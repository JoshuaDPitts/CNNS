#!/usr/bin/env python3
"""
Unified 200-epoch training harness for all CNN configurations evaluated in:
  Fischer, M. M. & Pitts, J. D. "Batch normalization rather than nominal or
  effective depth governs convolutional network accuracy while topology governs
  efficiency."

Shared optimization protocol (Methods):
  - Dataset        : Official CIFAR-10 (50,000 train / 10,000 test; no carved split),
                     bilinearly resized to 224 x 224 pixels
  - Augmentation   : 4-px zero-padding -> RandomCrop(224) -> RandomHorizontalFlip(0.5)
                     -> [0, 1] scaling -> per-channel RGB mean subtraction
                     (0.491400, 0.482158, 0.446531) with unit divisors
  - Optimizer      : SGD, momentum 0.9 (non-Nesterov), weight decay 5e-4 across
                     all trainable parameters, mini-batch size 128
  - Schedule       : Initial LR 0.01, StepLR(step_size=30, gamma=0.1), 200 epochs
  - Initialization : He-normal (kaiming_normal_, fan_in, ReLU) on Conv2d & Linear;
                     BatchNorm weight=1, bias=0
  - Precision      : Automatic mixed precision (torch.cuda.amp) with dynamic
                     GradScaler and no gradient clipping
  - Objective      : Cross-entropy without label smoothing (GoogLeNet includes
                     both auxiliary heads weighted at 0.3 during training;
                     evaluation uses main head only)
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import get_model
from models.depth_info import get_depth_info
from scripts.metrics_sampler import MetricsSampler
from scripts.run_logger import RunLogger

try:
    from thop import profile as thop_profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

CIFAR10_NATIVE_MEAN = [0.491400, 0.482158, 0.446531]


def he_normal_init_(model: nn.Module) -> None:
    """Apply He-normal initialization to Conv2d/Linear and 1/0 to BatchNorm."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
        ):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_macs(model: nn.Module, img_size: int, device) -> float:
    """Single-image evaluation-mode MACs in G-MACs (1 MAC = 2 FLOPs)."""
    if not THOP_AVAILABLE:
        return 0.0
    try:
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        model.train(False)
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        return macs / 1e9
    except Exception as exc:
        print(f"Warning: could not profile MACs: {exc}")
        return 0.0


def compute_gradient_norms(model: nn.Module) -> dict:
    """Compute global parameter-gradient L2 norm across all trainable tensors."""
    grad_norms = []
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            grad_norms.append(param_norm)
            total_norm_sq += param_norm**2
    if not grad_norms:
        return {
            "grad_norm_total": 0.0,
            "grad_norm_mean": 0.0,
            "grad_norm_std": 0.0,
            "grad_norm_max": 0.0,
        }
    total_norm = total_norm_sq**0.5
    mean_norm = sum(grad_norms) / len(grad_norms)
    std_norm = (sum((n - mean_norm) ** 2 for n in grad_norms) / len(grad_norms)) ** 0.5
    return {
        "grad_norm_total": total_norm,
        "grad_norm_mean": mean_norm,
        "grad_norm_std": std_norm,
        "grad_norm_max": max(grad_norms),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset, transform=None):
        self.dataset = tensor_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_train_transforms(img_size: int, mean: list) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, (1.0, 1.0, 1.0)),
        ]
    )


def _build_test_transforms(mean: list) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, (1.0, 1.0, 1.0)),
        ]
    )


def get_data(
    mean_file: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    img_size: int = 224,
    preresized_dir: str = "",
):
    if mean_file and os.path.exists(mean_file):
        mean = np.load(mean_file).tolist()
    else:
        mean = list(CIFAR10_NATIVE_MEAN)

    if preresized_dir and img_size > 32:
        train_pt = os.path.join(preresized_dir, f"cifar10_{img_size}_train.pt")
        test_pt = os.path.join(preresized_dir, f"cifar10_{img_size}_test.pt")
        if os.path.exists(train_pt) and os.path.exists(test_pt):
            train_tensor_ds = torch.load(train_pt, weights_only=False)
            test_tensor_ds = torch.load(test_pt, weights_only=False)
            train_ds = AugmentedTensorDataset(
                train_tensor_ds, transform=_build_train_transforms(img_size, mean)
            )
            test_ds = AugmentedTensorDataset(
                test_tensor_ds, transform=_build_test_transforms(mean)
            )
            persistent = num_workers > 0
            return (
                DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent,
                ),
                DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent,
                ),
            )

    resize_ops = [transforms.Resize(img_size, antialias=True)] if img_size != 32 else []
    tf_train = transforms.Compose(
        resize_ops
        + [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, (1.0, 1.0, 1.0)),
        ]
    )
    tf_test = transforms.Compose(
        resize_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean, (1.0, 1.0, 1.0)),
        ]
    )
    train_ds = CIFAR10(root="data", train=True, download=True, transform=tf_train)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=tf_test)
    persistent = num_workers > 0
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        ),
    )


def accuracy(logits, targets) -> float:
    _, pred = logits.max(1)
    return pred.eq(targets).float().mean().item()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    train: bool = True,
    track_gradients: bool = False,
):
    model.train(mode=train)
    total_loss, total_acc = 0.0, 0.0
    grad_norms_total, grad_norms_mean, grad_norms_max = [], [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        with (
            torch.cuda.amp.autocast(enabled=scaler is not None),
            torch.set_grad_enabled(train),
        ):
            logits = model(imgs)
            if isinstance(logits, tuple):
                main_logits, *aux_logits = logits
                loss = criterion(main_logits, targets)
                for aux in aux_logits:
                    if aux is not None:
                        loss = loss + 0.3 * criterion(aux, targets)
                logits = main_logits
            else:
                loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if track_gradients:
                    gs = compute_gradient_norms(model)
                    grad_norms_total.append(gs["grad_norm_total"])
                    grad_norms_mean.append(gs["grad_norm_mean"])
                    grad_norms_max.append(gs["grad_norm_max"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if track_gradients:
                    gs = compute_gradient_norms(model)
                    grad_norms_total.append(gs["grad_norm_total"])
                    grad_norms_mean.append(gs["grad_norm_mean"])
                    grad_norms_max.append(gs["grad_norm_max"])
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy(logits, targets) * imgs.size(0)

    n = len(loader.dataset)
    avg_loss, avg_acc = total_loss / n, total_acc / n
    if track_gradients and grad_norms_total:
        mu = sum(grad_norms_total) / len(grad_norms_total)
        epoch_grad_stats = {
            "grad_norm_total_mean": mu,
            "grad_norm_mean_mean": sum(grad_norms_mean) / len(grad_norms_mean),
            "grad_norm_max_mean": sum(grad_norms_max) / len(grad_norms_max),
            "grad_norm_total_max": max(grad_norms_total),
            "grad_norm_total_std": (
                sum((x - mu) ** 2 for x in grad_norms_total) / len(grad_norms_total)
            )
            ** 0.5,
        }
        return avg_loss, avg_acc, epoch_grad_stats
    return avg_loss, avg_acc


def main(cfg, progress_cb: Optional[Callable] = None) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    train_ld, test_ld = get_data(
        cfg.mean_file,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        img_size=cfg.img_size,
        preresized_dir=cfg.preresized_dir,
    )

    sampler = MetricsSampler(interval=0.5)
    sampler.start()
    n_train = len(train_ld.dataset)

    logger = RunLogger(
        tb_dir=cfg.log_dir,
        csv_dir=cfg.ckpt_dir,
        csv_name=f"{cfg.model}_run_metrics.csv",
    )

    model = get_model(cfg.model, cfg.dropout).to(device)
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        he_normal_init_(model)

    num_params_m = count_parameters(model) / 1e6
    macs_g = compute_macs(model, cfg.img_size, device)
    depth_info = get_depth_info(cfg.model)
    nominal_depth = depth_info["nominal_depth"]
    effective_depth = depth_info["effective_depth"]
    print(
        f"Model: {cfg.model} | Seed: {cfg.seed} | Params: {num_params_m:.3f}M | "
        f"G-MACs: {macs_g:.3f} | D_nom: {nominal_depth} | D_eff: {effective_depth:.1f}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_acc = 0.0
    endpoint_acc = 0.0
    all_grad_stats = []
    start = time.time()

    for epoch in range(cfg.epochs):
        train_loss, train_acc, grad_stats = run_epoch(
            model,
            train_ld,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            track_gradients=True,
        )
        all_grad_stats.append(grad_stats)

        # Observational test-set logging (never influences updates, LR schedule, or endpoint choice)
        test_loss, test_acc = run_epoch(
            model,
            test_ld,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
        )
        if not math.isfinite(train_loss) or not math.isfinite(test_loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, test={test_loss}"
            )
        scheduler.step()

        endpoint_acc = test_acc
        if test_acc > best_acc:
            best_acc = test_acc

        logger.log_epoch(
            epoch,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            optimizer.param_groups[0]["lr"],
            grad_stats=grad_stats,
        )
        if progress_cb:
            progress_cb(epoch, train_loss, test_loss, test_acc, cfg)

    if cfg.ckpt_dir:
        ckpt_path = (
            Path(cfg.ckpt_dir) / f"{cfg.model}_s{cfg.seed}_epoch{cfg.epochs}.pth"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    sampler.stop()
    metrics = sampler.summary()
    run_time = time.time() - start
    images_per_sec = (
        (n_train * cfg.epochs) / metrics["wall_time_s"]
        if metrics.get("wall_time_s")
        else None
    )

    finite_late_norms = [
        g["grad_norm_total_mean"]
        for g in all_grad_stats[-20:]
        if g
        and g.get("grad_norm_total_mean") is not None
        and math.isfinite(g["grad_norm_total_mean"])
    ]
    extra = {
        "num_params_m": num_params_m,
        "macs_g": macs_g,
        "nominal_depth": nominal_depth,
        "effective_depth": effective_depth,
        "grad_norm_mean": float(np.mean(finite_late_norms))
        if finite_late_norms
        else None,
        "grad_norm_std": float(np.std(finite_late_norms, ddof=1))
        if len(finite_late_norms) > 1
        else None,
    }
    logger.write_csv_row(
        cfg, endpoint_acc, best_acc, metrics, images_per_sec, extra=extra
    )
    logger.close()
    print(
        f"Completed {cfg.epochs} epochs in {run_time / 60:.1f} min | "
        f"Endpoint test accuracy: {endpoint_acc * 100:.2f}% | Best-to-date: {best_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 200-epoch CNN training loop")
    parser.add_argument(
        "--model",
        required=True,
        help="Architecture name (vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, "
        "resnet18, resnet34, resnet50, googlenet, vgg11, vgg16, vgg19)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr_step_size", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mean_file", type=str, default="data/cifar10_mean.npy")
    parser.add_argument("--preresized_dir", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="runs")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    main(parser.parse_args())
