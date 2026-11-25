"""
Per-epoch and per-run CSV / TensorBoard logging utilities used by scripts/train.py.
"""

from pathlib import Path
import csv
from torch.utils.tensorboard import SummaryWriter

SUMMARY_FIELDS = [
    "model",
    "epochs",
    "batch_size",
    "lr",
    "dropout",
    "weight_decay",
    "num_params_m",
    "macs_g",
    "nominal_depth",
    "effective_depth",
    "grad_norm_mean",
    "grad_norm_std",
    "wall_time_s",
    "images_per_sec",
    "gpu_name",
    "gpu_util_mean",
    "gpu_util_peak",
    "gpu_mem_mb_peak",
    "gpu_power_mean_w",
    "gpu_power_peak_w",
    "energy_j",
    "energy_wh",
    "gpu_temp_peak_c",
    "cpu_util_mean",
    "cpu_util_p95",
    "rss_mb_peak",
    "disk_read_mb",
    "disk_write_mb",
    "net_recv_mb",
    "net_sent_mb",
    "endpoint_test_acc",
    "best_test_acc",
]

PROGRESS_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
    "lr",
    "grad_norm_total",
    "grad_norm_mean",
    "grad_norm_max",
    "grad_norm_std",
]


class RunLogger:
    def __init__(
        self,
        tb_dir: str = "",
        csv_dir: str = "",
        csv_name: str = "run_metrics.csv",
        progress_name: str = "epoch_progress.csv",
    ):
        self.writer = SummaryWriter(tb_dir) if tb_dir else None
        self.csv_path = None
        self.progress_path = None
        if csv_dir:
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
            self.csv_path = Path(csv_dir) / csv_name
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()
            self.progress_path = Path(csv_dir) / progress_name
            if not self.progress_path.exists():
                with open(self.progress_path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=PROGRESS_FIELDS).writeheader()

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer and value is not None:
            self.writer.add_scalar(tag, value, step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        lr: float,
        grad_stats: dict | None = None,
    ):
        self.log_scalar("Loss/train", train_loss, epoch)
        self.log_scalar("Loss/test", test_loss, epoch)
        self.log_scalar("Acc/train", train_acc, epoch)
        self.log_scalar("Acc/test", test_acc, epoch)
        self.log_scalar("LR", lr, epoch)

        if grad_stats:
            self.log_scalar(
                "Gradient/norm_total", grad_stats.get("grad_norm_total_mean"), epoch
            )
            self.log_scalar(
                "Gradient/norm_mean", grad_stats.get("grad_norm_mean_mean"), epoch
            )
            self.log_scalar(
                "Gradient/norm_max", grad_stats.get("grad_norm_max_mean"), epoch
            )
            self.log_scalar(
                "Gradient/norm_std", grad_stats.get("grad_norm_total_std"), epoch
            )

        if self.progress_path:
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": lr,
                "grad_norm_total": (grad_stats or {}).get("grad_norm_total_mean"),
                "grad_norm_mean": (grad_stats or {}).get("grad_norm_mean_mean"),
                "grad_norm_max": (grad_stats or {}).get("grad_norm_max_mean"),
                "grad_norm_std": (grad_stats or {}).get("grad_norm_total_std"),
            }
            with open(self.progress_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=PROGRESS_FIELDS).writerow(row)

    def write_csv_row(
        self,
        cfg,
        endpoint_test_acc: float,
        best_test_acc: float,
        metrics: dict,
        images_per_sec,
        extra: dict | None = None,
    ):
        if not self.csv_path:
            return
        m = dict(metrics) if metrics else {}
        if extra:
            m.update(extra)
        if "energy_j" not in m and m.get("energy_wh") is not None:
            m["energy_j"] = m["energy_wh"] * 3600.0
        row = {
            "model": cfg.model,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "dropout": cfg.dropout,
            "weight_decay": cfg.weight_decay,
            "endpoint_test_acc": endpoint_test_acc,
            "best_test_acc": best_test_acc,
            "images_per_sec": images_per_sec,
        }
        row.update({k: m.get(k, "") for k in SUMMARY_FIELDS if k not in row})
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(row)

    def close(self):
        if self.writer:
            self.writer.close()
