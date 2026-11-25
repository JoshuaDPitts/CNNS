#!/usr/bin/env python3
"""
Pre-resize CIFAR-10 images to 224x224 and save as TensorDatasets.

This avoids redundant on-the-fly resizing during training (50k images x N epochs).
Output files:
  - <output_dir>/cifar10_224_train.pt  (~3.5 GB)
  - <output_dir>/cifar10_224_test.pt   (~700 MB)
  - <output_dir>/cifar10_224_mean.npy  (channel means at 224x224)

Usage:
    python3 scripts/preprocess_cifar.py --output_dir /workspace
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def preprocess_split(dataset, img_size: int, desc: str) -> torch.Tensor:
    """
    Resize all images in a CIFAR-10 split and return as a uint8 tensor.

    Storing as uint8 (1 byte/pixel) instead of float32 (4 bytes/pixel) cuts
    the in-memory and on-disk footprint by 4x. For the 50k-image train split
    at 224x224, this is the difference between ~30 GB (won't fit on a
    g2-standard-4 with 16 GB RAM) and ~7.5 GB. The training-time transforms
    in scripts/train.py convert uint8 -> float32 [0,1] before normalisation.
    """
    resize = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.PILToTensor(),  # keep uint8 [0, 255], C x H x W
    ])

    n = len(dataset)
    images = torch.empty(n, 3, img_size, img_size, dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.long)

    for i in range(n):
        img, label = dataset[i]  # PIL image, int
        images[i] = resize(img)
        labels[i] = label
        if (i + 1) % 5000 == 0:
            print(f"  {desc}: {i + 1}/{n}")

    return images, labels


def main():
    parser = argparse.ArgumentParser(
        description="Pre-resize CIFAR-10 to 224x224 for faster training"
    )
    parser.add_argument("--img_size", type=int, default=224,
                        help="Target image size (default: 224)")
    parser.add_argument("--output_dir", type=str, default="/workspace",
                        help="Directory to save .pt files")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="CIFAR-10 download directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw CIFAR-10 (no transform — we apply our own)
    print("Downloading/loading CIFAR-10...")
    train_ds = CIFAR10(root=args.data_dir, train=True, download=True)
    test_ds = CIFAR10(root=args.data_dir, train=False, download=True)

    # Resize train split
    print(f"Resizing train split to {args.img_size}x{args.img_size}...")
    train_images, train_labels = preprocess_split(train_ds, args.img_size, "train")

    # Resize test split
    print(f"Resizing test split to {args.img_size}x{args.img_size}...")
    test_images, test_labels = preprocess_split(test_ds, args.img_size, "test")

    # Channel-mean computation. The uint8 -> float32 cast on the full train
    # tensor would allocate ~30 GB in one shot (50k x 3 x 224^2 x 4 bytes)
    # which overruns a 16 GB-RAM VM. Stream the conversion in chunks of
    # 1000 images so the working set stays under ~700 MB at a time.
    def _channel_mean_uint8(images: torch.Tensor, chunk: int = 1000) -> torch.Tensor:
        n_imgs = images.shape[0]
        total_px = n_imgs * images.shape[2] * images.shape[3]
        acc = torch.zeros(3, dtype=torch.float64)
        for i in range(0, n_imgs, chunk):
            c = images[i:i + chunk].to(torch.float64).div_(255.0)
            acc += c.sum(dim=(0, 2, 3))
        return (acc / total_px).to(torch.float32)

    mean = _channel_mean_uint8(train_images)
    print(f"Channel means at {args.img_size}x{args.img_size}: {mean.tolist()}")

    # Save
    train_path = os.path.join(args.output_dir, f"cifar10_{args.img_size}_train.pt")
    test_path = os.path.join(args.output_dir, f"cifar10_{args.img_size}_test.pt")
    mean_path = os.path.join(args.output_dir, f"cifar10_{args.img_size}_mean.npy")

    print(f"Saving train dataset to {train_path}...")
    torch.save(TensorDataset(train_images, train_labels), train_path)

    print(f"Saving test dataset to {test_path}...")
    torch.save(TensorDataset(test_images, test_labels), test_path)

    print(f"Saving mean file to {mean_path}...")
    np.save(mean_path, mean.numpy())

    # Report sizes
    train_size = os.path.getsize(train_path) / (1024 ** 3)
    test_size = os.path.getsize(test_path) / (1024 ** 3)
    print(f"\nDone! Train: {train_size:.2f} GB, Test: {test_size:.2f} GB")


if __name__ == "__main__":
    main()
