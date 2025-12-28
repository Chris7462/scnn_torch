"""
Compute mean and standard deviation for CULane dataset.

Computes statistics for both original and resized images.
Run this once to get normalization values for training.

Usage:
    python tools/compute_mean_std.py --data_dir data/CULane
    python tools/compute_mean_std.py --data_dir data/CULane --batch_size 128 --num_workers 4 --resize_height 288 --resize_width 800
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from scnn_torch.datasets import CULane


def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std for CULane')
    parser.add_argument('--data_dir', type=str, default='data/CULane',
                        help='Path to CULane dataset root')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    parser.add_argument('--resize_height', type=int, default=288,
                        help='Resize height (default: 288)')
    parser.add_argument('--resize_width', type=int, default=800,
                        help='Resize width (default: 800)')
    return parser.parse_args()


def get_original_transforms() -> A.Compose:
    """Transforms for original size (only convert to tensor)."""
    return A.Compose([
        ToTensorV2(),
    ])


def get_resized_transforms(height: int, width: int) -> A.Compose:
    """Transforms for resized images."""
    return A.Compose([
        A.Resize(height=height, width=width),
        ToTensorV2(),
    ])


def compute_mean_std(dataloader: DataLoader, desc: str) -> tuple[tuple, tuple]:
    """
    Compute mean and std from dataloader.

    Uses Welford's online algorithm for numerical stability.

    Args:
        dataloader: DataLoader yielding batches
        desc: Description for progress bar

    Returns:
        mean: Tuple of (mean_r, mean_g, mean_b)
        std: Tuple of (std_r, std_g, std_b)
    """
    # Accumulators
    channels_sum = torch.zeros(3)
    channels_sum_sq = torch.zeros(3)
    total_pixels = 0

    for batch in tqdm(dataloader, desc=desc):
        img = batch['img']  # (B, C, H, W)

        # Convert to float and scale to [0, 1]
        img = img.float() / 255.0

        batch_size, channels, height, width = img.shape
        num_pixels = batch_size * height * width

        # Sum over batch, height, width (keep channel dimension)
        channels_sum += img.sum(dim=[0, 2, 3])
        channels_sum_sq += (img ** 2).sum(dim=[0, 2, 3])
        total_pixels += num_pixels

    # Compute mean and std
    mean = channels_sum / total_pixels
    std = torch.sqrt(channels_sum_sq / total_pixels - mean ** 2)

    return tuple(mean.tolist()), tuple(std.tolist())


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print("=" * 60)
    print("Computing mean and std for CULane dataset")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Resize:      {args.resize_height}x{args.resize_width}")
    print("=" * 60)

    # Compute for original size
    print("\n[1/2] Original size")
    print("-" * 40)

    original_transforms = get_original_transforms()
    original_dataset = CULane(
        root=str(data_dir),
        image_set='train',
        transforms=original_transforms,
    )
    original_loader = DataLoader(
        original_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=original_dataset.collate,
        pin_memory=True,
    )

    print(f"Train set: {len(original_dataset)} images")
    original_mean, original_std = compute_mean_std(original_loader, "Computing")

    # Compute for resized
    print(f"\n[2/2] Resized ({args.resize_height}x{args.resize_width})")
    print("-" * 40)

    resized_transforms = get_resized_transforms(args.resize_height, args.resize_width)
    resized_dataset = CULane(
        root=str(data_dir),
        image_set='train',
        transforms=resized_transforms,
    )
    resized_loader = DataLoader(
        resized_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=resized_dataset.collate,
        pin_memory=True,
    )

    resized_mean, resized_std = compute_mean_std(resized_loader, "Computing")

    # Print config snippet
    print("\nTo use in config file (configs/scnn_culane.yaml):")
    print("-" * 60)
    print("# Original size")
    print(f"# mean: [{original_mean[0]:.4f}, {original_mean[1]:.4f}, {original_mean[2]:.4f}]")
    print(f"# std: [{original_std[0]:.4f}, {original_std[1]:.4f}, {original_std[2]:.4f}]")
    print()
    print("# Resized")
    print(f"# mean: [{resized_mean[0]:.4f}, {resized_mean[1]:.4f}, {resized_mean[2]:.4f}]")
    print(f"# std: [{resized_std[0]:.4f}, {resized_std[1]:.4f}, {resized_std[2]:.4f}]")


if __name__ == '__main__':
    main()
