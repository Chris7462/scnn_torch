"""
Inference script for running SCNN on arbitrary images (e.g., KITTI).

Usage:
    python script/kitti_infer.py --checkpoint checkpoints/best.pth --image path/to/image.png
    python script/kitti_infer.py --checkpoint checkpoints/best.pth --image_dir path/to/images/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import SCNN
from utils import prob2lines
from utils import visualize_lanes


def parse_args():
    parser = argparse.ArgumentParser(description='Run SCNN inference on images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to directory of images')
    parser.add_argument('--output_dir', type=str, default='outputs/infer',
                        help='Directory to save results')
    parser.add_argument('--target_height', type=int, default=288,
                        help='Target height for resizing (default: 288)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--thresh', type=float, default=0.3,
                        help='Probability threshold for lane detection')
    return parser.parse_args()


def build_model(checkpoint_path: str, device: torch.device) -> SCNN:
    """Build and load SCNN model."""
    model = SCNN(ms_ks=9, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")

    return model


def preprocess_image(
    image: Image.Image,
    target_height: int = 288,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Preprocess image for inference.

    Args:
        image: PIL Image
        target_height: Target height for resizing
        mean: Normalization mean
        std: Normalization std

    Returns:
        input_tensor: Preprocessed tensor (1, 3, H, W)
        original_size: Original image size (H, W)
    """
    # Original size
    original_size = (image.height, image.width)

    # Calculate target width preserving aspect ratio, divisible by 8
    target_width = round(original_size[1] * target_height / original_size[0] / 8) * 8
    resized_size = (target_height, target_width)

    # Build transform
    transform = transforms.Compose([
        transforms.Resize(resized_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Apply transform
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, original_size


def resize_seg_pred(
    seg_pred: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize segmentation prediction to target size.

    Args:
        seg_pred: Segmentation probabilities (5, H, W)
        target_size: Target size (H, W)

    Returns:
        Resized segmentation probabilities (5, target_H, target_W)
    """
    target_h, target_w = target_size
    resized = np.zeros((seg_pred.shape[0], target_h, target_w), dtype=seg_pred.dtype)
    for i in range(seg_pred.shape[0]):
        resized[i] = cv2.resize(seg_pred[i], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return resized


def infer_single_image(
    model: SCNN,
    image_path: str,
    device: torch.device,
    target_height: int = 288,
    thresh: float = 0.3
) -> tuple[list, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Run inference on a single image.

    Args:
        model: SCNN model
        image_path: Path to image
        device: Torch device
        target_height: Target height for resizing
        thresh: Probability threshold

    Returns:
        lanes: List of lane coordinates in original image space
        seg_pred: Segmentation probabilities (5, H, W) at original size
        exist_pred: Lane existence probabilities (4,)
        original_size: Original image size (H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Preprocess
    input_tensor, original_size = preprocess_image(image, target_height)
    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        seg_pred, exist_pred = model(input_tensor)

        # Convert seg_pred logits to probabilities
        seg_pred = F.softmax(seg_pred, dim=1)

    # Convert to numpy
    seg_pred = seg_pred.squeeze(0).cpu().numpy()
    exist_pred = exist_pred.squeeze(0).cpu().numpy()

    # Resize seg_pred to original image size
    seg_pred = resize_seg_pred(seg_pred, original_size)

    # Post-process: get lane coordinates (seg_pred is now at original size)
    lanes = prob2lines(
        seg_pred=seg_pred,
        exist=exist_pred,
        thresh=thresh,
    )

    return lanes, seg_pred, exist_pred, original_size


def main():
    args = parse_args()

    # Validate args
    if args.image is None and args.image_dir is None:
        raise ValueError("Must specify either --image or --image_dir")

    # Device
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Build model
    model = build_model(args.checkpoint, device)

    # Collect images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    print(f"Found {len(image_paths)} images")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")

        lanes, seg_pred, exist_pred, original_size = infer_single_image(
            model=model,
            image_path=str(image_path),
            device=device,
            target_height=args.target_height,
            thresh=args.thresh,
        )

        # Print results
        print(f"  Original size: {original_size[1]} x {original_size[0]}")
        print(f"  Lane existence: {[f'{p:.2f}' for p in exist_pred]}")
        print(f"  Detected lanes: {len(lanes)}")

        # Save visualization
        if args.visualize:
            # Load original image
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate overlay (seg_pred is already at original size)
            img_overlay, _ = visualize_lanes(img, seg_pred, exist_pred)

            # Save
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
            vis_path = output_dir / f"{image_path.stem}_vis.png"
            cv2.imwrite(str(vis_path), img_overlay)
            print(f"  Saved: {vis_path}")


if __name__ == '__main__':
    main()
