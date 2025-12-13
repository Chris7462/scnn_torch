from pathlib import Path

import cv2
import numpy as np


def get_lane_coords(
    prob_map: np.ndarray,
    y_px_gap: int,
    pts: int,
    thresh: float,
    resize_shape: tuple[int, int],
) -> np.ndarray:
    """
    Extract lane coordinates from probability map for CULane format.

    Args:
        prob_map: Probability map for single lane (h, w)
        y_px_gap: Y pixel gap for sampling
        pts: Number of points to sample per lane
        thresh: Probability threshold
        resize_shape: Target shape (H, W)

    Returns:
        X coordinates bottom up every y_px_gap px, 0 for non-exist
    """
    h, w = prob_map.shape
    H, W = resize_shape

    coords = np.zeros(pts)
    for i in range(pts):
        y = int(h - i * y_px_gap / H * h - 1)
        if y < 0:
            break
        line = prob_map[y, :]
        idx = np.argmax(line)
        if line[idx] > thresh:
            coords[i] = int(idx / w * W)

    if (coords > 0).sum() < 2:
        coords = np.zeros(pts)

    return coords


def prob2lines(
    seg_pred: np.ndarray,
    exist: list[int],
    resize_shape: tuple[int, int],
    smooth: bool = True,
    y_px_gap: int = 20,
    pts: int = 18,
    thresh: float = 0.3,
) -> list[list[tuple[int, int]]]:
    """
    Convert probability map to lane coordinates for CULane format.

    Args:
        seg_pred: Segmentation prediction (5, h, w)
        exist: Lane existence list [0/1, 0/1, 0/1, 0/1]
        resize_shape: Target shape (H, W)
        smooth: Whether to smooth the probability map
        y_px_gap: Y pixel gap for sampling
        pts: Number of points per lane
        thresh: Probability threshold

    Returns:
        List of lane coordinates, each lane is list of (x, y) tuples
    """
    H, W = resize_shape
    coordinates = []

    # Transpose to (H, W, C) for easier processing
    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))

    for i in range(4):
        if exist[i] == 0:
            continue

        prob_map = seg_pred[..., i + 1]

        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)

        coords = get_lane_coords(prob_map, y_px_gap, pts, thresh, resize_shape)

        if (coords > 0).sum() < 2:
            continue

        # Convert to (x, y) tuples, only include valid points
        lane_coords = [
            (int(coords[j]), H - 1 - j * y_px_gap)
            for j in range(pts)
            if coords[j] > 0
        ]
        coordinates.append(lane_coords)

    return coordinates


def get_save_path(img_name: str, base_dir: Path, suffix: str) -> Path:
    """
    Get save path for output file.

    Preserves the CULane directory structure:
        base_dir/driver_xxx/xxx/xxx{suffix}

    Args:
        img_name: Original image path
        base_dir: Base directory for saving
        suffix: File suffix (e.g., '.lines.txt' or '.jpg')

    Returns:
        Path to save file
    """
    img_path = Path(img_name)

    # Build output path: base_dir/driver_xxx/xxx/xxx{suffix}
    save_name = img_path.stem + suffix
    save_path = base_dir / img_path.parts[-3] / img_path.parts[-2] / save_name

    return save_path
