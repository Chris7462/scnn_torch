"""
CULane Evaluation in Python

Mirrors the C++ evaluator logic:
1. Read lane coordinates from .txt files
2. Draw lanes on binary masks
3. Compute IoU between predicted and ground truth lanes
4. Use Hungarian matching for optimal lane assignment
5. Count TP, FP, FN based on IoU threshold
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev


class CULaneEvaluator:
    """
    CULane lane detection evaluator.

    Args:
        img_width: Image width for evaluation (default: 1640)
        img_height: Image height for evaluation (default: 590)
        iou_thresh: IoU threshold for matching (default: 0.5)
        lane_width: Width of lane lines for drawing (default: 30)
    """

    def __init__(
        self,
        img_width: int = 1640,
        img_height: int = 590,
        iou_thresh: float = 0.5,
        lane_width: int = 30,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.iou_thresh = iou_thresh
        self.lane_width = lane_width

    def read_lanes(self, file_path: str | Path) -> list[np.ndarray]:
        """
        Read lane coordinates from a .txt file.

        Each line contains: x1 y1 x2 y2 x3 y3 ...

        Args:
            file_path: Path to .txt file

        Returns:
            List of lanes, each lane is ndarray of shape (N, 2)
        """
        lanes = []
        file_path = Path(file_path)

        if not file_path.exists():
            return lanes

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = list(map(float, line.split()))
                if len(values) < 4:  # Need at least 2 points
                    continue

                # Reshape to (N, 2) array of (x, y) points
                points = np.array(values).reshape(-1, 2)
                lanes.append(points)

        return lanes

    def interp_lane(self, lane: np.ndarray, n_points: int = 50) -> np.ndarray:
        """
        Interpolate lane points using spline.

        Args:
            lane: Lane points of shape (N, 2)
            n_points: Number of output points

        Returns:
            Interpolated points of shape (n_points, 2)
        """
        if len(lane) < 2:
            return lane

        if len(lane) == 2:
            # Linear interpolation for 2 points
            t = np.linspace(0, 1, n_points)
            x = lane[0, 0] + t * (lane[1, 0] - lane[0, 0])
            y = lane[0, 1] + t * (lane[1, 1] - lane[0, 1])
            return np.column_stack([x, y])

        try:
            # Spline interpolation for 3+ points
            tck, u = splprep([lane[:, 0], lane[:, 1]], s=0, k=min(3, len(lane) - 1))
            u_new = np.linspace(0, 1, n_points)
            x_new, y_new = splev(u_new, tck)
            return np.column_stack([x_new, y_new])
        except Exception:
            # Fallback to linear interpolation
            t = np.linspace(0, 1, n_points)
            x = np.interp(t, np.linspace(0, 1, len(lane)), lane[:, 0])
            y = np.interp(t, np.linspace(0, 1, len(lane)), lane[:, 1])
            return np.column_stack([x, y])

    def draw_lane(self, lane: np.ndarray) -> np.ndarray:
        """
        Draw lane on a binary mask.

        Args:
            lane: Lane points of shape (N, 2)

        Returns:
            Binary mask of shape (img_height, img_width)
        """
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        if len(lane) < 2:
            return mask

        # Interpolate lane for smooth drawing
        lane_interp = self.interp_lane(lane)

        # Convert to integer points
        points = lane_interp.astype(np.int32)

        # Draw polyline
        cv2.polylines(mask, [points], isClosed=False, color=1, thickness=self.lane_width)

        return mask

    def compute_iou(self, lane1: np.ndarray, lane2: np.ndarray) -> float:
        """
        Compute IoU between two lanes.

        Args:
            lane1: First lane points of shape (N, 2)
            lane2: Second lane points of shape (M, 2)

        Returns:
            IoU score between 0 and 1
        """
        mask1 = self.draw_lane(lane1)
        mask2 = self.draw_lane(lane2)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)

        if union == 0:
            return 0.0

        return intersection / union

    def match_lanes(
        self,
        gt_lanes: list[np.ndarray],
        pred_lanes: list[np.ndarray],
    ) -> tuple[list[int], list[int]]:
        """
        Match predicted lanes to ground truth using Hungarian algorithm.

        Args:
            gt_lanes: List of ground truth lanes
            pred_lanes: List of predicted lanes

        Returns:
            gt_match: For each GT lane, index of matched pred lane (-1 if unmatched)
            pred_match: For each pred lane, index of matched GT lane (-1 if unmatched)
        """
        n_gt = len(gt_lanes)
        n_pred = len(pred_lanes)

        gt_match = [-1] * n_gt
        pred_match = [-1] * n_pred

        if n_gt == 0 or n_pred == 0:
            return gt_match, pred_match

        # Build IoU similarity matrix
        iou_matrix = np.zeros((n_gt, n_pred))
        for i, gt_lane in enumerate(gt_lanes):
            for j, pred_lane in enumerate(pred_lanes):
                iou_matrix[i, j] = self.compute_iou(gt_lane, pred_lane)

        # Hungarian matching (minimize cost, so use negative IoU)
        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Apply IoU threshold
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if iou_matrix[gt_idx, pred_idx] >= self.iou_thresh:
                gt_match[gt_idx] = pred_idx
                pred_match[pred_idx] = gt_idx

        return gt_match, pred_match

    def evaluate_image(
        self,
        gt_lanes: list[np.ndarray],
        pred_lanes: list[np.ndarray],
    ) -> tuple[int, int, int]:
        """
        Evaluate a single image.

        Args:
            gt_lanes: List of ground truth lanes
            pred_lanes: List of predicted lanes

        Returns:
            tp: True positives (matched lanes)
            fp: False positives (unmatched predictions)
            fn: False negatives (unmatched ground truth)
        """
        if len(gt_lanes) == 0:
            return 0, len(pred_lanes), 0

        if len(pred_lanes) == 0:
            return 0, 0, len(gt_lanes)

        gt_match, pred_match = self.match_lanes(gt_lanes, pred_lanes)

        tp = sum(1 for m in gt_match if m >= 0)
        fn = len(gt_lanes) - tp
        fp = len(pred_lanes) - tp

        return tp, fp, fn

    def evaluate_file(
        self,
        gt_file: str | Path,
        pred_file: str | Path,
    ) -> tuple[int, int, int]:
        """
        Evaluate a single file pair.

        Args:
            gt_file: Path to ground truth .txt file
            pred_file: Path to prediction .txt file

        Returns:
            tp, fp, fn counts
        """
        gt_lanes = self.read_lanes(gt_file)
        pred_lanes = self.read_lanes(pred_file)

        return self.evaluate_image(gt_lanes, pred_lanes)


def evaluate_culane(
    pred_dir: str | Path,
    anno_dir: str | Path,
    list_file: str | Path,
    img_width: int = 1640,
    img_height: int = 590,
    iou_thresh: float = 0.5,
    lane_width: int = 30,
    verbose: bool = True,
) -> dict:
    """
    Evaluate CULane predictions.

    Args:
        pred_dir: Directory containing prediction .txt files
        anno_dir: Directory containing ground truth annotation files
        list_file: File containing list of images to evaluate
        img_width: Image width
        img_height: Image height
        iou_thresh: IoU threshold for matching
        lane_width: Lane width for drawing
        verbose: Print progress

    Returns:
        Dictionary with precision, recall, f1, tp, fp, fn
    """
    pred_dir = Path(pred_dir)
    anno_dir = Path(anno_dir)

    evaluator = CULaneEvaluator(
        img_width=img_width,
        img_height=img_height,
        iou_thresh=iou_thresh,
        lane_width=lane_width,
    )

    # Read image list
    with open(list_file, 'r') as f:
        img_list = [line.strip() for line in f if line.strip()]

    total_tp, total_fp, total_fn = 0, 0, 0

    for i, img_name in enumerate(img_list):
        # Build file paths
        # img_name format: /driver_xxx/xxx/xxx.jpg
        txt_name = img_name.replace('.jpg', '.lines.txt').lstrip('/')

        gt_file = anno_dir / txt_name
        pred_file = pred_dir / txt_name

        tp, fp, fn = evaluator.evaluate_file(gt_file, pred_file)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(img_list)} images")

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
