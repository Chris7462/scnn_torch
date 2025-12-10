from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from utils.visualization import visualize_lanes


class Evaluator:
    """
    Evaluator for SCNN model.

    Runs inference on test set and saves predictions to files.

    Args:
        model: SCNN model
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to run on
        visualize: Whether to save visualization images
        num_visualize: Number of images to visualize
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        config: dict,
        device: torch.device,
        visualize: bool = False,
        num_visualize: int = 20,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.visualize = visualize
        self.num_visualize = num_visualize

        # Output settings
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.pred_dir = self.output_dir / 'predictions'
        self.vis_dir = self.output_dir / 'visualizations'
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        if self.visualize:
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation settings from config
        eval_cfg = config.get('evaluation', {})
        self.resize_shape = tuple(eval_cfg.get('resize_shape', [590, 1640]))  # (H, W) for CULane
        self.y_px_gap = eval_cfg.get('y_px_gap', 20)
        self.pts = eval_cfg.get('pts', 18)
        self.thresh = eval_cfg.get('thresh', 0.3)

        # Visualization counter
        self.vis_count = 0

    def evaluate(self) -> Path:
        """
        Run inference on test set and save predictions.

        Returns:
            output_dir: Path to directory containing predictions
        """
        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.test_loader, desc="Evaluating"):
                img = sample['img'].to(self.device)
                img_names = sample['img_name']

                # Forward pass
                seg_pred, exist_pred = self.model(img)
                seg_pred = F.softmax(seg_pred, dim=1)
                exist_pred = torch.sigmoid(exist_pred)

                # Convert to numpy
                seg_pred = seg_pred.detach().cpu().numpy()
                exist_pred = exist_pred.detach().cpu().numpy()

                # Process each image in batch
                for i in range(len(seg_pred)):
                    self.process_prediction(
                        seg_pred[i],
                        exist_pred[i],
                        img_names[i],
                    )

        print(f"\nPredictions saved to: {self.pred_dir}")
        if self.visualize:
            print(f"Visualizations saved to: {self.vis_dir}")
        return self.output_dir

    def process_prediction(
        self,
        seg_pred: np.ndarray,
        exist_pred: np.ndarray,
        img_name: str,
    ) -> None:
        """
        Process single prediction and save to file.

        Args:
            seg_pred: Segmentation prediction (5, H, W)
            exist_pred: Existence prediction (4,)
            img_name: Original image path
        """
        # Get lane coordinates
        exist = [1 if exist_pred[i] > 0.5 else 0 for i in range(4)]
        lane_coords = self.prob2lines(seg_pred, exist)

        # Build output path
        save_path = self.get_save_path(img_name, self.pred_dir, '.lines.txt')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        with open(save_path, 'w') as f:
            for lane in lane_coords:
                for (x, y) in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

        # Save visualization
        if self.visualize and self.vis_count < self.num_visualize:
            self.save_visualization(seg_pred, exist_pred, img_name)
            self.vis_count += 1

    def save_visualization(
        self,
        seg_pred: np.ndarray,
        exist_pred: np.ndarray,
        img_name: str,
    ) -> None:
        """
        Save visualization of prediction.

        Args:
            seg_pred: Segmentation prediction (5, H, W)
            exist_pred: Existence prediction (4,)
            img_name: Original image path
        """
        # Load original image
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to match prediction size
        pred_h, pred_w = seg_pred.shape[1], seg_pred.shape[2]
        img_resized = cv2.resize(img, (pred_w, pred_h), interpolation=cv2.INTER_CUBIC)

        # Generate overlay
        img_overlay, _ = visualize_lanes(img_resized, seg_pred, exist_pred)

        # Create side-by-side image
        side_by_side = np.concatenate([img_resized, img_overlay], axis=1)

        # Convert to BGR for saving
        side_by_side = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

        # Save visualization
        save_path = self.get_save_path(img_name, self.vis_dir, '.jpg')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), side_by_side)

    def prob2lines(
        self,
        seg_pred: np.ndarray,
        exist: list[int],
        smooth: bool = True,
    ) -> list[list[tuple[int, int]]]:
        """
        Convert probability map to lane coordinates.

        Args:
            seg_pred: Segmentation prediction (5, H, W)
            exist: Lane existence list [0/1, 0/1, 0/1, 0/1]
            smooth: Whether to apply smoothing

        Returns:
            List of lane coordinates, each lane is list of (x, y) tuples
        """
        _, h, w = seg_pred.shape
        H, W = self.resize_shape

        coordinates = []

        # Transpose to (H, W, C) for easier processing
        seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))

        for i in range(4):
            if exist[i] == 0:
                continue

            prob_map = seg_pred[..., i + 1]

            # Apply smoothing
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)

            # Get lane coordinates
            coords = self.get_lane_coords(prob_map, h, w, H, W)

            if len(coords) >= 2:
                coordinates.append(coords)

        return coordinates

    def get_lane_coords(
        self,
        prob_map: np.ndarray,
        h: int,
        w: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int]]:
        """
        Extract lane coordinates from probability map.

        Args:
            prob_map: Probability map for single lane (h, w)
            h, w: Input probability map size
            H, W: Output resize shape

        Returns:
            List of (x, y) coordinates
        """
        coords = []

        for i in range(self.pts):
            y = int(h - i * self.y_px_gap / H * h - 1)
            if y < 0:
                break

            line = prob_map[y, :]
            idx = np.argmax(line)

            if line[idx] > self.thresh:
                x = int(idx / w * W)
                coords.append((x, H - 1 - i * self.y_px_gap))

        return coords

    def get_save_path(self, img_name: str, base_dir: Path, suffix: str) -> Path:
        """
        Get save path for output file.

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
