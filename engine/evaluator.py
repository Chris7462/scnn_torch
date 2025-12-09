from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


class Evaluator:
    """
    Evaluator for SCNN model.

    Runs inference on test set and saves predictions to files.

    Args:
        model: SCNN model
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Output settings
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation settings from config
        eval_cfg = config.get('evaluation', {})
        self.resize_shape = tuple(eval_cfg.get('resize_shape', [590, 1640]))  # (H, W) for CULane
        self.y_px_gap = eval_cfg.get('y_px_gap', 20)
        self.pts = eval_cfg.get('pts', 18)
        self.thresh = eval_cfg.get('thresh', 0.3)

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
                exist_pred = torch.sigmoid(exist_pred)  # Apply sigmoid

                # Convert to numpy
                seg_pred = seg_pred.detach().cpu().numpy()
                exist_pred = exist_pred.detach().cpu().numpy()

                # Process each image in batch
                for i in range(len(seg_pred)):
                    self._process_prediction(
                        seg_pred[i],
                        exist_pred[i],
                        img_names[i],
                    )

        print(f"\nPredictions saved to: {self.output_dir}")
        return self.output_dir

    def _process_prediction(
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
        lane_coords = self._prob2lines(seg_pred, exist)

        # Build output path
        save_path = self._get_save_path(img_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        with open(save_path, 'w') as f:
            for lane in lane_coords:
                for (x, y) in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

    def _prob2lines(
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
            coords = self._get_lane_coords(prob_map, h, w, H, W)

            if len(coords) >= 2:
                coordinates.append(coords)

        return coordinates

    def _get_lane_coords(
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

    def _get_save_path(self, img_name: str) -> Path:
        """
        Get save path for prediction file.

        Args:
            img_name: Original image path

        Returns:
            Path to save prediction file
        """
        img_path = Path(img_name)

        # Build output path: output_dir/driver_xxx/xxx/xxx.lines.txt
        save_name = img_path.stem + '.lines.txt'
        save_path = self.output_dir / img_path.parts[-3] / img_path.parts[-2] / save_name

        return save_path
