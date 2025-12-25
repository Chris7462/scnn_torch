from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import prob2lines, get_save_path, resize_seg_pred
from utils import visualize_lanes


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
        self.output_dir = Path(config['output_dir'])
        self.pred_dir = self.output_dir / 'predictions'
        self.vis_dir = self.output_dir / 'visualizations'
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        if self.visualize:
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation settings from config
        eval_cfg = config['evaluation']
        self.y_px_gap = eval_cfg['y_px_gap']
        self.pts = eval_cfg['pts']
        self.thresh = eval_cfg['thresh']

        # Visualization counter
        self.vis_count = 0

    def evaluate(self) -> Path:
        """
        Run inference on test set and save predictions.

        Returns:
            output_dir: Path to directory containing predictions
        """
        self.model.eval()

        num_batches = len(self.test_loader)
        print(f"Evaluating... ({num_batches} batches)")

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.test_loader):
                img = sample['img'].to(self.device)
                img_name = sample['img_name']
                original_size = sample['original_size']

                # Forward pass
                seg_pred, exist_pred = self.model(img)

                # Convert logits to probabilities
                seg_pred = F.softmax(seg_pred, dim=1)
                exist_pred = torch.sigmoid(exist_pred)

                # Convert to numpy
                seg_pred = seg_pred.cpu().numpy()
                exist_pred = exist_pred.cpu().numpy()

                # Process each image in batch
                for i in range(len(seg_pred)):
                    # Resize seg_pred to original image size
                    seg_pred_resized = resize_seg_pred(seg_pred[i], original_size[i])

                    self._save_prediction(seg_pred_resized, exist_pred[i], img_name[i])

                    if self.visualize and self.vis_count < self.num_visualize:
                        self._save_visualization(seg_pred_resized, exist_pred[i], img_name[i])
                        self.vis_count += 1

                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Processed {batch_idx + 1}/{num_batches} batches")

        print(f"\nPredictions saved to: {self.pred_dir}")
        if self.visualize:
            print(f"Visualizations saved to: {self.vis_dir}")
        return self.output_dir

    def _save_prediction(
        self,
        seg_pred: np.ndarray,
        exist_pred: np.ndarray,
        img_name: str,
    ) -> None:
        """
        Save lane prediction to file.

        Args:
            seg_pred: Segmentation probabilities (5, H, W) at original image size
            exist_pred: Existence probabilities (4,)
            img_name: Original image path
        """
        # Get lane coordinates
        lane_coords = prob2lines(
            seg_pred,
            exist_pred,
            y_px_gap=self.y_px_gap,
            pts=self.pts,
            thresh=self.thresh,
        )

        # Build output path
        save_path = get_save_path(img_name, self.pred_dir, '.lines.txt')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        with open(save_path, 'w') as f:
            for lane in lane_coords:
                for (x, y) in lane:
                    f.write(f"{x} {y} ")
                f.write("\n")

    def _save_visualization(
        self,
        seg_pred: np.ndarray,
        exist_pred: np.ndarray,
        img_name: str,
    ) -> None:
        """
        Save visualization of prediction.

        Args:
            seg_pred: Segmentation probabilities (5, H, W) at original image size
            exist_pred: Existence probabilities (4,)
            img_name: Original image path
        """
        # Load original image
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate overlay
        img_overlay, _ = visualize_lanes(img, seg_pred, exist_pred)

        # Convert to BGR for saving
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)

        # Save visualization
        save_path = get_save_path(img_name, self.vis_dir, '.jpg')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img_overlay)
