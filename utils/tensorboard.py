from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str | Path) -> None:
        """
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar variable.

        Args:
            tag: Name of the scalar
            value: Value to log
            step: Training step or epoch
        """
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag: str, images: list[np.ndarray], step: int) -> None:
        """
        Log a list of images.

        Args:
            tag: Name of the image group
            images: List of images (numpy arrays)
            step: Training step or epoch
        """
        for i, img in enumerate(images):
            # Convert to (C, H, W) format if needed
            if len(img.shape) == 3 and img.shape[2] in [1, 3, 4]:
                img = np.transpose(img, (2, 0, 1))
            self.writer.add_image(f'{tag}/{i}', img, step)

    def flush(self) -> None:
        """Flush the writer."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()
