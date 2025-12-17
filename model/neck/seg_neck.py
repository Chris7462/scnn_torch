import torch.nn as nn
from torch import Tensor


class SegNeck(nn.Module):
    """
    Segmentation neck for SCNN.

    Converts message passing features to class predictions (before upsampling).
    This output is shared by both seg_head and exist_head.

    Architecture:
        Dropout → Conv(128→5, 1x1)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 5,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 128, H, W)

        Returns:
            Segmentation features of shape (B, 5, H, W)
        """
        return self.layers(x)
