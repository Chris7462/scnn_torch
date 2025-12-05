import torch.nn as nn


class SegNeck(nn.Module):
    """
    Segmentation neck for SCNN.

    Converts message passing features to class predictions (before upsampling).
    This output is shared by both seg_head and exist_head.

    Architecture:
        Dropout → Conv(128→5, 1x1)
    """

    def __init__(self, in_channels=128, out_channels=5, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 128, H, W)

        Returns:
            Segmentation features of shape (B, 5, H, W)
        """
        return self.layers(x)
