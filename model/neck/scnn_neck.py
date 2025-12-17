import torch.nn as nn
from torch import Tensor


class SCNNNeck(nn.Module):
    """
    Channel reduction neck for SCNN.

    Reduces backbone features from 512 channels to 128 channels
    with dilated convolution for larger receptive field.

    Architecture:
        Conv(512→1024, 3x3, dilation=4) → BN → ReLU →
        Conv(1024→128, 1x1) → BN → ReLU
    """

    def __init__(
        self,
        in_channels: int = 512,
        mid_channels: int = 1024,
        out_channels: int = 128,
        dilation: int = 4
    ) -> None:
        super().__init__()

        # Compute padding that preserves spatial dimensions
        kernel_size = 3
        padding = (dilation * (kernel_size - 1)) // 2

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 512, H, W)

        Returns:
            Feature tensor of shape (B, 128, H, W)
        """
        return self.layers(x)
