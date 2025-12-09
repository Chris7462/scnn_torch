import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SegHead(nn.Module):
    """
    Segmentation head for SCNN.

    Upsamples segmentation features to original image resolution.

    Architecture:
        Bilinear Upsample 8x

    Output classes:
        0: Background
        1-4: Lane 1-4
    """

    def __init__(self, upsample_scale: int = 8) -> None:
        super().__init__()

        self.upsample_scale = upsample_scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 5, H, W)

        Returns:
            Segmentation logits of shape (B, 5, H*8, W*8)
        """
        return F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=True)
