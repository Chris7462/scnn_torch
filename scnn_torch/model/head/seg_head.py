import torch.nn as nn
from torch import Tensor


class SegHead(nn.Module):
    """
    Segmentation head for SCNN.

    Upsamples segmentation features to original image resolution using
    learnable transposed convolutions (FCN-style).

    Architecture:
        3-stage upsampling (2× each stage = 8× total):
        (5, H/8, W/8) → ConvT → BN → ReLU →
        (5, H/4, W/4) → ConvT → BN → ReLU →
        (5, H/2, W/2) → ConvT →
        (5, H, W)

    Uses kernel_size=4, stride=2, padding=1 to avoid checkerboard artifacts.
    No activation after final stage since output is logits.

    Output classes:
        0: Background
        1-4: Lane 1-4
    """

    def __init__(self, in_channels: int = 5, num_stages: int = 3) -> None:
        """
        Args:
            in_channels: Number of input/output channels (default: 5)
            num_stages: Number of 2× upsample stages (default: 3 for 8× total)
        """
        super().__init__()

        self.num_stages = num_stages

        # Build upsample stages
        stages = []
        for i in range(num_stages):
            # Transposed conv: 2× upsample
            # kernel=4, stride=2, padding=1 gives clean 2× with no overlap issues
            stages.append(
                nn.ConvTranspose2d(
                    in_channels, in_channels,
                    kernel_size=4, stride=2, padding=1, bias=False
                )
            )

            # Add BN + ReLU for all but the last stage
            # (last stage outputs logits, no activation needed)
            if i < num_stages - 1:
                stages.append(nn.BatchNorm2d(in_channels))
                stages.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*stages)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 5, H, W)

        Returns:
            Segmentation logits of shape (B, 5, H*8, W*8)
        """
        return self.upsample(x)
