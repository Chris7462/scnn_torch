import torch.nn as nn
from torch import Tensor


class ExistHead(nn.Module):
    """
    Lane existence head for SCNN.

    Predicts whether each lane exists in the image using spatial convolutions
    followed by Global Max Pooling.

    Architecture:
        Select lane channels (drop background) →
        Conv(num_lanes→mid, 5×3, dilation=2×1) → GN → ReLU →
        Conv(mid→mid, 5×3, dilation=2×1) → GN → ReLU →
        Conv(mid→num_lanes, 1×1) →
        GlobalMaxPool(1,1) → Flatten

    Uses GroupNorm instead of BatchNorm for stability with small effective
    batch sizes (4 lanes per image). Both conv layers use tall dilated kernels
    (effective receptive field 9×3 each) to capture vertical lane structure.

    Output:
        num_lanes logits, one for each lane (use BCEWithLogitsLoss for training)
    """

    def __init__(
        self,
        in_channels: int = 5,
        mid_channels: int = 16,
        num_lanes: int = 4,
        num_groups: int = 4
    ) -> None:
        super().__init__()

        self.in_channels = in_channels

        # 3-layer conv with GroupNorm for stable training
        # Both spatial layers use tall dilated kernels for vertical lane detection
        self.conv = nn.Sequential(
            # Layer 1: Tall dilated conv to capture vertical lane structure
            # Kernel (5,3) with dilation (2,1) → effective receptive field (9,3)
            nn.Conv2d(num_lanes, mid_channels, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1), bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),

            # Layer 2: Another tall dilated conv for deeper vertical feature extraction
            # Kernel (5,3) with dilation (2,1) → effective receptive field (9,3)
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(5, 3), padding=(4, 1), dilation=(2, 1), bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),

            # Layer 3: 1×1 conv to output per-lane predictions
            nn.Conv2d(mid_channels, num_lanes, kernel_size=1),
        )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W) - segmentation logits before upsampling
               Channel 0: background
               Channel 1-num_lanes: lane 1-num_lanes

        Returns:
            Existence logits of shape (B, num_lanes)
        """
        x = x[:, 1:self.in_channels, :, :]  # Drop background, keep lanes: (B, num_lanes, H, W)
        x = self.conv(x)                     # (B, num_lanes, H, W) - spatially filtered
        x = self.pool(x)                     # (B, num_lanes, 1, 1)
        return x.flatten(1)                  # (B, num_lanes)
