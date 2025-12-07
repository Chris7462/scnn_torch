import torch.nn as nn
from torch import Tensor
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class VGGBackbone(nn.Module):
    """
    Modified VGG16-BN backbone for SCNN.

    Modifications from standard VGG16:
    1. Last 3 conv layers use dilation=2 for larger receptive field
    2. Last 2 max pooling layers are removed to preserve spatial resolution

    Output stride is 8 instead of 32.
    """

    DILATED_LAYER_INDICES = [34, 37, 40]
    REMOVED_POOL_INDICES = [33, 43]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg = vgg16_bn(weights=weights)

        # Modify conv layers and remove pooling layers in one pass
        layers = []
        for idx, layer in enumerate(vgg.features):
            if idx in self.REMOVED_POOL_INDICES:
                continue
            if idx in self.DILATED_LAYER_INDICES:
                layer = self._make_dilated_conv(layer)
            layers.append(layer)

        self.features = nn.Sequential(*layers)

    def _make_dilated_conv(self, conv: nn.Conv2d, dilation: int = 2) -> nn.Conv2d:
        """Convert a conv layer to use dilation while preserving spatial dims."""
        # Compute padding that preserves spatial dimensions
        padding = tuple((dilation * (k - 1)) // 2 for k in conv.kernel_size)

        dilated_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=padding,
            dilation=dilation,
            bias=(conv.bias is not None)
        )
        dilated_conv.load_state_dict(conv.state_dict())
        return dilated_conv

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, 512, H/8, W/8)
        """
        return self.features(x)
