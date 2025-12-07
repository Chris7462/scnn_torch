import torch.nn as nn
from torch import Tensor

from ..backbone import VGGBackbone
from ..neck import SCNNNeck, SegNeck
from ..spatial import MessagePassing
from ..head import SegHead, ExistHead


class SCNN(nn.Module):
    """
    Spatial CNN for lane detection.

    Architecture:
        Input (B, 3, 288, 800)
            │
            ▼
        backbone ──────────────── (B, 512, 36, 100)
            │
            ▼
        scnn_neck ─────────────── (B, 128, 36, 100)
            │
            ▼
        message_passing ───────── (B, 128, 36, 100)
            │
            ▼
        seg_neck ──────────────── (B, 5, 36, 100)    # shared features
            │
            ├──────────────────────────────────┐
            ▼                                  ▼
        seg_head                          exist_head
            │                                  │
            ▼                                  ▼
        seg_pred ───── (B,5,288,800)      exist_pred ───── (B, 4)

    Reference:
        "Spatial As Deep: Spatial CNN for Traffic Scene Understanding"
        https://arxiv.org/abs/1712.06080
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (800, 288),
        ms_ks: int = 9,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            input_size: (width, height) of input image
            ms_ks: Kernel size for message passing convolutions
            pretrained: Whether to use pretrained VGG16 backbone
        """
        super().__init__()

        input_width, input_height = input_size
        feature_height = input_height // 8
        feature_width = input_width // 8

        self.backbone = VGGBackbone(pretrained=pretrained)
        self.scnn_neck = SCNNNeck(in_channels=512, mid_channels=1024, out_channels=128)
        self.message_passing = MessagePassing(channels=128, kernel_size=ms_ks)
        self.seg_neck = SegNeck(in_channels=128, out_channels=5)
        self.seg_head = SegHead(upsample_scale=8)
        self.exist_head = ExistHead(in_channels=5, input_height=feature_height, input_width=feature_width)

        if not pretrained:
            self._initialize_weights()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            seg_pred: Segmentation logits of shape (B, 5, H, W)
            exist_pred: Existence probabilities of shape (B, 4)
        """
        x = self.backbone(x)
        x = self.scnn_neck(x)
        x = self.message_passing(x)
        x = self.seg_neck(x)

        seg_pred = self.seg_head(x)
        exist_pred = self.exist_head(x)

        return seg_pred, exist_pred

    def _initialize_weights(self) -> None:
        """
        Initialize model weights using best practices.

        Applies:
        - Kaiming normal initialization for Conv2d layers (optimal for ReLU)
        - Constant initialization for BatchNorm2d (weight=1, bias=0)
        - Normal initialization for Linear layers (mean=0, std=0.01)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
