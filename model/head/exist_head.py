import torch.nn as nn
from torch import Tensor


class ExistHead(nn.Module):
    """
    Lane existence head for SCNN.

    Predicts whether each lane exists in the image.

    Architecture:
        Softmax → AdaptiveAvgPool(18x50) → Flatten → FC(4500→128) → ReLU → FC(128→4)

    This head accepts any input spatial size due to AdaptiveAvgPool2d.
    The output size (18, 50) matches the CULane image with size of 288×800.

    Output:
        4 logits, one for each lane (use BCEWithLogitsLoss for training)
    """

    def __init__(
        self,
        in_channels: int = 5,
        pool_size: tuple[int, int] = (18, 50),
        num_lanes: int = 4
    ) -> None:
        super().__init__()

        self.pool = nn.Sequential(
            nn.Softmax(dim=1),
            nn.AdaptiveAvgPool2d(pool_size)
        )

        fc_input_features = in_channels * pool_size[0] * pool_size[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_lanes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, 5, H, W) - segmentation logits before upsampling

        Returns:
            Existence logits of shape (B, 4)
        """
        x = self.pool(x)
        x = self.fc(x)
        return x
