import torch.nn as nn


class ExistHead(nn.Module):
    """
    Lane existence head for SCNN.

    Predicts whether each lane exists in the image.

    Architecture:
        Softmax → AvgPool(2x2) → Flatten → FC(N→128) → ReLU → FC(128→4)

    Output:
        4 logits, one for each lane (use BCEWithLogitsLoss for training)
    """

    def __init__(self, in_channels=5, input_height=36, input_width=100, num_lanes=4):
        super().__init__()

        # After avg pool with kernel=2, spatial dims are halved
        pooled_height = input_height // 2
        pooled_width = input_width // 2
        fc_input_features = in_channels * pooled_height * pooled_width

        self.pool = nn.Sequential(
            nn.Softmax(dim=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_input_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_lanes),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 5, H, W) - segmentation logits before upsampling

        Returns:
            Existence logits of shape (B, 4)
        """
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
