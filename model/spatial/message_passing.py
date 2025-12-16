import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MessagePassing(nn.Module):
    """
    Spatial message passing module for SCNN.

    Propagates information across the feature map in 4 directions:
        - Top to Bottom (down)
        - Bottom to Top (up)
        - Left to Right (right)
        - Right to Left (left)

    This captures the long, continuous structure of lane lines by allowing
    each pixel to "see" information from the entire row/column.
    """

    def __init__(self, channels: int = 128, kernel_size: int = 9) -> None:
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        # Vertical propagation: kernel (1, k) captures horizontal context at each row for lane width
        self.conv_down = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.conv_up = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), bias=False)

        # Horizontal propagation: kernel (k, 1) captures vertical context at each column for lane height
        self.conv_right = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        self.conv_left = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, C, H, W) with spatial context propagated
        """
        x = self._propagate(x, self.conv_down, dim=2, reverse=False)  # Top to Bottom
        x = self._propagate(x, self.conv_up, dim=2, reverse=True)     # Bottom to Top
        x = self._propagate(x, self.conv_right, dim=3, reverse=False) # Left to Right
        x = self._propagate(x, self.conv_left, dim=3, reverse=True)   # Right to Left
        return x

    def _propagate(self, x: Tensor, conv: nn.Conv2d, dim: int, reverse: bool) -> Tensor:
        """
        Propagate information along one direction.

        Args:
            x: Input tensor of shape (B, C, H, W)
            conv: Convolution layer for message passing
            dim: Dimension to propagate along (2 for vertical, 3 for horizontal)
            reverse: If True, propagate in reverse direction

        Returns:
            Feature tensor with information propagated along the specified direction
        """
        # Split tensor into slices along the specified dimension
        slices = list(x.split(1, dim=dim))

        if reverse:
            slices = slices[::-1]

        # Sequential propagation: each slice receives info from previous slice
        out = [slices[0]]
        for slice_tensor in slices[1:]:
            out.append(slice_tensor + F.relu(conv(out[-1])))

        if reverse:
            out = out[::-1]

        return torch.cat(out, dim=dim)
