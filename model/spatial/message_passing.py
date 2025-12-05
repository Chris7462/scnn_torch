import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, channels=128, kernel_size=9):
        super().__init__()

        # Vertical convolutions: kernel (1, k) - wide to capture lane width
        self.conv_down = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.conv_up = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), bias=False)

        # Horizontal convolutions: kernel (k, 1) - tall to capture lane height
        self.conv_right = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        self.conv_left = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)

    def forward(self, x):
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

    def _propagate(self, x, conv, dim, reverse):
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
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))

        if reverse:
            out = out[::-1]

        return torch.cat(out, dim=dim)
