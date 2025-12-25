import torch
import torch.nn as nn
from torch import Tensor


class SCNNLoss(nn.Module):
    """
    Combined loss for SCNN.

    Combines:
        1. CrossEntropyLoss for segmentation (with background weight reduction)
        2. BCEWithLogitsLoss for lane existence prediction

    Total loss = seg_loss * seg_weight + exist_loss * exist_weight
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        exist_weight: float = 0.1,
        background_weight: float = 0.4
    ) -> None:
        super().__init__()

        self.seg_weight = seg_weight
        self.exist_weight = exist_weight

        # Background class (0) weighted less since it dominates the image
        class_weights = torch.tensor([background_weight, 1.0, 1.0, 1.0, 1.0])
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights)

        # BCEWithLogitsLoss expects logits (more numerically stable)
        self.exist_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        seg_pred: Tensor,
        exist_pred: Tensor,
        seg_gt: Tensor,
        exist_gt: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            seg_pred: Segmentation logits of shape (B, 5, H, W)
            exist_pred: Existence logits of shape (B, 4)
            seg_gt: Segmentation ground truth of shape (B, H, W)
            exist_gt: Existence ground truth of shape (B, 4)

        Returns:
            loss: Total combined loss
            loss_seg: Segmentation loss (for logging)
            loss_exist: Existence loss (for logging)
        """
        loss_seg = self.seg_loss(seg_pred, seg_gt)
        loss_exist = self.exist_loss(exist_pred, exist_gt)
        loss = loss_seg * self.seg_weight + loss_exist * self.exist_weight

        return loss, loss_seg, loss_exist
