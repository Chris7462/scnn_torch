import torch
import torch.nn as nn


class SCNNLoss(nn.Module):
    """
    Combined loss for SCNN.

    Combines:
        1. CrossEntropyLoss for segmentation (with background weight reduction)
        2. BCELoss for lane existence prediction

    Total loss = seg_loss * seg_weight + exist_loss * exist_weight
    """

    def __init__(self, seg_weight=1.0, exist_weight=0.1, background_weight=0.4):
        super().__init__()

        self.seg_weight = seg_weight
        self.exist_weight = exist_weight

        # Background class (0) weighted less since it dominates the image
        class_weights = torch.tensor([background_weight, 1.0, 1.0, 1.0, 1.0])
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.exist_loss = nn.BCELoss()

    def forward(self, seg_pred, exist_pred, seg_gt, exist_gt):
        """
        Args:
            seg_pred: Segmentation logits of shape (B, 5, H, W)
            exist_pred: Existence probabilities of shape (B, 4)
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
