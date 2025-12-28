import torch


class Metrics:
    """
    Tracks and computes running metrics for SCNN training.

    Accumulates losses and accuracies across batches,
    then computes averages when requested.

    Example:
        >>> metrics = Metrics()
        >>> for batch in dataloader:
        >>>     # ... forward pass ...
        >>>     metrics.update(loss, loss_seg, loss_exist, seg_pred, exist_pred, seg_gt, exist_gt)
        >>>
        >>> results = metrics.get_metrics()
        >>> print(f"Average loss: {results['loss']:.4f}")
        >>> metrics.reset()
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics to zero."""
        self.running_loss = 0.0
        self.running_loss_seg = 0.0
        self.running_loss_exist = 0.0
        self.running_seg_correct = 0
        self.running_seg_pixels = 0
        self.running_exist_correct = 0
        self.running_exist_samples = 0
        self.count = 0

    def update(
        self,
        loss: float,
        loss_seg: float,
        loss_exist: float,
        seg_pred: torch.Tensor,
        exist_pred: torch.Tensor,
        seg_gt: torch.Tensor,
        exist_gt: torch.Tensor,
    ) -> None:
        """
        Update metrics with current batch results.

        Args:
            loss: Total loss value
            loss_seg: Segmentation loss value
            loss_exist: Existence loss value
            seg_pred: Segmentation logits (B, 5, H, W)
            exist_pred: Existence logits (B, 4)
            seg_gt: Segmentation ground truth (B, H, W)
            exist_gt: Existence ground truth (B, 4)
        """
        # Accumulate losses
        self.running_loss += loss
        self.running_loss_seg += loss_seg
        self.running_loss_exist += loss_exist
        self.count += 1

        with torch.no_grad():
            # Segmentation accuracy
            seg_pred_class = seg_pred.argmax(dim=1)
            self.running_seg_correct += (seg_pred_class == seg_gt).sum().item()
            self.running_seg_pixels += seg_gt.numel()

            # Existence accuracy (exist_pred is logits, threshold at 0)
            exist_pred_binary = (exist_pred > 0).float()
            self.running_exist_correct += (exist_pred_binary == exist_gt).sum().item()
            self.running_exist_samples += exist_gt.numel()

    def get_metrics(self) -> dict:
        """
        Get averaged metrics.

        Returns:
            Dictionary with averaged metrics:
                - loss: Average total loss
                - loss_seg: Average segmentation loss
                - loss_exist: Average existence loss
                - seg_acc: Segmentation accuracy
                - exist_acc: Existence accuracy

        Raises:
            ValueError: If no metrics have been accumulated (count is 0)
        """
        if self.count == 0:
            raise ValueError("No metrics to compute. Call update() first.")

        return {
            'loss': self.running_loss / self.count,
            'loss_seg': self.running_loss_seg / self.count,
            'loss_exist': self.running_loss_exist / self.count,
            'seg_acc': self.running_seg_correct / self.running_seg_pixels,
            'exist_acc': self.running_exist_correct / self.running_exist_samples,
        }
