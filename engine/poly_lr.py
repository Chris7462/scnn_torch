"""
Polynomial Learning Rate Scheduler

Implements polynomial decay as used in the reference SCNN implementation.
"""

from torch.optim.lr_scheduler import LRScheduler


class PolyLR(LRScheduler):
    """
    Polynomial learning rate decay scheduler.

    Learning rate is decayed according to:
        lr = base_lr * (1 - current_iter / max_iter) ^ power

    This provides smooth, continuous decay throughout training, which is
    better than ReduceLROnPlateau for fixed-iteration training.

    Args:
        optimizer: Wrapped optimizer
        max_iter: Maximum number of training iterations
        power: Polynomial power (default: 0.9)
        last_epoch: The index of last iteration (default: -1)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> scheduler = PolyLR(optimizer, max_iter=90000, power=0.9)
        >>> for iteration in range(90000):
        >>>     train_step()
        >>>     optimizer.step()
        >>>     scheduler.step()  # Call after EVERY iteration

    Note:
        - Call scheduler.step() after every optimizer.step()
        - Do NOT call it only at validation time
        - This is different from ReduceLROnPlateau!
    """

    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current iteration."""
        # Compute polynomial decay factor
        # At iter 0: factor = 1.0
        # At iter max_iter: factor = 0.0
        factor = (1 - self.last_epoch / self.max_iter) ** self.power

        # Prevent negative learning rates
        factor = max(0.0, factor)

        return [base_lr * factor for base_lr in self.base_lrs]
