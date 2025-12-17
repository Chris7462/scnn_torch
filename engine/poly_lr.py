"""
Polynomial Learning Rate Scheduler

Implements polynomial decay with warmup as used in the reference SCNN implementation.
"""

from torch.optim.lr_scheduler import LRScheduler


class PolyLR(LRScheduler):
    """
    Polynomial learning rate decay scheduler with warmup.

    During warmup:
        lr = base_lr / warmup * (current_iter + 1)

    After warmup:
        lr = (base_lr - min_lr) * (1 - (current_iter - warmup) / (max_iter - warmup)) ^ power + min_lr

    Args:
        optimizer: Wrapped optimizer
        max_iter: Maximum number of training iterations
        power: Polynomial power (default: 0.9)
        warmup: Number of warmup iterations (default: 0)
        min_lr: Minimum learning rate (default: 1e-20)
        last_epoch: The index of last iteration (default: -1)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.16)
        >>> scheduler = PolyLR(optimizer, max_iter=130000, power=0.9, warmup=800)
        >>> for iteration in range(130000):
        >>>     train_step()
        >>>     optimizer.step()
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        max_iter: int,
        power: float = 0.9,
        warmup: int = 0,
        min_lr: float = 1e-20,
        last_epoch: int = -1
    ):
        self.max_iter = max_iter
        self.power = power
        self.warmup = warmup

        if not isinstance(min_lr, (list, tuple)):
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            self.min_lrs = list(min_lr)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current iteration."""
        # Warmup phase: linearly increase LR
        if self.last_epoch < self.warmup:
            return [
                base_lr / self.warmup * (self.last_epoch + 1)
                for base_lr in self.base_lrs
            ]

        # Poly decay phase
        if self.last_epoch < self.max_iter:
            coeff = (
                1 - (self.last_epoch - self.warmup) / (self.max_iter - self.warmup)
            ) ** self.power
        else:
            coeff = 0

        return [
            (base_lr - min_lr) * coeff + min_lr
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
