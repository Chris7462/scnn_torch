from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class PolyLR(LRScheduler):
    """
    Polynomial learning rate scheduler with warmup.

    Learning rate is decayed using polynomial function:
        lr = (base_lr - min_lr) * (1 - iter/max_iter)^power + min_lr

    During warmup, learning rate linearly increases from 0 to base_lr.

    Args:
        optimizer: Wrapped optimizer
        power: Power of polynomial decay
        max_iter: Maximum number of iterations
        min_lr: Minimum learning rate (can be single value or list for each param group)
        warmup: Number of warmup iterations
        last_epoch: Index of last epoch (for resuming)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        power: float,
        max_iter: int,
        min_lr: float | list[float] = 1e-10,
        warmup: int = 0,
        last_epoch: int = -1,
    ) -> None:
        self.power = power
        self.max_iter = max_iter
        self.warmup = max(warmup, 0)

        # Handle min_lr for each param group
        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current iteration."""
        # Warmup phase: linearly increase lr
        if self.warmup > 0 and self.last_epoch < self.warmup:
            return [base_lr * (self.last_epoch + 1) / self.warmup for base_lr in self.base_lrs]

        # Polynomial decay phase
        if self.last_epoch < self.max_iter:
            coeff = (1 - (self.last_epoch - self.warmup) / (self.max_iter - self.warmup)) ** self.power
        else:
            coeff = 0

        return [
            (base_lr - min_lr) * coeff + min_lr
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
