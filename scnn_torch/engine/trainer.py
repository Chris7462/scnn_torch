from pathlib import Path

import torch
import torch.nn as nn

from ..utils import Logger, Metrics, infinite_loader
from .poly_lr import PolyLR


class Trainer:
    """
    Trainer for SCNN model.

    Args:
        model: SCNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (SCNNLoss)
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler (PolyLR)
        config: Configuration dictionary
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: PolyLR,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device

        # Training settings
        self.max_iter = config['train']['max_iter']
        self.checkpoint_interval = config['checkpoint']['interval']
        self.print_interval = config['logging']['print_interval']

        # Checkpoint settings
        self.save_dir = Path(config['checkpoint']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logger and metrics
        self.logger = Logger(self.save_dir)
        self.metrics = Metrics()

        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')

    def train(self) -> None:
        """Main training loop over all iterations."""
        train_iter = iter(infinite_loader(self.train_loader))

        self.model.train()

        for cur_iter in range(self.start_iter, self.max_iter):
            # Get next batch
            sample = next(train_iter)
            img = sample['img'].to(self.device)
            seg_gt = sample['seg_label'].to(self.device)
            exist_gt = sample['exist'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            seg_pred, exist_pred = self.model(img)

            # Compute loss
            loss, loss_seg, loss_exist = self.criterion(seg_pred, exist_pred, seg_gt, exist_gt)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update metrics
            self.metrics.update(
                loss.item(), loss_seg.item(), loss_exist.item(),
                seg_pred, exist_pred, seg_gt, exist_gt
            )

            # Print training metrics
            if (cur_iter + 1) % self.print_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                m = self.metrics.get_metrics()
                print(
                    f"Iter [{cur_iter + 1}/{self.max_iter}] LR: {lr:.6f} | "
                    f"Loss: {loss.item():.4f} (seg: {loss_seg.item():.4f}, exist: {loss_exist.item():.4f}) | "
                    f"Seg Acc: {m['seg_acc']:.4f}, Exist Acc: {m['exist_acc']:.4f}"
                )

            # Validation and checkpoint at intervals
            if (cur_iter + 1) % self.checkpoint_interval == 0:
                # Get training metrics
                train_metrics = self.metrics.get_metrics()
                self.metrics.reset()

                # Validate
                val_metrics = self.validate()

                # Log metrics
                self.logger.update(cur_iter + 1, train_metrics, val_metrics)

                # Print summary
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.print_iteration(cur_iter + 1, self.max_iter, lr)

                # Plot training history
                self.logger.plot()

                # Save checkpoint after each validation
                self.save_checkpoint(cur_iter + 1)

                # Save best model
                val_loss = val_metrics['loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(cur_iter + 1, is_best=True)
                    print(f"Best model saved! (loss: {val_loss:.4f})")

                print("-" * 60)

                # Switch back to training mode
                self.model.train()

        print("\nTraining completed!")

    def validate(self) -> dict:
        """Validation logic."""
        self.model.eval()

        val_metrics = Metrics()

        num_batches = len(self.val_loader)
        print(f"Validating... ({num_batches} batches)")

        with torch.no_grad():
            for sample in self.val_loader:
                img = sample['img'].to(self.device)
                seg_gt = sample['seg_label'].to(self.device)
                exist_gt = sample['exist'].to(self.device)

                # Forward pass
                seg_pred, exist_pred = self.model(img)

                # Compute loss
                loss, loss_seg, loss_exist = self.criterion(seg_pred, exist_pred, seg_gt, exist_gt)

                # Update metrics
                val_metrics.update(
                    loss.item(), loss_seg.item(), loss_exist.item(),
                    seg_pred, exist_pred, seg_gt, exist_gt
                )

        return val_metrics.get_metrics()

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            iteration: Current iteration number
            is_best: Whether this is the best model so far
        """
        state = {
            'iteration': iteration,
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.logger.get_history(),
        }

        # Save latest checkpoint
        save_path = self.save_dir / 'latest.pth'
        torch.save(state, save_path)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(state, best_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['net'])

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # Restore training state
        self.start_iter = checkpoint['iteration']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Restore history
        if 'history' in checkpoint:
            self.logger.set_history(checkpoint['history'])

        print(f"  Resumed from iteration {checkpoint['iteration']}")
