from pathlib import Path

import torch
import torch.nn as nn

from utils import Logger


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


class Trainer:
    """
    Trainer for SCNN model.

    Args:
        model: SCNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (SCNNLoss)
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
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
        lr_scheduler,
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
        self.val_interval = config['validation']['interval']
        self.print_interval = config['logging']['print_interval']

        # Checkpoint settings
        self.save_dir = Path(config['checkpoint']['save_dir'])
        self.save_interval = config['checkpoint']['save_interval']
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = Logger(self.save_dir)

        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')

        # Running metrics for training (accumulated between validations)
        self._reset_running_metrics()

    def _reset_running_metrics(self) -> None:
        """Reset running metrics for training."""
        self.running_loss = 0.0
        self.running_loss_seg = 0.0
        self.running_loss_exist = 0.0
        self.running_seg_correct = 0
        self.running_seg_pixels = 0
        self.running_exist_correct = 0
        self.running_exist_samples = 0
        self.running_count = 0

    def _update_running_metrics(
        self,
        loss: float,
        loss_seg: float,
        loss_exist: float,
        seg_pred: torch.Tensor,
        exist_pred: torch.Tensor,
        seg_gt: torch.Tensor,
        exist_gt: torch.Tensor,
    ) -> None:
        """Update running metrics with current batch."""
        self.running_loss += loss
        self.running_loss_seg += loss_seg
        self.running_loss_exist += loss_exist
        self.running_count += 1

        with torch.no_grad():
            # Segmentation accuracy
            seg_pred_class = seg_pred.argmax(dim=1)
            self.running_seg_correct += (seg_pred_class == seg_gt).sum().item()
            self.running_seg_pixels += seg_gt.numel()

            # Existence accuracy
            exist_pred_binary = (torch.sigmoid(exist_pred) > 0.5).float()
            self.running_exist_correct += (exist_pred_binary == exist_gt).sum().item()
            self.running_exist_samples += exist_gt.numel()

    def _get_running_metrics(self) -> dict:
        """Get averaged running metrics."""
        return {
            'loss': self.running_loss / self.running_count,
            'loss_seg': self.running_loss_seg / self.running_count,
            'loss_exist': self.running_loss_exist / self.running_count,
            'seg_acc': self.running_seg_correct / self.running_seg_pixels,
            'exist_acc': self.running_exist_correct / self.running_exist_samples,
        }

    def _print_training_metrics(self, cur_iter: int, loss: float, loss_seg: float, loss_exist: float) -> None:
        """Print current training metrics."""
        lr = self.optimizer.param_groups[0]['lr']
        metrics = self._get_running_metrics()

        print(
            f"Iter [{cur_iter + 1}/{self.max_iter}] "
            f"LR: {lr:.6f} | "
            f"Loss: {loss:.4f} (seg: {loss_seg:.4f}, exist: {loss_exist:.4f}) | "
            f"Seg Acc: {metrics['seg_acc']:.4f}, Exist Acc: {metrics['exist_acc']:.4f}"
        )

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

            # Handle DataParallel
            if isinstance(self.model, nn.DataParallel):
                loss = loss.mean()
                loss_seg = loss_seg.mean()
                loss_exist = loss_exist.mean()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update running metrics
            self._update_running_metrics(
                loss.item(), loss_seg.item(), loss_exist.item(),
                seg_pred, exist_pred, seg_gt, exist_gt
            )

            # Print training metrics
            if (cur_iter + 1) % self.print_interval == 0:
                self._print_training_metrics(cur_iter, loss.item(), loss_seg.item(), loss_exist.item())

            # Validation and checkpoint at intervals
            if (cur_iter + 1) % self.val_interval == 0:
                # Get training metrics
                train_metrics = self._get_running_metrics()
                self._reset_running_metrics()

                # Validate
                val_metrics = self.validate()

                # Step scheduler with validation loss
                self.lr_scheduler.step(val_metrics['loss'])

                # Log metrics
                self.logger.update(cur_iter + 1, train_metrics, val_metrics)

                # Print summary
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.print_iteration(cur_iter + 1, self.max_iter, lr)

                # Plot training history
                self.logger.plot()

                # Save checkpoint
                if (cur_iter + 1) % self.save_interval == 0:
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

        total_loss = 0
        total_loss_seg = 0
        total_loss_exist = 0
        total_seg_correct = 0
        total_seg_pixels = 0
        total_exist_correct = 0
        total_exist_samples = 0

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

                # Handle DataParallel
                if isinstance(self.model, nn.DataParallel):
                    loss = loss.mean()
                    loss_seg = loss_seg.mean()
                    loss_exist = loss_exist.mean()

                # Accumulate losses
                total_loss += loss.item()
                total_loss_seg += loss_seg.item()
                total_loss_exist += loss_exist.item()

                # Accumulate accuracy metrics
                # Segmentation accuracy
                seg_pred_class = seg_pred.argmax(dim=1)
                total_seg_correct += (seg_pred_class == seg_gt).sum().item()
                total_seg_pixels += seg_gt.numel()

                # Existence accuracy
                exist_pred_binary = (torch.sigmoid(exist_pred) > 0.5).float()
                total_exist_correct += (exist_pred_binary == exist_gt).sum().item()
                total_exist_samples += exist_gt.numel()

        return {
            'loss': total_loss / num_batches,
            'loss_seg': total_loss_seg / num_batches,
            'loss_exist': total_loss_exist / num_batches,
            'seg_acc': total_seg_correct / total_seg_pixels,
            'exist_acc': total_exist_correct / total_exist_samples,
        }

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            iteration: Current iteration number
            is_best: Whether this is the best model so far
        """
        state = {
            'iteration': iteration,
            'net': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
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
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['net'])
        else:
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
