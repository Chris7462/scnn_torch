from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import TrainLogger


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
        self.epochs = config['train']['epochs']

        # Checkpoint settings
        self.save_dir = Path(config['checkpoint']['save_dir'])
        self.save_interval = config['checkpoint']['save_interval']
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = TrainLogger(self.save_dir)

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def train(self) -> None:
        """Main training loop over all epochs."""
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train one epoch
            train_metrics = self.train_one_epoch()
            self.logger.update('train', **train_metrics)

            # Validate
            val_metrics = self.validate()
            self.logger.update('val', **val_metrics)

            # Print epoch summary
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.print_epoch(epoch + 1, lr)

            # Plot training history
            self.logger.plot()

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)

            # Save best model
            val_loss = val_metrics['loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"Best model saved! (loss: {val_loss:.4f})")

            print("-" * 60)

        print("\nTraining completed!")

    def train_one_epoch(self) -> dict:
        """Training logic for one epoch."""
        self.model.train()

        total_loss = 0
        total_loss_seg = 0
        total_loss_exist = 0
        total_seg_correct = 0
        total_seg_pixels = 0
        total_exist_correct = 0
        total_exist_samples = 0

        for sample in tqdm(self.train_loader, desc="Training"):
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
            self.lr_scheduler.step()

            # Accumulate losses
            total_loss += loss.item()
            total_loss_seg += loss_seg.item()
            total_loss_exist += loss_exist.item()

            # Accumulate accuracy metrics
            with torch.no_grad():
                # Segmentation accuracy
                seg_pred_class = seg_pred.argmax(dim=1)
                total_seg_correct += (seg_pred_class == seg_gt).sum().item()
                total_seg_pixels += seg_gt.numel()

                # Existence accuracy
                exist_pred_binary = (torch.sigmoid(exist_pred) > 0.5).float()
                total_exist_correct += (exist_pred_binary == exist_gt).sum().item()
                total_exist_samples += exist_gt.numel()

        num_batches = len(self.train_loader)

        return {
            'loss': total_loss / num_batches,
            'loss_seg': total_loss_seg / num_batches,
            'loss_exist': total_loss_exist / num_batches,
            'seg_acc': total_seg_correct / total_seg_pixels,
            'exist_acc': total_exist_correct / total_exist_samples,
        }

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

        with torch.no_grad():
            for sample in tqdm(self.val_loader, desc="Validating"):
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

        num_batches = len(self.val_loader)

        return {
            'loss': total_loss / num_batches,
            'loss_seg': total_loss_seg / num_batches,
            'loss_exist': total_loss_exist / num_batches,
            'seg_acc': total_seg_correct / total_seg_pixels,
            'exist_acc': total_exist_correct / total_exist_samples,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        state = {
            'epoch': epoch,
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
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Restore history
        if 'history' in checkpoint:
            self.logger.set_history(checkpoint['history'])

        print(f"  Resumed from epoch {checkpoint['epoch'] + 1}")
