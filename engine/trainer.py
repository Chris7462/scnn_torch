import os
import time

import torch
import torch.nn as nn

from utils.tensorboard import TensorBoard
from utils.visualization import prepare_visualization_batch
from datasets.transforms import Resize


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
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
        config,
        device,
    ):
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
        self.print_interval = config['logging']['print_interval']

        # Checkpoint settings
        self.save_dir = config['checkpoint']['save_dir']
        self.save_interval = config['checkpoint']['save_interval']
        os.makedirs(self.save_dir, exist_ok=True)

        # TensorBoard
        self.tensorboard = TensorBoard(self.save_dir)

        # Visualization transform
        resize_shape = tuple(config['dataset']['resize_shape'])
        self.transform_val_img = Resize(resize_shape)

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

    def train(self):
        """Main training loop over all epochs."""
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nEpoch {epoch}/{self.epochs - 1}")
            print("-" * 40)

            # Train one epoch
            self.train_one_epoch(epoch)

            # Validate
            if self.val_loader is not None:
                print(f"\nValidation at epoch {epoch}")
                print(time.strftime('%H:%M:%S', time.localtime()))
                self.validate(epoch)

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)

        self.tensorboard.close()
        print("\nTraining completed!")

    def train_one_epoch(self, epoch):
        """
        Training logic for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()

        total_loss = 0
        total_loss_seg = 0
        total_loss_exist = 0

        for batch_idx, sample in enumerate(self.train_loader):
            img = sample['img'].to(self.device)
            seg_gt = sample['segLabel'].to(self.device)
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

            # Update statistics
            total_loss += loss.item()
            total_loss_seg += loss_seg.item()
            total_loss_exist += loss_exist.item()
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.print_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"(seg: {loss_seg.item():.4f}, exist: {loss_exist.item():.4f}) "
                      f"LR: {lr:.6f}")

            # TensorBoard logging
            self.tensorboard.scalar_summary('train/loss', loss.item(), self.global_step)
            self.tensorboard.scalar_summary('train/loss_seg', loss_seg.item(), self.global_step)
            self.tensorboard.scalar_summary('train/loss_exist', loss_exist.item(), self.global_step)
            self.tensorboard.scalar_summary('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        # Epoch summary
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_seg = total_loss_seg / len(self.train_loader)
        avg_loss_exist = total_loss_exist / len(self.train_loader)
        print(f"\n  Epoch {epoch} Summary - "
              f"Avg Loss: {avg_loss:.4f} "
              f"(seg: {avg_loss_seg:.4f}, exist: {avg_loss_exist:.4f})")

        self.tensorboard.flush()

    def validate(self, epoch):
        """
        Validation logic.

        Args:
            epoch: Current epoch number
        """
        self.model.eval()

        total_loss = 0
        total_loss_seg = 0
        total_loss_exist = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['img'].to(self.device)
                seg_gt = sample['segLabel'].to(self.device)
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

                total_loss += loss.item()
                total_loss_seg += loss_seg.item()
                total_loss_exist += loss_exist.item()

                # Visualize first few batches
                if batch_idx < 5:
                    self._visualize_batch(sample, seg_pred, exist_pred, epoch, batch_idx)

        # Compute averages
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_seg = total_loss_seg / len(self.val_loader)
        avg_loss_exist = total_loss_exist / len(self.val_loader)

        print(f"  Validation - "
              f"Avg Loss: {avg_loss:.4f} "
              f"(seg: {avg_loss_seg:.4f}, exist: {avg_loss_exist:.4f})")

        # TensorBoard logging
        val_step = (epoch + 1) * len(self.train_loader)
        self.tensorboard.scalar_summary('val/loss', avg_loss, val_step)
        self.tensorboard.scalar_summary('val/loss_seg', avg_loss_seg, val_step)
        self.tensorboard.scalar_summary('val/loss_exist', avg_loss_exist, val_step)
        self.tensorboard.flush()

        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
            print(f"  New best model saved! (loss: {avg_loss:.4f})")

    def _visualize_batch(self, sample, seg_pred, exist_pred, epoch, batch_idx):
        """
        Visualize predictions for a batch.

        Args:
            sample: Input sample dictionary
            seg_pred: Segmentation predictions
            exist_pred: Existence predictions
            epoch: Current epoch
            batch_idx: Current batch index
        """
        seg_pred_np = seg_pred.detach().cpu().numpy()
        exist_pred_np = exist_pred.detach().cpu().numpy()
        img_names = sample['img_name']

        vis_imgs = prepare_visualization_batch(
            imgs=img_names,
            seg_preds=seg_pred_np,
            exist_preds=exist_pred_np,
            transform_img=self.transform_val_img,
        )

        self.tensorboard.image_summary(f'val/batch_{batch_idx}', vis_imgs, epoch)

    def save_checkpoint(self, epoch, is_best=False):
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
        }

        # Save latest checkpoint
        save_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(state, save_path)
        print(f"  Checkpoint saved: {save_path}")

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(state, best_path)

    def load_checkpoint(self, checkpoint_path):
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
        self.global_step = self.start_epoch * len(self.train_loader)

        print(f"  Resumed from epoch {checkpoint['epoch']}, best_val_loss: {self.best_val_loss:.4f}")
