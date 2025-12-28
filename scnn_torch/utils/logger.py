from pathlib import Path
import os

import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Logger:
    """
    Training logger with history tracking and plotting.

    Tracks losses and accuracies during training, prints summaries,
    and generates training history plots. Logs at each validation point.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """
        Args:
            log_dir: Directory to save plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'iteration': [],
            'train_loss': [],
            'train_loss_seg': [],
            'train_loss_exist': [],
            'train_seg_acc': [],
            'train_exist_acc': [],
            'val_loss': [],
            'val_loss_seg': [],
            'val_loss_exist': [],
            'val_seg_acc': [],
            'val_exist_acc': [],
        }

    def update(
        self,
        iteration: int,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """
        Update history with metrics from one validation point.

        Args:
            iteration: Current iteration number
            train_metrics: Training metrics dict with keys:
                loss, loss_seg, loss_exist, seg_acc, exist_acc
            val_metrics: Validation metrics dict with same keys
        """
        self.history['iteration'].append(iteration)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_loss_seg'].append(train_metrics['loss_seg'])
        self.history['train_loss_exist'].append(train_metrics['loss_exist'])
        self.history['train_seg_acc'].append(train_metrics['seg_acc'])
        self.history['train_exist_acc'].append(train_metrics['exist_acc'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_loss_seg'].append(val_metrics['loss_seg'])
        self.history['val_loss_exist'].append(val_metrics['loss_exist'])
        self.history['val_seg_acc'].append(val_metrics['seg_acc'])
        self.history['val_exist_acc'].append(val_metrics['exist_acc'])

    def print_iteration(self, iteration: int, max_iter: int, lr: float) -> None:
        """
        Print iteration summary to screen.

        Args:
            iteration: Current iteration number
            max_iter: Maximum iterations
            lr: Current learning rate
        """
        train_loss = self.history['train_loss'][-1]
        train_loss_seg = self.history['train_loss_seg'][-1]
        train_loss_exist = self.history['train_loss_exist'][-1]
        train_seg_acc = self.history['train_seg_acc'][-1]
        train_exist_acc = self.history['train_exist_acc'][-1]

        val_loss = self.history['val_loss'][-1]
        val_loss_seg = self.history['val_loss_seg'][-1]
        val_loss_exist = self.history['val_loss_exist'][-1]
        val_seg_acc = self.history['val_seg_acc'][-1]
        val_exist_acc = self.history['val_exist_acc'][-1]

        print(f"\nIteration {iteration}/{max_iter} Summary:")
        print(f"  LR: {lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f} (seg: {train_loss_seg:.4f}, exist: {train_loss_exist:.4f}), "
              f"Seg Acc: {train_seg_acc:.4f}, Exist Acc: {train_exist_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} (seg: {val_loss_seg:.4f}, exist: {val_loss_exist:.4f}), "
              f"Seg Acc: {val_seg_acc:.4f}, Exist Acc: {val_exist_acc:.4f}")

    def plot(self, save_name: str = 'training_history.png') -> None:
        """
        Plot training history and save to file.

        Args:
            save_name: Filename for the plot
        """
        plt.style.use('ggplot')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = self.history['iteration']

        # Plot 1: Total Loss
        axes[0, 0].plot(iterations, self.history['train_loss'], label='Train',
                        marker='o', markersize=3, linewidth=2)
        axes[0, 0].plot(iterations, self.history['val_loss'], label='Val',
                        marker='s', markersize=3, linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Seg Loss & Exist Loss
        axes[0, 1].plot(iterations, self.history['train_loss_seg'], label='Train Seg',
                        marker='o', markersize=3, linewidth=2)
        axes[0, 1].plot(iterations, self.history['val_loss_seg'], label='Val Seg',
                        marker='s', markersize=3, linewidth=2)
        axes[0, 1].plot(iterations, self.history['train_loss_exist'], label='Train Exist',
                        marker='o', markersize=3, linewidth=2, linestyle='--')
        axes[0, 1].plot(iterations, self.history['val_loss_exist'], label='Val Exist',
                        marker='s', markersize=3, linewidth=2, linestyle='--')
        axes[0, 1].set_title('Segmentation & Existence Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Seg Accuracy
        axes[1, 0].plot(iterations, self.history['train_seg_acc'], label='Train',
                        marker='o', markersize=3, linewidth=2)
        axes[1, 0].plot(iterations, self.history['val_seg_acc'], label='Val',
                        marker='s', markersize=3, linewidth=2)
        axes[1, 0].set_title('Segmentation Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Exist Accuracy
        axes[1, 1].plot(iterations, self.history['train_exist_acc'], label='Train',
                        marker='o', markersize=3, linewidth=2)
        axes[1, 1].plot(iterations, self.history['val_exist_acc'], label='Val',
                        marker='s', markersize=3, linewidth=2)
        axes[1, 1].set_title('Existence Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Iteration', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.log_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def get_history(self) -> dict:
        """Return the history dictionary."""
        return self.history

    def set_history(self, history: dict) -> None:
        """Set the history dictionary (for resuming training)."""
        self.history = history
