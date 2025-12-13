import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets import CULane
from datasets.transforms import get_train_transforms, get_val_transforms
from model import SCNN
from model.loss import SCNNLoss
from engine import Trainer
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train SCNN for lane detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def build_transforms(config: dict, is_train: bool = True):
    """Build transforms for training or validation."""
    resize_shape = tuple(config['dataset']['resize_shape'])
    mean = tuple(config['normalize']['mean'])
    std = tuple(config['normalize']['std'])

    if is_train:
        aug_cfg = config['augmentation']
        return get_train_transforms(
            resize_shape,
            mean,
            std,
            rotation=aug_cfg['rotation'],
            horizontal_flip_prob=aug_cfg['horizontal_flip_prob'],
            color_jitter_prob=aug_cfg['color_jitter_prob'],
            motion_blur_prob=aug_cfg['motion_blur_prob'],
        )
    else:
        return get_val_transforms(resize_shape, mean, std)


def build_dataloader(config: dict, image_set: str, transforms):
    """Build dataloader for given image set."""
    dataset = CULane(
        root=config['dataset']['root'],
        image_set=image_set,
        transforms=transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=(image_set == 'train'),
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True,
        drop_last=(image_set == 'train'),
    )

    return dataloader


def build_model(config: dict):
    """Build SCNN model."""
    input_size = tuple(config['model']['input_size'])
    ms_ks = config['model']['ms_ks']
    pretrained = config['model']['pretrained']
    model = SCNN(input_size=input_size, ms_ks=ms_ks, pretrained=pretrained)

    return model


def build_optimizer(config: dict, model):
    """Build optimizer."""
    optimizer_cfg = config['optimizer']

    optimizer = optim.SGD(
        model.parameters(),
        lr=optimizer_cfg['lr'],
        momentum=optimizer_cfg['momentum'],
        weight_decay=optimizer_cfg['weight_decay'],
    )

    return optimizer


def build_lr_scheduler(config: dict, optimizer):
    """Build learning rate scheduler."""
    scheduler_cfg = config['lr_scheduler']

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg['mode'],
        factor=scheduler_cfg['factor'],
        patience=scheduler_cfg['patience'],
        min_lr=scheduler_cfg['min_lr'],
    )

    return scheduler


def build_criterion(config: dict):
    """Build loss function."""
    loss_cfg = config['loss']

    criterion = SCNNLoss(
        seg_weight=loss_cfg['seg_weight'],
        exist_weight=loss_cfg['exist_weight'],
        background_weight=loss_cfg['background_weight'],
    )

    return criterion


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build transforms
    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)

    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config, 'train', train_transforms)
    val_loader = build_dataloader(config, 'val', val_transforms)
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(config).to(device)

    # Build optimizer
    optimizer = build_optimizer(config, model)

    # Build lr scheduler
    lr_scheduler = build_lr_scheduler(config, optimizer)

    # Build criterion
    criterion = build_criterion(config).to(device)

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print(f"  Max iterations: {config['train']['max_iter']}")
    print(f"  Checkpoint interval: {config['checkpoint']['interval']}")
    trainer.train()


if __name__ == '__main__':
    main()
