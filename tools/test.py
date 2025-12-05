import argparse

import torch
from torch.utils.data import DataLoader
import yaml

from datasets import CULane
from datasets.transforms import Compose, Resize, ToTensor, Normalize
from model import SCNN
from engine import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Test SCNN for lane detection')
    parser.add_argument('--config', type=str, default='configs/scnn_culane.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save predictions')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_transforms(config):
    """Build transforms for testing."""
    resize_shape = tuple(config['dataset']['resize_shape'])
    mean = tuple(config['normalize']['mean'])
    std = tuple(config['normalize']['std'])

    return Compose(
        Resize(resize_shape),
        ToTensor(),
        Normalize(mean=mean, std=std),
    )


def build_dataloader(config, transforms):
    """Build test dataloader."""
    dataset = CULane(
        root=config['dataset']['root'],
        image_set='test',
        transforms=transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True,
    )

    return dataloader


def build_model(config, device):
    """Build SCNN model."""
    input_size = tuple(config['model']['input_size'])
    ms_ks = config['model']['ms_ks']

    model = SCNN(input_size=input_size, ms_ks=ms_ks, pretrained=False)
    model = model.to(device)

    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint['net'])

    print(f"  Loaded from epoch {checkpoint['epoch']}")
    return model


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Add output_dir to config
    config['output_dir'] = args.output_dir

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build transforms
    transforms = build_transforms(config)

    # Build dataloader
    print("Building dataloader...")
    test_loader = build_dataloader(config, transforms)
    print(f"  Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(config, device)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)

    # Build evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    output_dir = evaluator.evaluate()

    print(f"\nEvaluation complete!")
    print(f"Predictions saved to: {output_dir}")
    print(f"\nTo run official CULane evaluation:")
    print(f"  cd utils/lane_evaluation/CULane")
    print(f"  ./evaluate -a <anno_dir> -d {output_dir} -i <img_dir> -l <list_file>")


if __name__ == '__main__':
    main()
