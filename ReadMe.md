# SCNN Lane Detection

PyTorch implementation of [Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://arxiv.org/abs/1712.06080).

## Installation
```bash
pip install -r requirements.txt
```

## Dataset

Download [CULane](https://xingangpan.github.io/projects/CULane.html) dataset and create a symlink:
```bash
mkdir -p data
ln -s /path/to/CULane data/CULane
```

Expected structure:
```
data/CULane/
├── driver_100_30frame/       # Test example, no ground truth
├── driver_161_90frame/       # Training example
├── driver_182_30frame/       # Training example
├── driver_193_90frame/       # Test example, no ground truth
├── driver_23_30frame/        # Training & Validation example
├── driver_37_30frame/        # Test example, no ground truth
├── laneseg_label_w16/        # Ground truth for training & validation
├── laneseg_label_w16_test/   # No ground truth for test set
└── list/
    ├── train_gt.txt
    ├── val_gt.txt
    └── test.txt
```

## Training
```bash
python tools/train.py --config configs/scnn_culane.yaml
```

Resume from checkpoint:
```bash
python tools/train.py --config configs/scnn_culane.yaml --resume checkpoints/latest.pth
```

Training outputs:
- Checkpoints saved to `checkpoints/` (configurable in config file)
- Training history plot saved as `training_history.png`

### Training Configuration

The training is iteration-based. Key settings in `configs/scnn_culane.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `train.max_iter` | 90000 | Total training iterations |
| `validation.interval` | 2000 | Validate every N iterations |
| `checkpoint.save_interval` | 2000 | Save checkpoint every N iterations |
| `optimizer.lr` | 0.01 | Base learning rate |
| `lr_scheduler.power` | 0.9 | Polynomial decay power |

## Testing
```bash
python tools/test.py --config configs/scnn_culane.yaml --checkpoint checkpoints/best.pth
```

With visualization (saves first 20 images with lane overlay):
```bash
python tools/test.py --config configs/scnn_culane.yaml --checkpoint checkpoints/best.pth --visualize
```

Customize number of visualizations:
```bash
python tools/test.py --config configs/scnn_culane.yaml --checkpoint checkpoints/best.pth --visualize --num_visualize 100
```

Test outputs:
- Predictions saved to `output/predictions/`
- Visualizations saved to `output/visualizations/` (if `--visualize` enabled)

## Project Structure
```
├── configs/          # Configuration files
├── datasets/         # Dataset and transforms
├── model/            # Model architecture
│   ├── backbone/     # VGG16 backbone
│   ├── neck/         # Channel reduction
│   ├── spatial/      # Message passing module
│   ├── head/         # Segmentation and existence heads
│   └── loss/         # Loss functions
├── engine/           # Training and evaluation
├── utils/            # Utilities (logging, visualization, config)
└── tools/            # Train and test scripts
```

## Reference
```bibtex
@inproceedings{pan2018spatial,
  title={Spatial as deep: Spatial cnn for traffic scene understanding},
  author={Pan, Xingang and Shi, Jianping and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={AAAI},
  year={2018}
}
```
