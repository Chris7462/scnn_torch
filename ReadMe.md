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
    ├── test.txt
    └── test_split/
        ├── test0_normal.txt
        ├── test1_crowd.txt
        ├── test2_hlight.txt
        ├── test3_shadow.txt
        ├── test4_noline.txt
        ├── test5_arrow.txt
        ├── test6_curve.txt
        ├── test7_cross.txt
        └── test8_night.txt
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
| `model.input_size` | [288, 800] | Input size [height, width] |
| `train.max_iter` | 90000 | Total training iterations |
| `checkpoint.interval` | 2000 | Validate and save checkpoint every N iterations |
| `logging.print_interval` | 100 | Print training metrics every N iterations |
| `optimizer.lr` | 0.01 | Base learning rate |
| `lr_scheduler.patience` | 5 | Reduce LR after N validations without improvement |
| `lr_scheduler.factor` | 0.5 | LR reduction factor |
| `lr_scheduler.min_lr` | 0.00001 | Minimum learning rate |

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
- Predictions saved to `outputs/predictions/`
- Visualizations saved to `outputs/visualizations/` (if `--visualize` enabled)

## Evaluation

Evaluate predictions against ground truth:
```bash
python tools/evaluate.py --pred_dir outputs/predictions --data_dir data/CULane
```

With different IoU threshold:
```bash
python tools/evaluate.py --pred_dir outputs/predictions --data_dir data/CULane --iou 0.3
```

Evaluation outputs:
- Per-category results saved to `outputs/evaluate/out_<category>.txt`
- Summary saved to `outputs/evaluate/summary_iou<threshold>.txt`

### Evaluation Metrics

The evaluator computes precision, recall, and F1-measure for each category:

| Category | Description |
|----------|-------------|
| Normal | Normal driving conditions |
| Crowd | Crowded traffic |
| HLight | High light / glare |
| Shadow | Shadows on road |
| No line | Faded or missing lane markings |
| Arrow | Arrow markings on road |
| Curve | Curved lanes |
| Cross | Crossroads (FP only, no GT lanes) |
| Night | Nighttime driving |

## Project Structure
```
├── configs/              # Configuration files
├── datasets/             # Dataset and transforms
├── model/                # Model architecture
│   ├── backbone/         # VGG16 backbone
│   ├── neck/             # Channel reduction
│   ├── spatial/          # Message passing module
│   ├── head/             # Segmentation and existence heads
│   └── loss/             # Loss functions
├── engine/               # Training and evaluation
├── utils/                # Utilities
│   ├── config.py         # Config loading
│   ├── data.py           # Data utilities
│   ├── logger.py         # Training logger
│   ├── metrics.py        # Metrics tracking
│   ├── postprocessing.py # Lane coordinate extraction
│   ├── visualization.py  # Lane visualization
│   └── lane_evaluation/  # CULane evaluation
│       └── culane_eval.py
└── tools/                # Scripts
    ├── train.py          # Training script
    ├── test.py           # Testing script
    └── evaluate.py       # Evaluation script
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
