# SCNN Lane Detection

PyTorch implementation of [Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://arxiv.org/abs/1712.06080) with modern architectural improvements.

## Key Differences from Original Paper

This implementation introduces several architectural improvements over the original SCNN:

| Component | Original | This Implementation |
|-----------|----------|---------------------|
| **Segmentation Head** | Bilinear interpolation (8×) | FCN-style transposed convolutions (3-stage 2× upsample) |
| **Existence Head** | AdaptiveAvgPool + FC layers (~580K params) | FCOS-style conv layers + GlobalMaxPool (~550 params) |
| **ONNX Export** | Problematic (fixed input size) | Fully compatible (resolution-agnostic) |

### Segmentation Head (FCN-style)

The original uses fixed bilinear upsampling. We replace it with learnable transposed convolutions following the FCN methodology:

```
(5, H/8, W/8) → ConvT → BN → ReLU →
(5, H/4, W/4) → ConvT → BN → ReLU →
(5, H/2, W/2) → ConvT →
(5, H, W)
```

Benefits:
- Learnable upsampling adapts to lane structures
- Uses kernel=4, stride=2, padding=1 to avoid checkerboard artifacts

### Existence Head (FCOS-style)

The original uses a large FC network with fixed spatial pooling. We replace it with convolutional layers inspired by FCOS detection heads:

```
Select lane channels (drop background) →
Conv(4→8, 5×3, dilation=2×1) → BN → ReLU →
Conv(8→4, 1×1) →
GlobalMaxPool(1,1) → Flatten
```

Benefits:
- Resolution-agnostic (works with any input size)
- ONNX-exportable
- 1000× fewer parameters (~550 vs ~580K)
- 5×3 kernel with dilation of 2 (effective RF 9×3) captures vertical lane structure

## Installation

Install the package in development mode:
```bash
cd scnn_torch
pip install -e .
```

For training, install with optional dependencies:
```bash
pip install -e .[train]
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
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
├── driver_193_90frame/
├── driver_23_30frame/
├── driver_37_30frame/
├── laneseg_label_w16/
├── laneseg_label_w16_test/
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

## Computing Dataset Statistics

Before training, you can compute the mean and standard deviation for the CULane dataset to use for normalization:
```bash
python tools/compute_mean_std.py --data_dir data/CULane
```

With custom settings:
```bash
python tools/compute_mean_std.py --data_dir data/CULane --batch_size 128 --num_workers 4 --resize_height 288 --resize_width 800
```

This will compute statistics for both original and resized images. Copy the output values to your config file (`configs/scnn_culane.yaml`) under the `normalize` section.

**Note**: The default config already includes pre-computed normalization values, so this step is optional unless you want to verify or use different resize dimensions.

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
| `model.pretrained` | true | Use ImageNet pretrained VGG16 backbone |
| `train.max_iter` | 25000 | Total training iterations |
| `checkpoint.interval` | 1000 | Validate and save checkpoint every N iterations |
| `logging.print_interval` | 100 | Print training metrics every N iterations |
| `optimizer.lr` | 0.04 | Learning rate (scaled for `batch_size=32`) |
| `optimizer.weight_decay` | 0.001 | Weight decay |
| `optimizer.nesterov` | true | Use Nesterov momentum |
| `lr_scheduler.power` | 0.9 | Polynomial decay power |
| `lr_scheduler.warmup` | 200 | Warmup iterations |

**Note on batch size and learning rate scaling:**

The original paper uses `batch_size=12` with `lr=0.01`. When changing batch size, scale learning rate proportionally:

| Batch Size | Learning Rate | Max Iter | Warmup |
|------------|---------------|----------|--------|
| 128 | 0.16 | 8,000 | 50 |
| 64 | 0.08 | 16,000 | 100 |
| 32 | 0.04 | 32,000 | 200 |
| 8 | 0.01 | 128,000 | 800 |

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
python tools/evaluate.py --config configs/scnn_culane.yaml --pred_dir outputs/predictions
```

With different IoU threshold:
```bash
python tools/evaluate.py --config configs/scnn_culane.yaml --pred_dir outputs/predictions --iou 0.3
```

Evaluation outputs:
- Per-category results saved to `outputs/evaluate/out_<category>.txt`
- Summary saved to `outputs/evaluate/summary_iou<threshold>.txt`

## Results

Trained model on CULane dataset (IoU threshold: 0.5, lane width: 30):

| Category | F1 | Precision | Recall | TP | FP | FN |
|----------|------|-----------|--------|-------|-------|-------|
| Normal | 0.9059 | 0.9045 | 0.9073 | 29739 | 3140 | 3038 |
| Crowd | 0.7072 | 0.7115 | 0.7030 | 19685 | 7981 | 8318 |
| HLight | 0.6194 | 0.6246 | 0.6142 | 1035 | 622 | 650 |
| Shadow | 0.7013 | 0.6969 | 0.7058 | 2030 | 883 | 846 |
| No line | 0.4461 | 0.4624 | 0.4309 | 6042 | 7026 | 7979 |
| Arrow | 0.8567 | 0.8661 | 0.8476 | 2697 | 417 | 485 |
| Curve | 0.6701 | 0.7067 | 0.6372 | 836 | 347 | 476 |
| Cross | N/A | N/A | N/A | 0 | 3036 | 0 |
| Night | 0.6711 | 0.6672 | 0.6750 | 14195 | 7081 | 6835 |
| **Overall** | **0.7310** | **0.7350** | **0.7271** | **76259** | **27497** | **28627** |

**Note:** Cross category only measures false positives (no ground truth lanes at crossroads).

## Model Architecture

```
Input (B, 3, H, W)       ← Any size divisible by 8
    │
    ▼
VGG16 Backbone ────────── (B, 512, H/8, W/8)
    │
    ▼
SCNN Neck ─────────────── (B, 128, H/8, W/8)
    │                     Conv(512→1024, 3×3, dilation=4) → BN → ReLU →
    │                     Conv(1024→128, 1×1) → BN → ReLU
    ▼
Message Passing ───────── (B, 128, H/8, W/8)
    │                     4-direction spatial propagation
    ▼
Seg Neck ──────────────── (B, 5, H/8, W/8)
    │                     Dropout → Conv(128→5, 1×1)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Seg Head (FCN)                    Exist Head (FCOS)
    │                                  │
    │ 3× TransposedConv                │ Conv(5×3, dilation=2) → MaxPool
    ▼                                  ▼
seg_pred (B, 5, H, W)            exist_pred (B, 4)
```

## Project Structure
```
├── pyproject.toml        # Package configuration
├── configs/              # Configuration files
│   └── scnn_culane.yaml  # CULane training config
├── scnn_torch/           # Python package
│   ├── __init__.py
│   ├── datasets/         # Dataset and transforms
│   │   ├── __init__.py
│   │   ├── culane.py     # CULane dataset class
│   │   └── transforms.py # Data augmentation transforms
│   ├── model/            # Model architecture
│   │   ├── __init__.py
│   │   ├── backbone/     # VGG16 backbone
│   │   │   └── vgg.py
│   │   ├── neck/         # Channel reduction
│   │   │   ├── scnn_neck.py  # 512→128 channel reduction
│   │   │   └── seg_neck.py   # Dropout + 128→5 segmentation output
│   │   ├── spatial/      # Message passing module
│   │   │   └── message_passing.py
│   │   ├── head/         # Output heads
│   │   │   ├── seg_head.py   # FCN-style upsampling (3-stage 2×)
│   │   │   └── exist_head.py # FCOS-style conv + pooling
│   │   ├── loss/         # Loss functions
│   │   │   └── scnn_loss.py  # Combined seg + exist loss
│   │   └── net/          # Full network
│   │       └── scnn.py   # SCNN model
│   ├── engine/           # Training and evaluation
│   │   ├── __init__.py
│   │   ├── trainer.py    # Training loop
│   │   ├── evaluator.py  # Inference and prediction saving
│   │   └── poly_lr.py    # Polynomial LR scheduler with warmup
│   └── utils/            # Utilities
│       ├── __init__.py
│       ├── config.py     # Config loading
│       ├── culane_eval.py    # CULane evaluation metrics
│       ├── data.py       # Data utilities (infinite loader)
│       ├── logger.py     # Training logger with plots
│       ├── metrics.py    # Metrics tracking
│       ├── postprocessing.py # Lane coordinate extraction
│       └── visualization.py  # Lane visualization
└── tools/                # Scripts
    ├── train.py          # Training script
    ├── test.py           # Testing script
    ├── evaluate.py       # Evaluation script
    └── compute_mean_std.py # Dataset statistics computation
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
