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
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
├── driver_193_90frame/
├── driver_23_30frame/
├── driver_37_30frame/
├── laneseg_label_w16/
├── laneseg_label_w16_test/
└── list/
```

## Training
```bash
python tools/train.py --config configs/scnn_culane.yaml
```

Resume from checkpoint:
```bash
python tools/train.py --config configs/scnn_culane.yaml --resume checkpoints/latest.pth
```

## Testing
```bash
python tools/test.py --config configs/scnn_culane.yaml --checkpoint checkpoints/best.pth
```

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
├── utils/            # Utilities
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
