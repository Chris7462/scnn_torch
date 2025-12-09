from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


ImageSet = Literal['train', 'val', 'test']


class CULane(Dataset):
    """
    CULane dataset for lane detection.

    Args:
        root: Path to CULane dataset root directory
        image_set: One of 'train', 'val', or 'test'
        transforms: Optional transforms to apply to samples

    Dataset structure expected:
        root/
        ├── driver_23_30frame/  # Training & Validation example
        ├── driver_161_90frame/ # Training example
        ├── driver_182_30frame/ # Training example
        ├── driver_37_30frame/  # Test example, no ground truth
        ├── driver_100_30frame/ # Test example, no ground truth
        ├── driver_193_90frame/ # Test example, no ground truth
        ├── laneseg_label_w16/  # Ground truth for training & validation
        ├── laneseg_label_w16_test/ # No ground truth for test set
        └── list/
            ├── train_gt.txt
            ├── val_gt.txt
            └── test.txt
    """

    def __init__(
        self,
        root: str | Path,
        image_set: ImageSet,
        transforms: callable = None,
    ) -> None:
        super().__init__()
        assert image_set in ('train', 'val', 'test'), "image_set must be 'train', 'val', or 'test'"

        self.root = Path(root)
        self.image_set = image_set
        self.transforms = transforms

        self.img_list: list[Path] = []
        self.seg_label_list: list[Path] = []
        self.exist_list: list[list[int]] = []

        if image_set != 'test':
            self._load_annotations()
        else:
            self._load_test_list()

    def _load_annotations(self) -> None:
        """Load annotations for train/val sets."""
        list_file = self.root / 'list' / f'{self.image_set}_gt.txt'

        with open(list_file) as f:
            for line in f:
                parts = line.strip().split(' ')

                # Remove leading '/' for path joining
                self.img_list.append(self.root / parts[0].lstrip('/'))
                self.seg_label_list.append(self.root / parts[1].lstrip('/'))
                self.exist_list.append([int(x) for x in parts[2:]])

    def _load_test_list(self) -> None:
        """Load image list for test set."""
        list_file = self.root / 'list' / 'test.txt'

        with open(list_file) as f:
            for line in f:
                self.img_list.append(self.root / line.strip().lstrip('/'))

    def __getitem__(self, idx: int) -> dict:
        img = cv2.imread(str(self.img_list[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_set != 'test':
            seg_label = cv2.imread(str(self.seg_label_list[idx]))[:, :, 0]
            exist = np.array(self.exist_list[idx])
        else:
            seg_label = None
            exist = None

        sample = {
            'img': img,
            'seg_label': seg_label,
            'exist': exist,
            'img_name': str(self.img_list[idx]),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.img_list)

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Dictionary with batched tensors
        """
        if isinstance(batch[0]['img'], Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['seg_label'] is None:
            seg_label = None
            exist = None
        elif isinstance(batch[0]['seg_label'], Tensor):
            seg_label = torch.stack([b['seg_label'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            seg_label = [b['seg_label'] for b in batch]
            exist = [b['exist'] for b in batch]

        return {
            'img': img,
            'seg_label': seg_label,
            'exist': exist,
            'img_name': [b['img_name'] for b in batch],
        }
