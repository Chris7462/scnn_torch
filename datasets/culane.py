from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A


ImageSet = Literal['train', 'val', 'test']


class CULane(Dataset):
    """
    CULane dataset for lane detection.

    Args:
        root: Path to CULane dataset root directory
        image_set: One of 'train', 'val', or 'test'
        transforms: Optional albumentations transforms to apply to samples
    """

    def __init__(
        self,
        root: str | Path,
        image_set: ImageSet,
        transforms: A.Compose = None,
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
            exist = np.array(self.exist_list[idx], dtype=np.float32)
        else:
            seg_label = None
            exist = None

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=seg_label)
            img = transformed['image']
            if seg_label is not None:
                seg_label = transformed['mask'].long()

        if exist is not None:
            exist = torch.from_numpy(exist)

        return {
            'img': img,
            'seg_label': seg_label,
            'exist': exist,
            'img_name': str(self.img_list[idx]),
        }

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
        img = torch.stack([b['img'] for b in batch])
        img_name = [b['img_name'] for b in batch]

        if batch[0]['seg_label'] is None:
            return {'img': img, 'seg_label': None, 'exist': None, 'img_name': img_name}

        seg_label = torch.stack([b['seg_label'] for b in batch])
        exist = torch.stack([b['exist'] for b in batch])

        return {'img': img, 'seg_label': seg_label, 'exist': exist, 'img_name': img_name}
