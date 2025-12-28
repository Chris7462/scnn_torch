import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A


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
        root: str,
        image_set: str,
        transforms: A.Compose = None,
    ) -> None:
        super().__init__()
        assert image_set in ('train', 'val', 'test'), "image_set must be 'train', 'val', or 'test'"

        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        self.img_list: list[str] = []
        self.seg_label_list: list[str] = []
        self.exist_list: list[list[int]] = []

        if image_set != 'test':
            self._load_annotations()
        else:
            self._load_test_list()

    def _load_annotations(self) -> None:
        """Load annotations for train/val sets."""
        list_file = os.path.join(self.root, 'list', f'{self.image_set}_gt.txt')

        with open(list_file) as f:
            for line in f:
                parts = line.strip().split(' ')

                self.img_list.append(os.path.join(self.root, parts[0].lstrip('/')))
                self.seg_label_list.append(os.path.join(self.root, parts[1].lstrip('/')))
                self.exist_list.append([int(x) for x in parts[2:]])

    def _load_test_list(self) -> None:
        """Load image list for test set."""
        list_file = os.path.join(self.root, 'list', 'test.txt')

        with open(list_file) as f:
            for line in f:
                self.img_list.append(os.path.join(self.root, line.strip().lstrip('/')))

    def __getitem__(self, idx: int) -> dict:
        img = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Store original size before transforms
        original_size = img.shape[:2]  # (H, W)

        if self.image_set != 'test':
            seg_label = cv2.imread(self.seg_label_list[idx], cv2.IMREAD_GRAYSCALE)
            exist = np.array(self.exist_list[idx], dtype=np.float32)
        else:
            seg_label = None
            exist = None

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=seg_label)
            img = transformed['image']
            if seg_label is not None:
                seg_label = transformed['mask'].long()

        return {
            'img': img,
            'seg_label': seg_label,
            'exist': exist,
            'img_name': self.img_list[idx],
            'original_size': original_size,
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
        original_size = [b['original_size'] for b in batch]

        if batch[0]['seg_label'] is None:
            return {
                'img': img,
                'seg_label': None,
                'exist': None,
                'img_name': img_name,
                'original_size': original_size,
            }

        seg_label = torch.stack([b['seg_label'] for b in batch])
        exist = torch.from_numpy(np.stack([b['exist'] for b in batch]))

        return {
            'img': img,
            'seg_label': seg_label,
            'exist': exist,
            'img_name': img_name,
            'original_size': original_size,
        }
