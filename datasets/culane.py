import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CULane(Dataset):
    """
    CULane dataset for lane detection.

    Args:
        root: Path to CULane dataset root directory
        image_set: One of 'train', 'val', or 'test'
        transforms: Optional transforms to apply to samples

    Dataset structure expected:
        root/
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
            └── test.txt
    """

    def __init__(self, root, image_set, transforms=None):
        super().__init__()
        assert image_set in ('train', 'val', 'test'), "image_set must be 'train', 'val', or 'test'"

        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []

        if image_set != 'test':
            self._load_annotations()
        else:
            self._load_test_list()

    def _load_annotations(self):
        """Load annotations for train/val sets."""
        list_file = os.path.join(self.root, "list", f"{self.image_set}_gt.txt")

        with open(list_file) as f:
            for line in f:
                line = line.strip()
                parts = line.split(" ")

                # Remove leading '/' for os.path.join to work correctly
                img_path = os.path.join(self.root, parts[0][1:])
                seg_path = os.path.join(self.root, parts[1][1:])
                exist = [int(x) for x in parts[2:]]

                self.img_list.append(img_path)
                self.segLabel_list.append(seg_path)
                self.exist_list.append(exist)

    def _load_test_list(self):
        """Load image list for test set."""
        list_file = os.path.join(self.root, "list", "test.txt")

        with open(list_file) as f:
            for line in f:
                line = line.strip()
                # Remove leading '/' for os.path.join to work correctly
                img_path = os.path.join(self.root, line[1:])
                self.img_list.append(img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]
            exist = np.array(self.exist_list[idx])
        else:
            segLabel = None
            exist = None

        sample = {
            'img': img,
            'segLabel': segLabel,
            'exist': exist,
            'img_name': self.img_list[idx],
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Dictionary with batched tensors
        """
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        return {
            'img': img,
            'segLabel': segLabel,
            'exist': exist,
            'img_name': [b['img_name'] for b in batch],
        }
