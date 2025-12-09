import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import Normalize as TorchNormalize


class CustomTransform:
    """Base class for all custom transforms."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, name) -> bool:
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name) -> bool:
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """Compose multiple transforms together."""

    def __init__(self, *transforms) -> None:
        self.transforms = [*transforms]

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    """
    Resize image and segmentation label.

    Args:
        size: Target size as (width, height) or single int for square
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (W, H)

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img')
        seg_label = sample.get('seg_label', None)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        if seg_label is not None:
            seg_label = cv2.resize(seg_label, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['seg_label'] = seg_label
        return _sample

    def reset_size(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class RandomResize(Resize):
    """
    Resize to random size within specified range.

    Args:
        minW: Minimum width
        maxW: Maximum width
        minH: Minimum height (defaults to minW if not specified)
        maxH: Maximum height (defaults to maxW if not specified)
    """

    def __init__(
        self,
        minW: int,
        maxW: int,
        minH: int = None,
        maxH: int = None,
    ) -> None:
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super().__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH

    def random_set_size(self) -> None:
        w = np.random.randint(self.minW, self.maxW + 1)
        h = np.random.randint(self.minH, self.maxH + 1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    """
    Random rotation augmentation.

    Args:
        theta: Maximum rotation angle in degrees (rotation will be in range [-theta/2, theta/2])
    """

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img')
        seg_label = sample.get('seg_label', None)

        u = np.random.uniform()
        degree = (u - 0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if seg_label is not None:
            seg_label = cv2.warpAffine(seg_label, R, (seg_label.shape[1], seg_label.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['seg_label'] = seg_label
        return _sample

    def reset_theta(self, theta: float) -> None:
        self.theta = theta


class Normalize(CustomTransform):
    """
    Normalize image using mean and std.

    Args:
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.transform = TorchNormalize(mean, std)

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img')

        img = self.transform(img)

        _sample = sample.copy()
        _sample['img'] = img
        return _sample


class ToTensor(CustomTransform):
    """
    Convert image and labels to PyTorch tensors.

    Args:
        dtype: Data type for image tensor (default: torch.float)
    """

    def __init__(self, dtype: torch.dtype = torch.float) -> None:
        self.dtype = dtype

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img')
        seg_label = sample.get('seg_label', None)
        exist = sample.get('exist', None)

        # Convert image: (H, W, C) -> (C, H, W), scale to [0, 1]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.

        if seg_label is not None:
            seg_label = torch.from_numpy(seg_label).type(torch.long)
        if exist is not None:
            exist = torch.from_numpy(exist).type(torch.float32)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['seg_label'] = seg_label
        _sample['exist'] = exist
        return _sample


class RandomFlip(CustomTransform):
    """
    Random horizontal and/or vertical flip.

    Args:
        prob_x: Probability of horizontal flip
        prob_y: Probability of vertical flip
    """

    def __init__(self, prob_x: float = 0, prob_y: float = 0) -> None:
        self.prob_x = prob_x
        self.prob_y = prob_y

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img').copy()
        seg_label = sample.get('seg_label', None)
        if seg_label is not None:
            seg_label = seg_label.copy()

        flip_x = np.random.choice([False, True], p=(1 - self.prob_x, self.prob_x))
        flip_y = np.random.choice([False, True], p=(1 - self.prob_y, self.prob_y))

        if flip_x:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            if seg_label is not None:
                seg_label = np.ascontiguousarray(np.flip(seg_label, axis=1))

        if flip_y:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            if seg_label is not None:
                seg_label = np.ascontiguousarray(np.flip(seg_label, axis=0))

        _sample = sample.copy()
        _sample['img'] = img
        _sample['seg_label'] = seg_label
        return _sample


class Darkness(CustomTransform):
    """
    Random brightness reduction.

    Args:
        coeff: Maximum darkness coefficient (must be >= 1.0)
    """

    def __init__(self, coeff: float) -> None:
        assert coeff >= 1.0, "Darkness coefficient must be >= 1.0"
        self.coeff = coeff

    def __call__(self, sample: dict) -> dict:
        img = sample.get('img')

        coeff = np.random.uniform(1.0, self.coeff)
        img = (img.astype('float32') / coeff).astype('uint8')

        _sample = sample.copy()
        _sample['img'] = img
        return _sample
