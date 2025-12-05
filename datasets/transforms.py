import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize as TorchNormalize


class CustomTransform:
    """Base class for all custom transforms."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """Compose multiple transforms together."""

    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
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

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (W, H)

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        if segLabel is not None:
            segLabel = cv2.resize(segLabel, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_size(self, size):
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

    def __init__(self, minW, maxW, minH=None, maxH=None):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super().__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW + 1)
        h = np.random.randint(self.minH, self.maxH + 1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    """
    Random rotation augmentation.

    Args:
        theta: Maximum rotation angle in degrees (rotation will be in range [-theta/2, theta/2])
    """

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        degree = (u - 0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if segLabel is not None:
            segLabel = cv2.warpAffine(segLabel, R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta


class Normalize(CustomTransform):
    """
    Normalize image using mean and std.

    Args:
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
    """

    def __init__(self, mean, std):
        self.transform = TorchNormalize(mean, std)

    def __call__(self, sample):
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

    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        exist = sample.get('exist', None)

        # Convert image: (H, W, C) -> (C, H, W), scale to [0, 1]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.

        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)
        if exist is not None:
            exist = torch.from_numpy(exist).type(torch.float32)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        _sample['exist'] = exist
        return _sample


class RandomFlip(CustomTransform):
    """
    Random horizontal and/or vertical flip.

    Args:
        prob_x: Probability of horizontal flip
        prob_y: Probability of vertical flip
    """

    def __init__(self, prob_x=0, prob_y=0):
        self.prob_x = prob_x
        self.prob_y = prob_y

    def __call__(self, sample):
        img = sample.get('img').copy()
        segLabel = sample.get('segLabel', None)
        if segLabel is not None:
            segLabel = segLabel.copy()

        flip_x = np.random.choice([False, True], p=(1 - self.prob_x, self.prob_x))
        flip_y = np.random.choice([False, True], p=(1 - self.prob_y, self.prob_y))

        if flip_x:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            if segLabel is not None:
                segLabel = np.ascontiguousarray(np.flip(segLabel, axis=1))

        if flip_y:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            if segLabel is not None:
                segLabel = np.ascontiguousarray(np.flip(segLabel, axis=0))

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample


class Darkness(CustomTransform):
    """
    Random brightness reduction.

    Args:
        coeff: Maximum darkness coefficient (must be >= 1.0)
    """

    def __init__(self, coeff):
        assert coeff >= 1.0, "Darkness coefficient must be >= 1.0"
        self.coeff = coeff

    def __call__(self, sample):
        img = sample.get('img')

        coeff = np.random.uniform(1.0, self.coeff)
        img = (img.astype('float32') / coeff).astype('uint8')

        _sample = sample.copy()
        _sample['img'] = img
        return _sample
