import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    resize_shape: tuple[int, int],
    rotation: float,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> A.Compose:
    """
    Build transforms for training.

    Args:
        resize_shape: Target size as (width, height)
        rotation: Maximum rotation angle in degrees (rotation will be in range [-rotation/2, rotation/2])
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Albumentations Compose transform
    """
    # Note: Albumentations.Normalize scales to [0,1] and normalizes in one step,
    # then ToTensorV2 converts to tensor. This is different from pytorch's Normalize
    # and ToTensor.
    return A.Compose([
        A.Resize(height=resize_shape[1], width=resize_shape[0]),
        A.Rotate(limit=rotation / 2, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    resize_shape: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> A.Compose:
    """
    Build transforms for validation and testing.

    Args:
        resize_shape: Target size as (width, height)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Albumentations Compose transform
    """
    # Note: Albumentations.Normalize scales to [0,1] and normalizes in one step,
    # then ToTensorV2 converts to tensor. This is different from pytorch's Normalize
    # and ToTensor.
    return A.Compose([
        A.Resize(height=resize_shape[1], width=resize_shape[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
