import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    resize_shape: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    rotation: float = 2.0,
    horizontal_flip_prob: float = 0.5,
    color_jitter_prob: float = 0.5,
    motion_blur_prob: float = 0.2,
) -> A.Compose:
    """
    Build transforms for training.

    Args:
        resize_shape: Target size as (height, width)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
        rotation: Maximum rotation angle in degrees
        horizontal_flip_prob: Probability of horizontal flip
        color_jitter_prob: Probability of color jitter
        motion_blur_prob: Probability of motion blur

    Returns:
        Albumentations Compose transform
    """
    # Note: Albumentations.Normalize scales to [0,1] and normalizes in one step,
    # then ToTensorV2 converts to tensor. This is different from pytorch's Normalize
    # and ToTensor.
    return A.Compose([
        A.Resize(height=resize_shape[0], width=resize_shape[1]),
        A.Rotate(limit=rotation, border_mode=0, p=0.5),
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=color_jitter_prob),
        A.MotionBlur(blur_limit=7, p=motion_blur_prob),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    resize_shape: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> A.Compose:
    """
    Build transforms for validation and testing.

    Args:
        resize_shape: Target size as (height, width)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Albumentations Compose transform
    """
    # Note: Albumentations.Normalize scales to [0,1] and normalizes in one step,
    # then ToTensorV2 converts to tensor. This is different from pytorch's Normalize
    # and ToTensor.
    return A.Compose([
        A.Resize(height=resize_shape[0], width=resize_shape[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
