import cv2
import numpy as np


# Lane colors: 4 lanes with distinct colors (BGR format for cv2)
LANE_COLORS = np.array([
    [255, 125, 0],    # Lane 1: Orange
    [0, 255, 0],      # Lane 2: Green
    [0, 0, 255],      # Lane 3: Red
    [0, 255, 255],    # Lane 4: Yellow
], dtype='uint8')


def visualize_lanes(
    img: np.ndarray,
    seg_pred: np.ndarray,
    exist_pred: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Visualize lane predictions on image.

    Args:
        img: Original image (H, W, 3) in RGB format
        seg_pred: Segmentation prediction (5, H, W) or (H, W) as argmax
        exist_pred: Existence prediction (4,) probabilities
        threshold: Threshold for existence prediction

    Returns:
        img_overlay: Image with lane overlay (H, W, 3) in RGB format
        lane_img: Lane mask image (H, W, 3) in RGB format
    """
    img = img.copy()
    lane_img = np.zeros_like(img)

    # Get lane mask from segmentation prediction
    if len(seg_pred.shape) == 3:
        coord_mask = np.argmax(seg_pred, axis=0)
    else:
        coord_mask = seg_pred

    # Draw each lane if it exists
    for i in range(4):
        if exist_pred[i] > threshold:
            lane_img[coord_mask == (i + 1)] = LANE_COLORS[i]

    # Create overlay
    img_overlay = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1.0, gamma=0.0)

    return img_overlay, lane_img


def add_exist_text(img: np.ndarray, exist_pred: np.ndarray) -> np.ndarray:
    """
    Add existence prediction probabilities to image.

    Args:
        img: Image to draw on (H, W, 3)
        exist_pred: Existence prediction (4,) probabilities

    Returns:
        Image with text overlay
    """
    img = img.copy()
    exist_probs = [f"{p:.2f}" for p in exist_pred]
    cv2.putText(img, f"{exist_probs}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def prepare_visualization_batch(
    imgs: list[str],
    seg_preds: np.ndarray,
    exist_preds: np.ndarray,
    resize_shape: tuple[int, int],
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """
    Prepare a batch of visualizations for TensorBoard.

    Args:
        imgs: List of image paths
        seg_preds: Segmentation predictions (B, 5, H, W)
        exist_preds: Existence predictions (B, 4)
        resize_shape: Target size as (width, height)
        threshold: Threshold for existence prediction

    Returns:
        List of visualization images for TensorBoard
    """
    result_imgs = []

    for i in range(len(seg_preds)):
        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)

        seg_pred = seg_preds[i]
        exist_pred = exist_preds[i]

        # Generate visualizations
        img_overlay, lane_img = visualize_lanes(img, seg_pred, exist_pred, threshold)
        lane_img = add_exist_text(lane_img, exist_pred)

        # Convert to RGB for TensorBoard
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB) if img_overlay.shape[2] == 3 else img_overlay
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB) if lane_img.shape[2] == 3 else lane_img

        result_imgs.append(img_overlay)
        result_imgs.append(lane_img)

    return result_imgs
