"""Event and inference output visualization utilities.

Render events as red/blue frames, visualize time-surfaces, keypoints, and
descriptor PCA maps, and compose summary mosaics.

Example
-------
>>> import numpy as np
>>> frame = Render_events_to_frame(
...     np.array([[0.0, 10, 20, 1]], dtype=np.float64), height=40, width=40,
... )
>>> print(frame.shape, frame.dtype)
(40, 40, 3) uint8
"""

from __future__ import annotations

import cv2
import numpy as np


def Render_events_to_frame(
    events_t_x_y_p: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Accumulate events into a BGR image: ON=red, OFF=blue, background=white.

    Parameters
    ----------
    events_t_x_y_p : np.ndarray
        ``[N, 4]`` array with columns ``[t, x_col, y_row, p01]``.
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.

    Returns
    -------
    np.ndarray
        BGR ``uint8`` image of shape ``(height, width, 3)``.

    Example
    -------
    >>> import numpy as np
    >>> evs = np.array([[0.0, 5, 3, 1], [0.001, 8, 6, 0]], dtype=np.float64)
    >>> img = Render_events_to_frame(evs, height=10, width=10)
    >>> print(img[3, 5, 2] > 0)  # red channel for ON event at (row=3, col=5)
    True
    """
    frame = np.full((height, width, 3), 255, dtype=np.uint8)

    if len(events_t_x_y_p) == 0:
        return frame

    x_col = events_t_x_y_p[:, 1].astype(np.int32)
    y_row = events_t_x_y_p[:, 2].astype(np.int32)
    p = events_t_x_y_p[:, 3].astype(np.int32)

    # Clip to image bounds
    valid = (y_row >= 0) & (y_row < height) & (x_col >= 0) & (x_col < width)
    x_col = x_col[valid]
    y_row = y_row[valid]
    p = p[valid]

    on_mask = p == 1
    off_mask = p == 0

    # BGR: blue channel = index 0, red channel = index 2
    # ON events → red
    frame[y_row[on_mask], x_col[on_mask]] = [0, 0, 255]
    # OFF events → blue
    frame[y_row[off_mask], x_col[off_mask]] = [255, 0, 0]

    return frame


def Render_time_surface(
    ts_tensor: np.ndarray,
) -> np.ndarray:
    """Convert a time-surface tensor (H, W, C) to a BGR uint8 visualization.

    For 10-channel MCTS, reuses the channel-selection from ``ts2image``.
    For other channel counts, takes the max across channels.

    Parameters
    ----------
    ts_tensor : np.ndarray
        Time-surface of shape ``(H, W, C)`` with values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        BGR ``uint8`` image of shape ``(H, W, 3)``.

    Example
    -------
    >>> import numpy as np
    >>> ts = np.random.rand(10, 10, 10).astype(np.float32)
    >>> img = Render_time_surface(ts)
    >>> print(img.shape, img.dtype)
    (10, 10, 3) uint8
    """
    h, w, c = ts_tensor.shape

    if c == 10:
        # Reuse the ts2image convention: channels 3 (OFF) and 8 (ON)
        ts_out = np.ones((h, w, 3), dtype=np.float64)
        blue_mask = ts_tensor[..., 3] > 0
        red_mask = ts_tensor[..., 8] > 0
        blue_vals = ts_tensor[blue_mask, 3]
        red_vals = ts_tensor[red_mask, 8]
        ts_out[blue_mask] = np.column_stack([
            np.ones_like(blue_vals), 1.0 - blue_vals, 1.0 - blue_vals,
        ])
        ts_out[red_mask] = np.column_stack([
            1.0 - red_vals, 1.0 - red_vals, np.ones_like(red_vals),
        ])
        return np.rint(ts_out * 255.0).astype(np.uint8)

    # Generic: max across channels → grayscale → BGR
    ts_max = np.max(ts_tensor, axis=-1)
    gray = np.rint(ts_max * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def Render_keypoints_on_image(
    image: np.ndarray,
    keypoints: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Draw keypoint circles colored by detection probability.

    Parameters
    ----------
    image : np.ndarray
        BGR ``uint8`` background image of shape ``(H, W, 3)``.
    keypoints : np.ndarray
        ``[N, 2]`` array of ``(row, col)`` positions.
    probabilities : np.ndarray
        ``[N]`` array of detection probabilities in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Copy of the image with keypoint circles drawn.

    Example
    -------
    >>> import numpy as np
    >>> img = np.zeros((50, 50, 3), dtype=np.uint8)
    >>> kp = np.array([[10, 20]], dtype=np.float32)
    >>> p = np.array([0.9], dtype=np.float32)
    >>> out = Render_keypoints_on_image(img, kp, p)
    >>> print(out.shape)
    (50, 50, 3)
    """
    out = image.copy()
    if len(keypoints) == 0:
        return out

    p_min = probabilities.min()
    p_max = probabilities.max()
    p_range = p_max - p_min if p_max > p_min else 1.0

    for i in range(len(keypoints)):
        row, col = int(keypoints[i, 0]), int(keypoints[i, 1])
        norm_p = (probabilities[i] - p_min) / p_range
        # Green → Red gradient: low probability = green, high = red
        color = (0, int(255 * (1 - norm_p)), int(255 * norm_p))
        cv2.circle(out, (col, row), radius=3, color=color, thickness=1)

    return out


def Render_descriptors_pca(
    descriptors: np.ndarray,
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """PCA-based descriptor colorization: first 3 components mapped to RGB.

    Parameters
    ----------
    descriptors : np.ndarray
        ``[N, D]`` L2-normalized descriptor vectors.
    keypoints : np.ndarray
        ``[N, 2]`` array of ``(row, col)`` positions.
    image_shape : tuple[int, int]
        ``(height, width)`` of the output image.

    Returns
    -------
    np.ndarray
        BGR ``uint8`` image of shape ``(height, width, 3)``.

    Example
    -------
    >>> import numpy as np
    >>> desc = np.random.randn(10, 256).astype(np.float32)
    >>> kp = np.column_stack([np.arange(10), np.arange(10)]).astype(np.float32)
    >>> img = Render_descriptors_pca(desc, kp, (50, 50))
    >>> print(img.shape, img.dtype)
    (50, 50, 3) uint8
    """
    h, w = image_shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if len(descriptors) < 3:
        return out

    # Simple PCA via SVD on centered descriptors
    mean = descriptors.mean(axis=0)
    centered = descriptors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T  # [N, 3]

    # Normalize to [0, 255]
    for dim in range(3):
        col = projected[:, dim]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            projected[:, dim] = (col - col_min) / (col_max - col_min) * 255.0
        else:
            projected[:, dim] = 128.0

    for i in range(len(keypoints)):
        row, col = int(keypoints[i, 0]), int(keypoints[i, 1])
        if 0 <= row < h and 0 <= col < w:
            out[row, col] = projected[i].astype(np.uint8)
            # Draw a small circle for visibility
            color = tuple(int(c) for c in projected[i].astype(int))
            cv2.circle(out, (col, row), radius=3, color=color, thickness=-1)

    return out


def Create_inference_summary(
    event_frame: np.ndarray,
    ts_image: np.ndarray,
    detection_image: np.ndarray,
    descriptor_image: np.ndarray,
) -> np.ndarray:
    """Compose a 2x2 grid mosaic from four visualization images.

    All images are resized to the same dimensions before tiling.

    Parameters
    ----------
    event_frame : np.ndarray
        Event accumulation image (BGR uint8).
    ts_image : np.ndarray
        Time-surface visualization (BGR uint8).
    detection_image : np.ndarray
        Keypoint overlay image (BGR uint8).
    descriptor_image : np.ndarray
        Descriptor PCA image (BGR uint8).

    Returns
    -------
    np.ndarray
        Mosaic of shape ``(2*H, 2*W, 3)``.

    Example
    -------
    >>> import numpy as np
    >>> imgs = [np.zeros((100, 120, 3), dtype=np.uint8) for _ in range(4)]
    >>> mosaic = Create_inference_summary(*imgs)
    >>> print(mosaic.shape)
    (200, 240, 3)
    """
    target_h = max(img.shape[0] for img in [event_frame, ts_image, detection_image, descriptor_image])
    target_w = max(img.shape[1] for img in [event_frame, ts_image, detection_image, descriptor_image])

    def _Resize(img: np.ndarray) -> np.ndarray:
        if img.shape[0] != target_h or img.shape[1] != target_w:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return img

    top = np.hstack([_Resize(event_frame), _Resize(ts_image)])
    bottom = np.hstack([_Resize(detection_image), _Resize(descriptor_image)])
    return np.vstack([top, bottom])
