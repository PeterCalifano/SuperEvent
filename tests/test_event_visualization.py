"""Interface tests for inference.event_visualization."""

from __future__ import annotations

import numpy as np
import pytest

from inference.event_visualization import (
    Create_inference_summary,
    Render_descriptors_pca,
    Render_events_to_frame,
    Render_keypoints_on_image,
    Render_time_surface,
)


# ---------------------------------------------------------------------------
# Render_keypoints_on_image
# ---------------------------------------------------------------------------

def test_render_keypoints_on_image_draws_circles() -> None:
    """Output differs from input at pixel locations where keypoints exist."""
    bg = np.zeros((50, 50, 3), dtype=np.uint8)
    kp = np.array([[25, 25]], dtype=np.float32)
    probs = np.array([0.9], dtype=np.float32)
    out = Render_keypoints_on_image(bg, kp, probs)
    # The circle should have changed some pixels near the keypoint
    assert not np.array_equal(out, bg)


def test_render_keypoints_on_image_empty_keypoints() -> None:
    """No keypoints returns an identical copy of the input."""
    bg = np.full((30, 30, 3), 128, dtype=np.uint8)
    kp = np.zeros((0, 2), dtype=np.float32)
    probs = np.zeros(0, dtype=np.float32)
    out = Render_keypoints_on_image(bg, kp, probs)
    assert np.array_equal(out, bg)


# ---------------------------------------------------------------------------
# Create_inference_summary
# ---------------------------------------------------------------------------

def test_create_inference_summary_shape() -> None:
    """2x2 mosaic has dimensions (2*H, 2*W, 3)."""
    h, w = 60, 80
    imgs = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(4)]
    mosaic = Create_inference_summary(*imgs)
    assert mosaic.shape == (2 * h, 2 * w, 3)


def test_create_inference_summary_different_sizes() -> None:
    """Mosaic resizes input images to the largest dimensions."""
    small = np.zeros((30, 40, 3), dtype=np.uint8)
    large = np.zeros((60, 80, 3), dtype=np.uint8)
    mosaic = Create_inference_summary(small, large, small, large)
    assert mosaic.shape == (120, 160, 3)


# ---------------------------------------------------------------------------
# Render_events_to_frame
# ---------------------------------------------------------------------------

def test_render_events_to_frame_shape_and_dtype() -> None:
    """Output is a uint8 BGR image of the requested size."""
    evs = np.array([[0.0, 5, 3, 1], [0.001, 8, 6, 0]], dtype=np.float64)
    img = Render_events_to_frame(evs, height=20, width=20)
    assert img.shape == (20, 20, 3)
    assert img.dtype == np.uint8


def test_render_events_to_frame_on_events_are_red() -> None:
    """ON events (p=1) produce a red pixel (BGR [0, 0, 255])."""
    evs = np.array([[0.0, 5, 3, 1]], dtype=np.float64)  # col=5, row=3, ON
    img = Render_events_to_frame(evs, height=20, width=20)
    np.testing.assert_array_equal(img[3, 5], [0, 0, 255])


def test_render_events_to_frame_off_events_are_blue() -> None:
    """OFF events (p=0) produce a blue pixel (BGR [255, 0, 0])."""
    evs = np.array([[0.0, 8, 6, 0]], dtype=np.float64)  # col=8, row=6, OFF
    img = Render_events_to_frame(evs, height=20, width=20)
    np.testing.assert_array_equal(img[6, 8], [255, 0, 0])


def test_render_events_to_frame_empty_events() -> None:
    """Empty event array returns a white image."""
    evs = np.zeros((0, 4), dtype=np.float64)
    img = Render_events_to_frame(evs, height=10, width=10)
    assert img.shape == (10, 10, 3)
    assert np.all(img == 255)


# ---------------------------------------------------------------------------
# Render_time_surface
# ---------------------------------------------------------------------------

def test_render_time_surface_handles_mcts_channels() -> None:
    """10-channel MCTS produces a BGR uint8 image of the right shape."""
    ts = np.random.rand(15, 20, 10).astype(np.float32)
    img = Render_time_surface(ts)
    assert img.shape == (15, 20, 3)
    assert img.dtype == np.uint8


def test_render_time_surface_generic_channels() -> None:
    """Non-10-channel TS produces max-pooled grayscale BGR image."""
    ts = np.random.rand(15, 20, 4).astype(np.float32)
    img = Render_time_surface(ts)
    assert img.shape == (15, 20, 3)
    assert img.dtype == np.uint8


def test_render_time_surface_single_channel() -> None:
    """Single-channel TS is handled by the generic path."""
    ts = np.random.rand(10, 12, 1).astype(np.float32)
    img = Render_time_surface(ts)
    assert img.shape == (10, 12, 3)
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Render_descriptors_pca
# ---------------------------------------------------------------------------

def test_render_descriptors_pca_shape_and_dtype() -> None:
    """Output is a BGR uint8 image matching image_shape."""
    desc = np.random.randn(20, 256).astype(np.float32)
    kp = np.column_stack([
        np.random.randint(0, 50, 20), np.random.randint(0, 60, 20),
    ]).astype(np.float32)
    img = Render_descriptors_pca(desc, kp, image_shape=(50, 60))
    assert img.shape == (50, 60, 3)
    assert img.dtype == np.uint8


def test_render_descriptors_pca_fewer_than_3_descriptors_returns_black() -> None:
    """With fewer than 3 descriptors, returns a black image (PCA undefined)."""
    desc = np.random.randn(2, 256).astype(np.float32)
    kp = np.array([[10, 10], [20, 20]], dtype=np.float32)
    img = Render_descriptors_pca(desc, kp, image_shape=(40, 40))
    assert img.shape == (40, 40, 3)
    assert np.all(img == 0)


def test_render_descriptors_pca_non_zero_output() -> None:
    """With enough descriptors, at least some pixels are non-zero."""
    rng = np.random.default_rng(0)
    desc = rng.standard_normal((10, 256)).astype(np.float32)
    kp = np.column_stack([
        rng.integers(5, 45, 10), rng.integers(5, 55, 10),
    ]).astype(np.float32)
    img = Render_descriptors_pca(desc, kp, image_shape=(50, 60))
    assert np.any(img > 0)
