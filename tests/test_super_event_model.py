"""Interface tests for inference.super_event_model.SuperEventModel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from eventDataGenLibPy import EventStream
from inference.super_event_model import (
    Compute_crop_mask,
    Compute_crop_offset,
    InferenceResult,
    Load_config,
    SuperEventModel,
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_super_event_model_init(super_event_model: SuperEventModel) -> None:
    """Model initializes with config, model, and correct shapes."""
    assert super_event_model.config is not None
    assert super_event_model.model is not None
    assert super_event_model.resolution == [180, 240]
    assert len(super_event_model.cropped_shape) == 2
    assert super_event_model.device == "cpu"


def test_super_event_model_config_has_backbone(super_event_model: SuperEventModel) -> None:
    """Merged config contains backbone_config from YAML."""
    assert "backbone_config" in super_event_model.config
    assert super_event_model.config["backbone"] == "maxvit"


def test_load_config_returns_dict() -> None:
    """Load_config returns a dict with expected keys."""
    cfg = Load_config("config/super_event.yaml")
    assert isinstance(cfg, dict)
    assert "backbone" in cfg
    assert "input_channels" in cfg


def test_compute_crop_mask_shape() -> None:
    """Compute_crop_mask returns mask and cropped shape with correct types."""
    mask, shape = Compute_crop_mask([180, 240], {"grid_size": 8})
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert len(shape) == 2
    assert shape[0] <= 180
    assert shape[1] <= 240


def test_compute_crop_mask_maxvit() -> None:
    """Crop mask with full MaxViT config on [180, 240] matches expected shape."""
    cfg = Load_config("config/super_event.yaml")
    _, shape = Compute_crop_mask([180, 240], cfg)
    # Shape must be divisible by the MaxViT factor
    factor = (
        2 ** (len(cfg["backbone_config"]["num_blocks"]) - 1)
        * cfg["backbone_config"]["stem"]["patch_size"]
        * int(np.max(cfg["backbone_config"]["stage"]["attention"]["partition_size"]))
    )
    assert shape[0] % factor == 0
    assert shape[1] % factor == 0


# ---------------------------------------------------------------------------
# Event I/O — multi-format
# ---------------------------------------------------------------------------

def test_load_events_from_aedat4(
    super_event_model: SuperEventModel,
    synthetic_aedat4_path: Path,
) -> None:
    """Loading AEDAT4 returns valid EventStream."""
    stream = super_event_model.load_events_from_file(synthetic_aedat4_path)
    assert isinstance(stream, EventStream)
    assert len(stream.t_s) > 0
    assert stream.t_s.dtype == np.float64


def test_load_events_from_h5(
    super_event_model: SuperEventModel,
    synthetic_h5_path: Path,
) -> None:
    """Loading HDF5 returns valid EventStream."""
    stream = super_event_model.load_events_from_file(synthetic_h5_path)
    assert isinstance(stream, EventStream)
    assert len(stream.t_s) > 0


def test_load_events_from_txt(
    super_event_model: SuperEventModel,
    synthetic_txt_path: Path,
) -> None:
    """Loading TXT returns valid EventStream."""
    stream = super_event_model.load_events_from_file(synthetic_txt_path)
    assert isinstance(stream, EventStream)
    assert len(stream.t_s) > 0


def test_load_events_explicit_format(
    super_event_model: SuperEventModel,
    synthetic_txt_path: Path,
) -> None:
    """Explicit format parameter works."""
    stream = super_event_model.load_events_from_file(synthetic_txt_path, format="txt")
    assert isinstance(stream, EventStream)


# ---------------------------------------------------------------------------
# Inference from events
# ---------------------------------------------------------------------------

def test_infer_from_events_returns_inference_result(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Infer_from_events returns a well-formed InferenceResult."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = super_event_model.infer_from_events(batch, timestamp=0.05)

    assert isinstance(result, InferenceResult)
    assert result.timestamp == 0.05
    assert result.keypoints.ndim == 2
    assert result.keypoints.shape[1] == 2  # [row, col]
    assert result.probabilities.ndim == 1
    assert len(result.probabilities) == len(result.keypoints)
    assert result.descriptors.ndim == 2
    assert result.time_surface.ndim == 3


def test_infer_from_events_empty_batch(super_event_model: SuperEventModel) -> None:
    """Empty event batch still returns a valid InferenceResult."""
    batch = torch.zeros((0, 4))
    result = super_event_model.infer_from_events(batch, timestamp=0.0)
    assert isinstance(result, InferenceResult)
    assert result.keypoints.shape[1] == 2
    assert result.time_surface.ndim == 3


def test_infer_from_events_keypoints_in_bounds(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Detected keypoints are within the cropped image bounds."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = super_event_model.infer_from_events(batch, timestamp=0.05)

    if len(result.keypoints) > 0:
        h, w = super_event_model.cropped_shape
        assert np.all(result.keypoints[:, 0] >= 0)
        assert np.all(result.keypoints[:, 0] < h)
        assert np.all(result.keypoints[:, 1] >= 0)
        assert np.all(result.keypoints[:, 1] < w)


def test_infer_from_events_top_k_respected(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Number of keypoints does not exceed top_k."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = super_event_model.infer_from_events(batch, timestamp=0.05)
    # super_event_model fixture sets top_k=50
    assert len(result.keypoints) <= 50


def test_infer_from_events_descriptors_l2_normalized(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Descriptors have approximately unit L2 norm."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = super_event_model.infer_from_events(batch, timestamp=0.05)

    if len(result.descriptors) > 0:
        norms = np.linalg.norm(result.descriptors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Inference from time-surface
# ---------------------------------------------------------------------------

def test_infer_from_time_surface_numpy(super_event_model: SuperEventModel) -> None:
    """Infer_from_time_surface works with numpy array input."""
    ts = np.random.rand(180, 240, 10).astype(np.float32)
    result = super_event_model.infer_from_time_surface(ts, timestamp=1.0)

    assert isinstance(result, InferenceResult)
    assert result.timestamp == 1.0
    assert result.keypoints.ndim == 2
    assert result.time_surface.shape == (180, 240, 10)


def test_infer_from_time_surface_torch(super_event_model: SuperEventModel) -> None:
    """Infer_from_time_surface works with torch tensor input."""
    ts = torch.rand(180, 240, 10)
    result = super_event_model.infer_from_time_surface(ts, timestamp=2.0)

    assert isinstance(result, InferenceResult)
    assert result.timestamp == 2.0


def test_infer_from_time_surface_descriptors_shape(super_event_model: SuperEventModel) -> None:
    """Descriptors from TS inference have correct second dimension."""
    ts = np.random.rand(180, 240, 10).astype(np.float32) * 0.5
    result = super_event_model.infer_from_time_surface(ts)

    if len(result.descriptors) > 0:
        assert result.descriptors.shape[1] == super_event_model.config["descriptor_size"]


# ---------------------------------------------------------------------------
# Stream inference
# ---------------------------------------------------------------------------

def test_infer_from_event_stream_window_count(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Infer_from_event_stream returns one result per time window."""
    time_window = 0.025
    results = super_event_model.infer_from_event_stream(
        synthetic_event_stream, time_window_s=time_window,
    )
    duration = synthetic_event_stream.t_s[-1] - synthetic_event_stream.t_s[0]
    expected = int(np.ceil(duration / time_window))
    assert len(results) == expected
    assert all(isinstance(r, InferenceResult) for r in results)


def test_infer_from_event_stream_timestamps_increasing(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """Result timestamps are strictly increasing."""
    results = super_event_model.infer_from_event_stream(
        synthetic_event_stream, time_window_s=0.025,
    )
    timestamps = [r.timestamp for r in results]
    assert all(t1 < t2 for t1, t2 in zip(timestamps, timestamps[1:]))


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_reset_time_surface() -> None:
    """After reset, time-surface generator internal timestamps are zero."""
    # Use a fresh model so we don't disturb the session-scoped fixture
    model = SuperEventModel(
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        resolution=[180, 240],
    )
    events = torch.tensor([
        [0.0, 100.0, 90.0, 1.0],
        [0.001, 120.0, 80.0, 0.0],
    ])
    model.infer_from_events(events, timestamp=0.001)

    ts_stamps_before = model._ts_gen.time_stamps.clone()
    assert torch.any(ts_stamps_before != 0), "Timestamps should be non-zero after events"

    model.Reset_time_surface()
    ts_stamps_after = model._ts_gen.time_stamps
    assert torch.all(ts_stamps_after == 0), "Timestamps should be zero after reset"


def test_reset_preserves_camera_matrix() -> None:
    """Reset_time_surface keeps camera_matrix intact."""
    K = np.array([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]], dtype=np.float64)
    D = np.zeros(5)
    model = SuperEventModel(
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        resolution=[180, 240],
        camera_matrix=K,
        dist_coeffs=D,
    )
    model.Reset_time_surface()
    assert model.camera_matrix is not None
    np.testing.assert_array_equal(model.camera_matrix, K)


# ---------------------------------------------------------------------------
# Crop offset
# ---------------------------------------------------------------------------

def test_compute_crop_offset_grid_only() -> None:
    """With grid_size=8, [180, 240] has row_offset=2, col_offset=0."""
    offset = Compute_crop_offset([180, 240], {"grid_size": 8})
    assert offset == [2, 0]


def test_compute_crop_offset_symmetric() -> None:
    """Offset + cropped_shape + bottom/right remainder = full resolution."""
    cfg = {"grid_size": 8}
    _, shape = Compute_crop_mask([183, 245], cfg)
    offset = Compute_crop_offset([183, 245], cfg)
    # Top + kept + bottom == 183; left + kept + right == 245
    crop_h = 183 - shape[0]
    crop_w = 245 - shape[1]
    assert offset[0] == math.ceil(crop_h / 2)
    assert offset[1] == math.ceil(crop_w / 2)


# ---------------------------------------------------------------------------
# Map keypoints to sensor frame
# ---------------------------------------------------------------------------

def test_map_keypoints_to_sensor_frame_applies_offset(
    super_event_model: SuperEventModel,
) -> None:
    """Mapped keypoints are shifted by exactly the crop offset."""
    kp = np.array([[10., 15.], [20., 30.]], dtype=np.float32)
    mapped = super_event_model.map_keypoints_to_sensor_frame(kp)
    expected = kp + np.array(super_event_model.crop_offset, dtype=np.float32)
    np.testing.assert_array_equal(mapped, expected)


def test_map_keypoints_to_sensor_frame_empty() -> None:
    """Empty keypoint array is returned unchanged (zero rows)."""
    model = super_event_model_factory()
    kp = np.zeros((0, 2), dtype=np.float32)
    mapped = model.map_keypoints_to_sensor_frame(kp)
    assert mapped.shape == (0, 2)


def test_map_keypoints_to_sensor_frame_within_sensor_bounds(
    super_event_model: SuperEventModel,
    synthetic_event_stream: EventStream,
) -> None:
    """All sensor-frame keypoints are within sensor resolution."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = super_event_model.infer_from_events(batch, timestamp=0.05)
    kp_sensor = super_event_model.map_keypoints_to_sensor_frame(result.keypoints)

    if len(kp_sensor) > 0:
        h, w = super_event_model.resolution
        assert np.all(kp_sensor[:, 0] >= 0) and np.all(kp_sensor[:, 0] < h)
        assert np.all(kp_sensor[:, 1] >= 0) and np.all(kp_sensor[:, 1] < w)


# ---------------------------------------------------------------------------
# Unproject keypoints (requires camera_matrix)
# ---------------------------------------------------------------------------

def super_event_model_factory(**kwargs) -> SuperEventModel:
    """Create a fresh SuperEventModel with optional overrides (for parameter tests)."""
    defaults = dict(
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        resolution=[180, 240],
    )
    defaults.update(kwargs)
    return SuperEventModel(**defaults)


def test_unproject_keypoints_raises_without_camera_matrix(
    super_event_model: SuperEventModel,
) -> None:
    """Unproject_keypoints raises ValueError when no camera_matrix was given."""
    kp = np.array([[10., 20.]], dtype=np.float32)
    with pytest.raises(ValueError, match="camera_matrix"):
        super_event_model.unproject_keypoints(kp)


def test_unproject_keypoints_empty_returns_empty() -> None:
    """Empty keypoint array returns zero-row (N, 2) array."""
    K = np.array([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]], dtype=np.float64)
    model = super_event_model_factory(camera_matrix=K, dist_coeffs=np.zeros(5))
    result = model.unproject_keypoints(np.zeros((0, 2), dtype=np.float32))
    assert result.shape == (0, 2)


def test_unproject_keypoints_shape() -> None:
    """Output shape is (N, 2) for N keypoints."""
    K = np.array([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]], dtype=np.float64)
    model = super_event_model_factory(camera_matrix=K, dist_coeffs=np.zeros(5))
    kp = np.array([[90., 120.], [45., 60.], [10., 30.]], dtype=np.float32)
    out = model.unproject_keypoints(kp)
    assert out.shape == (3, 2)


def test_unproject_keypoints_principal_point_maps_to_origin() -> None:
    """Principal point (cx, cy) in sensor space maps to (0, 0) normalized."""
    cx, cy = 120., 90.
    K = np.array([[200., 0., cx], [0., 200., cy], [0., 0., 1.]], dtype=np.float64)
    model = super_event_model_factory(camera_matrix=K, dist_coeffs=np.zeros(5))
    # A keypoint at [row=cy, col=cx] → (x=cx, y=cy) in pixel space
    kp = np.array([[cy, cx]], dtype=np.float32)
    out = model.unproject_keypoints(kp)
    np.testing.assert_allclose(out[0], [0., 0.], atol=1e-5)


def test_unproject_keypoints_focal_length_scaling() -> None:
    """A pixel offset of 1 focal-length unit maps to ±1.0 in normalized coords."""
    fx, fy = 200., 200.
    cx, cy = 120., 90.
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)
    model = super_event_model_factory(camera_matrix=K, dist_coeffs=np.zeros(5))
    # col = cx + fx → x_norm = 1.0; row = cy
    kp = np.array([[cy, cx + fx]], dtype=np.float32)
    out = model.unproject_keypoints(kp)
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-5)


def test_model_accepts_camera_matrix_without_dist_coeffs() -> None:
    """camera_matrix alone (no dist_coeffs) is valid and disables TS undistortion."""
    K = np.array([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]], dtype=np.float64)
    model = super_event_model_factory(camera_matrix=K)
    assert model.camera_matrix is not None
    assert not model._undistort_ts  # no dist_coeffs → no TS undistortion
    # Unprojection still works (treats pixels as already undistorted)
    kp = np.array([[90., 120.]], dtype=np.float32)
    out = model.unproject_keypoints(kp)
    assert out.shape == (1, 2)
