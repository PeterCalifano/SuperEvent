"""Backward-compatibility tests for inference.event_inference pipeline.

Verifies that the ``EventInference`` / ``Aedat4Inference`` API still works
after refactoring to delegate to ``SuperEventModel``. Inference correctness
(keypoint bounds, L2 norm, window counts) is covered by test_super_event_model.py;
this file focuses on the wrapper surface and deprecated aliases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eventDataGenLibPy import EventStream
from inference.event_inference import (
    EventInference,
    EventInferenceSettings,
    InferenceResult,
    _Compute_crop_mask,
    _Load_config,
)
from inference.super_event_model import SuperEventModel


# ---------------------------------------------------------------------------
# Fixture: EventInference pipeline (local — not shared with other test files)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inference_pipeline() -> EventInference:
    """Create a pipeline with the default pretrained model."""
    settings = EventInferenceSettings(
        resolution=[180, 240],
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        detection_threshold=0.001,
        top_k=50,
    )
    return EventInference(settings)


# ---------------------------------------------------------------------------
# Settings and aliases
# ---------------------------------------------------------------------------

def test_event_inference_settings_defaults() -> None:
    """Verify EventInferenceSettings dataclass defaults are sensible."""
    s = EventInferenceSettings()
    assert s.resolution == [180, 240]
    assert s.detection_threshold == 0.01
    assert s.nms_box_size == 5
    assert s.device == "cpu"
    assert s.top_k is None


def test_backward_compat_aliases_exist() -> None:
    """Aedat4Inference and Aedat4InferenceSettings importable from old path."""
    from inference.aedat4_inference import Aedat4Inference, Aedat4InferenceSettings

    assert Aedat4Inference is EventInference
    assert Aedat4InferenceSettings is EventInferenceSettings


def test_event_inference_delegates_to_super_event_model(
    inference_pipeline: EventInference,
) -> None:
    """EventInference wraps a SuperEventModel internally."""
    assert isinstance(inference_pipeline._model, SuperEventModel)


def test_event_inference_exposes_cropped_shape(
    inference_pipeline: EventInference,
) -> None:
    """EventInference exposes cropped_shape from the inner model."""
    shape = inference_pipeline.cropped_shape
    assert len(shape) == 2
    assert shape[0] <= 180
    assert shape[1] <= 240


# ---------------------------------------------------------------------------
# Config / crop mask re-exports
# ---------------------------------------------------------------------------

def test_load_config_loads_backbone() -> None:
    """_Load_config (re-exported from super_event_model) merges backbone YAML."""
    config = _Load_config("config/super_event.yaml")
    assert "backbone_config" in config


def test_compute_crop_mask_grid_only() -> None:
    """_Compute_crop_mask with grid_size=8 on [180, 240] yields [176, 240]."""
    mask, shape = _Compute_crop_mask([180, 240], {"grid_size": 8})
    assert shape == [176, 240]


# ---------------------------------------------------------------------------
# Multi-format loading (via wrapper API)
# ---------------------------------------------------------------------------

def test_load_events_from_aedat4(
    synthetic_aedat4_path: Path,
    inference_pipeline: EventInference,
) -> None:
    """Loading AEDAT4 via EventInference.Load_events_from_file."""
    stream = inference_pipeline.Load_events_from_file(synthetic_aedat4_path)
    assert isinstance(stream, EventStream)
    assert len(stream.t_s) > 0


def test_backward_compat_load_events_from_aedat4(
    synthetic_aedat4_path: Path,
    inference_pipeline: EventInference,
) -> None:
    """Deprecated Load_events_from_aedat4 method still works."""
    stream = inference_pipeline.Load_events_from_aedat4(synthetic_aedat4_path)
    assert isinstance(stream, EventStream)


# ---------------------------------------------------------------------------
# Wrapper-specific methods
# ---------------------------------------------------------------------------

def test_process_single_window_returns_inference_result(
    inference_pipeline: EventInference,
    synthetic_event_stream: EventStream,
) -> None:
    """Process_single_window (wrapper method) returns InferenceResult."""
    mat = synthetic_event_stream.To_matrix_t_x_y_p()
    batch = torch.from_numpy(mat).float()
    result = inference_pipeline.Process_single_window(batch, timestamp=0.05)

    assert isinstance(result, InferenceResult)
    assert result.keypoints.ndim == 2
    assert result.descriptors.ndim == 2


def test_process_event_stream_window_count(
    inference_pipeline: EventInference,
    synthetic_event_stream: EventStream,
) -> None:
    """Process_event_stream (wrapper method) returns correct number of windows."""
    time_window = 0.025
    results = inference_pipeline.Process_event_stream(
        synthetic_event_stream, time_window_s=time_window,
    )
    duration = synthetic_event_stream.t_s[-1] - synthetic_event_stream.t_s[0]
    expected = int(np.ceil(duration / time_window))
    assert len(results) == expected
