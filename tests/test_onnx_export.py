"""ONNX export tests for SuperEvent.

These tests require ``onnx``, ``onnxruntime``, and ``pyTorchAutoForge``.
Marked ``@pytest.mark.slow`` as export takes several seconds.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from inference.export_onnx import Export_model_to_onnx
from inference.super_event_model import Compute_crop_mask, Load_config
from models.super_event import SuperEvent, SuperEventFullRes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """Temporary directory for ONNX exports."""
    return tmp_path / "onnx_exports"


@pytest.fixture
def exported_onnx_path(export_dir: Path) -> str:
    """Export the model and return the .onnx file path."""
    return Export_model_to_onnx(
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        output_path=str(export_dir),
        resolution=[180, 240],
        opset_version=17,
        validate=False,
        simplify=False,
    )


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_export_onnx_produces_file(exported_onnx_path: str) -> None:
    """ONNX export creates a file on disk."""
    assert os.path.isfile(exported_onnx_path)
    assert exported_onnx_path.endswith(".onnx")
    assert os.path.getsize(exported_onnx_path) > 0


@pytest.mark.slow
def test_export_onnx_validates(exported_onnx_path: str) -> None:
    """Exported ONNX model passes onnx.checker.check_model."""
    model = onnx.load(exported_onnx_path)
    onnx.checker.check_model(model, full_check=True)


@pytest.mark.slow
def test_export_onnx_has_correct_io_names(exported_onnx_path: str) -> None:
    """ONNX model has expected input/output names."""
    model = onnx.load(exported_onnx_path)
    input_names = [inp.name for inp in model.graph.input]
    output_names = [out.name for out in model.graph.output]

    assert "time_surface" in input_names
    assert "prob" in output_names
    assert "descriptors" in output_names


@pytest.mark.slow
def test_export_onnx_inference_runs(exported_onnx_path: str) -> None:
    """ONNX model runs inference via onnxruntime."""
    config = Load_config("config/super_event.yaml")
    _, cropped_shape = Compute_crop_mask([180, 240], config)

    session = ort.InferenceSession(
        exported_onnx_path, providers=["CPUExecutionProvider"],
    )
    dummy = np.random.randn(
        1, config["input_channels"], cropped_shape[0], cropped_shape[1],
    ).astype(np.float32)

    outputs = session.run(None, {"time_surface": dummy})
    assert len(outputs) == 2  # prob, descriptors

    prob = outputs[0]
    descriptors = outputs[1]
    assert prob.ndim >= 2
    assert descriptors.ndim >= 3


@pytest.mark.slow
def test_export_onnx_inference_equivalence(exported_onnx_path: str) -> None:
    """PyTorch and ONNX Runtime produce approximately equal outputs."""
    config = Load_config("config/super_event.yaml")
    _, cropped_shape = Compute_crop_mask([180, 240], config)

    # PyTorch model with tracing
    if config["pixel_wise_predictions"]:
        torch_model = SuperEventFullRes(config, tracing=True)
    else:
        torch_model = SuperEvent(config, tracing=True)
    torch_model.load_state_dict(
        torch.load("saved_models/super_event_weights.pth", weights_only=True, map_location="cpu"),
    )
    torch_model.eval()

    dummy_np = np.random.randn(
        1, config["input_channels"], cropped_shape[0], cropped_shape[1],
    ).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)

    # PyTorch inference
    with torch.inference_mode():
        torch_prob, torch_desc = torch_model(dummy_torch)
    torch_prob_np = torch_prob.numpy()
    torch_desc_np = torch_desc.numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(
        exported_onnx_path, providers=["CPUExecutionProvider"],
    )
    ort_outputs = session.run(None, {"time_surface": dummy_np})
    ort_prob = ort_outputs[0]
    ort_desc = ort_outputs[1]

    np.testing.assert_allclose(torch_prob_np, ort_prob, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(torch_desc_np, ort_desc, rtol=1e-3, atol=1e-5)


@pytest.mark.slow
def test_export_onnx_with_validation(export_dir: Path) -> None:
    """Export with validate=True runs without error."""
    path = Export_model_to_onnx(
        output_path=str(export_dir / "validated"),
        validate=True,
        simplify=False,
    )
    assert os.path.isfile(path)
