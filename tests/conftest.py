from __future__ import annotations

"""Shared pytest fixtures for all SuperEvent tests.

Includes:
- Dataset fixtures for TsDataset interface tests
- Inference fixtures (EventStream, file paths, pretrained model) shared across
  test_super_event_model, test_inference_aedat4, and test_event_sources

The fixtures in this module build a tiny on-disk sequence that mirrors the
folder structure expected by `data.dataset.TsDataset`:
- `time_surfaces/*.npz`
- `sg_matches/*.npz`
- `frames/*.png`

Using synthetic files keeps tests deterministic and fast while still exercising
real file I/O boundaries.
"""

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest

from data_preparation.util.data_io import save_ts_sparse
from eventDataGenLibPy import EventStream, Save_events
from inference.super_event_model import SuperEventModel


def _Build_time_surface(image_index: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Create a minimal multi-channel time-surface tensor for one image id.

    The tensor shape follows `(H, W, C)` with two channels populated
    (`4` and `9`) so downstream representation conversions have signal.
    """
    ts = np.zeros(shape, dtype=np.float32)
    ts[1 + image_index, 2 + image_index, 4] = 0.5
    ts[1 + image_index, 2 + image_index, 9] = 1.0
    return ts


def _Write_minimal_sequence(data_root: Path,
                            dataset_name: str = "fpv",
                            split: str = "test",
                            sequence_name: str = "seq0") -> Path:
    """Write a minimal sequence tree compatible with dataset loading.

    Returns
    -------
    Path
        Absolute path to the created sequence directory.
    """
    sequence_dir = data_root / split / dataset_name / sequence_name
    time_surfaces_dir = sequence_dir / "time_surfaces"
    matches_dir = sequence_dir / "sg_matches"
    frames_dir = sequence_dir / "frames"
    time_surfaces_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    ts_shape = (8, 10, 10)
    for image_index in (0, 1):
        ts = _Build_time_surface(image_index=image_index, shape=ts_shape)
        save_ts_sparse(str(time_surfaces_dir / f"{image_index:08d}"), ts)

        frame = np.full((ts_shape[0], ts_shape[1], 3), 32 + 64 * image_index, dtype=np.uint8)
        wrote_image = cv2.imwrite(str(frames_dir / f"{image_index:08d}.png"), frame)
        assert wrote_image, "Failed to write synthetic frame for tests."

    matches = {
        "keypoints0": np.array([[2.0, 1.0], [5.0, 3.0]], dtype=np.float32),
        "keypoints1": np.array([[2.0, 1.0], [5.0, 3.0]], dtype=np.float32),
        "matches0": np.array([0, 1], dtype=np.int64),
        # Keep numpy string scalars to guard path-building against dtype regressions.
        "image_id0": np.array("00000000"),
        "image_id1": np.array("00000001"),
    }
    np.savez_compressed(matches_dir / "00000000_00000001.npz", **matches)
    return sequence_dir


@pytest.fixture
def minimal_data_root(tmp_path: Path) -> Path:
    """Provide a temporary data root containing one synthetic sequence."""
    _Write_minimal_sequence(tmp_path)
    return tmp_path


@pytest.fixture
def dataset_config_factory(minimal_data_root: Path) -> Callable[[str], dict[str, Any]]:
    """Return a factory for minimal dataset configs by input representation.

    Parameters
    ----------
    input_representation : str
        One of the representations supported by `TsDataset` (e.g. `mcts`,
        `ts`, `mcts_1`, `tencode`).
    """
    def _Create_config(input_representation: str) -> dict[str, Any]:
        return {
            "train_data_path": str(minimal_data_root),
            "dataset_names": "fpv",
            "test_sequences": [{"fpv": ["seq0"]}],
            "temporal_matching": {"enable": True},
            "homography_adaptation": {
                "enable": False,
                "difficulty": 0.0,
                "max_angle": 0.0,
                "resize_factor": 0.0,
            },
            "shuffle_data": False,
            "grid_size": 1,
            "input_representation": input_representation,
            "batch_size": 1,
        }

    return _Create_config


# ---------------------------------------------------------------------------
# Shared inference fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_event_stream() -> EventStream:
    """Deterministic synthetic EventStream used across inference tests."""
    n_events = 5000
    rng = np.random.default_rng(42)
    return EventStream(
        t_s=np.sort(rng.uniform(0.0, 0.1, n_events)).astype(np.float64),
        x=rng.integers(0, 240, n_events).astype(np.int32),
        y=rng.integers(0, 180, n_events).astype(np.int32),
        p01=rng.integers(0, 2, n_events).astype(np.uint8),
        width=240,
        height=180,
    )


@pytest.fixture
def synthetic_aedat4_path(
    tmp_path: Path,
    synthetic_event_stream: EventStream,
) -> Path:
    """Write synthetic events to a temporary AEDAT4 file."""
    path = tmp_path / "synthetic.aedat4"
    Save_events(synthetic_event_stream, path, Format="aedat4", Width=240, Height=180)
    return path


@pytest.fixture
def synthetic_h5_path(
    tmp_path: Path,
    synthetic_event_stream: EventStream,
) -> Path:
    """Write synthetic events to a temporary HDF5 file."""
    path = tmp_path / "synthetic.h5"
    Save_events(synthetic_event_stream, path, Format="h5")
    return path


@pytest.fixture
def synthetic_txt_path(
    tmp_path: Path,
    synthetic_event_stream: EventStream,
) -> Path:
    """Write synthetic events to a temporary TXT file."""
    path = tmp_path / "synthetic.txt"
    Save_events(synthetic_event_stream, path, Format="txt")
    return path


@pytest.fixture(scope="session")
def super_event_model() -> SuperEventModel:
    """Pretrained SuperEventModel (loaded once per test session)."""
    return SuperEventModel(
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        resolution=[180, 240],
        detection_threshold=0.001,
        top_k=50,
    )
