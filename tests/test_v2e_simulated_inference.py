"""End-to-end test: v2e synthetic events → AEDAT4 → SuperEvent inference.

Uses ``v2ecore.emulator.EventEmulator`` to generate events from a synthetic
moving-edge scene, writes them to AEDAT4, and runs the full pipeline.

Marked ``pytest.mark.slow`` because v2e emulation takes ~1 s.
"""

from __future__ import annotations
from v2ecore.emulator import EventEmulator

from pathlib import Path

import numpy as np
import pytest
import torch

from eventDataGenLibPy import EventStream, Save_events
from inference.event_inference import EventInference, EventInferenceSettings

# Try importing v2e; skip entire module if unavailable
v2ecore = pytest.importorskip("v2ecore")


HEIGHT = 180
WIDTH = 240

# Auxiliary helpers for synthetic data


def _Generate_moving_edge_frames(n_frames: int = 30,
                                 ) -> list[np.ndarray]:
    """Create grayscale frames with a vertical edge moving left to right.

    Example
    -------
    >>> frames = _Generate_moving_edge_frames(5)
    >>> print(len(frames), frames[0].shape)
    5 (180, 240)
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        frame = np.full((HEIGHT, WIDTH), 50, dtype=np.uint8)
        edge_col = int(WIDTH * 0.2 + (WIDTH * 0.6) * i / max(n_frames - 1, 1))
        frame[:, edge_col:] = 200
        frames.append(frame)
    return frames


def _Emulate_events(frames: list[np.ndarray],
                    dt_s: float = 0.001,
                    ) -> EventStream:
    """Run v2e EventEmulator on a sequence of frames and return an EventStream.

    Example
    -------
    >>> frames = _Generate_moving_edge_frames(10)
    >>> stream = _Emulate_events(frames)  # doctest: +SKIP
    """
    emulator = EventEmulator(pos_thres=0.2,
                             neg_thres=0.2,
                             sigma_thres=0.03,
                             output_folder=None,
                             dvs_h5=None,
                             dvs_aedat2=None,
                             dvs_text=None,
                             )

    all_events: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        t_s = i * dt_s
        frame_log = np.log(frame.astype(np.float32) + 1.0)
        events = emulator.generate_events(frame_log, t_s)
        if events is not None and len(events) > 0:
            all_events.append(events)

    if not all_events:
        # Return empty stream if emulator produced nothing
        return EventStream(
            t_s=np.array([], dtype=np.float64),
            x=np.array([], dtype=np.int32),
            y=np.array([], dtype=np.int32),
            p01=np.array([], dtype=np.uint8),
            width=WIDTH,
            height=HEIGHT,
        )

    combined = np.concatenate(all_events, axis=0)
    # v2e format: [t_s, x, y, polarity] where polarity is -1 or +1
    t = combined[:, 0].astype(np.float64)
    x = combined[:, 1].astype(np.int32)
    y = combined[:, 2].astype(np.int32)
    p_signed = combined[:, 3]
    p01 = np.where(p_signed > 0, 1, 0).astype(np.uint8)

    return EventStream(t_s=t, x=x, y=y, p01=p01,
                       width=WIDTH, height=HEIGHT,
                       )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def v2e_aedat4_path(tmp_path: Path) -> Path:
    """Generate synthetic events with v2e and write to AEDAT4."""
    frames = _Generate_moving_edge_frames(n_frames=30)
    stream = _Emulate_events(frames, dt_s=0.001)
    pytest.assume_events = len(stream.t_s)
    if len(stream.t_s) == 0:
        pytest.skip("v2e emulator produced no events for this scene")

    aedat4_file = tmp_path / "v2e_synthetic.aedat4"
    Save_events(stream, aedat4_file, Format="aedat4",
                Width=WIDTH, Height=HEIGHT)
    return aedat4_file


@pytest.fixture
def v2e_pipeline() -> EventInference:
    """Pipeline configured for v2e synthetic resolution."""
    settings = EventInferenceSettings(
        resolution=[HEIGHT, WIDTH],
        config_path="config/super_event.yaml",
        model_path="saved_models/super_event_weights.pth",
        device="cpu",
        detection_threshold=0.001,
        top_k=100,
    )
    return EventInference(settings)


@pytest.mark.slow
def test_v2e_events_produced(v2e_aedat4_path: Path) -> None:
    """v2e emulator generates events that can be written to AEDAT4."""
    assert v2e_aedat4_path.exists()
    assert v2e_aedat4_path.stat().st_size > 0


@pytest.mark.slow
def test_v2e_full_pipeline_loads_and_infers(
    v2e_aedat4_path: Path,
    v2e_pipeline: EventInference,
) -> None:
    """Full pipeline: AEDAT4 → events → time-surfaces → keypoints."""
    stream = v2e_pipeline.Load_events_from_file(v2e_aedat4_path)
    assert len(stream.t_s) > 0

    results = v2e_pipeline.Process_event_stream(stream, time_window_s=0.005)
    assert len(results) > 0

    # At least some windows should have non-zero time-surfaces
    ts_sums = [np.sum(r.time_surface) for r in results]
    assert any(s > 0 for s in ts_sums), "All time-surfaces are zero"


@pytest.mark.slow
def test_v2e_keypoints_detected_on_edges(
    v2e_aedat4_path: Path,
    v2e_pipeline: EventInference,
) -> None:
    """Keypoints should be detected somewhere in the scene."""
    stream = v2e_pipeline.Load_events_from_file(v2e_aedat4_path)
    results = v2e_pipeline.Process_event_stream(stream, time_window_s=0.005)

    total_keypoints = sum(len(r.keypoints) for r in results)
    assert total_keypoints > 0, "No keypoints detected across all windows"


@pytest.mark.slow
def test_v2e_descriptors_correct_dimensionality(
    v2e_aedat4_path: Path,
    v2e_pipeline: EventInference,
) -> None:
    """Descriptors have the expected dimensionality (256)."""
    stream = v2e_pipeline.Load_events_from_file(v2e_aedat4_path)
    results = v2e_pipeline.Process_event_stream(stream, time_window_s=0.005)

    for r in results:
        if len(r.descriptors) > 0:
            assert r.descriptors.shape[1] == 256
            # Check L2 normalization
            norms = np.linalg.norm(r.descriptors, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=0.05)
            return

    pytest.skip("No keypoints detected to verify descriptor dimensionality")
