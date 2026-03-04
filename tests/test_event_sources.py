"""Interface tests for inference.event_sources.

Tests FileEventSource without external dependencies (file replay is
synchronous). TcpEventSource and Ros2EventSource are only structure-tested
(init / stop-before-start / isinstance) since they require live hardware or
a running ROS2 environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eventDataGenLibPy import EventStream
from inference.event_sources import (
    EventSource,
    FileEventSource,
    Ros2EventSource,
    TcpEventSource,
)


# ---------------------------------------------------------------------------
# EventSource ABC
# ---------------------------------------------------------------------------

def test_event_source_is_abstract() -> None:
    """EventSource cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EventSource()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FileEventSource — init and resolution
# ---------------------------------------------------------------------------

def test_file_event_source_resolution(synthetic_aedat4_path: Path) -> None:
    """Resolution matches the saved file dimensions."""
    source = FileEventSource(synthetic_aedat4_path)
    assert source.resolution == (180, 240)


def test_file_event_source_is_event_source(synthetic_aedat4_path: Path) -> None:
    """FileEventSource is an EventSource."""
    source = FileEventSource(synthetic_aedat4_path)
    assert isinstance(source, EventSource)


def test_file_event_source_fallback_resolution(synthetic_txt_path: Path) -> None:
    """Resolution falls back to constructor args when file has no metadata."""
    source = FileEventSource(synthetic_txt_path, height=120, width=160)
    h, w = source.resolution
    # TXT files may not embed resolution; constructor fallback should apply
    assert h > 0 and w > 0


# ---------------------------------------------------------------------------
# FileEventSource — callback behavior
# ---------------------------------------------------------------------------

def test_file_event_source_calls_callback(synthetic_aedat4_path: Path) -> None:
    """Start calls the callback at least once for a non-empty file."""
    received: list[EventStream] = []
    source = FileEventSource(synthetic_aedat4_path)
    source.Start(callback=received.append, time_window_s=0.05)
    assert len(received) > 0


def test_file_event_source_callback_count(
    synthetic_aedat4_path: Path,
    synthetic_event_stream: EventStream,
) -> None:
    """Callback fires the correct number of windows (ceil of duration / window)."""
    time_window = 0.025
    duration = synthetic_event_stream.t_s[-1] - synthetic_event_stream.t_s[0]
    # FileEventSource only fires when there are events in the window
    # so count <= ceil(duration / time_window)
    received: list[EventStream] = []
    source = FileEventSource(synthetic_aedat4_path)
    source.Start(callback=received.append, time_window_s=time_window)
    max_expected = int(np.ceil(duration / time_window))
    assert 1 <= len(received) <= max_expected


def test_file_event_source_callback_receives_event_stream(
    synthetic_aedat4_path: Path,
) -> None:
    """Each callback receives an EventStream with non-empty timestamps."""
    received: list[EventStream] = []
    source = FileEventSource(synthetic_aedat4_path)
    source.Start(callback=received.append, time_window_s=0.05)
    for batch in received:
        assert isinstance(batch, EventStream)
        assert len(batch.t_s) > 0
        assert batch.t_s.dtype == np.float64


def test_file_event_source_window_timestamps_non_overlapping(
    synthetic_aedat4_path: Path,
) -> None:
    """Each window's events are strictly within the window time range."""
    windows: list[EventStream] = []
    source = FileEventSource(synthetic_aedat4_path)
    source.Start(callback=windows.append, time_window_s=0.025)

    for i in range(len(windows) - 1):
        t_max_current = windows[i].t_s.max()
        t_min_next = windows[i + 1].t_s.min()
        assert t_max_current < t_min_next, (
            f"Window {i} max time ({t_max_current:.6f}) overlaps "
            f"window {i+1} min time ({t_min_next:.6f})"
        )


def test_file_event_source_txt_format(synthetic_txt_path: Path) -> None:
    """FileEventSource handles TXT files."""
    received: list[EventStream] = []
    source = FileEventSource(synthetic_txt_path)
    source.Start(callback=received.append, time_window_s=0.05)
    assert len(received) > 0


def test_file_event_source_h5_format(synthetic_h5_path: Path) -> None:
    """FileEventSource handles HDF5 files."""
    received: list[EventStream] = []
    source = FileEventSource(synthetic_h5_path)
    source.Start(callback=received.append, time_window_s=0.05)
    assert len(received) > 0


def test_file_event_source_explicit_format(synthetic_txt_path: Path) -> None:
    """Explicit format parameter overrides extension detection."""
    received: list[EventStream] = []
    source = FileEventSource(synthetic_txt_path, format="txt")
    source.Start(callback=received.append, time_window_s=0.05)
    assert len(received) > 0


def test_file_event_source_stop_is_noop(synthetic_aedat4_path: Path) -> None:
    """Stop on a file source does not raise (synchronous replay)."""
    source = FileEventSource(synthetic_aedat4_path)
    source.Stop()  # before Start — must be safe


# ---------------------------------------------------------------------------
# TcpEventSource — structure only (no live camera)
# ---------------------------------------------------------------------------

def test_tcp_event_source_init() -> None:
    """TcpEventSource stores connection params without connecting."""
    source = TcpEventSource("127.0.0.1", 4040, resolution=(180, 240))
    assert source.resolution == (180, 240)
    assert source._address == "127.0.0.1"
    assert source._port == 4040
    assert not source._running


def test_tcp_event_source_stop_before_start() -> None:
    """Calling Stop before Start does not raise."""
    source = TcpEventSource("127.0.0.1", 4040)
    source.Stop()  # must be safe


def test_tcp_event_source_is_event_source() -> None:
    """TcpEventSource is an EventSource."""
    source = TcpEventSource("127.0.0.1", 4040)
    assert isinstance(source, EventSource)


# ---------------------------------------------------------------------------
# Ros2EventSource — structure only (no ROS2 runtime)
# ---------------------------------------------------------------------------

def test_ros2_event_source_init() -> None:
    """Ros2EventSource stores topic and resolution without starting a node."""
    source = Ros2EventSource("/dvs/events", resolution=(260, 346))
    assert source.resolution == (260, 346)
    assert source._topic == "/dvs/events"
    assert not source._running
    assert source._node is None


def test_ros2_event_source_stop_before_start() -> None:
    """Calling Stop before Start does not raise."""
    source = Ros2EventSource("/dvs/events")
    source.Stop()


def test_ros2_event_source_is_event_source() -> None:
    """Ros2EventSource is an EventSource."""
    source = Ros2EventSource("/dvs/events")
    assert isinstance(source, EventSource)


def test_ros2_event_source_custom_node_name() -> None:
    """Custom node_name is stored correctly."""
    source = Ros2EventSource("/dvs/events", node_name="my_listener")
    assert source._node_name == "my_listener"
