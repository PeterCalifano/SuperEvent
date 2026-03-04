"""Event source abstraction for file, TCP, and ROS2 inputs.

Provides a uniform callback-based streaming interface for different event
sources. Each source accumulates events over a time window and fires a
user-provided callback with an ``EventStream`` batch.

Example
-------
>>> from inference.event_sources import FileEventSource
>>> source = FileEventSource("events.aedat4")  # doctest: +SKIP
>>> source.Start(callback=lambda stream: print(len(stream.t_s)), time_window_s=0.033)  # doctest: +SKIP
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Callable

import numpy as np

from eventDataGenLibPy import EventStream, Load_events


# TODO move to eventDataGenLibPy?
class EventSource(ABC):
    """Abstract base class for event sources.

    Subclasses must implement ``Start`` and ``Stop``.

    Example
    -------
    >>> # EventSource is abstract — use FileEventSource, TcpEventSource, etc.
    """

    @abstractmethod
    def Start(
        self,
        callback: Callable[[EventStream], None],
        time_window_s: float = 0.033,
    ) -> None:
        """Begin streaming events. Calls ``callback`` per time window.

        Parameters
        ----------
        callback : Callable[[EventStream], None]
            Function called with an ``EventStream`` for each time window.
        time_window_s : float
            Duration of each window in seconds.
        """

    @abstractmethod
    def Stop(self) -> None:
        """Stop streaming and release resources."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Return sensor resolution as ``(height, width)``."""


class FileEventSource(EventSource):
    """Event source that reads from a file and replays in time windows.

    Supports all formats handled by ``eventDataGenLibPy.Load_events``:
    AEDAT4, AEDAT2, HDF5, TXT.

    Example
    -------
    >>> source = FileEventSource("events.aedat4")  # doctest: +SKIP
    >>> source.Start(lambda s: print(len(s.t_s)), time_window_s=0.05)  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str | Path,
        format: str | None = None,
        height: int = 180,
        width: int = 240,
    ) -> None:
        """Load events from file.

        Parameters
        ----------
        file_path : str or Path
            Path to event file.
        format : str or None
            Force format, or auto-detect from extension.
        height : int
            Sensor height (used if file doesn't provide resolution).
        width : int
            Sensor width (used if file doesn't provide resolution).
        """
        self._stream = Load_events(file_path, Format=format)
        self._height = self._stream.height if self._stream.height else height
        self._width = self._stream.width if self._stream.width else width

    def Start(self,
              callback: Callable[[EventStream], None],
              time_window_s: float = 0.033,
              ) -> None:
        """Replay events in time windows, calling callback for each."""
        t = self._stream.t_s
        t_start = t[0]
        t_end = t[-1]
        window_start = t_start

        while window_start < t_end:
            window_end = window_start + time_window_s
            mask = (t >= window_start) & (t < window_end)

            if np.any(mask):
                window_stream = EventStream(
                    t_s=self._stream.t_s[mask],
                    x=self._stream.x[mask],
                    y=self._stream.y[mask],
                    p01=self._stream.p01[mask],
                    width=self._width,
                    height=self._height,
                )
                callback(window_stream)

            window_start = window_end

    def Stop(self) -> None:
        """No-op for file source (replay is synchronous)."""

    @property
    def resolution(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return (self._height, self._width)


class TcpEventSource(EventSource):
    """Event source using DV SDK TCP streaming protocol.

    Connects to a ``dv-processing`` network server (e.g. DVXplorer camera or
    DV software) and receives events over TCP.

    Requires ``dv-processing`` (``pip install dv-processing``).

    Example
    -------
    >>> source = TcpEventSource("127.0.0.1", 4040, resolution=(180, 240))  # doctest: +SKIP
    >>> source.Start(lambda s: print(len(s.t_s)), time_window_s=0.033)  # doctest: +SKIP
    >>> source.Stop()  # doctest: +SKIP
    """

    def __init__(self,
                 address: str,
                 port: int,
                 resolution: tuple[int, int] = (180, 240),
                 ) -> None:
        """Store connection parameters (connection is deferred to Start).

        Parameters
        ----------
        address : str
            Server IP address.
        port : int
            Server TCP port.
        resolution : tuple[int, int]
            Sensor resolution as ``(height, width)``.
        """
        self._address = address
        self._port = port
        self._resolution = resolution
        self._running = False
        self._thread: threading.Thread | None = None

    def Start(self,
              callback: Callable[[EventStream], None],
              time_window_s: float = 0.033,
              ) -> None:
        """Connect and start receiving events in a background thread."""
        import dv_processing as dv

        self._running = True

        def _Listen() -> None:
            reader = dv.io.NetworkReader(self._address, self._port)

            t_buf: list[float] = []
            x_buf: list[int] = []
            y_buf: list[int] = []
            p_buf: list[int] = []
            window_start: float | None = None

            while self._running:
                events = reader.getNextEventBatch()
                if events is None:
                    time.sleep(0.001)
                    continue

                for ev in events:
                    t_s = ev.timestamp * 1e-6  # µs → s
                    if window_start is None:
                        window_start = t_s

                    t_buf.append(t_s)
                    x_buf.append(int(ev.x))
                    y_buf.append(int(ev.y))
                    p_buf.append(1 if ev.polarity else 0)

                    if t_s - window_start >= time_window_s:
                        stream = EventStream(
                            t_s=np.array(t_buf, dtype=np.float64),
                            x=np.array(x_buf, dtype=np.int32),
                            y=np.array(y_buf, dtype=np.int32),
                            p01=np.array(p_buf, dtype=np.uint8),
                            width=self._resolution[1],
                            height=self._resolution[0],
                        )
                        callback(stream)
                        t_buf.clear()
                        x_buf.clear()
                        y_buf.clear()
                        p_buf.clear()
                        window_start = None

        self._thread = threading.Thread(target=_Listen, daemon=True)
        self._thread.start()

    def Stop(self) -> None:
        """Signal the listener thread to stop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return self._resolution


class Ros2EventSource(EventSource):
    """Event source subscribing to ``dvs_msgs/msg/EventArray`` on ROS2.

    Creates a minimal ``rclpy`` node that subscribes to the given topic,
    accumulates events per time window, and fires the callback.

    Requires ``rclpy`` and ``dvs_msgs``.

    Example
    -------
    >>> source = Ros2EventSource("/dvs/events", resolution=(180, 240))  # doctest: +SKIP
    >>> source.Start(lambda s: print(len(s.t_s)), time_window_s=0.033)  # doctest: +SKIP
    >>> source.Stop()  # doctest: +SKIP
    """

    def __init__(self,
                 topic: str,
                 resolution: tuple[int, int] = (180, 240),
                 node_name: str = "super_event_listener",
                 ) -> None:
        """Store subscription parameters (node is created on Start).

        Parameters
        ----------
        topic : str
            ROS2 topic publishing ``dvs_msgs/msg/EventArray``.
        resolution : tuple[int, int]
            Sensor resolution as ``(height, width)``.
        node_name : str
            Name for the ROS2 node.
        """
        self._topic = topic
        self._resolution = resolution
        self._node_name = node_name
        self._running = False
        self._thread: threading.Thread | None = None
        self._node = None

    def Start(self,
              callback: Callable[[EventStream], None],
              time_window_s: float = 0.033,
              ) -> None:
        """Create ROS2 node, subscribe, and spin in a background thread."""
        import rclpy
        from dvs_msgs.msg import EventArray
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()

        self._running = True

        t_buf: list[float] = []
        x_buf: list[int] = []
        y_buf: list[int] = []
        p_buf: list[int] = []
        window_start_ref: list[float | None] = [
            None]  # mutable ref for closure

        def _On_event_array(msg: EventArray) -> None:
            for ev in msg.events:
                t_s = float(ev.ts.sec) + float(ev.ts.nanosec) * 1e-9
                if window_start_ref[0] is None:
                    window_start_ref[0] = t_s

                t_buf.append(t_s)
                x_buf.append(int(ev.x))
                y_buf.append(int(ev.y))
                p_buf.append(1 if ev.polarity else 0)

                if t_s - window_start_ref[0] >= time_window_s:
                    stream = EventStream(
                        t_s=np.array(t_buf, dtype=np.float64),
                        x=np.array(x_buf, dtype=np.int32),
                        y=np.array(y_buf, dtype=np.int32),
                        p01=np.array(p_buf, dtype=np.uint8),
                        width=self._resolution[1],
                        height=self._resolution[0],
                    )
                    callback(stream)
                    t_buf.clear()
                    x_buf.clear()
                    y_buf.clear()
                    p_buf.clear()
                    window_start_ref[0] = None

        node = rclpy.create_node(self._node_name)
        self._node = node
        node.create_subscription(EventArray, self._topic, _On_event_array, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def _Spin() -> None:
            while self._running and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)

        self._thread = threading.Thread(target=_Spin, daemon=True)
        self._thread.start()

    def Stop(self) -> None:
        """Destroy the ROS2 node and stop spinning."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return self._resolution
