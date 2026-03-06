"""Event-file inference pipeline for SuperEvent.

This module wraps :class:`SuperEventModel` behind the ``EventInference``
interface and exposes convenience methods for loading event files and
processing streams.

Example
-------
>>> from inference.event_inference import EventInference, EventInferenceSettings
>>> settings = EventInferenceSettings(resolution=[180, 240])
>>> pipeline = EventInference(settings)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from eventDataGenLibPy import EventStream
from inference.super_event_model import (
    InferenceResult,
    SuperEventModel,
)


@dataclass
class EventInferenceSettings:
    """Configuration for the event-file inference pipeline.

    Example
    -------
    >>> settings = EventInferenceSettings()
    >>> print(settings.detection_threshold)
    0.01
    """

    resolution: list[int] = field(default_factory=lambda: [180, 240])
    delta_t: list[float] = field(default_factory=lambda: [
                                 0.001, 0.003, 0.01, 0.03, 0.1])
    config_path: str = "config/super_event.yaml"
    model_path: str = "saved_models/super_event_weights.pth"
    device: str = "cpu"
    detection_threshold: float = 0.01
    nms_box_size: int = 5
    top_k: int | None = None


class EventInference:
    """Event-file inference pipeline delegating to :class:`SuperEventModel`.

    Example
    -------
    >>> settings = EventInferenceSettings()
    >>> pipeline = EventInference(settings)  # doctest: +SKIP
    """

    def __init__(
        self,
        settings: EventInferenceSettings,
    ) -> None:
        """Create the underlying SuperEventModel from settings."""
        self.settings = settings
        self._model = SuperEventModel(
            config_path=settings.config_path,
            model_path=settings.model_path,
            device=settings.device,
            resolution=settings.resolution,
            delta_t=settings.delta_t,
            detection_threshold=settings.detection_threshold,
            nms_box_size=settings.nms_box_size,
            top_k=settings.top_k,
        )

    @property
    def config(self) -> dict:
        """Return merged config."""
        return self._model.config

    @property
    def cropped_shape(self) -> list[int]:
        """Return the cropped shape used for model input."""
        return self._model.cropped_shape

    def Load_events_from_file(self, path: str | Path,
                              format: str | None = None,
                              ) -> EventStream:
        """Load events from any supported file format.

        Example
        -------
        >>> pipeline = EventInference(EventInferenceSettings())  # doctest: +SKIP
        >>> stream = pipeline.Load_events_from_file("events.aedat4")  # doctest: +SKIP
        """
        return self._model.load_events_from_file(path, format=format)

    def Process_single_window(self,
                              event_batch: torch.Tensor,
                              timestamp: float,
                              ) -> InferenceResult:
        """Run one inference step.

        Example
        -------
        >>> # See tests for runnable examples.
        """
        return self._model.infer_from_events(event_batch, timestamp)

    def Process_event_stream(self,
                             event_stream: EventStream,
                             time_window_s: float = 0.033,
                             ) -> list[InferenceResult]:
        """Split stream into time windows and run inference on each.

        Example
        -------
        >>> # See tests for runnable examples.
        """
        return self._model.infer_from_event_stream(event_stream, time_window_s)
