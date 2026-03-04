"""Unified SuperEvent model wrapper for inference.

Provides a single API for loading events, running inference from events or
pre-computed time-surfaces, and managing time-surface generator state.

Example
-------
>>> from inference.super_event_model import SuperEventModel
>>> model = SuperEventModel()  # doctest: +SKIP
>>> result = model.Infer_from_time_surface(ts_tensor, timestamp=0.0)  # doctest: +SKIP
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from eventDataGenLibPy import EventStream, Load_events
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms
from ts_generation.generate_ts import TsGenerator


@dataclass
class InferenceResult:
    """Output of a single inference window.

    Example
    -------
    >>> import numpy as np
    >>> result = InferenceResult(
    ...     timestamp=0.033,
    ...     keypoints=np.zeros((5, 2), dtype=np.float32),
    ...     probabilities=np.ones(5, dtype=np.float32),
    ...     descriptors=np.zeros((5, 256), dtype=np.float32),
    ...     time_surface=np.zeros((180, 240, 10), dtype=np.float32),
    ... )
    >>> print(result.keypoints.shape)
    (5, 2)
    """

    timestamp: float
    keypoints: np.ndarray       # [N, 2] row, col
    probabilities: np.ndarray   # [N]
    descriptors: np.ndarray     # [N, descriptor_size]
    time_surface: np.ndarray    # [H, W, C]


def Load_config(
    config_path: str,
) -> dict[str, Any]:
    """Load and merge model + backbone YAML config files.

    Example
    -------
    >>> cfg = Load_config("config/super_event.yaml")  # doctest: +SKIP
    >>> print("backbone" in cfg)
    True
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "backbone" in config:
        backbone_config_path = os.path.join(
            os.path.dirname(config_path),
            "backbones",
            config["backbone"] + ".yaml",
        )
        if os.path.exists(backbone_config_path):
            with open(backbone_config_path, "r") as f:
                backbone_config = yaml.safe_load(f)
            config = config | backbone_config
            config["backbone_config"]["input_channels"] = config["input_channels"]

    return config


def Compute_crop_mask(
    ts_shape: list[int],
    config: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    """Compute a boolean crop mask and the resulting cropped shape.

    The mask removes border pixels so the spatial dimensions are divisible by
    the factor required by the backbone.

    Example
    -------
    >>> mask, shape = Compute_crop_mask([180, 240], {"grid_size": 8})
    >>> print(shape)
    [176, 240]
    """
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = (
            2 ** (len(config["backbone_config"]["num_blocks"]) - 1)
            * config["backbone_config"]["stem"]["patch_size"]
            * np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
        )

    crop = np.array(ts_shape) % max_factor_required
    crop_mask = torch.ones(ts_shape, dtype=bool)
    crop_mask[: math.ceil(crop[0] / 2)] = False
    crop_mask[:, : math.ceil(crop[1] / 2)] = False
    if crop[0] > 1:
        crop_mask[-math.floor(crop[0] / 2) :] = False
    if crop[1] > 1:
        crop_mask[:, -math.floor(crop[1] / 2) :] = False

    cropped_shape = [ts_shape[0] - int(crop[0]), ts_shape[1] - int(crop[1])]
    return crop_mask, cropped_shape


def Compute_crop_offset(
    ts_shape: list[int],
    config: dict[str, Any],
) -> list[int]:
    """Return the ``[row, col]`` offset of the top-left corner of the kept region.

    The crop removes border pixels symmetrically so that spatial dimensions are
    divisible by the backbone stride. The offset is the number of rows/columns
    removed from the top and left edges respectively.

    Adding this offset to a keypoint in cropped space gives the corresponding
    pixel in full-sensor space.

    Parameters
    ----------
    ts_shape : list[int]
        Full sensor resolution ``[H, W]``.
    config : dict[str, Any]
        Merged model config (output of ``Load_config``).

    Returns
    -------
    list[int]
        ``[row_offset, col_offset]``.

    Example
    -------
    >>> offset = Compute_crop_offset([180, 240], {"grid_size": 8})
    >>> print(offset)
    [2, 0]
    """
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = (
            2 ** (len(config["backbone_config"]["num_blocks"]) - 1)
            * config["backbone_config"]["stem"]["patch_size"]
            * int(np.max(config["backbone_config"]["stage"]["attention"]["partition_size"]))
        )

    crop = np.array(ts_shape) % max_factor_required
    return [int(math.ceil(crop[0] / 2)), int(math.ceil(crop[1] / 2))]


class SuperEventModel:
    """Unified wrapper for SuperEvent model operations.

    Handles config loading, model instantiation, time-surface generation,
    and keypoint/descriptor extraction. Visualization and export are handled
    by separate modules.

    Example
    -------
    >>> model = SuperEventModel(
    ...     config_path="config/super_event.yaml",
    ...     model_path="saved_models/super_event_weights.pth",
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        config_path: str = "config/super_event.yaml",
        model_path: str = "saved_models/super_event_weights.pth",
        device: str = "cpu",
        resolution: list[int] | None = None,
        delta_t: list[float] | None = None,
        detection_threshold: float = 0.01,
        nms_box_size: int = 5,
        top_k: int | None = None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        """Load config, model weights, and set up the time-surface generator.

        Parameters
        ----------
        config_path : str
            Path to the model config YAML.
        model_path : str
            Path to the ``.pth`` model weights.
        device : str
            PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
        resolution : list[int] or None
            Sensor resolution ``[H, W]``. Defaults to ``[180, 240]``.
        delta_t : list[float] or None
            Time constants for MCTS. Defaults to ``[0.001, 0.003, 0.01, 0.03, 0.1]``.
        detection_threshold : float
            Minimum keypoint probability to retain.
        nms_box_size : int
            NMS suppression radius in pixels.
        top_k : int or None
            Maximum number of keypoints per frame. ``None`` = no limit.
        camera_matrix : np.ndarray or None
            3×3 intrinsic matrix ``K``. Required for ``Unproject_keypoints``.
            When provided together with ``dist_coeffs``, the time-surface is
            undistorted before inference so that keypoints are in undistorted
            pixel space.
        dist_coeffs : np.ndarray or None
            Distortion coefficients compatible with ``cv2.undistort``.
            Ignored if ``camera_matrix`` is ``None``.
        """
        if resolution is None:
            resolution = [180, 240]
        if delta_t is None:
            delta_t = [0.001, 0.003, 0.01, 0.03, 0.1]

        self._device = device
        self._resolution = list(resolution)
        self._delta_t = list(delta_t)
        self._top_k = top_k

        # Camera intrinsics (optional)
        self._camera_matrix: np.ndarray | None = camera_matrix
        self._dist_coeffs: np.ndarray | None = dist_coeffs if camera_matrix is not None else None
        self._undistort_ts = camera_matrix is not None and dist_coeffs is not None

        # Config
        self._config = Load_config(config_path)
        self._config["detection_threshold"] = detection_threshold
        self._config["nms_box_size"] = nms_box_size

        # Crop mask and offset
        self._crop_mask, self._cropped_shape = Compute_crop_mask(
            self._resolution, self._config,
        )
        self._crop_offset: list[int] = Compute_crop_offset(self._resolution, self._config)
        self._crop_mask = self._crop_mask.to(device)

        # Model
        if self._config["pixel_wise_predictions"]:
            self._model = SuperEventFullRes(self._config)
        else:
            self._model = SuperEvent(self._config)

        self._model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device),
        )
        self._model.to(device)
        self._model.eval()

        # Time-surface generator
        self._ts_gen = self._Make_ts_generator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _Make_ts_generator(self) -> TsGenerator:
        """Construct a TsGenerator with current camera and resolution settings."""
        K = self._camera_matrix if self._camera_matrix is not None else np.identity(3)
        D = self._dist_coeffs if self._dist_coeffs is not None else np.zeros(5)
        return TsGenerator(
            camera_matrix=K,
            distortion_coeffs=D,
            settings={
                "shape": self._resolution,
                "delta_t": self._delta_t,
                "undistort": self._undistort_ts,
            },
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """Return the merged model configuration."""
        return self._config

    @property
    def model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model."""
        return self._model

    @property
    def resolution(self) -> list[int]:
        """Return the sensor resolution ``[H, W]``."""
        return self._resolution

    @property
    def cropped_shape(self) -> list[int]:
        """Return the cropped shape used for model input."""
        return self._cropped_shape

    @property
    def crop_offset(self) -> list[int]:
        """Return ``[row_offset, col_offset]`` removed from the top-left during cropping.

        Adding this offset to a keypoint ``[row, col]`` in cropped space gives
        the corresponding pixel in full-sensor space.

        Example
        -------
        >>> model = SuperEventModel()  # doctest: +SKIP
        >>> kp_sensor = result.keypoints + model.crop_offset  # doctest: +SKIP
        """
        return self._crop_offset

    @property
    def camera_matrix(self) -> np.ndarray | None:
        """Return the 3x3 intrinsic matrix, or ``None`` if not provided."""
        return self._camera_matrix

    @property
    def device(self) -> str:
        """Return the device string."""
        return self._device

    # ------------------------------------------------------------------
    # Event I/O
    # ------------------------------------------------------------------

    def Load_events_from_file(
        self,
        path: str | Path,
        format: str | None = None,
    ) -> EventStream:
        """Load events from any supported file format.

        Parameters
        ----------
        path : str or Path
            Event file (``.aedat4``, ``.aedat``, ``.h5``, ``.txt``, etc.).
        format : str or None
            Force format instead of auto-detecting from extension.

        Returns
        -------
        EventStream
            Canonical event container.

        Example
        -------
        >>> model = SuperEventModel()  # doctest: +SKIP
        >>> stream = model.Load_events_from_file("events.aedat4")  # doctest: +SKIP
        """
        return Load_events(path, Format=format)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _Run_model_and_extract(
        self,
        ts_input: torch.Tensor,
        ts_numpy: np.ndarray,
        timestamp: float,
    ) -> InferenceResult:
        """Run model forward pass, NMS, and keypoint/descriptor extraction.

        Parameters
        ----------
        ts_input : torch.Tensor
            Cropped, channels-first model input ``[1, C, H', W']``.
        ts_numpy : np.ndarray
            Original time-surface ``[H, W, C]`` for storing in result.
        timestamp : float
            Nominal timestamp.
        """
        with torch.inference_mode():
            pred = self._model(ts_input)

        kpts_batch, probs_batch = fast_nms(
            pred["prob"].clone(), self._config, top_k=self._top_k,
        )

        kpts = kpts_batch[0].cpu().numpy().astype(np.float32)
        probs = probs_batch[0].cpu().numpy()

        desc_map = pred["descriptors"][0]  # [C, H, W]
        if kpts.shape[0] > 0:
            descriptors = (
                desc_map[:, kpts[:, 0].astype(int), kpts[:, 1].astype(int)]
                .permute(1, 0)
                .cpu()
                .numpy()
            )
        else:
            descriptors = np.zeros(
                (0, self._config["descriptor_size"]), dtype=np.float32,
            )

        return InferenceResult(
            timestamp=timestamp,
            keypoints=kpts,
            probabilities=probs,
            descriptors=descriptors,
            time_surface=ts_numpy,
        )

    def _Prepare_ts_input(
        self,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """Convert channels-last TS to cropped channels-first model input."""
        ts_input = ts.permute(2, 0, 1).unsqueeze(0)
        ts_input = ts_input[..., self._crop_mask].reshape(
            list(ts_input.shape[:-2]) + self._cropped_shape,
        )
        return ts_input.to(self._device)

    def Infer_from_events(
        self,
        event_batch: torch.Tensor,
        timestamp: float,
    ) -> InferenceResult:
        """Run inference from a batch of events.

        Updates the internal time-surface generator, then runs the model.

        Parameters
        ----------
        event_batch : torch.Tensor
            ``[N, 4]`` tensor with columns ``[t_s, x_col, y_row, p01]``.
        timestamp : float
            Nominal timestamp for this inference step.

        Example
        -------
        >>> # See tests/test_super_event_model.py for runnable examples.
        """
        if len(event_batch) > 0:
            self._ts_gen.batch_update(event_batch)

        ts = self._ts_gen.get_ts()
        ts_numpy = ts.cpu().numpy()
        ts_input = self._Prepare_ts_input(ts)

        return self._Run_model_and_extract(ts_input, ts_numpy, timestamp)

    def Infer_from_time_surface(
        self,
        ts_tensor: torch.Tensor | np.ndarray,
        timestamp: float = 0.0,
    ) -> InferenceResult:
        """Run inference from a pre-computed time-surface.

        Bypasses the ``TsGenerator`` entirely — useful for dataset evaluation
        or pre-computed time-surfaces.

        Parameters
        ----------
        ts_tensor : torch.Tensor or np.ndarray
            Time-surface of shape ``[H, W, C]`` (channels last).
        timestamp : float
            Nominal timestamp.

        Example
        -------
        >>> # See tests/test_super_event_model.py for runnable examples.
        """
        if isinstance(ts_tensor, np.ndarray):
            ts_tensor = torch.from_numpy(ts_tensor).float()

        ts_numpy = ts_tensor.cpu().numpy()
        ts_input = self._Prepare_ts_input(ts_tensor)

        return self._Run_model_and_extract(ts_input, ts_numpy, timestamp)

    def Infer_from_event_stream(
        self,
        event_stream: EventStream,
        time_window_s: float = 0.033,
    ) -> list[InferenceResult]:
        """Split an event stream into time windows and infer on each.

        Parameters
        ----------
        event_stream : EventStream
            Loaded events with ``t_s``, ``x``, ``y``, ``p01`` fields.
        time_window_s : float
            Duration of each processing window in seconds.

        Example
        -------
        >>> # See tests/test_super_event_model.py for runnable examples.
        """
        t = event_stream.t_s
        x = event_stream.x
        y = event_stream.y
        p = event_stream.p01

        t_start = t[0]
        t_end = t[-1]

        results: list[InferenceResult] = []
        window_start = t_start

        while window_start < t_end:
            window_end = window_start + time_window_s
            mask = (t >= window_start) & (t < window_end)

            if np.any(mask):
                events_np = np.column_stack([
                    t[mask], x[mask], y[mask], p[mask],
                ])
                event_batch = torch.from_numpy(events_np).float().to(self._device)
            else:
                event_batch = torch.zeros((0, 4), device=self._device)

            result = self.Infer_from_events(event_batch, timestamp=window_end)
            results.append(result)

            window_start = window_end

        return results

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def Map_keypoints_to_sensor_frame(
        self,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """Shift keypoints from cropped-model space to full-sensor pixel space.

        The model operates on a spatially cropped time-surface. This method
        adds the ``crop_offset`` so that the returned coordinates correspond
        to actual pixel locations on the sensor (or undistorted image plane
        when ``dist_coeffs`` was provided).

        Parameters
        ----------
        keypoints : np.ndarray
            ``[N, 2]`` array of ``(row, col)`` in cropped image space.

        Returns
        -------
        np.ndarray
            ``[N, 2]`` array of ``(row, col)`` in full-sensor pixel space,
            same dtype as input.

        Example
        -------
        >>> model = SuperEventModel()  # doctest: +SKIP
        >>> result = model.Infer_from_events(batch, timestamp=0.05)  # doctest: +SKIP
        >>> kp_sensor = model.Map_keypoints_to_sensor_frame(result.keypoints)  # doctest: +SKIP
        """
        if len(keypoints) == 0:
            return keypoints.copy()
        offset = np.array(self._crop_offset, dtype=keypoints.dtype)
        return keypoints + offset

    def Unproject_keypoints(
        self,
        keypoints_sensor: np.ndarray,
    ) -> np.ndarray:
        """Convert sensor-frame pixel keypoints to normalized camera coordinates.

        Applies the inverse camera projection: ``K^{-1} * [x_px, y_px, 1]^T``.
        If distortion coefficients were provided at construction *and*
        time-surface undistortion is disabled, distortion is removed here.
        If the time-surface was already undistorted by ``TsGenerator``, only
        the intrinsic matrix is applied (no further distortion removal).

        Parameters
        ----------
        keypoints_sensor : np.ndarray
            ``[N, 2]`` array of ``(row, col)`` in full-sensor pixel space
            (output of ``Map_keypoints_to_sensor_frame``).

        Returns
        -------
        np.ndarray
            ``[N, 2]`` array of ``(x_norm, y_norm)`` normalized camera
            coordinates (i.e. with focal length and principal point divided
            out; distortion removed if applicable).

        Raises
        ------
        ValueError
            If ``camera_matrix`` was not provided at construction.

        Example
        -------
        >>> model = SuperEventModel(camera_matrix=K)  # doctest: +SKIP
        >>> kp_sensor = model.Map_keypoints_to_sensor_frame(result.keypoints)  # doctest: +SKIP
        >>> kp_norm = model.Unproject_keypoints(kp_sensor)  # doctest: +SKIP
        """
        if self._camera_matrix is None:
            raise ValueError(
                "camera_matrix was not provided at SuperEventModel construction. "
                "Pass camera_matrix (and optionally dist_coeffs) to enable unprojection."
            )

        if len(keypoints_sensor) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # keypoints are (row, col) = (y_px, x_px); cv2 expects (x, y)
        pts_xy = keypoints_sensor[:, ::-1].astype(np.float32).reshape(-1, 1, 2)

        # If TS was undistorted, keypoints are already in undistorted pixel space.
        # Use dist_coeffs only when we have NOT already undistorted the TS.
        dist = None if self._undistort_ts else self._dist_coeffs

        normalized = cv2.undistortPoints(
            pts_xy,
            self._camera_matrix.astype(np.float64),
            dist.astype(np.float64) if dist is not None else None,
        )
        return normalized.reshape(-1, 2)  # (x_norm, y_norm)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def Reset_time_surface(self) -> None:
        """Reset the time-surface generator to a zero state.

        Call this when switching to a new event sequence. Camera intrinsics
        are preserved across resets.

        Example
        -------
        >>> model = SuperEventModel()  # doctest: +SKIP
        >>> model.Reset_time_surface()  # doctest: +SKIP
        """
        self._ts_gen = self._Make_ts_generator()
