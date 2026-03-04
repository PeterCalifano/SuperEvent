"""ONNX export module for SuperEvent models.

Exports to ONNX using TorchScript backend with optional validation
and simplification.

Example
-------
>>> from inference.export_onnx import Export_model_to_onnx
>>> path = Export_model_to_onnx(output_path="exports/")  # doctest: +SKIP
>>> print(path)  # doctest: +SKIP
exports/super_event_0.onnx

CLI usage::

    python -m inference.export_onnx --output exports/ --validate
"""

from __future__ import annotations

import argparse
import os

import torch
from pyTorchAutoForge.api.onnx import ModelHandlerONNx

from inference.super_event_model import Compute_crop_mask, Load_config
from models.super_event import SuperEvent, SuperEventFullRes


def Export_model_to_onnx(config_path: str = "config/super_event.yaml",
                         model_path: str = "saved_models/super_event_weights.pth",
                         output_path: str = "exports",
                         resolution: list[int] | None = None,
                         opset_version: int = 17,
                         validate: bool = True,
                         simplify: bool = False,
                         ) -> str:
    """Export a SuperEvent model to ONNX format.

    Parameters
    ----------
    config_path : str
        Path to the model config YAML.
    model_path : str
        Path to the ``.pth`` model weights.
    output_path : str
        Directory for the exported ``.onnx`` file.
    resolution : list[int] or None
        Sensor resolution ``[H, W]``. Defaults to ``[180, 240]``.
    opset_version : int
        ONNX opset version.
    validate : bool
        Run ``onnx.checker.check_model`` and ORT inference after export.
    simplify : bool
        Run ``onnxsim.simplify`` on the exported model.

    Returns
    -------
    str
        Path to the exported ``.onnx`` file.

    Example
    -------
    >>> path = Export_model_to_onnx(output_path="/tmp/onnx_test")  # doctest: +SKIP
    """
    if resolution is None:
        resolution = [180, 240]

    # Load config
    config = Load_config(config_path)
    _, cropped_shape = Compute_crop_mask(resolution, config)

    # Create model with tracing=True (returns (prob, descriptors) tuple)
    if config["pixel_wise_predictions"]:
        model = SuperEventFullRes(config, tracing=True)
    else:
        model = SuperEvent(config, tracing=True)

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu"),
    )
    model.eval()

    # Dummy input
    input_channels = config["input_channels"]
    dummy_input = torch.randn(
        1, input_channels, cropped_shape[0], cropped_shape[1])

    # IO naming
    io_names = {
        "input": ["time_surface"],
        "output": ["prob", "descriptors"],
    }
    dynamic_axes: dict[str, dict[int, str]] = {
        "time_surface": {0: "batch_size"},
        "prob": {0: "batch_size"},
        "descriptors": {0: "batch_size"},
    }

    os.makedirs(output_path, exist_ok=True)
    onnx_handler = ModelHandlerONNx(model=model,
                                    dummy_input_sample=dummy_input,
                                    onnx_export_path=output_path,
                                    opset_version=opset_version,
                                    run_export_validation=validate,
                                    generate_report=False,
                                    run_onnx_simplify=simplify,
                                    )
    onnx_filepath = onnx_handler.export_onnx(
        model_inputs=dummy_input,
        onnx_model_name="super_event_0",
        dynamic_axes=dynamic_axes,
        IO_names=io_names,
        backend="legacy",
        fallback_to_legacy=False,
    )

    print(f"ONNX model exported to: {onnx_filepath}")
    return onnx_filepath


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export SuperEvent model to ONNX")
    parser.add_argument(
        "--config",
        default="config/super_event.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--model",
        default="saved_models/super_event_weights.pth",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--output",
        default="exports",
        help="Output directory for the .onnx file.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[180, 240],
        help="Sensor resolution as height width.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after export.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the ONNX model with onnxsim.",
    )
    args = parser.parse_args()

    Export_model_to_onnx(config_path=args.config,
                         model_path=args.model,
                         output_path=args.output,
                         resolution=args.resolution,
                         opset_version=args.opset,
                         validate=args.validate,
                         simplify=args.simplify,
                         )


if __name__ == "__main__":
    main()
