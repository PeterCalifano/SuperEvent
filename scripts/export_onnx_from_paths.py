"""Export an ONNX graph for SuperEvent.

This script exports only runtime inference graph parts:
1) tensor pre-processing needed by model input (NHWC -> NCHW + crop),
2) core SuperEvent network forward pass.

Checkpoint loading is used only during export to materialize parameters into
the ONNX graph. It is not part of runtime inference graph.

Usage
-----
python scripts/export_onnx_from_paths.py \
    --config-path config/super_event.yaml \
    --checkpoint-path saved_models/super_event_weights.pth \
    --resolution 180 240 \
    --output-path exports \
    --onnx-name super_event_e2e

Example output
--------------
Model built from repository classes: SuperEvent
Checkpoint loaded from: saved_models/super_event_weights.pth
ONNX model exported to: exports/super_event_e2e.onnx
"""

from __future__ import annotations
from models.super_event import SuperEvent, SuperEventFullRes
from inference.super_event_model import (
    Compute_crop_offset,
    Compute_crop_mask,
    Load_config,
)

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from pyTorchAutoForge.api.onnx import ModelHandlerONNx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SuperEventModelONNx(nn.Module):
    """Wrap SuperEvent core model with in-graph pre-processing.

    Runtime input is expected as NHWC time-surface tensor:
    [N, H, W, C].
    """

    def __init__(self,
                 core_model: nn.Module,
                 crop_row_start: int,
                 crop_col_start: int,
                 crop_height: int,
                 crop_width: int,
                 ) -> None:
        
        super().__init__()
        self.core_model = core_model
        self.crop_row_start = crop_row_start
        self.crop_col_start = crop_col_start
        self.crop_height = crop_height
        self.crop_width = crop_width

    def forward(self, time_surface: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NHWC -> NCHW conversion, crop, then SuperEvent forward."""

        x = time_surface.permute(0, 3, 1, 2)
        x = x[
            :,
            :,
            self.crop_row_start:self.crop_row_start + self.crop_height,
            self.crop_col_start:self.crop_col_start + self.crop_width,
        ]
        prob, descriptors = self.core_model(x)
        return prob, descriptors


def Extract_state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a torch state_dict from common checkpoint formats."""

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary-like state_dict.")

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state_dict[normalized_key] = value

    if not normalized_state_dict:
        raise ValueError("No tensor parameters found in checkpoint.")
    return normalized_state_dict


def Build_super_event_core_model(config: dict[str, Any],
                                 checkpoint_path: str,
                                 strict_load: bool,
                                 ) -> nn.Module:
    """Create repository SuperEvent model class and load checkpoint into it."""

    if config["pixel_wise_predictions"]:
        core_model: nn.Module = SuperEventFullRes(config, tracing=True)
    else:
        core_model = SuperEvent(config, tracing=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = Extract_state_dict_from_checkpoint(checkpoint)
    core_model.load_state_dict(state_dict, strict=strict_load)
    core_model.eval()
    return core_model


def Export_to_onnx(model: nn.Module,
                   dummy_input: torch.Tensor,
                   output_path: str,
                   onnx_name: str,
                   opset_version: int,
                   validate: bool,
                   simplify: bool,
                   input_name: str,
                   ) -> str:
    """Export model with dynamo-first backend and legacy fallback."""
    output_names = ["prob", "descriptors"]

    dynamic_axes: dict[str, dict[int, str]] = {input_name: {0: "batch_size"}}
    dynamic_axes["prob"] = {0: "batch_size"}
    dynamic_axes["descriptors"] = {0: "batch_size"}

    io_names = {
        "input": [input_name],
        "output": output_names,
    }

    onnx_handler = ModelHandlerONNx(model=model,
                                    dummy_input_sample=dummy_input,
                                    onnx_export_path=output_path,
                                    opset_version=opset_version,
                                    run_export_validation=validate,
                                    generate_report=False,
                                    run_onnx_simplify=simplify,
                                    )

    return onnx_handler.export_onnx(model_inputs=dummy_input,
                                    onnx_model_name=onnx_name,
                                    dynamic_axes=dynamic_axes,
                                    IO_names=io_names,
                                    backend="dynamo",
                                    fallback_to_legacy=True,
                                    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export C++-oriented SuperEvent ONNX from repo model + checkpoint.",
    )
    parser.add_argument(
        "--config-path",
        default="config/super_event.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="saved_models/super_event_weights.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[180, 240],
        help="Full sensor resolution as H W.",
    )
    parser.add_argument(
        "--output-path",
        default="exports",
        help="Output directory for ONNX model.",
    )
    parser.add_argument(
        "--onnx-name",
        default="model_export",
        help="Output ONNX filename without extension.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--input-name",
        default="time_surface",
        help="ONNX input tensor name (NHWC input).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy export batch size.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ONNX validation after export.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model if onnx-simplifier is available.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Enable strict state_dict loading.",
    )

    args = parser.parse_args()

    config = Load_config(args.config_path)
    _, cropped_shape = Compute_crop_mask(args.resolution, config)
    crop_offset = Compute_crop_offset(args.resolution, config)

    core_model = Build_super_event_core_model(
        config=config,
        checkpoint_path=args.checkpoint_path,
        strict_load=args.strict_load,
    )
    print(
        f"Model built from repository classes: {core_model.__class__.__name__}",
    )
    print(f"Checkpoint loaded from: {args.checkpoint_path}")

    # Model to export
    model = SuperEventModelONNx(core_model=core_model,
                                crop_row_start=crop_offset[0],
                                crop_col_start=crop_offset[1],
                                crop_height=cropped_shape[0],
                                crop_width=cropped_shape[1],
                                )
    model.eval()

    input_channels = int(config["input_channels"])
    dummy_input = torch.randn(args.batch_size,
                              args.resolution[0],
                              args.resolution[1],
                              input_channels,
                              )

    onnx_path = Export_to_onnx(model=model,
                               dummy_input=dummy_input,
                               output_path=args.output_path,
                               onnx_name=args.onnx_name,
                               opset_version=args.opset,
                               validate=args.validate,
                               simplify=args.simplify,
                               input_name=args.input_name,
                               )
    print(f"ONNX model exported to: {onnx_path}")


if __name__ == "__main__":
    main()
