"""Runnable demo: event file --> SuperEvent inference with visualization.

Supports AEDAT4, AEDAT2, HDF5, and TXT event file formats.

Usage
-----
    python -m examples.demo_event_inference <path_to_event_file> \\
        [--format aedat4|aedat2|h5|txt] \\
        [--model saved_models/super_event_weights.pth] \\
        [--config config/super_event.yaml] \\
        [--time_window 0.033] \\
        [--resolution 180 240] \\
        [--save_dir results/]

Press any key for the next window, 'q' to quit.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

import data
from inference.event_inference import EventInference, EventInferenceSettings
from inference.event_visualization import (
    Create_inference_summary,
    Render_descriptors_pca,
    Render_events_to_frame,
    Render_keypoints_on_image,
    Render_time_surface,
)


def main() -> None:
    """Run the event-file inference demo."""
    parser = argparse.ArgumentParser(
        description="SuperEvent inference demo (AEDAT4, AEDAT2, HDF5, TXT)",
    )
    parser.add_argument(
        "event_file",
        type=str,
        help="Path to an event file (.aedat4, .aedat, .h5, .hdf5, .txt, .events)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["aedat4", "aedat2", "h5", "txt"],
        help="Force event file format instead of auto-detecting from extension.",
    )
    parser.add_argument(
        "--model",
        default="saved_models/super_event_weights.pth",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--config",
        default="config/super_event.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=0.033,
        help="Time window duration in seconds.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[180, 240],
        help="Sensor resolution as height width.",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="Directory to save output images. If empty, use imshow.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Maximum number of keypoints per window.",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Setup inference session
    settings = EventInferenceSettings(resolution=args.resolution,
                                      config_path=args.config,
                                      model_path=args.model,
                                      device=device,
                                      top_k=args.top_k,
                                      )

    print("Loading pipeline...")
    pipeline = EventInference(settings)

    # Load events from file using EventDataGenerationLib loader
    print(f"Loading events from {args.event_file}...")
    
    event_stream = pipeline.Load_events_from_file(args.event_file, 
                                                  format=args.format)
    
    num_events = len(event_stream.t_s)
    duration = event_stream.t_s[-1] - event_stream.t_s[0]
    fmt = event_stream.source_format or "auto-detected"
    print(
        f"Loaded {num_events} events spanning {duration:.3f} s (format: {fmt})")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Run inference in time windows and visualize
    print(f"Running inference with {args.time_window:.3f} s windows...")
    results = pipeline.Process_event_stream(
        event_stream, time_window_s=args.time_window)

    # Build events matrix for visualization windowing
    events_matrix = event_stream.to_matrix_t_x_y_p()

    print(f"Visualizing {len(results)} windows...")
    t_start = event_stream.t_s[0]
    height, width = args.resolution

    for i, result in enumerate(results):
        window_start = t_start + i * args.time_window
        window_end = window_start + args.time_window
        mask = (events_matrix[:, 0] >= window_start) & (
            events_matrix[:, 0] < window_end)
        window_events = events_matrix[mask]

        # Render panels for visualization
        event_frame = Render_events_to_frame(window_events, height, width)

        # Contruct time surface visualization
        ts_image = Render_time_surface(result.time_surface)

        # Add keypoints and projected descriptors to the time surface image
        detection_image = Render_keypoints_on_image(
            ts_image.copy(), result.keypoints, result.probabilities,
        )
        descriptor_image = Render_descriptors_pca(
            result.descriptors, result.keypoints, (height, width),
        )

        mosaic = Create_inference_summary(
            event_frame, ts_image, detection_image, descriptor_image,
        )

        print(
            f"  Window {i:04d} | t={result.timestamp:.4f}s | "
            f"{len(result.keypoints)} keypoints | "
            f"{len(window_events)} events",
        )

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir,
                        f"window_{i:04d}.png"), mosaic)
        else:
            cv2.imshow("SuperEvent Inference", mosaic)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    print("Done.")


# %% Manual run
if __name__ == "__main__":

    # Setup options for manual run
    this_repo_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    data_folder = os.getenv("SCRATCH_PRO")
    assert data_folder is not None, "Please set SCRATCH_PRO environment variable to run the demo."
    data_folder = os.path.join(data_folder, "event_test_files")

    event_filepath = "dvSave-2026_01_14_17_55_45.aedat4"  # Set event file path here
    config_filepath = os.path.join(this_repo_folder, "config/super_event.yaml")  # Set config file path here
    # Set model checkpoint path here
    model_checkpoint_filepath = os.path.join(this_repo_folder, "saved_models/super_event_weights.pth")
    
    resolution = (480, 640)  # Set sensor resolution here (height, width)

    python_inputs = {
        "event_file": os.path.join(data_folder, event_filepath),  # placeholder
        "format": None,  # e.g. "aedat4", "aedat2", "h5", "txt"
        "model": model_checkpoint_filepath,
        "config": config_filepath,
        "time_window": 0.033,
        "resolution": resolution,
        "save_dir": "",
        "top_k": None,
    }

    cli_args = ["demo_event_inference.py", python_inputs["event_file"]]
    # TODO understand window size requirement and display options. Replace implementation of viz functions with functions from EventDatasetGen library

    if python_inputs["format"] is not None:
        cli_args.extend(["--format", str(python_inputs["format"])])

    cli_args.extend(
        [
            "--model",
            str(python_inputs["model"]),
            "--config",
            str(python_inputs["config"]),
            "--time_window",
            str(python_inputs["time_window"]),
            "--resolution",
            str(python_inputs["resolution"][0]),
            str(python_inputs["resolution"][1]),
        ],
    )

    if python_inputs["save_dir"]:
        cli_args.extend(["--save_dir", str(python_inputs["save_dir"])])
    if python_inputs["top_k"] is not None:
        cli_args.extend(["--top_k", str(python_inputs["top_k"])])

    sys.argv = cli_args
    main()
