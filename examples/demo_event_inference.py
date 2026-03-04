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

import cv2
import numpy as np
import torch

from inference.event_inference import EventInference, EventInferenceSettings
from inference.event_visualization import (
    Create_inference_summary,
    Render_descriptors_pca,
    Render_events_to_frame,
    Render_keypoints_on_image,
    Render_time_surface,
)


def Main() -> None:
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

    settings = EventInferenceSettings(
        resolution=args.resolution,
        config_path=args.config,
        model_path=args.model,
        device=device,
        top_k=args.top_k,
    )

    print("Loading pipeline...")
    pipeline = EventInference(settings)

    print(f"Loading events from {args.event_file}...")
    event_stream = pipeline.Load_events_from_file(args.event_file, format=args.format)
    num_events = len(event_stream.t_s)
    duration = event_stream.t_s[-1] - event_stream.t_s[0]
    fmt = event_stream.source_format or "auto-detected"
    print(f"Loaded {num_events} events spanning {duration:.3f} s (format: {fmt})")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"Running inference with {args.time_window:.3f} s windows...")
    results = pipeline.Process_event_stream(event_stream, time_window_s=args.time_window)

    # Build events matrix for visualization windowing
    events_matrix = event_stream.To_matrix_t_x_y_p()

    print(f"Visualizing {len(results)} windows...")
    t_start = event_stream.t_s[0]
    height, width = args.resolution

    for i, result in enumerate(results):
        window_start = t_start + i * args.time_window
        window_end = window_start + args.time_window
        mask = (events_matrix[:, 0] >= window_start) & (events_matrix[:, 0] < window_end)
        window_events = events_matrix[mask]

        # Render panels
        event_frame = Render_events_to_frame(window_events, height, width)
        ts_image = Render_time_surface(result.time_surface)
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
            cv2.imwrite(os.path.join(args.save_dir, f"window_{i:04d}.png"), mosaic)
        else:
            cv2.imshow("SuperEvent Inference", mosaic)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    print("Done.")


if __name__ == "__main__":
    Main()
