"""Live streaming demo: TCP or ROS2 events → SuperEvent inference.

Usage
-----
TCP (DV SDK)::

    python -m examples.demo_streaming --source tcp --address 127.0.0.1 --port 4040 \\
        [--model ...] [--config ...] [--time_window 0.033] [--resolution 180 240]

ROS2 (dvs_msgs/EventArray)::

    python -m examples.demo_streaming --source ros2 --topic /dvs/events \\
        [--model ...] [--config ...] [--time_window 0.033] [--resolution 180 240]

Press 'q' to quit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from eventDataGenLibPy import EventStream
from inference.event_sources import Ros2EventSource, TcpEventSource
from inference.event_visualization import (
    Create_inference_summary,
    Render_descriptors_pca,
    Render_events_to_frame,
    Render_keypoints_on_image,
    Render_time_surface,
)
from inference.super_event_model import SuperEventModel


def main() -> None:
    """Run the streaming inference demo."""
    parser = argparse.ArgumentParser(description="SuperEvent live streaming demo")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["tcp", "ros2"],
        help="Event source type.",
    )
    parser.add_argument("--address", default="127.0.0.1", help="TCP server address.")
    parser.add_argument("--port", type=int, default=4040, help="TCP server port.")
    parser.add_argument("--topic", default="/dvs/events", help="ROS2 topic.")
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
        "--top_k",
        type=int,
        default=None,
        help="Maximum number of keypoints per window.",
    )
    args = parser.parse_args()

    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = SuperEventModel(
        config_path=args.config,
        model_path=args.model,
        device=device,
        resolution=args.resolution,
        top_k=args.top_k,
    )

    height, width = args.resolution
    window_count = [0]

    def _On_events(stream: EventStream) -> None:
        events_matrix = stream.To_matrix_t_x_y_p()
        event_batch = torch.from_numpy(events_matrix).float().to(device)
        timestamp = float(stream.t_s[-1]) if len(stream.t_s) > 0 else 0.0

        result = model.infer_from_events(event_batch, timestamp)

        event_frame = Render_events_to_frame(events_matrix, height, width)
        ts_image = Render_time_surface(result.time_surface)
        det_image = Render_keypoints_on_image(
            ts_image.copy(), result.keypoints, result.probabilities,
        )
        desc_image = Render_descriptors_pca(
            result.descriptors, result.keypoints, (height, width),
        )
        mosaic = Create_inference_summary(event_frame, ts_image, det_image, desc_image)

        window_count[0] += 1
        print(
            f"  Window {window_count[0]:04d} | t={timestamp:.4f}s | "
            f"{len(result.keypoints)} keypoints | {len(stream.t_s)} events",
        )

        cv2.imshow("SuperEvent Streaming", mosaic)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            source.Stop()
            cv2.destroyAllWindows()

    resolution_tuple = (args.resolution[0], args.resolution[1])

    if args.source == "tcp":
        source = TcpEventSource(args.address, args.port, resolution=resolution_tuple)
        print(f"Connecting to TCP server at {args.address}:{args.port}...")
    elif args.source == "ros2":
        source = Ros2EventSource(args.topic, resolution=resolution_tuple)
        print(f"Subscribing to ROS2 topic {args.topic}...")

    print(f"Streaming with {args.time_window:.3f} s windows. Press 'q' to quit.")
    source.Start(callback=_On_events, time_window_s=args.time_window)

    # For TCP, the listener runs in a thread — keep main alive
    if args.source == "tcp":
        try:
            import time

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            source.Stop()
            cv2.destroyAllWindows()
            print("\nStopped.")


if __name__ == "__main__":
    main()
