"""Interface contract tests for ts_generation.generate_ts.TsGenerator.

These tests pin the observable behaviour of TsGenerator that the inference
pipeline depends on:
- Output shape: (H, W, 2 * len(delta_t))
- Output range: [0, 1]
- Batch update == sequential update for the same events
- Coordinate convention: input column 1 → internal x(row), column 2 → y(col)
"""

from __future__ import annotations

import torch

from ts_generation.generate_ts import TsGenerator


def test_ts_generator_get_ts_shape_and_range_contract() -> None:
    """get_ts returns (H, W, 2*len(delta_t)) tensor with values in [0, 1]."""
    gen = TsGenerator(
        settings={"shape": [6, 7], "delta_t": [0.01, 0.05], "undistort": False},
    )
    events = [
        (0.10, 2, 1, 0),
        (0.15, 2, 1, 1),
        (0.20, 4, 3, 0),
    ]
    for t, x, y, p in events:
        gen.update(torch.tensor(t), int(x), int(y), int(p))

    ts = gen.get_ts()
    assert tuple(ts.shape) == (6, 7, 4)
    assert float(ts.min()) >= 0.0
    assert float(ts.max()) <= 1.0


def test_ts_generator_batch_update_matches_sequential_update_contract() -> None:
    """batch_update produces identical timestamps to per-event sequential updates."""
    event_batch = torch.tensor(
        [
            [0.10, 1.0, 2.0, 0.0],
            [0.20, 1.0, 2.0, 0.0],  # same pixel/polarity, newer timestamp wins
            [0.30, 1.0, 2.0, 1.0],  # same pixel, different polarity
            [0.05, 3.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )

    gen_batch = TsGenerator(
        settings={"shape": [8, 8], "delta_t": [0.02], "undistort": False},
    )
    gen_batch.batch_update(event_batch)

    gen_seq = TsGenerator(
        settings={"shape": [8, 8], "delta_t": [0.02], "undistort": False},
    )
    # batch_update interprets col-1 as x(row), col-2 as y(col)
    for event in event_batch:
        gen_seq.update(
            event[0],
            int(event[2].item()),  # row
            int(event[1].item()),  # col
            int(event[3].item()),
        )

    assert torch.allclose(gen_batch.time_stamps, gen_seq.time_stamps)


def test_ts_generator_fresh_state_is_normalized_to_one() -> None:
    """A fresh TsGenerator with no events returns a time-surface of all ones."""
    gen = TsGenerator(
        settings={"shape": [4, 5], "delta_t": [0.01], "undistort": False},
    )
    ts = gen.get_ts()
    assert torch.all(ts == 1.0)


def test_ts_generator_reset_via_reinit_clears_timestamps() -> None:
    """Re-initializing TsGenerator zeroes internal timestamps."""
    gen = TsGenerator(
        settings={"shape": [4, 5], "delta_t": [0.01], "undistort": False},
    )
    gen.update(torch.tensor(0.05), 1, 2, 0)
    assert torch.any(gen.time_stamps != 0)

    gen2 = TsGenerator(
        settings={"shape": [4, 5], "delta_t": [0.01], "undistort": False},
    )
    assert torch.all(gen2.time_stamps == 0)
