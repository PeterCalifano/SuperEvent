from __future__ import annotations

import torch

from ts_generation.generate_ts import TsGenerator


def test_ts_generator_get_ts_shape_and_range_contract() -> None:
    gen = TsGenerator(settings={"shape": [6, 7], "delta_t": [0.01, 0.05], "undistort": False})
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
    event_batch = torch.tensor(
        [
            [0.10, 1.0, 2.0, 0.0],
            [0.20, 1.0, 2.0, 0.0],  # same pixel/polarity, newer timestamp
            [0.30, 1.0, 2.0, 1.0],  # same pixel, different polarity
            [0.05, 3.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )

    gen_batch = TsGenerator(settings={"shape": [8, 8], "delta_t": [0.02], "undistort": False})
    gen_batch.batch_update(event_batch)

    gen_seq = TsGenerator(settings={"shape": [8, 8], "delta_t": [0.02], "undistort": False})
    for event in event_batch:
        # `batch_update` uses column 2 as x(row) and column 1 as y(column).
        gen_seq.update(event[0], int(event[2].item()), int(event[1].item()), int(event[3].item()))

    assert torch.allclose(gen_batch.time_stamps, gen_seq.time_stamps)
