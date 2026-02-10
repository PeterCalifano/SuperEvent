from __future__ import annotations

import numpy as np
from pathlib import Path

from data_preparation.util.data_io import load_ts_sparse, save_ts_sparse


def test_sparse_time_surface_roundtrip_preserves_interface(tmp_path: Path) -> None:
    ts = np.zeros((6, 7, 4), dtype=np.float32)
    ts[1, 2, 0] = 0.3
    ts[2, 3, 1] = 0.9
    ts[4, 5, 3] = 1.0

    output_path = tmp_path / "surface_00000000"
    save_ts_sparse(str(output_path), ts)
    loaded = load_ts_sparse(str(output_path) + ".npz")

    assert loaded.shape == ts.shape
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, ts, rtol=0.0, atol=1e-8)
