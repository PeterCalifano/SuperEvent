from __future__ import annotations

from typing import Any, Callable

import numpy as np

from data.dataset import DataSplit, DatasetCollection, TsDataset


def test_tsdataset_returns_expected_default_interface(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("mcts")
    dataset = TsDataset(DataSplit.test, config, vis_mode=False)

    assert len(dataset) == 1
    ts0, ts1, kp_map0, kp_map1 = dataset[0]
    assert ts0.shape == (10, 8, 10)
    assert ts1.shape == (10, 8, 10)
    assert kp_map0.shape == (8, 10)
    assert kp_map1.shape == (8, 10)
    assert np.max(kp_map0) >= 1.0


def test_tsdataset_representation_ts_interface(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("ts")
    dataset = TsDataset(DataSplit.test, config, vis_mode=False)
    ts0, ts1, *_ = dataset[0]
    assert ts0.shape[0] == 1
    assert ts1.shape[0] == 1


def test_tsdataset_representation_mcts1_interface(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("mcts_1")
    dataset = TsDataset(DataSplit.test, config, vis_mode=False)
    ts0, ts1, *_ = dataset[0]
    assert ts0.shape[0] == 2
    assert ts1.shape[0] == 2


def test_tsdataset_representation_tencode_interface(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("tencode")
    dataset = TsDataset(DataSplit.test, config, vis_mode=False)
    sample = dataset[0]
    assert len(sample) == 6
    tencode_l0, tencode_l1, kp_map0, kp_map1, tencode_m0, tencode_h0 = sample
    assert tencode_l0.shape == (1, 8, 10)
    assert tencode_l1.shape == (1, 8, 10)
    assert tencode_m0.shape == (1, 8, 10)
    assert tencode_h0.shape == (1, 8, 10)
    assert kp_map0.shape == (8, 10)
    assert kp_map1.shape == (8, 10)


def test_tsdataset_vis_mode_interface_includes_frames_and_identifier(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("mcts")
    dataset = TsDataset(DataSplit.test, config, vis_mode=True)
    sample = dataset[0]
    assert len(sample) == 7

    _, _, _, _, frame0, frame1, identifier = sample
    assert frame0.shape == (8, 10, 3)
    assert frame1.shape == (8, 10, 3)
    assert identifier == "fpv_seq0_00000000_00000001"


def test_dataset_collection_interface_wraps_tsdataset(
    dataset_config_factory: Callable[[str], dict[str, Any]],
) -> None:
    config = dataset_config_factory("mcts")
    collection = DatasetCollection(DataSplit.test, config, vis_mode=False)
    assert len(collection) == 1
    sample = collection[0]
    assert len(sample) == 4
