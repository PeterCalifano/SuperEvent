# SuperEvent Repository Overview

## Purpose

This repository contains:
- data preparation tools for event-camera datasets,
- model definitions for event-based keypoint detection and description,
- training and evaluation scripts,
- utilities for visualization and diagnostics.

The core model is `models/super_event.py`.

## Directory Layout

- `config/`
  - Experiment and backbone YAML configuration files.
- `data/`
  - Dataset loaders and geometric augmentation helpers used by training/inference scripts.
- `data_preparation/`
  - End-to-end preprocessing pipeline:
    - frame filtering/undistortion,
    - time-surface generation,
    - pseudo-label generation via SuperPoint + SuperGlue,
    - split assignment (train/val/test).
- `models/`
  - Backbones, heads, losses, and model wrappers.
- `ts_generation/`
  - Online time-surface generator used for streaming inference/evaluation.
- `util/`
  - Shared training/evaluation/visualization helpers.
- `saved_models/`
  - Example pretrained weights.
- `train.py`
  - Main training entry point.
- `evaluate_pose_estimation.py`
  - Pose-consistency evaluation entry point.
- `visualize_matches.py`
  - Qualitative match visualization entry point.

## Installation

Install in editable mode from the repository root:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[train]"
pip install -e ".[dev]"
```

## Data Flow

1. Raw event-camera sequence input.
2. `data_preparation.convert_images`: frame filtering/undistortion + timestamp export.
3. `data_preparation.events2ts`: event stream -> sparse multi-channel time-surfaces.
4. `data_preparation.prepare_pseudo_groundtruth`: pseudo matches from frame pairs.
5. `data_preparation.split_data`: folder move into train/val/test structure.
6. `data.dataset.TsDataset`: load paired time-surfaces + pseudo-label maps.
7. `models.super_event.SuperEvent` / `models.super_event.SuperEventFullRes`: prediction.
8. `models.util.fast_nms` + descriptor extraction for downstream matching.

## Core Interfaces

- Time-surface generator:
  - `ts_generation.generate_ts.TsGenerator`
- Dataset loaders:
  - `data.dataset.DatasetCollection`
  - `data.dataset.TsDataset`
- Models:
  - `models.super_event.SuperEvent`
  - `models.super_event.SuperEventFullRes`
- Heads and losses:
  - `models.heads.*`
  - `models.losses.super_event_loss`
- Post-processing:
  - `models.util.fast_nms`
  - `util.eval_utils.extract_keypoints_and_descriptors`

## Notes on External/Vendored Code

- `data_preparation/SuperGluePretrainedNetwork/` is vendored third-party code for pseudo-label generation.
- `models/backbones/maxvit_backbone/` contains third-party/adapted backbone components.

These directories are used as dependencies by first-party scripts, but are maintained separately from the repository's primary interfaces.
