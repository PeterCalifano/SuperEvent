# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperEvent is a PyTorch library for event-based keypoint detection and descriptor learning for SLAM. It was published at ICCV 2025 and won the IROS 2025 EvSLAM Challenge. The core idea is cross-modal learning: using frames with SuperPoint+SuperGlue pseudo labels to train a model that operates on event-based time-surfaces.

Technical reference: `docs/REPOSITORY_OVERVIEW.md`

## Setup

```bash
conda create --name se python=3.12
conda activate se
pip install -e .
pip install -e ".[train]"  # for tensorboard
pip install -e ".[dev]"    # for pytest + build
```

## Commands

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_interface_dataset.py::test_tsdataset_returns_expected_default_interface

# Demo (uses pre-generated example_data/)
python visualize_matches.py

# Training
python train.py --config config/super_event.yaml

# Evaluation (pose consistency on Event-Camera Dataset or EDS)
python evaluate_pose_estimation.py /path/to/evaluation/dataset

# Data preparation pipeline
./data_preparation/prepare_training_data.sh -d /path/to/dataset
```

## Code Style (from AGENTS.md)

- Python 3.12+; PyTorch for ML, sklearn for statistics, seaborn for statistical plots, PIL/OpenCV for images
- **Type hints are mandatory**
- Function names: `CapitalCase` (e.g., `ComputeLoss`); Classes: `CapitalCase` (e.g., `TsDataset`)
- ONNX export compatibility required
- Every new class or function must include a runnable example with printed output in its docstring
- Signature style: first argument on new line after `(`

## Architecture

### Data Flow

```
Raw events → Time-Surfaces (MCTS/TS/TEncode) → Model → Keypoints + Descriptors
                                                              ↓
                                         Matching via SuperPoint+SuperGlue → SLAM
```

**Time-surface representations** (set via `input_representation` in config):

- `mcts`: Multi-Channel Time Surfaces, 10 channels (default)
- `ts`: Single-channel, 1 channel
- `tencode`: Temporal encoding, 4 channels

### Core Model (`models/super_event.py`)

Two variants controlled by `pixel_wise_predictions` config flag:

- `SuperEvent`: Grid-based predictions with 8×8 cell downsampling (default)
- `SuperEventFullRes`: Full-resolution pixel-wise predictions

Both share the same dual-head design:

- `DetectorHead` / `DetectorHeadFullRes` (`models/heads.py`): outputs keypoint probability map
- `DescriptorHead` (`models/heads.py`): outputs L2-normalized descriptor map

Supported backbones (`backbone` config key): `vgg` or `maxvit`

### Loss (`models/losses.py`)

`super_event_loss()` combines:

- `detector_loss()`: cross-entropy (grid) or focal loss (pixel-wise)
- `descriptor_loss()`: contrastive margin loss on positive/negative descriptor pairs
- Weights: `lambda_d` (descriptor vs detector balance), `lambda_loss` (total descriptor loss scale)

### Dataset (`data/dataset.py`)

- `TsDataset`: loads paired time-surface `.npz` + pseudo-label keypoint maps
- `DatasetCollection`: wraps multiple `TsDataset` instances with shuffling across datasets
- Augmentations: homography adaptation (`homography_adaptation` config), temporal matching (`temporal_matching` config)

### Online Inference (`ts_generation/generate_ts.py`)

`TsGenerator` maintains a rolling event buffer to produce time-surface tensors on-the-fly. Used in `evaluate_pose_estimation.py` for streaming inference.

### Data Preparation (`data_preparation/`)

Pipeline stages run by `prepare_training_data.sh`:

1. `convert_images.py` — filter frames without events, optional undistortion
2. `events2ts.py` — generate MCTS at image timestamps
3. `prepare_pseudo_groundtruth.py` — run SuperPoint+SuperGlue to create labels
4. `split_data.py` — organize into `train/`/`val/`/`test/`

Multi-dataset I/O for MVSEC, FPV, DDD20, ViVID, GRIFFIN, ECD, EDS lives in `data_preparation/util/data_io.py`.

## Key Configuration (`config/super_event.yaml`)

Important fields:

- `backbone`: `maxvit` or `vgg`
- `input_representation`: `mcts`, `ts`, `mcts_1`, or `tencode`
- `input_channels`: must match representation (10 for mcts, 1 for ts, 4 for tencode)
- `pixel_wise_predictions`: `False` for grid-based, `True` for full-res
- `train_data_path`: root of prepared training data
- `detection_threshold` / `nms_box_size`: inference post-processing

## Testing

Tests use synthetic data fixtures defined in `tests/conftest.py` — no real dataset needed. The root-level `conftest.py` excludes `ts_generation/test_ts.py` and the vendored MaxViT test file from pytest collection.
