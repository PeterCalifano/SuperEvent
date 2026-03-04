# SuperEvent ONNX Model: C++ Integration Guide

This document describes what is included in the ONNX exported by:

- `scripts/export_onnx_from_paths.py`

and how to run it from C++ with ONNX Runtime.

## What The Exported ONNX Includes

The exported graph is deployment-oriented and contains:

1. Input tensor pre-processing:

- layout conversion `NHWC -> NCHW`
- spatial crop to the model-valid region (computed from config + resolution)

1. Core network forward:

- repository model class from `models/super_event.py`
- `SuperEvent` or `SuperEventFullRes` selected from config
- exported with tracing outputs `(prob, descriptors)`

1. Runtime outputs:

- `prob`
- `descriptors`

## What Is Not In The ONNX

The graph intentionally excludes Python-only harness logic:

1. Checkpoint loading
2. Event-stream accumulation / `TsGenerator`
3. Post-processing helpers (`fast_nms`, keypoint extraction, descriptor sampling)
4. File I/O and visualization

These steps are expected to be handled by your C++ application around the ONNX inference.

## Input / Output Contract

### Input

- Name: `time_surface` (default; configurable with `--input-name`)
- Layout: `NHWC`
- Type: `float32`
- Shape: `[N, H, W, C]`
  - `N`: dynamic batch
  - `H, W`: full sensor resolution passed at export (`--resolution`)
  - `C`: `config["input_channels"]` (e.g. `10` for default config)

Important:

- Feed a time-surface tensor already computed by your C++ pipeline.
- Do not pre-crop or permute to NCHW; ONNX graph already does that.

### Outputs

- `prob`: detector probability map
- `descriptors`: dense descriptor map

Both output tensors have dynamic batch axis. Spatial/output channel dimensions depend on config/backbone and whether `pixel_wise_predictions` is enabled.

## Export Command (Reference)

From repository root:

```bash
python scripts/export_onnx_from_paths.py \
  --config-path config/super_event.yaml \
  --checkpoint-path saved_models/super_event_weights.pth \
  --resolution 180 240 \
  --output-path exports \
  --onnx-name super_event_e2e \
  --validate
```

## C++ ONNX Runtime Integration

## 1) Build-time dependencies

You need:

1. ONNX Runtime C++ package
2. Standard C++17 toolchain
3. Your own time-surface generator that outputs contiguous `float32` NHWC data

## 2) Session initialization

```cpp
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

struct SuperEventOnnx {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "superevent"};
  Ort::SessionOptions session_opts;
  Ort::Session session{nullptr};

  std::string input_name;
  std::vector<std::string> output_names;

  explicit SuperEventOnnx(const std::string& onnx_path) {
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, onnx_path.c_str(), session_opts);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name = session.GetInputNameAllocated(0, allocator).get();

    const size_t num_outputs = session.GetOutputCount();
    output_names.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    }
  }
};
```

## 3) Inference call

```cpp
// Example single-batch inference input.
// Expected layout: NHWC float32.
// Shape example: [1, 180, 240, 10]
std::vector<float> time_surface_nhwc;
std::array<int64_t, 4> input_shape = {1, 180, 240, 10};

Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault);

Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info,
    time_surface_nhwc.data(),
    time_surface_nhwc.size(),
    input_shape.data(),
    input_shape.size());

std::array<const char*, 1> input_names = {"time_surface"};
std::vector<const char*> output_names = {"prob", "descriptors"};

// session is an Ort::Session created as shown above
auto outputs = session.Run(
    Ort::RunOptions{nullptr},
    input_names.data(),
    &input_tensor,
    1,
    output_names.data(),
    output_names.size());

// outputs[0] -> prob
// outputs[1] -> descriptors
```

## 4) Output tensor reading

```cpp
float* prob_ptr = outputs[0].GetTensorMutableData<float>();
float* desc_ptr = outputs[1].GetTensorMutableData<float>();

auto prob_info = outputs[0].GetTensorTypeAndShapeInfo();
auto desc_info = outputs[1].GetTensorTypeAndShapeInfo();

std::vector<int64_t> prob_shape = prob_info.GetShape();
std::vector<int64_t> desc_shape = desc_info.GetShape();
```

You can then implement C++ post-processing equivalent to repository Python pipeline (`NMS`, thresholding, descriptor sampling at keypoints).

## Practical Notes

1. Use the same config assumptions at export and deployment time (`input_channels`, `resolution`).
2. Feed normalized/typed input exactly as `float32`.
3. Keep ONNX Runtime CPU/GPU provider selection explicit for reproducibility.
4. If dynamo export fails at export time, script already falls back to legacy exporter.
