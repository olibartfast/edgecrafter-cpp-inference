# EdgeCrafter C++ Inference

C++/OpenCV inference runner for [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) detection and instance segmentation ONNX exports.

This repo includes the full local flow: fetch EdgeCrafter, download a checkpoint, export ONNX, build the C++ app, and run
inference.

## One Command Demo

```bash
scripts/run_demo.sh ecdet_s
```

For segmentation:

```bash
scripts/run_demo.sh ecseg_s
```

The demo writes results under `outputs/`.

## Export ONNX

Export a detection checkpoint:

```bash
scripts/export_edgecrafter_onnx.sh ecdet_s
```

Export a segmentation checkpoint:

```bash
scripts/export_edgecrafter_onnx.sh ecseg_s
```

Supported models: `ecdet_s`, `ecdet_m`, `ecdet_l`, `ecdet_x`, `ecseg_s`, `ecseg_m`, `ecseg_l`, `ecseg_x`,
`ecpose_s`, `ecpose_m`.

The exporter clones `https://github.com/Intellindust-AI-Lab/EdgeCrafter` into `third_party/EdgeCrafter`, creates a Python
3.11 venv there, downloads the selected `.pth` checkpoint into `models/`, and writes `models/<model>.onnx`. PyTorch may
also write `models/<model>.onnx.data`; keep it next to the `.onnx` file.

The exported graph must expose the EdgeCrafter deploy contract:

- inputs: `images` (`NCHW` float tensor) and `orig_target_sizes` (`[N,2]`, width/height)
- detection outputs: `labels`, `boxes`, `scores`
- segmentation outputs: `labels`, `boxes`, `scores`, `masks`
- pose estimation outputs: `labels`, `scores`, `keypoints` (shape `[1, N, 17, 2]` or `[1, N, 17, 3]` — x, y, and optional confidence per keypoint)

## Build C++

Requirements:

- CMake 3.12+
- C++20 compiler
- OpenCV development package
- Network access on first configure, unless ONNX Runtime is already present in the build directory

```bash
cmake -S . -B build -DUSE_ONNX_RUNTIME=ON
cmake --build build -j
```

The build downloads ONNX Runtime 1.21.0 into `build/_deps` when needed.

## Run C++

Detection:

```bash
./build/inference_app ./models/ecdet_s.onnx ./data/dog.jpg ./data/coco.names --threshold 0.5
```

Instance segmentation:

```bash
./build/inference_app ./models/ecseg_s.onnx ./data/dog.jpg ./data/coco.names --segmentation --threshold 0.5
```

Human pose estimation:

```bash
./build/inference_app ./models/ecpose_s.onnx ./data/dog.jpg ./data/coco.names --pose --threshold 0.5
```

Optional output path:

```bash
./build/inference_app ./ecdet_l.onnx ./image.jpg ./data/coco.names --output result.jpg
```

The app writes an annotated image and prints each result with class id, score, box, and mask pixel count for
segmentation.

## Notes

- Preprocessing mirrors EdgeCrafter's Python ONNX inference path: resize to model input size, BGR to RGB, scale to
  `[0,1]`, then ImageNet mean/std normalization.
- EdgeCrafter's exported ONNX graph already performs top-k selection and box scaling through `orig_target_sizes`, so the
  C++ side only filters by score and draws results.
- TensorRT support was intentionally not carried over because EdgeCrafter's deploy graph returns typed outputs,
  including `int64` labels and a second integer input.
- Pose estimation draws the standard COCO 17-keypoint skeleton. Only keypoints with confidence above the
  `keypoint_threshold` (default 0.3) are rendered.
