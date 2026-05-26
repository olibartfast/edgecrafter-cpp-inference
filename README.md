# EdgeCrafter C++ Inference

C++/OpenCV inference runner for [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) detection, instance
segmentation, and human pose estimation ONNX exports.

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

For pose estimation:

```bash
scripts/run_demo.sh ecpose_s
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

Export a pose checkpoint:

```bash
scripts/export_edgecrafter_onnx.sh ecpose_s
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
- Optional for TensorRT: CUDA toolkit/runtime plus a TensorRT installation

```bash
cmake -S . -B build -DUSE_ONNX_RUNTIME=ON
cmake --build build -j
```

The build downloads ONNX Runtime 1.21.0 into `build/_deps` when needed.

Build with TensorRT instead of ONNX Runtime:

```bash
cmake -S . -B build-trt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build-trt -j
```

The TensorRT backend expects a serialized TensorRT engine file. Create one from an exported EdgeCrafter ONNX model with
`trtexec`, using fixed shapes for EdgeCrafter's two inputs:

```bash
LD_LIBRARY_PATH=/path/to/TensorRT/lib:${LD_LIBRARY_PATH} \
  /path/to/TensorRT/bin/trtexec \
  --onnx=models/ecdet_s.onnx \
  --saveEngine=models/ecdet_s.trt.engine \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --skipInference
```

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
./build/inference_app ./models/ecpose_s.onnx ./data/person.jpg ./data/coco.names --pose --threshold 0.5
```

Optional output path:

```bash
./build/inference_app ./ecdet_l.onnx ./image.jpg ./data/coco.names --output result.jpg
```

TensorRT detection:

```bash
LD_LIBRARY_PATH=/path/to/TensorRT/lib:${LD_LIBRARY_PATH} \
  ./build-trt/inference_app ./models/ecdet_s.trt.engine ./data/dog.jpg ./data/coco.names --threshold 0.5
```

The app writes an annotated image and prints each result with class id, score, box, and mask pixel count for
segmentation or visible keypoint count for pose estimation.

## Notes

- Preprocessing mirrors EdgeCrafter's Python ONNX inference path: resize to model input size, BGR to RGB, scale to
  `[0,1]`, then ImageNet mean/std normalization.
- EdgeCrafter's exported ONNX graph already performs top-k selection and box scaling through `orig_target_sizes`, so the
  C++ side only filters by score and draws results.
- TensorRT support requires TensorRT 10-style named tensor APIs and an engine whose input/output names match the
  EdgeCrafter deploy contract. The runtime handles EdgeCrafter's `int64` `orig_target_sizes` input and `labels` output.
- Pose estimation draws the standard COCO 17-keypoint skeleton. Only keypoints with confidence above
  `keypoint_threshold` (default 0.3) are rendered. The `--pose` flag also applies a `label_offset` of -1 because the
  pose model uses binary classification (background=0, person=1) rather than COCO indexing.
- Detection and segmentation models output `boxes` for every result. The pose model omits `boxes`; the C++ side derives
  bounding boxes from visible keypoints for drawing labels and rectangles.
