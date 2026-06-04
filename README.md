# EdgeCrafter C++ Inference

C++ inference runner for [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) ONNX exports.
It targets EdgeCrafter dense prediction models after upstream PyTorch export and runs them through ONNX Runtime or an optional TensorRT engine backend.

This repo is not a training fork. It keeps a local deployment path: fetch upstream EdgeCrafter, download one released checkpoint, export ONNX, build C++, and write annotated image output.

## EdgeCrafter Scope

Upstream EdgeCrafter provides compact ViT models for three COCO dense prediction tasks:

| Task | Upstream family | Helper model names | C++ flag | Outputs consumed |
|------|-----------------|--------------------|----------|------------------|
| Object detection | ECDet S/M/L/X | `ecdet_s`, `ecdet_m`, `ecdet_l`, `ecdet_x` | default | `labels`, `boxes`, `scores` |
| Instance segmentation | ECSeg S/M/L/X | `ecseg_s`, `ecseg_m`, `ecseg_l`, `ecseg_x` | `--segmentation` | `labels`, `boxes`, `scores`, `masks` |
| Human pose estimation | ECPose S/M/L/X | `ecpose_s`, `ecpose_m`, `ecpose_l`, `ecpose_x` | `--pose` | `labels`, `scores`, `keypoints` |

All listed upstream model-zoo checkpoints use 640 input size. EdgeCrafter publishes TensorRT latency numbers for FP16 batch-1 inference; this repo can either run exported ONNX directly with ONNX Runtime or run a TensorRT engine built from that ONNX file.

## One Command Demo

```bash
scripts/run_demo.sh ecdet_s
scripts/run_demo.sh ecseg_s
scripts/run_demo.sh ecpose_s
```

The demo exports the requested model if `models/<model>.onnx` is missing, configures the C++ build, runs inference, and writes output under `outputs/`.

## Export ONNX

```bash
scripts/export_edgecrafter_onnx.sh ecdet_s
scripts/export_edgecrafter_onnx.sh ecseg_s
scripts/export_edgecrafter_onnx.sh ecpose_s
```

The exporter clones upstream EdgeCrafter into `third_party/EdgeCrafter`, creates a Python 3.11 virtual environment there, installs task-specific upstream requirements, downloads `models/<model>.pth` from the EdgeCrafter release assets, and writes `models/<model>.onnx`.

Model-to-upstream mapping:

| Helper prefix | Upstream directory | Config path pattern |
|---------------|--------------------|---------------------|
| `ecdet_*` | `ecdetseg` | `configs/ecdet/<model>.yml` |
| `ecseg_*` | `ecdetseg` | `configs/ecseg/<model>.yml` |
| `ecpose_*` | `ecpose` | `configs/ecpose/<model>_coco.yml` |

PyTorch export can also create `models/<model>.onnx.data`; keep it next to the `.onnx` file.

Expected deploy contract:

- inputs: `images` (`NCHW` float tensor) and `orig_target_sizes` (`[N,2]`, width/height)
- detection outputs: `labels`, `boxes`, `scores`
- segmentation outputs: `labels`, `boxes`, `scores`, `masks`
- pose outputs: `labels`, `scores`, `keypoints` with shape `[1, N, 17, 2]` or `[1, N, 17, 3]`

## Build

Requirements:

- CMake 3.12+
- Ninja or another CMake generator
- C++20 compiler
- OpenCV development package
- Network access on first ONNX Runtime configure, unless ONNX Runtime 1.21.0 is already present under `build/_deps`

ONNX Runtime build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX_RUNTIME=ON
cmake --build build --parallel
```

`CMakeLists.txt` downloads ONNX Runtime 1.21.0 into `build/_deps` when needed and copies `libonnxruntime.so.1.21.0` next to `build/inference_app`.

Strict warnings:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX_RUNTIME=ON -DWERROR=ON
cmake --build build --parallel
```

## Run ONNX Runtime

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
./build/inference_app ./models/ecdet_l.onnx ./data/dog.jpg ./data/coco.names --output outputs/ecdet_l_dog.jpg
```

The app writes an annotated image and prints class id, score, box, and either mask pixel count or visible keypoint count when present.

## Run TensorRT

Build TensorRT backend instead of ONNX Runtime:

```bash
cmake -S . -B build-trt -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ONNX_RUNTIME=OFF \
  -DUSE_TENSORRT=ON \
  -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build-trt --parallel
```

Create a fixed-shape TensorRT engine from an exported EdgeCrafter ONNX model:

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

Run engine:

```bash
LD_LIBRARY_PATH=/path/to/TensorRT/lib:${LD_LIBRARY_PATH} \
  ./build-trt/inference_app ./models/ecdet_s.trt.engine ./data/dog.jpg ./data/coco.names --threshold 0.5
```

TensorRT backend uses TensorRT 10-style named tensor APIs. Engine input and output names must match EdgeCrafter deploy contract.

## Output Details

- Preprocessing mirrors upstream ONNX inference: resize to model input, BGR to RGB, scale to `[0,1]`, then ImageNet mean/std normalization.
- EdgeCrafter ONNX graph handles top-k selection and box scaling through `orig_target_sizes`; C++ filters by score and draws results.
- `--pose` applies `label_offset=-1` because pose labels are binary (`background=0`, `person=1`) while `data/coco.names` is 0-indexed COCO.
- Pose models can omit boxes; C++ derives boxes from visible keypoints for labels and rectangles.
- Segmentation masks are alpha-blended at 45%, then boxes and labels draw at full opacity.
- Detection and pose annotations draw at full opacity.
- Per-class colors come from `get_color_for_class()` in `src/processing_utils.cpp`.

## Quality Checks

```bash
find src -name "*.cpp" -o -name "*.hpp" | xargs clang-format-18 --dry-run --Werror
cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --error-exitcode=1 -I src src/
```

## Acknowledgements

This project depends on upstream [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) for model definitions, training code, checkpoints, and ONNX export tooling. EdgeCrafter builds on RT-DETR, D-FINE, DEIM, lightly-train, DETRPose, RF-DETR, and DINOv3.

## Citation

If you use this deployment wrapper or upstream EdgeCrafter models in research, cite EdgeCrafter:

```bibtex
@article{liu2026edgecrafter,
  title={EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation},
  author={Liu, Longfei and Hou, Yongjie and Li, Yang and Wang, Qirui and Sha, Youyang and Yu, Yongjun and Wang, Yinzhi and Ru, Peizhe and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2026}
}
```
