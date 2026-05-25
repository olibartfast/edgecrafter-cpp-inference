#!/usr/bin/env bash
set -euo pipefail

model="${1:-ecdet_s}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

onnx_path="${repo_root}/models/${model}.onnx"
if [[ ! -f "${onnx_path}" ]]; then
  onnx_path="$("${repo_root}/scripts/export_edgecrafter_onnx.sh" "${model}" | tee /dev/stderr | tail -n 1)"
fi

cmake -S "${repo_root}" -B "${repo_root}/build" -DUSE_ONNX_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "${repo_root}/build" -j

extra_args=()
if [[ "${model}" == ecseg_* ]]; then
  extra_args+=(--segmentation)
elif [[ "${model}" == ecpose_* ]]; then
  extra_args+=(--pose)
fi

"${repo_root}/build/inference_app" \
  "${onnx_path}" \
  "${repo_root}/data/dog.jpg" \
  "${repo_root}/data/coco.names" \
  --threshold 0.5 \
  --output "${repo_root}/outputs/${model}_dog.jpg" \
  "${extra_args[@]}"
