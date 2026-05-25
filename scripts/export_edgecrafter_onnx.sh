#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_edgecrafter_onnx.sh [model]

Models:
  ecdet_s  ecdet_m  ecdet_l  ecdet_x
  ecseg_s  ecseg_m  ecseg_l  ecseg_x
  ecpose_s  ecpose_m

Environment overrides:
  EDGECRAFTER_DIR   EdgeCrafter checkout directory (default: third_party/EdgeCrafter)
  PYTHON            Python executable used to create the venv (default: python3.11 when available, else python3)

Output:
  models/<model>.onnx
USAGE
}

model="${1:-ecdet_s}"
case "${model}" in
  ecdet_s|ecdet_m|ecdet_l|ecdet_x)
    task_dir="ecdetseg"
    config="configs/ecdet/${model}.yml"
    ;;
  ecseg_s|ecseg_m|ecseg_l|ecseg_x)
    task_dir="ecdetseg"
    config="configs/ecseg/${model}.yml"
    ;;
  ecpose_s|ecpose_m)
    task_dir="ecpose"
    config="configs/ecpose/${model}.yml"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unsupported model: ${model}" >&2
    usage >&2
    exit 1
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
edgecrafter_dir="${EDGECRAFTER_DIR:-${repo_root}/third_party/EdgeCrafter}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="${PYTHON}"
elif command -v python3.11 >/dev/null 2>&1; then
  python_bin="python3.11"
else
  python_bin="python3"
fi
venv_dir="${edgecrafter_dir}/.venv"
deps_marker="${venv_dir}/.edgecrafter_cpp_inference_deps"
weights_dir="${repo_root}/models"
checkpoint="${weights_dir}/${model}.pth"
onnx_out="${weights_dir}/${model}.onnx"
checkpoint_url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/${model}.pth"

mkdir -p "${weights_dir}" "$(dirname "${edgecrafter_dir}")"

if [[ ! -d "${edgecrafter_dir}/.git" ]]; then
  git clone --depth 1 https://github.com/Intellindust-AI-Lab/EdgeCrafter "${edgecrafter_dir}"
fi

if [[ -x "${venv_dir}/bin/python" ]]; then
  venv_version="$("${venv_dir}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${venv_version}" != "${requested_version}" ]]; then
    rm -rf "${venv_dir}"
  fi
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi

if [[ ! -f "${deps_marker}" ]]; then
  "${venv_dir}/bin/python" -m pip install --upgrade pip
  "${venv_dir}/bin/python" -m pip install -r "${edgecrafter_dir}/${task_dir}/requirements.txt"
  "${venv_dir}/bin/python" -m pip install onnx onnxsim onnxscript
  touch "${deps_marker}"
fi

if [[ ! -f "${checkpoint}" ]]; then
  curl -L --fail --retry 3 -o "${checkpoint}" "${checkpoint_url}"
fi

pushd "${edgecrafter_dir}/${task_dir}" >/dev/null
"${venv_dir}/bin/python" tools/deployment/export_onnx.py \
  -c "${config}" \
  -r "${checkpoint}" \
  --check \
  --simplify
popd >/dev/null

generated="${checkpoint%.pth}.onnx"
if [[ ! -f "${generated}" ]]; then
  echo "Expected export not found: ${generated}" >&2
  exit 1
fi

if [[ "$(realpath "${generated}")" != "$(realpath -m "${onnx_out}")" ]]; then
  mv "${generated}" "${onnx_out}"
fi
echo "${onnx_out}"
