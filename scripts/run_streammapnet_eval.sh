#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"

PYTHON_BIN="${PYTHON_BIN:-/home/dj/.conda/envs/maptr4090/bin/python}"
GPUS="${GPUS:-1}"

cd "${STREAMMAPNET_ROOT}"
export PYTHONPATH="${STREAMMAPNET_ROOT}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${MAPECHO_ROOT}/.cache"
export MMCV_HOME="${MAPECHO_ROOT}/.cache/mmcv"
export MPLCONFIGDIR="${MAPECHO_ROOT}/.cache/matplotlib"
export LD_LIBRARY_PATH="/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}"

if [[ "${GPUS}" == "1" ]]; then
  "${PYTHON_BIN}" tools/test.py "${STREAMMAPNET_CONFIG}" "${STREAMMAPNET_CKPT}" --eval
else
  bash tools/dist_test.sh "${STREAMMAPNET_CONFIG}" "${STREAMMAPNET_CKPT}" "${GPUS}" --eval
fi
