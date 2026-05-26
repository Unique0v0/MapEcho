#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/paths.env"

export PYTHONPATH="${STREAMMAPNET_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XDG_CACHE_HOME="${MAPECHO_ROOT}/.cache"
export MMCV_HOME="${MAPECHO_ROOT}/.cache/mmcv"
export MPLCONFIGDIR="${MAPECHO_ROOT}/.cache/matplotlib"
export LD_LIBRARY_PATH="/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}"

echo "MAPECHO_ROOT=${MAPECHO_ROOT}"
echo "STREAMMAPNET_ROOT=${STREAMMAPNET_ROOT}"
echo "NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "MAPECHO_DATA_ROOT=${MAPECHO_DATA_ROOT}"
