#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/paths.env"

export PYTHONPATH="${ATTACK_ROOT}:${ATTACK_ROOT}/mmdetection3d:${ATTACK_ROOT}/projects:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "MAPECHO_ROOT=${MAPECHO_ROOT}"
echo "ATTACK_ROOT=${ATTACK_ROOT}"
echo "NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "MAPECHO_DATA_ROOT=${MAPECHO_DATA_ROOT}"
