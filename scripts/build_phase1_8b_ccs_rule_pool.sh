#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

: "${PYTHON_BIN:=/home/dj/.conda/envs/maptr4090/bin/python}"
: "${DATA_ROOT:=/data/yuy/dataset/nuScenes/full}"
: "${NEWSPLIT_VAL_ANN:=/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}"
: "${OUT_DIR:=/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild}"
: "${LIMIT:=0}"
: "${MAX_PER_SCENE_INPUT:=0}"

MPLCONFIGDIR=/tmp "${PYTHON_BIN}" scripts/build_phase1_8b_ccs_rule_pool.py \
  --data-root "${DATA_ROOT}" \
  --newsplit-val-ann "${NEWSPLIT_VAL_ANN}" \
  --out-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --max-per-scene-input "${MAX_PER_SCENE_INPUT}"
