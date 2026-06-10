#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

: "${PYTHON_BIN:=python}"
: "${RULE_DIR:=/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild}"
: "${NEWSPLIT_VAL_ANN:=/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}"
: "${OUT_DIR:=/data/dj/MapEcho/artifacts/phase1_8b_assets}"
: "${WARMUP:=10}"
: "${RECOVERY:=9}"
: "${MAX_PER_SCENE:=5}"
: "${TARGET_MAX:=120}"

"${PYTHON_BIN}" scripts/build_phase1_8b_assets.py \
  --rule-dir "${RULE_DIR}" \
  --newsplit-val-ann "${NEWSPLIT_VAL_ANN}" \
  --out-dir "${OUT_DIR}" \
  --warmup "${WARMUP}" \
  --recovery "${RECOVERY}" \
  --max-per-scene "${MAX_PER_SCENE}" \
  --target-max "${TARGET_MAX}"
