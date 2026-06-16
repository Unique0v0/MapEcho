#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE="${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/h5_primary_scene_pilot_tokens.txt}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
RUN_ROOT="${RUN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/n_attack3_controlled_check}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/n_attack3_recovery_summary}"
N_ATTACK="${N_ATTACK:-3}"
OFFSET_START="${OFFSET_START:-${N_ATTACK}}"
OFFSET_END="${OFFSET_END:-9}"
PYTHON_BIN="${PYTHON_BIN:-/home/dj/.conda/envs/maptr4090/bin/python}"

cd /home/dj/MapEcho

LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-} \
"${PYTHON_BIN}" scripts/package_phase1_h3_recovery_curve.py \
  --tokens-file "${TOKENS_FILE}" \
  --asset-csv "${ASSET_CSV}" \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --offset-start "${OFFSET_START}" \
  --offset-end "${OFFSET_END}"
