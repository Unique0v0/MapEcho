#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

PYTHON_BIN="${PYTHON_BIN:-/home/dj/.conda/envs/maptr4090/bin/python}"
TOKENS_FILE="${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
RUN_ROOT="${RUN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_h3_recovery_curve}"

"${PYTHON_BIN}" scripts/package_phase1_h3_recovery_curve.py \
  --tokens-file "${TOKENS_FILE}" \
  --asset-csv "${ASSET_CSV}" \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}"
