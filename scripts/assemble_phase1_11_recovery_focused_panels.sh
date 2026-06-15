#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

PYTHON_BIN="${PYTHON_BIN:-/home/dj/.conda/envs/maptr4090/bin/python}"
CASE_CSV="${CASE_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/phase1_11_selected_for_visual_review.csv}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
RUN_ROOT="${RUN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/recovery_focused_panels}"

"${PYTHON_BIN}" scripts/assemble_phase1_11_recovery_focused_panels.py \
  --case-csv "${CASE_CSV}" \
  --asset-csv "${ASSET_CSV}" \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}"
