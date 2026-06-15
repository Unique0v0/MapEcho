#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

CASE_CSV="${CASE_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence/cases/qualitative_case_selection.csv}"
RUN_ROOT="${RUN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_10_qualitative_figures/focused_map_panels}"
TOKENS="${TOKENS:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/dj/.conda/envs/maptr4090/bin/python}"

"${PYTHON_BIN}" scripts/assemble_phase1_10_focused_map_panels.py \
  --case-csv "${CASE_CSV}" \
  --run-root "${RUN_ROOT}" \
  --asset-csv "${ASSET_CSV}" \
  --out-dir "${OUT_DIR}" \
  --tokens "${TOKENS}"
