#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

CASE_CSV="${CASE_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence/cases/qualitative_case_selection.csv}"
RUN_ROOT="${RUN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_10_qualitative_figures}"
MAX_CASES="${MAX_CASES:-15}"

python scripts/assemble_phase1_10_qualitative_figures.py \
  --case-csv "${CASE_CSV}" \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --max-cases "${MAX_CASES}"
