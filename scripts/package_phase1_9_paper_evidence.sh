#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

SUMMARY_DIR="${SUMMARY_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence}"

python scripts/package_phase1_9_paper_evidence.py \
  --summary-dir "${SUMMARY_DIR}" \
  --asset-csv "${ASSET_CSV}" \
  --out-dir "${OUT_DIR}"
