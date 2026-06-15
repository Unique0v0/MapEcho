#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

H3_DIR="${H3_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_h3_recovery_curve}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset}"

python scripts/select_phase1_11_qualitative_clean_subset.py \
  --h3-dir "${H3_DIR}" \
  --asset-csv "${ASSET_CSV}" \
  --out-dir "${OUT_DIR}"
