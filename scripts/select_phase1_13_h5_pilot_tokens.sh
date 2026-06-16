#!/usr/bin/env bash
set -euo pipefail

H3_DIR="${H3_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_h3_recovery_curve}"
ASSET_CSV="${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}"
PHASE11_TABLE="${PHASE11_TABLE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/phase1_11_candidate_table.csv}"
OUT_DIR="${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot}"

cd /home/dj/MapEcho

python scripts/select_phase1_13_h5_pilot_tokens.py \
  --h3-dir "${H3_DIR}" \
  --asset-csv "${ASSET_CSV}" \
  --phase11-table "${PHASE11_TABLE}" \
  --out-dir "${OUT_DIR}"
