#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa}"
TOKENS_FILE="${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}"
OUT_CSV="${OUT_CSV:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_assets_merged.csv}"
OUT_TOKENS="${OUT_TOKENS:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_tokens.txt}"

cd /home/dj/MapEcho

python scripts/merge_ccs_model_scored_assets.py \
  --out-root "${OUT_ROOT}" \
  --tokens-file "${TOKENS_FILE}" \
  --asset-filename ccs_model_scored_best_location_asset_rsa.csv \
  --out-csv "${OUT_CSV}" \
  --out-tokens "${OUT_TOKENS}"
