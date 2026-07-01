#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dj/MapEcho
PYTHON=${PYTHON:-/home/dj/.conda/envs/maptr4090/bin/python}

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}
CLEAN_ANN_ROOT=${CLEAN_ANN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_eta_candidates}

mkdir -p "$OUT_ROOT"

"$PYTHON" "$ROOT/scripts/build_ccs_patch_candidates.py" \
  --asset-csv "$ASSET_CSV" \
  --clean-ann-root "$CLEAN_ANN_ROOT" \
  --tokens "$TOKENS_FILE" \
  --out-csv "$OUT_ROOT/patch_eta_top20_candidates.csv" \
  --all-candidates-out "$OUT_ROOT/patch_eta_all_candidates.csv" \
  --sample-interval 0.5 \
  --total-locs 400 \
  --step-per-loc 20 \
  --samples-per-loc 2 \
  --sample-range 1.0 \
  --heading-jitter-deg 30 \
  --patch-width 3 \
  --patch-height 2 \
  --max-beam-angle-deg 20
