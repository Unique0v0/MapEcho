#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}
NUM_TOKENS=${NUM_TOKENS:-5}
OUT_FILE=${OUT_FILE:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_tokens.txt}

mkdir -p "$(dirname "$OUT_FILE")"
head -n "$NUM_TOKENS" "$TOKENS_FILE" > "$OUT_FILE"

echo "[MapEcho] wrote $(grep -c . "$OUT_FILE" || true) tokens to $OUT_FILE"
