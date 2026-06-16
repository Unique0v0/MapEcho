#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE="${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}"
DENSE_CANDIDATES_CSV="${DENSE_CANDIDATES_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_top_locations.csv}"
OUT_ROOT="${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa}"
MAX_CANDIDATES="${MAX_CANDIDATES:-400}"
WARMUP="${WARMUP:-10}"
RECOVERY="${RECOVERY:-0}"
POWER="${POWER:-3000.0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORMAT_RESULTS="${FORMAT_RESULTS:-0}"
SAVE_DEBUG="${SAVE_DEBUG:-0}"

cd /home/dj/MapEcho

ATTACK_OBJECTIVE=rsa \
TOKENS_FILE="${TOKENS_FILE}" \
DENSE_CANDIDATES_CSV="${DENSE_CANDIDATES_CSV}" \
OUT_ROOT="${OUT_ROOT}" \
MAX_CANDIDATES="${MAX_CANDIDATES}" \
WARMUP="${WARMUP}" \
RECOVERY="${RECOVERY}" \
POWER="${POWER}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
FORMAT_RESULTS="${FORMAT_RESULTS}" \
SAVE_DEBUG="${SAVE_DEBUG}" \
  bash scripts/run_phase1_8b_selected114_location_scoring.sh
