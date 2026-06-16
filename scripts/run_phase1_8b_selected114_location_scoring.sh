#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

: "${TOKENS_FILE:=/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt}"
: "${DENSE_CANDIDATES_CSV:=/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_top_locations.csv}"
: "${OUT_ROOT:=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114}"
: "${MAX_CANDIDATES:=400}"
: "${WARMUP:=10}"
: "${RECOVERY:=0}"
: "${POWER:=3000.0}"
: "${ATTACK_OBJECTIVE:=eta}"
: "${SKIP_COMPLETED:=1}"
: "${FORMAT_RESULTS:=0}"
: "${SAVE_DEBUG:=0}"

while IFS= read -r TARGET_TOKEN; do
  [[ -z "${TARGET_TOKEN}" ]] && continue
  echo "[MapEcho] scoring CCS-style top-${MAX_CANDIDATES} locations for ${TARGET_TOKEN}"
  TARGET_TOKEN="${TARGET_TOKEN}" \
  DENSE_CANDIDATES_CSV="${DENSE_CANDIDATES_CSV}" \
  OUT_ROOT="${OUT_ROOT}" \
  MAX_CANDIDATES="${MAX_CANDIDATES}" \
  WARMUP="${WARMUP}" \
  RECOVERY="${RECOVERY}" \
  POWER="${POWER}" \
  ATTACK_OBJECTIVE="${ATTACK_OBJECTIVE}" \
  SKIP_COMPLETED="${SKIP_COMPLETED}" \
  FORMAT_RESULTS="${FORMAT_RESULTS}" \
  SAVE_DEBUG="${SAVE_DEBUG}" \
    bash scripts/run_ccs_location_scoring_fast.sh
done < "${TOKENS_FILE}"
