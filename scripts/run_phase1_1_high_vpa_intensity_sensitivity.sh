#!/usr/bin/env bash
set -euo pipefail

POWERS=${POWERS:-"6000 9000"}
TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset/phase1_1_high_vpa_tokens.txt}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset/phase1_1_high_vpa_assets.csv}
BASE_OUT_ROOT=${BASE_OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_intensity}

cd /home/dj/MapEcho

for POWER in ${POWERS}; do
  SAFE_POWER=${POWER//./p}
  echo "[MapEcho] running high-VPA intensity power=${POWER}"
  TOKENS_FILE="${TOKENS_FILE}" \
  ASSET_CSV="${ASSET_CSV}" \
  OUT_ROOT="${BASE_OUT_ROOT}/power_${SAFE_POWER}" \
  ATTACK_POWER="${POWER}" \
  WARMUP=10 \
  RECOVERY=9 \
  bash scripts/run_phase1_1_probe_ablation.sh
done
