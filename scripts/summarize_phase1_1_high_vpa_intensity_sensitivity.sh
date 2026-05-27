#!/usr/bin/env bash
set -euo pipefail

POWERS=${POWERS:-"6000 9000"}
TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset/phase1_1_high_vpa_tokens.txt}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset/phase1_1_high_vpa_assets.csv}
BASE_OUT_ROOT=${BASE_OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_1_high_vpa_intensity}

cd /home/dj/MapEcho

for POWER in ${POWERS}; do
  SAFE_POWER=${POWER//./p}
  ROOT="${BASE_OUT_ROOT}/power_${SAFE_POWER}"
  echo "[MapEcho] summarizing high-VPA intensity power=${POWER}"
  TOKENS_FILE="${TOKENS_FILE}" \
  ASSET_CSV="${ASSET_CSV}" \
  OUT_ROOT="${ROOT}" \
  bash scripts/summarize_phase1_1_probe_ablation.sh
done

/home/dj/.conda/envs/maptr4090/bin/python scripts/compare_phase1_1_intensity_summaries.py \
  --baseline-summary /data/dj/MapEcho/artifacts/phase1_1_probe_ablation/summary/phase1_1_map_matched_deltas_enriched.csv \
  --baseline-label power_3000_full_high_vpa_subset \
  --tokens-file "${TOKENS_FILE}" \
  --intensity-root "${BASE_OUT_ROOT}" \
  --powers "${POWERS}" \
  --out-dir "${BASE_OUT_ROOT}/comparison"
