#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_tokens.txt}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_1_probe_ablation}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv}

TOKENS_FILE="${TOKENS_FILE}" \
OUT_ROOT="${OUT_ROOT}" \
ASSET_CSV="${ASSET_CSV}" \
bash /home/dj/MapEcho/scripts/summarize_phase1_0_overlap_mini_ablation.sh

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/analyze_phase1_1_probe_results.py \
  --summary-dir "${OUT_ROOT}/summary" \
  --asset-csv "${ASSET_CSV}" \
  --out-dir "${OUT_ROOT}/summary"
