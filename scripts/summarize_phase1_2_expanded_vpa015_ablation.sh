#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_tokens.txt \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000 \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv \
bash scripts/summarize_phase1_1_probe_ablation.sh
