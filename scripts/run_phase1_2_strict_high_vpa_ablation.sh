#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_tokens.txt \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_2_strict_high_vpa_ablation_power6000 \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_assets.csv \
WARMUP=10 \
RECOVERY=9 \
ATTACK_POWER=6000.0 \
bash scripts/run_phase1_1_probe_ablation.sh
