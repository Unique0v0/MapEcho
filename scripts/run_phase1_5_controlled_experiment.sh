#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment/phase1_5_high_quality_relaxed_v2_tokens.txt \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment/high_quality_relaxed_v2_ablation_power6000 \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment/phase1_5_high_quality_relaxed_v2_assets.csv \
WARMUP=10 \
RECOVERY=9 \
ATTACK_POWER=6000.0 \
bash scripts/run_phase1_1_probe_ablation.sh
