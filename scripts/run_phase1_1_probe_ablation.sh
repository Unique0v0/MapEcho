#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_tokens.txt}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_1_probe_ablation}
STREAM_ANN=${STREAM_ANN:-/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv}
CONFIG=${CONFIG:-/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py}
CHECKPOINT=${CHECKPOINT:-/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth}
WARMUP=${WARMUP:-10}
RECOVERY=${RECOVERY:-9}
ATTACK_POWER=${ATTACK_POWER:-3000.0}

TOKENS_FILE="${TOKENS_FILE}" \
OUT_ROOT="${OUT_ROOT}" \
STREAM_ANN="${STREAM_ANN}" \
ASSET_CSV="${ASSET_CSV}" \
CONFIG="${CONFIG}" \
CHECKPOINT="${CHECKPOINT}" \
WARMUP="${WARMUP}" \
RECOVERY="${RECOVERY}" \
ATTACK_POWER="${ATTACK_POWER}" \
bash /home/dj/MapEcho/scripts/run_phase1_0_overlap_mini_ablation.sh
