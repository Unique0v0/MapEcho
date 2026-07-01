#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}
OPT_ROOT=${OPT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_selected114_optimizer}
REPLAY_ROOT=${REPLAY_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_selected114_replay}
OUT_DIR=${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_selected114_summary}

cd /home/dj/MapEcho
TOKENS_FILE="$TOKENS_FILE" \
OPT_ROOT="$OPT_ROOT" \
REPLAY_ROOT="$REPLAY_ROOT" \
OUT_DIR="$OUT_DIR" \
bash scripts/package_phase1_14_patch_rsa_pilot.sh
