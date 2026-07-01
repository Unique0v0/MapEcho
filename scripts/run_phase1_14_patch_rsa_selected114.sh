#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}
OPT_ROOT=${OPT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_selected114_optimizer}
REPLAY_ROOT=${REPLAY_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_selected114_replay}
MAX_LOCATIONS=${MAX_LOCATIONS:-20}
PATCH_STEPS=${PATCH_STEPS:-20}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

cd /home/dj/MapEcho
TOKENS_FILE="$TOKENS_FILE" \
OPT_ROOT="$OPT_ROOT" \
REPLAY_ROOT="$REPLAY_ROOT" \
MAX_LOCATIONS="$MAX_LOCATIONS" \
PATCH_STEPS="$PATCH_STEPS" \
SKIP_COMPLETED="$SKIP_COMPLETED" \
bash scripts/run_phase1_14_patch_rsa_pilot.sh
