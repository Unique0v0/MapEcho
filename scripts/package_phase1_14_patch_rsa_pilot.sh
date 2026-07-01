#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_tokens.txt}
OPT_ROOT=${OPT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_optimizer}
REPLAY_ROOT=${REPLAY_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_replay}
OUT_DIR=${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_summary}

cd /home/dj/MapEcho
/home/dj/.conda/envs/maptr4090/bin/python scripts/package_phase1_14_patch_rsa_pilot.py \
  --tokens-file "$TOKENS_FILE" \
  --optimizer-root "$OPT_ROOT" \
  --replay-root "$REPLAY_ROOT" \
  --out-dir "$OUT_DIR"
