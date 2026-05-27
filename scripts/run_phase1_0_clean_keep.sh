#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho/src/StreamMapNet

export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:/home/dj/MapEcho:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

CONFIG=/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_phase1_0_clean_keep_debug.py
CHECKPOINT=/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth
ANN=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl
OUT_DIR=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep

/home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --ann-file "${ANN}" \
  --out-dir "${OUT_DIR}" \
  --condition clean_keep \
  --reset-mode none
