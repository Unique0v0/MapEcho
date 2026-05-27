#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

export MPLCONFIGDIR=/tmp/mapecho_matplotlib

/home/dj/.conda/envs/maptr4090/bin/python scripts/summarize_phase1_0_map_level.py \
  --ann-file /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --hook-root /data/dj/MapEcho/artifacts/streammapnet_hook_sanity \
  --out-dir /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_map_level \
  --offsets 0,1,2
