#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr/bin/python scripts/build_attack_at_t_sequence_ann.py \
  --clean-ann /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --out-ann /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl \
  --out-dir /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep \
  --attack-objective eta \
  --source-frame lidar

cd /home/dj/MapEcho/src/StreamMapNet

export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

/home/dj/.conda/envs/maptr4090/bin/python tools/test.py \
  plugin/configs/mapecho_nusc_newsplit_phase1_0_attack_keep_debug.py \
  /home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth \
  --format-only \
  --work-dir /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep \
  --eval-options prefix=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep
