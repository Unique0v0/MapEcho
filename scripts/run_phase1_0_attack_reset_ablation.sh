#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr/bin/python scripts/build_attack_at_t_sequence_ann.py \
  --clean-ann /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --out-ann /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl \
  --out-dir /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation/attack_assets \
  --attack-objective eta \
  --source-frame lidar

cd /home/dj/MapEcho/src/StreamMapNet

export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:/home/dj/MapEcho:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

CONFIG=/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py
CHECKPOINT=/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth
ANN=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl
OUT_ROOT=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation

/home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --ann-file "${ANN}" \
  --out-dir "${OUT_ROOT}/attack_keep" \
  --condition attack_keep \
  --reset-mode none

for MODE in all query bev; do
  /home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --ann-file "${ANN}" \
    --out-dir "${OUT_ROOT}/attack_reset_${MODE}" \
    --condition "attack_reset_${MODE}" \
    --reset-mode "${MODE}" \
    --reset-after-offset 0
done
