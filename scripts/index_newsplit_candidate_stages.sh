#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/index_newsplit_candidate_stages.py \
  --newsplit-val-ann /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl \
  --newsplit-train-ann /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_train_newsplit.pkl \
  --ccs-dataset-root /home/dj/physical-online-map-attack/dataset \
  --out-dir /data/dj/MapEcho/artifacts/newsplit_candidates \
  --windows 10:19,10:9,5:9
