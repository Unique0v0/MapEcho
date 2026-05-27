#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/sanity_target_boundary_vpa.py \
  --stream-ann /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_asymmetric_dist_eta_like_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/vpa_sanity \
  --attack-objective eta \
  --source-frame lidar \
  --max-samples 100000 \
  --coverage-threshold 0.05 \
  --render-overlays \
  --render-max-samples 12
