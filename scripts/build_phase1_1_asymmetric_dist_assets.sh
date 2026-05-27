#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/build_phase1_1_asymmetric_dist_assets.py \
  --membership-csv /data/dj/MapEcho/artifacts/newsplit_candidates/ccs_stage_newsplit_membership.csv \
  --newsplit-val-ann /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl \
  --source-stage ccs_asymmetric_dist \
  --eligibility-key eligible_W10_L9 \
  --out-dir /data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist
