#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr/bin/python scripts/build_phase1_6_ccs_style_pool.py \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv \
  --vpa-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool \
  --min-vpa 0.15 \
  --max-per-scene 2
