#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/select_phase1_2_high_vpa_assets.py \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv \
  --tag-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/asymmetric_dist_boundary_tags.csv \
  --vpa-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa \
  --high-vpa 0.25 \
  --medium-vpa 0.20 \
  --max-frames 60
