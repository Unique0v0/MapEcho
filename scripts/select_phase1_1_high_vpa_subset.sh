#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/select_phase1_1_high_vpa_subset.py \
  --probe-assets /data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_1_high_vpa_subset \
  --min-vpa 0.265 \
  --max-frames 16
