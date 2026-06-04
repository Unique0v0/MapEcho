#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/build_phase1_3_sample_gates.py \
  --summary-csv /data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000/summary/phase1_1_map_matched_deltas_enriched.csv \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_3_sample_gates \
  --source-stage ccs_candidate_expanded_vpa015 \
  --warmup 10 \
  --recovery 9 \
  --attack-power 6000
