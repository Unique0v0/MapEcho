#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/build_phase1_4_geometry_gate_v2.py \
  --gate-table /data/dj/MapEcho/artifacts/phase1_3_sample_gates/phase1_3_gate_table.csv \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2
