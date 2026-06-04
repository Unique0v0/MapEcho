#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/prepare_phase1_5_visual_audit.py \
  --geometry-v2-table /data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/geometry_quality_v2_table.csv \
  --top-bottom-csv /data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_top_bottom_failure_cases.csv \
  --relaxed-tokens /data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_tokens.txt \
  --relaxed-assets /data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_assets.csv \
  --run-root /data/dj/MapEcho/artifacts/phase1_2_vpa015_expanded_ablation_power6000 \
  --out-dir /data/dj/MapEcho/artifacts/phase1_5_visual_audit \
  --freeze-dir /data/dj/MapEcho/artifacts/phase1_5_controlled_experiment
