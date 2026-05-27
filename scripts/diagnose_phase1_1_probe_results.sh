#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/diagnose_phase1_1_probe_results.py \
  --summary-dir /data/dj/MapEcho/artifacts/phase1_1_probe_ablation/summary \
  --run-root /data/dj/MapEcho/artifacts/phase1_1_probe_ablation \
  --asset-csv /data/dj/MapEcho/artifacts/phase1_1_asymmetric_dist/phase1_1_probe_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_1_probe_ablation/diagnostics
