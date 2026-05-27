#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/diagnose_phase1_2_asset_quality.py \
  --out-dir /data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics
