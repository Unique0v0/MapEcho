#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

/home/dj/.conda/envs/maptr4090/bin/python scripts/diagnose_phase1_3_gate_dropoff.py \
  --gate-table /data/dj/MapEcho/artifacts/phase1_3_sample_gates/phase1_3_gate_table.csv \
  --out-dir /data/dj/MapEcho/artifacts/phase1_3_sample_gates/dropoff
