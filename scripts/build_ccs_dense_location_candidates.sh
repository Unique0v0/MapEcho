#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_high_quality_pool_assets.csv}
TOKENS=${TOKENS:-}
OUT_DIR=${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates}
STREAM_ANN=${STREAM_ANN:-/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}

cmd=(
  /home/dj/.conda/envs/maptr/bin/python
  scripts/build_ccs_dense_location_candidates.py
  --asset-csv "${ASSET_CSV}"
  --stream-ann "${STREAM_ANN}"
  --out-dir "${OUT_DIR}"
  --total-locs 400
  --sample-interval 0.5
  --locs-height-num 4
  --samples-per-loc 2
  --sample-range 1.0
  --max-beam-angle-deg 40.0
)

if [[ -n "${TOKENS}" ]]; then
  cmd+=(--tokens "${TOKENS}")
fi

"${cmd[@]}"
