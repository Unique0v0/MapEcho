#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

: "${PYTHON_BIN:=/home/dj/.conda/envs/maptr/bin/python}"
: "${ASSET_CSV:=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv}"
: "${VPA_CSV:=/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv}"
: "${OUT_DIR:=/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild}"
: "${MIN_VPA:=0.05}"
: "${PREFERRED_VPA:=0.15}"
: "${MAX_PER_SCENE:=5}"
: "${TARGET_MAX:=80}"

"${PYTHON_BIN}" scripts/build_phase1_8_pool.py \
  --asset-csv "${ASSET_CSV}" \
  --vpa-csv "${VPA_CSV}" \
  --out-dir "${OUT_DIR}" \
  --min-vpa "${MIN_VPA}" \
  --preferred-vpa "${PREFERRED_VPA}" \
  --max-per-scene "${MAX_PER_SCENE}" \
  --target-max "${TARGET_MAX}"
