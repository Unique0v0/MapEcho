#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

TARGET_TOKEN=${TARGET_TOKEN:?Set TARGET_TOKEN to one sample token}
DENSE_CANDIDATES_CSV=${DENSE_CANDIDATES_CSV:-/data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates/ccs_dense_top_locations.csv}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/model_scoring_fast_top20}
STREAM_ANN=${STREAM_ANN:-/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}
CONFIG=${CONFIG:-/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py}
CHECKPOINT=${CHECKPOINT:-/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth}
MAX_CANDIDATES=${MAX_CANDIDATES:-20}
WARMUP=${WARMUP:-10}
RECOVERY=${RECOVERY:-0}
POWER=${POWER:-3000.0}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}
FORMAT_RESULTS=${FORMAT_RESULTS:-0}
SAVE_DEBUG=${SAVE_DEBUG:-0}

ARGS=()
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  ARGS+=(--skip-completed)
fi
if [[ "${FORMAT_RESULTS}" == "1" ]]; then
  ARGS+=(--format-results)
fi
if [[ "${SAVE_DEBUG}" == "1" ]]; then
  ARGS+=(--save-debug)
fi

export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:/home/dj/MapEcho:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

/home/dj/.conda/envs/maptr4090/bin/python scripts/run_ccs_location_scoring_fast.py \
  --target-token "${TARGET_TOKEN}" \
  --dense-candidates-csv "${DENSE_CANDIDATES_CSV}" \
  --out-root "${OUT_ROOT}" \
  --stream-ann "${STREAM_ANN}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --warmup "${WARMUP}" \
  --recovery "${RECOVERY}" \
  --power "${POWER}" \
  "${ARGS[@]}"
