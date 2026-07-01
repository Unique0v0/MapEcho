#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dj/MapEcho
PYTHON=${PYTHON:-/home/dj/.conda/envs/maptr4090/bin/python}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/home/dj/.conda/envs/maptr4090/lib}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mapecho_matplotlib}

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt}
TOKEN=${TOKEN:-$(head -n 1 "$TOKENS_FILE")}

PATCH_CANDIDATES_CSV=${PATCH_CANDIDATES_CSV:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_eta_candidates/patch_eta_top20_candidates.csv}
CLEAN_ANN=${CLEAN_ANN:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/${TOKEN}/anns/clean_sequence_ann.pkl}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_smoke}

MAX_LOCATIONS=${MAX_LOCATIONS:-2}
PATCH_STEPS=${PATCH_STEPS:-2}

if [[ ! -f "$PATCH_CANDIDATES_CSV" ]]; then
  echo "[MapEcho] patch candidates missing, building them first: $PATCH_CANDIDATES_CSV"
  bash "$ROOT/scripts/build_phase1_14_patch_eta_candidates.sh"
fi

cd "$ROOT"
"$PYTHON" "$ROOT/scripts/run_ccs_patch_scoring_streammapnet.py" \
  --target-token "$TOKEN" \
  --clean-ann "$CLEAN_ANN" \
  --patch-candidates-csv "$PATCH_CANDIDATES_CSV" \
  --out-root "$OUT_ROOT" \
  --objective rsa \
  --max-locations "$MAX_LOCATIONS" \
  --patch-steps "$PATCH_STEPS"
