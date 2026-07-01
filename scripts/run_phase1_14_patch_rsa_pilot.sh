#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dj/MapEcho

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_tokens.txt}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}
PATCH_CANDIDATES_CSV=${PATCH_CANDIDATES_CSV:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_eta_candidates/patch_eta_top20_candidates.csv}
CLEAN_ANN_ROOT=${CLEAN_ANN_ROOT:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check}

OPT_ROOT=${OPT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_optimizer}
REPLAY_ROOT=${REPLAY_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_replay}

MAX_LOCATIONS=${MAX_LOCATIONS:-20}
PATCH_STEPS=${PATCH_STEPS:-20}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

cd "$ROOT"

if [[ ! -f "$TOKENS_FILE" ]]; then
  echo "[MapEcho] token file missing; creating default pilot file: $TOKENS_FILE"
  OUT_FILE="$TOKENS_FILE" bash scripts/select_phase1_14_patch_rsa_pilot_tokens.sh
fi

if [[ ! -f "$PATCH_CANDIDATES_CSV" ]]; then
  echo "[MapEcho] patch candidates missing; building: $PATCH_CANDIDATES_CSV"
  bash scripts/build_phase1_14_patch_eta_candidates.sh
fi

while IFS= read -r TOKEN; do
  [[ -z "$TOKEN" ]] && continue
  echo "[MapEcho] patch_rsa pilot token: $TOKEN"

  OPT_SUMMARY="${OPT_ROOT}/${TOKEN}/patch_scoring_summary.json"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$OPT_SUMMARY" ]]; then
    echo "[MapEcho] skipping completed optimizer: $TOKEN"
  else
    TOKEN="$TOKEN" \
    PATCH_CANDIDATES_CSV="$PATCH_CANDIDATES_CSV" \
    CLEAN_ANN="${CLEAN_ANN_ROOT}/${TOKEN}/anns/clean_sequence_ann.pkl" \
    OUT_ROOT="$OPT_ROOT" \
    MAX_LOCATIONS="$MAX_LOCATIONS" \
    PATCH_STEPS="$PATCH_STEPS" \
    bash scripts/run_phase1_14_patch_rsa_smoke.sh
  fi

  REPLAY_SUMMARY="${REPLAY_ROOT}/${TOKEN}/phase1_0_map_level/phase1_0_single_sequence_map_summary.json"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$REPLAY_SUMMARY" ]]; then
    echo "[MapEcho] skipping completed replay: $TOKEN"
  else
    TOKEN="$TOKEN" \
    PATCH_ROOT="${OPT_ROOT}/${TOKEN}" \
    CLEAN_ANN="${CLEAN_ANN_ROOT}/${TOKEN}/anns/clean_sequence_ann.pkl" \
    OUT_ROOT="$REPLAY_ROOT" \
    ASSET_CSV="$ASSET_CSV" \
    bash scripts/run_phase1_14_patch_rsa_one_token_replay.sh
  fi
done < "$TOKENS_FILE"

echo "[MapEcho] patch_rsa pilot done"
echo "[MapEcho] optimizer root: $OPT_ROOT"
echo "[MapEcho] replay root: $REPLAY_ROOT"
