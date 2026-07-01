#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dj/MapEcho
CONFIG=${CONFIG:-/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py}
CHECKPOINT=${CHECKPOINT:-/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv}

TOKEN=${TOKEN:-5d4b194ee07d418b9a60704991e647eb}
PATCH_ROOT=${PATCH_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_full_one_token/${TOKEN}}
CLEAN_ANN=${CLEAN_ANN:-/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/${TOKEN}/anns/clean_sequence_ann.pkl}
PATCH_ANN=${PATCH_ANN:-${PATCH_ROOT}/anns/patch_rsa_sequence_ann.pkl}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_full_one_token_replay}
CONDITIONS=${CONDITIONS:-clean_keep,clean_reset_all,clean_reset_query,clean_reset_bev,attack_keep,attack_reset_all,attack_reset_query,attack_reset_bev}

cd "$ROOT"
export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:/home/dj/MapEcho:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

TOKEN_ROOT="${OUT_ROOT}/${TOKEN}"
mkdir -p "${TOKEN_ROOT}/anns"
cp "$CLEAN_ANN" "${TOKEN_ROOT}/anns/clean_sequence_ann.pkl"
cp "$PATCH_ANN" "${TOKEN_ROOT}/anns/attack_sequence_ann.pkl"

cd /home/dj/MapEcho/src/StreamMapNet
/home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_multi_condition.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --clean-ann "${TOKEN_ROOT}/anns/clean_sequence_ann.pkl" \
  --attack-ann "${TOKEN_ROOT}/anns/attack_sequence_ann.pkl" \
  --token-root "$TOKEN_ROOT" \
  --conditions "$CONDITIONS" \
  --reset-after-offset 0 \
  --skip-completed

cd "$ROOT"
/home/dj/.conda/envs/maptr4090/bin/python scripts/summarize_phase1_0_attack_reset_ablation.py \
  --clean-root "${TOKEN_ROOT}/phase1_0_clean_keep" \
  --clean-reset-root "${TOKEN_ROOT}/phase1_0_reset_sanity" \
  --ablation-root "${TOKEN_ROOT}/phase1_0_attack_reset_ablation" \
  --ann "${TOKEN_ROOT}/anns/attack_sequence_ann.pkl" \
  --offsets 0,1,2,3

/home/dj/.conda/envs/maptr4090/bin/python scripts/summarize_phase1_0_map_level.py \
  --ann-file "${TOKEN_ROOT}/anns/clean_sequence_ann.pkl" \
  --asset-csv "$ASSET_CSV" \
  --hook-root "$TOKEN_ROOT" \
  --out-dir "${TOKEN_ROOT}/phase1_0_map_level" \
  --offsets 0,1,2

echo "[MapEcho] patch_rsa replay complete: ${TOKEN_ROOT}"
