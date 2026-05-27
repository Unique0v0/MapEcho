#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_tokens.txt}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation}
STREAM_ANN=${STREAM_ANN:-/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv}
CONFIG=${CONFIG:-/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py}
CHECKPOINT=${CHECKPOINT:-/home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
WARMUP=${WARMUP:-10}
RECOVERY=${RECOVERY:-19}
ATTACK_POWER=${ATTACK_POWER:-3000.0}

cd /home/dj/MapEcho

mkdir -p "${OUT_ROOT}"

while IFS= read -r TOKEN; do
  [[ -z "${TOKEN}" ]] && continue
  ROOT="${OUT_ROOT}/${TOKEN}"
  CLEAN_ANN="${ROOT}/anns/clean_sequence_ann.pkl"
  ATTACK_ANN="${ROOT}/anns/attack_sequence_ann.pkl"

  if [[ "${SKIP_COMPLETED}" == "1" ]] \
    && [[ -f "${ROOT}/phase1_0_clean_keep/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_all/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_query/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_bev/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_keep/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_all/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_query/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_bev/outputs.pkl" ]]; then
    echo "[MapEcho] skipping completed token ${TOKEN}"
    continue
  fi

  echo "[MapEcho] building clean sequence ann for ${TOKEN}"
  /home/dj/.conda/envs/maptr/bin/python scripts/build_sequence_ann_subset.py \
    --stream-ann "${STREAM_ANN}" \
    --tokens "${TOKENS_FILE}" \
    --target-token "${TOKEN}" \
    --out "${CLEAN_ANN}" \
    --summary-out "${ROOT}/anns/clean_sequence_ann_summary.json" \
    --warmup "${WARMUP}" \
    --recovery "${RECOVERY}"

  echo "[MapEcho] building attack-at-t ann for ${TOKEN}"
  /home/dj/.conda/envs/maptr/bin/python scripts/build_attack_at_t_sequence_ann.py \
    --clean-ann "${CLEAN_ANN}" \
    --asset-csv "${ASSET_CSV}" \
    --out-ann "${ATTACK_ANN}" \
    --out-dir "${ROOT}/attack_assets" \
    --attack-objective eta \
    --source-frame lidar \
    --power "${ATTACK_POWER}"

  cd /home/dj/MapEcho/src/StreamMapNet
  export MPLCONFIGDIR=/tmp/mapecho_matplotlib
  export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:/home/dj/MapEcho:${PYTHONPATH:-}
  export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}

  echo "[MapEcho] running clean_keep for ${TOKEN}"
  /home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --ann-file "${CLEAN_ANN}" \
    --out-dir "${ROOT}/phase1_0_clean_keep" \
    --condition clean_keep \
    --reset-mode none

  for MODE in all query bev; do
    echo "[MapEcho] running clean_reset_${MODE} for ${TOKEN}"
    /home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --ann-file "${CLEAN_ANN}" \
      --out-dir "${ROOT}/phase1_0_reset_sanity/reset_${MODE}" \
      --condition "clean_reset_${MODE}" \
      --reset-mode "${MODE}" \
      --reset-after-offset 0
  done

  echo "[MapEcho] running attack_keep for ${TOKEN}"
  /home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --ann-file "${ATTACK_ANN}" \
    --out-dir "${ROOT}/phase1_0_attack_reset_ablation/attack_keep" \
    --condition attack_keep \
    --reset-mode none

  for MODE in all query bev; do
    echo "[MapEcho] running attack_reset_${MODE} for ${TOKEN}"
    /home/dj/.conda/envs/maptr4090/bin/python /home/dj/MapEcho/scripts/run_streammapnet_sequence_condition.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --ann-file "${ATTACK_ANN}" \
      --out-dir "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_${MODE}" \
      --condition "attack_reset_${MODE}" \
      --reset-mode "${MODE}" \
      --reset-after-offset 0
  done

  cd /home/dj/MapEcho
done < "${TOKENS_FILE}"
