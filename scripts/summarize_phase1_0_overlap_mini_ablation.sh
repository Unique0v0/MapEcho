#!/usr/bin/env bash
set -euo pipefail

TOKENS_FILE=${TOKENS_FILE:-/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_tokens.txt}
OUT_ROOT=${OUT_ROOT:-/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv}

cd /home/dj/MapEcho

export MPLCONFIGDIR=/tmp/mapecho_matplotlib

SUMMARY_DIR="${OUT_ROOT}/summary"
COMPLETED_TOKENS_FILE="${SUMMARY_DIR}/completed_tokens.txt"
MISSING_TOKENS_FILE="${SUMMARY_DIR}/missing_tokens.txt"
mkdir -p "${SUMMARY_DIR}"
: > "${COMPLETED_TOKENS_FILE}"
: > "${MISSING_TOKENS_FILE}"

while IFS= read -r TOKEN; do
  [[ -z "${TOKEN}" ]] && continue
  ROOT="${OUT_ROOT}/${TOKEN}"
  if [[ -f "${ROOT}/anns/attack_sequence_ann.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_clean_keep/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_all/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_query/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_reset_sanity/reset_bev/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_keep/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_all/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_query/outputs.pkl" ]] \
    && [[ -f "${ROOT}/phase1_0_attack_reset_ablation/attack_reset_bev/outputs.pkl" ]]; then
    echo "${TOKEN}" >> "${COMPLETED_TOKENS_FILE}"
  else
    echo "${TOKEN}" >> "${MISSING_TOKENS_FILE}"
  fi
done < "${TOKENS_FILE}"

COMPLETED_COUNT=$(grep -c . "${COMPLETED_TOKENS_FILE}" || true)
MISSING_COUNT=$(grep -c . "${MISSING_TOKENS_FILE}" || true)
echo "[MapEcho] completed tokens: ${COMPLETED_COUNT}"
echo "[MapEcho] missing/incomplete tokens: ${MISSING_COUNT}"
if [[ "${COMPLETED_COUNT}" == "0" ]]; then
  echo "[MapEcho] no completed tokens to summarize; see ${MISSING_TOKENS_FILE}" >&2
  exit 1
fi

while IFS= read -r TOKEN; do
  [[ -z "${TOKEN}" ]] && continue
  ROOT="${OUT_ROOT}/${TOKEN}"

  echo "[MapEcho] summarizing internal reset ablation for ${TOKEN}"
  /home/dj/.conda/envs/maptr4090/bin/python scripts/summarize_phase1_0_attack_reset_ablation.py \
    --clean-root "${ROOT}/phase1_0_clean_keep" \
    --clean-reset-root "${ROOT}/phase1_0_reset_sanity" \
    --ablation-root "${ROOT}/phase1_0_attack_reset_ablation" \
    --ann "${ROOT}/anns/attack_sequence_ann.pkl" \
    --offsets 0,1,2,3

  echo "[MapEcho] summarizing map-level boundary residue for ${TOKEN}"
  /home/dj/.conda/envs/maptr4090/bin/python scripts/summarize_phase1_0_map_level.py \
    --ann-file "${ROOT}/anns/clean_sequence_ann.pkl" \
    --asset-csv "${ASSET_CSV}" \
    --hook-root "${ROOT}" \
    --out-dir "${ROOT}/phase1_0_map_level" \
    --offsets 0,1,2
done < "${COMPLETED_TOKENS_FILE}"

/home/dj/.conda/envs/maptr4090/bin/python scripts/aggregate_phase1_0_overlap_mini_ablation.py \
  --tokens-file "${COMPLETED_TOKENS_FILE}" \
  --out-root "${OUT_ROOT}" \
  --out-dir "${SUMMARY_DIR}"
