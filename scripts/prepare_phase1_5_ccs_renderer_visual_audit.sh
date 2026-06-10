#!/usr/bin/env bash
set -euo pipefail

cd /home/dj/MapEcho

MANIFEST=${MANIFEST:-/data/dj/MapEcho/artifacts/phase1_5_visual_audit/visual_audit_manifest.csv}
OUT_DIR=${OUT_DIR:-/data/dj/MapEcho/artifacts/phase1_5_ccs_renderer_visual_audit}
STREAM_ANN=${STREAM_ANN:-/home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl}
ASSET_CSV=${ASSET_CSV:-/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv}
WARMUP=${WARMUP:-10}
RECOVERY=${RECOVERY:-9}
ATTACK_POWER=${ATTACK_POWER:-3000.0}

mkdir -p "${OUT_DIR}"

/home/dj/.conda/envs/maptr/bin/python - <<PY
import csv
from pathlib import Path

manifest = Path("${MANIFEST}")
out = Path("${OUT_DIR}") / "phase1_5_ccs_renderer_audit_tokens.txt"
rows = list(csv.DictReader(manifest.open()))
tokens = []
seen = set()
for row in rows:
    token = row["sample_token"]
    if token not in seen:
        seen.add(token)
        tokens.append(token)
out.write_text("\\n".join(tokens) + "\\n")
print(f"[MapEcho] wrote {len(tokens)} audit tokens to {out}")
PY

TOKENS_FILE="${OUT_DIR}/phase1_5_ccs_renderer_audit_tokens.txt"

while IFS= read -r TOKEN; do
  [[ -z "${TOKEN}" ]] && continue
  ROOT="${OUT_DIR}/${TOKEN}"
  CLEAN_ANN="${ROOT}/anns/clean_sequence_ann.pkl"
  ATTACK_ANN="${ROOT}/anns/attack_sequence_ann.pkl"

  if [[ -f "${ROOT}/attack_assets/attack_at_t_ann_summary.json" ]]; then
    echo "[MapEcho] skipping existing CCS renderer audit token ${TOKEN}"
    continue
  fi

  echo "[MapEcho] building clean sequence ann for renderer audit ${TOKEN}"
  /home/dj/.conda/envs/maptr/bin/python scripts/build_sequence_ann_subset.py \
    --stream-ann "${STREAM_ANN}" \
    --tokens "${TOKENS_FILE}" \
    --target-token "${TOKEN}" \
    --out "${CLEAN_ANN}" \
    --summary-out "${ROOT}/anns/clean_sequence_ann_summary.json" \
    --warmup "${WARMUP}" \
    --recovery "${RECOVERY}"

  echo "[MapEcho] building CCS-style six-camera attack assets for ${TOKEN}"
  /home/dj/.conda/envs/maptr/bin/python scripts/build_attack_at_t_sequence_ann.py \
    --clean-ann "${CLEAN_ANN}" \
    --asset-csv "${ASSET_CSV}" \
    --out-ann "${ATTACK_ANN}" \
    --out-dir "${ROOT}/attack_assets" \
    --attack-objective eta \
    --source-frame lidar \
    --power "${ATTACK_POWER}" \
    --renderer ccs \
    --camera-mode all
done < "${TOKENS_FILE}"

echo "[MapEcho] CCS renderer visual audit assets written to ${OUT_DIR}"

