#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    printf "[ok]   %-34s %s\n" "${label}" "${path}"
  else
    printf "[miss] %-34s %s\n" "${label}" "${path}"
    return 1
  fi
}

status=0

check_path "${STREAMMAPNET_ROOT}/plugin/configs/nusc_baseline_480_60x30_30e.py" "StreamMapNet oldsplit config" || status=1
check_path "${STREAMMAPNET_CONFIG}" "MapEcho eval config" || status=1
check_path "${STREAMMAPNET_ROOT}/plugin/models/mapers/StreamMapNet.py" "Phase0.5 StreamMapNet code" || status=1
check_path "${STREAMMAPNET_ROOT}/datasets/nuScenes/nuscenes_map_infos_val.pkl" "StreamMapNet val ann" || status=1
check_path "${STREAMMAPNET_CKPT}" "StreamMapNet checkpoint" || status=1

check_path "${ATTACK_ROOT}/tools/attack.py" "CCS25 attack entry" || status=1
check_path "${CCS25_ASYM_TOKENS}" "CCS25 asymmetric seeds" || status=1

check_path "${NUSCENES_ROOT}/samples/CAM_FRONT" "nuScenes CAM_FRONT" || status=1
check_path "${NUSCENES_ROOT}/maps" "nuScenes maps" || status=1
check_path "${NUSCENES_ROOT}/v1.0-trainval/sample.json" "nuScenes trainval metadata" || status=1

check_path "${MAPECHO_DATA_ROOT}/results" "private results dir" || status=1
check_path "${MAPECHO_DATA_ROOT}/dumps" "private hook dumps dir" || status=1

exit "${status}"
