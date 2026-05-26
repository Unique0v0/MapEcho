#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/paths.env"

rsync -a --delete \
  --exclude='.git/' \
  --exclude='/datasets/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  "${STREAMMAPNET_PHASE05_SOURCE}/" \
  "${STREAMMAPNET_ROOT}/"

ln -sfn ../../datasets "${STREAMMAPNET_ROOT}/datasets"

echo "Synced StreamMapNet Phase0.5 source into ${STREAMMAPNET_ROOT}"
