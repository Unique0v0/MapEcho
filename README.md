# MapEcho

MapEcho is the working directory for the temporal-residue study built on:

- CCS'25 physical online map attack code: `/home/dj/physical-online-map-attack`
- Phase 0.5 modified StreamMapNet source: `/home/dj/StreamMapNet`
- StreamMapNet oldsplit checkpoint: `ckpts/nusc_baseline_480_60x30_30e.pth`
- Shared read-only nuScenes: `/data/yuy/dataset/nuScenes`
- Private writable experiment storage: `/data/dj/MapEcho`

## Layout

```text
src/StreamMapNet/              # editable copy of the Phase 0.5 StreamMapNet source
external/                      # symlinks to original external repositories
datasets/nuScenes/             # local dataset view: annotations + symlinks to shared nuScenes
datasets/ccs25_seeds/          # copied CCS'25 asymmetric seed metadata
ckpts/                         # StreamMapNet checkpoint
scripts/                       # project entrypoints
configs/paths.env              # canonical path configuration
data_private -> /data/dj/MapEcho
```

The public nuScenes directory is only referenced through symlinks and should not
be modified. Hook dumps, experiment outputs, and generated temporal-evaluation
metadata should go under `data_private/`.

## Quick Checks

```bash
bash scripts/verify_project_layout.sh
```

Load StreamMapNet paths:

```bash
source scripts/env_streammapnet.sh
```

Run the oldsplit StreamMapNet validation wrapper:

```bash
PYTHON_BIN=/home/dj/.conda/envs/maptr/bin/python GPUS=1 bash scripts/run_streammapnet_eval.sh
```

This validation can be slow; use it as a sanity target before Phase 1 mini-probe
runs.

The wrapper uses `plugin/configs/mapecho_nusc_baseline_480_60x30_30e_eval.py`,
which inherits the official oldsplit config but disables the redundant
open-mmlab backbone preinitialization before loading the full checkpoint.

## Notes

- `src/StreamMapNet` was copied from `/home/dj/StreamMapNet` with the Phase 0.5
  instrumentation changes and without its `.git` directory or dataset cache.
- `scripts/sync_streammapnet_from_phase05_source.sh` refreshes the copied source
  from `/home/dj/StreamMapNet` and restores the local dataset link.
- The CCS'25 attack repository is large, so this project references it by path
  instead of copying the full 11GB tree.
