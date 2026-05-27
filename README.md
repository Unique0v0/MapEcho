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

## Data Construction Artifacts

Current generated metadata lives under `/data/dj/MapEcho/artifacts/`:

```text
seed_matching/ccs25_seed_streammapnet_match.csv
seed_matching/temporal_eligible_metadata_W10_L19.csv
phase1/phase1_probe_selection.csv
ccs25_attack_assets/ccs25_attack_asset_index.csv
ccs25_attack_assets/temporal_eligible_attack_assets_W10_L19.csv
ccs25_attack_assets/phase1_attack_assets.csv
```

Rebuild the CCS'25 attack asset index with:

```bash
/home/dj/.conda/envs/maptr/bin/python scripts/index_ccs25_attack_assets.py \
  --ccs25-root /home/dj/physical-online-map-attack \
  --match-csv /data/dj/MapEcho/artifacts/seed_matching/ccs25_seed_streammapnet_match.csv \
  --temporal-metadata-csv /data/dj/MapEcho/artifacts/seed_matching/temporal_eligible_metadata_W10_L19.csv \
  --phase1-selection-csv /data/dj/MapEcho/artifacts/phase1/phase1_probe_selection.csv \
  --out-dir /data/dj/MapEcho/artifacts/ccs25_attack_assets
```

Run the ETA attack-point coordinate/projection sanity check with:

```bash
/home/dj/.conda/envs/maptr/bin/python scripts/sanity_attack_point_projection.py \
  --stream-ann datasets/nuScenes/nuscenes_map_infos_val.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/rendering_sanity/attack_point_projection \
  --attack-objective eta \
  --source-frame lidar \
  --max-samples 5 \
  --offsets=-2,-1,0,1,2,5 \
  --render-overlays \
  --render-max-samples 1
```

Run target-boundary VPA sanity with:

```bash
/home/dj/.conda/envs/maptr/bin/python scripts/sanity_target_boundary_vpa.py \
  --stream-ann datasets/nuScenes/nuscenes_map_infos_val.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa \
  --attack-objective eta \
  --source-frame lidar \
  --max-samples 20 \
  --coverage-threshold 0.05 \
  --render-overlays \
  --render-max-samples 5
```

Run raw-image attack rendering injection smoke with:

```bash
/home/dj/.conda/envs/maptr/bin/python scripts/smoke_attack_rendering_injection.py \
  --stream-ann datasets/nuScenes/nuscenes_map_infos_val.pkl \
  --asset-csv /data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv \
  --out-dir /data/dj/MapEcho/artifacts/rendering_sanity/injection_smoke \
  --attack-objective eta \
  --source-frame lidar \
  --max-samples 5 \
  --offsets=-2,-1,0,1,2
```

Before model-level hook sanity, audit whether the selected config/checkpoint is
actually temporal:

```bash
/home/dj/.conda/envs/maptr/bin/python scripts/audit_streammapnet_temporal_readiness.py \
  --config src/StreamMapNet/plugin/configs/mapecho_nusc_baseline_480_60x30_30e_eval.py \
  --checkpoint ckpts/nusc_baseline_480_60x30_30e.pth \
  --stream-ann datasets/nuScenes/nuscenes_map_infos_val.pkl \
  --phase1-tokens /data/dj/MapEcho/artifacts/phase1/phase1_probe_tokens.txt \
  --out /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/temporal_readiness_audit.json
```

The current oldsplit checkpoint is a non-streaming baseline, so it cannot be
used for query/BEV temporal hook or reset sanity.

The original `/home/dj/StreamMapNet` tree does contain temporal NuScenes configs
such as `plugin/configs/nusc_newsplit_480_60x30_24e.py`, and the MapEcho copy is
in sync for the key temporal hook files. The downloaded newsplit temporal
checkpoint is available at `ckpts/nusc_newsplit_480_60x30_24e.pth`; use
`src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py`
for local clean/reset/attack sanity. This wrapper disables the extra backbone
preload and uses batch size 1 for streaming state buffers.

The official newsplit checkpoint is temporal-ready, but the existing CCS'25
oldsplit Phase 1 set only overlaps newsplit validation in 5 / 20 frames. Treat
oldsplit-seed runs with the newsplit checkpoint as sanity/diagnostic unless a
newsplit-val asymmetric candidate set is rebuilt.

The current split strategy is documented in
`doc/experiment_plan_v11_newsplit_strategy.md`. Phase 1.0 hook sanity should use
`/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_tokens.txt`.
The first clean_keep debug config is
`src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_phase1_0_clean_keep_debug.py`.
Run it with `bash scripts/run_phase1_0_clean_keep.sh`; this wrapper adds the CCS
repository root to `PYTHONPATH` so the CCS-modified `mmdet3d.apis` can import
`attack_toolkit`.

Run reset sanity with `bash scripts/run_phase1_0_reset_sanity.sh`. It resets
after target frame `t` and before `t+1`, matching the later attack recovery
ablation timing.

Run the first attack-at-t dry run with `bash scripts/run_phase1_0_attack_keep.sh`,
then summarize clean vs attack effects with
`bash scripts/summarize_phase1_0_attack_dry_run.sh`.

Run the single-sequence attack reset ablation with
`bash scripts/run_phase1_0_attack_reset_ablation.sh`, then summarize reduction
ratios with `bash scripts/summarize_phase1_0_attack_reset_ablation.sh`.

## Notes

- `src/StreamMapNet` was copied from `/home/dj/StreamMapNet` with the Phase 0.5
  instrumentation changes and without its `.git` directory or dataset cache.
- `scripts/sync_streammapnet_from_phase05_source.sh` refreshes the copied source
  from `/home/dj/StreamMapNet` and restores the local dataset link.
- The CCS'25 attack repository is large, so this project references it by path
  instead of copying the full 11GB tree.
