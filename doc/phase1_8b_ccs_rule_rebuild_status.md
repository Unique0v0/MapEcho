# Phase 1.8-B CCS Rule-Based Rebuild Status

## Question

Can we reuse the original CCS project for rebuilding the newsplit-val data pool, or do we need to manually select from raw nuScenes?

## Finding

The original project contains a reusable rule-based data construction pipeline:

```text
create_asymmetry_dataset.py
dataset_processing/rule_based_classifier.py
dataset_processing/map_utils.py
dataset_processing/curvature_analysis.py
dataset_processing/geometry_utils.py
```

The relevant original stages are:

```text
nuScenes infos
  -> scenes_candidate
  -> scenes_asymmetric_dist
  -> scenes_asymmetric_curvature
  -> final VLM-refined scenes_asymmetric
```

For MapEcho, the useful part is the rule-based portion:

```text
preprocess_scenes
classify_by_lane_width
classify_by_curvature
```

The VLM/API stage is not needed for the next step.

## Safety Of The Wrapper

The original script writes to `/home/dj/physical-online-map-attack/dataset` by default, so it should not be run directly.

MapEcho now has a wrapper that redirects all outputs into the private artifact directory:

```text
scripts/build_phase1_8b_ccs_rule_pool.py
scripts/build_phase1_8b_ccs_rule_pool.sh
```

Default output:

```text
/data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild
```

The wrapper reads:

```text
original CCS code:
  /home/dj/physical-online-map-attack

nuScenes:
  /data/yuy/dataset/nuScenes/full

StreamMapNet newsplit val annotation:
  /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl
```

It does not write to the original CCS project.

## Adaptation Needed

The original code expects MapTR-style info dicts:

```text
infos['infos']
map_location
ego2global_rotation
ego2global_translation
```

The StreamMapNet newsplit file is a list and uses:

```text
location
e2g_rotation
e2g_translation
```

The wrapper normalizes these fields before calling the original rule-based code.

## Smoke Tests

### First 50 Frames

This was only a plumbing check. The first 50 frames are from one scene:

```text
candidates:             35 frames / 1 scene
asymmetric_dist:         0 frames / 0 scenes
asymmetric_curvature:    0 frames / 0 scenes
```

Result: pipeline runs, but the sample is not representative.

### Cross-Scene Smoke: 3 Frames Per Scene

Command:

```bash
MPLCONFIGDIR=/tmp /home/dj/.conda/envs/maptr4090/bin/python \
  scripts/build_phase1_8b_ccs_rule_pool.py \
  --max-per-scene-input 3 \
  --out-dir /data/dj/MapEcho/artifacts/phase1_8b_ccs_rule_rebuild_smoke_scene3
```

Result:

```text
input subset:             444 frames
candidates:               203 frames / 77 scenes
asymmetric_dist:           84 frames / 34 scenes
symmetric_dist:           119 frames / 47 scenes
asymmetric_curvature:      18 frames / 10 scenes
symmetric_curvature:       56 frames / 21 scenes
invalid_curvature:         10 frames / 7 scenes
```

This is a positive signal. Even a sparse cross-scene pass produces more scene diversity than the current strict Phase 1.8-A pool.

## Interpretation

The original CCS rule-based pipeline is usable and should be migrated before manually selecting from raw images.

Manual visual selection should remain a small audit layer, not the primary data construction method.

## Recommended Next Step

Run the full newsplit-val rule-based rebuild:

```bash
bash scripts/build_phase1_8b_ccs_rule_pool.sh
```

Expected runtime is minutes to tens of minutes because it processes all newsplit-val frames and writes candidate visualizations.

After full rebuild:

```text
sample_tokens_asymmetric_curvature.txt
  -> temporal eligibility W=10/L=9
  -> scene-diverse selection
  -> dense glare-source candidates
  -> geometric top-400
  -> StreamMapNet frame-t scoring
  -> controlled temporal check
```

If the full rule-based rebuild is still too small, then consider raw nuScenes map-level sampling. But the current evidence says the original CCS rule-based pipeline should be tried first.

## Full Rebuild Result

The full newsplit-val rule-based rebuild has completed:

```text
candidates:               2841 frames / 109 scenes
asymmetric_dist:          1242 frames / 87 scenes
symmetric_dist:           1599 frames / 98 scenes
asymmetric_curvature:      315 frames / 51 scenes
symmetric_curvature:       722 frames / 58 scenes
invalid_curvature:         205 frames / 57 scenes
```

This is a strong positive result. The original CCS rule-based data construction,
when run from the StreamMapNet newsplit-val annotation entry point, produces a
large and scene-diverse candidate pool.

## Temporal Eligibility And Asset Table

The Phase 1.8-B asset builder converts the full rule-based output into MapEcho
asset CSVs and applies W=10/L=9 temporal eligibility:

```bash
bash scripts/build_phase1_8b_assets.sh
```

Outputs:

```text
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_all_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_temporal_eligible_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_scene_coverage.csv
```

Result:

```text
all asymmetric_curvature assets:
  315 frames / 51 scenes

W10/L9 temporal eligible:
  180 frames / 38 scenes

scene-diverse selected set:
  114 frames / 38 scenes
  max 5 frames per scene
```

The selected set is now large enough for the downstream location-selection
pipeline:

```text
phase1_8b_selected_assets.csv
  -> dense glare-source candidates
  -> geometric top-400
  -> StreamMapNet frame-t scoring
  -> controlled temporal check
```

The remaining caution is that a few selected samples have low boundary-tag
confidence. They can be retained for scene coverage or filtered into a stricter
subset before expensive model scoring.
