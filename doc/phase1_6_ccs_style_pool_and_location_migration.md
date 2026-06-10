# Phase 1.6 CCS-style Pool and Location Migration

Date: 2026-06-04

## Purpose

Phase 1.6 is a pipeline-alignment checkpoint after the manual visual audit.
It does not run StreamMapNet inference. It checks whether the current newsplit
candidate pool can support a CCS-style asymmetric robustness set, and records
what remains to migrate from the original CCS workflow.

## Manual Audit Inputs

The latest visual audit found that many Phase 1.5 candidates are not suitable
as asymmetric target-boundary cases:

```text
straight / repeated scene-token prefixes:
  005cfc, 51c1ee, 87eb02, 282c9a, bb9632,
  c6ceba, c943bd, d85a67, e9c518

center-road glare-source placement:
  d0a9cf
```

These prefixes are used as a manual token-prefix exclusion list in the
Phase 1.6 pool builder.

## Original CCS Data Filtering

The original CCS project does not start from arbitrary frames. Its rule-based
asymmetric-scene construction has three important layers.

### 1. Preprocess Scenes

The original preprocessing:

```text
requires ego to be on a lane / road segment
rejects intersections
rejects overly complex road polygons
extracts current lane left/right boundaries
matches both sides to different GT boundary instances
requires enough forward boundary points
requires the boundaries to be close enough to ego
requires each boundary to have sufficient endpoint separation
```

This full preprocessing layer has not yet been rerun on StreamMapNet newsplit
val. The current MapEcho Phase 1.6 script only applies the later CCS-style
geometry gates to the already-indexed `ccs_candidate` derived pool.

### 2. Lane-width Asymmetry

The original lane-width gate aligns the left and right boundaries from the
closest ego point forward, then checks whether future lane width exceeds the
initial lane width by more than:

```text
DIST_OFFSET_THRESHOLD = 5 m
```

### 3. Curvature Asymmetry

The original curvature stage:

```text
rejects tail-heading symmetric boundaries
identifies the more diverging side by direction-change and turning scores
requires max curvature difference > 0.1
requires reference regional curvature < 0.3
requires large-curvature-difference points to be 3-15 m from ego
requires those points to be in front of ego, y >= 3 m
requires the diverging boundary to be within 10 m of ego
removes CCS problematic scenes
```

The CCS scene exclusion list is also included:

```text
scene-0329, scene-0907, scene-0908, scene-0557, scene-0560,
scene-0561, scene-0632, scene-0109, scene-0784
```

## Phase 1.6 Builder

Run:

```bash
bash scripts/build_phase1_6_ccs_style_pool.sh
```

Implementation:

```text
scripts/build_phase1_6_ccs_style_pool.py
```

Outputs:

```text
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_ccs_rule_table.csv
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_high_quality_pool_assets.csv
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_high_quality_pool_tokens.txt
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_high_quality_selected_assets.csv
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_high_quality_selected_tokens.txt
/data/dj/MapEcho/artifacts/phase1_6_ccs_style_pool/phase1_6_summary.json
```

## Phase 1.6 Results

Using:

```text
min_vpa = 0.15
max_per_scene = 2
manual visual-audit exclusion = enabled
CCS scene exclusion = enabled
```

| Set | Frames | Scenes | Median VPA | Median curvature diff | Median lane-width gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_assets | 243 | 15 | 0.0620 | 0.0118 | 0.0082 |
| ccs_rule_pass | 19 | 4 | 0.1719 | 0.1173 | 12.2353 |
| vpa_gate_pass | 55 | 11 | 0.2333 | 0.1365 | 4.8503 |
| phase1_6_high_quality_pool | 8 | 3 | 0.1978 | 0.1626 | 13.7095 |
| phase1_6_scene_diverse_selected | 6 | 3 | 0.1999 | 0.1626 | 13.8108 |

Failure counts among rejected rows:

```text
lane_width: 193
curvature: 185
point_distance: 213
vpa: 188
not_ccs_scene_blacklist: 21
not_manual_blacklist: 10
diverge_near_ego: 6
```

## Interpretation

The current broad `ccs_candidate` derived pool is not enough to produce a
scene-diverse CCS-style asymmetric set. After applying CCS-style geometry
gates, target-boundary VPA, the manual visual exclusion list, and the CCS scene
exclusion list, only:

```text
8 frames / 3 scenes
```

remain.

Therefore, the next data-pool step should not be another round of looser
post-hoc filtering on the same pool. The project should rebuild the candidate
pool from an earlier newsplit-val entry point, ideally by porting the original
CCS:

```text
preprocess_scenes -> classify_by_lane_width -> classify_by_curvature
```

workflow to StreamMapNet newsplit annotations.

## Current Camera-glare Pipeline Status

Already migrated:

```text
six-camera replacement at frame t
CCS-style lens-flare renderer
power = 3000 default
target-frame-only schedule
```

Still not migrated:

```text
original CCS dense glare-source location selection
original ETA/RSA model-evaluated candidate scoring
StreamMapNet-specific target-frame scoring function
```

The current Phase 1.5/1.6 location source remains:

```text
mapecho_loc_method = diverge_boundary_anchor_heuristic
```

It should be described as:

```text
CCS-style six-camera rendering with ETA-like heuristic locations.
```

It should not be described as:

```text
full CCS ETA location selection on newsplit.
```

## Original CCS Location Selection

Original camera-glare configuration:

```text
total_locs = 400
sample_interval = 0.5 m
locs_height_num = 4
sample = true
samples_per_loc = 2
sample_range = 1.0 m
power = 3000
beam_angle = 40 degrees
```

Original location-selection procedure:

```text
1. sample points densely along the diverging boundary
2. optionally sample local random offsets around each boundary point
3. enumerate heights from -1.84 to 0
4. rank by geometric feasibility score:
   camera visibility + alignment to asymmetry anchors + distance
5. keep top 400 candidates
6. render six-camera lens flare for each candidate
7. run the map model for each candidate
8. compute ETA/RSA boundary score
9. save the selected location per sample
```

## StreamMapNet Migration Plan

### S1. Rebuild Newsplit Data Pool

Port the original CCS data filtering to newsplit val:

```text
preprocess_scenes
classify_by_lane_width
classify_by_curvature
temporal eligibility W=10/L=9
target-boundary VPA sanity
lightweight visual audit for ambiguous cases
```

Target:

```text
>= 25 frames / >= 8 scenes for a controlled set
>= 40 frames / >= 10 scenes for a stronger main set
```

### S2. Port Candidate-location Generation

Create a StreamMapNet location-selection asset builder that reproduces:

```text
boundary sampling at interval = 0.5 m
local sampled offsets with sample_range = 1.0 m and samples_per_loc = 2
height enumeration over 4 values from -1.84 to 0
geometric feasibility ranking with max_beam_angle = 40 degrees
top 400 candidates
```

### S3. Adapt Location Selection to Temporal StreamMapNet

For each candidate location:

```text
run warm-up frames clean
apply camera-glare perturbation to frame t only
render all six cameras using CCS renderer
evaluate target-frame map-level score
```

This matters because StreamMapNet's frame-t prediction depends on temporal
state. A cold single-frame search would not match the recovery experiment.

### S4. Define StreamMapNet ETA/RSA Scores

Two options:

```text
Port original ETA/RSA boundary scores where output tensors are compatible.
Use MapEcho target-boundary metrics as StreamMapNet-native surrogates.
```

Recommended first pass:

```text
ETA main: maximize target-frame diverging-boundary delta using
          delta CD / outward-inward surrogate.
RSA optional: compare against wrong reference boundary where available.
```

### S5. Pilot Before Full Cost

Run a small pilot before committing to full cost:

```text
5-10 high-confidence samples
top 100 candidate locations
compare heuristic location vs selected location
measure frame-t delta CD and t+1/t+2 residue
```

If the pilot improves frame-t target-boundary delta, expand to:

```text
top 400 candidate locations
25-40 selected samples
```

## Current Decision

The data-pool issue and the location-selection issue should be handled
separately:

```text
Data pool:
  current ccs_candidate-derived pool is too diluted and repetitive.
  rebuild from newsplit val with original CCS preprocess/lane-width/curvature
  logic.

Location selection:
  renderer is now aligned with CCS.
  location selection is not yet aligned.
  migrate dense CCS location selection to StreamMapNet after the candidate
  pool is cleaner, or run a small pilot on promising samples first.
```
