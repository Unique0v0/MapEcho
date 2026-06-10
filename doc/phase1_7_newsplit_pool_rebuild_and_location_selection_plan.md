# Phase 1.7 Newsplit Pool Rebuild and Location Selection Plan

Date: 2026-06-04

## Purpose

Phase 1.7 should resolve the two issues exposed by manual visual review and
Phase 1.6:

```text
1. The current candidate pool is diluted by repeated scenes and straight /
   symmetric road structures.
2. The current glare-source location is still a heuristic anchor, not the full
   CCS dense location-selection process.
```

The goal is to improve evaluation quality before running another controlled
StreamMapNet experiment.

## Current Evidence

Manual visual review found many unsuitable cases:

```text
straight / repeated prefixes:
  005cfc, 51c1ee, 87eb02, 282c9a, bb9632,
  c6ceba, c943bd, d85a67, e9c518

center-road placement:
  d0a9cf
```

Phase 1.6 applied CCS-style rule gates to the current broad pool and obtained:

```text
all_assets:                     243 frames / 15 scenes
ccs_rule_pass:                   19 frames / 4 scenes
phase1_6_high_quality_pool:       8 frames / 3 scenes
phase1_6_scene_diverse_selected:  6 frames / 3 scenes
```

This shows that the current pool is not an adequate source for a scene-diverse
CCS-style asymmetric set.

## Direction A: Rebuild the Newsplit Data Pool

### A1. Start from an earlier newsplit-val entry point

Do not continue tightening the current `ccs_candidate` derived pool. Rebuild
from the full newsplit validation annotation entry where possible.

Required stages:

```text
preprocess_scenes
classify_by_lane_width
classify_by_curvature
temporal eligibility W=10/L=9
scene-diversity selection
target-boundary VPA sanity
lightweight visual audit
```

### A2. Port CCS preprocessing faithfully

The preprocessing should reproduce the CCS checks:

```text
ego on lane / road segment
not intersection
road polygon not overly complex
left/right lane boundaries extracted from current lane
left/right boundaries matched to different GT boundary instances
enough forward points
boundaries close enough to ego
sufficient endpoint separation
```

This step is the main missing piece in the current MapEcho pool construction.

### A3. Use CCS rule thresholds as defaults

Default thresholds:

```text
MIN_BOUNDARY_LENGTH = 10
MIN_ENDPOINT_DISTANCE = 5
MAX_DISTANCE_FROM_EGO = 10
DIST_OFFSET_THRESHOLD = 5
MIN_Y_COORDINATE = 0
CURVATURE_DIFF_THRESHOLD = 0.1
REGION_CURVATURE_THRESHOLD = 0.3
MIN_POINTS = 5
CHUNK_SIZE = 5
ANGLE_THRESHOLD_DEG = 30
MIN_DIST_CLOSEST = 3
MAX_DIST_CLOSEST = 15
MIN_Y = 3
MIN_DIST_TO_DIVERGE_BOUNDARY = 10
```

### A4. Outputs

Recommended output root:

```text
/data/dj/MapEcho/artifacts/phase1_7_newsplit_pool_rebuild
```

Recommended files:

```text
newsplit_preprocess_candidates.csv
newsplit_lane_width_asymmetric.csv
newsplit_curvature_asymmetric.csv
newsplit_temporal_eligible_W10_L9.csv
newsplit_scene_diverse_pool.csv
newsplit_pool_rebuild_summary.json
```

### A5. Acceptance target

Minimum useful target:

```text
>= 25 frames / >= 8 scenes
```

Preferred target:

```text
>= 40 frames / >= 10 scenes
```

If this cannot be reached, the paper should report a smaller controlled
mechanism set rather than overstating broad coverage.

## Direction B: Migrate CCS Dense Location Selection

### B1. Keep renderer fixed

The renderer should remain:

```text
renderer = ccs
camera_mode = all
power = 3000
```

The six-camera rendering now visually matches the original CCS style closely
enough for the next evaluation stage.

### B2. Replace heuristic location with dense candidate generation

Current source:

```text
mapecho_loc_method = diverge_boundary_anchor_heuristic
```

Target source:

```text
CCS-style dense glare-source location selection
```

Candidate generation now reproduces:

```text
sample along diverging boundary every 0.5 m
sample local offsets within 1.0 m
enumerate 4 heights from -1.84 to 0
rank by geometric feasibility score
keep top 400 candidates
```

Implemented:

```text
scripts/build_ccs_dense_location_candidates.py
scripts/build_ccs_dense_location_candidates.sh
```

Current output:

```text
/data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates
8 samples / 3 scenes
3200 top-location rows
```

Important status:

```text
This is dense candidate generation + geometric ranking only.
It is not yet full-scale model-scored final location selection.
```

### B3. Temporal StreamMapNet scoring

Pilot implementation:

```text
scripts/run_ccs_location_scoring_pilot.py
scripts/run_ccs_location_scoring_pilot.sh
scripts/run_ccs_location_scoring_fast.py
scripts/run_ccs_location_scoring_fast.sh
```

For each candidate location, the pilot does:

```text
run W warm-up frames clean
apply camera-glare perturbation at frame t only
render all six cameras
score frame-t target-boundary delta
save selected location and per-candidate trace
```

This should be temporal-state aware because StreamMapNet frame-t output depends
on the warm-up history.

Smoke status:

```text
target_token = 1db65e17354d4873bd22d3186e7dfcdf
max_candidates = 1
warmup = 10
recovery = 0
power = 3000
output = /tmp/mapecho_location_scoring_smoke

clean_keep forward: passed
candidate forward: passed
six-camera frame-t replacement: passed
candidate_model_scores.csv: written
model-scored best-location asset: written
```

Top-20 single-sample pilot:

```text
target_token = 1db65e17354d4873bd22d3186e7dfcdf
output = /data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/model_scoring_pilot_top20
num_candidates_scored = 20
model-scored best rank = 6
best_xyz_lidar = (-1.6064419686, 2.3311605667, -0.6133333333)
best frame-t delta CD to diverging boundary = +1.6665 m
geometric rank-1 frame-t delta CD to diverging boundary = +0.7611 m
```

Interpretation:

```text
The StreamMapNet-aware model-scored location can differ from the geometric
top-1 location. Therefore the CCS per-location model-scoring stage is
functionally necessary and should not be replaced by geometric top-1.
```

Current limitation:

```text
The pilot ranks candidates with a StreamMapNet-native frame-t target-boundary
CD delta. This is a model-evaluated score and is suitable for StreamMapNet
location selection, but it is not the exact original CCS ETA centerline loss.
```

Fast runner status:

```text
scripts/run_ccs_location_scoring_fast.py loads StreamMapNet once per target
token and resets temporal state before each clean/candidate sequence. It keeps
the same scoring semantics as the transparent pilot: W clean warm-up frames plus
frame-t candidate rendering.

Smoke:
  target_token = 1db65e17354d4873bd22d3186e7dfcdf
  max_candidates = 2
  output = /tmp/mapecho_location_scoring_fast_smoke
  best rank = 2
  rank-1 delta CD = +0.7609 m
  rank-2 delta CD = +1.2790 m
  model_loads_per_token = 1
```

Remaining speed limitation:

```text
The fast runner still reruns W warm-up frames for each candidate to preserve
clean temporal-state semantics. A later optimization can snapshot the warm-up
temporal state and restore it before each candidate frame, but that requires
careful state serialization for query memory and BEV memory.
```

### B4. Pilot before full cost

Run a pilot first:

```text
5-10 visually confirmed samples
top 20-100 candidate locations
compare heuristic location vs dense-selected location
measure frame-t delta CD and t+1/t+2 residue
```

Expand only if the pilot improves frame-t target-boundary delta.

### B5. Outputs

Recommended output root:

```text
/data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot
```

Recommended files:

```text
candidate_locations.csv
location_selection_trace.csv
selected_locations.csv
heuristic_vs_selected_location_summary.csv
location_selection_pilot_summary.json
```

## Recommended Execution Order

1. Rebuild the newsplit data pool using CCS-style preprocessing and rule gates.
2. Run a lightweight visual audit on the rebuilt pool.
3. If the rebuilt pool reaches at least 25 frames / 8 scenes, run a small dense
   location-selection pilot.
4. If the pilot improves frame-t target-boundary delta, freeze a Phase 1.7
   controlled set.
5. Run the next controlled StreamMapNet experiment only after the data pool and
   location source are both cleaner.

## Reporting Language

Use neutral paper-writing terminology:

```text
camera-glare perturbation
robustness evaluation
glare-source location selection
target-frame delta
conditional temporal residue
BEV-memory removal
query-state internal effect
```

Avoid describing the current pipeline as full CCS location selection until the
dense candidate scoring loop is actually migrated.
