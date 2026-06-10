# CCS25 Full-chain Migration Audit

Date: 2026-06-04

## Verdict

The current MapEcho pipeline is **not yet a strict full-chain migration** of the
CCS25 camera-glare workflow.

What is strict enough now:

```text
six-camera frame-t replacement
CCS-style lens-flare formula
power = 3000 default
single-camera / simplified renderer disabled in the frame-t builder
CCS-style dense location candidate generation and geometric ranking
StreamMapNet-aware per-location scoring pilot
```

What is not strict yet:

```text
newsplit data-pool construction
full-scale top-400 / multi-sample glare-source location selection
ETA target centerline source
exact CCS ETA centerline objective
tensor-level insertion point equivalence
```

Therefore the current chain should be described as:

```text
CCS-style six-camera rendering with ETA-like heuristic locations.
```

It should not be described as:

```text
full CCS25 camera-glare workflow migrated to StreamMapNet.
```

## Original CCS25 Chain

The original CCS25 chain has the following stages:

| Stage | Original CCS25 behavior |
| --- | --- |
| Data entry | NuScenes val infos and map annotations |
| Scene preprocessing | ego on lane/road segment, not intersection, road polygon sanity, left/right lane boundaries, matched different GT boundary instances |
| Rule filtering | lane-width divergence, tail-heading symmetry rejection, curvature difference, reference curvature, ego-distance gates, scene blacklist |
| Visual refinement | VLM/manual-style asymmetric/symmetric refinement |
| ETA target | diverging route centerline JSON |
| Location candidates | sample diverging boundary every 0.5 m, local offsets within 1.0 m, 4 height values |
| Geometric ranking | camera visibility, alignment to asymmetry anchors, distance score |
| Candidate budget | top 400 locations |
| Rendering | six-camera CCS lens-flare utility |
| Scoring | model forward for each candidate and ETA/RSA boundary score |
| Selected location | location with best model-evaluated score |
| Final evaluation | rerender selected location, run model, save map outputs |

## Current MapEcho Chain

| Stage | Current MapEcho behavior | Status |
| --- | --- | --- |
| Data entry | uses existing indexed `ccs_candidate` / `ccs_asymmetric_dist` derived pools | PARTIAL |
| Scene preprocessing | full CCS preprocess has not been rerun on newsplit val | NOT STRICT |
| Rule filtering | Phase 1.6 applies CCS-style lane-width/curvature/VPA gates to existing pool | PARTIAL |
| Visual refinement | manual audit used for exclusion prefixes; no full VLM/manual labeling pass | PARTIAL |
| ETA target | current newsplit assets usually have `has_centerline_json = False` | NOT STRICT |
| Location source | active experiments still use `mapecho_loc_method = diverge_boundary_anchor_heuristic` unless explicitly switched | NOT STRICT |
| Location candidates | CCS-style dense 0.5 m boundary sampling + local offsets + height enumeration implemented as candidate generation | STRICT-STYLE |
| Geometric ranking | top-400 CCS-style feasibility ranking implemented | STRICT-STYLE |
| Candidate scoring | StreamMapNet-aware pilot implemented; full top-400 / multi-sample scoring not yet run | PARTIAL |
| Rendering | CCS-style six-camera renderer, power 3000 | STRICT-STYLE |
| Camera replacement | all six frame-t camera files replaced; unaffected views copied clean | STRICT-STYLE |
| Schedule | warm-up clean, frame `t` rendered, recovery clean | STREAMMAPNET-SPECIFIC |
| Reset experiments | MapEcho temporal-state ablations | STREAMMAPNET-SPECIFIC |

## Renderer Check

Current implementation:

```text
CCS renderer shim
```

Matches CCS constants:

```text
BASE_RADIUS_DIVISOR = 2
MIN_RADIUS_DIVISOR = 8
MAX_DISTANCE_FACTOR = 30
INTENSITY_SCALE = 0.02
MIN_INTENSITY = 0.6
FALLOFF_POWER = 1.5
BLUE_TINT_FACTOR = 1.1
CAMERA_ANGLE = 60 degrees
```

Current frame-t builder:

```text
frame-t annotation builder
```

Now accepts only:

```text
--renderer ccs
--camera-mode all
```

This removes the previous simplified renderer and single-camera mode from the
active frame-t builder.

Important difference:

```text
Original CCS renders on the model image tensor after model-specific image
loading / resizing / normalization.

MapEcho currently renders BGR image files before the StreamMapNet image
pipeline loads, resizes, normalizes, and pads them.
```

This is appropriate for a file-level StreamMapNet evaluation pipeline, but it
is not byte-for-byte identical to the original tensor-level insertion point.
If strict insertion equivalence is required, a StreamMapNet data-pipeline or
model-runner hook must apply the CCS renderer after `LoadMultiViewImages` and
`ResizeMultiViewImages`, before `Normalize3D`.

## Data-pool Check

Phase 1.6 result after applying CCS-style gates to the current pool:

```text
all_assets:                     243 frames / 15 scenes
ccs_rule_pass:                   19 frames / 4 scenes
phase1_6_high_quality_pool:       8 frames / 3 scenes
phase1_6_scene_diverse_selected:  6 frames / 3 scenes
```

Interpretation:

```text
The current derived pool is too diluted and repetitive for a strict CCS-style
asymmetric evaluation set.
```

Required fix:

```text
Port CCS preprocess_scenes -> classify_by_lane_width -> classify_by_curvature
to StreamMapNet newsplit val from an earlier annotation entry point.
```

## Location-selection Check

Original CCS location selection:

```text
sample along diverging boundary every 0.5 m
sample local offsets within 1.0 m
enumerate 4 heights from -1.84 to 0
rank by geometric feasibility score
keep top 400 locations
render six cameras for each candidate
run model forward for each candidate
score with ETA/RSA boundary objective
save selected location
```

Current MapEcho location source:

```text
active controlled runs: diverge_boundary_anchor_heuristic
dense candidate artifacts: ccs_dense_geometric_top1_unscored
```

Current status:

```text
candidate generation and geometric top-400 ranking are migrated.
StreamMapNet-aware per-location scoring pilot is implemented.
full top-400 / multi-sample model-scored selection is not yet complete.
```

Migrated candidate generation:

```text
script:
  scripts/build_ccs_dense_location_candidates.py

wrapper:
  scripts/build_ccs_dense_location_candidates.sh

outputs:
  /data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates/ccs_dense_top_locations.csv
  /data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates/ccs_dense_selected_top1_assets_unscored.csv
  /data/dj/MapEcho/artifacts/phase1_7_location_selection_pilot/dense_candidates/ccs_dense_location_summary.csv
```

Generated on the Phase 1.6 high-quality pool:

```text
8 samples / 3 scenes
top locations: 3200 rows = 8 * 400
sample_interval = 0.5 m
sample_range = 1.0 m
samples_per_loc = 2
locs_height_num = 4
max_beam_angle = 40 degrees
```

Required fix:

```text
Scale the StreamMapNet-aware dense location-selection pilot:
  W clean warm-up frames
  frame t rendered with candidate location
  six-camera CCS renderer
  frame-t target-boundary score
  selected location and per-candidate trace
```

Pilot implementation:

```text
scripts/run_ccs_location_scoring_pilot.py
scripts/run_ccs_location_scoring_pilot.sh
scripts/run_ccs_location_scoring_fast.py
scripts/run_ccs_location_scoring_fast.sh
```

Smoke check:

```text
target_token = 1db65e17354d4873bd22d3186e7dfcdf
max_candidates = 1
warmup = 10
recovery = 0
power = 3000
output = /tmp/mapecho_location_scoring_smoke

clean forward: passed
candidate frame-t ann: six camera files replaced
candidate forward: passed
candidate_model_scores.csv: written
ccs_model_scored_best_location_asset.csv: written
```

Fast runner smoke:

```text
target_token = 1db65e17354d4873bd22d3186e7dfcdf
max_candidates = 2
model_loads_per_token = 1
rank-1 delta CD = +0.7609 m
rank-2 delta CD = +1.2790 m
best rank = 2
```

The fast runner preserves full warm-up semantics for each candidate. It avoids
repeated model loading, but it does not yet snapshot and restore temporal state
after warm-up.

## ETA Target Check

Original CCS ETA uses:

```text
diverge_route_centerlines_<split>/<sample_token>.json
outward_inward_loss_interpolated(...)
```

Current newsplit assets often record:

```text
has_centerline_json = False
```

Current MapEcho map-level evaluation instead uses target-boundary delta metrics.

Status:

```text
not a strict ETA objective migration.
```

Required fix:

```text
Either port the CCS diverging-route centerline generation to newsplit val, or
explicitly define the StreamMapNet-native target-boundary score as a surrogate
and do not claim exact ETA objective equivalence.
```

## Immediate Changes Already Made

The active frame-t builder now disables simplified modes:

```text
renderer choices = ["ccs"]
camera-mode choices = ["all"]
```

Verified:

```text
py_compile passed for the frame-t builder, CCS renderer shim, and Phase 1.6
pool builder. The dense candidate generator also passes py_compile and has a
two-sample smoke check plus full 8-sample candidate generation output.
```

## Required Work Before Claiming Full Migration

1. Rebuild the newsplit data pool from the original CCS preprocessing entry.
2. Add StreamMapNet-aware per-location scoring for the migrated dense
   candidate list.
4. Decide whether strict tensor-level renderer insertion is required.
5. Port ETA centerline generation or clearly define a StreamMapNet-native
   surrogate score.
6. Run a small pilot comparing heuristic location vs dense-selected location.

## Safe Paper Wording

Current accurate wording:

```text
We port the CCS-style six-camera lens-flare renderer and evaluate temporal
recovery in StreamMapNet using ETA-like heuristic glare-source locations.
```

Wording to avoid until migration is complete:

```text
We fully migrate the CCS25 end-to-end evaluation pipeline to StreamMapNet.
```
