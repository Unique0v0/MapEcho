# Phase 1.14 Stronger Single-Frame Settings

## Goal

Phase 1.14 keeps the main MapEcho setting fixed:

```text
single-frame perturbation at t
clean recovery t+1...t+9
matched reset baselines
StreamMapNet newsplit checkpoint
selected114 / 38 scenes as the main temporal set
```

Only the perturbation type and objective are expanded to match the four
original CCS branches.

## Original CCS Branches

The original project exposes four combinations through one unified entry:

```text
tools/attack.py
attack.type in {blind, patch}
attack.loss in {eta, rsa}
```

Shell entries:

```text
run_eta_blind.sh
run_rsa_blind.sh
run_eta_patch.sh
run_rsa_patch.sh
```

The four branches are:

| Branch | Original type | Original objective | Current MapEcho status |
| --- | --- | --- | --- |
| blind_eta | camera-glare rendering | early-turn objective | already migrated and used for selected114 |
| blind_rsa | camera-glare rendering | road-straightening objective | Phase 1.14 first migration target |
| patch_eta | perspective physical patch + optimization | early-turn objective | candidate generation migrated; optimization blocked until newsplit ETA centerlines are available |
| patch_rsa | perspective physical patch + optimization | road-straightening objective | pending full migration |

## Original Search Flow

The original flow is not a pure geometry heuristic. It is:

```text
1. sample candidate positions along the diverging boundary
2. add local random samples around each boundary point
3. rank by geometric feasibility score
4. keep top-K candidates
5. render candidate effect / optimize patch
6. run model forward
7. compute objective-specific model loss
8. choose the best model-scored candidate
```

For camera-glare:

```text
blind.total_locs = 400
blind.sample_interval = 0.5
blind.locs_height_num = 4
blind.sample = True
blind.samples_per_loc = 2
blind.sample_range = 1.0
power = 3000
```

For patch:

```text
patch.total_locs = 400
patch.sample_interval = 0.5
patch.step_per_loc = 20
patch.sample = True
patch.samples_per_loc = 2
patch.sample_range = 1.0
patch.type = vertical
patch.width = 3
patch.height = 2
```

Patch is not equivalent to placing a static image at a point. The original
implementation initializes a patch tensor and mask, projects it through the
current camera geometry, applies it to every visible camera, and optimizes the
patch tensor for up to 20 steps per candidate location.

## Current Implemented Migration

### blind_eta

Already used in Phase 1.8B selected114:

```text
CCS-style six-camera renderer
top-400 model-scored location search
objective score = delta CD to diverging boundary
```

### blind_rsa

Added in Phase 1.14:

```text
scripts/run_ccs_location_scoring_fast.py --attack-objective rsa
scripts/run_phase1_14_blind_rsa_location_scoring.sh
scripts/merge_phase1_14_blind_rsa_assets.sh
```

The RSA score uses the existing target/reference boundary metrics:

```text
wrong_reference_preference = CD_to_diverge - CD_to_reference
```

The selected location maximizes:

```text
delta_wrong_reference_preference
then delta_cd_to_diverge
then candidate closeness to reference
then geometric feasibility score
```

This is the StreamMapNet-compatible proxy for the original RSA objective,
which selected the candidate that moves the predicted diverging boundary toward
the straightened/reference-like target.

Outputs are separate from the ETA main results:

```text
/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/
  model_scoring_fast_top400_selected114_blind_rsa/
```

## Commands

Run top-400 model-scored camera-glare RSA locations:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_14_blind_rsa_location_scoring.sh
```

Merge per-token RSA assets:

```bash
cd /home/dj/MapEcho
bash scripts/merge_phase1_14_blind_rsa_assets.sh
```

Then run the existing single-frame controlled temporal check with the RSA asset
CSV:

```bash
TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_tokens.txt \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_assets_merged.csv \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/blind_rsa_selected114_controlled_check \
ATTACK_POWER=3000.0 \
ATTACK_OBJECTIVE=rsa \
ATTACK_RENDERER=ccs \
ATTACK_CAMERA_MODE=all \
bash scripts/run_phase1_8b_selected114_controlled_check_fast.sh
```

Package recovery curve:

```bash
TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_tokens.txt \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/model_scoring_fast_top400_selected114_blind_rsa/ccs_model_scored_top400_selected114_blind_rsa_assets_merged.csv \
RUN_ROOT=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/blind_rsa_selected114_controlled_check \
OUT_DIR=/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/blind_rsa_h3_recovery_curve \
bash scripts/package_phase1_h3_recovery_curve.sh
```

## Pending Full Migrations

## Completed blind_rsa Results

`blind_rsa` has now been run on the same selected114 / 38-scene temporal set as
the main `blind_eta` setting.

```text
location scoring: 114 / 114 completed
controlled temporal check: 114 / 114 completed
H3 recovery package: 114 / 114 completed
```

Outputs:

```text
/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/
  model_scoring_fast_top400_selected114_blind_rsa/
  blind_rsa_selected114_controlled_check/
  blind_rsa_h3_recovery_curve/
```

### Recovery Curve

`blind_rsa` produces a clear but slightly weaker recovery curve than the main
`blind_eta` setting:

| Setting | t+1 median Delta CD | t+1 positive | t+2 median Delta CD | t+2 positive | Median AUC_CD |
| --- | ---: | ---: | ---: | ---: | ---: |
| blind_eta | +0.0333 m | 82/114 | +0.0179 m | 69/114 | 0.1166 |
| blind_rsa | +0.0264 m | 76/114 | +0.0101 m | 58/114 | 0.0996 |

### Reset Pattern

The reset mechanism remains consistent with the main `blind_eta` result:

| Setting | reset_all median AUC | reset_BEV median AUC | reset_query median AUC |
| --- | ---: | ---: | ---: |
| blind_eta | 0.0000 | 0.0020 | 0.1099 |
| blind_rsa | 0.0000 | 0.0021 | 0.0986 |

This means `blind_rsa` confirms the same channel interpretation:

```text
reset_all removes the recovery residue completely
reset_BEV nearly removes map-level boundary residue
reset_query preserves most map-level boundary residue
```

### Interpretation

`blind_rsa` is not stronger than `blind_eta` on the selected114 set, but it is a
useful complementary sanity setting. It reproduces the H3 recovery pattern and
the BEV-dominant map-level reset mechanism under a different objective for
choosing the single-frame location.

For the main paper evidence, `blind_eta` should remain the primary single-frame
setting. `blind_rsa` can be used as an auxiliary objective check, especially if
it yields better qualitative examples in specific scenes.

### patch_eta and patch_rsa

These require full migration of the original patch chain:

```text
candidate location generation
patch center offset by diverging side
heading facing ego
pseudo-area creation
3D patch projection to all visible cameras
patch tensor and mask initialization
20-step Adam optimization per location
objective-specific model loss
best patch tensor + geometry asset serialization
single-frame temporal replay on StreamMapNet
```

Do not replace this with a simplified static overlay if the goal is strict CCS
migration.

### patch candidate generation

The original patch candidate-generation stage has been ported into MapEcho:

```text
scripts/ccs_patch_utils.py
scripts/build_ccs_patch_candidates.py
scripts/build_phase1_14_patch_eta_candidates.sh
```

This stage reproduces the original patch-location geometry:

```text
sample diverging boundary every 0.5 m
offset patch center by width / 2 according to diverging side
set vertical patch size to width=3 m, height=2 m
orient patch heading to face ego
add local random samples with sample_range=1.0 and +/-30 deg heading jitter
rank by geometric feasibility with max_beam_angle=20 deg
keep total_locs / step_per_loc = 400 / 20 = 20 patch locations
```

The output path is intentionally separate from all blind-branch artifacts:

```text
/data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_eta_candidates/
  patch_eta_top20_candidates.csv
  patch_eta_all_candidates.csv
```

Command:

```bash
cd /home/dj/MapEcho
bash scripts/build_phase1_14_patch_eta_candidates.sh
```

### strict patch_eta blocker

The original `patch_eta` optimization loss is not only a target-boundary
Chamfer score. It uses the CCS ETA route-centerline target:

```text
diverge_route_centerlines_asymmetric/<sample_token>.json
outward_inward_loss_interpolated(
  predicted_diverging_boundary,
  diverging_boundary,
  diverging_route_centerline,
  reference_boundary
)
```

The selected114 newsplit assets currently come from a rebuilt newsplit pool, not
from the original CCS 100-seed asymmetric set. Their scene JSON files do not
carry `diverge_route_centerlines_asymmetric/<token>.json`. Therefore, a strict
`patch_eta` migration requires one of the following before optimization can be
claimed equivalent to the original branch:

```text
1. port the original CCS centerline-generation step to the selected114 newsplit
   samples; or
2. explicitly define a StreamMapNet-native surrogate objective and avoid calling
   it an exact ETA migration.
```

For the current strict-migration route, option 1 is the correct next dependency.
`patch_rsa` does not require the ETA route-centerline file and can be migrated
earlier if the goal is to validate the physical-patch renderer/optimizer before
the centerline dependency is solved.

### patch_rsa optimizer smoke

A one-token physical-patch optimizer smoke entry has been added:

```text
scripts/run_ccs_patch_scoring_streammapnet.py
scripts/run_phase1_14_patch_rsa_smoke.sh
```

The smoke path uses the migrated patch candidates and exercises the full patch
runtime components on StreamMapNet:

```text
clean W=10 warm-up
snapshot temporal state before target frame t
initialize patch tensor and mask
project patch to all visible cameras
optimize patch tensor with Adam
score with the original RSA-style target boundary
save best_patch_rsa.pkl
write a patch_rsa_sequence_ann.pkl with all target-frame camera paths replaced
```

Default smoke command:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_14_patch_rsa_smoke.sh
```

Default smoke settings:

```text
TOKEN = first selected114 token
MAX_LOCATIONS = 2
PATCH_STEPS = 2
OUT_ROOT = /data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_smoke
```

This smoke is for implementation validation only. It is not a paper result
because the original patch branch uses 20 locations and 20 optimization steps
per target sample.

Smoke result:

```text
target_token = 5d4b194ee07d418b9a60704991e647eb
num_locations_optimized = 2
patch_steps = 2
clean_loss = 29.3066
best_loss = 29.2881
best_rank = 1
visible_cameras = CAM_FRONT, CAM_FRONT_RIGHT
patch_sequence_ann = written
best_patch_rsa.pkl = written
```

Interpretation:

```text
patch projection, visible-camera selection, patch tensor optimization,
best-patch serialization, and target-frame ann replacement are functional.
The smoke is intentionally too small to evaluate effect strength.
```

Full one-token result:

```text
target_token = 5d4b194ee07d418b9a60704991e647eb
num_locations_optimized = 20
patch_steps = 20
clean_loss = 29.3066
best_loss = 29.1862
best_rank = 1
visible_cameras = CAM_FRONT, CAM_FRONT_RIGHT
patch_sequence_ann = written
best_patch_rsa.pkl = written
```

Candidate-level optimization was consistent:

```text
20 / 20 locations completed with status=ok
20 / 20 locations improved the RSA loss relative to clean
best loss_delta_vs_clean = -0.1204
```

This verifies that the original patch-branch scale settings are runnable for a
single StreamMapNet target:

```text
top patch locations = 20
optimization steps per location = 20
```

The next check is temporal replay with the optimized patch frame:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_14_patch_rsa_one_token_replay.sh
```

This reuses:

```text
clean_ann = selected114 clean sequence
attack_ann = patch_rsa_sequence_ann.pkl from the full one-token optimizer
```

and produces the same keep/reset directory structure as the main Phase 1
pipeline.

Replay result:

```text
target_token = 5d4b194ee07d418b9a60704991e647eb
conditions completed = 8 / 8
query dumps per condition = 20
BEV dumps per condition = 20
map-level summaries = written for offsets t, t+1, t+2
```

Map-level matched deltas:

| Condition | t Delta CD | t+1 Delta CD | t+2 Delta CD |
| --- | ---: | ---: | ---: |
| patch_rsa keep | +0.6355 m | +0.2790 m | +0.1157 m |
| patch_rsa reset_all | +0.6355 m | 0.0000 m | 0.0000 m |
| patch_rsa reset_query | +0.6355 m | +0.3749 m | +0.2499 m |
| patch_rsa reset_BEV | +0.6355 m | +0.0049 m | +0.0137 m |

Internal matched-reset pattern:

```text
reset_all removes t+1 internal residue completely
reset_query reduces query/pred residue but keeps fused BEV residue
reset_BEV removes fused BEV residue completely
```

Interpretation:

```text
The physical-patch RSA branch now reproduces the main MapEcho temporal pattern
on a full one-token replay: strong target-frame boundary corruption, clean-input
recovery residue at t+1/t+2, reset_all removal, and BEV-dominant map-level
removal under reset_BEV.
```

## patch_rsa Pilot-5 Plan

Since the full one-token replay is positive, the next step is a small
multi-token pilot rather than jumping directly to selected114.

Scripts:

```text
scripts/select_phase1_14_patch_rsa_pilot_tokens.sh
scripts/run_phase1_14_patch_rsa_pilot.sh
scripts/package_phase1_14_patch_rsa_pilot.py
scripts/package_phase1_14_patch_rsa_pilot.sh
```

Default pilot:

```text
tokens = first 5 selected114 tokens
max patch locations per token = 20
patch optimization steps per location = 20
optimizer root = /data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_optimizer
replay root = /data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_replay
summary root = /data/dj/MapEcho/artifacts/phase1_14_stronger_single_frame/patch_rsa_pilot5_summary
```

Commands:

```bash
cd /home/dj/MapEcho
bash scripts/select_phase1_14_patch_rsa_pilot_tokens.sh
bash scripts/run_phase1_14_patch_rsa_pilot.sh
bash scripts/package_phase1_14_patch_rsa_pilot.sh
```

The runner supports resumption:

```text
optimizer step skips a token if patch_scoring_summary.json exists
replay step skips a token if phase1_0_single_sequence_map_summary.json exists
```

Pilot pass condition:

```text
not every token needs to be strong,
but the majority should show attack_keep t/t+1 positive map-level deltas,
and reset_all / reset_BEV should reduce or remove t+1/t+2 residue when present.
```

Pilot-5 result:

```text
requested tokens = 5
completed tokens = 5
missing tokens = 0
patch locations per token = 20
patch optimization steps per location = 20
```

Condition summary:

| Condition | t median Delta CD | t positive | t+1 median Delta CD | t+1 positive | t+2 median Delta CD | t+2 positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| patch_rsa keep | +0.5617 m | 5/5 | +0.2980 m | 5/5 | +0.0771 m | 3/5 |
| patch_rsa reset_all | +0.5617 m | 5/5 | 0.0000 m | 0/5 | 0.0000 m | 0/5 |
| patch_rsa reset_BEV | +0.5617 m | 5/5 | -0.0119 m | 1/5 | +0.0020 m | 0/5 |
| patch_rsa reset_query | +0.5617 m | 5/5 | +0.2802 m | 5/5 | +0.1774 m | 5/5 |

Interpretation:

```text
patch_rsa pilot-5 strongly reproduces the MapEcho mechanism:
target-frame boundary corruption appears in all 5 tokens;
clean-input recovery residue remains at t+1 for all 5 tokens;
reset_all removes the t+1/t+2 map-level residue;
reset_BEV nearly removes the t+1/t+2 map-level residue;
reset_query preserves map-level residue.
```

This is stronger than a pure functionality check. It is still not the full
selected114 result, but it is enough to justify expanding `patch_rsa` beyond the
pilot stage.

## patch_rsa Selected114 Result

The full selected114 patch_rsa run has completed:

```text
requested tokens = 114
completed tokens = 114
missing tokens = 0
patch locations per token = 20
patch optimization steps per location = 20
```

Optimizer diagnostics:

```text
median optimizer loss delta = -0.5284
mean optimizer loss delta = -1.1582
median best rank = 8.5
```

Most frequent visible-camera sets:

```text
CAM_FRONT_LEFT               54 / 114
CAM_FRONT + CAM_FRONT_RIGHT  21 / 114
CAM_FRONT_RIGHT              20 / 114
CAM_FRONT + CAM_FRONT_LEFT   17 / 114
CAM_FRONT_LEFT + CAM_BACK_LEFT 2 / 114
```

Condition summary:

| Condition | t median Delta CD | t positive | t+1 median Delta CD | t+1 positive | t+2 median Delta CD | t+2 positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| patch_rsa keep | +0.0323 m | 64/114 | +0.0170 m | 63/114 | +0.0025 m | 51/114 |
| patch_rsa reset_all | +0.0323 m | 64/114 | 0.0000 m | 0/114 | 0.0000 m | 0/114 |
| patch_rsa reset_BEV | +0.0323 m | 64/114 | -0.0016 m | 24/114 | -0.0001 m | 7/114 |
| patch_rsa reset_query | +0.0323 m | 64/114 | +0.0248 m | 68/114 | +0.0077 m | 54/114 |

Thresholded t+1 positive counts:

| Condition | Delta CD > 0.01 | > 0.05 | > 0.10 | > 0.20 |
| --- | ---: | ---: | ---: | ---: |
| patch_rsa keep | 63/114 | 43/114 | 31/114 | 19/114 |
| patch_rsa reset_all | 0/114 | 0/114 | 0/114 | 0/114 |
| patch_rsa reset_BEV | 24/114 | 5/114 | 3/114 | 1/114 |
| patch_rsa reset_query | 68/114 | 44/114 | 27/114 | 16/114 |

Interpretation:

```text
patch_rsa selected114 is weaker than the very strong pilot-5 subset but still
positive overall. The key mechanism remains stable: reset_all completely removes
the recovery residue, reset_BEV nearly removes map-level residue, and
reset_query preserves map-level residue. This makes patch_rsa a successful
stronger single-frame auxiliary setting rather than the primary setting.
```

## Recommended Next Step

Since `blind_rsa` is complete and is slightly weaker than `blind_eta`, the next
strict-migration target remains the patch branch. For exact `patch_eta`, first
port the CCS diverging-route centerline generation to selected114. In parallel,
`patch_rsa` is a practical renderer/optimizer smoke target because it exercises
the full physical-patch chain without the ETA centerline dependency. The patch
branch should only be treated as fully migrated after 3D patch projection,
visible-camera handling, patch tensor/mask initialization, 20-step per-location
optimization, objective-specific scoring, best patch serialization, and
StreamMapNet temporal replay are all verified.
