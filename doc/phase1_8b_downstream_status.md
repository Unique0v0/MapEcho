# Phase 1.8-B Downstream Status

## Data Entry

Manual visual audit confirmed that the Phase 1.8-B sampled scenes are asymmetric.

The active downstream input is:

```text
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt
```

Size:

```text
114 frames / 38 scenes
W=10, L=9 temporal eligible
max 5 frames per scene
```

## Dense Candidate Generation

Completed:

```bash
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_assets.csv \
OUT_DIR=/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates \
bash scripts/build_ccs_dense_location_candidates.sh
```

Output:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_top_locations.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_selected_top1_assets_unscored.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_location_summary.csv
```

Result:

```text
114 frames / 38 scenes
top-400 geometric candidates per frame
```

These are geometry-ranked candidates only, not final model-scored locations.

## Pilot20 For Model Scoring

Prepared a scene-diverse pilot set:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/pilot20/pilot20_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/pilot20/pilot20_tokens.txt
```

Size:

```text
20 frames / 20 scenes
```

Selection rule:

```text
one high-confidence frame per scene, sorted by tag confidence
```

## Top-400 Model Scoring Result

Completed top-400 StreamMapNet frame-t scoring for the pilot20 set:

```bash
bash scripts/run_phase1_8b_pilot20_location_scoring.sh
```

Defaults:

```text
TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_8b_downstream/pilot20/pilot20_tokens.txt
DENSE_CANDIDATES_CSV=/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_top_locations.csv
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20
MAX_CANDIDATES=400
WARMUP=10
RECOVERY=0
POWER=3000
SKIP_COMPLETED=1
```

This step is GPU/model-forward heavy. It can be interrupted and resumed because
`SKIP_COMPLETED=1` is enabled by default.

Result:

```text
20 / 20 samples completed
20 / 20 frame-t delta-CD-to-diverge > 0
median frame-t delta-CD-to-diverge = +0.0476 m
best-rank median = 100.5
best-rank max = 370
```

This confirms that top-400 model-scored location selection is effective on the
scene-diverse pilot set. The max best rank of 370 also indicates that reducing
the geometric candidate budget to top-100 would miss valid high-impact
locations.

## Merged Model-Scored Assets

Merge best locations:

```bash
python scripts/merge_ccs_model_scored_assets.py \
  --out-root /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20 \
  --tokens-file /data/dj/MapEcho/artifacts/phase1_8b_downstream/pilot20/pilot20_tokens.txt \
  --out-csv /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20/ccs_model_scored_top400_pilot20_assets_merged.csv \
  --out-tokens /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20/ccs_model_scored_top400_pilot20_tokens.txt
```

Output:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20/ccs_model_scored_top400_pilot20_assets_merged.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_pilot20/ccs_model_scored_top400_pilot20_tokens.txt
```

## Controlled Temporal Check

Completed controlled temporal evaluation on the merged model-scored pilot20 set:

```text
input size = 20 frames / 20 scenes
W=10, L=9
renderer = ccs
camera mode = all six cameras
power = 3000
```

Map-level matched-delta summary:

```text
attack_keep t+1:
  median delta-CD-to-diverge = +0.0111 m
  positive rate > 0.01 m = 10 / 20
  scene-clustered median CI = [+0.0024, +0.0511]

attack_keep t+2:
  median delta-CD-to-diverge = +0.0091 m
  positive rate > 0.01 m = 10 / 20
  scene-clustered median CI = [+0.0023, +0.0392]

attack_reset_all:
  t+1/t+2 positive = 0 / 20
  median delta-CD-to-diverge = 0

attack_reset_BEV:
  t+1/t+2 positive = 0 / 20
  median delta-CD-to-diverge approximately 0

attack_reset_query:
  t+1 positive = 8 / 20
  t+2 positive = 9 / 20
```

Internal matched-reduction summary:

```text
reset_all:
  query / pred / embedding / fused-BEV reductions = 1.0

reset_query at t+1:
  query-score reduction median = 0.929
  pred-vector reduction median = 0.984
  fused-BEV reduction median = 0.0

reset_BEV:
  fused-BEV reduction median = 1.0
  query / pred reductions are partial but substantial
```

Interpretation:

```text
The CCS-style asymmetric pool plus top-400 model-scored camera-glare locations
produces a clear pilot20 temporal residue signal. Reset-all closes the temporal
state causal loop, while reset-BEV removes map-level boundary residue. Reset-query
mainly suppresses immediate internal query/prediction residue but does not remove
map-level boundary residue. This matches the earlier BEV-dominant geometry-residue
mechanism, now on a scene-diverse 20-scene set.
```

## Next Scale-Up: Selected114

The next scale-up should use the full Phase 1.8-B selected set:

```text
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_assets.csv
/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt
```

Size:

```text
114 frames / 38 scenes
W=10, L=9 temporal eligible
max 5 frames per scene
```

Run top-400 model scoring:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_8b_selected114_location_scoring.sh
```

Defaults:

```text
TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt
DENSE_CANDIDATES_CSV=/data/dj/MapEcho/artifacts/phase1_8b_downstream/dense_candidates/ccs_dense_top_locations.csv
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114
MAX_CANDIDATES=400
WARMUP=10
RECOVERY=0
POWER=3000
SKIP_COMPLETED=1
```

Completed selected114 top-400 model scoring:

```text
114 / 114 samples completed
114 / 114 frame-t delta-CD-to-diverge > 0
median frame-t delta-CD-to-diverge = +0.0954 m
min frame-t delta-CD-to-diverge = +0.0012 m
max frame-t delta-CD-to-diverge = +2.9281 m
best-rank median = 73.5
best-rank max = 381
best-rank >= 360: 5 samples
```

This confirms that the full selected114 set remains frame-t effective after
top-400 model-scored location selection. The rank distribution again supports
keeping top-400 rather than reducing to top-100.

Merge completed best-location assets:

```bash
cd /home/dj/MapEcho

python scripts/merge_ccs_model_scored_assets.py \
  --out-root /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114 \
  --tokens-file /data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt \
  --out-csv /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv \
  --out-tokens /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt
```

Merged selected114 model-scored assets:

```text
requested_tokens = 114
merged_assets = 114
missing = 0
```

Run controlled temporal evaluation:

```bash
cd /home/dj/MapEcho

TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check \
WARMUP=10 \
RECOVERY=9 \
ATTACK_POWER=3000.0 \
ATTACK_RENDERER=ccs \
ATTACK_CAMERA_MODE=all \
bash scripts/run_phase1_1_probe_ablation.sh
```

Summarize:

```bash
cd /home/dj/MapEcho

TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt \
ASSET_CSV=/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check \
bash scripts/summarize_phase1_1_probe_ablation.sh
```

## Controlled Check Fast Runner

To reduce repeated checkpoint loading, an equivalent fast runner is available:

```text
scripts/run_streammapnet_multi_condition.py
scripts/run_phase1_8b_selected114_controlled_check_fast.sh
```

It keeps the same clean/attack annotations, same reset timing, same output
directory structure, and same summarization path, but loads StreamMapNet once per
token and runs all conditions in a single process.

A one-token A/B equivalence check was run on:

```text
5d4b194ee07d418b9a60704991e647eb
```

Comparison:

```text
old runner output root:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/equiv_check_old

fast runner output root:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/equiv_check_fast
```

The following summary CSVs were exactly identical:

```text
overlap_map_matched_deltas_all.csv
overlap_map_matched_deltas_summary.csv
overlap_internal_matched_baseline_all.csv
overlap_internal_matched_baseline_summary.csv
overlap_internal_matched_reductions_all.csv
overlap_internal_matched_reductions_summary.csv
phase1_1_map_residue_summary.csv
phase1_1_internal_reduction_summary.csv
```

Result:

```text
max numeric absolute difference = 0.0
non-numeric fields equal = true
A/B equivalence = PASS
```

Therefore, the fast runner is approved for selected114 controlled temporal
evaluation.

## Selected114 Controlled Result

Completed controlled temporal evaluation and summary:

```text
tokens = 114
scenes = 38
W=10, L=9
renderer = ccs
camera mode = all six cameras
power = 3000
```

Map-level target-boundary residue:

```text
attack_keep t+1:
  median delta-CD-to-diverge = +0.0333 m
  positive rate > 0.01 m = 82 / 114 = 71.9%
  scene-clustered median CI = [+0.0179, +0.0481]
  scene-clustered positive-rate CI = [59.5%, 83.3%]

attack_keep t+2:
  median delta-CD-to-diverge = +0.0179 m
  positive rate > 0.01 m = 69 / 114 = 60.5%
  scene-clustered median CI = [+0.0095, +0.0289]
  scene-clustered positive-rate CI = [48.8%, 71.3%]
```

Reset controls:

```text
attack_reset_all:
  t+1/t+2 median delta-CD-to-diverge = 0
  t+1/t+2 positive = 0 / 114

attack_reset_BEV:
  t+1 median delta-CD-to-diverge = +0.00008 m
  t+1 positive = 10 / 114 = 8.8%
  t+2 median delta-CD-to-diverge = -0.00001 m
  t+2 positive = 3 / 114 = 2.6%

attack_reset_query:
  t+1 median delta-CD-to-diverge = +0.0374 m
  t+1 positive = 78 / 114 = 68.4%
  t+2 median delta-CD-to-diverge = +0.0174 m
  t+2 positive = 67 / 114 = 58.8%
```

One-primary-frame-per-scene conservative analysis:

```text
primary selection:
  38 frames / 38 scenes
  source = first scene_pos per scene fallback

attack_keep t+1:
  median delta-CD-to-diverge = +0.0410 m
  positive = 26 / 38 = 68.4%

attack_keep t+2:
  median delta-CD-to-diverge = +0.0194 m
  positive = 24 / 38 = 63.2%

attack_reset_all:
  t+1/t+2 positive = 0 / 38

attack_reset_BEV:
  t+1 median delta-CD-to-diverge = +0.0006 m
  t+1 positive = 5 / 38 = 13.2%
  t+2 median delta-CD-to-diverge = -0.00005 m
  t+2 positive = 2 / 38 = 5.3%

attack_reset_query:
  t+1 median delta-CD-to-diverge = +0.0373 m
  t+1 positive = 27 / 38 = 71.1%
  t+2 median delta-CD-to-diverge = +0.0184 m
  t+2 positive = 24 / 38 = 63.2%
```

Internal matched-reduction summary:

```text
reset_all:
  query / pred / embedding / fused-BEV reductions = 1.0

reset_query at t+1:
  query-score reduction median = 0.923
  pred-vector reduction median = 0.982
  fused-BEV reduction median = 0.0

reset_BEV:
  fused-BEV reduction median = 1.0
  query / pred reductions are partial but substantial
```

Interpretation:

```text
The selected114 controlled evaluation gives strong scene-clustered evidence of
attack-off temporal residue on the target boundary. Reset-all fully removes the
effect, closing the temporal-state causal loop. Reset-BEV nearly eliminates
map-level boundary residue, while reset-query mostly preserves map-level residue
but removes immediate query/prediction internal residue. This supports the
BEV-dominant geometry-residue mechanism on 114 frames from 38 scenes.
```
