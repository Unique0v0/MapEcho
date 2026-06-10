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

Merge completed best-location assets:

```bash
cd /home/dj/MapEcho

python scripts/merge_ccs_model_scored_assets.py \
  --out-root /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114 \
  --tokens-file /data/dj/MapEcho/artifacts/phase1_8b_assets/phase1_8b_selected_tokens.txt \
  --out-csv /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv \
  --out-tokens /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt
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
