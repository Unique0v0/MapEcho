# Phase 1.9 Paper Evidence Status

## Status

```text
Phase 1.9 result consolidation and paper-ready evidence packaging: PASS
```

This stage packages the selected114 controlled temporal evaluation into
paper-ready tables, recovery plots, qualitative case lists, and a concise
evidence narrative. It does not rerun StreamMapNet inference.

## Input

```text
summary_dir:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary

asset_csv:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv
```

## Output

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence/
  phase1_9_paper_evidence_summary.md
  phase1_9_paper_evidence_manifest.json
  tables/
  curves/
  plots/
  cases/
```

Generated tables:

```text
tables/main_temporal.csv/.md
tables/scene_clustered_ci.csv/.md
tables/primary_scene.csv/.md
tables/internal_reduction.csv/.md
```

Generated figures:

```text
plots/recovery_curve_median_delta_cd.png/.pdf
plots/positive_rate_t1_t2.png/.pdf
```

Generated qualitative case list:

```text
cases/qualitative_case_selection.csv
cases/qualitative_case_selection.md
```

## Key Packaged Result

```text
selected114:
  114 frames / 38 scenes

attack_keep:
  t+1 median Delta CD = +0.0333 m
  t+1 positive = 82 / 114
  t+2 median Delta CD = +0.0179 m
  t+2 positive = 69 / 114

reset_all:
  t+1 positive = 0 / 114
  t+2 positive = 0 / 114

reset_BEV:
  t+1 positive = 10 / 114
  t+2 positive = 3 / 114

reset_query:
  t+1 positive = 78 / 114
  t+2 positive = 67 / 114
```

The conservative one-primary-frame-per-scene analysis remains positive:

```text
38 frames / 38 scenes
attack_keep t+1 median Delta CD = +0.0410 m
attack_keep t+1 positive = 26 / 38
attack_keep t+2 median Delta CD = +0.0194 m
attack_keep t+2 positive = 24 / 38
```

## Regeneration Command

```bash
bash scripts/package_phase1_9_paper_evidence.sh
```

Optional overrides:

```bash
SUMMARY_DIR=/path/to/summary \
ASSET_CSV=/path/to/assets.csv \
OUT_DIR=/path/to/out \
bash scripts/package_phase1_9_paper_evidence.sh
```

## Interpretation

The selected114 evidence package supports the current main mechanism story:

```text
BEV memory dominates map-level target-boundary geometry residue.
Query memory mainly carries immediate internal query/prediction residue.
Reset-all closes the total temporal-state causal loop.
```
