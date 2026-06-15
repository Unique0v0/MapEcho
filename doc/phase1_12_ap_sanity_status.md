# Phase 1.12 AP Sanity Status

## Status

```text
Phase 1.12 AP sanity: READY / DEFERRED
```

This stage is optional and should not replace the primary target-boundary
Delta CD / AUC_CD metrics.

## Purpose

The goal is only to check whether standard StreamMapNet AP moves in a broadly
consistent direction on the selected114 temporal subset. It is not the main
paper metric because MapEcho studies localized target-boundary recovery
residue.

## Correct Evaluation Scope

Do not evaluate a selected114-only submission against the full newsplit
validation set. That would count all missing full-val samples as empty
predictions and artificially depress AP.

The correct scope is:

```text
selected114 target frames
offsets: t, t+1, t+2
subset annotation built from the same offset tokens
submission tokens aligned to the same subset annotation
```

## Script

```text
scripts/package_phase1_12_ap_sanity.py
```

Default exact AP sanity conditions now cover the full matched set:

```text
clean_keep
clean_reset_all
clean_reset_query
clean_reset_bev
attack_keep
attack_reset_all
attack_reset_query
attack_reset_bev
```

Command:

```bash
LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib \
/home/dj/.conda/envs/maptr4090/bin/python scripts/package_phase1_12_ap_sanity.py
```

Fast minimal smoke command:

```bash
LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib \
/home/dj/.conda/envs/maptr4090/bin/python scripts/package_phase1_12_ap_sanity.py --minimal
```

## Reporting Metrics

The script reports AP degradation as a drop:

```text
mAP Drop = mAP_clean - mAP_condition
Boundary AP Drop = AP_boundary_clean - AP_boundary_condition
Relative Drop = Drop / Clean
```

Positive values indicate AP degradation. This is intentionally different from
`condition - clean`, whose negative sign is easy to misread in a paper table.

The summary table emphasizes:

```text
t:
  attack_keep vs clean_keep

t+1 / t+2:
  attack_keep vs clean_keep
  attack_reset_all vs clean_reset_all
  attack_reset_BEV vs clean_reset_BEV
  attack_reset_query vs clean_reset_query
```

Reset rows at frame `t` are computed if requested but should not be emphasized,
because reset takes effect before recovery frames.

## Sanity Records

The script writes:

```text
phase1_12_ap_sanity_frame_stats.csv
  requested_target_tokens
  unique_eval_frames
  duplicate_sample_tokens_removed

phase1_12_ap_sanity_prediction_stats.csv
  missing_prediction_targets
  missing_prediction_unique_samples
  denormalized coordinate range
  coord_range_within_roi

phase1_12_ap_sanity_manifest.json
  config
  offsets
  conditions
  AP thresholds
  output files
```

Coordinate sanity should remain within the StreamMapNet 60m x 30m ROI:

```text
x in [-30, 30]
y in [-15, 15]
```

## Current Note

The exact AP computation reached the official `instance_match` stage, so the
subset data path is viable. However, exact AP matching is much slower than the
localized Delta CD summaries and should not block the current paper-evidence
pipeline. Run it later only if an appendix AP sanity table is needed.

## Reporting Guidance

If this stage is completed later, report it as:

```text
AP sanity check on the selected114 temporal subset
```

Do not present it as the primary result or as full-validation AP.
