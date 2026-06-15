# Phase 1.11 Qualitative-friendly and Clean-quality Status

## Status

```text
Phase 1.11 H3-based qualitative and clean-quality selection: PASS
```

This stage uses existing H3 recovery outputs. It does not rerun StreamMapNet.

## Selection Gates

```text
attack_keep_t1_delta_cd > 0.05
attack_keep_AUC_CD > 0.15
attack_reset_all_AUC_CD = 0
attack_reset_BEV_AUC_CD < 0.02
attack_reset_query_AUC_CD > 0.10
clean_keep_t1_CD percentile low enough
clean recovery CD std percentile low enough
```

## Selection Result

```text
all selected114 tokens = 114
H3 qualitative signal pass = 31
clean_quality_strict_pass = 30 frames / 13 scenes
clean_quality_relaxed_pass = 68 frames / 26 scenes
qualitative_strict_pass = 11 frames / 7 scenes
qualitative_relaxed_pass = 24 frames / 14 scenes
strict review cases written = 11
recovery-focused panels written = 11
```

The strict review set is selected by combining t+1 target-boundary damage,
full-window AUC_CD, matched reset behavior, and clean-quality screening. It is
intended for manual selection of final 3-5 paper qualitative cases, not as a
new quantitative evaluation set.

## Clean-quality Robustness

The clean-quality subsets preserve the main temporal pattern:

```text
full selected114:
  attack_keep t+1 median delta CD = +0.0333 m, positive = 82/114
  attack_keep t+2 median delta CD = +0.0179 m, positive = 69/114
  attack_keep median AUC_CD = 0.1166
  reset_all median AUC_CD = 0.0000
  reset_BEV median AUC_CD = 0.0020
  reset_query median AUC_CD = 0.1099
  AUC_CD > 0.03:
    attack_keep = 92/114
    reset_all = 0/114
    reset_BEV = 10/114
    reset_query = 92/114

clean_quality_strict:
  attack_keep t+1 median delta CD = +0.0514 m, positive = 24/30
  attack_keep t+2 median delta CD = +0.0301 m, positive = 21/30
  attack_keep median AUC_CD = 0.1629
  reset_all median AUC_CD = 0.0000
  reset_BEV median AUC_CD = 0.0020
  reset_query median AUC_CD = 0.1679
  AUC_CD > 0.03:
    attack_keep = 26/30
    reset_all = 0/30
    reset_BEV = 3/30
    reset_query = 25/30

clean_quality_relaxed:
  attack_keep t+1 median delta CD = +0.0455 m, positive = 51/68
  attack_keep t+2 median delta CD = +0.0261 m, positive = 45/68
  attack_keep median AUC_CD = 0.1418
  reset_all median AUC_CD = 0.0000
  reset_BEV median AUC_CD = 0.0020
  reset_query median AUC_CD = 0.1496
  AUC_CD > 0.03:
    attack_keep = 57/68
    reset_all = 0/68
    reset_BEV = 6/68
    reset_query = 58/68
```

This supports the robustness claim that the recovery residue is not solely
driven by poor clean predictions. The BEV reset still removes the map-level
residue in the clean-quality subsets, while query reset largely preserves the
map-level AUC pattern.

Reporting note: avoid using `AUC_CD > 0` as the main positive AUC rate because
very small numerical positives make reset_BEV look positive for nearly every
sample. Use median AUC reduction and thresholded counts such as `AUC_CD > 0.03`
instead.

## Outputs

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/
  phase1_11_candidate_table.csv
  phase1_11_qualitative_candidates_strict.csv
  phase1_11_qualitative_candidates_relaxed.csv
  phase1_11_selected_for_visual_review.csv
  phase1_11_selected_for_visual_review_tokens.txt
  phase1_11_clean_quality_subset_summary.csv
  phase1_11_clean_quality_auc_summary.csv
  phase1_11_selection_summary.md
  phase1_11_selection_summary.json
  recovery_focused_panels/
    phase1_11_recovery_focused_panel_metrics.csv
    phase1_11_recovery_focused_panel_paths.txt
    phase1_11_recovery_focused_panel_summary.json
    phase1_11_strict_*_recovery_panel.png
```

## Commands

```bash
bash scripts/select_phase1_11_qualitative_clean_subset.sh
bash scripts/assemble_phase1_11_recovery_focused_panels.sh
```

## Intended Manual Step

Inspect the recovery focused panels and select final 3-5 paper qualitative
cases. The final paper cases should be chosen for both numerical mechanism
strength and visual clarity.

The recovery-focused panel format is:

```text
rows:
  t+1
  t+2

columns:
  clean_keep
  attack_keep
  attack_reset_all
  attack_reset_BEV
  attack_reset_query
```

Each panel highlights the diverging GT boundary, reference boundary, and
metric-selected predicted boundary for the corresponding condition.

## Manual Review Shortlist

The first manual review identified `scene-0749` and `scene-0962` as visually
usable in the Phase 1.11 strict/relaxed qualitative pool. These cases have been
written to:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/
  phase1_11_final_qualitative_shortlist_user_review.csv
```

Current usable candidates:

```text
scene-0749:
  a2f618c613ae4a9281e0981180a5b6c0  strict rank 6
  8192dbec2b4f4f2c80e642e123370f31  relaxed rank 4
  bac2b24bdf094bb6ab9469bc25daa469  relaxed rank 7

scene-0962:
  c56adbccfeff449f959f25cc41f8b1e5  strict rank 7
  661e92f2f19f44c283f0b540e68fdef8  strict rank 11
```

`scene-0384` was also mentioned in manual review, but it is not present in the
current Phase 1.11 selected114 candidate tables. It should be traced back to
the original visualization source before being used in the paper figure set.

The five valid user-reviewed candidates have also been packaged into a final
manual-review panel directory:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_11_qualitative_clean_subset/
  phase1_11_final_qualitative_shortlist_valid_tokens.csv
  final_review_recovery_panels/
    phase1_11_final_strict_1_a2f618c6_recovery_panel.png
    phase1_11_final_strict_2_c56adbcc_recovery_panel.png
    phase1_11_final_strict_3_661e92f2_recovery_panel.png
    phase1_11_final_relaxed_4_8192dbec_recovery_panel.png
    phase1_11_final_relaxed_5_bac2b24b_recovery_panel.png
```

Figure-selection note:

```text
Prefer the strict candidates for the main paper figure. The relaxed scene-0749
backup panels have strong AUC behavior but their reset columns can look less
intuitive when read as raw CD overlays, because reset conditions should be
interpreted against their matched clean-reset baselines rather than directly
against clean_keep.
```
