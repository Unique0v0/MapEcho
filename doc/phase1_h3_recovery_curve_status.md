# Phase H3 Recovery Curve Status

## Status

```text
Phase H3 recovery curve packaging: PASS
```

This stage recomputes target-boundary matched Delta CD for selected114 recovery
offsets `t+1 ... t+9` from existing outputs. It does not rerun StreamMapNet.

## Input

```text
tokens_file:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt

asset_csv:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv

run_root:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check
```

## Output

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_h3_recovery_curve/
  h3_recovery_matched_deltas_all.csv
  h3_recovery_curve_summary.csv
  h3_auc_cd_by_token.csv
  h3_auc_cd_summary.csv
  h3_recovery_curve_median_delta_cd.png/.pdf
  h3_auc_cd_bar.png/.pdf
  phase1_h3_recovery_curve_summary.md
  phase1_h3_recovery_curve_manifest.json
```

Generation completed:

```text
requested_tokens = 114
completed_tokens = 114
missing = 0
offsets = t+1 ... t+9
```

## Regeneration Command

```bash
bash scripts/package_phase1_h3_recovery_curve.sh
```

## Metrics

```text
Delta CD_i = CD_condition(t+i) - CD_matched_clean(t+i)
positive residue = Delta CD_i > 0.01 m
AUC_CD = sum_i max(0, Delta CD_i), i = 1...9
```

## Key Result

Attack-keep recovery curve:

| Offset | Median Delta CD | Positive Rate |
| ---: | ---: | ---: |
| t+1 | +0.0333 m | 82/114 = 71.9% |
| t+2 | +0.0179 m | 69/114 = 60.5% |
| t+3 | +0.0101 m | 57/114 = 50.0% |
| t+4 | +0.0057 m | 46/114 = 40.4% |
| t+5 | +0.0039 m | 30/114 = 26.3% |
| t+6 | +0.0017 m | 26/114 = 22.8% |
| t+7 | +0.0006 m | 21/114 = 18.4% |
| t+8 | +0.0007 m | 28/114 = 24.6% |
| t+9 | +0.0007 m | 20/114 = 17.5% |

AUC_CD over `t+1...t+9`:

| Condition | Median AUC_CD | Mean AUC_CD | AUC > 0.03 | AUC > 0.05 | AUC > 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attack_keep | 0.1166 | 0.2978 | 92/114 | 80/114 | 62/114 |
| attack_reset_all | 0.0000 | 0.0000 | 0/114 | 0/114 | 0/114 |
| attack_reset_BEV | 0.0020 | 0.0098 | 10/114 | 1/114 | 1/114 |
| attack_reset_query | 0.1099 | 0.3254 | 92/114 | 83/114 | 60/114 |

Important reporting note:

```text
Do not emphasize AUC_CD > 0 as a positive AUC rate. For reset_BEV, tiny
numerical positive values make AUC_CD > 0 overly sensitive. The meaningful
statement is that reset_BEV reduces median recovery AUC from 0.1166 to 0.0020,
nearly eliminating map-level residue. Thresholded AUC counts are used when a
positive AUC rate is needed.
```

Interpretation:

```text
attack_keep shows a clear recovery-delay curve: median residue is strongest at
t+1 and decays toward zero by t+7...t+9. reset_all removes the full curve.
reset_BEV nearly removes the map-level curve, while reset_query largely follows
attack_keep. This supports H3 and reinforces BEV memory as the dominant
map-level geometry-residue channel.
```
