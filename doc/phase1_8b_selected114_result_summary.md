# Phase 1.8-B Selected114 Result Summary

## Status

```text
Phase 1.8-B selected114 controlled temporal evaluation: PASS
```

This stage uses the rebuilt CCS-style asymmetric scene pool, top-400 geometry
candidates, StreamMapNet frame-t model scoring, CCS six-camera rendering, and
matched reset controls.

## Data

```text
split: nuScenes newsplit val
model: official StreamMapNet newsplit temporal checkpoint
window: W=10, L=9
evaluation set: 114 frames / 38 scenes
renderer: CCS six-camera glare renderer
power: 3000
```

Manual visual audit confirmed the sampled scenes are asymmetric.

## Location Scoring

Top-400 model-scored location selection completed for all selected114 samples.

```text
114 / 114 samples completed
114 / 114 frame-t delta-CD-to-diverge > 0
median frame-t delta-CD-to-diverge = +0.0954 m
best-rank median = 73.5
best-rank max = 381
best-rank >= 360: 5 samples
```

The rank distribution supports keeping top-400 candidates. A smaller top-100
budget would miss several high-impact locations.

## Main Temporal Result

Map-level target-boundary residue:

| Condition | Offset | Median ΔCD_diverge | Positive Rate |
| --- | ---: | ---: | ---: |
| attack_keep | t+1 | +0.0333 m | 82/114 = 71.9% |
| attack_keep | t+2 | +0.0179 m | 69/114 = 60.5% |
| attack_reset_all | t+1 | 0.0000 m | 0/114 |
| attack_reset_all | t+2 | 0.0000 m | 0/114 |
| attack_reset_BEV | t+1 | +0.00008 m | 10/114 = 8.8% |
| attack_reset_BEV | t+2 | -0.00001 m | 3/114 = 2.6% |
| attack_reset_query | t+1 | +0.0374 m | 78/114 = 68.4% |
| attack_reset_query | t+2 | +0.0174 m | 67/114 = 58.8% |

Scene-clustered bootstrap confidence intervals:

```text
attack_keep t+1:
  median CI = [+0.0179, +0.0481]
  positive-rate CI = [59.5%, 83.3%]

attack_keep t+2:
  median CI = [+0.0095, +0.0289]
  positive-rate CI = [48.8%, 71.3%]
```

## Conservative Scene-Level Analysis

One primary frame per scene was selected using the first scene position per
scene fallback:

```text
38 frames / 38 scenes
```

| Condition | Offset | Median ΔCD_diverge | Positive Rate |
| --- | ---: | ---: | ---: |
| attack_keep | t+1 | +0.0410 m | 26/38 = 68.4% |
| attack_keep | t+2 | +0.0194 m | 24/38 = 63.2% |
| attack_reset_all | t+1 | 0.0000 m | 0/38 |
| attack_reset_all | t+2 | 0.0000 m | 0/38 |
| attack_reset_BEV | t+1 | +0.0006 m | 5/38 = 13.2% |
| attack_reset_BEV | t+2 | -0.00005 m | 2/38 = 5.3% |
| attack_reset_query | t+1 | +0.0373 m | 27/38 = 71.1% |
| attack_reset_query | t+2 | +0.0184 m | 24/38 = 63.2% |

This confirms that the result is not driven only by multiple frames from the
same scene.

## Internal Mechanism

Matched internal reduction:

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

## Interpretation

The selected114 controlled evaluation provides strong scene-clustered evidence
that a one-frame target-side camera-glare perturbation can produce attack-off
temporal residue on the target boundary. Reset-all fully removes the residue,
closing the temporal-state causal loop.

The channel-level reset pattern is consistent:

```text
BEV memory dominates map-level target-boundary geometry residue.
Query memory mainly carries immediate internal query/prediction residue.
```

In particular, reset-BEV nearly eliminates map-level boundary residue, while
reset-query preserves most of the map-level residue but strongly suppresses
query-score and predicted-vector internal differences.

## Paper-Ready Wording

```text
On the selected114 newsplit validation set, consisting of 114 target frames from
38 scenes, the one-frame camera-glare perturbation produces clear attack-off
target-boundary residue. At t+1, attack_keep increases target-boundary Chamfer
distance by a median of 0.0333 m, with 82/114 frames exceeding the 0.01 m
positive-residue threshold. At t+2, the median remains positive at 0.0179 m
with 69/114 positive frames. Scene-level clustered bootstrap confidence
intervals remain positive for both t+1 and t+2.

Reset-all removes the residue completely, while reset-BEV nearly eliminates the
map-level effect. In contrast, reset-query preserves most of the map-level
target-boundary residue while removing immediate query/prediction internal
differences. This indicates that BEV memory is the dominant channel for
geometry-level temporal residue, whereas query memory primarily carries
internal query/prediction residue.
```

## Output Files

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary/phase1_1_map_residue_summary.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary/phase1_1_primary_scene_map_residue_summary.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary/phase1_1_internal_reduction_summary.csv
/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check/summary/phase1_1_probe_analysis_summary.json
```
