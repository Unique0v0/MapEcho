# Phase 1.4 Geometry-Gate Refinement

Date: 2026-06-03

## Purpose

Phase 1.4 refines the pre-attack geometry gate and expands the high-quality
candidate set without running additional model inference.

Input:

```text
Phase 1.3 gate table
Phase 1.2 expanded VPA>=0.15 asset table
```

Script:

```bash
bash scripts/build_phase1_4_geometry_gate_v2.sh
```

Output directory:

```text
/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2
```

## Outputs

```text
geometry_quality_v2_table.csv
high_quality_strict_v2_tokens.txt
high_quality_strict_v2_assets.csv
high_quality_relaxed_v2_tokens.txt
high_quality_relaxed_v2_assets.csv
geometry_false_negative_cases.csv
geometry_false_positive_cases.csv
phase1_4_geometry_v2_set_summary.csv
phase1_4_geometry_v2_summary.json
```

## Geometry v2 Rule

The old geometry gate relied too strongly on `tag_confidence >= 0.4`. Phase
1.3 showed that this is too crude: some low-tag-confidence samples are
attack-effective, while some high-tag-confidence samples are not.

Geometry v2 treats tag confidence as a weak feature and combines it with
source-geometry features:

```text
tag_confidence
asymmetry_score
centrality_score
visible camera
```

The v2 rule is:

```text
VPA >= 0.15
camera != CAM_FRONT_RIGHT

strict:
  clean_correct
  and clean_stable
  and moderate geometry signal

relaxed:
  (clean_correct or clean_stable)
  and weak geometry signal
```

Weak geometry signal:

```text
tag_confidence >= 0.15
or asymmetry_score >= 0.10
or centrality_score >= 0.70
```

Moderate geometry signal:

```text
tag_confidence >= 0.20
or asymmetry_score >= 0.12
or centrality_score >= 0.75
```

The rule does not use attack-frame success as a selection criterion. Attack
outcomes are used only for post-hoc validation.

## Set Results

| Set | Frames | Scenes | t+1 positive | t+1 median delta CD | t+2 positive | t+2 median delta CD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broad_report_set | 55 | 11 | 10/55 | -0.0002 m | 9/55 | -0.0016 m |
| high_quality_strict_v2 | 20 | 8 | 5/20 | +0.0011 m | 3/20 | -0.0017 m |
| high_quality_relaxed_v2 | 34 | 10 | 9/34 | +0.0002 m | 7/34 | -0.0002 m |

Acceptance target:

```text
high_quality_relaxed_v2 >= 25 frames / 8 scenes
```

Result:

```text
PASS: 34 frames / 10 scenes
```

## Conditional Residue Check

To check that the relaxed set does not obviously dilute the conditional
temporal-residue signal, we evaluate the attack-effective subset inside each
set using the Phase 1.3 gate:

```text
attack-frame delta CD > 0.01 m
```

| Set | Attack-effective frames | Scenes | t+1 positive | t+1 median delta CD | t+2 positive | t+2 median delta CD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broad_report_set | 22 | 9 | 8/22 | +0.0089 m | 8/22 | +0.0047 m |
| high_quality_strict_v2 | 9 | 5 | 3/9 | +0.0088 m | 2/9 | +0.0035 m |
| high_quality_relaxed_v2 | 17 | 8 | 7/17 | +0.0091 m | 6/17 | +0.0048 m |

The relaxed v2 set preserves the conditional trend:

```text
attack-effective relaxed_v2:
  17 frames / 8 scenes
  t+1 positive = 7 / 17
  t+2 positive = 6 / 17
```

This is consistent with Phase 1.3 and does not show obvious dilution of the
conditional temporal-residue effect.

## False Negative / False Positive Diagnostics

Old geometry false negatives:

```text
4 frames / 3 scenes
t+1 positive = 3 / 4
t+2 positive = 4 / 4
```

These are samples rejected by the old geometry gate but showing attack-frame
corruption and recovery residue. They confirm that the old tag-confidence-heavy
gate was too strict.

Old geometry false positives:

```text
12 frames / 4 scenes
t+1 positive = 0 / 12
t+2 positive = 0 / 12
```

These are samples passing the old geometry gate but showing neither
attack-frame corruption nor recovery residue. They confirm that tag confidence
alone is not sufficient.

## Current Status

Phase 1.4 meets the proposed acceptance criterion:

```text
high_quality_relaxed_v2:
  34 frames / 10 scenes
```

The current recommended sets are:

```text
broad_report_set:
  report broad-pool robustness

attack_effective_set_delta001:
  conditional mechanism analysis

high_quality_relaxed_v2:
  pre-attack high-quality candidate pool for the next controlled run
```

Next decision:

```text
Use high_quality_relaxed_v2 for the next controlled experiment,
or inspect geometry_false_negative / false_positive cases visually before running.
```
