# Phase 1.3 Gate-Based Sample Construction

Date: 2026-05-27

## Purpose

Phase 1.3 formalizes when map-level temporal residue is expected to be
observable. It separates broad-pool robustness reporting from conditional
temporal-residue mechanism analysis.

Input:

```text
Phase 1.2 ccs_candidate expanded VPA>=0.15 ablation
55 frames / 11 scenes
W=10, L=9, attack_power=6000
```

Script:

```bash
bash scripts/build_phase1_3_sample_gates.sh
```

Output directory:

```text
/data/dj/MapEcho/artifacts/phase1_3_sample_gates
```

## Gate Table

Main output:

```text
/data/dj/MapEcho/artifacts/phase1_3_sample_gates/phase1_3_gate_table.csv
```

Each sample is annotated with:

```text
source_stage
W / L / attack_power
geometry gate
clean-correct gate
clean-stable gate
VPA gates at 0.15 / 0.20 / 0.25
attack-frame delta-CD gates at 0.005 / 0.01 / 0.02
recovery residue fields for attack_keep and reset variants
```

The current pre-attack quality thresholds are data-derived:

```text
clean_correct_threshold_p75 = 0.9875 m
clean_stable_threshold_p75  = 0.2564 m
geometry_tag_threshold      = 0.4
VPA threshold               = 0.15
```

## Constructed Sets

| Set | Frames | Scenes | Purpose |
| --- | ---: | ---: | --- |
| broad_report_set | 55 | 11 | Broad-pool robustness and unconditional effect |
| attack_effective_set_delta0005 | 22 | 9 | Attack-frame gate sensitivity |
| attack_effective_set_delta001 | 22 | 9 | Main conditional mechanism set |
| attack_effective_set_delta002 | 12 | 5 | Strong attack-frame gate sensitivity |
| high_quality_candidate_set | 12 | 4 | Conservative pre-attack quality set |

Token and asset files are written for each set:

```text
broad_report_set_tokens.txt
broad_report_set_assets.csv
attack_effective_set_delta0005_tokens.txt
attack_effective_set_delta0005_assets.csv
attack_effective_set_delta001_tokens.txt
attack_effective_set_delta001_assets.csv
attack_effective_set_delta002_tokens.txt
attack_effective_set_delta002_assets.csv
high_quality_candidate_set_tokens.txt
high_quality_candidate_set_assets.csv
```

## Set-Level Results

| Set | t+1 median delta CD | t+1 positive | t+2 median delta CD | t+2 positive |
| --- | ---: | ---: | ---: | ---: |
| broad_report_set | -0.0002 m | 10/55 | -0.0016 m | 9/55 |
| attack_effective_delta0005 | +0.0089 m | 8/22 | +0.0047 m | 8/22 |
| attack_effective_delta001 | +0.0089 m | 8/22 | +0.0047 m | 8/22 |
| attack_effective_delta002 | +0.0142 m | 7/12 | +0.0073 m | 5/12 |
| high_quality_candidate_set | +0.0023 m | 4/12 | -0.0001 m | 3/12 |

The `0.005` and `0.01` attack-effective gates currently select the same 22
samples. The `0.02` gate is stricter and gives a clearer median effect, but the
scene coverage drops to 5 scenes.

## Reset Summary

For `attack_effective_set_delta001`:

```text
attack_keep:
  t+1 positive = 8 / 22
  t+2 positive = 8 / 22

reset_all:
  t+1 positive = 0 / 22
  t+2 positive = 0 / 22

reset_BEV:
  t+1 positive = 1 / 22
  t+2 positive = 0 / 22

reset_query:
  t+1 positive = 11 / 22
  t+2 positive = 7 / 22
```

This reproduces the mechanism pattern: BEV reset removes map-level boundary
residue, while query reset does not.

## Interpretation

Phase 1.3 supports a three-layer interpretation:

```text
1. Broad-pool robustness:
   unconditional map-level effect is weak in the 55-frame expanded set.

2. Attack-effective conditional mechanism:
   when attack frame t has target-boundary corruption,
   t+1/t+2 attack-off residue becomes more visible.

3. Reset mechanism:
   reset-all and reset-BEV remove map-level residue,
   supporting BEV memory as the dominant geometry-level residue channel.
```

The conservative `high_quality_candidate_set` is currently too small for a main
statistical experiment:

```text
12 frames / 4 scenes
```

This means the next data step should be either:

```text
1. refine pre-attack geometry/clean-quality gates and inspect top/bottom cases;
2. add more candidates using stronger asymmetric filtering;
3. run true ETA search on promising candidates if heuristic placement is the bottleneck.
```

The current best main analysis set is:

```text
attack_effective_set_delta001:
  22 frames / 9 scenes
```

This set should be used for conditional temporal-residue mechanism analysis,
while `broad_report_set` should be reported as the broad-pool robustness result.

## Gate Drop-Off Diagnostic

The pre-attack high-quality set is small, so we diagnose which gates remove the
most samples.

Script:

```bash
bash scripts/diagnose_phase1_3_gate_dropoff.sh
```

Output directory:

```text
/data/dj/MapEcho/artifacts/phase1_3_sample_gates/dropoff
```

Single-gate pass counts:

| Gate | Pass frames | Pass scenes | Fail frames | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| geometry | 25 | 6 | 30 | 0.455 |
| clean_correct | 41 | 10 | 14 | 0.745 |
| clean_stable | 41 | 9 | 14 | 0.745 |
| VPA>=0.15 | 55 | 11 | 0 | 1.000 |

Cumulative drop-off:

| Step | Frames | Scenes | Dropped from previous |
| --- | ---: | ---: | ---: |
| geometry | 25 | 6 | 30 |
| geometry + clean_correct | 16 | 5 | 9 |
| geometry + clean_correct + clean_stable | 12 | 4 | 4 |
| geometry + clean_correct + clean_stable + VPA | 12 | 4 | 0 |

Thus, the current bottleneck is not VPA. It is primarily the geometry gate,
followed by clean correctness and clean stability.

Relaxation scenarios:

| Scenario | Frames | Scenes | t+1 positive | t+1 median delta CD | t+2 positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_gates | 12 | 4 | 4/12 | +0.0023 m | 3/12 |
| drop_geometry | 33 | 9 | 5/33 | -0.0001 m | 3/33 |
| drop_clean_correct | 16 | 4 | 5/16 | +0.0014 m | 4/16 |
| drop_clean_stable | 16 | 5 | 5/16 | +0.0000 m | 3/16 |
| geometry_vpa_only | 25 | 6 | 6/25 | +0.0002 m | 5/25 |
| vpa_only | 55 | 11 | 10/55 | -0.0002 m | 9/55 |

This shows that simply dropping the geometry gate expands the set to 33 frames
and 9 scenes, but it weakens the map-level signal. Therefore, the geometry gate
should not be blindly removed. Instead, the current tag-confidence threshold
should be treated as a weak feature and refined with source-geometry and visual
diagnostics.

One notable failure-pattern group is:

```text
fail:geometry,clean_stable
  4 frames / 3 scenes
  t+1 positive = 3 / 4
  t+2 positive = 3 / 4
```

This means some samples rejected by the current geometry gate are actually
attack-effective. The current geometry gate is therefore too crude if it relies
strongly on tag confidence.

Updated next step:

```text
1. Keep attack_effective_set_delta001 as the current conditional mechanism set.
2. Keep broad_report_set as the broad-pool robustness set.
3. Refine pre-attack geometry quality instead of simply relaxing the geometry gate.
4. Inspect top/bottom cases and add source-geometry features before trying true ETA search.
```
