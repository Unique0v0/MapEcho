# Phase 1.2 High-VPA Expansion Status

Date: 2026-05-27

## Goal

Expand Phase 1.1 from the `ccs_asymmetric_dist` probe set to a broader
newsplit-val pool with better scene coverage:

```text
source = ccs_candidate W10/L9
target = 40-60 frames / 10-15 scenes
main attack power = 6000
```

The key filter is target-boundary VPA coverage, because Phase 1.1 showed that
map-level residue is much clearer on high-VPA samples.

## Scripts

```bash
bash scripts/build_phase1_2_ccs_candidate_assets.sh
bash scripts/run_phase1_2_ccs_candidate_vpa_sanity.sh
bash scripts/select_phase1_2_high_vpa_assets.sh
```

The selector keeps a strict high-VPA set at `VPA >= 0.25` and a medium reserve
at `0.20 <= VPA < 0.25`.

## Candidate And VPA Results

The `ccs_candidate W10/L9` pool produced:

```text
assets = 243 frames / 15 scenes
median tag confidence = 0.335
```

Target-boundary VPA sanity:

```text
diverge boundary:
  visible        = 243 / 243
  on-boundary    = 243 / 243
  VPA pass       = 176 / 243
  median VPA     = 0.062

reference boundary:
  VPA pass       = 0 / 243
```

This keeps the attack placement target-specific: it covers the diverging
boundary while not covering the reference boundary.

## Threshold Tradeoff

| VPA threshold | Frames | Scenes |
| ---: | ---: | ---: |
| 0.30 | 23 | 9 |
| 0.25 | 25 | 9 |
| 0.20 | 32 | 11 |
| 0.15 | 55 | 11 |
| 0.10 | 68 | 11 |
| 0.05 | 176 | 15 |

The original strict high-VPA target of 40-60 frames at `VPA >= 0.25` is not
reachable with the current ETA-like heuristic locations. The closest strict
set is:

```text
VPA >= 0.25:
  25 frames / 9 scenes
  median VPA = 0.418
  min VPA    = 0.264
```

An expanded VPA-stratified set is available:

```text
VPA >= 0.15:
  55 frames / 11 scenes
  median VPA = 0.233
  min VPA    = 0.152
```

## Outputs

Strict high-VPA set:

```text
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_assets.csv
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_tokens.txt
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_2_high_vpa_selection_summary.json
```

Expanded VPA-stratified set:

```text
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_assets.csv
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_tokens.txt
/data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_vpa015_expanded/phase1_2_high_vpa_selection_summary.json
```

## Ablation Results

Both Phase 1.2 candidate sets were run with `attack_power = 6000` and matched
clean-reset baselines.

Strict high-VPA set:

```text
25 frames / 9 scenes

attack_keep:
  t+1 median delta CD = +0.0020 m, positive = 6 / 25
  t+2 median delta CD = -0.0020 m, positive = 6 / 25

reset_all:
  t+1/t+2 positive = 0 / 25

reset_BEV:
  t+1 positive = 1 / 25
  t+2 positive = 0 / 25
```

Expanded VPA-stratified set:

```text
55 frames / 11 scenes

attack_keep:
  t+1 median delta CD = -0.0002 m, positive = 10 / 55
  t+2 median delta CD = -0.0016 m, positive = 9 / 55

reset_all:
  t+1/t+2 positive = 0 / 55

reset_BEV:
  t+1 positive = 1 / 55
  t+2 positive = 0 / 55
```

The internal reset mechanism remains stable:

```text
reset_query at t+1:
  query-score reduction median = 0.944
  pred-vector reduction median = 0.986
  fused-BEV reduction          = 0

reset_BEV:
  fused-BEV reduction          = 1.0
```

Thus Phase 1.2 confirms the temporal-state mechanism, but the broader
`ccs_candidate` pool does not provide a strong unconditional map-level main
effect. VPA alone is not sufficient to define an attack-effective evaluation
set from this broader pool.

## Asset-Quality Diagnostic

The diagnostic script:

```bash
bash scripts/diagnose_phase1_2_asset_quality.sh
```

produces:

```text
/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_asset_quality_by_group.csv
/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_asset_quality_stratified.csv
/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_scene_quality_summary.csv
/data/dj/MapEcho/artifacts/phase1_2_asset_quality_diagnostics/phase1_2_top_bottom_failure_cases.csv
```

The key diagnostic result is the attack-frame gate. In the expanded set:

```text
all expanded:
  55 frames / 11 scenes
  attack-frame median delta CD = -0.0002 m
  t+1 positive = 10 / 55
  t+2 positive = 9 / 55

attack-effective at t:
  condition: attack-frame delta CD > 0.01 m
  22 frames / 9 scenes
  t+1 positive = 8 / 22
  t+2 positive = 8 / 22

attack-weak at t:
  33 frames / 10 scenes
  t+1 positive = 2 / 33
  t+2 positive = 1 / 33
```

The same pattern appears in the strict set:

```text
all strict:
  25 frames / 9 scenes
  t+1 positive = 6 / 25
  t+2 positive = 6 / 25

attack-effective at t:
  12 frames / 7 scenes
  t+1 positive = 5 / 12
  t+2 positive = 5 / 12

attack-weak at t:
  13 frames / 7 scenes
  t+1 positive = 1 / 13
  t+2 positive = 1 / 13
```

This shows that map-level temporal residue should be analyzed conditionally on
attack-frame corruption. The full-set result remains necessary as a robustness
report, but it should not be interpreted as the main mechanism estimate.

## Updated Recommendation

Use a three-tier Phase 1.2 interpretation:

```text
Full ccs_candidate-derived evaluation:
  Report as broad-pool robustness.
  Conclusion: mechanism persists, but unconditional map-level effect is weak.

Attack-effective subset:
  Gate: attack-frame delta CD > 0.01 m.
  Use for conditional temporal-residue mechanism analysis.

High-quality future subset:
  Gate should combine:
    clear asymmetric geometry,
    clean-correct boundary,
    target-boundary VPA,
    attack-frame boundary corruption.
```

The current evidence supports the conditional chain:

```text
asymmetric vulnerable geometry
+ clean-correct boundary
+ effective target-side perturbation
-> attack-frame boundary corruption
-> BEV-dominant temporal residue
-> reset-BEV/reset-all removes residue
```

The next step should be sample-quality refinement rather than more blind
ablation on the broad `ccs_candidate` pool.
