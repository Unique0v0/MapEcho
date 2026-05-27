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

## Recommendation

Use a two-tier Phase 1.2 design:

```text
Main strict high-VPA subset:
  VPA >= 0.25
  25 frames / 9 scenes
  attack_power = 6000

Expanded statistical subset:
  VPA >= 0.15
  55 frames / 11 scenes
  attack_power = 6000
  report VPA-stratified results
```

This is more accurate than claiming that 40-60 strict high-VPA samples are
available. The strict subset tests the cleanest physical condition; the
expanded subset provides the scene coverage needed for clustered statistics.

