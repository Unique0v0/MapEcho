# Phase 1.0 Overlap Sanity Result

Date: 2026-05-27

## Status

Phase 1.0 overlap sanity: PASS.

This subset uses 5 CCS-aligned samples that are also in the official StreamMapNet newsplit validation split. It is used as an engineering and mechanism sanity set, not as the final main evaluation set.

## Evidence

- 5 / 5 samples completed.
- Internal causal temporal-state evidence replicated.
- Map-level target-boundary residue observed on 4 / 5 samples.
- `reset_all` removes both internal and map-level residue.
- `reset_BEV` removes target-boundary geometry residue almost completely.
- `reset_query` removes immediate query/pred internal residue but does not remove map-level boundary residue.

## Key Results

Map-level target-boundary Chamfer delta to diverging boundary:

| Condition | t+1 median | t+2 median |
| --- | ---: | ---: |
| `attack_keep` | +0.0517 m | +0.0307 m |
| `attack_reset_all` | 0.0000 m | 0.0000 m |
| `attack_reset_query` | +0.0684 m | +0.0348 m |
| `attack_reset_BEV` | -0.0002 m | -0.0000 m |

Internal t+1 matched reset reductions:

| Reset | Query score median | Pred vector median | Fused BEV median |
| --- | ---: | ---: | ---: |
| `reset_all` | 1.000 | 1.000 | 1.000 |
| `reset_query` | 0.948 | 0.986 | 0.000 |
| `reset_BEV` | 0.568 | 0.562 | 1.000 |

## Interpretation

The 5-sample overlap subset provides preliminary causal evidence that a one-frame physical perturbation can leave attack-off temporal residue in StreamMapNet. The residue is not caused by current-frame input at t+1/t+2 because `reset_all` removes it.

The mechanism is two-level:

- Query memory primarily carries immediate query-score and predicted-vector internal residue.
- BEV memory dominates map-level diverging-boundary geometry residue.

Subsequent Phase 1 experiments should report internal query/pred residue and map-level target-boundary residue separately.

## Artifacts

- Summary root: `/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation/summary`
- Internal matched baseline summary: `/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation/summary/overlap_internal_matched_baseline_summary.csv`
- Internal matched reductions summary: `/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation/summary/overlap_internal_matched_reductions_summary.csv`
- Map-level matched deltas summary: `/data/dj/MapEcho/artifacts/phase1_0_overlap_mini_ablation/summary/overlap_map_matched_deltas_summary.csv`

