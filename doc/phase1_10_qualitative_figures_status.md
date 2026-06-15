# Phase 1.10 Qualitative Figures Status

## Status

```text
Phase 1.10 qualitative figure assembly: PASS
```

This stage converts the Phase 1.9 qualitative case list into paper-facing
figure assets. It does not rerun model inference. It reuses existing selected114
outputs:

```text
six-camera target-frame contact sheet
map overlay at t
map overlay at t+1
map overlay at t+2
```

## Input

```text
case_csv:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence/cases/qualitative_case_selection.csv

run_root:
  /data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check
```

## Output

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_10_qualitative_figures/
  phase1_10_qualitative_contact_sheet.png
  phase1_10_qualitative_figure_index.md
  phase1_10_qualitative_figure_manifest.csv
  phase1_10_qualitative_summary.json
  figures/
  case_assets/
```

Generation completed for all default cases:

```text
num_cases = 15
num_with_six_camera = 15
num_with_map_t0 = 15
num_with_map_t1 = 15
num_with_map_t2 = 15
```

## Regeneration Command

```bash
bash scripts/assemble_phase1_10_qualitative_figures.sh
```

Optional overrides:

```bash
CASE_CSV=/path/to/cases.csv \
RUN_ROOT=/path/to/run_root \
OUT_DIR=/path/to/out \
MAX_CASES=15 \
bash scripts/assemble_phase1_10_qualitative_figures.sh
```

## Case Groups

The default case list comes from Phase 1.9:

```text
top_residue: 5
median_residue: 2
reset_bev_clear_removal: 3
weak_or_failure: 5
```

The intended manual step after assembly is to inspect the generated contact
sheet and pick the final 3-5 panels for the paper.

## Manual Review Note

Manual review on 2026-06-15 found that many default Phase 1.10 panels are not
ideal paper examples. In several cases, the five map columns look visually
similar because the original overlay draws multiple high-score boundary
predictions in the same light color and does not highlight the exact
metric-selected boundary used by the CD calculation. Some cases also have
clean predictions that are already visibly imperfect, making them poor
qualitative examples even when their numerical reset pattern is valid.

Focused diagnostic panels were added to address this:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_10_qualitative_figures/focused_map_panels/
  focused_map_panel_metrics.csv
  focused_map_panel_paths.txt
  *_focused.png
```

For the manually accepted top-residue #3 case:

```text
target_token = 4a1972f8731b4cdea40fc69a38a735b1
t+1 attack_keep CD = 5.7014
t+1 clean_keep CD = 5.0747
t+1 attack_keep delta = +0.6267
t+1 attack_reset_all delta = 0.0000
t+1 attack_reset_BEV delta = -0.0041
t+1 attack_reset_query delta = +0.5530
```

This confirms that the numerical reset pattern is real, while the original
qualitative overlay is visually under-informative. Final paper figures should
use focused panels and manually selected cases, not the unfiltered contact
sheet alone.
