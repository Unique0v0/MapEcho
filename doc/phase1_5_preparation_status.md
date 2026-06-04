# Phase 1.5 Controlled Experiment Preparation

Date: 2026-06-03

## Purpose

Phase 1.5 will run a controlled experiment on the frozen
`high_quality_relaxed_v2` set from Phase 1.4.

Before running the model, we prepare a lightweight visual audit package for:

```text
old geometry false negatives: 4
old geometry false positives: 12
top residue cases: 5
bottom failure cases: 5
```

No model inference is run during this preparation step.

## Visual Audit

Script:

```bash
bash scripts/prepare_phase1_5_visual_audit.sh
```

Output directory:

```text
/data/dj/MapEcho/artifacts/phase1_5_visual_audit
```

Outputs:

```text
visual_audit_manifest.csv
old_geometry_false_negative_contact_sheet.png
old_geometry_false_positive_contact_sheet.png
top_residue_t1_contact_sheet.png
bottom_failure_t1_contact_sheet.png
phase1_5_visual_audit_summary.json
```

The manifest contains 26 rows:

```text
old_geometry_false_negative: 4
old_geometry_false_positive: 12
top_residue_t1: 5
bottom_failure_t1: 5
```

Each row includes:

```text
attack image overlay
map overlay at t
map overlay at t+1
map overlay at t+2
scene JSON path
VPA / geometry / delta-CD metrics
```

## Frozen Phase 1.5 Input

Frozen input directory:

```text
/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment
```

Frozen files:

```text
phase1_5_high_quality_relaxed_v2_tokens.txt
phase1_5_high_quality_relaxed_v2_assets.csv
```

Current frozen set:

```text
34 frames / 10 scenes
W=10, L=9
attack_power=6000
```

This set is copied from:

```text
/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_tokens.txt
/data/dj/MapEcho/artifacts/phase1_4_geometry_gate_v2/high_quality_relaxed_v2_assets.csv
```

## Phase 1.5 Run Commands

Run the controlled experiment after visual audit:

```bash
bash scripts/run_phase1_5_controlled_experiment.sh
```

Summarize results:

```bash
bash scripts/summarize_phase1_5_controlled_experiment.sh
```

Output root:

```text
/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment/high_quality_relaxed_v2_ablation_power6000
```

## Evaluation Targets

Phase 1.5 should evaluate:

```text
1. broad high-quality-set unconditional effect
2. attack-effective subset conditional residue
3. reset_all / reset_BEV map-level removal
4. reset_query internal-only effect
```

The expected interpretation is:

```text
high_quality_relaxed_v2:
  broad high-quality robustness set

attack-frame delta CD > 0.01 within high_quality_relaxed_v2:
  conditional temporal-residue mechanism subset
```
