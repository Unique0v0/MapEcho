# Phase 1.13 H5 Continuous Perturbation Pilot

## Goal

Phase 1.13 is a small stress-test branch for H5 temporal accumulation. It does
not replace the single-frame main experiment. The main question is:

```text
Does a short N=3 camera-glare perturbation produce stronger post-perturbation
recovery residue than the existing N=1 setting, and do reset_all / reset_BEV
still remove map-level residue?
```

## Design

```text
N_attack in {1, 3}
main single-frame baseline: existing selected114 N=1 outputs
pilot N=3 samples: primary-scene or qualitative-friendly selected114 subset
renderer: CCS-style six-camera renderer
power: 3000
camera mode: all six cameras
```

For `N_attack=3`, the perturbation schedule is:

```text
t, t+1, t+2: perturbed
t+3 ... t+9: clean recovery
```

Therefore `t+1` and `t+2` are not post-perturbation recovery frames for N=3.
They are in-perturbation damage frames. Recovery AUC for N=3 should start at
`t+3`.

Reset timing:

```text
N=1: reset before t+1, reset_after_offset = 0
N=3: reset before t+3, reset_after_offset = 2
```

Matched clean baselines must use the same reset timing as their paired
perturbed condition.

## Implementation

New files:

```text
scripts/build_multi_frame_glare_sequence_ann.py
scripts/select_phase1_13_h5_pilot_tokens.py
scripts/select_phase1_13_h5_pilot_tokens.sh
scripts/run_phase1_13_h5_continuous_pilot.sh
scripts/package_phase1_13_h5_continuous_pilot.sh
```

`build_multi_frame_glare_sequence_ann.py` fixes the glare source in the global
coordinate system once, then transforms that same world point into each current
frame's LiDAR coordinate before calling the CCS six-camera renderer. This keeps
the physical source stationary in the world rather than moving with the ego
vehicle.

## Suggested Commands

Select pilot tokens:

```bash
bash scripts/select_phase1_13_h5_pilot_tokens.sh
```

Run N=3 on the primary-scene pilot tokens:

```bash
bash scripts/run_phase1_13_h5_continuous_pilot.sh
```

Package N=3 recovery summaries over true post-perturbation recovery frames
`t+3...t+9`:

```bash
bash scripts/package_phase1_13_h5_continuous_pilot.sh
```

Optional qualitative-friendly run:

```bash
TOKENS_FILE=/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/h5_qualitative_friendly_pilot_tokens.txt \
OUT_ROOT=/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/n_attack3_qualitative_friendly_controlled_check \
bash scripts/run_phase1_13_h5_continuous_pilot.sh
```

## Metrics

Primary:

```text
AUC_CD over post-perturbation recovery window
median Delta CD at first recovery step
positive rate at first recovery step
reset_all median AUC_CD
reset_BEV median AUC_CD
```

N=1 and N=3 should be compared by recovery step:

```text
N=1 recovery step 1 = t+1
N=3 recovery step 1 = t+3
```

For a fair common-window comparison, use seven recovery steps:

```text
N=1: t+1...t+7
N=3: t+3...t+9
```

## Interpretation

Strong H5 pilot signal:

```text
N=3 attack_keep recovery AUC_CD > N=1 over the common recovery window
N=3 qualitative maps are visibly clearer than N=1
reset_all and reset_BEV still reduce map-level residue near zero
reset_query mainly affects internal query/prediction residue
```

If N=3 is stronger, present it as:

```text
continuous perturbation stress test
```

The main experiment should remain the single-frame setting.

## Result: Primary-Scene Pilot

Run completed on:

```text
tokens = 38 frames / 38 scenes
N_attack = 3
perturbed offsets = t, t+1, t+2
post-perturbation recovery offsets = t+3...t+9
```

Outputs:

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_13_h5_continuous_pilot/
  n_attack3_controlled_check/
  n_attack3_recovery_summary/
  n_attack1_primary38_common7_summary/
```

N=3 post-perturbation recovery curve:

| Offset | attack_keep median Delta CD | Positive Count |
| ---: | ---: | ---: |
| t+3 | +0.0105 m | 19/38 |
| t+4 | +0.0109 m | 22/38 |
| t+5 | +0.0030 m | 16/38 |
| t+6 | +0.0020 m | 15/38 |
| t+7 | +0.0003 m | 11/38 |
| t+8 | +0.0019 m | 14/38 |
| t+9 | -0.0009 m | 8/38 |

N=3 AUC over `t+3...t+9`:

| Condition | Median AUC_CD | AUC > 0.03 | AUC > 0.05 | AUC > 0.10 |
| --- | ---: | ---: | ---: | ---: |
| attack_keep | 0.0976 | 25/38 | 22/38 | 18/38 |
| attack_reset_all | 0.0000 | 0/38 | 0/38 | 0/38 |
| attack_reset_BEV | 0.0051 | 4/38 | 3/38 | 1/38 |
| attack_reset_query | 0.0812 | 24/38 | 22/38 | 15/38 |

Matched N=1 common-window baseline on the same 38 tokens, `t+1...t+7`:

| Condition | Median AUC_CD | AUC > 0.03 | AUC > 0.05 | AUC > 0.10 |
| --- | ---: | ---: | ---: | ---: |
| attack_keep | 0.1546 | 34/38 | 31/38 | 22/38 |
| attack_reset_all | 0.0000 | 0/38 | 0/38 | 0/38 |
| attack_reset_BEV | 0.0031 | 3/38 | 0/38 | 0/38 |
| attack_reset_query | 0.1474 | 33/38 | 32/38 | 25/38 |

## Interpretation

The primary-scene H5 pilot does not support a simple nonlinear-amplification
claim. On the same 38 tokens and a common seven-step recovery window, N=3 has a
lower median post-perturbation AUC than N=1:

```text
N=1 attack_keep median AUC_CD = 0.1546
N=3 attack_keep median AUC_CD = 0.0976
```

However, the reset mechanism remains consistent:

```text
reset_all removes map-level residue completely
reset_BEV nearly removes map-level residue
reset_query preserves most map-level residue
```

Therefore the current H5 conclusion should be:

```text
Continuous N=3 perturbation is useful as a stress-test, but it should not be
claimed as stronger than the single-frame setting on the primary-scene pilot.
The main result should remain the single-frame selected114 experiment.
```

The next optional step is qualitative inspection of N=3 panels only if we want
to check whether some individual cases become visually clearer despite the
aggregate AUC being weaker.
