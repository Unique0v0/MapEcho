# MapEcho Experiment Plan v11: Newsplit Temporal Route

## Core Decision

The main experimental route is switched to the official StreamMapNet temporal
NuScenes newsplit checkpoint.

```text
Main split  = NuScenes newsplit
Main model  = nusc_newsplit_480_60x30_24e.pth
Main data   = asymmetric temporal candidates rebuilt on newsplit val
```

The old CCS'25 oldsplit seed pool remains useful for engineering sanity and
attack-rendering validation, but it is not the main evaluation set for the
official newsplit temporal checkpoint.

## Why This Change Is Needed

The oldsplit baseline checkpoint is non-streaming and cannot support H1/H3
temporal-residue experiments. The downloaded newsplit checkpoint is
temporal-ready:

```text
streaming query config       = true
streaming BEV config         = true
checkpoint head.query_update = present
checkpoint stream_fusion_neck= present
can run clean hook sanity    = true
```

However, the CCS'25 seed pool was built on oldsplit validation scenes. Under
newsplit, most of these samples belong to training scenes:

```text
CCS'25 100 seeds:
  newsplit train = 88 frames / 26 scenes
  newsplit val   = 12 frames / 5 scenes

old temporal-eligible W=10/L=19 subset:
  newsplit train = 26 frames / 11 scenes
  newsplit val   = 7 frames / 3 scenes

old Phase 1 subset:
  newsplit train = 15 frames / 11 scenes
  newsplit val   = 5 frames / 3 scenes
```

Therefore, using all oldsplit CCS'25 seeds with the official newsplit checkpoint
would be split-mixed and should only be treated as diagnostic.

## Phase Definitions

### Phase 1.0: Temporal Hook Sanity

Purpose:

```text
Validate model-level temporal plumbing before rebuilding newsplit candidates.
```

Data:

```text
old Phase 1 ∩ newsplit val overlap
5 frames / 3 scenes
```

Use this phase for:

```text
clean_keep
clean_reset_all
clean_reset_query
clean_reset_BEV
single-sequence attack_keep dry run
```

Artifacts:

```text
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_tokens.txt
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_selection.csv
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/newsplit_overlap_summary.json
```

The first overlap target has also been expanded into a 30-frame clean_keep
debug annotation:

```text
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann.pkl
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep_one_sequence_ann_summary.json
```

Use this config for the first clean_keep hook sanity run:

```text
/home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_phase1_0_clean_keep_debug.py
```

Expected command in a GPU-capable StreamMapNet environment:

```bash
cd /home/dj/MapEcho/src/StreamMapNet
export MPLCONFIGDIR=/tmp/mapecho_matplotlib
export PYTHONPATH=/home/dj/physical-online-map-attack:/home/dj/MapEcho/src/StreamMapNet:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/dj/.conda/envs/maptr4090/lib:${LD_LIBRARY_PATH:-}
/home/dj/.conda/envs/maptr4090/bin/python tools/test.py \
  plugin/configs/mapecho_nusc_newsplit_phase1_0_clean_keep_debug.py \
  /home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth \
  --format-only \
  --work-dir /data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep \
  --eval-options prefix=/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_clean_keep
```

Equivalent wrapper:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_0_clean_keep.sh
```

Environment note: `maptr` sees CUDA but fails on the old mmdet3d ABI.
`maptr4090` uses the CCS-modified mmdetection3d package, whose
`mmdet3d.apis.__init__` imports attack entrypoints. Therefore the CCS repository
root must be present in `PYTHONPATH` so that `attack_toolkit` resolves.

First clean_keep run result:

```text
inference frames = 30 / 30
query dumps      = 30 / 30
BEV dumps        = 30 / 30
query first frame count = 1
BEV first frame count   = 1
propagated query mask   = 0 on first frame, 33 on following frames
BEV warped history      = absent on first frame, present on 29 following frames
NaN / Inf               = none observed
```

The run reached the final submission-formatting step and initially failed only
because `format_results` needs a `prefix`. The command above now supplies the
prefix via `--eval-options`.

### Phase 1.1: Newsplit Candidate Probe

Purpose:

```text
Test H1/H3 on a small newsplit-val asymmetric candidate set.
```

Data funnel:

```text
NuScenes newsplit val frames
  -> CCS'25-style asymmetric candidate selection
  -> temporal eligibility W=10/L=19
  -> ETA attack-location generation
  -> target-boundary VPA sanity
  -> clean-quality filter
  -> Phase 1.1 mini probe
```

### Main Experiment

Use all final valid newsplit-val temporal candidates after clean-quality and VPA
filters. If multiple target frames come from the same scene, report clustered
scene-level bootstrap confidence intervals.

## Role of Oldsplit CCS'25 Assets

Oldsplit CCS'25 assets are now scoped as:

```text
engineering sanity set
attack-rendering validation set
optional overlap sanity subset
```

They should not be described as the main newsplit evaluation set unless an
oldsplit temporal StreamMapNet checkpoint is obtained or trained.

## Attack Location Strategy

For newsplit-val candidates:

1. Prefer reusing the CCS'25 attack-location search code to generate ETA best
   locations.
2. Use RSA best locations only as a supplementary subset if available.
3. If search is too slow, use an ETA-like geometry heuristic only for Phase 1.1
   probing and clearly label it as heuristic.

## Immediate Next Steps

1. Run Phase 1.0 clean hook sanity with the newsplit temporal wrapper config.
2. Run reset sanity on 1-2 overlap sequences.
3. Run one attack-at-t dry run on an overlap sequence.
4. Rebuild asymmetric candidates on newsplit val.
5. Generate ETA attack locations and repeat target-boundary VPA sanity.

## Reset Sanity Runner

Run the Phase 1.0 reset sanity sweep with:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_0_reset_sanity.sh
```

The runner uses the same 30-frame clean sequence as clean_keep and applies reset
after target frame `t` and before `t+1`:

```text
warm-up clean
t clean
reset before t+1
t+1... recovery clean
```

Outputs:

```text
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_reset_sanity/reset_all/reset_sanity_summary.json
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_reset_sanity/reset_query/reset_sanity_summary.json
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_reset_sanity/reset_bev/reset_sanity_summary.json
```

Pass criteria at the first post-reset frame (`t+1`):

| Mode | Query propagated after reset | Warped BEV after reset |
| --- | ---: | --- |
| reset_all | 0 | absent |
| reset_query | 0 | present |
| reset_BEV | 33 | absent |

Observed Phase 1.0 result:

| Mode | Query dumps | BEV dumps | Prop before | Prop after | Warped after | Pass |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| reset_all | 30/30 | 30/30 | 33 | 0 | absent | true |
| reset_query | 30/30 | 30/30 | 33 | 0 | present | true |
| reset_BEV | 30/30 | 30/30 | 33 | 33 | absent | true |

The reset controls are selective: query reset does not clear BEV history, BEV
reset does not clear propagated queries, and reset-all clears both.

## Attack-At-T Dry Run

Run the first clean attack-at-t dry run with:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_0_attack_keep.sh
```

This builds a 30-frame annotation in which only the target frame `t` uses an
attacked raw image. Warm-up and recovery frames remain clean.

Generated attack annotation and image artifacts:

```text
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep_one_sequence_ann.pkl
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep/attack_at_t_ann_summary.json
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_keep/images/
```

For the current overlap sequence:

```text
target token       = 6e147994a5e3493d86a928a612ff5791
target sample_idx  = 572
attack objective   = ETA
attack camera      = CAM_FRONT_LEFT
N_attack           = 1
```

After the model run, summarize clean vs attack effects with:

```bash
cd /home/dj/MapEcho
bash scripts/summarize_phase1_0_attack_dry_run.sh
```

The dry-run pass criteria are:

```text
attack query dumps       = 30 / 30
attack BEV dumps         = 30 / 30
N_attack                 = 1
target-frame query delta > 0
target-frame BEV delta   > 0
t+1 input schedule       = clean
NaN / Inf                = none
```

Observed Phase 1.0 attack-at-t dry run:

```text
attack query dumps       = 30 / 30
attack BEV dumps         = 30 / 30
submission saved         = yes
N_attack                 = 1
pass                     = true
```

Clean vs attack deltas:

| Frame | Query score mean_abs | Query score max_abs | Pred vector mean_abs | Top-k embedding mean_abs | Current BEV norm delta | Fused BEV norm delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t | 0.414945 | 3.979524 | 0.126144 | 0.369988 | 0.078979 | 0.184998 |
| t+1 clean input | 0.467813 | 3.854012 | 0.178403 | 0.347421 | 0.000000 | 0.015320 |

The `t+1` input schedule is clean, so the nonzero `t+1` query and fused-BEV
deltas are preliminary temporal-residue evidence for the next mini probe.

## Attack Reset Ablation

Run the single-sequence attack reset ablation with:

```bash
cd /home/dj/MapEcho
bash scripts/run_phase1_0_attack_reset_ablation.sh
```

This runs:

```text
attack_keep
attack_reset_all
attack_reset_query
attack_reset_bev
```

All reset modes use the same timing:

```text
attack at t
reset before t+1
t+1... clean recovery
```

Summarize internal differences and reset reduction ratios with:

```bash
cd /home/dj/MapEcho
bash scripts/summarize_phase1_0_attack_reset_ablation.sh
```

Outputs:

```text
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation/phase1_0_single_sequence_reset_ablation_summary.csv
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation/phase1_0_single_sequence_reset_ablation_reductions.csv
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/phase1_0_attack_reset_ablation/phase1_0_single_sequence_reset_ablation_summary.json
```

Primary readout at `t+1`:

```text
Reduction_reset_query =
  1 - diff(attack_reset_query, clean_keep) / diff(attack_keep, clean_keep)

Reduction_reset_BEV =
  1 - diff(attack_reset_BEV, clean_keep) / diff(attack_keep, clean_keep)

Reduction_reset_all =
  1 - diff(attack_reset_all, clean_keep) / diff(attack_keep, clean_keep)
```

Interpretation:

```text
near 1  = reset strongly removes attack-off difference
near 0  = reset has little effect
below 0 = reset increases difference, inspect reset distribution shift
```

Observed result and analysis note:

```text
All four attack conditions completed with 30 / 30 query dumps and 30 / 30 BEV
dumps. Direct comparison of reset conditions to clean_keep shows large negative
reductions because reset itself changes the state distribution. The causal
readout should therefore use matched clean-reset baselines:

attack_keep        vs clean_keep
attack_reset_all   vs clean_reset_all
attack_reset_query vs clean_reset_query
attack_reset_BEV   vs clean_reset_BEV
```

Matched-baseline t+1 reduction ratios:

| Condition | Query score | Pred vector | Top-k embedding | Fused BEV |
| --- | ---: | ---: | ---: | ---: |
| reset_all | 1.000 | 1.000 | 1.000 | 1.000 |
| reset_query | 0.925 | 0.976 | 0.379 | 0.000 |
| reset_BEV | 0.568 | 0.623 | 0.551 | 1.000 |

Interpretation:

```text
reset_all fully removes the single-sequence t+1 attack-off difference.
reset_query primarily removes query/pred residue while leaving BEV residue.
reset_BEV removes fused-BEV residue and partially reduces query/pred residue.
```

This is the first single-sequence causal temporal-state evidence. It is still
not a main-result claim until repeated on more newsplit-val candidates and tied
to map-level target-boundary metrics.
