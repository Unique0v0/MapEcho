# Current Data Construction Status

## Summary

The CCS'25 seed reuse path is valid for StreamMapNet oldsplit.

```text
CCS'25 100 asymmetric seeds
  ↓ StreamMapNet oldsplit token matching
matched = 100 / 100
missing = 0
  ↓ temporal eligibility, W=10, L=19
eligible = 33 target frames / 14 scenes
```

The original 100 CCS'25 asymmetric seeds cover 31 distinct nuScenes scenes.

The original CCS'25 repository also contains the reusable attack assets needed
for the next stage:

```text
scene JSON geometry       = 100 / 100 seeds
ETA centerline JSON       = 100 / 100 seeds
camera-blind ETA best loc = 100 / 100 seeds
camera-blind RSA best loc = 30 / 100 seeds
patch RSA/ETA result pkl  = present
```

For the current temporal-eligible pool:

```text
33 temporal-eligible frames / 14 scenes
ETA best loc available = 33 / 33 frames, 14 / 14 scenes
RSA best loc available = 13 / 33 frames, 6 / 14 scenes
```

For the current Phase 1 set:

```text
20 selected frames / 14 scenes
ETA best loc available = 20 / 20 frames
RSA best loc available = 7 / 20 frames
```

Note: the `has_attack_config` field in the first temporal metadata CSV only
checks whether attack-like keys exist inside the CCS'25 seed pickle. The actual
CCS'25 attack results are stored separately under the original repository's
`dataset/maptr-bevpool/.../results/map/attack/` folders. Use the attack asset
index below for attack availability.

## Implication

Phase 1 can proceed directly with the 33 temporal-eligible frames. These are enough for clean sanity, hook sanity, reset sanity, and a 10-20 target-frame mini probe.

Main-experiment sample size is tight. If the main analysis uses only one primary frame per scene, the temporal-eligible pool has at most 14 samples before clean-quality and VPA filters. Therefore the current main-analysis strategy is:

```text
Use all temporal-eligible frames after clean-quality and VPA filters.
Use scene-level clustered bootstrap for statistics.
Use one-primary-frame-per-scene as conservative analysis.
Expand candidates only if final valid samples fall below 20 frames or 10 scenes.
```

## Generated Files

Seed matching:

```text
/data/dj/MapEcho/artifacts/seed_matching/ccs25_seed_streammapnet_match.csv
/data/dj/MapEcho/artifacts/seed_matching/ccs25_seed_streammapnet_match_summary.json
/data/dj/MapEcho/artifacts/seed_matching/temporal_eligible_tokens_W10_L19.txt
```

Temporal metadata:

```text
/data/dj/MapEcho/artifacts/seed_matching/temporal_eligible_metadata_W10_L19.csv
```

Phase 1 probe selection:

```text
/data/dj/MapEcho/artifacts/phase1/phase1_probe_tokens.txt
/data/dj/MapEcho/artifacts/phase1/phase1_probe_selection.csv
/data/dj/MapEcho/artifacts/phase1/phase1_probe_selection_summary.json
```

CCS'25 attack asset index:

```text
/data/dj/MapEcho/artifacts/ccs25_attack_assets/ccs25_attack_asset_index.csv
/data/dj/MapEcho/artifacts/ccs25_attack_assets/ccs25_attack_asset_index_summary.json
/data/dj/MapEcho/artifacts/ccs25_attack_assets/temporal_eligible_attack_assets_W10_L19.csv
/data/dj/MapEcho/artifacts/ccs25_attack_assets/phase1_attack_assets.csv
```

Attack-point projection sanity:

```text
/data/dj/MapEcho/artifacts/rendering_sanity/attack_point_projection/eta_attack_point_coordinate_sanity_summary.csv
/data/dj/MapEcho/artifacts/rendering_sanity/attack_point_projection/eta_attack_point_frame_projection_sanity.csv
/data/dj/MapEcho/artifacts/rendering_sanity/attack_point_projection/eta_attack_point_overlay_index.csv
/data/dj/MapEcho/artifacts/rendering_sanity/attack_point_projection/overlays/
```

Initial ETA sanity run:

```text
checked samples = 5
visible at t    = 5 / 5
visible at t+1  = 4 / 5
visible at t+5  = 5 / 5
overlays        = 6 frames from one sequence
```

The checked sequence shows the expected fixed-world behavior: the projected
source moves naturally across cameras over time instead of staying fixed in the
same ego-frame coordinate or image pixel.

Target-boundary VPA sanity:

```text
/data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa/eta_target_boundary_vpa_sanity.csv
/data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa/eta_target_boundary_vpa_sanity_summary.json
/data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa/rsa_target_boundary_vpa_sanity.csv
/data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa/rsa_target_boundary_vpa_sanity_summary.json
/data/dj/MapEcho/artifacts/rendering_sanity/target_boundary_vpa/overlays/
```

Phase 1 ETA VPA sanity:

```text
checked samples                 = 20
diverge_boundary visible         = 20 / 20
diverge_boundary BEV on-boundary = 20 / 20
diverge_boundary overlay VPA     = 20 / 20
reference_boundary overlay VPA   = 0 / 20
median dist to diverge boundary  = 0.105 m
median diverge-boundary coverage = 0.537
```

Phase 1 RSA subset VPA sanity:

```text
checked samples                 = 7
diverge_boundary visible         = 7 / 7
diverge_boundary BEV on-boundary = 7 / 7
diverge_boundary overlay VPA     = 7 / 7
reference_boundary overlay VPA   = 0 / 7
median dist to diverge boundary  = 0.012 m
median diverge-boundary coverage = 0.337
```

Important interpretation: strict ground-projected center distance is not the
primary VPA criterion because the physical light source has height while the
road boundary lies on the ground plane. The sanity check therefore reports
ground-center, height-aligned-center, and glare-region coverage separately.

Attack rendering injection smoke:

```text
/data/dj/MapEcho/artifacts/rendering_sanity/injection_smoke/eta_injection_smoke_frames.csv
/data/dj/MapEcho/artifacts/rendering_sanity/injection_smoke/eta_injection_smoke_samples.csv
/data/dj/MapEcho/artifacts/rendering_sanity/injection_smoke/eta_injection_smoke_summary.json
/data/dj/MapEcho/artifacts/rendering_sanity/injection_smoke/eta/*/
```

Initial ETA rendering smoke:

```text
checked samples       = 5
written frame rows    = 23
attacked frame rows   = 5
N_attack=1 schedule   = pass
raw uint8             = pass
image shape unchanged = pass
```

The smoke test writes raw clean images, scheduled images, overlays, and metadata
for `t-2, t-1, t, t+1, t+2` when the attack point is visible. Only offset `0`
receives glare rendering; warm-up and recovery frames remain clean. The script
explicitly records the intended injection order:

```text
raw image -> attack rendering -> resize/pad/normalization -> model input
```

StreamMapNet temporal readiness audit:

```text
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/temporal_readiness_audit.json
/data/dj/MapEcho/artifacts/streammapnet_hook_sanity/original_newsplit_config_with_oldsplit_ckpt_audit.json
```

Current audit result:

```text
Phase 1 temporal sequences       = OK
config streaming query           = false
config streaming BEV             = false
checkpoint query temporal weights= absent
checkpoint BEV temporal weights  = absent
can run clean hook sanity        = false
```

Blocking reason:

```text
The current oldsplit config/checkpoint pair is a non-streaming baseline.
It cannot produce query-memory / BEV-memory temporal dumps and cannot validate
reset-all / reset-query / reset-BEV behavior.
```

Do not treat a run with the current checkpoint as clean StreamMapNet temporal
hook sanity. A temporal StreamMapNet config and matching temporal checkpoint are
needed before entering model-level clean/reset/attack sanity.

Rechecking `/home/dj/StreamMapNet`:

```text
original temporal NuScenes config exists:
  /home/dj/StreamMapNet/plugin/configs/nusc_newsplit_480_60x30_24e.py

original temporal config enables:
  streaming query = true
  streaming BEV   = true

MapEcho copy matches original for:
  nusc_newsplit_480_60x30_24e.py
  StreamMapNet.py
  MapDetectorHead.py

downloaded temporal checkpoint in /home/dj/StreamMapNet:
  not found
```

Using the original temporal newsplit config with the current oldsplit baseline
checkpoint still fails readiness because the checkpoint has no temporal module
weights (`head.query_update`, `stream_fusion_neck`). The official temporal
NuScenes checkpoint referenced by the StreamMapNet README is for newsplit, not
the current oldsplit baseline checkpoint.

Updated after downloading the official newsplit temporal checkpoint:

```text
newsplit temporal config:
  /home/dj/StreamMapNet/plugin/configs/nusc_newsplit_480_60x30_24e.py

newsplit temporal checkpoint:
  /home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth

MapEcho eval wrapper:
  /home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py

generated newsplit annotations:
  /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_train_newsplit.pkl
  /home/dj/MapEcho/datasets/nuScenes/nuscenes_map_infos_val_newsplit.pkl
```

The newsplit config/checkpoint pair is temporal-ready:

```text
streaming query config     = true
streaming BEV config       = true
checkpoint query weights   = present: head.query_update
checkpoint BEV weights     = present: stream_fusion_neck
can run clean hook sanity  = true
```

However, the existing CCS'25 seed pool was built on oldsplit validation scenes.
Under newsplit:

```text
CCS'25 100 seeds:
  newsplit train = 88 frames / 26 scenes
  newsplit val   = 12 frames / 5 scenes

old temporal-eligible W=10/L=19 subset:
  newsplit train = 26 frames / 11 scenes
  newsplit val   = 7 frames / 3 scenes

old Phase 1 20-frame subset:
  newsplit train = 15 frames / 11 scenes
  newsplit val   = 5 frames / 3 scenes
```

Therefore, the newsplit temporal checkpoint is the right model for clean/reset
hook sanity, but a strict newsplit-val main experiment should either use the
small overlapping subset only or rebuild asymmetric candidates on newsplit val.

Phase 1.0 overlap artifacts were generated for immediate hook sanity:

```text
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_tokens.txt
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/phase1_0_overlap_selection.csv
/data/dj/MapEcho/artifacts/phase1_0_newsplit_overlap/newsplit_overlap_summary.json
```

Summary:

```text
old Phase 1 ∩ newsplit val ∩ W=10/L=19
  = 5 frames / 3 scenes
```

The v11 split strategy is documented in:

```text
doc/experiment_plan_v11_newsplit_strategy.md
```

Phase 1.0 clean_keep hook sanity status:

```text
inference frames          = 30 / 30
query_memory dumps        = 30 / 30
bev_memory dumps          = 30 / 30
query first-frame count   = 1
BEV first-frame count     = 1
propagated mask first     = 0
propagated mask later     = 33
warped BEV later frames   = 29 / 29
NaN / Inf                 = none observed
```

The first run crashed only after inference, during `format_results`, because
`prefix` was missing under `--format-only`. `scripts/run_phase1_0_clean_keep.sh`
now passes `--eval-options prefix=...`.

Phase 1.0 reset sanity status:

```text
reset_all:
  query dumps              = 30 / 30
  BEV dumps                = 30 / 30
  prop mask before reset   = 33
  prop mask after reset    = 0
  warped BEV after reset   = absent
  query first after reset  = true
  BEV first after reset    = true
  pass                     = true

reset_query:
  query dumps              = 30 / 30
  BEV dumps                = 30 / 30
  prop mask before reset   = 33
  prop mask after reset    = 0
  warped BEV after reset   = present
  query first after reset  = true
  BEV first after reset    = false
  pass                     = true

reset_BEV:
  query dumps              = 30 / 30
  BEV dumps                = 30 / 30
  prop mask before reset   = 33
  prop mask after reset    = 33
  warped BEV after reset   = absent
  query first after reset  = false
  BEV first after reset    = true
  pass                     = true
```

This confirms that reset hooks are selective and can be used as causal controls:
`reset_query` clears query propagation while retaining BEV history, `reset_BEV`
clears BEV history while retaining query propagation, and `reset_all` clears
both temporal paths.

Phase 1.0 attack-at-t dry run status:

```text
target token              = 6e147994a5e3493d86a928a612ff5791
target sample_idx         = 572
attack objective          = ETA
attack camera             = CAM_FRONT_LEFT
N_attack                  = 1
attack query dumps        = 30 / 30
attack BEV dumps          = 30 / 30
submission saved          = yes
NaN / Inf                 = none observed
pass                      = true
```

Clean vs attack deltas:

```text
frame t:
  query score mean_abs    = 0.414945
  query score max_abs     = 3.979524
  pred vector mean_abs    = 0.126144
  topk embedding mean_abs = 0.369988
  current BEV norm delta  = 0.078979
  fused BEV norm delta    = 0.184998

frame t+1, clean input:
  query score mean_abs    = 0.467813
  query score max_abs     = 3.854012
  pred vector mean_abs    = 0.178403
  topk embedding mean_abs = 0.347421
  current BEV norm delta  = 0.000000
  fused BEV norm delta    = 0.015320
```

This confirms that the attacked target frame changes model internals, and that a
clean `t+1` input can still carry a measurable temporal-state difference.

Phase 1.0 attack reset ablation status:

```text
conditions:
  attack_keep        = 30 query dumps / 30 BEV dumps
  attack_reset_all   = 30 query dumps / 30 BEV dumps
  attack_reset_query = 30 query dumps / 30 BEV dumps
  attack_reset_BEV   = 30 query dumps / 30 BEV dumps
```

Important analysis note:

```text
Reset conditions must be compared to matched clean-reset baselines.
Comparing reset conditions directly to clean_keep mixes attack residue with
reset cold-start / distribution-shift effects.
```

Matched-baseline t+1 results:

```text
attack_keep vs clean_keep:
  query score mean_abs    = 0.467813
  pred vector mean_abs    = 0.178403
  topk embedding mean_abs = 0.347421
  fused BEV norm delta    = 0.015320

attack_reset_all vs clean_reset_all:
  query score mean_abs    = 0.000000
  pred vector mean_abs    = 0.000000
  topk embedding mean_abs = 0.000000
  fused BEV norm delta    = 0.000000

attack_reset_query vs clean_reset_query:
  query score mean_abs    = 0.034926
  pred vector mean_abs    = 0.004273
  topk embedding mean_abs = 0.215586
  fused BEV norm delta    = 0.015320

attack_reset_BEV vs clean_reset_BEV:
  query score mean_abs    = 0.201872
  pred vector mean_abs    = 0.067172
  topk embedding mean_abs = 0.156096
  fused BEV norm delta    = 0.000000
```

Matched t+1 reduction ratios:

```text
reset_all:
  query score reduction    = 1.000
  pred vector reduction    = 1.000
  topk embedding reduction = 1.000
  fused BEV reduction      = 1.000

reset_query:
  query score reduction    = 0.925
  pred vector reduction    = 0.976
  topk embedding reduction = 0.379
  fused BEV reduction      = 0.000

reset_BEV:
  query score reduction    = 0.568
  pred vector reduction    = 0.623
  topk embedding reduction = 0.551
  fused BEV reduction      = 1.000
```

This provides single-sequence causal temporal-state evidence: reset-all removes
the attack-off t+1 difference under matched reset baselines; query reset mainly
removes query/pred residue while retaining BEV residue; BEV reset removes fused
BEV residue and partially reduces query/pred residue.

## Phase 1 Probe Set

The current Phase 1 set contains:

```text
20 target frames
14 scenes
14 primary scene-level samples
6 extra within-scene frames
```

Selection rule:

1. Select one primary target frame from every temporal-eligible scene.
2. Prefer larger temporal eligibility margin and more central scene position.
3. Add extra eligible frames until reaching 20 target frames.

## Scripts

```text
scripts/match_ccs25_seeds.py
scripts/build_temporal_phase1_sets.py
scripts/index_ccs25_attack_assets.py
scripts/sanity_attack_point_projection.py
scripts/sanity_target_boundary_vpa.py
scripts/smoke_attack_rendering_injection.py
scripts/audit_streammapnet_temporal_readiness.py
scripts/build_newsplit_overlap_sets.py
scripts/build_sequence_ann_subset.py
scripts/run_phase1_0_clean_keep.sh
scripts/run_streammapnet_reset_sanity.py
scripts/run_phase1_0_reset_sanity.sh
scripts/build_attack_at_t_sequence_ann.py
scripts/run_phase1_0_attack_keep.sh
scripts/summarize_phase1_0_attack_dry_run.py
scripts/summarize_phase1_0_attack_dry_run.sh
scripts/run_streammapnet_sequence_condition.py
scripts/run_phase1_0_attack_reset_ablation.sh
scripts/summarize_phase1_0_attack_reset_ablation.py
scripts/summarize_phase1_0_attack_reset_ablation.sh
```
