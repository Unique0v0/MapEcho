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
```
