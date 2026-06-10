# Phase 1.8 Pool Rebuild Status

## Goal

Phase 1.8-A tests whether the current newsplit-val `ccs_candidate`-derived pool can be converted into a larger high-quality evaluation pool after strict CCS-style geometry filtering and the manual visual-audit exclusions.

This step does not run StreamMapNet inference. It only rebuilds the candidate table and quality gates.

## Inputs

```text
asset_csv:
  /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/phase1_1_asymmetric_dist_eta_like_assets.csv

vpa_csv:
  /data/dj/MapEcho/artifacts/phase1_2_ccs_candidate_high_vpa/vpa_sanity/eta_target_boundary_vpa_sanity.csv

source size:
  243 frames / 15 scenes
```

Manual visual-audit exclusions are applied for straight, repeated, invalid, or center-road placement cases:

```text
005cfc, 51c1ee, 87eb02, 282c9a, bb9632,
c6ceba, c943bd, d85a67, e9c518, d0a9cf
```

The CCS scene blacklist is also applied.

## Outputs

```text
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/geometry_quality_v2_table.csv
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/phase1_8_candidate_pool.csv
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/phase1_8_selected_candidates.csv
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/phase1_8_selected_tokens.txt
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/phase1_8_scene_coverage.csv
/data/dj/MapEcho/artifacts/phase1_8_pool_rebuild/phase1_8_pool_summary.json
```

Reproduction command:

```bash
bash scripts/build_phase1_8_pool.sh
```

## Result

```text
all_assets:
  243 frames / 15 scenes
  median VPA = 0.0620
  median curvature diff = 0.0118
  median lane-width gain = 0.0082
  median diverge turn = 1.95 deg

phase1_8_pass_pool:
  10 frames / 3 scenes
  median VPA = 0.1947
  median curvature diff = 0.1404
  median lane-width gain = 14.7395
  median diverge turn = 68.39 deg
```

Scene coverage:

```text
scene-0269: 4 frames
scene-0913: 2 frames
scene-0962: 4 frames
```

## Gate Drop-Off

```text
length_gate:             243 / 243
lane_width_gate:          50 / 243
curvature_gate:           58 / 243
point_distance_gate:      30 / 243
diverge_near_ego_gate:   237 / 243
vpa_gate >= 0.05:        176 / 243
preferred_vpa >= 0.15:    55 / 243
ccs_geometry_pass:        19 / 243
phase1_8_pass:            10 / 243
```

The limiting factor is not VPA coverage. The main bottleneck is strict CCS-style geometry: lane-width divergence and large-curvature point placement reject most of the broad pool.

## Interpretation

Phase 1.8-A does not produce a sufficiently large high-quality set from the current broad pool. This is an important negative diagnostic:

```text
Current ccs_candidate-derived pool
  + strict CCS-style geometry gates
  + visual-audit exclusions
  -> only 10 frames / 3 scenes
```

Therefore, the current broad pool is too diluted and repetitive to serve as the final rebuild source. Continuing to tune thresholds inside this pool would either keep the set too small or reintroduce visually invalid straight/repeated cases.

## Recommended Next Step

Proceed to Phase 1.8-B:

```text
Rebuild the newsplit-val data pool from an earlier annotation entry point,
using the original CCS preprocessing logic for lane-width divergence,
curvature asymmetry, scene exclusion, temporal eligibility, and scene diversity.
```

The already validated downstream path should then be applied:

```text
CCS-style data pool
  -> dense glare-source candidates
  -> geometric top-400
  -> six-camera CCS renderer
  -> StreamMapNet frame-t scoring
  -> controlled temporal check
```

The Phase 1.7 top-400 pilot remains the strongest evidence that the migrated location-selection and rendering chain works once the input samples are geometrically valid.
