# Camera-glare Pipeline CCS Migration Status

Date: 2026-06-04

## Motivation

The Phase 1.5 visual audit exposed two robustness-evaluation fidelity issues:

1. The previous MapEcho renderer was a simplified single-image glare overlay.
2. The previous target-frame annotation replaced only one camera image, while
   the original CCS camera-glare evaluation applies the same light source over
   all six nuScenes cameras.

This can understate the strength of the camera-glare perturbation even when a
larger numeric `power` value is used.

## Current Migration Decision

Phase 1.5 is switched to:

```text
renderer    = ccs
camera_mode = all
power       = 3000
```

The previous simplified renderer and single-camera mode are now disabled in the
target-frame annotation builder. The builder accepts only:

```text
--renderer ccs --camera-mode all
```

## Current MapEcho vs Original CCS

| Stage | Original CCS camera-glare evaluation | Previous MapEcho Phase 1.x | Migrated MapEcho Phase 1.5 |
| --- | --- | --- | --- |
| Input views | six nuScenes cameras | StreamMapNet six-camera input, but only one camera image was replaced | six camera files replaced at frame `t` |
| Frame schedule | current evaluated sample only | target frame `t` only | target frame `t` only |
| Location source | dense boundary candidate selection + model-evaluated location scoring on MapTR | ETA-like diverge-boundary anchor heuristic + VPA/gates | unchanged for now: ETA-like heuristic |
| Renderer | CCS lens-flare utility | simplified raw BGR Gaussian alpha glare | CCS-style raw BGR lens flare equivalent |
| Camera handling | loop over six cameras and apply affected-camera logic | choose one visible/central camera | loop over six cameras and apply affected-camera logic |
| Power | 3000 | 3000/6000 sensitivity | 3000 |
| Output image paths | tensor-level image update inside original runner | one replaced image path | six replaced image paths |

## What Is Now Migrated

### Six-camera Replacement

The frame-t annotation builder now defaults to:

```text
--renderer ccs
--camera-mode all
```

For frame `t`, it saves six camera files and replaces all six `img_fpath`
entries in the annotation. Cameras not affected by the light-source geometry
are still written as clean copies, so the model always consumes a full
six-camera bundle from the generated artifact directory.

### CCS-style Renderer

The CCS renderer shim ports the lens-flare logic for raw BGR StreamMapNet
images:

```text
BASE_RADIUS_DIVISOR = 2
MIN_RADIUS_DIVISOR  = 8
MAX_DISTANCE_FACTOR = 30
INTENSITY_SCALE     = 0.02
MIN_INTENSITY       = 0.6
FALLOFF_POWER       = 1.5
BLUE_TINT_FACTOR    = 1.1
CAMERA_ANGLE        = 60 degrees
```

It also ports the affected-camera check:

```text
camera/light angle threshold
same-side vehicle quadrant check
```

### Phase 1.5 Wrapper

The Phase 1.5 wrapper writes to:

```text
/data/dj/MapEcho/artifacts/phase1_5_controlled_experiment/high_quality_relaxed_v2_ablation_ccs_renderer_power3000
```

and sets:

```text
power = 3000
renderer = ccs
camera_mode = all
```

Some legacy environment-variable names are kept in scripts for compatibility,
but the target-frame builder itself no longer accepts simplified rendering or
single-camera replacement.

## What Is Not Yet Migrated

The glare-source location selection is still not the original CCS dense
location-selection process.

Current newsplit Phase 1.5 still uses:

```text
mapecho_loc_method = diverge_boundary_anchor_heuristic
```

Original CCS location selection uses:

```text
dense boundary sampling
+ optional local random sampling
+ height enumeration
+ geometric feasibility ranking
+ MapTR forward for each candidate
+ ETA/RSA boundary-score selection
```

Therefore Phase 1.5 should be described as:

```text
CCS-style six-camera rendering with ETA-like heuristic locations.
```

It should not yet be described as:

```text
full CCS ETA location selection on newsplit.
```

## Smoke Check

A local `/tmp` smoke check was run on:

```text
target = b2d77fbfe24e4cdb988949cc2565652b
```

Observed:

```text
target_sample_count          = 1
replaced_camera_count        = 6
affected_camera_count        = 2
pass_replaced_expected_cams  = true
pass_raw_uint8               = true
pass_shape_unchanged         = true
contact_sheet                = generated
```

## Lightweight Six-camera Visual Audit

Before running model inference, regenerate the audit cases with the migrated
renderer:

```bash
bash scripts/prepare_phase1_5_ccs_renderer_visual_audit.sh
```

Default output:

```text
/data/dj/MapEcho/artifacts/phase1_5_ccs_renderer_visual_audit
```

For each audit token, inspect the six-camera contact sheet and the generated
frame-t summary JSON under the token artifact directory. The underlying paths
retain legacy script naming, but the visual content should be interpreted as a
six-camera camera-glare robustness case.

This audit covers:

```text
old false negatives
old false positives
top residue 5
bottom failure 5
```

## Recommended Next Steps

1. Re-run a lightweight visual audit using the new six-camera contact sheets.
2. Run Phase 1.5 controlled experiment with the migrated renderer.
3. Evaluate the same four claims:

```text
broad high-quality unconditional effect
target-frame-delta subset conditional residue
reset_all / reset_BEV map-level removal
reset_query internal-only effect
```

4. If renderer migration improves frame-t target-boundary delta but sample
   quality remains the bottleneck, migrate or reuse the original CCS dense
   location-selection process for promising newsplit candidates.
