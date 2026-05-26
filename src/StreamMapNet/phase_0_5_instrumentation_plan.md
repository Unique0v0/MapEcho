# Phase 0.5 Instrumentation and Hook Plan

Date: 2026-05-21

Scope: static code review plus hook preparation. No checkpoint loading, nuScenes inference, or evaluation was run.

## 0. Static Conclusion Checklist

- [x] Temporal pipeline: BEVFormer backbone -> BEV memory fusion -> MapDetectorHead -> query propagation -> vector prediction -> memory update.
- [x] Query memory update path: final decoder outputs are ranked by max class logit; top-k query embeddings and reference points are saved.
- [x] BEV memory update path: previous fused BEV is ego-motion warped and fused with current BEV by ConvGRU; fused output is saved.
- [x] Scene reset: `StreamTensorMemory.get()` resets when memory is empty or `scene_name` changes.
- [x] First-frame behavior: query propagation is disabled; BEV uses detached current BEV as pseudo-history.
- [x] Top-k query selection: ranking-gated with `scores.max(-1).topk(k=self.topk_query)`.
- [x] Confidence hard threshold: no absolute threshold before query memory update.
- [x] `score_thr=0.1`: constructor argument only; it does not control propagation.
- [x] `prop_add_stage=1`: insertion happens after decoder layer 0 and before decoder layer 1 runs.
- [x] NuScenes config: `num_queries=100`, `topk=int(100 * 1/3)=33`.
- [x] H1 prior: medium.
- [x] Residue source prior: query propagation medium; BEV fusion medium-to-high.

## 1. Query Propagation Review

### Memory Definition

Relevant code:

- `plugin/models/heads/MapDetectorHead.py:54-72`
- `plugin/models/heads/MapDetectorHead.py:267-278`
- `plugin/models/heads/MapDetectorHead.py:476-478`
- `plugin/models/heads/MapDetectorHead.py:592-593`
- `plugin/models/utils/memory_buffer.py:30-39`

Findings:

- `query_memory` stores final decoder query embeddings selected from `inter_queries[-1][i]`.
- `reference_points_memory` stores final normalized line/reference coordinates selected from `outputs[-1]['lines']`.
- `target_memory` is train-only and stores matched GT targets for transition loss.
- `query_update` is `MotionMLP(c_dim=12, f_dim=embed_dims, identity=True)`.
- Stored tensors are cloned and detached by `StreamTensorMemory.update()`.
- Memory update happens before post-processing and is not affected by `post_process(thr=...)`.

### Top-k Selection

Relevant code:

- Train: `plugin/models/heads/MapDetectorHead.py:453-478`
- Test: `plugin/models/heads/MapDetectorHead.py:569-593`

Findings:

- `_scores` are raw logits from `cls_branches`, not sigmoid scores.
- `max(-1)` is over the class dimension.
- In the active NuScenes config, `loss_cls.use_sigmoid=True`, so `cls_out_channels=num_classes`; there is no explicit background class in the score tensor used by top-k.
- Top-k is per sample within the batch, not global across batch.
- `topk_idx` indexes both query embeddings and reference points.
- Train and test use the same top-k ranking logic; train additionally updates `target_memory`.
- `topk_query=33` comes from config, not a hard-coded constant.

Static wording:

> Query propagation is ranking-gated, not threshold-gated.

### Decoder Insertion

Relevant code:

- `plugin/models/transformer_utils/MapTransformer.py:99-130`
- `plugin/models/heads/MapDetectorHead.py:539-567`

Findings:

- `prop_add_stage=1` means propagated queries are inserted when the decoder loop reaches `lid == 1`.
- Layer 0 runs on current-frame queries only.
- Before layer 1 runs, the model prepends all propagated queries and keeps the best `num_queries - topk` current queries.
- The concatenated tensor remains length `num_queries`; there is no concat-then-grow path.
- Propagated queries do not get a separate positional encoding; `query_pos=None` in this code path.
- Propagated reference points replace the front `topk` entries of `reference_points` and are passed to deformable attention.
- After insertion, propagated and current queries are processed equivalently by subsequent decoder layers.
- Final top-k memory update is over all final outputs, so propagated queries can be selected again.

Interpretation:

> Once a wrong query enters memory, it can be refined at `t+1` rather than acting only as auxiliary context.

## 2. BEV Memory Review

Relevant code:

- `plugin/models/mapers/StreamMapNet.py:150-248`
- `plugin/models/necks/gru.py:47-77`

Findings:

- `bev_memory` stores fused BEV, not raw current BEV.
- Previous fused BEV is ego-motion warped with `curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix`.
- `F.grid_sample()` samples previous-memory coordinates for the current BEV grid.
- ConvGRU input order is `(history, current)`, called as `self.stream_fusion_neck(warped_feat, curr_bev_feats[i])`.
- Fused output is detached when stored by `StreamTensorMemory.update()`.
- BEV memory updates every frame when `streaming_bev=True`.
- BEV memory is not controlled by query prediction confidence.

ConvGRU gate:

```python
z = sigmoid(convz([h, x]))
r = sigmoid(convr([h, x]))
q = convq([r * h, x])
out = (1 - z) * h + z * q
```

Meaning:

- Small `z` retains more previous/history BEV `h`.
- Large `z` moves more toward candidate update `q`.
- `q` is not pure current BEV; it mixes current BEV `x` with reset-gated history `r * h`.

## 3. Reset and First-frame Constraints

Relevant code:

- `plugin/models/utils/memory_buffer.py:39-67`
- `plugin/datasets/nusc_dataset.py:91-105`

Findings:

- Reset condition is batch-wise: each sample slot resets independently when its `scene_name` changes or memory is empty.
- Query memory, reference-point memory, target memory, and BEV memory all use `StreamTensorMemory`.
- Scene change clears both tensor memory and stored pose metadata for that memory object.
- First-frame query propagation is disabled by `is_first_frame_list`.
- First-frame BEV path uses detached current BEV as pseudo-history.

Experiment constraint:

> Target sequences must stay inside one scene, must not start attack at a scene first frame, and must include warm-up frames.

## 4. Hook Preparation Implemented

The hooks are default-off. They do not write debug files unless `debug_cfg` enables them.

### Query Memory Hook

Implemented in `plugin/models/heads/MapDetectorHead.py`.

Dump point: immediately before `query_memory.update()`.

Fields prepared:

- `scene_name`
- `frame_idx`
- `token`
- `is_first_frame`
- `topk_idx`
- `topk_score_raw_logit`
- `topk_score_sigmoid`
- `topk_label`
- `topk_query_embedding`
- `topk_reference_points`
- `topk_pred_vectors`
- `all_query_scores_raw_logit`
- `all_query_scores_sigmoid`
- `all_query_labels`
- `all_query_pred_vectors`
- `propagated_query_mask`

### Propagated Query Mask Hook

Implemented in `plugin/models/transformer_utils/MapTransformer.py`.

Prepared internal state:

- `last_prop_mask`: reliable mask over final query positions.
- `last_prop_valid_idx`: current-query indices retained when propagated queries are inserted.
- `last_prop_add_lid`: decoder layer id where insertion occurred.

`forward_test()` now uses this internal mask for exported `prop_mask`, avoiding the previous all-False behavior.

### BEV Memory Hook

Implemented in `plugin/models/mapers/StreamMapNet.py` and `plugin/models/necks/gru.py`.

Light fields prepared:

- `scene_name`
- `frame_idx`
- `token`
- `is_first_frame`
- `history_bev_norm`
- `warped_history_bev_norm`
- `current_bev_norm`
- `fused_bev_norm`
- `convgru.z_mean`
- `convgru.z_p10`
- `convgru.z_p50`
- `convgru.z_p90`
- `convgru.z_p95`
- `ego_motion_matrix`

Full mode fields, enabled only by `save_full=True`:

- `current_bev`
- `previous_fused_bev`
- `warped_previous_bev`
- `fused_bev`
- `sampling_grid`
- full ConvGRU `z`

### Reset Hooks

Implemented reset entry points:

- `StreamTensorMemory.reset_all()`
- `MapDetectorHead.reset_streaming_query()`
- `StreamMapNet.reset_streaming_bev()`
- `StreamMapNet.reset_temporal_state(mode='all'|'query'|'bev')`

Phase 1 should start with batch size 1 for reset ablations.

## 5. Debug Config Template

Example config addition for a small debug run:

```python
model = dict(
    # existing fields...
    debug_cfg=dict(
        query_memory=dict(
            enabled=True,
            out_dir='debug/query_memory',
        ),
        bev_memory=dict(
            enabled=True,
            out_dir='debug/bev_memory',
            save_full=False,
        ),
    ),
)
```

Full BEV dump for a tiny scene probe:

```python
debug_cfg=dict(
    query_memory=dict(enabled=True, out_dir='debug/query_memory'),
    bev_memory=dict(enabled=True, out_dir='debug/bev_memory_full', save_full=True),
)
```

Expected file layout:

```text
debug/query_memory/{scene_name}/{frame_idx}.pt
debug/bev_memory/{scene_name}/{frame_idx}.pt
```

## 6. Wrong Query Tracking Definition

Target-boundary query:

```text
argmin_q CD(pred_vector_q, GT_target_boundary)
```

Wrong target query:

```text
CD(pred, wrong_reference) + margin < CD(pred, GT_target_boundary)
```

Propagated wrong query:

```text
wrong at t
and query index in final top-k at t
and written to query_memory
and appears in decoder propagated mask at t+1
```

Persistent wrong query:

```text
propagated across t+i
and geometry remains closer to wrong_reference than GT
and embedding cosine similarity to t wrong query > 0.85
```

The `0.85` cosine threshold is a starting point and requires sensitivity analysis.

## 7. Frozen Items

Do not claim H1 is empirically true yet.

Do not run or report:

- Clean mAP reproduction.
- Attack rendering evaluation.
- Full recovery curves.
- Planner-level risk.
- Final attack success rate.

Allowed current claim:

> Static code supports the possibility of temporal residue through ranking-gated query propagation and confidence-agnostic recurrent BEV fusion. Dynamic validation is still required.
