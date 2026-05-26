# 实验方案 v10（最终整合版）
# Temporal Effects of Symmetry-Bias Physical Perturbations in Streaming Online HD Map Construction

**版本定位**：本方案在 v9 基础上，整合关于数据集构建、NuScenes oldsplit、CCS'25 100 个 asymmetric seed frames、H3 恢复延迟假设、MapTR 公平对照、patch 定位与 Phase 1 决策规则的全部讨论结果。

**当前状态**：
- Phase 0.5 已完成；
- 已完成 StreamMapNet 静态代码分析；
- 已完成 hook 设计与默认关闭的 instrumentation 准备；
- 已确认 StreamMapNet README 提供 NuScenes oldsplit 配置与 checkpoint；
- 主实验统一使用 NuScenes oldsplit；
- 数据集构建采用 **CCS'25 100 个 asymmetric target frames 作为 seed frames + 本文 streaming-specific filters**；
- 尚未加载 checkpoint、尚未跑 nuScenes 推理、尚未完成动态实验验证。

---

## 版本更新摘要

相比 v9，本版主要修改如下：

1. **主实验统一使用 NuScenes oldsplit**，以对齐 CCS'25 提供的 100 个 asymmetric target frames。
2. StreamMapNet 使用官方 oldsplit 配置：
   - `plugin/configs/nusc_baseline_480_60x30_30e.py`
   - 官方 full-validation sanity target：AP = 63.4，AP_ped = 61.7，AP_div = 66.3，AP_bound = 62.1。
3. 数据集构建明确采用：
   - 复用 CCS'25 已挑选的 100 个 asymmetric 单帧作为 seed frames；
   - 加入本文自己的 streaming-specific filters；
   - 得到最终 **Temporal Evaluation Set**，该集合很可能少于 100，这是正常且合理的。
4. 不默认在完整 nuScenes 上重新筛选 100 个新场景；完整重筛只作为样本不足或 token 不匹配时的备选方案。
5. 论文中不声称“提出新的 asymmetric scene selection 方法”，而声称：
   - follow CCS'25 for asymmetric seed selection；
   - extend it for streaming temporal evaluation。
6. MapTR 公平对照使用同一个最终 Temporal Evaluation Set 中的 target frame `t`。
7. 恢复曲线假设 H3 重新加入，命名为 **Recovery Delay Hypothesis**。
8. 路径 B 保留但降调：只有机制解释充分时才作为 robustness / self-healing 论文路径。
9. Patch 保持 should-have supplementary experiment，但不进入 one-frame recovery 主线。
10. Defense 仍为 optional，不影响主论文成立。

---

# 一、研究定位与核心问题

## 1.1 背景

已有 CCS'25 工作证明，online HD map construction model 在 asymmetric road scenes 中存在 **symmetry bias**。在 fork、merge、one-side diverging boundary 等场景中，模型容易将非对称道路结构预测为更对称的道路结构。攻击者可以通过 roadside flashlight 或 adversarial patch 触发 road straightening 或 early turn，从而影响下游规划安全。

但是，CCS'25 主要研究的是**单帧 online map construction model**，例如 MapTR 和 VectorMapNet。本文进一步研究 streaming online HD map construction model，代表模型为 StreamMapNet。Streaming model 引入 temporal state，包括：

- **Streaming Query Propagation**：上一帧 selected instance queries 被传播到下一帧；
- **Historical BEV Fusion**：上一帧 fused BEV feature 被 ego-motion warp 后，与当前 BEV feature 融合。

这带来一个关键安全问题：

> 单帧 transient physical perturbation 是否会污染 streaming model 的 temporal state，使攻击停止后的 clean frames 仍然产生错误 map prediction？

## 1.2 核心研究问题

本文研究：

> 在 streaming online HD map construction 中，一次发生在 frame `t` 的 transient camera-blinding perturbation，是否会通过 query propagation 或 BEV memory 在 `t+1 ... t+L` 的 clean recovery frames 中留下 temporal residue？

这个问题有两种可能答案：

1. **Temporal residue**：攻击停止后，错误仍持续若干帧；
2. **Temporal self-healing**：攻击停止后，clean observation、query ranking 与 BEV update 迅速抑制错误。

因此，本研究不是单纯证明攻击一定传播，而是系统判定：

> Streaming temporal state 到底是放大 transient physical attack，还是抑制 transient physical attack？

---

# 二、研究假设

## H1：Temporal Residue Hypothesis

单帧 camera blinding attack 在 frame `t` 造成错误 map prediction 后，可能污染 StreamMapNet 的 temporal state，使 `t+1 ... t+L` 的 clean recovery frames 仍然产生错误预测。

核心验证标准：

- attack frame `t` 产生明显 target-boundary error；
- wrong target query 被写入 query memory，或 BEV feature 出现 attack residue；
- recovery frames 中 map quality 持续低于 paired clean baseline；
- reset-all 后 residue 明显下降；
- reset-query 或 reset-BEV 至少一个能解释 residue 来源。

## H2：Propagation Channel Hypothesis

若 H1 成立，residue 通过以下一个或多个通道传播：

1. **Query propagation channel**：错误 boundary query 进入 final top-k，被保存到 query memory，并在 `t+1` 作为 propagated query 参与 decoder refinement。
2. **BEV memory channel**：攻击帧污染 BEV feature，上一帧 fused BEV 被 ego-motion warp 后进入 ConvGRU，与当前 clean BEV 融合。
3. **Mixed channel**：query propagation 与 BEV memory 同时贡献 residue。

## H3：Recovery Delay Hypothesis

攻击停止后，StreamMapNet 可能需要多个 clean frames 才能恢复到 paired clean baseline。该恢复延迟用以下指标度量：

- RecoveryRatio curve；
- RFC；
- APD；
- AUC_residue；
- Target-boundary Chamfer Delta；
- persistent RSS rate。

同时，必须通过 reset ablation 判断恢复延迟是否来自 temporal state contamination：

- reset-all；
- reset-query-only；
- reset-BEV-only。

注意：

> MapTR 没有 temporal state contamination channel，因此不应表现出由 frame `t` 攻击状态导致的恢复延迟。但 MapTR 在 `t+1` 的错误仍可能来自该帧本身的场景难度，所以 MapTR 不能被当作“理想恢复上界”。

## H4：Single-frame Control Hypothesis

MapTR 没有 temporal state，因此不存在从 `t` 到 `t+i` 的 state contamination channel。MapTR 只作为 **single-frame difficulty control**，而不是 recovery upper bound。

## H5：Temporal Amplification Hypothesis

连续 `N` 帧攻击可能造成超过独立 `N` 次单帧攻击平均效果的额外损伤。主统计应使用 additive interaction：

```text
Interaction_state = Damage_keep-state - Damage_reset-state
Interaction_ind = Damage_continuous - mean(Damage_single_j)
```

Ratio 形式可以作为辅助展示，但不作为主统计结论。

## H6：Temporal Self-healing Alternative

若 H1 / H3 不成立，则本文转为 self-healing / robustness 路径。该路径不应简单写成“模型没被攻击传播”，而应解释：

- attack-frame error 是否进入 temporal state；
- wrong query 是否被 top-k ranking 丢弃；
- current clean observation 是否修正 propagated query；
- BEV ConvGRU gate 是否快速吸收当前帧信息；
- reset-query / reset-BEV 与 keep-state 的差异如何解释 self-healing。

路径 B 只有在机制证据充分时才有投稿价值；否则只能作为 negative result 或项目报告。

---

# 三、Phase 0.5 已完成结论

## 3.1 静态分析目标

Phase 0.5 的目标是，在运行任何评估之前，先从源码判断 H1 是否在机制上可能成立。

当前已完成：

- query memory update path 复查；
- top-k selection 逻辑复查；
- decoder propagated query insertion 复查；
- BEV memory / ConvGRU fusion 复查；
- scene reset 与 first-frame behavior 复查；
- instrumentation hook 设计与实现。

结论：

> **H1 prior = medium。静态分析没有触发 kill condition，Phase 1 值得继续。**

## 3.2 Query propagation 机制

已确认：

- `query_memory` 保存 final decoder query embeddings；
- `reference_points_memory` 保存 final normalized line / reference coordinates；
- memory update 发生在 post-processing 之前；
- `score_thr=0.1` 不影响 propagation；
- top-k 使用 raw class logits，而不是 sigmoid score；
- top-k 是 per-sample，不是 batch-global；
- active NuScenes config 中 `num_queries=100`，`topk=33`。

因此本文使用以下表述：

> Query propagation is ranking-gated, not threshold-gated.

## 3.3 Propagated query 插入位置

已确认：

- `prop_add_stage=1` 表示 decoder layer 0 之后、layer 1 之前插入 propagated queries；
- layer 0 只处理 current-frame queries；
- layer 1 开始，模型将 propagated top-k queries 与 current-frame retained queries 组成总长仍为 `num_queries` 的 query set；
- propagated queries 和 current queries 在后续 decoder layers 中等价参与 refinement；
- final top-k memory update 覆盖所有 final outputs，因此 propagated query 可以再次被选中并继续传播。

这意味着：

> 一旦 wrong query 进入 memory，它在 `t+1` 不是被动辅助信息，而是可以继续参与 map prediction refinement。

## 3.4 BEV memory 机制

已确认：

- `bev_memory` 保存 fused BEV，而不是 raw current BEV；
- previous fused BEV 通过 ego-motion warp 对齐到当前帧；
- ConvGRU 输入顺序为 `(history, current)`；
- fused output 被 detach 后保存；
- BEV memory 每帧更新；
- BEV memory 不受 query confidence 控制。

ConvGRU gate 公式：

```python
z = sigmoid(convz([h, x]))
r = sigmoid(convr([h, x]))
q = convq([r * h, x])
out = (1 - z) * h + z * q
```

含义：

- `z` 越小，越保留历史 BEV；
- `z` 越大，越走向 candidate update；
- candidate update `q` 不是纯 current BEV，而是混合了 reset-gated history 和 current BEV。

因此，后续实验必须记录 ConvGRU gate statistics，尤其是 `z_mean / z_p50 / z_p90`。

## 3.5 已实现 instrumentation hooks

已完成默认关闭 hooks。

### Query memory hook 保存字段

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

### Reliable propagated query mask hook

已在 decoder 内部记录：

- `last_prop_mask`
- `last_prop_valid_idx`
- `last_prop_add_lid`

该 hook 绕开了原 `forward_test()` 中 exported `prop_mask` 可能全 False 的问题。

### BEV memory / ConvGRU hook

light mode 保存：

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

full mode 保存：

- `current_bev`
- `previous_fused_bev`
- `warped_previous_bev`
- `fused_bev`
- `sampling_grid`
- full ConvGRU `z`

### Reset hooks

已实现：

- `StreamTensorMemory.reset_all()`
- `MapDetectorHead.reset_streaming_query()`
- `StreamMapNet.reset_streaming_bev()`
- `StreamMapNet.reset_temporal_state(mode='all'|'query'|'bev')`

Phase 1 的 reset ablation 建议先使用 batch size = 1。

---

# 四、实验优先级

## 4.1 Must-have：核心实验

这些实验决定论文是否成立，必须完成。

1. **Phase 1 Mini Probe / Kill-or-Continue**：验证 attack-frame corruption、wrong query top-k、BEV residue、reset ablation。
2. **Clean baseline + Warm-up convergence**：确认 StreamMapNet clean performance 和 W=10 是否稳定。
3. **Temporal Residue 主实验**：Camera blinding，`N_attack=1`，观察 `t+1 ... t+L` recovery。
4. **Reset ablation**：keep-state、reset-all、reset-query、reset-BEV。
5. **Mechanism analysis**：wrong query tracking + BEV gate statistics + geometry tracking。

## 4.2 Should-have：增强说服力的实验

这些实验尽量做，但不阻塞主线。

1. **MapTR fair rerun**：相同 target frames、相同 rendering、相同 evaluation script 下重跑 MapTR。不要直接引用 CCS'25 Table 2 作为本文主对照。
2. **Planner-level proxy risk**：做 proxy planner，但必须报告 planner-window matching。
3. **Continuous attack interaction**：做 `N ∈ {1,3,5}` 或 `{1,3}`，主统计用 additive interaction，不把 TAF ratio 当主结论。
4. **Patch single-frame / transfer supplement**：用于说明攻击载体不仅限于 flashlight，但不进入 transient recovery 主线。

## 4.3 Optional：时间充足再做

这些实验不影响主论文成立。

1. adaptive patch；
2. full patch continuous visibility temporal effect；
3. defense；
4. MapTracker 第二 streaming model；
5. attention map 可视化；
6. full BEV tensor dump 大规模分析。

---

# 五、模型与数据集

## 5.1 主实验 split：NuScenes oldsplit

本文主实验采用 **NuScenes oldsplit**。

原因：

1. CCS'25 攻击代码提供的 100 个 asymmetric target frames 来自 oldsplit；
2. StreamMapNet 官方也提供 NuScenes oldsplit 配置和 checkpoint；
3. 使用 oldsplit 可以避免 CCS'25 target frames 与 StreamMapNet newsplit checkpoint 之间的 split mismatch；
4. 所有模型、所有攻击与所有 paired clean / attack 结果都在同一 split、同一 target frames 上评估。

StreamMapNet oldsplit 配置：

```text
plugin/configs/nusc_baseline_480_60x30_30e.py
```

官方 full-validation sanity target：

| Metric | Value |
|---|---:|
| AP_ped | 61.7 |
| AP_div | 66.3 |
| AP_bound | 62.1 |
| AP | 63.4 |

注意：

> 官方 AP 63.4 是 full oldsplit validation 上的 sanity reference，不等于本文最终 Temporal Evaluation Set 上的 clean baseline。本文主实验应报告在最终 target set 上的 paired clean / attack metrics。

## 5.2 主要模型：StreamMapNet

StreamMapNet 是本文核心研究对象。

关注机制：

- query propagation；
- historical BEV fusion；
- query matching / confidence ranking；
- ego-motion compensation；
- temporal memory reset；
- first-frame behavior；
- clean observation 是否 self-heal attack residue。

## 5.3 对照模型：MapTR

MapTR 用作单帧对照模型。

约束：

- 使用最终 Temporal Evaluation Set 中的相同 target frame `t`；
- 使用相同 camera blinding rendering；
- 使用相同 evaluation script；
- 使用相同 planner setting；
- 不引用 CCS'25 中已有 MapTR 表格作为本文主结果。

用途：

- 判断 StreamMapNet 的 attack-frame 单帧脆弱性是否强于 / 弱于 MapTR；
- 解释 StreamMapNet 的 temporal effect 是否来自 streaming state，而不是 scene 本身难。

CCS'25 Table 2 数字只能作为 related work / background reference，不能作为本文公平比较的主实验结果。

---

# 六、数据集构建：Temporal Evaluation Set

## 6.1 最终选择：方案 1

本文采用以下数据集构建策略：

> 复用 CCS'25 已挑选的 100 个 asymmetric 单帧作为 seed frames，然后加上本文自己的 streaming-specific filters，得到最终 Temporal Evaluation Set。

不默认采用以下策略：

> 在完整 NuScenes oldsplit 上重新跑 CCS'25 selection code，强行重新构建一个等于 100 个的新数据集。

完整重筛只作为备选：当 CCS'25 100 seeds 经过 temporal filters 后样本量过少，或 sample token / scene token 无法匹配 oldsplit annotation 时，再考虑补充 candidates。

## 6.2 为什么选择方案 1

本文研究目标不是提出新的 asymmetric scene mining 方法，而是研究：

> 已知 single-frame vulnerable asymmetric scenes 在 streaming model 中是否会产生 temporal state contamination。

因此，使用 CCS'25 的 100 个 asymmetric target frames 作为 seeds 有三个优势：

1. 保持与前作可比；
2. 避免重新筛选场景引入 cherry-picking 风险；
3. 将论文重点集中在 temporal residue，而不是 scene mining。

本文真正新增的数据集构建贡献是：

- temporal sequence construction；
- temporal eligibility filter；
- clean-quality stability filter；
- target-boundary visibility / VPA filter；
- scene-level independence control。

## 6.3 论文中的准确表述

推荐英文表述：

```text
We construct a single temporal evaluation set for all experiments. Starting from the 100 asymmetric target frames produced by the CCS'25 selection pipeline on NuScenes oldsplit, we extend each target frame into a same-scene temporal sequence. We then apply three streaming-specific filters: temporal eligibility, clean-quality stability, and target-boundary visibility. The resulting Temporal Evaluation Set is used throughout the paper. StreamMapNet is evaluated on the full sequences, while MapTR is evaluated on the target frame t extracted from each sequence.
```

推荐中文表述：

```text
我们为所有实验构建一个统一的 Temporal Evaluation Set。具体而言，我们从 CCS'25 在 NuScenes oldsplit 上筛选得到的 100 个 asymmetric target frames 出发，将每个 target frame 扩展为同一 scene 内的 temporal sequence。随后加入三个 streaming-specific filters：temporal eligibility、clean-quality stability 和 target-boundary visibility。最终得到的 Temporal Evaluation Set 用于全文实验。StreamMapNet 使用完整序列评估时序传播，MapTR 则只使用每个序列中的 target frame t 进行单帧对照。
```

## 6.4 Dataset Construction Funnel

正式实验报告一个最终 Temporal Evaluation Set 的构建过程，而不是报告两个正式 datasets。

```text
CCS'25 asymmetric seed frames
  ↓ temporal eligibility filter
  ↓ clean-quality stability filter
  ↓ VPA / target-boundary visibility filter
  ↓ scene-level statistical control
Final Temporal Evaluation Set
```

建议表格：

| Step | Remaining samples | Purpose |
|---|---:|---|
| CCS'25 asymmetric seed frames | 100 | asymmetric single-frame seeds |
| Temporal eligibility | N1 | enough warm-up / recovery context |
| Clean-quality stability | N2 | reliable paired clean baseline |
| VPA / target-boundary visibility | N3 | attack source visible to target region |
| Final Temporal Evaluation Set | N4 | paired temporal evaluation |

N1、N2、N3、N4 必须由实际实验填入。

当前数据前置验证结果：

| Step | Remaining samples | Purpose |
|---|---:|---|
| CCS'25 asymmetric seed frames | 100 frames / 31 scenes | asymmetric single-frame seeds |
| StreamMapNet oldsplit token matching | 100 / 100 matched | no token / split mismatch |
| Temporal eligibility, W=10, L=19 | 33 frames / 14 scenes | enough warm-up / recovery context |

这说明 Phase 1 可直接使用这 33 个 temporal-eligible frames；但主实验若严格每 scene 一个 primary frame，最多只有 14 个样本。因此主实验统计策略需要优先采用 scene-level clustered bootstrap，并将每 scene 一个 primary frame 作为 conservative analysis。

当前 CCS'25 原项目攻击资产索引结果：

| Asset | Available samples | Note |
|---|---:|---|
| scene JSON geometry | 100 / 100 | `dataset/scenes_asymmetric` |
| ETA centerline JSON | 100 / 100 | `dataset/diverge_route_centerlines_asymmetric` |
| camera-blind ETA best loc | 100 / 100 | covers all 33 temporal-eligible frames |
| camera-blind RSA best loc | 30 / 100 | covers 13 / 33 temporal-eligible frames |
| patch RSA / ETA result pkl | present | large MapTR-optimized patch artifacts |

因此 Phase 1 的 camera-blinding smoke test 可以先用 ETA loc 做全 20 样本覆盖；RSA loc 只覆盖当前 Phase 1 中的 7 个样本。如果主线需要严格 road-straightening RSA loc，则需要对剩余 temporal-eligible samples 重新生成 RSA attack location，或将 RSA-only 分析作为子集结果。

## 6.5 Temporal Eligibility Filter

对每个 CCS'25 seed frame `t`，要求：

```text
scene_index(t) >= W
scene_index(t) + L < scene_length
```

默认：

```text
W = 10
L = 19
```

也就是：

- target frame 前面至少有 10 个 clean warm-up frames；
- target frame 后面至少有 19 个 clean recovery frames；
- target sequence 必须处于同一 nuScenes scene 内；
- 不能跨 scene 构造 temporal residue。

## 6.6 Target Sequence Construction

每个最终样本构造为：

```text
[t-W, ..., t-1]    clean warm-up frames
[t]                attack frame
[t+1, ..., t+L]    clean recovery frames
```

默认：

```text
W = 10
L = 19
K = 3
tau = 0.9
```

因为 nuScenes keyframe 为 2 Hz，`L=19` 对应 9.5 秒 recovery window。

## 6.7 Warm-up Convergence Analysis

正式实验前测试：

```text
W ∈ {0, 3, 5, 10, 15, 20}
```

观察指标：

- mAP；
- AP_boundary；
- target-boundary Chamfer；
- propagated query count；
- query confidence distribution；
- BEV memory norm；
- ConvGRU gate statistics。

判定：

- 若 `W=10` 与 `W=15/20` 无显著差异，保留 `W=10`；
- 若 `W=10` 不稳定，增大 W 或只选择 scene 后段 frames。

## 6.8 Clean-quality Stability Filter

正式主分析样本必须满足：

1. `mAP_clean_t` 高于 asymmetric validation frames 的第 25 百分位；
2. `AP_boundary_clean_t` 高于阈值；
3. target-boundary Chamfer 不超过阈值；
4. clean prediction 未发生 road straightening；
5. recovery window 内 clean performance 不剧烈震荡；
6. target boundary 在 recovery window 内仍可见；
7. target frame 不在 scene 初始 temporal state 未稳定阶段。

目的：

> 避免把 StreamMapNet 本来就预测错或 clean recovery 本来就震荡的样本误判为 attack residue。

## 6.9 VPA / Target-boundary Visibility Filter

保留满足以下条件的样本：

- attack source 在 frame `t` 的至少一个 relevant camera 中可见；
- glare region 覆盖 target-boundary image region；
- VPA 高于阈值；
- VPA 定义为 glare / blur region 在 target-boundary image region 内的覆盖比例，而不是全图面积占比。

注意：

> VPA filter 只能基于物理可见性和目标区域覆盖，不能基于 attack 是否成功或是否产生 residue。不能用 attack success 过滤样本。

## 6.10 Scene-level Independence Control

由于 CCS'25 100 seeds 在 `W=10, L=19` temporal eligibility 后只剩：

```text
33 target frames / 14 scenes
```

主实验不再默认每个 scene 只取一个 primary frame。当前策略调整为：

```text
主分析：全部 temporal-eligible frames after clean-quality and VPA filters
统计：scene-level clustered bootstrap
保守分析：每个 scene 选一个 primary target frame
```

### 主分析：全部 eligible frames + clustered bootstrap

优点：样本量更现实，同时不把同一 scene 内多个 frames 误当成完全独立样本。

要求：

- 统计时按 scene cluster；
- 报告 `N_frames / N_scenes`；
- bootstrap / confidence interval 以 scene 为 cluster；
- 不按 attack success 选择或过滤样本；
- clean-quality 与 VPA filter 必须在 attack success 判定之前完成。

推荐论文表述：

```text
Our final temporal-eligible set contains 33 target frames from 14 nuScenes scenes. Since multiple target frames may come from the same scene, we use scene-level clustered bootstrap for confidence intervals and significance testing.
```

### 保守分析：每个 scene 一个 primary target frame

用途：作为 robustness / conservative analysis，而不是默认主分析。

选择规则：

- temporal window 完整；
- clean prediction 正确；
- VPA 足够；
- target-boundary clean quality 最好；
- 不按 attack success 选择。

注意：

> 如果 clean-quality / VPA 后主实验有效样本少于 20 frames 或少于 10 scenes，才启动 full oldsplit candidate expansion。

## 6.11 MapTR 如何使用同一数据集

所有模型使用同一个最终 Temporal Evaluation Set。

区别在于：

```text
StreamMapNet input = full temporal sequence [t-W, ..., t-1, t, t+1, ..., t+L]
MapTR input = target frame t only
```

MapTR 条件：

```text
MapTR clean at t
MapTR attack at t
```

MapTR 输出：

- mAP drop；
- AP_boundary drop；
- Target-BD-CD delta；
- RSS；
- RDR；
- CVD。

MapTR 不参与 recovery residue 结论，因为 MapTR 没有 temporal state。

## 6.12 什么时候启用完整重筛作为备选

默认不重新构建新的 100 个样本。

仅在以下情况下启用完整重筛：

### 情况 A：Temporal / clean-quality / VPA filters 后样本太少

例如：

```text
100 → < 20 valid frames
或
< 10 valid scenes
```

这时可用 CCS'25 selection code 在 NuScenes oldsplit 上补充 candidates，再施加同样的 streaming-specific filters。

### 情况 B：CCS'25 100 seeds 无法匹配 StreamMapNet oldsplit annotation

如果 sample token、scene token、map annotation 对不上，则必须重新筛选。

### 情况 C：需要附录泛化性验证

可在附录中从完整 oldsplit 重筛额外 candidates，验证主结论不是 100 seeds 的 artifact。

即使启用补充，也建议写成：

```text
We extend the released CCS'25 seed set with additional asymmetric candidates generated by the same selection pipeline.
```

而不是完全抛弃原 100 seeds。

---

# 七、攻击设置

## 7.1 主攻击：Camera Blinding

主线只使用 camera blinding，因为它能自然定义：

```text
attack on at frame t
attack off after frame t
clean recovery from t+1
```

攻击定位：

> controlled one-frame physical-effect perturbation based on parametric glare approximation

不声称完整复现真实光学过程。

CCS'25 原项目可复用两类 camera-blind attack location：

```text
RSA best loc: road-straightening objective, available for 30 / 100 seeds
ETA best loc: early-turn objective, available for 100 / 100 seeds
```

当前 temporal-eligible 集合中 ETA loc 覆盖 33 / 33，RSA loc 覆盖 13 / 33。Phase 1 应同时记录 attack objective，并避免把 ETA 与 RSA 混成同一个主结果。

Attack-asset indexing. We index reusable attack assets from the CCS'25 repository rather than relying only on the seed-info pkl. For the 100 CCS'25 asymmetric seed frames, scene geometry JSON, ETA centerline JSON, and camera-blind ETA best locations are available for all samples. Among the W=10/L=19 temporal-eligible subset, ETA best locations cover 33/33 target frames from 14/14 scenes, while RSA best locations cover 13/33 target frames from 6/14 scenes. Therefore, ETA best locations are used as the primary camera-blinding source, and RSA best locations are used only as a supplementary subset analysis.

## 7.2 Attack Source Projection

CCS'25 `best_attack_locs.json` 中的位置是 target frame 的 local LiDAR coordinate。用于 StreamMapNet temporal recovery 时，先在 attack frame `t` 将其转换到 ego coordinate，再转换到 global coordinate：

```text
p_ego_t  = T_ego←lidar(t) · p_lidar_t
p_global = T_global←ego(t) · p_ego_t
```

随后攻击源固定在 global coordinate：

```text
p_global
```

每帧投影：

```text
p_cam = T_cam←ego · T_ego←global · p_global
```

注意：

> 不能直接把 calibrated sensor extrinsic 当作 world-to-camera transform。

每帧执行：

1. visibility check；
2. 3D to 2D projection；
3. distance-based intensity approximation；
4. Gaussian glare rendering；
5. VPA logging。

This prevents the attack source from unrealistically moving with the ego vehicle. Directly reusing the target-frame local coordinate for every recovery frame is invalid because it would make the roadside source follow the vehicle.

## 7.3 Valid Projection Area，VPA

VPA 定义为：

```text
glare / blur region 在 target-boundary image region 内的覆盖比例
```

不是全图面积占比。

## 7.4 Intensity Sensitivity

不声称具体 lumen 是真实物理值。

使用：

```text
Intensity ∈ {low, medium, high}
```

主实验使用 medium。附录报告 low / high sensitivity。

## 7.5 Patch Attack：Should-have Supplement

Patch 不是 one-frame recovery 主线。

### Patch 不做

- 不做 `attack at t, clean recovery from t+1`；
- 不把 patch 用于 RFC / AUC_residue 主结论。

### Patch 应做

1. **Single-frame transfer patch**
   - MapTR-optimized patch → StreamMapNet；
   - 观察 StreamMapNet 是否对 CCS'25 patch 有迁移脆弱性。

2. **Single-frame adaptive patch，可选增强**
   - StreamMapNet-optimized patch → StreamMapNet；
   - 区分“patch 无效”是 transfer gap 还是模型鲁棒。

3. **Continuous visibility patch，可选**
   - patch 固定世界坐标；
   - 随车辆运动重投影；
   - 只讨论 sustained visibility temporal effect，不讨论 attack-off recovery。

---

# 八、核心指标

## 8.1 Paired Recovery Ratio

用于越高越好的指标，例如 mAP、AP_boundary：

```text
RecoveryRatio_i = Metric_attack(t+i) / Metric_clean(t+i)
```

解释：

- `RecoveryRatio_i = 1`：完全恢复到 paired clean；
- `< 1`：仍有 residue；
- `> 1`：attack condition 反而更好，按无 residue 处理。

## 8.2 Chamfer Delta

对于越低越好的 Chamfer Distance，使用绝对差：

```text
DeltaCD_i = CD_attack(t+i) - CD_clean(t+i)
```

主分析使用：

```text
AUC_CD = Σ max(0, DeltaCD_i)
```

## 8.3 Recovery Frame Count，RFC

给定阈值 `tau` 和连续窗口 `K`：

```text
RFC = min i such that
RecoveryRatio_i, RecoveryRatio_{i+1}, RecoveryRatio_{i+2} >= tau
```

默认：

```text
tau = 0.9
K = 3
```

若 `L` 内未恢复，标记为 censored sample。

附录报告：

```text
tau ∈ {0.8, 0.9, 0.95}
```

## 8.4 Attack Propagation Depth，APD

```text
APD = Σ 1[RecoveryRatio_i < tau], i = 1...L
```

## 8.5 Residual AUC，主统计指标

```text
AUC_residue = Σ max(0, 1 - RecoveryRatio_i), i = 1...L
```

这是主统计指标。RFC 只作为直观解释指标。

## 8.6 Target-boundary Chamfer

单独计算 target boundary：

```text
CD(B_pred_target, B_gt_target)
```

报告：

- attack frame CD delta；
- recovery frames CD delta；
- AUC_CD；
- per-scene distribution。

## 8.7 Road Straightening Success，RSS

定义：

```text
RSS = 1[
  CD(B_pred, B_straight_ref) < CD(B_pred, B_gt_target)
]
```

可加 margin 版本：

```text
RSS_margin = 1[
  CD(B_pred, B_straight_ref) + margin < CD(B_pred, B_gt_target)
]
```

## 8.8 Damage-based DID

为避免方向混乱，统一定义 damage。

对越高越好的指标：

```text
Damage = Metric_clean - Metric_attack
DID = Damage_keep-state - Damage_reset-state
```

对 Chamfer：

```text
Damage_CD = CD_attack - CD_clean
DID_CD = Damage_CD_keep - Damage_CD_reset
```

## 8.9 Relative Degradation Rate，RDR

单帧比较 StreamMapNet 与 MapTR：

```text
RDR = (Metric_clean - Metric_attack) / Metric_clean
```

## 8.10 Comparative Vulnerability Difference，CVD

```text
CVD = RDR_StreamMapNet(attack frame) - RDR_MapTR(attack frame)
```

解释：

| CVD | RFC | 解释 |
|---|---|---|
| > 0 | ≥ 2 | StreamMapNet 单帧更脆弱且有 residue |
| < 0 | ≥ 2 | StreamMapNet 单帧更稳，但错误进入 state 后残留 |
| > 0 | ≤ 1 | 只有瞬时脆弱性 |
| < 0 | ≤ 1 | 主攻击路径较弱，考虑 self-healing |

## 8.11 Planner-level Metrics

使用 Hybrid A* proxy planner。

指标：

- UGR：Unreachable Goal Rate；
- UPTR：Unsafe Planned Trajectory Rate；
- unsafe window length；
- unsafe travel distance；
- cumulative planner risk；
- paired delta：

```text
DeltaUGR = UGR_attack - UGR_clean
DeltaUPTR = UPTR_attack - UPTR_clean
```

必须额外报告：

> recovery window 内车辆是否仍需要经过 asymmetry anchor 附近。

措辞限制：

> 只说 planner-level risk，不声称商业 AV 一定碰撞。

---

# 九、Wrong Query Tracking 定义

## 9.1 Target-boundary query

```text
q_target = argmin_q CD(pred_vector_q, GT_target_boundary)
```

## 9.2 Wrong target query

```text
WrongTargetQuery =
CD(pred_q, wrong_reference) + margin < CD(pred_q, GT_target_boundary)
```

## 9.3 Propagated wrong query

frame `t` 的 wrong target query 满足：

```text
wrong at t
and query index in final top-k at t
and written to query_memory
and appears in decoder propagated mask at t+1
```

## 9.4 Persistent wrong query

frame `t+i` 中存在 query 满足：

```text
propagated across t+i
and geometry remains closer to wrong_reference than GT
and embedding cosine similarity to t wrong query > 0.85
```

`0.85` 是初始阈值，必须做 sensitivity analysis。

## 9.5 禁止使用的弱证据

不使用：

```text
all-query mean cosine similarity
```

作为机制因果证据。

---

# 十、实验组设计

## 实验组 0：Phase 0.5 静态分析与 Hook 准备

状态：已完成。

输出：

- `phase_0_5_query_propagation_static_analysis.md`
- `phase_0_5_instrumentation_plan.md`
- query memory hook；
- reliable propagated query mask；
- BEV / ConvGRU logging；
- reset hooks；
- `py_compile` 通过；
- `git diff --check` 通过。

结论：

> 不 kill H1，进入 Phase 1。

## 实验组 1：Phase 1 Mini Probe / Kill-or-Continue

### 目的

在小规模样本上判断是否进入路径 A、扩大 Phase 1，或切换路径 B。

### 规模

建议分两步：

```text
1–2 scenes: hook sanity + full debug
10–20 asymmetric scenes: mini probe
```

若结果 ambiguous，扩大到：

```text
30–40 asymmetric scenes
```

### 运行条件

每个 target sequence 跑：

```text
clean + keep-state
attack + keep-state
clean + reset-all
attack + reset-all
clean + reset-query-only
attack + reset-query-only
clean + reset-BEV-only
attack + reset-BEV-only
```

Phase 1 先使用：

```text
batch size = 1
debug light mode
```

### 必须输出

1. hook sanity report；
2. attack-frame target-boundary Chamfer delta；
3. attack-frame RSS；
4. wrong target query confidence distribution；
5. wrong query 是否进入 final top-k；
6. propagated wrong query ratio；
7. ConvGRU `z` statistics；
8. keep-state vs reset-all AUC_residue；
9. reset-query vs reset-BEV residue reduction；
10. preliminary RecoveryRatio curve。

## 10.1 Phase 1 三段式决策

### A. Strong Pass：进入路径 A

满足以下条件中的多数，尤其是 1、2、3：

1. attack frame `t` 有明显 target-boundary error；
2. `AUC_residue > 0` 且 bootstrap 95% CI 不跨 0；
3. `reset-all` 明显降低 AUC_residue；
4. median RFC ≥ 2；
5. wrong query 进入 top-k，或 reset-BEV 明显降低 residue；
6. 至少一个 channel attribution 成立：query / BEV / mixed。

### B. Weak / Ambiguous：扩大 Phase 1，不立即决策

出现以下情况：

- median RFC 在 1–2 之间；
- 只有少数 scene 有 residue；
- reset-all 有趋势但 CI 不稳定；
- attack frame 有错误，但 recovery 曲线噪声大；
- query channel 不明显，但 BEV channel 有弱信号。

处理：

- 扩大到 30–40 scenes；
- 分层 VPA；
- 做 intensity sensitivity；
- 检查 clean-quality filter；
- 暂不进入 full experiment。

### C. Fail：切换路径 B 或停止攻击主线

满足以下条件：

- attack frame 有错，但 `t+1` 基本恢复；
- median RFC ≤ 1；
- AUC_residue CI 跨 0；
- reset-all 与 keep-state 无差异；
- wrong query 不进 top-k，BEV reset 也无效果。

## 实验组 2：主数据集构建

执行完整 dataset funnel。

输出：

- final target frame list；
- scene id；
- sample token；
- target frame index；
- warm-up validity；
- recovery validity；
- clean-quality status；
- VPA status；
- primary frame selection reason；
- clustered bootstrap group id。

## 实验组 3：Single-frame Fair Comparison

### 目的

比较 StreamMapNet 与 MapTR 在相同 target frame 上的瞬时脆弱性。

### 条件

```text
MapTR clean at t
MapTR attack at t
StreamMapNet clean at t after warm-up
StreamMapNet attack at t after warm-up
```

### 指标

- mAP drop；
- AP_boundary drop；
- Target-BD-CD delta；
- RSS；
- RDR；
- CVD。

### 注意

MapTR 不参与 recovery residue 结论。

## 实验组 4：Temporal Residue / Recovery Delay 主实验

### 目的

验证 H1 和 H3 是否成立。

### 条件

```text
clean_keep
attack_keep
clean_reset_all
attack_reset_all
clean_reset_query
attack_reset_query
clean_reset_BEV
attack_reset_BEV
```

### 输出

- RecoveryRatio curve；
- AUC_residue；
- RFC；
- APD；
- AUC_CD；
- persistent RSS rate；
- per-scene distribution；
- censored sample ratio；
- DID table。

### 图表

- recovery curve with CI；
- per-scene AUC_residue boxplot；
- qualitative visualization：clean / attack / recovery；
- reset ablation bar plot。

## 实验组 5：Mechanism Analysis

### 目的

解释 residue 来源。

### Query channel 分析

- attack frame target query 是否 wrong；
- wrong query 是否进入 final top-k；
- wrong query 是否进入 query memory；
- `t+1` propagated mask 是否包含该 query；
- embedding cosine similarity；
- geometry 是否仍接近 wrong reference；
- reset-query 是否消除 residue。

### BEV channel 分析

- attack frame current BEV norm change；
- warped history BEV norm；
- fused BEV norm；
- ConvGRU `z` distribution；
- reset-BEV 是否降低 residue；
- BEV feature similarity over recovery frames。

### 强证据链

Query channel：

```text
attack frame wrong boundary
+ wrong query enters top-k
+ propagated mask confirms t+1 insertion
+ embedding cosine > 0.85
+ geometry closer to wrong reference than GT
+ reset-query reduces residue
```

BEV channel：

```text
attack frame BEV feature shifts
+ ConvGRU z indicates history retention
+ reset-BEV reduces residue
+ query reset alone cannot explain residue
```

## 实验组 6：Continuous Attack Interaction

测试：

```text
N ∈ {1, 2, 3, 5}
```

若资源有限，压缩为：

```text
N ∈ {1, 3}
```

主指标：

```text
Interaction_state = Damage_keep-state - Damage_reset-state
Interaction_ind = Damage_continuous - mean(Damage_single_j)
```

Ratio 只做辅助展示，不作为主结论。

## 实验组 7：Planner-level Safety

### 目的

判断 map residue 是否在 planning proxy 中有下游影响。

### 条件

- same start；
- same goal；
- same route；
- same planner parameters；
- paired clean / attack map inputs。

### 指标

- UGR；
- UPTR；
- unsafe window length；
- unsafe travel distance；
- cumulative planner risk；
- DeltaUGR；
- DeltaUPTR；
- planner-window matching ratio。

### 必须报告

```text
recovery window 中仍需经过 asymmetry anchor 的帧比例
```

若比例低，planner risk 只作为补充，不作为核心主张。

## 实验组 8：Patch Supplementary

Patch 不用于 transient recovery 主线。

### Should-have

1. MapTR-optimized patch → StreamMapNet transfer；
2. single-frame patch effect；
3. TransferGap 初步分析。

### Optional

1. StreamMapNet adaptive patch；
2. patch continuous visibility temporal effect；
3. patch failure cases。

## 实验组 9：Defense，可选

只有 H1 / H3 成立且机制信号稳定后再做。

候选 defense：

- temporal state reset after anomaly；
- reset-query-only；
- reset-BEV-only；
- reset-all；
- fallback to single-frame mode for K frames；
- query confidence consistency check；
- BEV feature anomaly detection。

指标：

- defended AUC_residue；
- clean mAP drop；
- false positive rate；
- UGR / UPTR reduction；
- reset cost。

注意：

> defense 不是主贡献，除非效果非常稳定。

---

# 十一、统计分析

## 11.1 主统计单位

主分析以 scene 为统计单位。

推荐：

- 主文：每个 scene 一个 primary target frame；
- 附录：clustered bootstrap 使用多个 frames。

## 11.2 置信区间

所有核心指标报告：

- mean；
- median；
- std；
- P75；
- P90；
- bootstrap 95% CI；
- censored ratio。

## 11.3 检验方法

| 指标 | 检验 |
|---|---|
| AUC_residue | paired bootstrap / Wilcoxon signed-rank |
| DID | paired bootstrap |
| Interaction_state | paired permutation test |
| RFC | survival-style summary + censored ratio |
| RSS persistence | paired proportion test |
| planner delta | paired bootstrap |

显著性：

```text
p < 0.05, two-sided
```

---

# 十二、8 周执行计划

## Week 0：已完成

- Phase 0.5 static analysis；
- instrumentation hooks；
- reliable propagated query mask；
- reset hooks；
- py_compile / diff check。

## Week 1：Sanity + Mini Probe

目标：确认工具链可用，不直接追求结果。

任务：

- clean inference sanity；
- hook dump sanity；
- reset sanity；
- attack rendering smoke test；
- 1–2 scenes full debug；
- 10 scenes mini probe。

验收：

- dump 文件完整；
- `propagated_query_mask` 合理；
- reset-all / query / BEV 不崩；
- attack frame 至少在部分 scene 造成 target-boundary error。

## Week 2：扩大 Phase 1 + 决策

任务：

- 扩大到 20–40 scenes；
- 跑 keep / reset-all / reset-query / reset-BEV；
- 计算 AUC_residue、RFC、APD、Target-BD-CD；
- 做初步 channel attribution；
- 决定路径 A / ambiguous / 路径 B。

验收：

- 给出 kill-or-continue report。

## Week 3：主数据集与 clean-quality filter

任务：

- 完整 dataset funnel；
- CCS'25 100 seed frames 匹配 oldsplit annotation；
- temporal eligibility filter；
- warm-up convergence；
- clean-quality filter；
- VPA / target-boundary visibility filter；
- scene-level primary frame selection；
- 生成 final Temporal Evaluation Set。

验收：

- Temporal Evaluation Set 冻结；
- 每个样本有 clean baseline、VPA、warm-up status；
- funnel 表格有完整数量。

## Week 4：Temporal Residue / Recovery Delay 主实验

任务：

- camera blinding `N_attack=1`；
- paired clean / attack；
- keep-state / reset-all / reset-query / reset-BEV；
- 计算 AUC_residue、RFC、APD、DID、AUC_CD。

验收：

- 主结果表；
- recovery curve；
- reset ablation 图。

## Week 5：Mechanism Analysis

任务：

- wrong query tracking；
- top-k selection rate；
- propagated wrong query rate；
- matched embedding similarity；
- geometry-level tracking；
- BEV z statistics；
- reset-query vs reset-BEV attribution。

验收：

- query channel / BEV channel / mixed channel 分类；
- 典型 case 可视化。

## Week 6：Fair Comparison + Planner + Patch Supplement

优先级顺序：

1. MapTR fair rerun；
2. planner-level proxy；
3. patch single-frame transfer；
4. continuous attack interaction，如果时间允许。

验收：

- MapTR 不再直接引用 CCS'25 作主对照；
- planner-window matching report；
- patch supplement 有基本结果。

## Week 7：写作主文

任务：

- Introduction；
- Problem formulation；
- Methodology；
- Experiments；
- Main results；
- Discussion。

## Week 8：补实验 + 定稿

任务：

- 补缺失统计；
- bootstrap CI；
- sensitivity analysis；
- limitation；
- appendix；
- figure polishing；
- submission package。

---

# 十三、投稿目标

## 主目标

- **IROS 2026**
- **IV 2026**

这两个目标适合 autonomous driving safety、perception robustness、online mapping vulnerability 方向。

## 条件性目标

若 H1 / H3 强成立，且机制证据、统计显著性、planner-level risk 都充分：

- 可尝试更偏 security / systems safety 的 venue。

若 H1 / H3 不成立但 self-healing 机制解释充分：

- 更适合 IV / IROS workshop / T-IV / T-ITS extension。

---

# 十四、论文路径

## 路径 A：H1 / H3 成立

候选标题：

```text
Temporal Residue of Physical Attacks on Streaming Online HD Map Construction
```

核心贡献：

1. 首次证明 streaming online HD map construction 在 transient physical perturbation 下存在 temporal residue；
2. 提出 paired temporal evaluation protocol；
3. 通过 reset ablation 定位 query propagation / BEV fusion residue channel；
4. 追踪 wrong query contamination；
5. 量化 recovery delay 与 planner-level risk 在 recovery window 内的持续影响；
6. 提供初步 mitigation，可选。

## 路径 B：H1 / H3 不成立

候选标题：

```text
Temporal State as a Double-Edged Sword in Streaming Online HD Map Construction
```

或：

```text
When Streaming Helps: Temporal Self-Healing in Online HD Map Construction under Transient Physical Perturbations
```

核心贡献：

1. 给出首个 paired temporal perturbation protocol；
2. 证明 attack-frame error 是否进入 temporal state 取决于 top-k ranking 与 BEV fusion；
3. 通过 reset ablation 解释 self-healing 来源；
4. 总结 streaming architecture 中哪些设计降低 temporal contamination：
   - top-k ranking bottleneck；
   - current observation refinement；
   - ConvGRU update gate；
   - scene reset；
   - low-confidence wrong query dropout。

路径 B 不应声称与路径 A 同等强度。只有机制解释充分时才作为投稿路径。

---

# 十五、当前禁止声称的内容

在 Phase 1 动态实验完成前，不能声称：

- H1 已经成立；
- H3 已经成立；
- StreamMapNet 一定存在 temporal residue；
- wrong query 已经被实验证明传播；
- BEV memory 已经被实验证明污染；
- attack 会导致真实商业 AV 碰撞；
- planner-level risk 已经显著增加。

当前允许声称：

> Phase 0.5 static analysis supports the possibility of temporal residue through ranking-gated query propagation and confidence-agnostic recurrent BEV fusion. Dynamic validation is still required.

---

# 十六、下一步最小执行清单

服务器恢复后，按以下顺序执行：

1. **CCS'25 seed matching**
   - 读取 100 个 asymmetric seed frames；
   - 匹配 NuScenes oldsplit sample token / scene token；
   - 检查 frame index；
   - 检查前后帧是否连续。

2. **Clean hook sanity run**
   - 检查 query dump；
   - 检查 BEV dump；
   - 检查 propagated mask；
   - 检查 ConvGRU z statistics。

3. **Reset sanity run**
   - keep-state；
   - reset-all；
   - reset-query；
   - reset-BEV。

4. **Attack rendering smoke test**
   - 检查 projection；
   - 检查 VPA；
   - 保存渲染图。

5. **10–20 scenes mini probe**
   - attack-frame RSS；
   - wrong query top-k；
   - propagated wrong query；
   - AUC_residue；
   - reset ablation。

6. **决策**
   - Strong Pass：进入路径 A；
   - Ambiguous：扩大 Phase 1；
   - Fail：切换路径 B 或停止攻击主线。

---

# 十七、总结

本方案的核心逻辑是：

> Phase 0.5 已经证明 H1 在代码机制上“可能成立”，但没有证明它在数据和攻击下“实际成立”。因此下一步必须用小规模 Phase 1 mini probe 验证 attack-frame corruption、wrong-query propagation、BEV-memory residue 和 reset-ablation effect。

最终执行原则：

- 主实验 split：NuScenes oldsplit；
- 数据集：CCS'25 100 asymmetric seed frames + streaming-specific filters；
- 最终数据集：一个统一的 Temporal Evaluation Set；
- StreamMapNet：使用完整 temporal sequences；
- MapTR：只使用每个 sequence 的 target frame `t`；
- 主线攻击：camera blinding；
- 主线证据：paired recovery + reset ablation + mechanism tracking；
- MapTR 对照：必须公平重跑；
- patch：作为 should-have supplement；
- defense：作为 optional；
- 路径 B：保留但降调；
- 投稿管理：按 8 周计划推进。
