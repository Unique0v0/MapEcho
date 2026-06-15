# MapEcho 当前阶段复盘：对照最初实验方案

## 0. 当前一句话结论

项目已经从最初的“验证 StreamMapNet 是否可能存在 temporal residue”推进到：

```text
已在 newsplit-val、114 frames / 38 scenes、CCS-style 六相机渲染、top-400 模型评分位置选择下，
观察到稳定的 attack-off target-boundary temporal residue；
reset-all 完全清除；
reset-BEV 基本清除 map-level residue；
reset-query 主要清除 internal query/pred residue，但 map-level residue 大多保留。
```

但当前也发现：

```text
qualitative 图例质量不足；
很多默认可视化 case clean 本身不够好，或者图中没有高亮 metric-selected boundary；
目前人工目测真正适合论文定性图的 case 很少，top-residue #3 是当前较可信的机制示例。
```

因此，当前项目状态可以定义为：

```text
主定量链路：基本成立
机制归因：较清楚，偏 BEV-dominant map-level residue
qualitative 展示：需要重新筛选与重画
后续重点：整理证据、清理可视化、决定是否补 clean-quality / qualitative-friendly 样本
```

---

## 1. 最初实验方案的核心目标

最初 v10 方案的核心问题是：

```text
在 streaming online HD map construction 中，
frame t 的一次短时 camera-glare perturbation
是否会污染 StreamMapNet temporal state，
使 t+1 ... t+L 的 clean recovery frames 仍然产生错误 map prediction？
```

对应两个主要假设：

### H1：Temporal Residue Hypothesis

如果 frame `t` 产生 map-level 错误，该错误可能进入：

```text
query memory
BEV memory
query + BEV mixed channel
```

并在后续 clean recovery frames 中留下 residue。

### H3：Recovery Delay Hypothesis

攻击停止后，模型可能需要多个 clean frames 才恢复到 paired clean baseline。原方案计划用：

```text
AUC_residue
Target-boundary Chamfer Delta
persistent RSS rate
reset ablation
```

来衡量恢复延迟和 temporal-state 来源。

---

## 2. 最初方案与当前实际路线的关键差异

### 2.1 Split 路线变化

最初方案：

```text
主实验使用 NuScenes oldsplit
复用 CCS'25 100 asymmetric seed frames
```

实际执行中发现：

```text
oldsplit checkpoint 不包含完整 temporal StreamMapNet 权重；
newsplit checkpoint 才包含 query temporal weights 和 BEV temporal weights；
oldsplit CCS seeds 大部分落在 newsplit train，不适合作为 newsplit 主实验集。
```

因此路线改为：

```text
主模型：official StreamMapNet newsplit temporal checkpoint
主数据：NuScenes newsplit val
oldsplit CCS assets：工程 sanity / overlap sanity / 迁移参考
```

这是一个必要转向。它牺牲了直接复用 CCS'25 100 seeds 的便利性，但保证了：

```text
模型 checkpoint 是真正 temporal model；
数据 split 与官方 StreamMapNet temporal checkpoint 对齐；
主实验不会被 train/val split mismatch 质疑。
```

### 2.2 攻击位置来源变化

早期路线：

```text
复用 CCS'25 ETA best loc
或使用 ETA-like heuristic loc
```

后来人工审查发现：

```text
heuristic loc 容易选到直路、车、人行道、电线杆、转弯中心等不理想位置；
VPA 高不等于 attack-frame map corruption 成立；
部分样本 geometry 不够符合 asymmetric vulnerability 条件。
```

最终当前主路线变为：

```text
CCS-style rule pool rebuild
↓
CCS dense geometric candidate generation
↓
top-400 geometric candidates per sample
↓
逐个六相机渲染 + StreamMapNet frame-t forward
↓
用 frame-t target-boundary Delta CD 选模型评分最强位置
```

这更接近原 CCS 的“候选位置搜索 + 模型输出评分”思想，但适配到 StreamMapNet/newsplit。

### 2.3 渲染路线变化

早期曾有简化 glare 渲染用于 smoke test。

当前主链路已经切换为：

```text
CCS-style lens flare renderer
all six cameras at frame t
power = 3000
raw image -> render -> model preprocessing
warm-up clean
target frame t perturbed
recovery frames clean
```

这一点很重要，因为用户人工审查确认：

```text
当前渲染视觉效果已经接近原攻击论文；
不再使用简化渲染作为主实验链路。
```

### 2.4 Window 选择变化

最初常用窗口：

```text
W=10, L=19
```

实际中为了样本量与运行成本，主阶段采用：

```text
W=10, L=9
```

原因：

```text
t+1 / t+2 已经是最敏感 recovery offsets；
L=9 仍覆盖约 4.5s recovery；
L=19 会显著减少 eligible samples 并增加成本。
```

当前核心结果主要报告：

```text
t+1
t+2
```

随后已补充完整 `t+1...t+9` H3 recovery curve 与 AUC_CD：

```text
attack_keep median Delta CD:
  t+1 = +0.0333 m
  t+2 = +0.0179 m
  t+3 = +0.0101 m
  t+4 = +0.0057 m
  t+5 = +0.0039 m
  t+6 = +0.0017 m
  t+7 = +0.0006 m
  t+8 = +0.0007 m
  t+9 = +0.0007 m

median AUC_CD over t+1...t+9:
  attack_keep = 0.1166
  attack_reset_all = 0.0000
  attack_reset_BEV = 0.0020
  attack_reset_query = 0.1099

thresholded AUC_CD counts:
  attack_keep AUC>0.03 = 92/114
  attack_reset_all AUC>0.03 = 0/114
  attack_reset_BEV AUC>0.03 = 10/114
  attack_reset_query AUC>0.03 = 92/114
```

因此，当前 H3 主叙事可以从单纯 `t+1/t+2` 扩展为：

```text
residue 在 t+1 最强，随后逐步衰减；
reset_all 清除整条 recovery curve；
reset_BEV 基本清除 map-level recovery curve；
reset_query 仍接近 keep-state recovery curve。
```

统计表述上需要避免强调 `AUC_CD > 0` 的 positive rate。因为
`AUC_CD = sum(max(0, Delta CD_i))` 对极小正值很敏感，`reset_BEV` 也会
因为数值级别的小正值呈现很高的 `AUC>0` rate。更稳的表述是：

```text
reset_BEV reduces median recovery AUC from 0.1166 to 0.0020,
nearly eliminating map-level residue.
```

如果需要报告 positive AUC rate，应使用阈值化版本，例如：

```text
AUC_CD > 0.03 / 0.05 / 0.10
```

---

## 3. 从最开始到当前阶段的实现流程

## 3.1 Phase 0.5：静态机制审查与 hook 准备

目标：

```text
确认 StreamMapNet 架构中是否存在 temporal residue 的机制入口。
```

完成内容：

```text
检查 query propagation
检查 BEV temporal memory / fusion
设计 query / BEV dump
设计 reset_all / reset_query / reset_BEV hooks
```

结论：

```text
H1 prior = medium
代码机制上存在 query memory 与 BEV memory 两条可能 residue channel
进入动态 Phase 1
```

## 3.2 oldsplit CCS seeds sanity

完成内容：

```text
CCS'25 100 asymmetric seeds 与 StreamMapNet oldsplit token matching
matched = 100 / 100
W=10, L=19 temporal eligible = 33 frames / 14 scenes
ETA best loc 覆盖 = 33 / 33
RSA best loc 覆盖 = 13 / 33
```

意义：

```text
证明数据链路和 CCS asset indexing 可用；
但 oldsplit temporal checkpoint 不满足主实验需要。
```

## 3.3 newsplit checkpoint 与 split 策略确认

完成内容：

```text
确认 newsplit checkpoint 包含 query_update 和 stream_fusion_neck
确认 old CCS seeds 在 newsplit val 中覆盖很小
将 oldsplit assets 降级为工程 sanity / overlap sanity
```

当前主策略：

```text
official newsplit temporal StreamMapNet
newsplit val data
重新构建 asymmetric temporal evaluation set
```

## 3.4 Phase 1.0 overlap sanity

使用：

```text
old CCS overlap subset
5 samples
```

完成：

```text
clean hook sanity
reset sanity
attack-at-t dry run
matched reset ablation
map-level validation
```

关键发现：

```text
internal temporal-state residue 可复现；
map-level residue 在 4/5 samples 出现；
reset_all 清除 internal 与 map-level residue；
reset_BEV 基本清除 map-level target-boundary residue；
reset_query 清除 query/pred internal residue，但不清除 map-level boundary residue。
```

机制初步修正为：

```text
query memory：主要解释 internal query/pred residue
BEV memory：主导 map-level boundary geometry residue
```

## 3.5 Phase 1.1 / 1.2：newsplit candidate 初筛与 broad pool 诊断

路线：

```text
newsplit val
ccs_asymmetric_dist
ccs_candidate
ETA-like / heuristic location
VPA filter
controlled ablation
```

发现：

```text
full 40-frame set:
  internal residue 强复现
  map-level residue 弱正向且 scene-sensitive

high-VPA subset:
  map-level residue 增强
  power=6000 比 3000 增强 t+2 persistence
  power=9000 无额外收益

expanded ccs_candidate broad pool:
  unconditional map-level effect weak
  VPA alone 不足以定义 attack-effective set
```

关键诊断：

```text
attack-frame corruption 是最强解释变量；
如果 frame t 没有产生 target-boundary corruption，
后续 recovery residue 自然不稳定。
```

## 3.6 Phase 1.3：gate-based sample construction

目标：

```text
把 broad pool、attack-effective set、pre-attack high-quality candidate set 分离。
```

输出集合：

```text
broad_report_set:
  55 frames / 11 scenes
  用于 broad-pool robustness report
  unconditional map-level effect weak

attack_effective_set_delta001:
  22 frames / 9 scenes
  用于 conditional temporal-residue mechanism analysis

high_quality_candidate_set:
  12 frames / 4 scenes
  太小，不适合作为主统计集
```

结论：

```text
Phase 1.3 成功把问题从“有没有 residue”
推进到“什么条件下 residue 可稳定观察”。
```

## 3.7 Phase 1.4 / 1.5：geometry gate refinement 与视觉审查

人工审查发现：

```text
部分 false positive / false negative 由视觉结构误判导致；
大量直路、重复 scene、不符合 asymmetric 条件的样本会稀释结果；
部分位置选在车、人行道、电线杆、道路中心等，不适合主实验。
```

因此决定：

```text
先严格迁移 CCS-style data selection / location search / rendering；
再重构数据池。
```

这一步是当前后续主结果变强的关键转折。

## 3.8 Phase 1.6 - 1.8B：严格 CCS-style 链路迁移与 selected114

当前主链路：

```text
CCS-style rule-based asymmetric pool rebuild
↓
人工确认 sampled scenes 非对称
↓
W=10, L=9 temporal eligibility
↓
selected114:
  114 frames / 38 scenes
  max 5 frames per scene
↓
CCS dense location candidates
  top-400 geometric candidates per frame
↓
StreamMapNet frame-t model scoring
  six-camera CCS rendering
  power=3000
  score = frame-t Delta CD to diverging boundary
↓
merge best model-scored loc
↓
controlled temporal evaluation
  clean_keep
  clean_reset_all
  clean_reset_query
  clean_reset_BEV
  attack_keep
  attack_reset_all
  attack_reset_query
  attack_reset_BEV
↓
matched clean-reset baseline summary
```

重要实现细节：

```text
攻击只在 target frame t 打开；
warm-up frames clean；
recovery frames clean；
所有六个 camera 在 frame t 按物理投影渲染；
不写入原 CCS 项目目录；
所有输出在 /data/dj/MapEcho/artifacts 下。
```

---

## 4. 当前使用的数据与模型

### 模型

```text
StreamMapNet official newsplit temporal checkpoint
ckpt:
  /home/dj/MapEcho/ckpts/nusc_newsplit_480_60x30_24e.pth

config:
  /home/dj/MapEcho/src/StreamMapNet/plugin/configs/mapecho_nusc_newsplit_480_60x30_24e_eval.py
```

### 数据

```text
nuScenes newsplit val
W=10, L=9
selected114 = 114 frames / 38 scenes
```

### 当前主要输入文件

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_assets_merged.csv

/data/dj/MapEcho/artifacts/phase1_8b_downstream/model_scoring_fast_top400_selected114/ccs_model_scored_top400_selected114_tokens.txt
```

### 当前主要结果目录

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/top400_selected114_controlled_check
```

### 论文证据打包目录

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_9_paper_evidence
```

### qualitative 图目录

```text
/data/dj/MapEcho/artifacts/phase1_8b_downstream/phase1_10_qualitative_figures
```

---

## 5. 当前使用的实验条件

每个 target sequence 跑 8 组：

```text
clean_keep
clean_reset_all
clean_reset_query
clean_reset_BEV

attack_keep
attack_reset_all
attack_reset_query
attack_reset_BEV
```

reset timing：

```text
warm-up: clean
frame t: clean or perturbed
reset condition: before t+1 生效
recovery t+1 ... t+L: clean
```

matched baseline：

```text
attack_keep        vs clean_keep
attack_reset_all   vs clean_reset_all
attack_reset_query vs clean_reset_query
attack_reset_BEV   vs clean_reset_BEV
```

这点非常关键，因为它避免把 reset 本身造成的 cold-start / distribution shift 误当成 perturbation residue。

---

## 6. 当前主要指标

## 6.1 Target-boundary Chamfer Delta

当前主指标：

```text
Delta CD to diverging boundary =
  CD_attack_condition_to_diverge - CD_matched_clean_condition_to_diverge
```

正值表示：

```text
perturbation condition 比 matched clean baseline 更远离 diverging GT boundary
```

当前 positive threshold：

```text
Delta CD > 0.01 m
```

用于 positive residue rate。

## 6.2 frame-t location scoring metric

在位置搜索中，对每个 candidate：

```text
frame t 渲染六相机
跑 StreamMapNet forward
计算 target-boundary Delta CD to diverging boundary
选择 Delta CD 最大的位置
```

当前 top-400 scoring 结果：

```text
114 / 114 samples completed
114 / 114 frame-t Delta CD > 0
median frame-t Delta CD = +0.0954 m
best-rank median = 73.5
best-rank max = 381
best-rank >= 360: 5 samples
```

含义：

```text
top-400 是必要的；
top-100 会漏掉部分有效位置。
```

## 6.3 reset reduction / internal metrics

内部机制指标包括：

```text
query_score_mean_abs_reduction
pred_vector_mean_abs_reduction
topk_embedding_mean_abs_reduction
fused_bev_norm_delta_reduction
```

解释：

```text
reset_all = 1.0:
  temporal-state residue 被完全清除

reset_query 高 query/pred reduction, fused-BEV reduction = 0:
  query path 被切断，但 BEV path 保留

reset_BEV fused-BEV reduction = 1.0:
  BEV temporal path 被切断
```

## 6.4 scene-clustered bootstrap

由于同一 scene 可能有多个 frames，不能把 114 frames 全当独立样本。

当前使用：

```text
frame-level descriptive stats
scene-level clustered bootstrap CI
one-primary-frame-per-scene conservative analysis
```

## 6.5 qualitative visualization 指标

当前 qualitative 图包含：

```text
six-camera target-frame rendered input
map overlay at t
map overlay at t+1
map overlay at t+2
```

但目前发现默认 overlay 不够适合作为最终论文图，因为：

```text
没有高亮 metric-selected best boundary；
多条 high-score prediction 淡色叠加，肉眼难以分辨 condition 差异；
部分 case clean 本身已经明显不理想。
```

因此新增 focused diagnostic panel：

```text
只高亮 metric-selected best boundary
同时显示 clean_keep / attack_keep / reset_all / reset_query / reset_BEV
```

---

## 7. 当前主结果

## 7.1 selected114 主结果

```text
114 frames / 38 scenes
W=10, L=9
renderer = CCS six-camera renderer
power = 3000
```

Map-level target-boundary residue：

| Condition | Offset | Median Delta CD | Positive Rate |
| --- | ---: | ---: | ---: |
| attack_keep | t+1 | +0.0333 m | 82/114 = 71.9% |
| attack_keep | t+2 | +0.0179 m | 69/114 = 60.5% |
| attack_reset_all | t+1 | 0.0000 m | 0/114 |
| attack_reset_all | t+2 | 0.0000 m | 0/114 |
| attack_reset_BEV | t+1 | +0.00008 m | 10/114 = 8.8% |
| attack_reset_BEV | t+2 | -0.00001 m | 3/114 = 2.6% |
| attack_reset_query | t+1 | +0.0374 m | 78/114 = 68.4% |
| attack_reset_query | t+2 | +0.0174 m | 67/114 = 58.8% |

Scene-clustered bootstrap：

```text
attack_keep t+1:
  median CI = [+0.0179, +0.0481]
  positive-rate CI = [59.5%, 83.3%]

attack_keep t+2:
  median CI = [+0.0095, +0.0289]
  positive-rate CI = [48.8%, 71.3%]
```

## 7.2 one-primary-frame-per-scene 保守分析

```text
38 frames / 38 scenes
```

结果：

| Condition | Offset | Median Delta CD | Positive Rate |
| --- | ---: | ---: | ---: |
| attack_keep | t+1 | +0.0410 m | 26/38 = 68.4% |
| attack_keep | t+2 | +0.0194 m | 24/38 = 63.2% |
| attack_reset_all | t+1 | 0.0000 m | 0/38 |
| attack_reset_all | t+2 | 0.0000 m | 0/38 |
| attack_reset_BEV | t+1 | +0.0006 m | 5/38 = 13.2% |
| attack_reset_BEV | t+2 | -0.00005 m | 2/38 = 5.3% |
| attack_reset_query | t+1 | +0.0373 m | 27/38 = 71.1% |
| attack_reset_query | t+2 | +0.0184 m | 24/38 = 63.2% |

含义：

```text
主结果不是仅由同一 scene 的重复 frames 驱动。
```

## 7.3 internal mechanism

```text
reset_all:
  query / pred / embedding / fused-BEV reductions = 1.0

reset_query at t+1:
  query-score reduction median = 0.923
  pred-vector reduction median = 0.982
  fused-BEV reduction median = 0.0

reset_BEV:
  fused-BEV reduction median = 1.0
  query / pred reductions partial but substantial
```

机制解释：

```text
BEV memory dominates map-level target-boundary geometry residue.
Query memory mainly carries immediate internal query/prediction residue.
Reset-all closes the total temporal-state causal loop.
```

---

## 8. 当前 qualitative 状态与问题

Phase 1.10 生成了 15 个默认 qualitative cases：

```text
top_residue: 5
median_residue: 2
reset_bev_clear_removal: 3
weak_or_failure: 5
```

人工审查发现：

```text
大部分图例中 clean_keep / attack_keep / reset_* 肉眼看起来非常接近；
很多 case clean 本身也已经预测不理想；
目前目测只有 top-residue #3 比较符合实验展示需求。
```

进一步检查 top-residue #3：

```text
target_token = 4a1972f8731b4cdea40fc69a38a735b1

t+1:
  clean_keep CD = 5.0747
  attack_keep CD = 5.7014
  attack_keep delta = +0.6267
  attack_reset_all delta = 0.0000
  attack_reset_BEV delta = -0.0041
  attack_reset_query delta = +0.5530
```

结论：

```text
数值 reset pattern 真实存在；
但默认 qualitative overlay 表达不清楚；
top#3 可作为机制示例，但 clean 本身 CD 偏大，不适合作为“clean-perfect”示例。
```

因此，当前 qualitative 部分不能直接作为最终论文图，需要：

```text
重新筛 clean 更好、结构更清晰的 case；
使用 focused panel，而不是默认 5-column overlay；
人工确认最终 3-5 个图例。
```

---

## 9. 与最初 kill-or-continue 标准的对照

最初 Phase 1 强信号标准包括：

```text
attack-frame target-boundary error
recovery frames 有 residue
reset-all 明显降低 residue
reset-query / reset-BEV 解释 residue 来源
```

当前 selected114 对照：

| 标准 | 当前状态 |
| --- | --- |
| attack-frame corruption | PASS，114/114 frame-t Delta CD > 0 |
| t+1/t+2 recovery residue | PASS，attack_keep t+1/t+2 median positive |
| reset-all 清除 | PASS，t+1/t+2 positive = 0/114 |
| reset-BEV 解释 map-level residue | PASS，map-level residue 基本清除 |
| reset-query 解释 internal residue | PASS，query/pred reduction 高，但 map-level 保留 |
| scene-level robustness | PASS，38-scene primary analysis 仍 positive |
| qualitative paper figure | NOT READY，需重筛与重画 |

因此：

```text
定量机制链路满足继续写论文主结果的条件；
qualitative 展示仍是当前最明显短板。
```

---

## 10. 当前项目的风险点

### 10.1 clean-quality 风险

部分 qualitative case 中：

```text
clean prediction 已经不够贴合 target boundary；
这会削弱论文图的说服力。
```

尽管 matched delta 指标仍有效，但视觉示例需要更严格 clean-quality。

### 10.2 qualitative selection 风险

当前自动选出的 top residue case 不一定最适合论文展示，因为：

```text
top residue 可能来自 clean already bad 或场景复杂；
最大数值不等于最清晰图例。
```

后续应单独定义：

```text
qualitative-friendly case selection
```

而不是直接用 top residue。

### 10.3 指标与视觉不一致风险

默认 map overlay 没有高亮 metric-selected best boundary，导致：

```text
图看起来与数值不一致；
审稿人可能误解为各 condition 没区别。
```

解决方向：

```text
论文图使用 focused best-boundary visualization；
caption 中明确 CD 计算对应哪条 boundary。
```

### 10.4 与最初 AUC_residue 计划的关系

最初计划主指标偏 AUC_residue / recovery curve。

当前已补充：

```text
t+1...t+9 recovery curve
AUC_CD = Σ max(0, Delta CD_i), i=1...9
```

但当前主指标仍建议保持：

```text
t+1/t+2 Delta CD
positive residue rate
reset ablation
scene-clustered CI
```

完整 recovery curve 和 AUC_CD 更适合作为 H3 supporting figure/table。

### 10.5 MapTR / planner 尚未做

最初方案包含：

```text
MapTR single-frame difficulty control
planner proxy
defense
```

当前尚未推进这些部分。

它们不影响 Phase 1 主机制结论，但影响论文完整度。

---

## 11. 当前阶段最合理的下一步选项

## 选项 A：先整理论文主结果，不再扩实验

适合条件：

```text
目标是先写出主实验章节和方法章节；
接受 qualitative 图后续再精修。
```

动作：

```text
整理 selected114 主表
整理 scene-clustered CI
整理 reset mechanism 表
整理 focused qualitative figure
写 method / result draft
```

优点：

```text
最快形成论文骨架；
当前定量证据已经足够强。
```

缺点：

```text
qualitative 图仍弱；
clean-quality 质疑需要文字解释。
```

## 选项 B：优先补 qualitative-friendly case

适合条件：

```text
你希望论文图非常直观；
不希望图上 clean 已经明显错。
```

动作：

```text
从 selected114 中筛：
  clean_keep CD 低
  attack_keep Delta CD 明显
  reset_BEV Delta 接近 0
  scene geometry 人工看起来清楚

生成 focused panels
人工选 3-5 个最终 case
```

优点：

```text
提高论文可读性与说服力。
```

缺点：

```text
需要额外筛图和人工审查；
可能不改变主统计结果。
```

## 选项 C：补 clean-quality filtered quantitative subset

适合条件：

```text
你担心 clean 本身质量会被审稿人质疑。
```

动作：

```text
定义 clean-quality gate：
  clean_keep CD_diverge_t/t+1/t+2 低于阈值或分位数
  recovery clean 稳定
  scene-level 不过度重复

在 clean-quality subset 上重复 summary：
  attack_keep
  reset_all
  reset_BEV
  reset_query
```

优点：

```text
直接回应 clean-quality 风险；
可能让 qualitative 和 quantitative 更一致。
```

缺点：

```text
需要小心避免被看成 post-hoc cherry-picking；
必须同时报告 full selected114。
```

## 选项 D：进入 MapTR / planner / defense

适合条件：

```text
当前主结果已完全认可；
准备扩论文贡献面。
```

建议暂缓。原因：

```text
当前 qualitative 与 clean-quality 问题还没完全收束；
先把主证据打磨稳定更重要。
```

---

## 12. 我的建议

当前最稳路线已经推进到：

```text
Priority 1 H3 recovery curve 已完成。
Phase 1.11：Qualitative-friendly and Clean-quality Subset Check 已完成。
```

Phase 1.11 已完成的内容：

```text
1. 用 H3 AUC_CD + t+1 Delta CD 重新筛 qualitative-friendly cases；
2. 生成 t+1 / t+2 recovery-focused panels；
3. 生成 clean-quality strict / relaxed subset summary；
4. 对 full selected114、clean-quality subset、qualitative subset 同时报告 AUC_CD；
5. 保持 selected114 full-set 为主定量集，避免 cherry-picking。
```

Phase 1.11 没有重新跑模型，只重用已有 H3 outputs 与 metrics。

实际筛选规则：

```text
candidate qualitative case =
  attack_keep t+1 Delta CD > 0.05
  attack_keep AUC_CD > 0.15
  attack_reset_all AUC_CD = 0
  attack_reset_BEV AUC_CD < 0.02
  attack_reset_query AUC_CD > 0.10
  clean_keep CD_diverge_t+1 在 selected114 中相对较低
  clean recovery CD std 在 selected114 中相对较低
```

Phase 1.11 输出结果：

```text
H3 qualitative signal pass = 31 / 114
clean_quality_strict_pass = 30 frames / 13 scenes
clean_quality_relaxed_pass = 68 frames / 26 scenes
qualitative_strict_pass = 11 frames / 7 scenes
qualitative_relaxed_pass = 24 frames / 14 scenes
recovery-focused panels = 11
```

clean-quality robustness 结果：

```text
clean_quality_strict:
  attack_keep t+1 median Delta CD = +0.0514 m, positive = 24/30
  attack_keep t+2 median Delta CD = +0.0301 m, positive = 21/30
  attack_keep median AUC_CD = 0.1629
  attack_reset_all median AUC_CD = 0
  attack_reset_BEV median AUC_CD = 0.0020
  attack_reset_query median AUC_CD = 0.1679
  AUC_CD > 0.03:
    attack_keep = 26/30
    attack_reset_all = 0/30
    attack_reset_BEV = 3/30
    attack_reset_query = 25/30

clean_quality_relaxed:
  attack_keep t+1 median Delta CD = +0.0455 m, positive = 51/68
  attack_keep t+2 median Delta CD = +0.0261 m, positive = 45/68
  attack_keep median AUC_CD = 0.1418
  attack_reset_all median AUC_CD = 0
  attack_reset_BEV median AUC_CD = 0.0020
  attack_reset_query median AUC_CD = 0.1496
  AUC_CD > 0.03:
    attack_keep = 57/68
    attack_reset_all = 0/68
    attack_reset_BEV = 6/68
    attack_reset_query = 58/68
```

这说明：

```text
即使只看 clean prediction 较好且 recovery 较稳定的 subset，
attack-off recovery AUC 仍然存在；
reset_all 清零；
reset_BEV 基本清除；
reset_query 保留 map-level recovery AUC。
```

当前接下来最实际的动作是：

```text
1. 人工审查 11 个 recovery-focused panels；
2. 选出 3-5 个最终论文 qualitative cases；
3. 如果 11 个 strict panels 里仍不够清楚，再从 24 个 relaxed cases 中补图；
4. 保留 full selected114 作为主定量结果，clean-quality subset 作为 robustness check。
```

最终论文叙事应保留两层：

```text
定量主结果：
  selected114 / 38 scenes full-set

定性图例：
  从 full-set 中挑 visually clear representative cases
```

这样最干净，也最不容易被质疑。
