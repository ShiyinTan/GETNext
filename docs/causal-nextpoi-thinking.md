# 在 GETNext 上用因果视角做 Next-POI 预测：思考笔记

本文档记录如何在现有 GETNext（Trajectory Flow Map + Transformer）上，用**因果去偏**思路缓解三类系统性偏差：

1. **可达性 / 距离偏差**（Accessibility / Distance bias）
2. **流行度偏差**（Popularity bias）
3. **区域属性混淆**（Area / Land-use confounding）

目标不是另起一套全新模型，而是：**明确“用户偏好”与“环境约束/曝光机制”的因果结构**，再把可落地的干预接到 GETNext 现有模块上（GCN 节点特征、轨迹流图 `A`、`NodeAttnMap`、Transformer logits、训练损失与评估）。

---

## 1. 问题从哪里来：观测数据 ≠ 用户偏好

Next-POI 训练数据是**观测到的 check-in**，不是“用户在无约束下自由选择”的结果。一次签到大致可写成：

\[
\text{Visit}(u, p, t) \;=\; f\big(\underbrace{Pref(u,p)}_{\text{想去}},\;
\underbrace{Access(u,p,t)}_{\text{能不能去}},\;
\underbrace{Expo(p,t)}_{\text{被曝光到}},\;
\underbrace{Area(p),\,Context(t)}_{\text{场景}}\big) + \epsilon
\]

标准 MLE / CrossEntropy 直接拟合 \(P(\text{next POI} \mid \text{history})\)，会把 **Access / Expo / Area** 的效应一并学进“用户偏好”和“POI 表征”里。于是：

| 现象 | 数据侧机制 | 模型侧表现（GETNext） |
|------|------------|------------------------|
| 近距离主导 | 多数转移发生在短距离；远距离访问稀疏 | GCN/`NodeAttnMap` 被高频近邻边主导；Transformer 也学到“靠近当前点就高分” |
| 热门主导 | `checkin_cnt` 高的 POI 在图与序列中反复出现 | 节点特征含 `checkin_cnt`；解码器偏向热门 logits |
| 区域模式主导 | 商圈/居民区决定“白天去哪、晚上回哪” | 经纬度+类别被当成偏好，实则是土地用途与作息的混杂 |

对**偶尔远行**、**反潮流个人兴趣**、**跨区非常规转移**的用户，整体 Acc@k 可能仍好看，但这类长尾样本几乎被忽略。

---

## 2. 对齐 GETNext 的因果图（Conceptual SCM）

把当前管线拆成可干预的变量会更清晰：

```text
                    ┌──────────────┐
   Area(p) ────────►│  Access(u,p,t) │◄──── Distance / 交通 / 时间窗
                    └──────┬───────┘
                           │
 Popularity(p) ──► Expo(p)─┼──────────────► Observed Transition / Check-in
                           │                        │
 User Pref(u,·) ───────────┘                        ▼
                                           Trajectory Flow Map A
                                           Node feats X (含 checkin_cnt, lat/lon, cat)
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                            GCN               NodeAttnMap            Transformer
                              └──────────► fused seq embed ──────────────┘
                                                    │
                                                    ▼
                                      y_pred = Transformer + attn_map[cur]
```

要点：

- **想估计的因果量**：在给定历史与当前上下文下，用户对候选 POI 的**偏好效应** \(P(\text{visit} \mid do(Pref),\; context)\)，或至少分离出“偏好驱动的分数”。
- **不该直接当偏好的量**：全局转移频次 `A`、节点 `checkin_cnt`、纯地理邻近、区域土地用途带来的“必经/常驻”模式。
- GETNext 的 `adjust_pred_prob_by_graph`（`attn_map[cur] + transformer_logits`）本质上是把**群体流动先验**硬加进预测；若 `A` 与近邻/热门高度相关，偏差会被**放大二次**（图侧一次 + logits 侧一次）。

---

## 3. 可达性（Accessibility）：邻近不等于偏好

### 3.1 因果表述

- **混杂 / 中介**：距离既限制可达性，又与居住地、工作地相关，从而与“真偏好”相关。
- **选择偏差**：远距离正样本极少 → 模型把“远 = 低分”当成稳定规律。
- **想要的反事实**：若把候选 \(p\) 的可达性提升到与近邻相当（\(do(\text{Access}=1)\)），用户是否仍会选它？若会，说明存在真实兴趣，而不只是“碰巧近”。

### 3.2 可落地思路（由浅到深）

#### A. 显式可达性因子 + 残差偏好（推荐优先试）

把下一 POI 分数拆成：

\[
s(u,p,t) = s_{\text{pref}}(u,p,t) + \lambda(t)\, s_{\text{access}}(u,p,t)
\]

- \(s_{\text{access}}\)：由距离、时间预算、历史活动半径、区域连通性等**非学习或弱学习**的项给出（可微或后处理均可）。
- \(s_{\text{pref}}\)：才是 Transformer / 用户嵌入要学的部分。
- 训练时可用 **backdoor / residual learning**：先拟合或固定 access 项，再让主模型拟合残差标签分布；或对 access 项 stop-grad，避免主网偷懒全靠距离。

接到 GETNext：

- 不要让 `NodeAttnMap` 无约束地吸收全部地理邻近；可对 `A` 做**距离归一化边权**（同距离桶内再比转移强度），或把 `e_ij * A_ij` 改成 `e_ij * g(A_ij / f(dist_ij))`。
- 解码时：`y = s_pref + α * s_graph + β * s_access`，并对 α、β 做消融，观察远距离命中是否上升。

#### B. 距离分层 / Inverse Propensity on Distance

把每次转移按距离分桶（如 0–0.5km / 0.5–2km / 2–5km / >5km），估计“出现在该桶的倾向” \(\pi(d)\)，训练权重：

\[
w = \frac{1}{\pi(d)^\gamma}
\]

远距离样本上权，迫使模型在稀有远跳上也产生梯度。这是对 **selection into nearby transitions** 的近似 IPS。

注意：IPS 方差大，需裁剪 \(w\)、或用自归一化 IPS；并按用户或轨迹做分层，避免被少数极端远跳主导。

#### C. 反事实数据增强（Counterfactual augmentation）

在保持类别/用户偏好信号的前提下，构造“若用户当时在另一位置”的轨迹：

- **空间平移 / 镜像**：把一段轨迹整体平移到另一区域，标签随 POI 映射到同类别、相似功能的候选（难，需 POI 对齐）。
- 更现实的变体：**硬负样本挖掘**——对每个正样本，采样同距离环带内的其他 POI 作强负例，让模型在“一样近”的集合里学偏好，而不是学“近 vs 远”。

同距离环带内对比学习，往往比盲目远距离上采样更稳。

#### D. 评估必须切开距离

整体 Acc@k 会被近邻样本淹没。至少报告：

- Acc@k \| 距离桶
- 远距离召回 / 远距离 nDCG
- “下一跳超出用户历史 P95 半径”子集上的指标

否则任何去偏是否有效都看不出来。

---

## 4. 流行度（Popularity）：热门曝光 ≠ 个人兴趣

### 4.1 因果表述

- **曝光机制**：越热门越容易出现在别人的轨迹里 → 进入全局图 `A` 与节点特征 `checkin_cnt` → GCN/`NodeAttnMap` 对热门节点给出更高结构分。
- **反馈回路**：模型再推荐热门 → 若用于在线系统会加剧马太效应（离线训练里则是拟合历史马太效应）。
- **异质性**：对“从众型”用户，流行度是有效信号；对“小众兴趣型”用户，流行度是有害混杂。需要的是 **条件效应**，而不是全局减流行度。

### 4.2 可落地思路

#### A. 两塔 / 两头分解：匹配分 + 流行度分

\[
s(u,p) = s_{\text{match}}(u,p) + g(\text{pop}(p),\, u)
\]

- \(s_{\text{match}}\)：用户–POI 个性化匹配（应用户兴趣）。
- \(g\)：可学习的、依赖用户类型的流行度门控（对部分用户 \(g\) 大，对小众用户 \(g\) 接近 0）。

GETNext 中：

- 节点特征里的 `checkin_cnt` 建议：**标准化 / log1p 后单独通道**，或从 GCN 输入中移出，改为解码期加性偏置，便于做因果消融（`do(pop=0)` ≈ 去掉该偏置看排序变化）。
- `NodeAttnMap` 用的 `A` 含转移频次，天然偏热门；可对行做 **流行度折减**：\(A'_{ij} = A_{ij} / \text{pop}(j)^\alpha\)，再进入注意力调制。

#### B. 倾向评分（Propensity）加权损失

令 \(\pi(p) \approx P(\text{exposed/clicked} \mid p) \propto \text{pop}(p)\)（或用更细的 user–POI 倾向模型），对 CE 做：

\[
\mathcal{L} = \sum -\frac{1}{\pi(p^+)^\gamma}\log \frac{e^{s_{p^+}}}{\sum_q e^{s_q}}
\]

也可只在负采样分布上做 **popularity-aware negative sampling 的逆操作**：负样本按热门过度采样，正样本按热门降权，迫使 match 头区分“热但我不喜欢”。

#### C. 因果表示：不变偏好子空间

用环境划分做不变学习（Invariant Risk Minimization / Group DRO 的简化版）：

- 环境例子：工作日 vs 周末、高流行度时段 vs 低、不同 borough。
- 要求用户嵌入中的“兴趣子空间”在多环境下都能预测下一 **类别** 或下一 **功能型 POI**，而对绝对 POI_id 的依赖可环境变化。

直觉：个人兴趣对“咖啡 vs 酒吧”相对稳定；对“哪家网红店”随流行度环境变化。多环境一致性惩罚有助于剥离流行度。

#### D. 反事实推理问题（用于分析与 re-ranking）

对同一用户历史，问：

- 若将候选集中所有 POI 的流行度设为同一常数，排序如何变？
- 哪些用户的 top-k 几乎不变（兴趣主导）？哪些剧烈变成长尾（说明原模型在吃流行度）？

这类 `do(pop)` 探针不改训练也能诊断；再决定是否加门控或 IPS。

---

## 5. 区域属性（Area）：商圈 vs 居民区等

### 5.1 为什么要单独建区

距离与流行度都还不够：

- 同样 1km，从居民区到地铁站 vs 从商圈到另一商圈，语义完全不同。
- “回家”“去上班”“周末逛街”是 **area-conditioned** 的转移模式，不是单纯 POI 偏好。
- 若不建模 area，GCN 会用 lat/lon 隐式拟合区域簇，但无法区分“区域约束”与“兴趣”。

### 5.2 Area 变量怎么定义（数据侧）

在 NYC 上可从粗到细：

1. **网格 / geohash / census tract**（纯空间分区）
2. **功能区标签**（residential / commercial / nightlife / transit hub…）  
   - 可用 POI 类别聚合：某格子内 Food/Shop/Office/Home-related 的比例 → soft land-use 向量
3. **用户个人锚点**：家/公司的推断（夜间众数格子、工作日白天众数格子）

建议把 area 做成：

- POI 的 `area_id` / `area_feat`
- 转移的 `area_i → area_j` 边（区域流图），与 POI 流图 `A` 分层

### 5.3 因果角色

Area 常常是 **混杂因子** 或 **效应修饰因子（moderator）**：

- 混杂：住在商圈附近的人更常打卡热门店 → 看起来像“喜欢热门”，其实是居住 area 的效应。
- 修饰：同一用户在“工作区午餐时间”与“居住区晚间”的偏好机制不同 → 应估计条件效应 \(Pref(u,p \mid area, t)\)。

对应干预：

- 控制 area：在同一 `(from_area, time_bin)` 内比较候选 POI（分层或条件建模）。
- 跨 area 的远跳：单独评估，避免被区内短跳指标掩盖。

### 5.4 接到 GETNext 的结构想法

1. **分层图**  
   - POI 层：现有 trajectory flow map  
   - Area 层：area 转移图 + area 嵌入  
   - 消息传递：POI embed ← 自身 + 所属 area embed；`NodeAttnMap` 增加 area 兼容项（跨区惩罚或跨区门控，而非一刀切禁止）

2. **解码分解**  
   \[
   s(p) = s_{\text{poi-pref}} + s_{\text{area-transit}}(a_{\text{cur}}\!\to\! a_p) + s_{\text{access}} + s_{\text{pop}}
   \]
   Transformer 主学 `s_poi-pref`；area-transit 可用小参数表或第二套轻量 GCN。

3. **条件归一化**  
   在 `(user, from_area, hour)` 条件下对候选做 softmax，相当于在同一场景内排序，削弱“永远推荐大商圈”的全局偏置。

---

## 6. 三者一起建模时的统一框架

可达性、流行度、区域不是三个独立补丁，而是同一生成过程的不同外生/中介变量。一个可操作的统一分数：

\[
\begin{aligned}
s(u,p,t)
&= \underbrace{s_{\theta}(u,p,t)}_{\text{个性化偏好（主模型）}}
+ \underbrace{\beta_a\, s_{\text{access}}(u,p,t)}_{\text{可达性}}
+ \underbrace{\beta_p(u)\, s_{\text{pop}}(p,t)}_{\text{流行度（用户门控）}}
+ \underbrace{\beta_r\, s_{\text{area}}(a_u,a_p,t)}_{\text{区域转移}}
\end{aligned}
\]

训练目标建议：

1. **主损失**：下一 POI CE（可 IPS 加权：距离桶 × 流行度倾向）。
2. **辅助损失**（GETNext 已有）：时间、类别 —— 类别头有助于兴趣信号，可对类别 CE **提高权重**，对 POI_id 头做去偏，形成“先功能、后具体地点”的两阶段因果直觉。
3. **去混淆正则**：  
   - 惩罚 \(s_\theta\) 与 `pop`、与 `dist` 的全局相关（软正交）；或  
   - adversarial：额外判别器想从 \(s_\theta\) 预测 pop/dist 桶，主模型最大化其损失（需小心伤及有用信息，最好只 adversarial 掉“与 label 无关的部分”）。
4. **反事实一致性**：对同一历史，扰动 pop/area 编码后，类别预测应更稳，POI_id 预测允许变。

推理时可提供两种模式：

- **写实模式**：保留 access/pop/area（贴近真实下一跳，刷线上指标）。
- **偏好模式**：`β_a, β_p` 缩小或 `do(pop=c)`（更适合“猜兴趣 / 探索推荐”）。

这对研究“模型到底学到了什么”很有价值。

---

## 7. 与 GETNext 模块的具体挂钩清单

| 模块 | 现状 | 因果向改动 |
|------|------|------------|
| `graph_X` / `checkin_cnt` | 直接进 GCN | 拆出为流行度偏置；GCN 更侧重 cat + 去中心化地理/区域特征 |
| `graph_A` | 原始转移频次 | 距离桶内归一化；`/ pop(j)^α`；或构建 access-conditioned 图 |
| `NodeAttnMap` | `e * (A+1)` 加强群体流 | 改为偏好注意力 × 结构门控；避免双重计入热门近邻 |
| `UserEmbeddings` | 单一用户向量 | 拆成兴趣子空间 + 从众/活跃度子空间；后者可与 pop 门控共享 |
| `TransformerModel` | 三头 POI/time/cat | 强化 cat 头；POI 头用去偏损失；可加 area 条件 |
| `adjust_pred_prob_by_graph` | 直接加 attn_map | 拆成 access / area / residual-flow 三项可开关 |
| 评估 | 全局 Acc@k / MRR | 加距离桶、流行度四分位、跨 area、小众用户子集 |

---

## 8. 建议的实验顺序（由易到难）

不必一上来上完整 SCM。建议：

1. **诊断**  
   - 统计训练转移距离分布、命中样本的距离分布、top-k 候选的平均 pop。  
   - `do(pop)` / 去掉 `checkin_cnt` / 去掉 `NodeAttnMap` 的消融，看指标与长尾子集变化。

2. **轻量干预**  
   - 距离分层 IPS + 流行度降权 CE。  
   - `A` 的 pop/距离归一化。  
   - 同距离环带负采样。

3. **结构分解**  
   - \(s = s_{\text{pref}} + s_{\text{access}} + s_{\text{pop}}(u) + s_{\text{area}}\)。  
   - area 特征与区域流图。

4. **反事实评估协议**  
   - 固定报告：整体指标 + 远跳 + 长尾 POI + 跨 area + 低从众用户组。  
   - 避免只看被近邻/热门刷高的 Acc@1。

---

## 9. 风险与边界

- **过度去偏**：可达性与流行度在真实世界里*确实*影响下一跳；完全 `do(access=1), do(pop=0)` 的预测会不切实际。要分清任务：是“预测真实下一签到”还是“推断潜在兴趣”。
- **不可识别性**：没有随机实验或强工具变量时，偏好与居住地/常驻 area 无法完美分开；应用条件化与敏感性分析（改变 β 看排序稳定性）。
- **方差**：IPS 与长尾上采样可能伤整体 Acc；需用多目标或 Group DRO，保证近邻主群体不明显崩坏。
- **图泄漏**：`A` 若含验证/测试时段转移，去偏结论会偏乐观；应严格只用训练期构图（当前 `build_graph.py` 流程需保持这一点）。

---

## 10. 一句话收束

GETNext 很强地拟合了**群体轨迹流 + 序列上下文**，但也因此把**邻近可达、热门曝光、区域土地用途**写进了“偏好”。用因果视角，不是扔掉这些信号，而是把它们从偏好里**显式剥离成可干预项**，让主模型学残差兴趣，并用距离/流行度/跨 area 的切片评估去检验：模型是否还能看见那些“偶尔走远、偏爱小众、跨区行动”的用户。
