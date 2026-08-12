# 在 GETNext 上用因果视角做 Next-POI 预测：思考笔记

本文档记录如何在现有 GETNext（Trajectory Flow Map + Transformer）上，用**因果去偏**思路缓解三类系统性偏差：

1. **可达性 / 距离偏差**（Accessibility / Distance bias）
2. **流行度偏差**（Popularity bias）
3. **区域属性混淆**（Area / Land-use confounding）

目标不是另起一套全新模型，而是：**明确“用户偏好”与“环境约束/曝光机制”的因果结构**，再把可落地的干预接到 GETNext 现有模块上（GCN 节点特征、轨迹流图 `A`、`NodeAttnMap`、Transformer logits、训练损失与评估）。

去偏路线不限于 IPS：更推荐 **SCM 因果图 → deconfounded training / 表征解耦**（详见 §6）；IPS 可作为稀有样本上的辅助稳定器。

> **公式说明**：正文数学使用 GitHub 友好的 `$...$`（行内）与 `$$...$$`（独立公式）。若本地预览不渲染，请用支持 KaTeX/MathJax 的查看器，或直接在 GitHub 上打开本文件。

---

## 1. 问题从哪里来：观测数据 ≠ 用户偏好

Next-POI 训练数据是**观测到的 check-in**，不是“用户在无约束下自由选择”的结果。一次签到大致可写成：


$$
\mathrm{Visit}(u,p,t)=f\big(\mathrm{Pref}(u,p),\;\mathrm{Access}(u,p,t),\;\mathrm{Expo}(p,t),\;\mathrm{Area}(p),\;\mathrm{Context}(t)\big)+\epsilon
$$


其中各项含义：

- $\mathrm{Pref}(u,p)$：想去（用户偏好）
- $\mathrm{Access}(u,p,t)$：能不能去（可达性）
- $\mathrm{Expo}(p,t)$：被曝光到（流行度/曝光）
- $\mathrm{Area}(p),\mathrm{Context}(t)$：场景（区域与上下文）

标准 MLE / CrossEntropy 直接拟合 $P(Y\mid H)$（下一 POI | 历史），会把 **Access / Expo / Area** 的效应一并学进“用户偏好”和“POI 表征”里。于是：

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

- **想估计的因果量**：在给定历史与当前上下文下，用户对候选 POI 的**偏好效应** $P(\mathrm{visit} \mid do(Pref),\; context)$，或至少分离出“偏好驱动的分数”。
- **不该直接当偏好的量**：全局转移频次 `A`、节点 `checkin_cnt`、纯地理邻近、区域土地用途带来的“必经/常驻”模式。
- GETNext 的 `adjust_pred_prob_by_graph`（`attn_map[cur] + transformer_logits`）本质上是把**群体流动先验**硬加进预测；若 `A` 与近邻/热门高度相关，偏差会被**放大二次**（图侧一次 + logits 侧一次）。

---

## 3. 可达性（Accessibility）：邻近不等于偏好

### 3.1 因果表述

- **混杂 / 中介**：距离既限制可达性，又与居住地、工作地相关，从而与“真偏好”相关。
- **选择偏差**：远距离正样本极少 → 模型把“远 = 低分”当成稳定规律。
- **想要的反事实**：若把候选 $p$ 的可达性提升到与近邻相当（$do(\mathrm{Access}=1)$），用户是否仍会选它？若会，说明存在真实兴趣，而不只是“碰巧近”。

### 3.2 可落地思路（由浅到深）

#### A. 显式可达性因子 + 残差偏好（推荐优先试）

把下一 POI 分数拆成：


$$
s(u,p,t) = s_{\mathrm{pref}}(u,p,t) + \lambda(t)\, s_{\mathrm{access}}(u,p,t)
$$


- $s_{\mathrm{access}}$：由距离、时间预算、历史活动半径、区域连通性等**非学习或弱学习**的项给出（可微或后处理均可）。
- $s_{\mathrm{pref}}$：才是 Transformer / 用户嵌入要学的部分。
- 训练时可用 **backdoor / residual learning**：先拟合或固定 access 项，再让主模型拟合残差标签分布；或对 access 项 stop-grad，避免主网偷懒全靠距离。

接到 GETNext：

- 不要让 `NodeAttnMap` 无约束地吸收全部地理邻近；可对 `A` 做**距离归一化边权**（同距离桶内再比转移强度），或把 `e_ij * A_ij` 改成 `e_ij * g(A_ij / f(dist_ij))`。
- 解码时：`y = s_pref + α * s_graph + β * s_access`，并对 α、β 做消融，观察远距离命中是否上升。

#### B. 距离分层 / Inverse Propensity on Distance

把每次转移按距离分桶（如 0–0.5km / 0.5–2km / 2–5km / >5km），估计“出现在该桶的倾向” $\pi(d)$，训练权重：


$$
w = \frac{1}{\pi(d)^\gamma}
$$


远距离样本上权，迫使模型在稀有远跳上也产生梯度。这是对 **selection into nearby transitions** 的近似 IPS。

注意：IPS 方差大，需裁剪 $w$、或用自归一化 IPS；并按用户或轨迹做分层，避免被少数极端远跳主导。

#### C. 反事实数据增强（Counterfactual augmentation）

在保持类别/用户偏好信号的前提下，构造“若用户当时在另一位置”的轨迹：

- **空间平移 / 镜像**：把一段轨迹整体平移到另一区域，标签随 POI 映射到同类别、相似功能的候选（难，需 POI 对齐）。
- 更现实的变体：**硬负样本挖掘**——对每个正样本，采样同距离环带内的其他 POI 作强负例，让模型在“一样近”的集合里学偏好，而不是学“近 vs 远”。

同距离环带内对比学习，往往比盲目远距离上采样更稳。

#### D. 评估必须切开距离

整体 Acc@k 会被近邻样本淹没。至少报告：

- Acc@k（按距离桶分层）
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


$$
s(u,p) = s_{\mathrm{match}}(u,p) + g(\mathrm{pop}(p),\, u)
$$


- $s_{\mathrm{match}}$：用户–POI 个性化匹配（应用户兴趣）。
- $g$：可学习的、依赖用户类型的流行度门控（对部分用户 $g$ 大，对小众用户 $g$ 接近 0）。

GETNext 中：

- 节点特征里的 `checkin_cnt` 建议：**标准化 / log1p 后单独通道**，或从 GCN 输入中移出，改为解码期加性偏置，便于做因果消融（`do(pop=0)` ≈ 去掉该偏置看排序变化）。
- `NodeAttnMap` 用的 `A` 含转移频次，天然偏热门；可对行做 **流行度折减**：$A'_{ij} = A_{ij} / \mathrm{pop}(j)^\alpha$，再进入注意力调制。

#### B. 倾向评分（Propensity）加权损失

令 $\pi(p) \approx P(\mathrm{exposed} \mid p) \propto \mathrm{pop}(p)$（或用更细的 user–POI 倾向模型），对 CE 做：


$$
\mathcal{L} = \sum -\frac{1}{\pi(p^+)^\gamma}\log \frac{e^{s_{p^+}}}{\sum_q e^{s_q}}
$$


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
- 修饰：同一用户在“工作区午餐时间”与“居住区晚间”的偏好机制不同 → 应估计条件效应 $Pref(u,p \mid area, t)$。

对应干预：

- 控制 area：在同一 `(from_area, time_bin)` 内比较候选 POI（分层或条件建模）。
- 跨 area 的远跳：单独评估，避免被区内短跳指标掩盖。

### 5.4 接到 GETNext 的结构想法

1. **分层图**  
   - POI 层：现有 trajectory flow map  
   - Area 层：area 转移图 + area 嵌入  
   - 消息传递：POI embed ← 自身 + 所属 area embed；`NodeAttnMap` 增加 area 兼容项（跨区惩罚或跨区门控，而非一刀切禁止）

2. **解码分解**  
   

$$
s(p) = s_{\mathrm{poi\_pref}} + s_{\mathrm{area\_transit}}(a_{\mathrm{cur}}\!\to\! a_p) + s_{\mathrm{access}} + s_{\mathrm{pop}}
$$


   Transformer 主学 `s_poi-pref`；area-transit 可用小参数表或第二套轻量 GCN。

3. **条件归一化**  
   在 `(user, from_area, hour)` 条件下对候选做 softmax，相当于在同一场景内排序，削弱“永远推荐大商圈”的全局偏置。

---

## 6. 不止 IPS：SCM 因果图 + Deconfounded Training + 表征因果

**可以，而且往往比纯 IPS 更适合 GETNext。**

IPS 把偏差当成“采样权重”问题：纠正 $P(\mathrm{observe})$ 与目标分布的差异。但 Next-POI 里，距离 / 流行度 / area 不只是采样偏差，它们是**进入数据生成过程的结构化混杂（或中介）**。此时更自然的路线是：

1. 画 **SCM / 因果图**，标明 $U$（偏好）、$C$（混杂：access/pop/area）、$X$（观测历史与图特征）、$Y$（下一 POI）；
2. 用后门/前门/干预公式确定**可识别的去混淆目标**；
3. 用 **deconfounded training** 或 **representation-based** 方法，让编码器学到对 $C$ 去混淆（或与 $C$ 解耦）的表征，再接到 GETNext 的预测头。

IPS 与这类方法**互补**：IPS 改损失权重；SCM/表征方法改**学什么表示、在什么条件分布下预测**。可以只做后者，也可以表征去混淆 + 轻度 IPS。

### 6.1 先把因果图写清楚（推荐的最小 SCM）

把一次“历史 → 下一 POI”写成：

```text
         Z (user latent interest)          C_pop (POI popularity)
                  │                              │
                  │         ┌────────────────────┤
                  ▼         ▼                    ▼
   H (history) ──► X_pref ──► Y (next POI) ◄── X_ctx
                  ▲         ▲                    ▲
                  │         │                    │
            C_area         C_access ◄── Dist, time budget
         (land-use)              ▲
                                 │
                            Area, home/work anchors
```

约定：

| 符号 | 含义 | GETNext 里大致对应 |
|------|------|-------------------|
| $Z$ | 用户稳定兴趣（想估计的因果因子） | `UserEmbeddings` 的兴趣子空间 + 序列里与 cat 相关的部分 |
| $C = \{C_{\mathrm{access}}, C_{\mathrm{pop}}, C_{\mathrm{area}}\}$ | 混杂 / 场景约束 | 距离与活动半径；`checkin_cnt` / 边权；区域标签 |
| $X$ | 编码器输入 | 轨迹 POI/time/cat 嵌入、GCN(`X`,`A`)、`NodeAttnMap` |
| $Y$ | 下一 POI（或下一类别） | `decoder_poi` / `decoder_cat` |

**识别问题（要预测的量）：**

- 若任务是“真实下一跳”：估计 $P(Y \mid H)$ 即可，但内部仍应用 SCM 避免把 $C$ 误当成 $Z$ 的全部内容（否则长尾用户差）。
- 若任务是“偏好驱动的下一跳 / 去混淆推荐”：更关心  
  

$$
P(Y \mid do(Z),\; H_{\mathrm{wo}\,C}) \quad\mathrm{or}\quad P(Y \mid X_{\mathrm{pref}},\; do(C=\bar{c}))
$$


  即阻断 $C \to Y$ 的后门路径后，看偏好表征还能不能预测。

后门调整的离散版（$C$ 可分层时）：


$$
P(Y \mid do(X_{\mathrm{pref}})) = \sum_c P(Y \mid X_{\mathrm{pref}}, c)\, P(c)
$$


这就是许多 **deconfounded recommender** 的公式原型：不是按 $P(c \mid X)$ 加权（那会保留混杂），而是按**边缘** $P(c)$（或干预分布）积分。

### 6.2 Deconfounded Training：在 GETNext 上怎么做

核心思想：**训练时显式条件化混杂 $C$，预测/推理时对 $C$ 做边缘化或干预**，使主表征无法走“偷懒走 $C$”的捷径。

#### 方案 A：后门调整头（Backdoor-adjusted prediction）

1. 为每个训练样本构造混杂向量 $c$：距离桶、`log pop(p)`、`area_id`（from/to）、时段等。
2. 预测头改为条件头：$P(Y \mid h, c)$，其中 $h$ 是 Transformer 输出的轨迹表征。
3. 推理时：
   - **写实**：代入真实 $c$；
   - **去混淆**：  
     

$$
P(Y \mid h) = \sum_{c'} P(Y \mid h, c')\, \hat{P}(c')
$$


     或取均匀 / 反事实 $c'=\bar{c}$（如所有候选同一 pop、同一距离环带中位数）。

接到现有代码：不必重写整网。可在 `y_pred_poi` 上增加一项 `f(c)` 的条件偏置，或对 decoder 做 FiLM/拼接 $c$；去混淆推理时对 $c$ 的若干原型做平均。

#### 方案 B：混杂条件化 + 不变风险（Deconfounded + IRM / Group DRO）

把 $C$ 的不同取值看作**环境** $e$（近 vs 远、热门 vs 长尾、商圈 vs 居民区）：


$$
\min_\theta \sum_e \mathcal{L}_e(\theta) + \lambda \cdot \mathrm{Penalty}(\{\nabla_{w\mid e}\})
$$


要求同一套偏好表征在多环境下都能预测 $Y$（或下一类别）。这直接针对“模型只在近邻/热门环境好用”的问题。

GETNext 落地：按距离桶 × 流行度四分位 分组算 CE，再加 Group DRO 或 IRMv1 惩罚；类别头作不变预测目标往往比 POI_id 更稳。

#### 方案 C：对抗 / 互信息去混淆（Adversarial deconfounding）

编码器出 $h = \mathrm{Enc}(H)$，额外判别器 $D$ 试图从 $h$ 预测 $C$（距离桶、pop 桶、area）：


$$
\min_{\mathrm{Enc},\,\mathrm{Pred}}\; \max_D\; \mathcal{L}_{Y}(h) - \lambda\, \mathcal{L}_{C}(D(h))
$$


迫使 $h$ 对 $C$ 信息最少（近似 $h \perp C$），再由单独的 $g(C)$ 支路提供可达性/流行度分数（见 §7 的分数分解）。  
这就是典型的 **representation-based causal** 做法：把“偏好表征”和“混杂表征”拆开。

注意：完全 $h \perp C$ 可能过强——兴趣本身与常驻 area 相关。更稳妥的是 **partial disentanglement**：

- $h = [h_z ; h_c]$，只对 $h_z$ 做对抗去 $C$；
- $h_c$ 允许预测 $C$，并单独进入 access/pop/area 支路；
- 预测 $Y$ 时写实模式用两者，偏好模式只用 $h_z$。

#### 方案 D：结构化因果表征（SCM-shaped encoders）

不只“去相关”，而是按 SCM 搭模块，使干预有明确旋钮：

```text
H ──► Enc_Z ──► h_z ──┐
                       ├──► Pred_Y
C ──► Enc_C ──► h_c ──┘
         │
         └──► (optional) recon C / predict dist, pop, area
```

- 训练：`L_Y(h_z, h_c) + L_recon(C) + L_ortho(h_z, h_c) + L_inv(h_z across env)`  
- 干预：`do(C=c*)` = 把 `h_c` 换成编码 `c*` 的向量，保持 `h_z` 不变，看 top-k 如何变。  
这比 IPS 更可解释，也更适合回答“若可达性相同，用户还会去吗？”

### 6.3 常见表征因果算法族，哪些能用

| 算法族 | 想法 | 是否适合 Next-POI / GETNext | 备注 |
|--------|------|------------------------------|------|
| Backdoor adjustment / PDA 类 | $\sum_c P(Y\mid X,c)P(c)$ | ✅ 很适合 | pop/距离/area 可离散分层；推理可边缘化 |
| DecConfounder / 替代混杂因子 | 用多因多果学替代混杂 $\hat{C}$ | ⚠️ 可试 | 需多个“因果”侧变量；POI 图上可把多用户转移当多因 |
| Disentangled / adversarial rep | $h_z \perp C$，$h_c$ 吃混杂 | ✅ 推荐 | 与 User/POI 双塔或双头天然兼容 |
| IRM / V-REx / Group DRO | 跨环境不变预测 | ✅ 推荐 | 环境=距离×pop×area 切片 |
| Front-door | 经中介 $M$ 识别 | ⚠️ 条件苛刻 | 若用“意图类别”作 $M$，需假设类别挡住全部偏好→POI 路径且无 $C\to M$ 后门——很难严格成立，可作启发（先预测 cat 再 POI） |
| Causal discovery（从数据学图） | 学 DAG 再建模型 | ❌ 优先级低 | 轨迹+强选择偏差下图难可靠；**专家因果图 + 敏感分析**更务实 |
| IPS / SNIPS | 按倾向重加权 | ✅ 可作辅助 | 方差大；作表征方法的补充而非唯一手段 |
| Counterfactual data augmentation | 干预 $C$ 后合成样本 | ✅ | 同距离环带替换、同 area 替换热门 POI 等 |

**结论：**  
用 **SCM 建模 + deconfounded training（后门调整 / 对抗解耦 / 跨环境不变）** 完全可行，且比“只上 IPS”更贴合“邻近与热门被学成偏好”的机制。IPS 保留为对极端稀有桶的稳定器即可。

### 6.4 相对纯 IPS 的优劣

| | IPS | SCM + Deconfounded / Rep-based |
|--|-----|--------------------------------|
| 建模对象 | 采样概率 | 数据生成与混杂路径 |
| 方差 | 易爆，需裁剪 | 通常更稳，但对抗训练可能不稳 |
| 可解释干预 | 弱 | 强（`do(C)` 有明确模块） |
| 实现成本 | 低 | 中（要定义 $C$、改头/损失） |
| 与 GETNext | 改 CE 权重即可 | 改表征与 `adjust_pred_prob_by_graph` 的信息来源 |

推荐默认组合：

1. **主路径**：双表征 $h_z, h_c$ + 后门调整或写实/偏好双模式推理；  
2. **正则**：跨环境 Group DRO（或轻量 IRM）；  
3. **可选**：对最稀有距离桶加裁剪 IPS。

### 6.5 最小可跑通的实现草图（仍挂在 GETNext 上）

```text
现有:  GCN(X,A) → poi_emb
       Transformer(seq) → h
       y = decoder(h) + NodeAttnMap(cur)

改为:
       C = [dist_bucket, log_pop, area_from, area_to, hour]
       h → split/project → h_z, h_c
       L = CE(y | h_z, h_c)
         + λ1 * CE_adv(C | h_z)     # 上升沿训练 Enc，使 h_z 难测 C
         + λ2 * CE(C | h_c)         # h_c 要能预测混杂
         + λ3 * GroupDRO(CE by env)
       推理_deconf: y ∝ softmax(decoder(h_z, c̄) + α * flow_residual)
       推理_factual: y ∝ softmax(decoder(h_z, h_c) + attn_map)
```

其中 `flow_residual` 建议是对 `A` 做过 pop/距离归一化后的残差流，避免 `NodeAttnMap` 再次把混杂灌回 $h_z$ 路径。

---

## 7. 三者一起建模时的统一框架

可达性、流行度、区域不是三个独立补丁，而是同一生成过程的不同外生/中介变量。一个可操作的统一分数：


$$
s(u,p,t)=s_{\theta}(u,p,t)+\beta_a\,s_{\mathrm{access}}(u,p,t)+\beta_p(u)\,s_{\mathrm{pop}}(p,t)+\beta_r\,s_{\mathrm{area}}(a_u,a_p,t)
$$


其中：

- $s_{\theta}(u,p,t)$：个性化偏好（主模型 / $h_z$）
- $s_{\mathrm{access}}$：可达性
- $s_{\mathrm{pop}}$：流行度（可对用户门控）
- $s_{\mathrm{area}}$：区域转移

训练目标建议：

1. **主损失**：下一 POI CE（条件化 $C$ 的 deconfounded 形式；必要时再加轻度 IPS）。
2. **辅助损失**（GETNext 已有）：时间、类别 —— 类别头有助于兴趣信号，可对类别 CE **提高权重**，对 POI_id 头做去偏，形成“先功能、后具体地点”的两阶段因果直觉；类别头也可作为近似前门中介（启发式，非严格识别）。
3. **表征去混淆正则**：  
   - $h_z \perp C$（对抗 / HSIC / 正交）；$h_c$ 重构 $C$；  
   - 跨环境 Group DRO / IRM；  
   - 惩罚 $s_\theta$ 与 `pop`、`dist` 的全局相关作轻量替代。
4. **反事实一致性**：对同一历史，扰动 pop/area 编码后，类别预测应更稳，POI_id 预测允许变（对应 `do(C)` 探针）。

推理时可提供两种模式：

- **写实模式**：保留 access/pop/area（贴近真实下一跳，刷线上指标）。
- **偏好 / 去混淆模式**：边缘化 $C$ 或 `do(C=\bar{c})`，主要用 $h_z$（更适合“猜兴趣 / 探索推荐”）。

这对研究“模型到底学到了什么”很有价值。

---

## 8. 与 GETNext 模块的具体挂钩清单

| 模块 | 现状 | 因果向改动 |
|------|------|------------|
| `graph_X` / `checkin_cnt` | 直接进 GCN | 拆出为 $C_{\mathrm{pop}}$ / $h_c$；GCN 更侧重 cat + 区域特征 |
| `graph_A` | 原始转移频次 | 距离桶内归一化；`/ pop(j)^α`；残差流进写实支路，不进 $h_z$ |
| `NodeAttnMap` | `e * (A+1)` 加强群体流 | 拆成偏好注意力 × 结构门控；或仅加入 factual 路径 |
| `UserEmbeddings` | 单一用户向量 | 显式 $h_z$（兴趣）+ $h_c$（从众/活跃/常驻 area） |
| `TransformerModel` | 三头 POI/time/cat | 条件化 $C$ 的 POI 头；强化 cat 头；可选后门边缘化推理 |
| `adjust_pred_prob_by_graph` | 直接加 attn_map | factual / deconfounded 两套聚合；deconf 路径避免再灌热门近邻 |
| 损失 | CE(+time+cat) | + 对抗去混淆 + Group DRO；IPS 仅作稀有桶辅助 |
| 评估 | 全局 Acc@k / MRR | 距离桶、流行度四分位、跨 area、小众用户；外加 `do(C)` 排序稳定性 |

---

## 9. 建议的实验顺序（由易到难）

不必一上来上完整可识别 SCM。建议：

1. **诊断**  
   - 统计训练转移距离分布、命中样本的距离分布、top-k 候选的平均 pop。  
   - `do(pop)` / 去掉 `checkin_cnt` / 去掉 `NodeAttnMap` 的消融，看指标与长尾子集变化。

2. **轻量干预（含或不含 IPS）**  
   - $A$ 的 pop/距离归一化；同距离环带负采样。  
   - 可选：距离分层 IPS（裁剪）作基线对照。

3. **SCM + Deconfounded / 表征方法（主推）**  
   - 定义 $C$，实现 $h_z/h_c$ 分解 + 对抗或正交约束。  
   - 条件头 + 推理时后门边缘化 / `do(C=\bar{c})`。  
   - 按环境做 Group DRO；对比“仅 IPS”与“仅表征去混淆”与“两者结合”。

4. **结构分数分解**  
   - $s = s_{\mathrm{pref}}(h_z) + s_{\mathrm{access}} + s_{\mathrm{pop}}(u) + s_{\mathrm{area}}$。  
   - area 特征与区域流图。

5. **反事实评估协议**  
   - 固定报告：整体指标 + 远跳 + 长尾 POI + 跨 area + 低从众用户组。  
   - 报告 factual vs deconfounded 两套指标，避免只看被近邻/热门刷高的 Acc@1。

---

## 10. 风险与边界

- **过度去偏**：可达性与流行度在真实世界里*确实*影响下一跳；完全 `do(access=1), do(pop=0)` 的预测会不切实际。要分清任务：是“预测真实下一签到”还是“推断潜在兴趣”。
- **不可识别性**：没有随机实验或强工具变量时，偏好与居住地/常驻 area 无法完美分开；应用条件化与敏感性分析（改变 β / 边缘化分布看排序稳定性）。因果发现学到的图不宜过度信任。
- **对抗训练不稳**：$h_z \perp C$ 过强会伤有用信号；优先 partial disentanglement + 重建 $C$ 的 $h_c$ 支路。
- **方差**：纯 IPS 与长尾上采样可能伤整体 Acc；表征方法相对更稳，仍需 Group DRO / 多目标，保证近邻主群体不明显崩坏。
- **图泄漏**：`A` 若含验证/测试时段转移，去偏结论会偏乐观；应严格只用训练期构图（当前 `build_graph.py` 流程需保持这一点）。

---

## 11. 一句话收束

GETNext 很强地拟合了**群体轨迹流 + 序列上下文**，但也因此把**邻近可达、热门曝光、区域土地用途**写进了“偏好”。用因果视角，不必停在 IPS：用 **SCM 因果图** 标明混杂路径，再用 **deconfounded training / 表征解耦（$h_z \perp C$）/ 跨环境不变** 把偏好与约束拆开，配合切片与 `do(C)` 评估，才能检验模型是否还看得见那些“偶尔走远、偏爱小众、跨区行动”的用户。
