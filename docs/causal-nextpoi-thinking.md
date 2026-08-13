# 从 Transformer 视角做 Next-POI 因果去偏：思考笔记

本文从 **Transformer / 序列建模** 的通用理念出发，讨论 next-POI 推荐中的因果去偏，而不是绑定某一篇具体模型（如 GETNext）的图模块。

核心立场：

- Next-POI 在多数现代方法里，本质是 **给定历史轨迹，预测下一个离散“词”（POI id）** 的自回归 / encoder 打分问题。
- **POI embedding 从哪来**（随机初始化查表、预训练、GCN、空间编码器……）只是输入表征的一种实现；因果问题主要出在 **序列目标、注意力聚合、词表 softmax、以及训练数据的生成机制** 上。
- GETNext 的 GCN / Trajectory Flow Map 只是“给 Transformer 更好的 POI 输入 / 额外先验”的一种特例；下文默认 **不依赖 GCN**，必要时在附录对照。

关注的三类偏差：

1. **可达性 / 距离偏差**（Accessibility / Distance bias）
2. **流行度偏差**（Popularity bias）
3. **区域属性混淆**（Area / Land-use confounding）

去偏主路线：**SCM 因果图 → deconfounded training / 表征解耦**；IPS 仅作稀有样本辅助。

> **公式说明**：使用 GitHub 友好的 `$...$`（行内）与 `$$...$$`（独立公式）。

---

## 1. Transformer 式 Next-POI：在优化什么？

把一条轨迹写成 token 序列：

$$
H=(p_1,t_1,c_1),\ldots,(p_T,t_T,c_T)
$$

典型 Transformer next-POI 管线（抽象）：

```text
POI / user / time / category id
        │
        ▼
   Embedding 层   ←── 任意实现：nn.Embedding / 预训练 / GCN / 空间编码 …
        │
        ▼
  融合为序列 token x_1..x_T
        │
        ▼
 Transformer Encoder（因果 mask 或只取最后位置）
        │
        ▼
     h_T（上下文表征）
        │
        ▼
  打分 s(h_T, p) 对词表 Softmax → P(p_{T+1} | H)
```

训练目标几乎总是：

$$
\max_\theta\; \log P_\theta(p_{T+1}\mid H)
\quad\text{即 CrossEntropy over POI vocabulary}
$$

这与语言模型相同：**观测到的下一 token 被当成“正确标签”**。但 check-in 不是自由写作，而是

$$
\mathrm{Visit}(u,p,t)=f\big(\mathrm{Pref}(u,p),\;\mathrm{Access}(u,p,t),\;\mathrm{Expo}(p,t),\;\mathrm{Area}(p),\;\mathrm{Context}(t)\big)+\epsilon
$$

因此 CE 会把 Access / Expo / Area **一并写进**：

| 位置 | 学到的“捷径” |
|------|----------------|
| Token embedding | 热门 POI 范数更大、更新更频；地理邻近的 id 在共现中被拉近 |
| Self-attention | 更关注“最近一次附近 POI”、重复访问的热门点 |
| 输出层 / 偏置 | 对高频 POI_id 给出更高先验 logits（类似 LM 的 unigram bias） |
| Softmax 竞争 | 长尾兴趣在近邻×热门候选面前被压掉 |

**结论：** 就算完全去掉 GCN，只要仍用“历史 → Transformer → 全词表 CE”，距离/流行度/区域偏差依然存在。图模块可能放大它们，但不是根因。

---

## 2. 通用因果图（与具体 POI encoder 解耦）

```text
   Z (user interest)          C_pop (popularity / exposure)
            │                         │
            │         ┌───────────────┤
            ▼         ▼               ▼
 H (history) ──► h = Transformer(H) ──► Y (next POI)
            ▲         ▲               ▲
            │         │               │
       C_area    C_access ◄── dist, time budget, mobility
     (land-use)       ▲
                      │
                 home/work anchors, Area
```

| 符号 | 含义 | Transformer 管线中的落点 |
|------|------|---------------------------|
| $Z$ | 相对稳定的兴趣 | 用户向量兴趣子空间；attention 聚合出的兴趣成分 |
| $C=\{C_{\mathrm{access}},C_{\mathrm{pop}},C_{\mathrm{area}}\}$ | 混杂 / 场景约束 | 不必进“兴趣表征”；应显式条件化或走独立支路 |
| $h$ | 序列上下文向量 | Transformer 最后位置（或池化）输出 |
| $Y$ | 下一 POI（或下一类别） | 词表上的 CE / 多任务头 |

想估计的量往往不是裸的 $P(Y\mid H)$，而是：

$$
P(Y\mid h_z,\; do(C=\bar{c}))
\quad\text{或}\quad
\sum_c P(Y\mid h_z,c)\,P(c)
$$

即：让 Transformer 学的主表征尽量接近偏好 $Z$，再对混杂 $C$ 做干预或边缘化。

POI embedding 来源（查表 / GCN / …）只影响 $H$ 如何被数值化，**不改变这张因果图的主干**。

---

## 3. 从 Transformer 机制看三类偏差

### 3.1 可达性：邻近被注意力当成“语义相关”

**现象：** 训练转移大量短距 → 模型发现“复制/偏向当前点附近 id”损失最低。

在 Transformer 里这表现为：

1. **位置捷径**：因果注意力过度依赖最后几个 token（recency + proximity 耦合）。
2. **共现几何**：embedding 空间里，常一起出现的近邻 POI 聚成一团；点积打分等价于“近邻检索”。
3. **负采样/全词表 CE**：远距离正样本稀少，梯度几乎不要求模型区分“同距离环带内谁更符合兴趣”。

**去偏（不靠 GCN）：**

- 分数分解：

$$
s(u,p,t)=s_{\mathrm{pref}}(h_z,p)+\lambda(t)\,s_{\mathrm{access}}(u,p,t)
$$

  其中 $s_{\mathrm{access}}$ 由距离、活动半径、时间预算等给出；$s_{\mathrm{pref}}$ 才是 Transformer 主头。

- **同距离环带负采样 / 对比学习**：逼迫 attention+输出层在“一样近”的集合里学偏好。
- **距离桶 IPS**（可选）：稀有远跳上权，方差需裁剪。
- **评估切片**：Acc@k 按距离桶；否则近邻样本淹没一切。

反事实问题：若 $do(\mathrm{Access}=1)$（候选同样可达），排序是否仍指向同一 POI？这才接近“想去”。

### 3.2 流行度：词表频率先验钻进 Softmax

**现象：** 热门 POI 在序列中出现多 → embedding 与输出偏置被更多更新 → 类似语言模型 unigram。

在 Transformer 里：

1. 输出层权重 / 偏置隐式学到 $P(p)$。
2. 用户向量容易塌成“跟热门对齐”，对小众兴趣用户失效。
3. Multi-head 可能用若干头专门“复读高频模式”，而非建模个人转移语法。

**去偏（表示层，而非改图）：**

- 两头打分：

$$
s(u,p)=s_{\mathrm{match}}(h_z,p)+g(u)\,\mathrm{pop}(p)
$$

  推理可关 $g$（偏好模式）或保留 $g$（写实模式）。

- 对 CE 做流行度倾向加权，或 popularity-aware 负采样的逆操作。
- **表征去混淆**：$h_z \perp C_{\mathrm{pop}}$（对抗 / 正交），流行度只进 $h_c$ 或 $g(\cdot)$。
- 辅助 **类别头**：兴趣对“咖啡 vs 酒吧”更稳，对“哪家网红店”更易受流行度污染；强化 category CE 有助于 $h_z$。

### 3.3 Area：场景切换被误学成兴趣切换

商圈午餐 vs 居民区晚间，是 **条件机制不同**，不是偏好向量整体翻转。

Transformer 视角：

- 若不告诉模型 area，它会用 lat/lon、最近 POI 类别统计去隐式拟合区域——这些信号与 $Z$ 纠缠在 $h$ 里。
- Self-attention 的“上下文”其实常是 **当前 area 的局部模式**，跨区远跳被当成异常。

**做法：**

- 显式 $C_{\mathrm{area}}$（网格 / 功能区 / 家公司锚点）。
- 条件化预测 $P(Y\mid h_z, a_{\mathrm{from}}, t)$，或加 $s_{\mathrm{area}}(a\to a_p)$ 支路。
- 跨 area 子集单独评估。

---

## 4. 契合 Transformer 理念的去偏训练（相对 IPS 更主推）

IPS 改的是样本权重；Transformer 更需要对齐其归纳偏置：**表征 $h$、注意力、词表打分头**。

### 4.1 双表征 + 后门调整

$$
h \xrightarrow{\mathrm{split}} (h_z, h_c),\quad
C=[\mathrm{dist\_bucket},\,\log\mathrm{pop},\,\mathrm{area},\,\mathrm{hour}]
$$

- 训练：$P(Y\mid h_z,h_c)$ 或 $P(Y\mid h_z,c)$  
- 约束：$h_z$ 难预测 $C$；$h_c$ 要能预测 $C$  
- 推理：  
  - **写实**：用真实 $c$ / $h_c$  
  - **去混淆**：$\sum_{c'} P(Y\mid h_z,c')\hat{P}(c')$ 或 $do(C=\bar{c})$

这直接对应 Transformer 的“上下文向量该编码什么”。

### 4.2 跨环境不变（IRM / Group DRO）

把距离×流行度×area 切片当环境 $e$，要求同一 $h_z$ 在多环境下都能预测 $Y$（或类别）：

$$
\min_\theta \sum_e \mathcal{L}_e(\theta)+\lambda\cdot\mathrm{Penalty}(\{\nabla_{w\mid e}\})
$$

符合 Transformer 想学的“可迁移序列规律”，而不是单环境捷径。

### 4.3 对抗解耦 / partial disentanglement

完全 $h\perp C$ 过强（兴趣与常驻 area 相关）。更稳：

- 只对 $h_z$ 去 $C$  
- $h_c$ 吃混杂并进入 access/pop/area 支路  
- 偏好模式推理只用 $h_z$

### 4.4 与“语言模型式”技巧的类比

| LM / Transformer NLP | Next-POI 对应 |
|----------------------|---------------|
| 频率偏置 / unigram | POI 流行度头，可开关 |
| 领域自适应 / 不变表示 | 跨 area、跨距离桶的 $h_z$ |
| 控制生成（steer by attribute） | `do(C)` 改写混杂后再打分 |
| 对比学习 / hard negatives | 同距离环带、同 area 内难负例 |
| 多任务（句法 vs 语义） | 类别/时间头 vs POI_id 头 |

不必引入 GCN，也能把因果干预做进 **encoder 输出与解码打分**。

---

## 5. 统一分数（仍是 Transformer 解码，而非图扩散）

$$
s(u,p,t)=s_{\theta}(h_z,p)+\beta_a\,s_{\mathrm{access}}+\beta_p(u)\,s_{\mathrm{pop}}+\beta_r\,s_{\mathrm{area}}
$$

- $s_{\theta}$：Transformer 主匹配分（点积 / MLP / tied embedding）  
- 后三项：显式混杂支路，**不要偷塞回无名 $h$**

训练建议：

1. 主 CE：条件化 $C$ 的 deconfounded 形式  
2. 辅助：时间 + 类别（类别更贴 $Z$）  
3. 正则：$h_z\perp C$ + Group DRO  
4. 可选：稀有距离桶裁剪 IPS  

推理双模式：写实（保留 $\beta$）vs 偏好/去混淆（边缘化或冻结 $C$）。

---

## 6. 最小实现草图（纯序列，无 GCN 假设）

```text
输入: 历史 POI/time/cat/user + 可算的 C 特征
Enc:  Embedding + Transformer → h
Split: h → h_z, h_c
Loss:  CE(Y | h_z, h_c)
     + λ1 * Adv(C | h_z)      # 最小化 C 可预测性
     + λ2 * Pred(C | h_c)
     + λ3 * GroupDRO(env)
解码:  s = Dot(h_z, e_p) + g_access + g_pop + g_area
推理_deconf:  用 c̄ 或边缘化 C；可关掉 g_pop / 缩小 g_access
推理_factual: 用真实 C
```

POI 向量 $e_p$ 可以是 `nn.Embedding`；若某系统用 GCN 生成 $e_p$，只需保证 **流行度与原始转移频次不要二次灌进 $h_z$**（见附录）。

---

## 7. 评估协议（与 backbone 无关）

只报整体 Acc@k / MRR 会被近邻×热门刷高。固定报告：

- 距离桶 Acc@k / 远跳召回  
- 流行度四分位、长尾 POI  
- 跨 area 转移子集  
- 低从众用户组  
- **factual vs deconfounded** 两套指标，以及 `do(C)` 下 top-k 稳定性  

---

## 8. 建议实验顺序

1. **诊断**：命中样本的距离/pop 分布；去掉输出偏置、打乱距离特征看跌多少。  
2. **序列侧轻量干预**：同距离环带负采样；$s_{\mathrm{pref}}+s_{\mathrm{access}}$ 分解。  
3. **表征去混淆**：$h_z/h_c$ + 后门边缘化 + Group DRO（主推）。  
4. **Area 条件化**与双模式推理。  
5. 需要时再对比“加/不加图先验”的消融——图不是前提。

---

## 9. 风险与边界

- 真实下一跳确实受可达性与流行度影响；完全 `do(C=\bar{c})` 适合“兴趣推断”，不一定适合“写实预测”。  
- $Z$ 与常驻 area 不可完美识别；做敏感性分析。  
- 对抗训练需 partial disentanglement，避免抽干有用信息。  
- 词表极大时，去偏后的全 Softmax 仍贵；可用候选生成 + 重排，但重排阶段同样要带 $C$ 分解。

---

## 10. 一句话收束

从 Transformer 理念看，next-POI 的因果问题是：**序列模型把观测转移的生成约束（近、热、区）当成了可注意力、可 Softmax 的“语义”**。  
去偏应落在 **上下文表征拆分、解码打分分解、跨环境不变与 `do(C)` 推理**，而不是依赖是否使用 GCN。GCN 只是 POI embedding 的一种来源；换查表嵌入，同一套 SCM 与 deconfounded training 仍然成立。

---

## 附录 A. 若对照 GETNext：GCN 只是可选放大镜

GETNext =（可选）轨迹流图 GCN POI 嵌入 + Transformer 序列 +（可选）`NodeAttnMap` 加性先验。

- **根因仍在序列 CE**；去掉 GCN/AttnMap，偏差不会自动消失。  
- GCN/`checkin_cnt`/原始转移边权会 **额外** 把 pop 与近邻灌进 $e_p$ 与加性 logits，可能造成“表征一次 + 先验一次”的双重计入。  
- 若保留图模块：让其只影响 factual 支路或 $h_c$，不要无约束打进 $h_z$。  
- 研究去偏时，建议先在 **纯 Transformer + id embedding** 上验证因果模块，再决定是否加图。

## 附录 B. 方法速查

| 方法 | 作用位置 | 是否需要 GCN |
|------|----------|--------------|
| $s_{\mathrm{pref}}+s_{\mathrm{access}}+s_{\mathrm{pop}}+s_{\mathrm{area}}$ | 解码打分 | 否 |
| $h_z/h_c$ 解耦 + 对抗 | Transformer 输出 | 否 |
| 后门调整 / `do(C)` | 条件头与推理 | 否 |
| Group DRO / IRM | 按环境划分的 CE | 否 |
| 同距离环带对比 | 负采样 / 损失 | 否 |
| IPS | 样本权重 | 否（辅助） |
| 图边权归一化 / 移出 checkin_cnt | 仅当使用图编码器时 | 是（可选） |
