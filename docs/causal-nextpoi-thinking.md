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

去偏主路线：**SCM 因果图 → 可识别目标量 → deconfounded training / 表征解耦**；IPS 仅作稀有样本辅助。

> **公式说明**：使用 GitHub 友好的 `$...$`（行内）与 `$$...$$`（独立公式）。

---

## 0. 阅读约定：因果层 vs 实现层

全文严格分两层写。混写会把 “Transformer encoding” 画进 DAG，也会把 “把 $C$ 拼进 embedding” 误当成已经做了 $do(C)$。

| 层 | 允许出现的内容 | 不允许 |
|----|----------------|--------|
| **因果层** | $Z$, $H$, $Y$, $C_{\mathrm{pop}}$, $C_{\mathrm{area}}$, $C_{\mathrm{access}}$，外生输入 | Transformer、GCN、CE、IPS、对抗头、embedding |
| **实现层** | Embedding / Encoder / 打分分解 / Loss / 采样 / 评估切片 | 把模块名画成 DAG 节点 |

- 因果层问：观测 $P(Y\mid H)$ 相对 $P(Y\mid do(\cdot))$ 差在哪、什么量可识别。
- 实现层问：用什么模块去逼近那个可识别量。
- $h_z=\phi_\theta(H)$ 是 $Z$ 的**可学习代理**，不是 $Z$ 本身。
- 把 $C$ 拼进 token 只是在拟合 $P(Y\mid H,C)$（条件化）；**干预**发生在训练约束 + 推理时固定 $\bar c$ 或对 $C$ 边缘化。

---

## 1. 观测生成与朴素目标

把一条轨迹写成 token 序列：

$$
H=(p_1,t_1,c_1),\ldots,(p_T,t_T,c_T)
$$

典型 Transformer next-POI 管线（**实现层抽象**，不是因果图）：

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

这与语言模型相同：**观测到的下一 token 被当成“正确标签”**。但 check-in 不是自由写作。生成机制更接近：

$$
\mathrm{Visit}(u,p,t)=f\big(\mathrm{Pref}(u,p),\;\mathrm{Access}(u,p,t),\;\mathrm{Expo}(p,t),\;\mathrm{Area}(p),\;\mathrm{Context}(t)\big)+\epsilon
$$

因此 CE 估计的是关联 $P(Y\mid H)$，会把 Access / Expo / Area **一并写进**实现模块：

| 位置 | 学到的“捷径” |
|------|----------------|
| Token embedding | 热门 POI 范数更大、更新更频；地理邻近的 id 在共现中被拉近 |
| Self-attention | 更关注“最近一次附近 POI”、重复访问的热门点 |
| 输出层 / 偏置 | 对高频 POI_id 给出更高先验 logits（类似 LM 的 unigram bias） |
| Softmax 竞争 | 长尾兴趣在近邻×热门候选面前被压掉 |

**结论：** 就算完全去掉 GCN，只要仍用“历史 → Transformer → 全词表 CE”，距离/流行度/区域偏差依然存在。图模块可能放大它们，但不是根因。估计 $P(Y\mid H)$ **不等于**估计兴趣干预效应；目标量见 §2 末与 §3。

---

## 2. 通用因果图（与具体 POI encoder 解耦）

### 变量定义与因果关系梳理（模型-变量对齐）

#### 变量分组

- $\mathbf{Z}$: 用户个体兴趣（长期/稳定的 preference，隐变量，需通过行为间接建模。）
- $\mathbf{H}$: 用户历史行为序列（已观测；模型主要输入之一。）
- $\mathbf{C}_{\text{pop}}$: POI 热度/曝光（部分可观测统计量，部分由词表频率/外部数据补全。）
- $\mathbf{C}_{\text{area}}$: 区域相关属性（如地理片区/功能区，对每个 POI 可查表获得。）
- $\mathbf{C}_{\text{access}}$: 可达性（动态变量，基于当前位置、距离、时间预算、出行方式、锚点等，需要基于外部数据或规则推算后输入模型。）
- $\mathbf{Y}$: 下一个可能访问 POI（模型输出/标签）。

#### 变量因果图（去实现层，只关注真实变量）

```text
      Z (用户真实兴趣)
         │      │
         ▼      ▼
       H(历史)  Y(下个POI)
         ▲   ▲  ▲
     ───┤   │  │─────────────┐
     │   │   │  │             │
 C_pop  C_area  C_access ← [距离/时间预算/锚点等外部变量]
```

- $Z \to H$：兴趣决定历史模式
- $Z \to Y$：兴趣直接影响选择
- $H \to Y$：历史惯性、短程记忆
- $C_{\mathrm{pop}}$、$C_{\mathrm{area}}$、$C_{\mathrm{access}} \to H/Y$：流行度、区域、可达性混杂共同影响历史与目标
- $C_{\mathrm{access}} \leftarrow$ 距离、时间预算、mobility、锚点：这些为外生输入

---

### 主要模型输入与处理方式

| 变量 | 是否模型输入 | 如何获得/预处理 | 需学习获取？ | 建议处理方式 |
|------|------------|----------------|-------------|------------|
| $H$（历史） | 是 | 直接输入（POI/user/time 等 id 序列） | 否 | Transformer 主输入 token 序列 |
| $C_{\mathrm{pop}}$（热度） | 是/可选 | POI 词频/外部曝光度查表 | 否 | 可作为统计特征拼接 embedding，也可用于纠偏 loss |
| $C_{\mathrm{area}}$（区域） | 是/可选 | POI 所属区域属性查表 | 否 | 做 embedding 拼接或评估分层 |
| $C_{\mathrm{access}}$（可达性） | 是/可选 | 当前 context 动态计算（基于距离、预算等） | 否 | 1）规则算法先筛候选 2）作为 side-info 拼接特征 |
| $Z$（稳定兴趣） | 间接 | 隐含在历史序列统计/主表征中 | 需端到端学习 | 通过历史序列 Transformer 内隐建模 |

---

### 各部分实现建议

**1. Transformer:**

- 主要作用：端到端从历史 ($H$) 学习兴趣 ($Z$) 与短时依赖，输出主表征 $h$。
- 输入：历史 token 序列（形式上可拼接/注入 $C_{\mathrm{pop}}$、$C_{\mathrm{area}}$、$C_{\mathrm{access}}$ 特征）。
- 注意：transformer 本身只近似学习 $H\to Y$、$Z\to Y$ 的综合路径。

**2. 额外变量的处理和注入：**

- $C_{\mathrm{pop}}$, $C_{\mathrm{area}}$, $C_{\mathrm{access}}$ 可由预处理脚本查表、计算后作为额外输入特征拼入 token embedding，也可用作训练或评估切片（如热度/区域分层评估）。
- $C_{\mathrm{access}}$ 涉及动态地理约束（如距离/预算/锚点），可在数据处理阶段为每个样本增加该属性，或用于候选生成/负采样。

**3. Loss/目标增强手段:**

- 对抗流行度/可达性偏差，可设计 reweighting loss（如 distance bucket 采样、流行度 IPS 提升长尾）。
- 也可将 $C_{\mathrm{pop}}$/$C_{\mathrm{access}}$ 做 confounder 干预（如 $do(C=\ldots)$ 分析）或边缘化处理。
- 评估指标应区分各类混杂因子下的效果（如分 distance、popularity 分桶 Acc@K），避免仅反映热门/近邻。

---

**总结建议与结论：**

- 因果图仅含 $Z$、$H$、$Y$ 和 $C_*$ 等真实因果变量，不出现“Transformer encoding”。
- Transformer/GCN 等只是为建模这些变量之间的关系、隐式拟合 $Z$ 等 latent 变量的实现细节，可单列实现层专门分析，不宜混淆为 DAG 节点。
- 各输入特征应按其获取来源、是否需 learned 加以标注，实现上分别由 embedding、预处理增强、候选过滤、loss 修改、评估分层等方式解决。
- 区分“要学习的核心因果变量”（如兴趣 $Z$）与“数据先验/可查外特征/约束变量”（$C$ 系列）至关重要。

想估计的量往往不是裸的 $P(Y\mid H)$，而是：

$$
P(Y\mid h_z,\; do(C=\bar{c}))
\quad\text{或}\quad
\sum_c P(Y\mid h_z,c)\,P(c)
$$

即：让 Transformer 学的主表征尽量接近偏好 $Z$，再对混杂 $C$ 做干预或边缘化。

POI embedding 来源（查表 / GCN / …）只影响 $H$ 如何被数值化，**不改变这张因果图的主干**。

识别假设、写实 vs 去混淆两种推理、以及各算法如何逼近上式，见 §3 与 §5。

---

## 3. 目标量与识别

### 3.1 朴素监督学习在估什么

CE 拟合的是：

$$
P(Y\mid H)=\sum_{z,c} P(Y\mid H,z,c)\,P(z,c\mid H)
$$

$C$ 同时影响 $H$ 与 $Y$，故 $P(z,c\mid H)$ 把混杂折进条件分布。模型即使“拟合得很好”，学到的也可以是近 / 热 / 区捷径。

### 3.2 两种正当目标（不要混成一个数）

**写实预测**（factual）：下一跳在真实约束下会去哪。

$$
P(Y\mid H, C_{\mathrm{obs}})
$$

**兴趣 / 去混淆**（deconfounded）：若把混杂固定或边缘化，偏好排序是否仍指向同一 POI。

$$
P(Y\mid do(C=\bar c),\, Z=z)
$$

$Z$ 不可观测，实现上用代理 $h_z\approx\phi(H)$：

$$
P(Y\mid h_z,\; do(C=\bar c))
\quad\text{或}\quad
\sum_c P(Y\mid h_z,c)\,P(c)
$$

后者即后门调整形式。写实 vs 去混淆是 **两种推理模式**，不是互相否定。

### 3.3 识别假设（写清楚才能谈算法）

要对 $C$ 做后门调整，至少需要：

1. **混杂可测（或有足够代理）**：$C$ 阻断 $Z\to Y$ 之外、经混杂进入 $Y$ 的路径。遗漏的 $U$ 若同时影响 $H$ 与 $Y$，调整仍有偏。
2. **正性（positivity）**：对关心的 $(h_z,c)$ 组合，$P(C=c\mid h_z)>0$。距离极远或极冷门桶经常违规，边缘化方差会爆。
3. **$Z$ 的代理充分性**：$h_z$ 需近似 $Z$ 且尽量不携带 $C$。这不可由数据单独证明，要用 §5 的解耦约束去逼近，并用敏感性分析（§8）检查。
4. **可加性等功能形式**（可选、更强）：分数分解还假设偏好与混杂在 logit 上近似可加；该假设失败时，$s_{\mathrm{pref}}$ 仍可能吸收 $C$。

没有这些假设时，§5 的算法仍可作为**归纳偏置 / 正则**，但不要宣称已识别因果效应。

把 $C$ 拼进 token（§2 实现建议）只是在拟合 $P(Y\mid H,C)$；**干预**发生在训练约束 + 推理时固定 $\bar c$ 或对 $C$ 边缘化。$C$ 的四条用法可并行、语义不同：条件化输入、打分支路、采样/损失权重、评估分层。

---

## 4. 三类偏差：因果路径 → 实现症状 → 对策

每小节先对应 DAG，再谈 Transformer 里看起来像什么，最后指向 §5 算法编号。

### 4.1 可达性

**因果：** $C_{\mathrm{access}}\to Y$ 且 $C_{\mathrm{access}}\to H$。相对 $Z\to Y$，邻近既是真实约束，也是后门 / 捷径。反事实问题：若 $do(C_{\mathrm{access}}=\text{同等可达})$，排序是否仍指向同一 POI。

**实现症状：** 训练转移大量短距。

1. 因果注意力过度依赖最后几个 token（recency 与 proximity 耦合）。
2. embedding 里常共现的近邻 POI 聚成一团；点积打分退化成近邻检索。
3. 远距离正样本稀少，梯度几乎不要求区分“同距离环带内谁更符合兴趣”。

**对策：** §5.1 分数分解；$s_{\mathrm{access}}$ 由距离 / 半径 / 预算给出，$s_{\mathrm{pref}}$ 才是 Transformer 主头。§5.3 同距离环带负采样。可选 §5.4 距离桶 IPS。评估必须按距离桶切片（§7）。

### 4.2 流行度

**因果：** $C_{\mathrm{pop}}\to H,Y$。热门更容易被写进历史，也更容易被选为 $Y$。观测 CE 把 $P(p)$ 学成“兴趣”。

**实现症状：**

1. 输出层权重 / 偏置隐式学到 unigram $P(p)$。
2. 用户向量塌成“跟热门对齐”，小众兴趣用户失效。
3. 若干 attention head 复读高频模式，而非个人转移语法。

**对策：** §5.1 两头打分 $s_{\mathrm{match}}(h_z,p)+g(u)\,\mathrm{pop}(p)$，推理可关 $g$。§5.2 要求 $h_z\perp C_{\mathrm{pop}}$（近似）。辅助类别头：兴趣对“咖啡 vs 酒吧”更稳，对“哪家网红店”更易受 pop 污染。评估按流行度四分位 / 长尾。

### 4.3 区域

**因果：** $C_{\mathrm{area}}\to H,Y$。商圈午餐 vs 居民区晚间是 **条件机制不同**，不是 $Z$ 整体翻转。若不把 area 当作 $C$，它会经 lat/lon、最近类别统计漏进 $h$，与 $Z$ 纠缠。

**实现症状：** Self-attention 的“上下文”经常是当前 area 的局部模式；跨区远跳被当成异常。

**对策：** 显式 $C_{\mathrm{area}}$。条件化 $P(Y\mid h_z,a_{\mathrm{from}},t)$ 或加 $s_{\mathrm{area}}(a\to a_p)$ 支路（§5.1、§5.2）。跨 area 子集单独评估。Group DRO 把 area 切片当环境（§5.5）。

---

## 5. 算法与理论分析

统一模板：**逼近的因果目标 → 关键假设 → 算法形式 → 偏差 / 方差**。
IPS 改样本权重；Transformer 更需要对齐其归纳偏置：**表征 $h$、注意力、词表打分头**。主推 5.1–5.2–5.5；5.3 对可达性最直接；5.4 仅辅助。实现层面的变量入模方式见 §2，此处只给理论。

### 5.1 分数分解

**目标：** 把生成式中的 Access / Pop / Area 从偏好路径剥离，使 $s_{\mathrm{pref}}$ 更接近 $Z\to Y$。

**假设：** 真实 log-odds 近似可加

$$
\log\frac{P(Y=p\mid z,c)}{P(Y=p_0\mid z,c)}
\approx m(z,p)+g(c,p).
$$

**形式：**

$$
s(u,p,t)=s_{\mathrm{pref}}(h_z,p)+\lambda(t)\,s_{\mathrm{access}}(u,p,t)
$$

流行度 / 区域同理并入 $g(c,p)$。$s_{\mathrm{access}}$ 由距离、活动半径、时间预算等给出，**不要**再偷塞回无名 $h$。

**理论要点：** 若可加性大致成立，把 $g$ 参数化为显式支路、$m$ 交给 Transformer，可降低 $C$ 泄漏进 $h$ 的诱因。这是功能形式约束，不是完整识别。

**风险：** 支路欠定或过弱时，$m$ 仍吸收 $C$（省略偏差）。支路过强（例如硬距离核）会把真实的“愿意走远看展览”也罚掉。推理时 $\lambda$ 可调：写实保留，去混淆缩小或边缘化。

### 5.2 双表征 + 后门调整（主推）

**目标：** 逼近

$$
\sum_c P(Y\mid h_z,c)\,P(c)
\quad\text{或}\quad
P(Y\mid h_z,c=\bar c).
$$

**假设：** §3.3 的后门 + 正性；且 $h_z$ 是 $Z$ 的充分代理。对抗项在经验分布上惩罚 $I(h_z;C)$，**不保证**因果充分，只减小后门残留。

**形式：**

$$
h \xrightarrow{\mathrm{split}} (h_z, h_c),\quad
C=[\mathrm{dist\_bucket},\,\log\mathrm{pop},\,\mathrm{area},\,\mathrm{hour}]
$$

$$
\mathcal{L}=\underbrace{\mathrm{CE}(Y\mid h_z,h_c)}_{\text{拟合 }P(Y\mid h_z,h_c)}
+\lambda_1\underbrace{\mathrm{Adv}(C\mid h_z)}_{\text{逼 }h_z\perp C}
+\lambda_2\underbrace{\mathrm{CE}(C\mid h_c)}_{\text{让 }h_c\text{ 吃混杂}}
$$

- 训练：用真实 $c$ / $h_c$ 拟合条件分布。
- 约束：$h_z$ 难预测 $C$；$h_c$ 要能预测 $C$。
- 推理写实：用真实 $c$。
- 推理去混淆：$\sum_{c'}P(Y\mid h_z,c')\hat P(c')$ 或 $do(C=\bar c)$。

完全 $h\perp C$ 过强（兴趣与常驻 area 相关）。只对 $h_z$ 去 $C$；$h_c$ 吃混杂并进入 access / pop / area 支路。

**风险：** 正性不足时边缘化方差大，需对 $c$ 分桶而非逐值求和。对抗过强会抽干 $h_z$ 中与常驻地绑定、但属于 $Z$ 的信息 → 保持 partial disentanglement。

### 5.3 同距离环带负采样 / 对比

**目标：** 在局部 $do(C_{\mathrm{access}}\approx\text{同桶})$ 下识别偏好序。

**假设：** 桶 $b$ 内 $C_{\mathrm{access}}$ 变化可忽略；桶内仍可能残留 pop / area 混杂。

**形式：** 负例从与正样本同一距离环带（及可选同一 area）抽取，逼迫 attention + 输出层在“一样近”的集合里学 $s_{\mathrm{pref}}$。

**理论要点：** 条件于距离桶 $b$ 的 softmax / contrastive 近似

$$
P(Y\mid H,\, C_{\mathrm{access}}\in b).
$$

这是 **分层条件化**，不是完整后门；对可达性捷径最直接。跨桶聚合或按桶报告评估，避免近邻样本淹没一切。

**风险：** 桶太粗 → 桶内仍有 pop；太细 → 样本稀疏、对比崩溃。应与 5.1 / 5.2 叠用，而不是单独当作 $do(C)$。

### 5.4 IPS（辅助）

**目标：** 用 $w=1/\hat\pi(\mathrm{unit})$ 提高稀有远跳 / 长尾曝光的梯度份额。

**假设：** 倾向模型 $\pi$ 正确、权重有界。Next-POI 的“处理”是多值的（哪个 POI、多远），倾向定义含糊。

**形式：** 对距离桶或流行度分位赋权，**必须裁剪**。不替代条件化结构。

**理论要点：** 正确 IPS 对 Horvitz–Thompson 型目标无偏，但方差随 $1/\pi$ 恶化。这里“处理”不是干净的二值试验，故 **只作稀有单元提权**，不作主估计量。

**与后门的关系：** IPS 改采样分布 / 样本权重；后门改条件化与推理。可并用，不要当成同一件事。

### 5.5 Group DRO / IRM

**目标：** 学在多环境 $\{e\}$（距离 × 流行度 × area 切片）上稳定的 $h_z\to Y$（或类别）机制，逼近跨 $C$ 不变的偏好成分。

**假设：** 环境主要改变 $P(C)$，而 $Y\leftarrow f_Y$ 中依赖 $Z$ 的部分跨环境不变。环境划分错误会学错“不变集”。

**形式：**

$$
\min_\theta \sum_e \mathcal{L}_e(\theta)+\lambda\cdot\mathrm{Penalty}(\{\nabla_{w\mid e}\})
$$

或 $\min_\theta\sup_e\mathcal{L}_e$。符合 Transformer 想学的可迁移序列规律，而不是单环境捷径。

**风险：** 切片过细导致每组过少；只优化最难组可能牺牲写实主指标。优先对类别头做不变约束（更贴 $Z$），POI_id 头仍可条件化 $C$。

### 5.6 与语言模型技巧的类比

| LM / Transformer NLP | Next-POI 对应 | 主要落在 |
|----------------------|---------------|----------|
| 频率偏置 / unigram | POI 流行度头，可开关 | 5.1 |
| 领域自适应 / 不变表示 | 跨 area、跨距离桶的 $h_z$ | 5.2 / 5.5 |
| 控制生成（steer by attribute） | `do(C)` 改写混杂后再打分 | 5.2 推理 |
| 对比学习 / hard negatives | 同距离环带、同 area 内难负例 | 5.3 |
| 多任务（句法 vs 语义） | 类别 / 时间头 vs POI_id 头 | 5.2 辅助 |

不必引入 GCN，也能把因果干预做进 **encoder 输出与解码打分**。

### 5.7 最小实现草图（纯序列，无 GCN 假设）

```text
输入: 历史 POI/time/cat/user + 可算的 C 特征     # §2
Enc:  Embedding + Transformer → h               # 实现层；不是 DAG 节点
Split: h → h_z, h_c                             # h_z 代理 Z
Loss:  CE(Y | h_z, h_c)                         # 拟合 P(Y|h_z,h_c)  §5.2
     + λ1 * Adv(C | h_z)                        # 逼 I(h_z;C)↓      §5.2
     + λ2 * Pred(C | h_c)                       # h_c 吃混杂        §5.2
     + λ3 * GroupDRO(env)                       # 跨 C 切片不变     §5.5
     + 可选 环带对比 / 裁剪 IPS                  # §5.3 / §5.4
解码:  s = Dot(h_z, e_p) + g_access + g_pop + g_area   # §5.1, §6
推理_deconf:  用 c̄ 或边缘化 C；可关掉 g_pop / 缩小 g_access
推理_factual: 用真实 C
```

POI 向量 $e_p$ 可以是 `nn.Embedding`；若某系统用 GCN 生成 $e_p$，只需保证 **流行度与原始转移频次不要二次灌进 $h_z$**（附录 A）。

---

## 6. 统一分数与双模式推理

仍是 Transformer 解码，而非图扩散：

$$
s(u,p,t)=s_{\theta}(h_z,p)+\beta_a\,s_{\mathrm{access}}+\beta_p(u)\,s_{\mathrm{pop}}+\beta_r\,s_{\mathrm{area}}
$$

- $s_{\theta}$：Transformer 主匹配分（点积 / MLP / tied embedding），对应 $m(z,p)$
- 后三项：显式混杂支路 $g(c,p)$，**不要偷塞回无名 $h$**

训练建议：

1. 主 CE：条件化 $C$ 的 $P(Y\mid h_z,h_c)$（§5.2）
2. 辅助：时间 + 类别（类别更贴 $Z$）
3. 正则：$h_z\perp C$ + Group DRO（§5.2、§5.5）
4. 可选：同距离环带对比；稀有距离桶裁剪 IPS

推理：

| 模式 | 用什么 | 回答的问题 |
|------|--------|------------|
| factual | 真实 $C$ / 保留 $\beta$ | 在真实约束下下一跳是谁 |
| deconfounded | $\bar c$ 或 $\sum_c P(Y\mid h_z,c)P(c)$；可关 $g_{\mathrm{pop}}$、缩小 $g_{\mathrm{access}}$ | 混杂固定 / 边缘化后偏好指向谁 |

两套分数都要报（§7），不要用去混淆分数去“刷”写实 Acc。

---

## 7. 评估协议（与 backbone 无关）

只报整体 Acc@k / MRR 会被近邻 × 热门刷高。固定报告：

- 距离桶 Acc@k / 远跳召回（对应 $C_{\mathrm{access}}$）
- 流行度四分位、长尾 POI（对应 $C_{\mathrm{pop}}$）
- 跨 area 转移子集（对应 $C_{\mathrm{area}}$）
- 低从众用户组（$Z$ 与 pop 更分离的子群）
- **factual vs deconfounded** 两套指标，以及 `do(C)` 下 top-k 稳定性

每套指标写明估的是 $P(Y\mid H,C_{\mathrm{obs}})$ 还是边缘化 / 固定 $C$ 的代理。诊断阶段可：看命中样本的距离 / pop 分布；去掉输出偏置、打乱距离特征，看跌多少。

---

## 8. 建议实验顺序与边界

### 8.1 顺序

1. **诊断**：命中样本的距离 / pop 分布；去掉输出偏置、打乱距离特征看跌多少。
2. **序列侧轻量干预**：同距离环带负采样（§5.3）；$s_{\mathrm{pref}}+s_{\mathrm{access}}$ 分解（§5.1）。
3. **表征去混淆**：$h_z/h_c$ + 后门边缘化 + Group DRO（§5.2、§5.5，主推）。
4. **Area 条件化**与双模式推理（§6）。
5. 需要时再对比“加 / 不加图先验”的消融——图不是前提。

### 8.2 风险与边界

- 真实下一跳确实受可达性与流行度影响；完全 `do(C=\bar c)` 适合兴趣推断，不一定适合写实预测。
- $Z$ 与常驻 area 不可完美识别；对 $C$ 的定义做敏感性分析（换网格粒度、换 pop 统计窗口）。
- 对抗训练需 partial disentanglement，避免抽干有用信息。
- 正性失败的桶（极远、极冷门）不要硬做边缘化，改为单独报告或合并桶。
- 词表极大时，去偏后的全 Softmax 仍贵；可用候选生成 + 重排，但重排阶段同样要带 $C$ 分解，否则偏差在第二阶段回流。

---

## 9. 一句话收束

从 Transformer 理念看，next-POI 的因果问题是：**序列模型把观测转移的生成约束（近、热、区）当成了可注意力、可 Softmax 的“语义”**。

因果层给出 $Z,H,Y,C$ 与可识别目标 $P(Y\mid h_z,do(C))$（或其边缘化）；实现层用 **表征拆分、打分分解、跨环境不变与双模式推理** 去逼近它。GCN 只是 POI embedding 的一种来源；换查表嵌入，同一套 SCM 与 deconfounded training 仍然成立。

---

## 附录 A. 若对照 GETNext：GCN 只是可选放大镜

GETNext =（可选）轨迹流图 GCN POI 嵌入 + Transformer 序列 +（可选）`NodeAttnMap` 加性先验。

- **根因仍在序列 CE**；去掉 GCN / AttnMap，偏差不会自动消失。
- GCN / `checkin_cnt` / 原始转移边权会 **额外** 把 pop 与近邻灌进 $e_p$ 与加性 logits，可能造成“表征一次 + 先验一次”的双重计入。
- 若保留图模块：让其只影响 factual 支路或 $h_c$，不要无约束打进 $h_z$。
- 研究去偏时，建议先在 **纯 Transformer + id embedding** 上验证因果模块，再决定是否加图。

## 附录 B. 方法速查

| 方法 | 作用位置 | 逼近的目标量 | 是否需要 GCN |
|------|----------|--------------|--------------|
| $s_{\mathrm{pref}}+s_{\mathrm{access}}+s_{\mathrm{pop}}+s_{\mathrm{area}}$ | 解码打分 | 可加 logit 下的 $m(z,p)+g(c,p)$ | 否 |
| $h_z/h_c$ 解耦 + 对抗 | Transformer 输出 | $h_z$ 代理 $Z$，$I(h_z;C)\downarrow$ | 否 |
| 后门调整 / `do(C)` | 条件头与推理 | $\sum_c P(Y\mid h_z,c)P(c)$ 或 $P(Y\mid h_z,\bar c)$ | 否 |
| Group DRO / IRM | 按环境划分的 CE | 跨 $P(C)$ 稳定的 $Z\to Y$ 成分 | 否 |
| 同距离环带对比 | 负采样 / 损失 | $P(Y\mid H,C_{\mathrm{access}}\in b)$ | 否 |
| IPS | 样本权重 | 稀有单元的 HT 加权（辅助） | 否 |
| 图边权归一化 / 移出 checkin_cnt | 仅当使用图编码器时 | 减少 $C$ 二次灌入 $e_p$ | 是（可选） |
