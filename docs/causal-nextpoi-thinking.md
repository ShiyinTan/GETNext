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
- 注意：transformer 本身只近似学习 $H\to Y$、$Z\to Y$ 的综合路径（展开见附录 C）。

**2. 额外变量的处理和注入：**

- $C_{\mathrm{pop}}$, $C_{\mathrm{area}}$, $C_{\mathrm{access}}$ 可由预处理脚本查表、计算后作为额外输入特征拼入 token embedding，也可用作训练或评估切片（如热度/区域分层评估）。
- $C_{\mathrm{access}}$ 涉及动态地理约束（如距离/预算/锚点），可在数据处理阶段为每个样本增加该属性，或用于候选生成/负采样。

**3. Loss/目标增强手段:**

- 对抗流行度/可达性偏差，可设计 reweighting loss（如 distance bucket 采样、流行度 IPS 提升长尾）。 “TODO: 具体怎么实现呢？distangle 各个部分的embedding，比如h_{pop}, h_{interest}, h_{short_interest}, h_{area}，可以看作是因果学习吗？”
- 也可将 $C_{\mathrm{pop}}$/$C_{\mathrm{access}}$ 做 confounder 干预（如 $do(C=\ldots)$ 分析）或边缘化处理。 “TODO: 什么是边缘化处理？具体化confounder干预”
- 评估指标应区分各类混杂因子下的效果（如分 distance、popularity 分桶 Acc@K），避免仅反映热门/近邻。 “TODO: 这里的意思是根据POI的距离评估模型，以及根据POI的popularity评估模型吗？”

---

TODO: 数据处理，怎么获取各个外部变量？交通网络层的access，而不是单纯经纬度的access。

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

即：让 Transformer 学的主表征尽量接近偏好 $Z$，再对混杂 $C$ 做干预或边缘化。 TODO: 这一步中，h_z获取了，但是不能单纯用h_z进行预测吧？其他的因素对预测准确性非常重要，应该怎么处理？

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

TODO: 这两者是等价的吗？do(C)和\sum P(Y|c)P(c)

后者即后门调整形式。写实 vs 去混淆是 **两种推理模式**，不是互相否定。 TODO: 具体说说写实和去混淆的区别，举例说明，两者都可以作为因果吗？

### 3.3 识别假设（写清楚才能谈算法）

要对 $C$ 做后门调整，至少需要：

1. **混杂可测（或有足够代理）**：$C$ 阻断 $Z\to Y$ 之外、经混杂进入 $Y$ 的路径。遗漏的 $U$ 若同时影响 $H$ 与 $Y$，调整仍有偏。 TODO: 这一部分没有看懂。
2. **正性（positivity）**：对关心的 $(h_z,c)$ 组合，$P(C=c\mid h_z)>0$。距离极远或极冷门桶经常违规，边缘化方差会爆。 TODO: 没看懂，这部分是用来干什么的？是后门调整的前提条件吗？什么叫违规？边缘化方差是什么，爆了会怎么样？
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

TODO: 具体解释这两个公式，有什么联系吗？是上面的概率公式推导出下面的score公式吗？以及各个符号、函数的意义，为什么这样设计。

**理论要点：** 若可加性大致成立，把 $g$ 参数化为显式支路、$m$ 交给 Transformer，可降低 $C$ 泄漏进 $h$ 的诱因。这是功能形式约束，不是完整识别。

**风险：** 支路欠定或过弱时，$m$ 仍吸收 $C$（省略偏差）。支路过强（例如硬距离核）会把真实的“愿意走远看展览”也罚掉。推理时 $\lambda$ 可调：写实保留，去混淆缩小或边缘化。

### 5.2 双表征 + 后门调整（主推）

**目标：** 逼近

$$
\sum_c P(Y\mid h_z,c)\,P(c)
\quad\text{或}\quad
P(Y\mid h_z,c=\bar c).
$$

**假设：** §3.3 的后门 + 正性；且 $h_z$ 是 $Z$ 的充分代理。对抗项在经验分布上惩罚 $I(h_z;C)$，**不保证**因果充分，只减小后门残留。 TODO: I(h_z;C)是什么？什么是不保证因果充分？

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

TODO: 具体CE(\cdot, \cdot)该怎么实现？添加损失，让h_c分别预测C吗？但是C作为输入，又作为损失的预测目标吗？推理去混杂的两个公式有区别的？是等价的吗？还是两种不同的去混杂方式？

完全 $h\perp C$ 过强（兴趣与常驻 area 相关）。只对 $h_z$ 去 $C$；$h_c$ 吃混杂并进入 access / pop / area 支路。

**风险：** 正性不足时边缘化方差大，需对 $c$ 分桶而非逐值求和。对抗过强会抽干 $h_z$ 中与常驻地绑定、但属于 $Z$ 的信息 → 保持 partial disentanglement。 TODO: 什么是边缘化方差过大？有什么影响？

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

TODO: 具体说明怎么叠加实现。

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

TODO: 不了解这个算法，DRO和IRM是什么？这个公式怎么来的？w是什么？

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

完整输入→输出→损失→双模式推理规格见 **附录 D**（§5.1 + §5.2 合并实现）。

```text
输入: 历史 POI/time/cat/user + 可算的 C 特征     # §2
Enc:  Embedding + Transformer → h               # 实现层；不是 DAG 节点
Split: h → h_z, h_c                             # h_z 代理 Z
Decode: s_pref(p)=⟨h_z,e_p⟩; s_conf(p)=g(C(p))+⟨W_c h_c,ψ(p)⟩; s=s_pref+s_conf
Loss:  L_main = CE(Y | s)                       # P(Y|H,C)，主损失      附录 D.4.1
     + λ_pref * L_pref (环带对比 on s_pref)     # 偏好通道            附录 D.4.2
     + λ_conf * L_conf (align s_conf ↔ g̃(C))   # 混杂通道            附录 D.4.3
     + λ_adv  * Adv(C | h_z)                    # h_z⊥C               附录 D.4.4
     + λ_recon* CE(C | h_c)                     # h_c 重建 C          附录 D.4.4
推理_deconf:  argmax s_pref；或边缘化 s_conf
推理_factual: argmax s = s_pref + s_conf
```

TODO: g_access, g_pop, g_area分别是什么？e_p又是什么，是transformer的序列输出吗？

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

---

## 附录 C. 补充说明：为何 Transformer「只近似」学习 $H\to Y$、$Z\to Y$ 的综合路径

> 对应 §2 实现建议中「注意：transformer 本身只近似学习 $H\to Y$、$Z\to Y$ 的综合路径」一句的展开说明。

### C.1 一句话版

标准 Transformer 的输入只有 $H$，监督目标是 $P(Y\mid H)$。它**不会**分别学出因果图里的 $H\to Y$ 和 $Z\to Y$ 两条机制，而是学一个**混在一起**的映射；其中 $Z\to Y$ 还是**经 $H$ 间接推断**的，所以只能说「近似」。

### C.2 为什么说学的是「综合路径」？

因果图里到 $Y$ 的路径不止一条：

```text
Z ──→ Y          （兴趣直接影响下一跳）
Z ──→ H ──→ Y    （兴趣先塑造历史，历史再影响下一跳）
H ──→ Y          （纯历史惯性 / 短程记忆）
C ──→ H, C ──→ Y （混杂也混进来）
```

但典型训练是：

$$
\min_\theta \; -\log P_\theta(Y \mid H)
$$

模型只看见 $(H, Y)$，**看不见 $Z$**。因此它拟合的是**一个**条件分布 $P(Y\mid H)$，里面同时包含：

| 因果成分 | 在 $P(Y\mid H)$ 里怎么体现 |
|----------|---------------------------|
| $H\to Y$ | 最近去过哪、重复访问、转移模式 |
| $Z\to Y$ | 只能通过 $H$ 里残留的偏好信号间接体现 |
| $Z\to H\to Y$ | 和上面纠缠在一起，分不开 |
| $C\to H, C\to Y$ | $C$ 已写进 $H$（近、热、区），又直接影响 $Y$，形成捷径 |

所以不是「Transformer 专门学 $H\to Y$，另外再学 $Z\to Y$」，而是**一个网络把多条路径压成一个函数**：

$$
h = \mathrm{TF}(H), \quad \hat{Y} = \mathrm{softmax}(s(h, \cdot))
$$

这在实现上等价于学 $P(Y\mid H)$，不是学干净的 $P(Y\mid Z)$ 或单独的 $P(Y\mid H, Z)$。

### C.3 为什么说「只近似」？

「近似」有三层意思：

**（1）$Z$ 不可直接观测**

$Z$ 是隐变量，只能从 $H$ 反推。数学上：

$$
P(Y\mid H) = \sum_{z,c} P(Y\mid H, z, c)\, P(z, c\mid H)
$$

Transformer 的 $h$ 是在逼近某个 $P(Y\mid H)$ 的充分统计，**不保证** $h \approx Z$，更不保证把 $H\to Y$ 与 $Z\to Y$ 拆开。

**（2）$H$ 里已经混了 $C$**

因为 $C\to H$，历史序列本身带近邻、热门、区域偏置。模型从 $H$ 预测 $Y$ 时，会把「兴趣」和「混杂捷径」一起学进去——§1 里 embedding / attention / softmax 那些捷径就是这个结果。

**（3）损失函数不区分路径**

CE 只要求「给定 $H$，把真实 $Y$ 打高分」。它**不奖励**「这条预测来自 $Z$」或「这条来自纯 $H$ 惯性」。哪条路径 loss 更低，模型就走哪条——往往是近、热、区捷径。

因此：Transformer **可以**从历史里**隐式**抽出一部分偏好（对应 $Z$），也会学历史惯性（$H\to Y$），但两者**缠在一个 $h$ 里**，且都带 $C$ 污染——这就是「近似」。

### C.4 与正文其他章节的对应

- §2 表格里写 $Z$「需端到端学习」「通过 Transformer 内隐建模」——$Z$ 没有独立输入，只能藏在 $h$ 里。
- §2 末的目标 $P(Y\mid h_z, do(C))$ —— 正因为标准 Transformer **只近似**综合路径，才需要后面把 $h$ 拆成 $h_z$ / $h_c$、打分分解、后门调整等（§5），去逼近「偏好」而不是「观测捷径」。

### C.5 直觉类比

把用户下一跳想成考试答题：

- $Z$：真实兴趣（内心）
- $H$：过去答题记录（已观测）
- $Y$：下一题选什么

老师只给你「过去记录 → 预测下一题」，不给你「内心兴趣」标签。你能从记录猜兴趣，但猜不准；记录里还有「最近常选附近选项」「常选热门选项」等噪声。你学到的是「综合猜题策略」，不是单独识别「兴趣驱动」和「惯性驱动」。

---

## 附录 D. 实现规格：分数分解（§5.1）+ 双表征后门调整（§5.2）

本节给出一种可落地的端到端方案，将 §5.1 的可加打分分解与 §5.2 的 $(h_z,h_c)$ 解耦及后门训练约束合并为单一模型。叙述采用「符号定义 → 数据构造 → 前向计算 → 损失 → 推理」的论文体例；不展开 Transformer、MLP 等模块的内部层结构。

### D.1 符号与任务定义

设 POI 词表 $\mathcal{P}=\{1,\ldots,|\mathcal{P}|\}$，用户集合 $\mathcal{U}$。一条训练样本由用户 $u\in\mathcal{U}$、历史轨迹与预测时刻构成：

$$
H=(p_1,t_1,a_1),\ldots,(p_T,t_T,a_T),\qquad p_i\in\mathcal{P},
$$

其中 $t_i$ 为时间特征（如日内归一化时刻），$a_i$ 为 POI 类别 id。记 **当前位** $p_T$ 为 origin，**监督标签** $Y=p_{T+1}\in\mathcal{P}$。

混杂向量取离散化形式（分桶后便于后门边缘化）：

$$
C=(c_{\mathrm{acc}},\,c_{\mathrm{pop}},\,c_{\mathrm{area}},\,c_{\mathrm{hour}}),
$$

分别对应距离桶、目标 POI 流行度档、区域 id、时刻桶。对 **候选 POI** $p\in\mathcal{P}$，记其候选级混杂

$$
C(p)=\big(c_{\mathrm{acc}}(p_T,p),\,c_{\mathrm{pop}}(p),\,c_{\mathrm{area}}(p),\,c_{\mathrm{hour}}(t_T)\big).
$$

**重要约定（无标签泄漏）：** $C(p)$ 仅依赖 $H$、候选 $p$ 的静态属性及 $p_T$，不将「到 $Y$ 的距离」注入所有候选共享的上下文向量；训练时 $C(Y)$ 只是 $C(p)$ 在 $p=Y$ 时的取值（见 D.3.4）。

模型学习条件分布 $P_\theta(Y\mid H,C)$ 的 deconfounded 近似，通过兴趣代理 $h_z$、混杂摘要 $h_c$ 与可加打分 $s=s_{\mathrm{pref}}+s_{\mathrm{conf}}$ 实现。

### D.2 数据预处理与查表

**静态 POI 表** $\mathcal{T}_{\mathrm{poi}}$（训练集一次统计，全体样本共享）：

| 字段 | 含义 | 用途 |
|------|------|------|
| $\mathrm{pop}(p)$ | POI $p$ 的训练集 check-in 频次 | $c_{\mathrm{pop}}(p)$ |
| $\mathrm{area}(p)$ | 网格 / 行政区 id | $c_{\mathrm{area}}(p)$ |
| $\mathrm{lat}(p),\mathrm{lon}(p)$ | 坐标 | 距离计算 |
| $e_p\in\mathbb{R}^{d_e}$ | POI 嵌入（查表，可选预训练） | 解码点积 |

**距离与分桶：** 预计算或运行时查询

$$
d(p_T,p)=\mathrm{Dist}(p_T,p)\quad(\text{Haversine 或路网距离}),
\qquad
c_{\mathrm{acc}}(p_T,p)=\mathrm{Bucket}(d(p_T,p)).
$$

**样本级量（仅来自 $H$）：** 用户 id $u$；当前时刻桶 $c_{\mathrm{hour}}(t_T)$。可选：由历史推断锚点 $\hat p_{\mathrm{home}}(u)$（与 $Y$ 无关）。

训练 DataLoader 输出元组：

$$
\bigl(u,\; H,\; Y,\; p_T,\; t_T,\; \{C(p)\}_{p\in\mathcal{S}}\bigr),
$$

其中 $\mathcal{S}\subseteq\mathcal{P}$ 为全词表或本步采样的候选子集（含正样本 $Y$）。

### D.3 模型结构与前向传播

整体分为 **编码器** $\mathcal{E}$、**表征拆分** $\mathcal{S}$、**解码器** $\mathcal{D}$、**混杂头** $\mathcal{G}$ 四部分。

```text
H  ──►  E  ──►  h  ──►  S  ──►  (h_z, h_c)
                              │
         候选 p ∈ S ──────────┼──►  D(h_z, h_c, e_p, C(p))  ──►  s(p)
                              │
                              └──►  G_conf(h_c)  ──►  ĉ  (训练期辅助)
```

#### D.3.1 编码

$$
h=\mathcal{E}(H,u)\in\mathbb{R}^{d},
$$

$\mathcal{E}$ 以 $(p_{1:T},t_{1:T},a_{1:T},u)$ 为输入，经序列编码器得到 **最后时间步** 上下文向量 $h$（或等价地 pooling 末位表征）。$h$ 不直接参与最终 POI 排序，仅作为后续拆分的输入。

#### D.3.2 表征拆分

$$
h_z=\mathcal{S}_z(h)\in\mathbb{R}^{d_z},\qquad h_c=\mathcal{S}_c(h)\in\mathbb{R}^{d_c}.
$$

语义约定：$h_z$ 为兴趣代理 $Z$ 的表示；$h_c$ 为混杂摘要，供条件化预测与 $C$ 重建。

#### D.3.3 解码：可加打分与条件分布（§5.1）

对每个候选 $p\in\mathcal{S}$，定义 **总 logit** 与 **分通道 logit**：

$$
s(p)=s_{\mathrm{pref}}(p)+s_{\mathrm{conf}}(p).
$$

**偏好通道**（仅含 $h_z$，对应 §5.1 的 $m(z,p)$）：

$$
s_{\mathrm{pref}}(p)=\langle h_z,\,e_p\rangle.
$$

**混杂通道**（候选级、仅依赖 $\phi(\cdot,p)$ 与 $h_c$，对应 §5.1 的 $g(c,p)$）：

$$
s_{\mathrm{conf}}(p)=g_{\mathrm{acc}}\bigl(\phi_{\mathrm{acc}}(p_T,p)\bigr)
+g_{\mathrm{pop}}\bigl(\phi_{\mathrm{pop}}(p)\bigr)
+g_{\mathrm{area}}\bigl(\phi_{\mathrm{area}}(p_T,p)\bigr)
+\langle W_c h_c,\,\psi(p)\rangle.
$$

$\phi_{\mathrm{acc}}$ 为 $d(p_T,p)$ 或 $c_{\mathrm{acc}}(p_T,p)$ 的嵌入；$\phi_{\mathrm{pop}}(p)=\log(1+\mathrm{pop}(p))$；$\phi_{\mathrm{area}}$ 为 $(\mathrm{area}(p_T),\mathrm{area}(p))$ 的联合嵌入；$g_{\cdot}$ 为标量输出头；$\psi(p)$ 为 POI 侧辅助嵌入。最后一项 $\langle W_c h_c,\psi(p)\rangle$ 编码 **无法完全查表** 的上下文混杂（时段、情境），可选。

**条件分布（主任务所拟合对象）：**

$$
P_\theta(Y=p\mid H,C)=\frac{\exp\bigl(s_{\mathrm{pref}}(p)+s_{\mathrm{conf}}(p)\bigr)}
{\sum_{p'\in\mathcal{S}}\exp\bigl(s_{\mathrm{pref}}(p')+s_{\mathrm{conf}}(p')\bigr)}.
$$

**要点：** $P(Y\mid H,C)$ 由 **总分数 $s$** 经 softmax 定义；$s_{\mathrm{pref}}$、$s_{\mathrm{conf}}$ 各自 **不是** 完整的 $P(Y\mid H,C)$，而是可分解、可单独约束与干预的子通道。全词表训练时 $\mathcal{S}=\mathcal{P}$；大规模词表可用 sampled softmax，$\mathcal{S}=\{Y\}\cup\mathcal{N}$。

#### D.3.4 候选级特征与无泄漏性

对任意 $p\in\mathcal{S}$，$s_{\mathrm{conf}}(p)$ 与 $s_{\mathrm{pref}}(p)$ 中凡依赖地理/流行的量，均通过 $\phi(\cdot,p)$ 计算，**推理时可对全体 $p$ 复现**。训练时

$$
s(Y)=s_{\mathrm{pref}}(Y)+s_{\mathrm{conf}}(Y)
$$

仅为 $p=Y$ 的特例，不构成将 $Y$ 独有信息注入共享 $h$ 的泄漏。$C$ 不作为 Transformer 输入 token 拼入 $H$；若需条件化，仅通过 $h_c$ 与 $s_{\mathrm{conf}}(p)$ 进入。

仅为 $p=Y$ 的特例，不构成将 $Y$ 独有信息注入共享 $h$ 的泄漏。$C$ 不作为 Transformer 输入 token 拼入 $H$；经 $h_c$ 与 $s_{\mathrm{conf}}(p)$ 进入模型。

### D.4 训练目标：主损失 + 分通道辅助损失

采用 **多任务** 结构：主损失用总 $s$ 拟合 $P(Y\mid H,C)$；$s_{\mathrm{pref}}$、$s_{\mathrm{conf}}$ 各有辅助损失，分别约束兴趣通道与混杂通道，避免两路信号全部挤进单一 CE。

$$
\mathcal{L}
=\underbrace{\mathcal{L}_{\mathrm{main}}}_{P(Y\mid H,C)}
+\lambda_{\mathrm{pref}}\underbrace{\mathcal{L}_{\mathrm{pref}}}_{s_{\mathrm{pref}}\text{ 通道}}
+\lambda_{\mathrm{conf}}\underbrace{\mathcal{L}_{\mathrm{conf}}}_{s_{\mathrm{conf}}\text{ 通道}}
+\lambda_{\mathrm{adv}}\underbrace{\mathcal{L}_{\mathrm{adv}}}_{h_z\perp C}
+\lambda_{\mathrm{recon}}\underbrace{\mathcal{L}_{\mathrm{recon}}}_{h_c\to C}.
$$

超参 $\lambda_{\mathrm{pref}},\lambda_{\mathrm{conf}},\lambda_{\mathrm{adv}},\lambda_{\mathrm{recon}}\ge 0$。推荐 $\mathcal{L}_{\mathrm{main}}$ 权重为 1，其余取 $10^{-2}\sim10^{-1}$ 量级并按验证集调节。

#### D.4.1 主损失：总 $s$ 拟合 $P(Y\mid H,C)$

$$
\mathcal{L}_{\mathrm{main}}
=-\log P_\theta(Y\mid H,C)
=-s(Y)+\log\sum_{p\in\mathcal{S}}\exp\bigl(s(p)\bigr),
\qquad s(p)=s_{\mathrm{pref}}(p)+s_{\mathrm{conf}}(p).
$$

这是 **唯一** 直接优化完整条件分布 $P(Y\mid H,C)$ 的项；factual 预测精度主要由 $\mathcal{L}_{\mathrm{main}}$ 决定。$\mathcal{L}_{\mathrm{pref}}$、$\mathcal{L}_{\mathrm{conf}}$ 不替代该项，只塑造两通道分工。

#### D.4.2 偏好通道损失 $\mathcal{L}_{\mathrm{pref}}$

约束 $s_{\mathrm{pref}}$ 在 **同等可达** 条件下学习偏好序，防止 $\mathcal{L}_{\mathrm{main}}$ 将全部预测压力泄入 $s_{\mathrm{conf}}$。

**（a）同距离环带对比（推荐，§5.3）：** 设 $\mathcal{N}_b(Y)$ 为与 $Y$ 同距离桶 $b=c_{\mathrm{acc}}(p_T,Y)$ 的负例集，

$$
\mathcal{L}_{\mathrm{pref}}
=-\log\frac{\exp\bigl(s_{\mathrm{pref}}(Y)\bigr)}
{\exp\bigl(s_{\mathrm{pref}}(Y)\bigr)+\sum_{p^{-}\in\mathcal{N}_b(Y)}\exp\bigl(s_{\mathrm{pref}}(p^{-})\bigr)}.
$$

仅在 $s_{\mathrm{pref}}$ 上做 softmax，逼近「$C_{\mathrm{access}}$ 近似固定时」的偏好排序。

**（b）类别辅助（可选）：** 从 $h_z$ 预测 $a_Y$（$Y$ 的 POI 类别），

$$
\mathcal{L}_{\mathrm{pref}}^{(\mathrm{cat})}=\mathrm{CE}\bigl(a_Y,\,\mathrm{Head}_{\mathrm{cat}}(h_z)\bigr),
$$

可与 (a) 加权相加：$\mathcal{L}_{\mathrm{pref}}=\mathcal{L}_{\mathrm{pref}}^{(\mathrm{ring})}+\eta\,\mathcal{L}_{\mathrm{pref}}^{(\mathrm{cat})}$。类别更贴 $Z$，有助于 $h_z$ 承载兴趣。

#### D.4.3 混杂通道损失 $\mathcal{L}_{\mathrm{conf}}$

约束 $s_{\mathrm{conf}}$ 显式承担近/热/区效应，减轻 $h_z$ 与 $e_p$ 吸收混杂捷径。

**（a）混杂打分对齐：** 用查表特征构造 **先验混杂分**（stop-gradient）：

$$
\tilde g(p)=\tilde g_{\mathrm{acc}}\bigl(\phi_{\mathrm{acc}}(p_T,p)\bigr)
+\tilde g_{\mathrm{pop}}\bigl(\phi_{\mathrm{pop}}(p)\bigr)
+\tilde g_{\mathrm{area}}\bigl(\phi_{\mathrm{area}}(p_T,p)\bigr),
$$

$\tilde g_{\cdot}$ 可为线性层或手工递减函数（如 $-\alpha\, d(p_T,p)-\beta\log\mathrm{pop}(p)$）。令

$$
\mathcal{L}_{\mathrm{conf}}^{(\mathrm{align})}
=\frac{1}{|\mathcal{S}'|}\sum_{p\in\mathcal{S}'}
\bigl(s_{\mathrm{conf}}(p)-\tilde g(p)\bigr)^2,
$$

$\mathcal{S}'\subseteq\mathcal{S}$ 为 mini-batch 内候选子集（含 $Y$ 与负例）。该项促使 $s_{\mathrm{conf}}$ 追踪可观测 $C(p)$，而不依赖 $Y$ 作为输入特征。

**（b）混杂通道弱 CE（可选）：** 仅用 $s_{\mathrm{conf}}$ 做预测，

$$
\mathcal{L}_{\mathrm{conf}}^{(\mathrm{aux})}
=-\log\frac{\exp\bigl(s_{\mathrm{conf}}(Y)\bigr)}
{\sum_{p\in\mathcal{S}}\exp\bigl(s_{\mathrm{conf}}(p)\bigr)}.
$$

权重宜小（$\ll \mathcal{L}_{\mathrm{main}}$），否则 $s_{\mathrm{conf}}$ 会吞掉本属 $s_{\mathrm{pref}}$ 的信号。默认 $\mathcal{L}_{\mathrm{conf}}=\mathcal{L}_{\mathrm{conf}}^{(\mathrm{align})}$；需更强 factual 校准时再加 (b)。

#### D.4.4 表征解耦：$\mathcal{L}_{\mathrm{adv}}$ 与 $\mathcal{L}_{\mathrm{recon}}$

**对抗项**（§5.2，作用于 $h_z$，非 $s_{\mathrm{pref}}$ 公式内）：

$$
\mathcal{L}_{\mathrm{adv}}=\sum_k \mathrm{CE}\bigl(c^{(k)}(Y),\,D_{\mathrm{adv}}^{(k)}(h_z)\bigr),
$$

编码器侧最小化 $\mathcal{L}_{\mathrm{main}}+\lambda_{\mathrm{pref}}\mathcal{L}_{\mathrm{pref}}+\cdots-\lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{adv}}$（梯度反转），使 $h_z$ 难预测 $C$。

**混杂重建**（§5.2，作用于 $h_c$）：

$$
\mathcal{L}_{\mathrm{recon}}=\sum_k \mathrm{CE}\bigl(c^{(k)}(Y),\,\mathcal{G}^{(k)}_{\mathrm{recon}}(h_c)\bigr).
$$

注意区分：$\mathcal{L}_{\mathrm{recon}}$ 监督 **向量 $h_c$** 重建 $C$；$\mathcal{L}_{\mathrm{conf}}$ 监督 **标量 $s_{\mathrm{conf}}(p)$** 对齐 $\tilde g(p)$。二者互补，不重复。

#### D.4.5 损失分工小结

| 损失 | 优化对象 | 作用 |
|------|----------|------|
| $\mathcal{L}_{\mathrm{main}}$ | $s=s_{\mathrm{pref}}+s_{\mathrm{conf}}$ | 拟合 $P(Y\mid H,C)$，主任务 |
| $\mathcal{L}_{\mathrm{pref}}$ | 仅 $s_{\mathrm{pref}}$ | 同距离环带内学偏好；$h_z$ 承载兴趣 |
| $\mathcal{L}_{\mathrm{conf}}$ | 仅 $s_{\mathrm{conf}}$ | 对齐 $\tilde g(C(p))$；近/热/区走混杂通道 |
| $\mathcal{L}_{\mathrm{adv}}$ | $h_z$ | $h_z\perp C$ |
| $\mathcal{L}_{\mathrm{recon}}$ | $h_c$ | $h_c$ 编码可观测 $C$ |

**设计可行性：** 多任务在推荐与去偏文献中常见。关键是 $\mathcal{L}_{\mathrm{main}}$ 始终对 **相加后的 $s$** 做 CE，保证整体仍建模 $P(Y\mid H,C)$；分通道损失提供 **归纳偏置**，使去混淆推理时 $s_{\mathrm{pref}}$ 可单独使用或搭配干预后的 $s_{\mathrm{conf}}$。

### D.5 推理：Factual 与 Deconfounded

记 $\mathcal{S}$ 为候选 POI 集合（全词表或召回集）。对 $p\in\mathcal{S}$ 计算 $s_{\mathrm{pref}}(p)$ 与 $s_{\mathrm{conf}}(p)$。

**（1）Factual 推理** — 估计 $P(Y\mid H,C_{\mathrm{obs}})$：

- 用历史编码得 $h_z,h_c$；
- $s_{\mathrm{conf}}(p)$ 使用 **真实** $\phi(p_T,p)$、$\mathrm{pop}(p)$、$\mathrm{area}(\cdot)$；
- $\hat Y=\arg\max_{p\in\mathcal{S}} s(p)$。

**（2）Deconfounded 推理（固定干预）** — 估计 $P(Y\mid h_z, do(C=\bar c))$：

- $h_z,h_c$ 同 factual；
- 将 $s_{\mathrm{conf}}(p)$ 中的 $\phi$ 换为常数 $\bar\phi$，或令 $s_{\mathrm{conf}}(p)\equiv 0$；
- $\hat Y_{\mathrm{deconf}}=\arg\max_{p\in\mathcal{S}} s_{\mathrm{pref}}(p)$，或 $\arg\max_p\bigl[s_{\mathrm{pref}}(p)+\tilde s_{\mathrm{conf}}(p;\bar\phi)\bigr]$。

因 $s_{\mathrm{pref}}$ 仅含 $h_z$，首选 **纯偏好排序** $\arg\max s_{\mathrm{pref}}$。

**（3）Deconfounded 推理（边缘化）** — 估计 $\sum_c P(Y\mid h_z,c)P(c)$：

对离散桶 $\{c_j\}_{j=1}^J$，用训练集频率 $\hat P(c_j)$：

$$
\bar s(p)=\sum_{j=1}^{J}\hat P(c_j)\,
\bigl[s_{\mathrm{pref}}(p)+s_{\mathrm{conf}}(p;\,c_j)\bigr].
$$

$s_{\mathrm{pref}}(p)$ 与 $c_j$ 无关，边缘化只作用于 $s_{\mathrm{conf}}$。$J$ 不宜过大，以免正性不足导致方差膨胀（§3.3）。

两套排序 $\hat Y$ 与 $\hat Y_{\mathrm{deconf}}$ **均应报告**（§7），分别对应写实预测与兴趣导向预测。

### D.6 训练与推理算法

**Algorithm 1** 训练一步（mini-batch）

```text
输入: batch {(u, H, Y)}，POI 表 T_poi，超参 λ_main=1, λ_pref, λ_conf, λ_adv, λ_recon
1.  p_T ← H 的最后 POI；h ← E(H, u)；(h_z, h_c) ← S(h)
2.  构造全体 p ∈ S 的 C(p)；s_pref(p) ← ⟨h_z, e_p⟩
3.  s_conf(p) ← g_acc(φ_acc) + g_pop(φ_pop) + g_area(φ_area) + ⟨W_c h_c, ψ(p)⟩
4.  s(p) ← s_pref(p) + s_conf(p)
5.  L_main ← -s(Y) + log Σ_{p∈S} exp(s(p))                    // P(Y|H,C)
6.  L_pref ← 环带对比 CE，仅在 s_pref 上（§D.4.2）              // 可选 + L_pref^(cat)
7.  L_conf ← MSE(s_conf(p), g̃(p))  on S'                       // 可选 + 弱 CE on s_conf
8.  L_adv ← Σ_k CE(c^(k)(Y), D_adv^(k)(h_z))；L_recon ← Σ_k CE(c^(k), G_recon^(k)(h_c))
9.  L ← L_main + λ_pref·L_pref + λ_conf·L_conf + λ_recon·L_recon - λ_adv·L_adv
10. 反向传播并更新参数
```

**Algorithm 2** Deconfounded 推理（单用户单步）

```text
输入: H, u, 候选集 S, 干预参数 c̄ 或边缘化分布 P̂(c)
1.  (h_z, h_c) ← S(E(H, u))
2.  factual:     对每个 p ∈ S 算 s(p) 用真实 C(p)；取 top-k
3.  deconf (do):  将 s_conf 中 φ 换为 φ̄；取 top-k
4.  deconf (sum): s̄(p) ← Σ_j P̂(c_j) · s(p; c_j)；取 top-k
```

### D.7 与理论目标的对应关系

| 组件 | 理论角色（§5） | 说明 |
|------|----------------|------|
| $h_z$ | $Z$ 的代理 | $\mathcal{L}_{\mathrm{adv}}$ 减小 $C$ 泄漏 |
| $h_c$ | 混杂摘要 | $\mathcal{L}_{\mathrm{recon}}$ 保证可重建 $C$ |
| 推理 §D.5(2)(3) | $do(C)$ / 后门边缘化 | 测试兴趣排序 |

本方案 **不保证** 因果效应可识别（§3.3），但通过 **主损失拟合 $P(Y\mid H,C)$** 与 **分通道辅助损失** 分工，将朴素 CE 分解为可解释的偏好与混杂路径，并支持双模式评估。

### D.8 实现边界与默认选择

1. **POI 嵌入 $e_p$：** 默认 `nn.Embedding`；若使用 GCN，$e_p$ 不得再含未归一化的 `checkin_cnt` 灌入 $h_z$（附录 A）。
2. **词表规模：** $|\mathcal{S}|=|\mathcal{P}|$ 不可行时，用负采样估计 $\mathcal{L}_{\mathrm{main}}$；$\mathcal{L}_{\mathrm{pref}}$、$\mathcal{L}_{\mathrm{conf}}^{(\mathrm{align})}$ 在相同 $\mathcal{S}$ 上计算。
3. **$C$ 粒度：** 距离 / 流行度 / 区域均用 **分桶** 而非连续值，便于 $\mathcal{L}_{\mathrm{recon}}$、$\mathcal{L}_{\mathrm{adv}}$ 与边缘化求和。
4. **$\lambda$ 调节：** 若 $\mathcal{L}_{\mathrm{conf}}^{(\mathrm{aux})}$ 过大导致 $s_{\mathrm{conf}}$ 主导 $\mathcal{L}_{\mathrm{main}}$，应降低 $\lambda_{\mathrm{conf}}$ 或去掉弱 CE，仅保留 align。
5. **不包含：** §5.4 IPS、§5.5 Group DRO 可作为正交扩展叠加，非本规格必需项；环带对比已纳入 $\mathcal{L}_{\mathrm{pref}}$。

