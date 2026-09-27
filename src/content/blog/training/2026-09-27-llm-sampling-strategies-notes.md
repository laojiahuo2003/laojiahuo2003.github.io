---
title: LLM 采样与解码策略综述：确定性解码、温度缩放、截断方法与自适应控制
description: 从 logits 到 token 的处理流水线讲起，逐个拆开确定性解码（Greedy / Beam Search）、温度缩放、截断类方法（Top-k、Top-p、Min-p、Top-nσ、Typical、Epsilon/Eta、Tail-free、p-LESS）、重复抑制（Repetition/Presence/Frequency Penalty、no_repeat_ngram、DRY、XTC）、自适应控制（Mirostat）与投机解码，最后落到采样阶段的计算开销、kernel 实现与参数配置建议。
pubDate: 2026-09-27T21:00:00+08:00
tags: [大模型, 训练工程]
---

## 1. 问题定义与处理流水线

模型在每个位置只做一件事：输出一个 **logits** 向量（维度等于词表大小 V，通常是几万到几十万）。logits 本身是未归一化的实数分数，它不告诉你「该选哪个 token」——**从 logits 到最终选出的那个 token，中间所有的规则，统称采样（Sampling）或解码（Decoding）策略。**

先给一个最小的具体例子。假设词表里只有 5 个 token，某一步输出的 logits 是：

```text
token :   A      B      C      D      E
logits:  2.20   1.60   1.10   0.70   0.20
```

经过 softmax（温度 T = 1）后，得到一组概率：

```text
p = [0.447, 0.245, 0.149, 0.100, 0.059]      （和 = 1.00）
```

这组概率就是全部讨论的起点。完整的处理流水线是：

```text
logits
  │
  ├─ 温度缩放（Temperature）        改分布的「形状」
  │
  ├─ 惩罚与抑制（Penalty / Ban）    改特定 token 的分数
  │
  ├─ 截断（Truncation）             砍候选集合
  │
  ├─ softmax                        归一化成概率
  │
  └─ 多项分布采样                    掷骰子
```

这里有个必须先分清的区分：

- **温度** 改的是分布的形状——拉大或压缩 token 之间的概率差距；
- **截断** 改的是候选集合——把某些 token 直接从候选里划掉。

两者正交。很多人把它们混在一起调，所以怎么调都别扭：温度调高想加创意，结果长尾里的垃圾 token 也一起被抬起来；温度调低想稳妥，结果输出千篇一律，而且该复读还是复读。

另外要注意截断的一个性质：**截断不是「把砍掉的概率扔掉」，而是重归一化**——被划掉的 token 的概率会被按比例分给留下的 token。所以截断之后，留下的 token 概率会比原来**更大**，这会给采样引入偏差（bias）。在第 4 节会解释为什么这个偏差是值得的。

## 2. 确定性解码：Greedy 与 Beam Search

### 2.1 Greedy（贪心解码）

每步取 `argmax`，等价于温度趋近 0 的极限。

```text
p = [0.447, 0.245, 0.149, 0.100, 0.059]
            ↓
       永远选 A
```

**适用**：有唯一正确答案的任务——分类、信息抽取、结构化输出、需要稳定复现的场景。
**局限**：

- 没有任何多样性，同一个 prompt 永远同一个输出；
- 容易进入重复循环，小模型尤其明显（模型「相信」自己刚才说过的话，于是继续说）；
- 逐步最优不等于整句最优——贪心只看当前这一步的最大概率，不看后面的走向。

严格来说，温度设为 0 也不保证逐位可复现的确定性，原因在第 12 节。

### 2.2 Beam Search（束搜索）

同时保留 k 条候选序列（k 叫 beam width，束宽），每步把每条序列向词表扩展，按**整句对数概率**排序，剪枝回 k 条。

```text
step 1:   [A: -1.20]        [B: -1.90]
              │                  │
step 2:   A→Ax, A→Ay        B→Bx, B→By
              │                  │
          按累计 logP 排序，保留前 k 条
```

必须配 **length normalization（长度归一化）**：对数概率越加越负，不归一化的话束搜索会系统性偏向短句。常见做法是除以长度，或者除以 `length^α`（α 一般取 0.6~1.0）。

**位置**：翻译、摘要这类「输出较短、有较明确的参考标准」的任务仍然用它。
**为什么对话系统基本不用**：

1. 最大化整句概率会偏向高频、安全的表达，输出冗长呆板（这个现象正是 [Holtzman 等人提出 nucleus sampling](https://arxiv.org/abs/1904.09751) 的直接动机）；
2. 计算量约 k 倍——k 条序列的 KV cache、显存、每步前向都要 ×k；
3. 开放式生成本来就不存在「唯一正确」，最大化整句概率是个错的目标函数。

## 3. 温度缩放（Temperature Scaling）

温度在 **softmax 之前**作用在 logits 上：

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

用开头的例子算一遍（T = 0.5 / 1 / 2，数值保留三位小数）：

| T | A | B | C | D | E |
|---|---|---|---|---|---|
| 0.5 | 0.676 | 0.204 | 0.075 | 0.034 | 0.012 |
| 1.0 | 0.447 | 0.245 | 0.149 | 0.100 | 0.059 |
| 2.0 | 0.317 | 0.235 | 0.183 | 0.150 | 0.117 |

三个 T 下的关键观察：

- **T < 1 让分布变尖**：头部更集中（A 从 0.447 涨到 0.676）；
- **T > 1 让分布变平**：尾部被抬起来（E 从 0.059 涨到 0.117）；
- **任何 T 都不改变 token 的排序**，只改变它们之间的差距。

**最后一条是最关键的**：因为温度只改差距、不改排序，所以**温度单独使用，永远无法阻止低概率 token 被抽到**。它只是把「被抽到的概率」调大或调小。要在结构上排除长尾，必须靠截断。

这也解释了参数调优里的一条常见反模式：为了减少胡言乱语而把温度调到 0.1——此时分布已经很尖，但尾部依然非零，一旦头部几个 token 都不合适（模型困惑的位置），你依然会从尾部抽，而且抽出来的是更突兀的东西，因为你顺手把头部也压没了。

**工程注意点**：

- 温度实现在 logits 上做除法，**不要**先 softmax 再对概率取幂——虽然数学上 `p^(1/T)` 归一化等价，但概率取幂会有下溢风险，logits 上做更稳；
- `T = 0` 在实现里是特例（直接 argmax），不要真的去写除法，否则会得到 `0/0`。

**Dynamic Temperature（动态温度）**：固定温度的问题是，同一次生成里有的位置模型极度确定（该压低温）、有的位置模型摇摆（该放宽）。动态温度只在**模型过于自信时抬高温度**，典型实现是

```text
if max_prob > 阈值:
    T = base_T + (max_prob - 阈值) × 放大系数
```

白话：模型越笃定，越要给它加噪声，避免它在一条路上越走越深。这属于工程启发式，没有严格论文支撑，但在角色扮演、创意写作的社区实现里很常见。

## 4. 截断方法的理论依据：Desmoothing

在逐个介绍截断方法之前，先说清一件事：**为什么截断这种「明显有偏」的操作能提升生成质量？**

[Hewitt 等人的论文](https://arxiv.org/abs/2210.15191)（标题即 *Truncation Sampling as Language Model Desmoothing*）给出了一个解释：

神经语言模型的分布在**尾部系统性地偏厚**。真实语言里几乎不可能出现的 token，模型会给它一个非零概率。因为模型是用最大似然在有限数据上训出来的，它学到的分布被「平滑」过——而真实语言的分布要稀疏得多。

于是截断的定位变了：它不是拍脑袋的工程 trick，而是**在估计一个更稀疏的分布**——把模型多给出来的那层「人为平滑」去掉。

这就回答了两个问题：

- **为什么截断值得引入偏差**：因为前提是「模型给的分布本身就是错的」。在这个前提下，忠实地从模型分布采样反而不是最优的。截断用一点偏差，换掉的是一堆本来不该出现的 token。
- **为什么阈值不能一刀切**：模型在不同位置的「尾部有多厚」是不一样的。模型笃定时尾部本来就该薄，模型犹豫时尾部本来就是真实的不确定性。**阈值应当随模型置信度变化**——这正是后面 Min-p、Top-nσ 这类「自适应阈值」方法的设计动机，也是截断方法演化的主线。

论文提出的 **η-sampling（Eta Sampling）** 就是从「最优的稀疏分布应该是什么样」反推出来的截断规则，见第 8 节。

## 5. 截断方法（一）：Top-k 与 Top-p

### 5.1 Top-k：固定数量

保留概率最高的 k 个 token，其余置零，然后重归一化。

```text
p = [0.447, 0.245, 0.149, 0.100, 0.059]，k = 3

候选 = {0.447, 0.245, 0.149}   和 = 0.841
重归一化后 = [0.531, 0.291, 0.177]
```

注意 D、E 的 0.159 没有消失，它被按比例分给了 A、B、C——所以 A 从 0.447 变成了 0.531。

**局限：k 是固定的绝对数量，不随分布形状变化。**

- 分布很平（模型困惑、有多种合理说法）时，k 太小会把合理选项截掉；
- 分布很尖（模型笃定）时，k 太大仍然会放进长尾垃圾。

选 k 变成了一件「必须同时适应两种极端」的难事，这是 Top-k 最根本的缺陷。[Top-k 最早出现在 story generation 的工作里](https://arxiv.org/abs/1805.04833)，现在基本只作为兜底参数（比如 `k = 20`）出现。

### 5.2 Top-p / Nucleus Sampling（核采样）

不固定数量，而是固定**累积概率**：保留概率从高到低累加、刚好达到 p 的那个最小集合。

```text
p = [0.447, 0.245, 0.149, 0.100, 0.059]，p_threshold = 0.9

累计：0.447 → 0.692 → 0.841 → 0.941 ≥ 0.9，停
候选 = 前 4 个（A, B, C, D），和 = 0.941
重归一化后 = [0.475, 0.260, 0.158, 0.106]
```

比 Top-k 好的地方：**候选数量随分布自动伸缩**。

**它仍然有的洞**：p 控制的是「留下多少总概率质量」，**不控制「单个 token 的质量」**。

- 模型笃定（top1 = 0.95）时，累积到 0.9 只需要 1 个 token，很好；
- 模型弥散（top1 = 0.05）时，累积到 0.9 可能要吞进几百个 token，里面就混着垃圾。

也就是说，**同一个 p，在模型自信时剪得合理，在模型犹豫时放得过宽**——而模型犹豫的位置，恰恰是最需要谨慎的地方。

## 6. 截断方法（二）：Min-p

Min-p 的核心改动只有一行：**阈值不设成绝对值，而是设成「最大概率 token 的某个比例」**。

$$
\text{候选} = \{\, i \;\mid\; p_i \;\geq\; \text{min\_p} \times \max_j p_j \,\}
$$

用两个方向相反的分布看它的行为。**情况一：模型较自信。**

```text
p        = [0.75, 0.10, 0.06, 0.04, 0.03, 0.02]
min_p    = 0.1  →  阈值 = 0.1 × 0.75 = 0.075
候选     = {0.75, 0.10}                                → 2 个

top_p    = 0.9  →  累计 0.75 → 0.85 → 0.91 ≥ 0.9
候选     = {0.75, 0.10, 0.06}                          → 3 个
```

**情况二：模型较犹豫。**

```text
p        = [0.30, 0.20, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01]
min_p    = 0.1  →  阈值 = 0.1 × 0.30 = 0.03
候选     = 前 8 个（到 0.03 为止）                      → 8 个

top_p    = 0.9  →  累计到第 6 个恰好 0.90
候选     = 前 6 个                                      → 6 个
```

把两种情况的结论放在一起：

| 情况 | Min-p 候选数 | Top-p 候选数 | 谁更紧 |
|---|---|---|---|
| 自信（top1 = 0.75） | 2 | 3 | Min-p 更紧 |
| 犹豫（top1 = 0.30） | 8 | 6 | Min-p 更松 |

**同一个 min_p，在模型自信时比 Top-p 收得更紧，在模型犹豫时比 Top-p 放得更松。** 这正是第 4 节说的「阈值应当随置信度变化」，Min-p 用一行代码实现了它。

Min-p 还有一个对调参很友好的性质：**它和温度解耦**。因为阈值是相对量，温度把分布整体压平之后，`min_p × p_max` 也跟着一起缩放，不会像 Top-p 那样「温度一高、尾巴膨胀、候选爆掉」。所以想加创意时，可以放心地「抬温度 + Min-p 兜底」，而不是「抬温度 + Top-p」。

论文是 [*Min-p Sampling for Creative and Coherent LLM Outputs*](https://arxiv.org/html/2407.01082v8)，经验值 `min_p ≈ 0.05 ~ 0.1`。

**局限**：分布极度弥散时（top1 只有 0.02 这种），阈值会被压到很低，退化成几乎不截断；另外 `min_p` 这个超参本身仍需按任务调，没有做到完全免调参。

## 7. 截断方法（三）：Top-nσ

换一个参照系：不看「和第一名比」，看「和全体统计量比」。**在 logits 上**（不是概率上）计算均值和标准差：

$$
\text{阈值} = \mu + n \cdot \sigma, \qquad \text{候选} = \{\, i \;\mid\; z_i \geq \mu + n\,\sigma \,\}
$$

用开头的 logits 算一遍：

```text
z     = [2.20, 1.60, 1.10, 0.70, 0.20]
mean  = 1.16            （5.80 / 5）
std   = 0.695           （方差 0.482，开方）

n = 1.0  →  阈值 = 1.16 + 0.695 = 1.855
候选 = {A: 2.20, B: 1.60}          → 2 个
```

**它为什么自适应**：σ 本身就是分布弥散程度的度量。分布集中时 σ 小、阈值紧；分布弥散时 σ 大、阈值松。不需要先做 softmax，也不需要排序。

论文是 [*Top-nσ: Not All Logits Are You Need*](https://arxiv.org/html/2411.07641)，卖点是「两行代码」——算一次 `mean(logits)` 和 `std(logits)`，再拿阈值 mask 一遍。

**局限**：

- 当 n 偏大或分布极尖时，阈值可能高过所有 logit，**候选集为空**。所有实现都必须加一个保护：候选为空时至少保留 top-1（退化成贪心）；
- 对 outlier logit 敏感——单个特别大的 logit 会把 μ 和 σ 一起抬起来，导致阈值失准。这和量化里 outlier 破坏 scale 是同一类问题。

**和 Min-p 的区别**：Min-p 看的是「和第一名比」，Top-nσ 看的是「和全体统计量比」。前者更关注头部主导性，后者更关注整体分布形状。

## 8. 截断方法（四）：Typical、Epsilon / Eta、Tail-free、p-LESS

### 8.1 Typical Sampling / Locally Typical

换掉参照系：不看概率大小，看**信息量（surprise）**。

```text
信息量：  I_i = -log p_i
期望熵：  H   = Σ_i p_i · I_i
候选：    I_i 接近 H 的那些 token
```

它同时惩罚两端：**过分自信**（I 很小，几乎无信息的填充词）和**过分随机**（I 很大，纯噪声）。直觉是匹配「典型」的用词——生成出来的 token 信息量应当接近这个位置的平均水平。

论文是 [*Locally Typical Sampling*](https://arxiv.org/abs/2202.00666)。

**局限**：实现比 Top-k/Top-p 复杂（要算熵、要排序按 |I_i − H| 取前缀），而且在实践中相对 Top-p 的增益不稳定，工程采用度有限。

### 8.2 Epsilon Sampling 与 Eta Sampling

- **Epsilon Sampling**：最简单的形态——设一个概率地板 ε，`p_i ≥ ε` 才保留，然后重归一化。极简，但 ε 是个绝对值，遇到不同量级分布要重调。
- **Eta Sampling**：第 4 节那篇 [desmoothing 论文](https://arxiv.org/abs/2210.15191) 提出的方法。它先从「最优稀疏分布」的目标反推截断规则，得到的结论是阈值应当取 `η = min(ε, sqrt(ε))`——注意这个定义：当 ε 很小时 `sqrt(ε)` 远大于 ε，所以阈值自动变紧；ε 接近 1 时两者收敛。**它相当于给累积概率截断配了一个概率地板**，兼顾「总质量」和「单个 token 质量」。

### 8.3 Tail-free Sampling（TFS）

用**二阶导**定位「尾部从哪里开始」。把 token 按概率降序排开，看概率曲线的形状：头部陡降、尾部趋于平坦。TFS 找曲率（二阶导）变号的位置，在那里切一刀。

理论上把「尾部」定义得很清楚（PDF 的平坦区），但在实际 logits 上二阶导对噪声非常敏感，且需要排序 + 数值微分，工程上不稳定，主流框架里基本只有 SillyTavern 一类前端支持。

### 8.4 p-LESS

更近的一个方向：**不做手工阈值，改用统计检验**。在 Neyman–Pearson 框架下判定「哪些 token 与最优 token 在统计上不可区分」——不可区分就保留，能区分就砍掉。卖点是完全免超参、跨模型鲁棒。代价是每次截断都要做一遍假设检验，计算比一行阈值贵。

### 8.5 这一类方法的共同定位

| 方法 | 参照系 | 需要排序 | 超参直观度 |
|---|---|---|---|
| Top-k | 固定数量 | 是 | 中 |
| Top-p | 累积概率 | 是 | 高 |
| Min-p | 相对最大概率 | 否（只需 max） | 高 |
| Top-nσ | logits 的 μ、σ | 否 | 高 |
| Typical | 信息量 vs 熵 | 是 | 低 |
| Eta | 累积概率 + 地板 | 是 | 低 |
| TFS | 概率曲线曲率 | 是 | 低 |
| p-LESS | 统计检验 | 是 | 无超参 |

## 9. 重复抑制：Penalty、no_repeat_ngram、DRY、XTC

截断解决的是「尾部垃圾」，重复抑制解决的是另一个独立问题：**模型陷入复读**。

### 9.1 Repetition Penalty / Presence Penalty / Frequency Penalty

```text
repetition_penalty:  出现过的 token，logit 除以 θ（θ > 1）
presence_penalty:    出现过（至少一次）就减固定分 α
frequency_penalty:   按出现次数减分，logit -= β × count_i
```

三个都要注意实现细节：

- **repetition_penalty 是除法，不是减法**。logit 为负时「除以 θ」会把它推向 0（即**抬高**概率），方向是反的。所以 HuggingFace 的实现是分正负处理：正 logit 除以 θ，负 logit 乘以 θ，保证两边都被压低。这个细节出自 [CTRL 论文](https://arxiv.org/abs/1909.05858)。
- presence / frequency 是减法，方向直观，这也是 OpenAI 风格 API 只暴露后两者的原因。

**共同缺陷**：它们都按 token 惩罚。合理的高频词——中文的「的」「了」、代码里的括号和缩进——会被一并误伤。`frequency_penalty` 一大，语法就开始崩。

### 9.2 no_repeat_ngram

硬性规则：任何已经出现过的 n-gram 一律禁止。简单、确定性强，但 `n` 太小会杀掉正常搭配（`n = 3` 时连固定短语都生成不了），很大时形同虚设。适合做兜底，不适合精调。

### 9.3 DRY（Don't Repeat Yourself）

针对前三者的缺陷设计的社区方案（源自采样器社区，llama.cpp / exllamav2 / text-generation-webui 都有实现）：

1. 取已生成文本的**末尾后缀**，找它在文中之前出现过的**最长重合**；
2. 对那段重复序列的**后继 token** 施罚，罚分随重合长度**指数上升**；
3. 遇「sequence breaker」（换行、句末标点等）时重置状态。

关键差别：**它惩罚的是「重复一段序列」而不是「重复单个 token」**，所以不会误伤正常高频词。目前是开源生态里抑制复读最主流的方案。

**局限**：超参不止一个（允许的最大重合长度、罚分基数、指数底数、breaker 集合），且各框架实现细节不统一，跨平台复现困难。

### 9.4 XTC（Exclude Top Choices）

方向完全相反的一种操作：**把概率最高的那几个 token 从候选里删掉**，从剩下的里面采样。

```text
通常做法：p_i ≥ 阈值 的 token 全部排除，从剩余集合采样
```

目的不是「准确」，而是**制造意外**——模型总是倾向那几个最安全的词，删掉它们能逼出更少见的表达。有 `probability` 和 `threshold` 两个参数控制「多大程度上启用」以及「删掉前几个」。

**局限**：只在模型足够自信时才应启用，且只适合创意写作/角色扮演。任何需要准确的任务都不能开。

## 10. 自适应控制：Mirostat

前面所有截断方法都控制「候选集合」，Mirostat 换了一个**控制目标**：[直接控制输出的困惑度](https://arxiv.org/abs/2007.14966)。

要解决的问题是：Top-p 的候选数量每一步都在剧烈波动，导致输出的「惊喜程度」忽高忽低。

- **Mirostat v1**：维护一个 surprise 预算。按概率降序累加每个 token 的 `-log p`，累计到目标交叉熵就停，截断边界由此确定。相当于「给模型一个目标困惑度，让它自己决定每步留几个候选」。
- **Mirostat v2**：把 v1 改成**反馈控制器**——维护一个截断阈值 μ，实测困惑度高于目标就调低阈值（多留候选），低于目标就调高阈值，如此跟踪目标值，`μ` 就是这个被跟踪的状态量。

**优点**：输出的「意外程度」稳定可预期，不像 Top-p 那样随上下文剧烈伸缩。
**局限**：目标困惑度这个超参本身不直观（「我要 PPL = 3」不是一句人能下判断的话），且与具体模型强相关，跨模型要重新标定。所以社区普及度远不如 Top-p / Min-p。

## 11. 加速方法：Speculative Decoding（投机解码）

严格来说它不是采样策略（**它不改变输出分布**），但它在 decode 阶段和采样紧密配合，是必须知道的一环。

思路：

```text
小模型（draft，快）连续猜 k 个 token
        ↓
大模型（target，准）一次前向，并行校验这 k 个位置
        ↓
逐个 accept / reject —— 接受概率 = min(1, p_target(x) / p_draft(x))
        ↓
在被拒的位置用 target 自己的分布重新采一个，从那里继续
```

**无损性的来源**：这套「提议 + 拒绝采样」的构造，保证了最终输出的分布和「直接拿 target 模型采样」**严格相同**。也就是说它是一个纯粹的加速手段，不改生成质量。

**为什么能加速**：decode 阶段是 **memory-bound** 的（每步只为 1 个 token 读取全部权重，算力远没用满）。所以 target 一次前向校验 k 个候选位置的耗时，和校验 1 个位置的耗时**相差不大**——并行校验几乎是白赚。实际加速比取决于 draft 模型的命中率。

后续工作的改进点都在「draft 从哪来」：

- [Medusa](https://arxiv.org/html/2401.10774v2)：给 target 模型加多个预测头，一次前向直接给出多个位置的建议；
- EAGLE：在特征层做预测，而不是在 token 层重新跑一个小模型；
- Lookahead Decoding：不用 draft 模型，靠 Jacobi 迭代自举。

**和采样的耦合点**：接受判定用的就是 target 与 draft 的分布比。所以温度、Top-p、Min-p 这些参数**必须在两侧用一致的规则处理**，否则「无损」这个性质会被破坏。

## 12. 工程实现：采样阶段的开销与 kernel

这一节讲「采样到底花多少时间、kernel 怎么写」。很多人以为采样是免费的收尾步骤，实际上在大词表 + 高并发下它是实打实的开销。

### 12.1 采样在一步 decode 里的位置

```text
一个 decode step
├── 模型前向（每层 Attention + MLP）
├── LM Head：hidden [1, d] × 权重 [d, V] → logits [1, V]
└── 采样管线：惩罚 → 温度 → 截断 → softmax → 采样 → token
```

量级感：以 d = 4096、V = 152k 为例，LM Head 的权重是 6.2 亿参数，FP16 下 1.24 GB——**每生成一个 token 都要把这 1.24 GB 读一遍**，这本身就是 decode 阶段的大头之一；输出的 logits 是 152k 个浮点数，约 608 KB。

结论：**采样必须在微秒级完成**。如果采样管线拖到几百微秒，它就会成为整个 step 的瓶颈（一步 decode 往往也就几十毫秒甚至更快）。

### 12.2 各操作的复杂度与访存遍数

```text
操作          复杂度                      需要排序    访存遍数
--------------------------------------------------------------
惩罚项        O(V)                        否          1（另查 token 历史）
温度          O(V)                        否          1（可与上一步融合）
Top-k         O(V)（部分选择 O(V log k)）  是          1~2
Top-p         O(V)（全排序则 O(V log V)）  是          2
Min-p         O(V)                        否          1
Top-nσ        O(V)                        否          1~2
softmax       O(V)                        否          1
采样          O(V)                        否          1
```

两个直接推论：

1. **排序是采样管线里最贵的一步，而 Top-k / Top-p 都需要排序。** 全排序在 V = 152k 时是上百万次比较，每条序列每步都做不起。实现上通常避开全排序：用「分块局部排序 + 阈值扫描」的两遍法（第一遍估计阈值、第二遍按阈值筛选），或基数选择（radix select）做到 O(V)。
2. **Min-p / Top-nσ 只需要一次归约**（前者要 max，后者要 μ、σ），不需要排序，在 kernel 友好度上明显占优。这是它们在大词表下更容易做出高效 kernel 的结构性原因——不只是「效果更好」。

### 12.3 kernel 实现要点

**（1）尽可能融合。** 把「惩罚 → 温度 → 截断 → softmax → 采样」尽量融进一个 kernel，避免 logits 在 global memory 上被反复读写。logits 是 608 KB，每多一遍读 + 写就是 1.2 MB 的额外显存流量；decode 阶段本来就是访存受限，这些额外流量会直接变成延迟。

**（2）用 Gumbel-max 技巧免掉「前缀和 + 二分查找」。** 从 `softmax(z)` 中采样，等价于：

$$
\arg\max_i \left( z_i + g_i \right), \qquad g_i \sim \text{Gumbel}(0, 1)
$$

等价形式（指数竞赛，exponential race）：

$$
\arg\min_i \left( q_i / p_i \right), \qquad q_i \sim \text{Exp}(1)
$$

好处是：只需要一次 elementwise 除法 + 一次归约式 argmax，**没有前缀和、没有二分查找、没有任何串行依赖**，完全并行。vLLM 的 `random_sample` 就是这一类写法。

**（3）批量采样要处理「per-row 参数」。** 服务端一个 batch 里每条序列的采样参数都不同（温度、min_p、惩罚各自独立），kernel 需要按行读取参数。更麻烦的是**路径分歧**：同一 batch 里只要有一条序列开了 Top-p，整个 batch 都得走带排序的那条路径——这是批量推理里「坏苹果」效应的来源之一，也是 Min-p 这类「无排序」方法在服务端的实际优势。

**（4）已有融合实现可参考。** FlashInfer 一类库提供把 Top-p / Top-k / Min-p + softmax + 采样融在一起的专用 kernel；SGLang、TensorRT-LLM 也各自有采样 kernel 的实现。

### 12.4 训练侧的对照

训练时 logits 是 `[B, S, V]`，比推理时大得多：B = 8、S = 2048、V = 152k、FP32 下约 **10 GB**。所以训练里普遍用**融合的 cross-entropy**（分块计算，不把完整 logits 落到显存，如 Liger Kernel 的 fused linear CE）。

动机和推理侧采样完全一样：**logits 太大，不要反复搬**。这两处是同一个工程思想在两个场景下的体现。

### 12.5 关于确定性

`T = 0` 也**不保证逐位可复现**，原因不在采样，而在前向：

- batch 组成变化会改变归约顺序、影响 kernel 选择（split-K 的切分方式随之变化）；
- MoE 的路由、原子加操作的累加顺序都会引入浮点差异。

要复现需要固定 batch 组成、固定 kernel、固定随机种子。**采样本身的随机性**则来自 RNG：服务端要为每条序列维护独立的随机数流（vLLM 用 per-request generator），否则并发请求会互相干扰，同一条序列在两次运行里拿到的噪声不同。

### 12.6 投机解码的额外开销

校验阶段要为每个候选位置算一次「接受/拒绝」：算 `min(1, p_target / p_draft)`，再做一次采样。这是一次 elementwise 比较加一次采样，代价很小。真正的额外成本在于：

- 要同时维护 draft 与 target 两套 logits；
- 被拒时要**回滚 KV cache**（把多写进去的 k 个位置丢弃），这部分在实现上比采样本身更容易出错。

## 13. 参数配置建议

| 场景 | 推荐配置 |
|---|---|
| 事实问答 / 信息抽取 / 分类 | `T = 0`（贪心）或 `T = 0.1` |
| 代码生成 | `T = 0 ~ 0.2`，`top_p = 0.95` |
| 通用对话 | `T = 0.7`，`top_p = 0.9` |
| 创意写作 / 角色扮演 | `T = 0.8 ~ 1.0`，`min_p = 0.05 ~ 0.1`，可选 DRY 或 XTC |
| 推理模型（CoT / thinking） | `T = 0.6 ~ 0.7`，`top_p = 0.95`，**不要用贪心** |

推理模型那一行要单独强调：**温度不能设 0**。CoT 需要在多条思路之间探索，贪心会让模型卡死在一条（往往错误的）路径上，甚至进入重复循环。主流推理模型的官方推荐都落在 `T = 0.6`、`top_p = 0.95` 这个区间。

三条调参纪律：

1. **截断只用一种。** `top_k` + `top_p` + `min_p` 叠在一起调，等于三个阈值互相打架，效果无法归因，也没法迁移到别的模型。要换就整个换掉。
2. **温度和截断分工明确。** 想加创意，优先「抬温度 + Min-p 兜底」；用「抬温度 + Top-p」会把尾巴一起放进来。
3. **惩罚类能不开就不开。** 优先用 DRY；非要用 frequency / presence penalty，值控制在 0.1 ~ 0.5，再大就开始破坏语法。

## 14. 方法对比总表

| 方法 | 阈值形式 | 自适应 | 需要排序 | 主要局限 |
|---|---|---|---|---|
| Greedy | — | — | 否 | 无多样性、易复读 |
| Beam Search | 束宽 k | — | 是 | 冗长、k 倍开销、不适合开放式 |
| Temperature | 无（改分布形状） | — | 否 | 单独用不能防长尾 |
| Top-k | 固定数量 k | 否 | 是 | 不随分布形状调整 |
| Top-p | 固定累积概率 p | 部分 | 是 | 只控总质量，不控单 token 质量 |
| **Min-p** | `p_max × min_p` | 是 | 否 | 极散分布时阈值过低 |
| Top-nσ | `μ + n·σ`（logits） | 是 | 否 | 对 outlier logit 敏感、可能空集 |
| Typical | 信息量 ≈ 熵 | 是 | 是 | 实现复杂、增益不稳 |
| Eta | 累积概率 + 地板 | 部分 | 是 | 超参不直观 |
| Tail-free | 概率曲线曲率 | 是 | 是 | 数值不稳定 |
| p-LESS | 统计检验 | 是 | 是 | 检验本身有计算开销 |
| DRY | 后缀重合长度（指数） | 是 | 否 | 超参多、跨框架不统一 |
| XTC | 排除头部 | — | 否 | 只为「意外」，不适合准确任务 |
| Mirostat | 目标困惑度 μ | 是 | 是 | μ 不直观、与模型强相关 |
| Speculative | — | — | 否 | 不改分布，只加速 |

## 15. 总结

- 模型只给 logits，**从 logits 到 token 的全部规则才是采样策略**。温度和截断各管一件事：温度改形状，截断改候选集合。
- 温度只改差距、**不改排序**，所以它单独使用无法阻止低概率 token 被抽到——要结构性地排除长尾，只能靠截断。
- 截断是有偏的（被砍掉的概率会重归一化给头部），它之所以值得，是因为**模型分布本身在尾部偏厚**（desmoothing），忠实采样反而不是最优解。
- 截断方法的演化主线：**从固定的绝对阈值（Top-k）→ 自适应阈值（Top-p、Min-p、Top-nσ）**，从「控制候选集大小」到「控制输出质量」（Mirostat）。Min-p 之所以成为近期主流，是因为它第一次把「模型置信度」显式地放进了阈值里，并且与温度解耦。
- 工程上，采样不是免费的：**排序是管线里最贵的操作**，Min-p / Top-nσ 的价值有一半来自「不需要排序」；融合、Gumbel-max 技巧、per-row 参数的批量处理，是把采样压进微秒级的关键手段。
- 最后的选型原则：**先用一种截断把候选集定对，再用温度调风格，惩罚项只在确实复读时才介入。**

## 全称速查

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| logits | — | 模型输出的未归一化实数分数向量，维度等于词表大小 |
| LLM | Large Language Model | 大语言模型 |
| Sampling | Sampling / Decoding | 采样 / 解码：从 logits 选出 token 的规则 |
| Greedy | Greedy Decoding | 贪心解码，每步取 argmax |
| Beam Search | Beam Search | 束搜索，同时保留 k 条候选序列 |
| beam width | — | 束宽 k，束搜索保留的序列数 |
| Temperature | Temperature Scaling | 温度缩放，softmax 前对 logits 除以 T |
| Top-k | Top-k Sampling | 保留概率最高的 k 个 token |
| Top-p | Nucleus Sampling | 核采样，保留累积概率达到 p 的最小集合 |
| Min-p | Min-p Sampling | 保留概率不低于 `min_p × p_max` 的 token |
| Top-nσ | Top-n-sigma Sampling | 保留 logits 高于 `μ + n·σ` 的 token |
| Typical | Locally Typical Sampling | 保留信息量接近期望熵的 token |
| Epsilon Sampling | — | 概率地板截断，`p_i ≥ ε` 才保留 |
| Eta Sampling | η-sampling | 从最优稀疏分布反推的截断，阈值 `η = min(ε, √ε)` |
| TFS | Tail-free Sampling | 用二阶导（曲率）定位尾部起点的截断 |
| p-LESS | — | 用统计检验决定截断，无超参 |
| Desmoothing | — | 神经语言模型分布尾部偏厚的现象，截断的理论依据 |
| Repetition Penalty | — | 对已出现 token 的 logit 做除法惩罚 |
| Presence Penalty | — | 对已出现 token 做一次性减分 |
| Frequency Penalty | — | 按出现次数累加减分 |
| no_repeat_ngram | — | 硬性禁止重复出现过的 n-gram |
| DRY | Don't Repeat Yourself | 按后缀重合长度指数惩罚的重复抑制方法 |
| XTC | Exclude Top Choices | 排除头部高概率 token，制造意外 |
| Mirostat | — | 以目标困惑度为控制目标的自适应截断 |
| PPL | Perplexity | 困惑度 |
| Speculative Decoding | — | 投机解码：draft 提议 + target 并行校验 |
| LM Head | Language Model Head | 把最后一层隐状态映射到词表 logits 的线性层 |
| GEMM | General Matrix Multiply | 通用矩阵乘 |
| KV cache | Key-Value Cache | 缓存的注意力 K/V，decode 阶段复用 |
| Gumbel-max | Gumbel-max Trick | `argmax(z + Gumbel噪声)` 等价于从 softmax(z) 采样 |
| Exponential Race | — | 指数竞赛，Gumbel-max 的等价形式 `argmin(q_i / p_i)` |
| RNG | Random Number Generator | 随机数发生器 |
| split-K | split-K GEMM | 把规约维切开并行，导致归约顺序依赖 batch 组成 |
| MoE | Mixture of Experts | 混合专家，路由引入浮点不确定性 |
| FlashInfer | — | 提供融合采样 kernel 的推理库 |

## 参考文献

1. Fan et al. *Hierarchical Neural Story Generation.* arXiv:1805.04833. （Top-k 采样）
2. Holtzman et al. *The Curious Case of Neural Text Degeneration.* arXiv:1904.09751. （Nucleus / Top-p）
3. Meister et al. *Locally Typical Sampling.* arXiv:2202.00666.
4. Hewitt et al. *Truncation Sampling as Language Model Desmoothing.* arXiv:2210.15191. （Eta Sampling / desmoothing 依据）
5. *Min-p Sampling for Creative and Coherent LLM Outputs.* arXiv:2407.01082.
6. *Top-nσ: Not All Logits Are You Need.* arXiv:2411.07641.
7. Basu et al. *Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity.* arXiv:2007.14966.
8. Keskar et al. *CTRL: A Conditional Transformer Language Model for Controllable Generation.* arXiv:1909.05858. （Repetition Penalty）
9. Li et al. *Contrastive Decoding: Open-ended Text Generation as Optimization.* arXiv:2210.15097.
10. Leviathan et al. *Fast Inference from Transformers via Speculative Decoding.* arXiv:2211.17192.
11. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318.
12. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* arXiv:2401.10774.
13. *Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation.* arXiv:2408.13586.
14. von Platen. *How to generate text: using different decoding methods for language generation with Transformers.* Hugging Face Blog. https://huggingface.co/blog/how-to-generate
