---
title: LLM Quantization（大模型量化）：GPTQ、AWQ、SmoothQuant 的核心思路
description: 先用最直白的例子讲清三个方法在干什么——GPTQ 量化一个、补偿其他；AWQ 保护重要权重；SmoothQuant 平滑 Activation——再逐步加深：为什么要 Hessian、为什么叫 Activation-aware、那个数学变换为什么是无损的，最后落到 kernel 级工程与三者对比。
pubDate: 2026-09-27T01:30:00+08:00
tags: [大模型, 训练工程]
---

## 1. 先记住三句话

$$\boxed{\text{GPTQ = 误差补偿}}$$

$$\boxed{\text{AWQ = 保护重要权重}}$$

$$\boxed{\text{SmoothQuant = 平滑 Activation}}$$

展开一点：

- **GPTQ**（GPT Quantization）：量化一个权重，顺手调整其他**还没量化**的权重，把误差补回来。
- **AWQ**（Activation-aware Weight Quantization）：先找出哪些权重重要，重点保护它们。
- **SmoothQuant**（平滑量化）：把 Activation 里的 outlier「搬」一部分到 Weight 上，让 Activation 变好量化。

三个方法都在跟「量化误差」较劲，但站位不同：前两个**只管权重**（Weight-only，常见 W4A16），第三个要**连 Activation 一起量化**（W8A8）。

下面一个个来，每个都从最简单的例子讲起。

## 2. 先补一课：量化误差从哪来

最简单粗暴的权重量化长这样：

```text
FP16 Weight
     ↓
   round
     ↓
   INT4
```

每个权重都会带上一点误差：

$$ \Delta W = W_q - W $$

举个例子：

```text
W1 = 1.23  →  W1_q = 1.0        误差 ΔW1 = −0.23
```

误差就这么产生了。接下来是两个不同的反应：

- 反应一：**误差能不能补回来？** → GPTQ
- 反应二：**能不能别让重要的权重吃这么大误差？** → AWQ

还有一个更根本的问题：我们到底该怕什么误差？

**真正要紧的不是权重误差本身，而是输出误差。**

$$ Y = XW \qquad\Longrightarrow\qquad Y_q = X W_q = Y + X\,\Delta W $$

于是有了一句关键的话：

> 同一个权重误差，乘上越大的 Activation，伤害越大。

这句话是后面两个方法的分水岭：

- 误差被放大的地方（Activation 大的通道）→ 就是 AWQ 要保护的地方；
- 想让整体输出误差最小 → 就得看校准集 Activation 的结构，这就是 GPTQ 用 Hessian 的原因。

（细节先按下，各自那一节再回来。）

## 3. GPTQ：量化一个，补偿其他

### 3.1 思路

GPTQ 主要用于 Weight-only Quantization（仅权重量化），尤其是 INT4。

它的出发点就是一句话：

> 如果我把某个权重量化产生了误差，能不能调整其他**还没量化**的权重，把这个误差补回来？

也就是说：**允许单个权重产生误差，但通过调整其他权重，让整体输出误差尽量小。**

### 3.2 举个例子

原始权重：

```text
W1 = 1.23
W2 = 2.35
W3 = 0.91
```

先把 W1 量化：

```text
W1 = 1.23 → 1.0        误差 ΔW1 = −0.23
```

GPTQ 不会就这么算了——它会利用权重之间的相关性（更准确地说，是校准数据里 Activation 之间的相关性），去调整还没量化的 W2、W3：

```text
W1 → 1.0
W2 → 2.42     ← 补偿
W3 → 0.86     ← 补偿
```

目标是让最终的 $X W_q$ 尽可能接近 $X W$。

（真实补偿多少是由 Hessian 算出来的，这三个数的例子只是为了看清「补偿」这个动作。）

### 3.3 为什么需要 Hessian？

注意 GPTQ 的优化目标不是

$$ \min |W - W_q| $$

（权重误差最小），而是

$$ \min \|XW - XW_q\|^2 $$

（**输出误差**最小）。这两件事不一样——权重误差大一点，只要不伤输出，就无所谓。

那怎么知道「动一个权重，输出会跟着动多少」？靠二阶信息：

$$ H = 2XX^{\mathsf T} $$

$H$ 就是校准集 Activation 的统计结构。直觉上一句话：

> **Hessian 描述的是「校准数据里，各输入通道的幅度、以及它们之间的相关性」**——激活大、又互相纠缠的通道，动它们的权重代价最大。

### 3.4 再往深看：补偿到底怎么算

（这节是公式，跳过不影响理解。）

逐列量化，量到第 q 列时产生误差 $e_q = w_q - \mathrm{quant}(w_q)$，再把它摊给还没量化的列 F：

$$ \delta_F = -\frac{e_q}{[H^{-1}]_{qq}}\, H^{-1}_{F,q} $$

三块各自的含义：

- $e_q$：这一列量化产生的误差有多大；
- $1/[H^{-1}]_{qq}$：这一列有多敏感；
- $H^{-1}_{F,q}$：误差应该往哪些方向、按什么比例摊。

**从 OBQ 到 GPTQ 的三个工程招**（为什么原方法跑不动 175B）：

1. **Hessian 全层共用**：同一层所有输出行共用同一个 $X$ 算出的 $H$，整个矩阵一次跑完；
2. **预 Cholesky 分解**：$H^{-1} = LL^{\mathsf T}$ 提前分解好，量化过程中原地更新，$O(d^3)$ 的求逆只做一次（对角线再加一点阻尼保证可逆）；
3. **惰性批量更新**：128 列攒成一批，批内的补偿合并成一次矩阵乘再应用——这是「快」的关键。

账：175B 模型 4-bit 量化约 **4 GPU 小时**（校准集 128 条序列）。

### 3.5 工程落地

- 运行时：**group 128** 的 4-bit 权重 + 融合反量化 kernel（Marlin）——权重只读一遍、解包直接产出 Tensor Core 的 fragment、scale 离线 repack，细节见[量化的计算开销](/blog/2026-09-27-quantization-compute-cost-notes/)那篇的 §6。
- 可选项 **act-order / desc_act**：按 Activation 重要性重排列顺序再量化，通常更准一点，但推理 kernel 要支持这个顺序（复现时得配对）。

### 3.6 局限

- **它只治权重，不治 Activation**（Activation 仍是 16-bit）；
- 依赖校准集，换了领域的数据会掉点；
- 量化过程本身不便宜（离线一次性）。

## 4. AWQ：重要权重不要乱量化

### 4.1 问题

AWQ（Activation-aware Weight Quantization，激活感知权重量化）问的是另一个问题：

> 量化的时候，哪些权重特别重要？

### 4.2 举个例子

不是所有权重都一样重要：

```text
Weight:   W1    W2    W3    W4    W5
           ↓     ↓     ↓     ↓     ↓
         普通   普通   重要   普通   普通
```

如果全部无差别地压成 INT4：

```text
W1 → INT4
W2 → INT4
W3 → INT4   ← 误差可能很严重
W4 → INT4
W5 → INT4
```

W3 一崩，模型能力可能明显下降。

AWQ 的思路是：**找出对 Activation 重要的权重，保护它们。**

```text
普通 Weight → INT4（照常）
重要 Weight → 尽量少留量化误差
```

### 4.3 为什么叫 Activation-aware？

因为它不只盯着 $W$，而是盯着 $XW$ 里的 **Activation $X$**。

回到 §2 那句结论：

> 同一个权重误差，乘上越大的 Activation，伤害越大。

所以「哪些权重重要」要看**与它对应的输入通道的 Activation 大不大**——这就是 activation-aware。

AWQ 论文里还有两个反直觉的发现：

- **权重幅度是个坏指标**：把「幅度最大的 1% 权重」留成 FP16，效果竟然不如随机留 1%；
- **真正显著（salient）的通道很少**：按 Activation 幅度看，只有约 **1%** 的输入通道是关键。

### 4.4 怎么做：不是留 FP16，而是「缩放」

一个自然想法是：把重要权重直接存成 FP16 不就行了？

AWQ 没这么做。原因很工程：**混合精度会让 GEMM 掉进慢路径**（硬件里 FP16 和 INT4 混算要走低效通道），省下来的精度还不够赔速度。

AWQ 用的是 scaling（缩放）——给输入通道乘一个因子 $s$：

$$ W' = W \cdot \mathrm{diag}(s), \qquad x' = \mathrm{diag}(s)^{-1} x $$

数学上完全等价（$W'x' = Wx$），但量化难度变了：

```text
s 越大 → 该通道权重被放大 → 4-bit 下的相对误差 Δw/w 越小 → 保住了
显著通道（Activation 大）：给大 s，重点保护
普通通道：给小 s，让位
Activation 本来就跑在 FP16，除以 s 几乎无损 —— 拿 Activation 的动态范围，换权重的相对精度
```

再往深看：$s$ 不是一个一个猜的，而是按通道算 + 搜一个指数 $\alpha$：

$$ s_j = \frac{(\mathrm{mean}\,|X_j|)^{\alpha}}{(\mathrm{mean}\,|W_j|)^{1-\alpha}} $$

对 $\alpha \in [0,1]$ 网格搜索，目标是「量化再反量化之后，输出误差最小」。

### 4.5 工程落地与局限

- $s$ 乘进权重（离线完成），$1/s$ 折进**前一层**（LayerNorm/RMSNorm 的缩放参数，或前一个 Linear 的输出通道）——**推理时零额外开销**。
- 量化成本比 GPTQ 轻得多：只要**前向**统计 Activation 幅度 + 一次 $\alpha$ 搜索，不需要 Hessian 和逐列回归。
- 运行时走 W4A16 融合反量化 kernel（AWQ GEMM / vLLM 的 awq kernel 等）。
- 局限：它只做「保护」，不做「补偿」——没有 GPTQ 那种误差修正机制；$s$ 是 per-channel 粒度，比 group-wise 粗；同样依赖校准集。

## 5. SmoothQuant：处理 Activation Outlier

### 5.1 问题：Activation 很难量化

前两个方法都只动权重。SmoothQuant 面对的是另一个对手：**Activation 难量化**。

看一组 Activation：

```text
0.1   0.2   0.3   0.5   20
                          ↑ 这就是 outlier
```

如果直接 INT8：

```text
scale 被 20 拉大 → 0.1 / 0.2 / 0.3 / 0.5 这些正常值连一格都占不到，全被舍成 0
```

更麻烦的是：Activation 跟着输入走，每个 token 都可能不一样，不像权重可以离线慢慢分析。

### 5.2 核心思想：一个无损的数学变换

既然 Activation 难搞，那就**把难度分一部分给 Weight**。做法是一个数学变换：

$$ Y = \bigl(X \cdot \mathrm{diag}(s)^{-1}\bigr)\,\bigl(\mathrm{diag}(s) \cdot W\bigr) = XW $$

也就是：**Activation 除以 $s$（压小 outlier），Weight 乘以 $s$（量级变大）**，两边一乘，模型数学结果完全不变。

```text
原来：  Activation ████████████████  ← outlier 严重
        Weight     ████

之后：  Activation ████████          ← 更平滑
        Weight     ████████          ← 量级增加
```

### 5.3 为什么这样有用？

因为：**Activation 比 Weight 更难量化。**

```text
Weight：    模型固定，可以离线分析、提前校准
Activation：跟输入有关，每个 token 都可能不同，动态变化
```

所以 SmoothQuant 的思路是：**牺牲一点 Weight 的量化难度，换 Activation 容易量化。**

搬多少由 $\alpha$ 控制：

$$ s_j = \frac{\max |X_j|^{\alpha}}{\max |W_j|^{1-\alpha}} $$

- $\alpha = 1$：难度全留在 Activation（不搬）；
- $\alpha = 0$：难度全推给 Weight；
- $\alpha = 0.5$：**对半劈**，最常用的默认值。

### 5.4 换来了什么：W8A8 全对称

Activation 被压平之后，链式反应是：

$$ \text{Activation 可 per-tensor 静态量化} \Rightarrow \text{W8A8 且全对称} \Rightarrow \text{原生 INT8 GEMM，零修正项} $$

这正是[量化的计算开销](/blog/2026-09-27-quantization-compute-cost-notes/)那篇算过的最省的账：GEMM 之外什么都不用做，还吃到 INT8 TensorCore 约 2 倍于 FP16 的吞吐。

### 5.5 工程落地与局限

- 折法同 AWQ：$s$ 乘进权重、$1/s$ 折进前一层（LayerNorm 的 $\gamma$ 或前一个 Linear），**运行时零额外开销**。
- 离线成本：跑校准集统计 $\max|X_j|$ 与 $\max|W_j|$，再调 $\alpha$（论文给 0.5 通用，per-layer 搜更好）。
- 局限：它解决的只是 **8-bit 这一档**；4-bit 的 Activation 是另一类难题（格点只剩 15 个，迁移不够用）。另外本身没有 outlier 的场景，收益有限。

## 6. 三者区别（面试爱问）

**GPTQ 的重点**：量化产生误差之后，怎么补偿？

```text
Weight
  ↓
量化
  ↓
产生误差
  ↓
用其他 Weight 补偿
  ↓
降低整体输出误差
```

**AWQ 的重点**：哪些 Weight 最重要？优先保护谁？

```text
Activation
  ↓
分析重要性
  ↓
找到敏感 Weight
  ↓
重点保护
  ↓
其余 Weight 量化
```

**SmoothQuant 的重点**：Activation 难量化，怎么办？

```text
Activation 有 outlier
  ↓
把量级搬到 Weight（数学无损）
  ↓
Activation 变平滑
  ↓
W8A8 全对称，吃到 INT8 GEMM
```

画成一棵树：

```text
                LLM Quantization
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        GPTQ          AWQ      SmoothQuant
          │            │            │
      误差补偿      重要性保护     平滑 Activation
          │            │            │
    Weight 量化    Weight 量化    Activation 量化
     （W4A16）      （W4A16）       （W8A8）
```

$$\boxed{\text{GPTQ: Error Compensation}}$$

$$\boxed{\text{AWQ: Weight Importance}}$$

$$\boxed{\text{SmoothQuant: Smooth Activation}}$$

一张表看得更快：

| 方法 | 量化什么 | 核心思路 | 精度从哪来 | 离线成本 | 运行时 |
|---|---|---|---|---|---|
| GPTQ | 权重 4-bit（W4A16） | 逐列量化 + 补偿误差 | 把误差摊给未量化的列 | 最重：校准集 + Hessian + Cholesky（175B 约 4 GPU 小时） | 融合反量化 kernel（Marlin） |
| AWQ | 权重 4-bit（W4A16） | 按 Activation 幅度缩放，保护显著通道 | 让重要通道的相对误差变小 | 轻：前向统计 + α 搜索 | 融合反量化 kernel（scale 已折进权重） |
| SmoothQuant | 权重 + Activation 8-bit（W8A8） | 把 Activation outlier 搬进权重 | 让 Activation 可 per-tensor 量化 | 轻：校准统计 + 调 α | 原生 INT8 GEMM（零额外开销） |

一句话定位：

- GPTQ 和 AWQ 是**同一条战线**（W4A16）的两种打法：一个用数学修正误差，一个用观察避开误差；
- SmoothQuant 站在**另一条战线**（W8A8）：它去打通那条本可以更快、更省的路。

## 7. 边界与组合

- **可以叠加**：权重用 GPTQ、Activation 侧用平滑/裁剪，就是 W4A8 一类的做法；AWQ 本身也用 group。
- **都依赖校准集**：三者都是 PTQ（Post-Training Quantization，训练后量化）——分布漂移（比如换了领域的数据）会掉点，这也是 QAT（量化感知训练）存在的原因。
- **不同战线的邻居**：QLoRA / NF4 面向**训练**（省显存，不图推理速度）；FP8 / MXFP4 是**硬件原生**路线（Hopper / Blackwell），跟这三个「软件方法」互补而非替代。
- **怎么选**：显存优先、能接受离线成本 → GPTQ；量化要快、要跨领域稳 → AWQ；想同时压 Activation、要 INT8 GEMM 的吞吐 → SmoothQuant。

## 8. 总结

- 量化误差 = $X\,\Delta W$，被 Activation 幅度加权。GPTQ 从这里走向「用 Hessian 补偿」，AWQ 从这里走向「按 Activation 保护」。
- GPTQ 的三个工程招（共用 Hessian、预 Cholesky、批量更新）把 OBQ 的昂贵逐列更新，变成了可用的 175B 方案。
- AWQ 用一次等价缩放，把精度花在 1% 的显著通道上；SmoothQuant 用同样的缩放，把难度从 Activation 搬给 Weight。
- $$\boxed{\text{GPTQ 补偿误差 · AWQ 保护重点 · SmoothQuant 搬移难点}}$$
- 三者都是「用离线的一点聪明，换运行时的一点精度」。

## 全称速查

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| PTQ | Post-Training Quantization | 训练后量化，不训练直接量 |
| QAT | Quantization-Aware Training | 量化感知训练，训练时就模拟量化 |
| W4A16 | Weight 4-bit / Activation 16-bit | 权重 4 位、激活 16 位的常见组合 |
| W8A8 | Weight 8-bit / Activation 8-bit | 权重、激活都 8 位，可走原生 INT8 GEMM |
| GPTQ | GPT Quantization | 逐列量化 + Hessian 误差补偿的 4-bit 权重量化方法 |
| AWQ | Activation-aware Weight Quantization | 激活感知的权重量化：按激活幅度缩放保护显著通道 |
| SmoothQuant | — | 平滑 + 量化：把激活 outlier 迁移进权重，打通 W8A8 |
| Weight-only | Weight-only Quantization | 只量化权重、激活保持 16 位的方案 |
| OBQ | Optimal Brain Quantization | GPTQ 的前身，逐列量化 + 二阶补偿的原型 |
| Hessian | — | 二阶导数矩阵，这里指 $2XX^{\mathsf{T}}$，描述校准集激活的相关结构 |
| Cholesky | Cholesky Decomposition | 矩阵分解，用于避免量化过程中每步求逆 |
| act-order / desc_act | activation order | 按激活重要性重排列顺序后再量化 |
| salient channel | — | 显著通道：激活幅度特别大的输入通道（约 1%） |
| outlier | — | 离群值，幅度远超同侪的元素 |
| PPL | Perplexity | 困惑度，量化精度的常用指标 |
| Marlin | — | GPTQ 4-bit 推理的高性能融合 kernel |

## 参考文献

1. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
2. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
3. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438.
4. Frantar et al. *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning.* arXiv:2208.11580.
5. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339.
6. Frantar et al. *Marlin: Nearly Ideal Inference Speed for 4-bit Large Language Models.* arXiv:2408.11743.
7. Yao et al. *ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers.* arXiv:2206.01861.
8. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
