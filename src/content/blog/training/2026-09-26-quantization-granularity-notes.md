---
title: 量化粒度：一个 scale 管多大一片地方
description: 从「量化到底在干什么」讲起——用一行 5 个权重手把手走通 q = round(x/s)，再逐个拆开三种粒度：per-tensor 一把尺子量全张量、per-channel 每个输出通道一把、group-wise 每 128 个元素一把。为什么尺子切得越细越准、per-channel 为什么在硬件上几乎免费、4-bit 为什么必须切到组。
pubDate: 2026-09-26T23:30:00+08:00
tags: [大模型, 训练工程]
---

## 1. 先理解量化到底在干什么

假设原始 FP16 权重：

$$
W = [-1.2,\ -0.5,\ 0.1,\ 0.8,\ 1.5]
$$

我们想把它变成 INT8。

**Quantization（量化）**，就是：

$$
\mathrm{FP16/FP32} \ \rightarrow\ \mathrm{INT8/INT4}
$$

核心问题只有一个：

> 怎么把连续的浮点数，映射成有限的整数？

最简单的做法是**对称量化**：

$$
q = \mathrm{round}(x / s)
$$

反量化：

$$
\hat{x} = q \times s
$$

其中：

- $x$：原始浮点数
- $q$：量化后的整数
- $s$：Scale（缩放因子）——就是「刻度宽度」
- $\hat{x}$：反量化回来的近似值（有误差，误差从哪来看下面）

**手把手走一遍。** 先定 $s$：int8 对称量化只有 $[-127, 127]$ 共 255 个格，$s$ 取「最大绝对值除以 127」：

$$
s = \frac{\max|x|}{127} = \frac{1.5}{127} \approx 0.0118
$$

然后逐个算 $q = \mathrm{round}(x/s)$，再反量化回去对账：

| 原始 $x$ | −1.2 | −0.5 | 0.1 | 0.8 | 1.5 |
|---|---|---|---|---|---|
| 量化 $q$ | −102 | −42 | 8 | 68 | 127 |
| 反量化 $\hat{x}$ | −1.20 | −0.50 | 0.09 | 0.80 | 1.50 |

每个数最多差半格（$s/2 \approx 0.006$）——**误差的大小由 $s$ 决定：$s$ 越大，格越粗。** 反量化不是「把量化撤销」，只是用格点代表值近似原值，信息在 round 那一步就已经丢了。

所以：

> **量化的关键就是：Scale 怎么分配。** 用谁的 max 来决定 $s$？多少元素共用同一个 $s$？

这就产生了 **Per-Tensor、Per-Channel、Group-wise** 三种粒度（granularity）。

（顺带一句：更完整的量化写法是 $x \approx s(q - z)$，多一个零点 $z$ 用来精确表示 0——叫仿射量化；这里 $z = 0$ 就是对称量化。到「边界与延伸」再展开。）

## 2. Per-Tensor：整个 Tensor 一个 Scale

**Per-Tensor Quantization（逐张量量化）**：整个矩阵只算一个 $s$，所有元素共用。

假设 $W \in \mathbb{R}^{4\times4}$：

```text
整个矩阵 → 一个 scale

┌────┬────┬────┬────┐
│    │    │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
```

即 $W \to s$，所有元素共享 $s$。

**优点**：

- 非常简单：Scale 数量最少（就 1 个）
- 存储开销最小
- Kernel 简单，硬件实现容易

**缺点**：如果不同区域的数值范围差别很大，就很吃亏。

比如：

- 区域 A：$[-0.1,\ 0.1]$
- 区域 B：$[-10,\ 10]$

整个 Tensor 用一个 scale。为了覆盖 $[-10, 10]$，$s$ 被撑大，区域 A 的精度就会很差——A 里的数连一格都占不满，全被舍成 0。

白话：**一把尺子要同时量得了蚂蚁和大象，就只能按大象做刻度，蚂蚁量出来全是 0。** LLM 权重里恰恰就有这种「大象」——outlier（离群值），极少数通道比别的通道大几十上百倍。所以 per-tensor 用在 LLM 权重上经常翻车。

## 3. Per-Channel：每个 Channel 一个 Scale

**Per-Channel Quantization（逐通道量化）**：不让整个矩阵共享 scale，而是每个通道一把尺子。

比如一个 Linear 权重：

$$
W \in \mathbb{R}^{out \times in}
$$

通常按 **Output Channel（输出通道）** 切——每一行一个 scale：

```text
             Input
        c0  c1  c2  c3
         ↓   ↓   ↓   ↓
Output 0 ──────────── → s0
Output 1 ──────────── → s1
Output 2 ──────────── → s2
Output 3 ──────────── → s3
```

于是量化公式变成：

$$
q_{ij} = \mathrm{round}(W_{ij} / s_i)
$$

每一行使用自己的 $s_i$——第 $i$ 行的 max 只决定第 $i$ 行的刻度，别的行管不着。

## 4. 为什么 Per-Channel 通常更准

假设三个通道的动态范围：

- Channel 0：$[-0.1,\ 0.1]$
- Channel 1：$[-5,\ 5]$
- Channel 2：$[-20,\ 20]$

**Per-Tensor**：三个通道共用一把尺子，$s$ 由最大的 $[-20, 20]$ 决定。Channel 0 的动态范围 $[-0.1, 0.1]$ 相对 $[-20, 20]$ 小 200 倍——刻度是按 20 做的，0.1 的量级连一格都不到，量化精度损失殆尽。

**Per-Channel**：三个通道各用自己的 $s$，各按各的动态范围做刻度：

- Channel 0 → $s_0$（量程只到 0.1，刻度极细）
- Channel 1 → $s_1$
- Channel 2 → $s_2$

每个通道的信息都保住了。所以 **Per-Channel 通常比 Per-Tensor 精度更好**。

代价就是：**Scale 更多**。但算一笔账：out 个 FP16 scale 只占 $2 \times out$ 字节，相对整个 $[out \times in]$ 的权重矩阵（比如 4096×4096）只有 **0.1%**——几乎免费。

### 更进一步：为什么偏偏是「输出通道」这个方向？

两个原因，缺一不可：

1. **数学上通道天然独立**：$y_i = \sum_j W_{ij} x_j$——输出第 $i$ 个元素只用到第 $i$ 行权重。第 3 行有 outlier，只坑 $y$ 的第 3 个元素，别的通道不该为它买单。所以「每行一把尺子」是误差隔离的天然边界。
2. **硬件上几乎免费**：把量化权重代回矩阵乘：

$$
y_i = \sum_j X_{ij}\, s_i\, W^q_{ij} = s_i \sum_j X_{ij} W^q_{ij}
$$

$s_i$ 只依赖输出下标 $i$，**可以提到求和号外面**——整数 GEMM 照常算（int8 乘、int32 累加），算完再给每个输出行乘上自己的 $s_i$ 就行。反过来，如果按**输入**方向（按列）切，$s$ 的下标就留在求和号里，提不出来，整数 GEMM 没法直接做。

**判断口诀：scale 的下标只要不出现在求和号（规约维度）里，就是免费的。** 这也是所有标准实现（TensorRT、cuBLAS int8、ONNX 的 per-axis）都沿输出通道的原因。

## 5. Group-wise：组大小可调的那一档

**Group-wise Quantization（分组量化）**：每 $g$ 个元素共用一把尺子，$g$ 叫 group size（组大小）。

先给一个统一视角——**三种粒度其实是同一件事：把矩阵按「组」切，差别只是组多大**：

```text
组 = 整个矩阵   → per-tensor    1 把尺子
组 = 一行       → per-channel   out 把尺子
组 = g 个元素   → group-wise    out·in/g 把尺子
```

组大小连续可调。比如 16 个权重：

- **group size = 16**（整个张量）→ 1 个 scale（就是 per-tensor）
- **group size = 8** → 2 个 scale
- **group size = 4** → 4 个 scale

```text
16 个权重，group size = 4：xxxx xxxx xxxx xxxx → s0 s1 s2 s3（4 个 scale）
16 个权重，group size = 8：xxxxxxxx xxxxxxxx     → s0 s1（2 个 scale）
```

组越小，尺子越多、每把尺子看到的值域越窄、格点越密。注意：组也可以切得比一行还粗（几个通道一组，当 per-tensor 和 per-channel 之间的中间档用）；但 LLM 里说的 **GroupSize = 128 是比一行更细的切法**（一行通常有几千个元素）：

```text
128 个权重 → 共享一个 scale；下一个 128 个 → 共享另一个 scale
```

**为什么 4-bit 时代需要它？** int8 有 127 格，格点密，per-channel 一把尺子量一整行也够用。但 **int4 只有 15 格，比 int8 粗约 8.5 倍**——同样的行内不均匀，8-bit 吃点亏不碍事，4-bit 直接崩。所以 4-bit 权重的尺子必须比 per-channel 更贴身：**bit 数越少，尺子必须切得越细。**

**代价**：尺子自己开始占地。group size 128 时，每 64 字节的权重背 4 字节的 scale+zero（FP16），元数据约 **6%**；另外 int4 没有原生整数 GEMM 硬件，group-wise 要靠「先反量化回 FP16 再算」或专用 kernel（GPTQ 的 Marlin、AWQ kernel 等）——粒度越细，kernel 越难写。

## 6. 三者放一起

```text
权重矩阵 W（4 行 × 8 列）的三种切法
per-tensor           per-channel          group-wise（g=4）
┌───────────────┐   ┌───────────────┐    ┌─────┬─────┐
│               │   │ s1 ────────── │    │ s1  │ s2  │
│   一把 (s,z)   │   │ s2 ────────── │    ├─────┼─────┤
│   量整个矩阵   │   │ s3 ────────── │    │ s3  │ s4  │
│               │   │ s4 ────────── │    ├─────┼─────┤
│               │   │               │    │ s5  │ s6  │
│               │   │               │    ├─────┼─────┤
│               │   │               │    │ s7  │ s8  │
└───────────────┘   └───────────────┘    └─────┴─────┘
1 把尺子             4 把尺子              8 把尺子
```

结论链：

$$
\mathrm{GroupSize} \downarrow \ \Rightarrow\ \text{Scale 数量} \uparrow \ \Rightarrow\ \text{粒度更细} \ \Rightarrow\ \text{通常精度更好}
$$

三种粒度的本质 trade-off：

| 粒度 | Scale 数量 | 精度 | 额外开销 |
|---|---|---|---|
| Per-Tensor | 最少（1 个） | 较低 | 最低 |
| Per-Channel | 较多（out 个） | 较高 | 很低（约 0.1%） |
| Group-wise（g=128） | 最多（out·in/128 个） | 最高 | 中等（约 6%） |

所以可以画成一条谱：

```text
粒度由粗到细（组大小由大到小）
per-tensor   → 组 = 整个矩阵   → 1 把尺子
per-channel  → 组 = 一行       → out 把尺子
group-wise   → 组 = 128 元素   → out·in/128 把尺子
越往右：尺子越多 · 值域越窄 · 格点越密 · 精度越好 · 元数据越多
```

## 7. 实际工作里谁在用

**8-bit 时代**（格点密，per-channel 就是甜点）：

- 标准套餐：**权重 per-channel + 激活 per-tensor**（CNN 和 QAT 时代的默认组合）。
- **SmoothQuant**：把激活里的 outlier「搬」到权重——激活第 $j$ 列除小、权重第 $j$ 行乘大。这招能成立，正是因为**权重本来就是 per-channel 的**：被抬高的那一行有自己的尺子，不会被别的行拖累；激活被抹平后可以安心 per-tensor，凑出硬件最爱的 W8A8。
- **LLM.int8()**：激活按 token、权重按输出通道量化（都满足 §4 的口诀，scale 能提出求和号）；再把 outlier 所在的个位数隐藏维度抠出来单独走 FP16——「少数维度用高精度」，是与粒度正交的另一条路。

**4-bit 时代**（格点只剩 15 个，必须切到组）：

- **GPTQ**：group size 128 逐组量化，每组一套 scale/zero-point；再用二阶 Hessian 信息「每量化一列，把误差补偿给还没量化的列」——粒度负责尺子贴身，Hessian 补偿负责剩下的误差。
- **QLoRA**：NF4 权重按块 64 存 scale；「双重量化」再把 scale 本身按 256 块量化成 8-bit，元数据总开销压到 **0.127 bit/参数**（约 3%）。
- **AWQ**：观察到约 1% 的「显著通道」承载激活幅度主体，按通道搜索缩放因子保护它们——4-bit 权重 + 16-bit 激活即可不掉点。
- **MX 微缩放格式**：块大小固定 32、scale 取 2 的幂，Blackwell 起硬件原生支持——粒度第一次进了硬件标准。

**选择三句话**：

1. **bit 数决定粒度下限**：8-bit 用 per-channel 就够；4-bit 必须 group-wise（或 AWQ 式激活感知缩放）兜底。
2. **outlier 决定切不切**：值域均匀就粗切甚至不切；有 outlier 就切到「每把尺子只看到均匀的一片」为止。
3. **再细没有意义**：每元素一把尺子 = 元数据和数据一样大，等于没量化。所以实际停在 group 32~128。

## 8. 边界与延伸

- **零点与对称**：$z$ 按同样粒度共享。权重几乎总用对称（$z = 0$）；激活常非对称（ReLU/GELU 之后非负、分布单边）。
- **粒度 vs 裁剪（clipping）是正交的两件事**：粒度决定「谁和谁共用尺子」，clipping 决定「量程取到极值的百分之几」（比如只量到 99.9% 分位，把最野的 outlier 裁掉）。GPTQ、AWQ 都是切细 + 裁剪并用。
- **激活侧的粒度**：per-tensor 激活同样怕 outlier token；LLM.int8、ZeroQuant 用 per-token（每行现算 max）。但 per-token 要多一个运行时算 max 的 kernel——SmoothQuant 抹平后能退回 per-tensor，正是它的卖点。
- **与本站其他文章联动**：8-bit Adam 的动量量化也是 block-wise（块 2048），把优化器状态从 8Ψ 压到 2Ψ，见 [训练显存账本](/blog/2026-09-26-training-memory-notes/)；QLoRA 把 NF4 双重量化与 LoRA 结合，见 [LoRA 那篇](/blog/2026-09-26-lora-notes/)。

## 全称速查

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| PTQ | Post-Training Quantization | 训练后量化，不训练直接量 |
| QAT | Quantization-Aware Training | 量化感知训练，训练时就模拟量化 |
| GEMM | General Matrix Multiply | 通用矩阵乘 |
| scale | — | 量化步长/缩放因子 |
| zero-point | — | 零点 |
| outlier | — | 离群值 |
| MX | Microscaling | 微缩放格式，块大小 32 的硬件量化格式 |

## 9. 总结

- 量化的精度 = 值域 ÷ 格数；**scale 怎么分配**是量化的一切。
- 三种粒度是同一个「组」概念的三个档位：组 = 全张量（per-tensor）、组 = 一行（per-channel）、组 = g 个元素（group-wise）。
- per-tensor 死于 outlier；per-channel 是数学和硬件共同选出的甜点（通道独立 + scale 后乘免费）；group-wise 是 4-bit 时代的刚需，代价是约 6% 的元数据和专用 kernel。
- 一句话：**粒度做的事，就是让每一把尺子的「值域」尽量窄、尽量均匀；而尺子越多，尺子自己也越占地。**

## 参考文献

1. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
2. Krishnamoorthi. *Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper.* arXiv:1806.08342.
3. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339.
4. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438.
5. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
6. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
7. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
