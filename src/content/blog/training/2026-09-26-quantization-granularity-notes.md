---
title: 量化粒度：一个 scale 管多大一片地方
description: 量化的精度由「值域 ÷ 格点数」决定，而粒度决定值域怎么划片：per-tensor 一把尺子量全张量、per-channel 每个输出通道一把、group-wise 每 128 个元素一把。逐档讲清各自为什么存在、outlier 为什么逼着把尺子切细、per-channel 为什么在整数 GEMM 里几乎免费，最后给一张三档元数据账本。
pubDate: 2026-09-26T23:30:00+08:00
tags: [大模型, 训练工程]
---

## 摘要

量化（quantization）的全部精度，取决于一句话：**把一段值域均匀切成 2ᵇ − 1 格，格宽 = 值域 ÷ 格数**。比特数 b 决定格数，而「值域」怎么划——是整张张量共享一段，还是一行一段、128 个元素一段——就是量化粒度（quantization granularity）。本篇把三档粒度逐个拆开：

- **per-tensor（逐张量）**：整张权重一把尺子。最省事，但一个 outlier 就会拖垮所有其他元素；
- **per-channel（逐通道）**：每个输出通道一把尺子。outlier 被隔离在自己的通道里，而且在整数 GEMM 里几乎免费——8-bit 量化的标准甜点；
- **group-wise（分组）**：通道内部再按 32~128 个元素切块。4-bit 量化保精度的刚需，代价是 3%~6% 的 scale 元数据和专用 kernel。

一句话：**粒度 = 一把尺子（scale/zero-point）管多大一片地方**，越细越准、越细越贵，工程上永远在两者之间找平衡。

## 0. 两个前置概念

### 0.1 量化公式：scale 与 zero-point

量化把浮点数 $x$ 映射到低比特整数 $q$，标准写法是仿射量化（affine quantization）：

$$
x \approx s\,(q - z) \tag{1}
$$

- $s$：scale（缩放因子，也叫步长 step size）——量化的刻度宽度；
- $z$：zero-point（零点）——$x = 0$ 对应哪个整数格点，用来保证「0 必须被精确表示」（卷积里的 padding 零、ReLU 后的零）；
- $z = 0$ 时叫对称量化（symmetric），否则叫非对称（asymmetric）。

$s$ 最直接的取法：$s = (\max - \min)/(2^b - 1)$。int8 对称量化下 $s = \max|x|/127$，4-bit 下 $s = \max|x|/15$。**格宽就是 $s$，落在同一格里的数都表示成同一个整数——量化误差最大是半格宽 $s/2$。** 这就是「值域 ÷ 格数」的含义：格数由比特数钉死，值域越小、$s$ 越小、误差越小。

### 0.2 粒度 = (s, z) 的共享范围

$(s, z)$ 就是一把尺子。量化粒度问的是：**多少个元素共用一把尺子**。同一把尺子下，所有元素共用由同一个 max/min 决定的值域；粒度越细，每把尺子看到的值域越窄、越均匀。

## 1. per-tensor：一把尺子量全张量

**定义**：per-tensor quantization（逐张量量化）——整张权重矩阵（或整个激活张量）只算一个 $(s, z)$。

**什么时候够用**：当张量里的值域本身均匀时，一把尺子完全够，元数据开销几乎为零（只有一个 scale）。8-bit 时代的经典套餐「权重 per-channel + 激活 per-tensor」里，激活就是 per-tensor：CNN 模型的激活经过 BatchNorm 之后分布规整，一把尺子量得动。这套组合从 2018 年的量化训练（QAT，Quantization-Aware Training）论文起就是标准配置，沿用至今。

**什么时候翻车**：LLM 权重里藏着 outlier（离群值）——极少数通道的数值比别的通道大几十上百倍。per-tensor 的尺子由全张量的 max 决定，outlier 把 max 撑大，$s$ 变粗，所有正常元素被挤进同一个格：

```text
per-tensor 的灾难：int8 对称量化（127 格），一行权重 4 个通道
数值：    C1=0.01   C2=0.02   C3=0.01   C4=10.0
         └──────────┴──────────┘        └─ outlier ─┘
尺子由全行最大值决定：s = 10/127 ≈ 0.079（一格宽 0.079）
量化结果：   0        0         0        127
            ↑ 0.01、0.02 连半格都不到，全被舍成 0——三个通道的信息清零
```

白话：**一把尺子要同时量得了蚂蚁和大象，就只能按大象做刻度，蚂蚁量出来全是 0。** outlier 的破坏力不是「把它自己量错」，而是「拖着所有正常元素一起变粗」。

## 2. per-channel：一行一把尺子

**定义**：per-channel quantization（逐通道量化）——沿**输出通道**方向，每个通道一套 $(s, z)$。LLM 线性层的权重形状是 [out × in]，per-channel 就是每**行**一把尺子，共 out 把。

### 2.1 为什么偏偏是「输出通道」这个方向

两个理由，缺一不可：

1. **数学上通道天然独立**：输出向量的第 $i$ 个元素 $y_i = \sum_j W_{ij} x_j$，只用权重第 $i$ 行和整个输入。通道之间互不搭界——第 3 行的 outlier 只影响 $y$ 的第 3 个元素，别的通道不该为它买单。「每行一把尺子」是误差隔离的天然边界。
2. **硬件上几乎免费**：把量化权重代回矩阵乘：

$$
y_i = \sum_j X_{ij}\, s_{W,i}\, W^q_{ij} = s_{W,i} \sum_j X_{ij} W^q_{ij} \tag{2}
$$

$s_{W,i}$ 只依赖输出下标 $i$，**可以提到求和号外面**——整数 GEMM 照常算（int8 乘、int32 累加），算完再给每个输出行乘上自己的 $s_{W,i}$ 就行。per-channel 不改变 GEMM 内核，只多一个「按行后乘」。反过来，如果尺子沿**输入**方向（按列），$s$ 的下标就留在求和号里，提不出来，整数 GEMM 没法直接做——这就是所有标准实现（TensorRT、cuBLAS int8、ONNX 的 per-axis）都沿输出通道的原因。

同理可推激活侧的免费组合：激活按 token（行）量化时，$s_X$ 同样提到求和号外，与 $s_{W,i}$ 以外积形式相乘。**判断口诀：scale 的下标只要不出现在规约维度里，就是免费的。**

### 2.2 效果：outlier 被关进自己的房间

```text
per-channel：每个通道自己一把尺子
C1~C3 的尺子：s ≈ 0.02/127 ≈ 0.00016 → 0.01 落在约 64 格处，信息保留 ✓
C4 的尺子：   s ≈ 10/127  ≈ 0.079   → 10.0 落在 127 格处，信息保留 ✓
大象用大量程的尺子，蚂蚁用小量程的尺子，谁也不拖累谁
```

### 2.3 per-channel 的三个著名用户

- **SmoothQuant（arXiv:2211.10438）**：把「激活 outlier 迁移到权重」的载体。平滑因子 $s_j = \max|X_j|^{\alpha} / \max|W_j|^{1-\alpha}$（论文取 $\alpha = 0.5$）把激活的第 $j$ 列除小、权重的第 $j$ 行乘大。这招之所以成立，正是因为**权重本来就是 per-channel 的**——被抬高的那一行有自己的尺子，不会被别的行拖累；激活被抹平后可以退回 per-tensor，凑出硬件最爱的 W8A8。
- **AWQ（arXiv:2306.00978）**：观察到 LLM 权重里约 1% 的「显著通道」承载了激活幅度的主体。按通道搜索缩放因子 $s = (\mathrm{mean}|X|)^{\alpha}$，把这 1% 通道保护起来，4-bit 权重 + 16-bit 激活即可不掉点。缩放因子按通道搜，是因为要保护的对象正是「通道」这个单位。
- **LLM.int8()（arXiv:2208.07339）**：激活按 token、权重按输出通道做向量式量化（vector-wise quantization），两个方向的 scale 都按 §2.1 的口诀以外积提出，整数 GEMM 照跑；再把 outlier 所在的个位数隐藏维度抠出来单独走 FP16。这是「分维度分裂精度」的思路，与粒度正交。

## 3. group-wise：通道内部再切块

### 3.1 为什么 per-channel 还不够

per-channel 的尺子对**整行**仍按该行的 max 定刻度——行内依然大小值混居，行内的小值依然吃亏。8-bit 时代无所谓：127 格很密，吃点亏不碍事。但 4-bit 只有 15 格，**比 8-bit 粗约 8.5 倍**（127 ÷ 15），每格容错只剩 1/8.5——同样的行内不均匀，4-bit 下直接崩。结论：**比特数越少，尺子必须越贴身，粒度必须越细。** 于是通道内部再切块：group-wise quantization（分组量化，也叫 block-wise 分块量化），每 $g$ 个元素一把尺子，$g$ 典型取 32/64/128。

### 3.2 代表工作

- **GPTQ（arXiv:2210.17323）**：权重按 128 列一组（group size 128）量化，每组一套 scale/zero-point（FP16）；配合二阶 Hessian 信息「每量化一列，就把误差补偿给还没量化的列」。粒度负责「尺子贴身」，Hessian 补偿负责「剩下的误差」。GPTQ 是 PTQ（Post-Training Quantization，训练后量化）方法——只用量化前跑几批数据算 Hessian，不训练。
- **QLoRA（arXiv:2305.14314）**：NF4 权重按**块大小 64** 存 scale，一级 scale 用 FP32；「双重量化」（double quantization）把一级 scale 再按 256 块量化成 8-bit——元数据从 32 bit/块压到 8 bit/块，总开销 0.127 bit/参数。
- **MX 微缩放格式（Microscaling）**：块大小固定 32，scale 取 2 的幂、用 8 位指数 E8M0 存。Blackwell 起硬件原生支持 MXFP4——**粒度第一次进了硬件标准**。

### 3.3 元数据账本

以权重矩阵 [out × in]、4-bit、组大小 $g$ 为例，数一数尺子的数量与开销：

| 粒度 | 尺子数量 | 元数据开销（out=4096, in=4096） |
|---|---|---|
| per-tensor | 1 | ≈ 0 |
| per-channel | out | 4096 × 2B = 8KB，占权重的 0.1%，可忽略 |
| group-wise（g=128，FP16 存 s 和 z） | out·in/g | 每 64B 的组背 4B 尺子，约 **6.25%** |
| QLoRA（块 64 + 双重量化） | out·in/64 | 平均 **0.127 bit/参数**，约 3% |

白话：**尺子越切越多，尺子自己也开始占地。** group-wise 的 6% 看着不多，但要在「精度收益」和「这 6% 的显存与搬运开销」之间算账——QLoRA 的双重量化就是专门来砍这笔账的。

### 3.4 代价：kernel

per-tensor / per-channel 是整数 GEMM 的原生支持（cuBLAS、TensorRT 直接吃）。group-wise 没有这个待遇：**int4 本来就没有原生整数 GEMM 硬件**，group-wise 的落地要么「先反量化回 FP16 再做 FP16 GEMM」，要么靠专用 kernel（GPTQ 的 Marlin、AWQ 的量化 kernel）把反量化融合进计算。粒度越细，kernel 越难写——这也是组大小止步于 32~128 的原因之一。

## 4. 三档总账与选择逻辑

```text
权重矩阵 W（4 行 × 8 列）的三种切法
per-tensor           per-channel          group-wise（g=4）
┌───────────────┐   ┌───────────────┐    ┌─────┬─────┐
│               │   │ s₁ ────────── │    │ s₁  │ s₂  │
│   一把 (s,z)   │   │ s₂ ────────── │    ├─────┼─────┤
│   量整个矩阵   │   │ s₃ ────────── │    │ s₃  │ s₄  │
│               │   │ s₄ ────────── │    ├─────┼─────┤
│               │   │               │    │ s₅  │ s₆  │
│               │   │               │    ├─────┼─────┤
│               │   │               │    │ s₇  │ s₈  │
└───────────────┘   └───────────────┘    └─────┴─────┘
元数据：1 把尺子      元数据：4 把尺子       元数据：8 把尺子
精度最差，最省        精度中，几乎免费       精度最好，开始占地
```

三档对照：

| 粒度 | 一把尺子管多少 | 抗 outlier | 元数据 | 典型用途 |
|---|---|---|---|---|
| per-tensor | 整个张量 | 差 | ≈ 0 | 8-bit 激活（CNN / SmoothQuant 抹平后） |
| per-channel | 一个输出通道 | 中 | 可忽略 | 8-bit 权重标准套餐；AWQ 缩放因子 |
| group-wise | 32~128 个元素 | 好 | 3%~6% | 4-bit LLM 权重（GPTQ / QLoRA / MX） |

选择逻辑可以记成三句话：

1. **比特数决定粒度下限**：8-bit 格点密，per-channel 权重 + per-tensor 激活就是成熟套餐；4-bit 格点粗 8.5 倍，必须 group-wise 或激活感知的 per-channel 缩放（AWQ）兜底。
2. **outlier 决定切不切**：值域均匀就粗切甚至不切；有 outlier 就切到「每把尺子只看到均匀的一片」为止。
3. **再细没有意义**：每元素一把尺子 = 元数据和数据一样大，等于没量化。所以实际粒度停在 group 32~128，再往下就是 MX 这类硬件格式的天下。

## 5. 边界与延伸

- **对称 vs 非对称**：zero-point 按同样粒度共享。权重几乎总用对称（$z = 0$），激活常非对称（ReLU/GELU 之后非负、分布单边）。
- **粒度 vs 裁剪（clipping）是正交的两件事**：粒度决定「谁和谁共用尺子」，clipping 决定「每把尺子的量程取到极值的百分之几」（比如只量到 99.9% 分位，把最野的 outlier 裁掉）。GPTQ、AWQ 都是「切细 + 裁剪」并用。
- **激活侧的粒度**：per-tensor 激活同样怕 outlier token；LLM.int8、ZeroQuant 用 per-token。但 per-token 的 scale 要在运行时对每行现算 max，多一个 kernel、框架支持不一——SmoothQuant 抹平后能退回 per-tensor，正是它的卖点。
- **混合粒度/混合精度**：LLM.int8 把 outlier 维度抠出来走 FP16、其余 int8——「少数维度用高精度」与「多数维度切细粒度」是两条独立的路。
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

## 6. 总结

粒度从粗到细 = 尺子越来越多、每把尺子看到的值域越来越窄、格点越来越密：

- **per-tensor** 死于 outlier——一把尺子量不了蚂蚁和大象；
- **per-channel** 是数学和硬件共同选出的甜点——通道天然独立、scale 后乘免费；
- **group-wise** 是 4-bit 时代的刚需——格点只剩 15 个，尺子必须贴身，代价是 3%~6% 的元数据和专用 kernel。

一句话：**量化精度 = 值域 ÷ 格数；粒度做的事，就是让每一把尺子的「值域」尽量窄、尽量均匀。**

## 参考文献

1. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
2. Krishnamoorthi. *Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper.* arXiv:1806.08342.
3. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339.
4. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438.
5. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
6. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
7. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
