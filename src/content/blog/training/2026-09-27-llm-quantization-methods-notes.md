---
title: LLM Quantization（大模型量化）：GPTQ、AWQ、SmoothQuant 的核心思路
description: 三个名字站在两条战线上：W4A16 的 GPTQ（逐列量化 + Hessian 补偿误差）与 AWQ（激活感知缩放，保护显著通道），W8A8 的 SmoothQuant（把激活 outlier 搬进权重）。从「量化误差伤在哪」推导三种思路的来源，再各自落到 kernel 与工程细节。
pubDate: 2026-09-27T01:30:00+08:00
tags: [大模型, 训练工程]
---

## 1. 三个方法，三种「态度」

大模型量化有两个战场：

- **只量化权重（W4A16）**：权重压到 4-bit，激活保持 16-bit（FP16/BF16）。
- **权重和激活都量化（W8A8）**：两边都压到 8-bit，才吃得到整数 GEMM 的吞吐。

三个名字正好站在这两条战线上，而且对「量化误差」有三种不同态度：

```text
只量权重（W4A16）                        权重 + 激活都量（W8A8）
├─ GPTQ：误差已经产生了 → 用数学补偿掉      └─ SmoothQuant：难点在激活 → 把难点搬走
└─ AWQ：与其补偿，不如一开始就别让重要通道误差大
```

各自一句话的核心思路：

- **GPTQ**（GPT Quantization）：逐列量化，把每一列产生的误差**补偿**给还没量化的列——用 Hessian 决定「怎么补」。
- **AWQ**（Activation-aware Weight Quantization）：权重的重要性由激活幅度决定，按激活幅度**缩放**权重，把少数显著通道**保护**起来。
- **SmoothQuant**：激活里的 outlier 是 W8A8 的拦路虎，那就把 outlier **搬**进权重——让激活变得好量。

三个思路的展开方式是一样的：一个问题 → 一个观察 → 一套做法 → 工程落地 → 局限。先看它们共同的出发点。

## 2. 共同的起点：量化误差伤在哪

一个线性层 $y = Wx$。量化权重得到 $\hat{W} = W + \Delta W$，于是：

$$
\hat{y} = (W + \Delta W)\,x = y + \Delta W\,x
$$

**输出误差就是 $\Delta W\,x$**。拆到每个输出通道：

$$
\Delta y_i = \sum_j \Delta W_{ij}\, x_j
$$

这个式子看着简单，但它是三个方法共同的出发点，直接推出两个结论：

1. **同一个权重误差，乘上越大的激活，伤害越大。** 也就是说「权重重不重要」不由权重本身决定，而由它对应的输入激活幅度决定——这是 AWQ 的全部立足点。
2. **想最小化整个校准集上的误差 $\lVert \Delta W X \rVert^2$，就必须利用 $X$ 的统计结构**（$XX^{\mathsf{T}}$，也就是 Hessian）——这是 GPTQ 的全部立足点。

（SmoothQuant 的出发点略不同：它的问题不在权重误差，而在**激活根本量化不动**——这个到 §5 再讲。）

白话：**权重误差本身不吓人，吓人的是「误差乘上了大的激活」。**

## 3. GPTQ：逐列量化 + 误差补偿

### 3.1 思路：量一列，改一列

如果一次性量化整个矩阵，量化误差产生了就产生了，没有挽回余地。GPTQ（以及它的前身 OBQ，Optimal Brain Quantization）换个玩法——**一列一列地量，每量完一列，就用「还没量化的列」去抵消它带来的输出误差**：

```text
逐列量化（沿输入维度从左到右）
   已量化列 | 当前列 q | 待量化列 F
              ↓ 量化：e_q = w_q − quant(w_q)
   用 F 去补偿：δ_F = − e_q / [H⁻¹]_qq · H⁻¹_{F,q}
   量完的 q 列从此固定，继续下一列
```

其中 $H = 2XX^{\mathsf{T}}$ 是校准集输入 $X$ 的 Hessian。补偿公式的直觉可以拆成三块：

- $e_q$：这一列量化产生的误差有多大；
- $1/[H^{-1}]_{qq}$：这一列「有多敏感」——它的重要性；
- $H^{-1}_{F,q}$：误差应该按什么方向、按什么比例摊给剩下的列。

### 3.2 从 OBQ 到 GPTQ：三个工程招

上面的公式 OBQ 就已经有了，但 OBQ 每量一列要做一次 $O(d^3)$ 的更新，175B 模型根本跑不完。GPTQ 的贡献主要在工程：

1. **Hessian 全层共用**：同一层所有输出行共用同一个 $X$ 算出的 $H$，所以整个权重矩阵的量化顺序一致、可以一次跑完（OBQ 是逐行独立处理）。
2. **预 Cholesky 分解**：$H^{-1} = LL^{\mathsf{T}}$ 提前分解好，量化过程中原地更新 $L$，把 $O(d^3)$ 的求逆只做一次。数值上再加一点阻尼（对角线加均值 1% 左右）保证可逆。
3. **惰性批量更新**：把 128 列攒成一批，批内产生的补偿更新合并成**一次矩阵乘**再应用——这是「快」的关键。

账：175B 模型 4-bit 量化约 **4 GPU 小时**（校准集 128 条序列），精度损失对下游任务几乎不可见。

### 3.3 工程落地与局限

- 运行时：**group 128** 的 4-bit 权重 + 融合反量化 kernel（Marlin），流程见[量化的计算开销](/blog/2026-09-27-quantization-compute-cost-notes/) §6——权重只读一遍、解包直接产出 MMA fragment、scale 离线 repack。
- 可选项 **act-order**（也称 desc_act）：把列按激活重要性重排后再量化，通常更准一点，但推理 kernel 要支持重排后的顺序（复现时得配对）。
- 局限：**它只治权重，不治激活**（激活仍是 16-bit）；依赖校准集，分布漂移会掉点；量化过程本身不便宜（离线一次性）。

白话：GPTQ 像「量一列、改一列」——用还没量化的列当橡皮擦，把已经产生的误差擦掉一部分。

## 4. AWQ：激活感知的缩放保护

### 4.1 观察：重要的不是权重，是激活

回到 §2 的结论「权重重不重要由激活幅度决定」，AWQ 论文把这个观察钉死了，还发现了两件反直觉的事：

- **权重幅度是个坏指标**：把「幅度最大的 1% 权重」保留成 FP16，效果竟然还不如随机保留 1%。
- **真正的显著通道很少**：按激活幅度看，只有约 **1%** 的输入通道是「显著」的；剩下 99% 的通道怎么量化都无所谓。

所以策略应该是：**别去猜哪些权重大，去看哪些通道的激活大。**

### 4.2 做法：缩放，而不是量化

AWQ 的核心手法是一个等价变换——**给输入通道乘一个缩放因子 $s$**：

$$
W' = W \cdot \mathrm{diag}(s), \qquad x' = \mathrm{diag}(s)^{-1} x
$$

$W' x' = W x$，数学上完全等价，但量化的难度变了：

```text
s 越大 → 该通道权重被放大 → 4-bit 下的相对量化误差 Δw/w 变小 → 精度保住
显著通道（激活大）：给大 s，重点保护
普通通道：给小 s，让位
激活本来就跑在 FP16，除以 s 几乎无损 —— 拿激活的动态范围，换权重的相对精度
```

$s$ 不能任性（放大权重也会放大权重本身的动态范围），所以要搜：

$$
s_j = \frac{(\mathrm{mean}\,|X_j|)^{\alpha}}{(\mathrm{mean}\,|W_j|)^{1-\alpha}}
$$

对 $\alpha \in [0,1]$ 做网格搜索，目标是让「量化后再反量化」的输出误差最小。注意这个公式的形状和 SmoothQuant 的几乎一样，但**目标不同**：AWQ 要保护通道，SmoothQuant 要压平激活。

### 4.3 工程落地与局限

- **为什么不用「显著通道留 FP16」？** 这是 AWQ 论文专门讨论过的：混合精度会让 GEMM 掉进慢路径（硬件里 FP16 与 INT4 混算要走低效通道），省下的精度还不够赔速度。而**缩放可以折进前后面的算子**，运行时零额外开销。
- 折法：$s$ 乘进权重（离线完成），$1/s$ 折进**前一层**（LayerNorm/RMSNorm 的缩放参数，或前一个 Linear 的输出通道）——推理时没人再算这个缩放。
- 量化成本很轻：只要**前向**统计激活幅度 + 一次 $\alpha$ 网格搜索，不像 GPTQ 要做 Hessian 和逐列回归。运行时走 W4A16 融合反量化 kernel（AWQ GEMM / vLLM 的 awq kernel 等）。
- 局限：它只做「保护」，不做「补偿」——没有 GPTQ 那种误差修正机制；$s$ 是 per-channel 粒度（比 group-wise 粗）；同样依赖校准集。

白话：AWQ 像给重要通道配了一副放大镜——放大之后，同样的绝对误差，相对就小了。

## 5. SmoothQuant：把 outlier 从激活搬到权重

### 5.1 问题：激活量化不动

W8A8 的死穴在激活：激活里存在 **outlier**（离群值），个别通道的幅度是中位数的上百倍，而且 token 之间的分布还在剧烈变化。

```text
某层激活的某一列：最大值 100，中位数 0.5
per-tensor 量化：scale 按 100 定 → 0.5 这种值连一格都占不到，全被舍成 0
per-token 动态量化：每 token 现算 scale 能救精度，但每次推理多一次 max 归约（见计算开销那篇 §4）
```

### 5.2 做法：把难度「对半劈」

SmoothQuant 的手法同样是等价变换，但方向相反——**把激活除以 $s$，把权重乘以 $s$**：

$$
W' = W \cdot \mathrm{diag}(s), \qquad x' = \mathrm{diag}(s)^{-1} x
$$

迁移力度由 $\alpha$ 控制：

$$
s_j = \frac{\max |X_j|^{\alpha}}{\max |W_j|^{1-\alpha}}
$$

- $\alpha = 1$：难度全留在激活（不搬）；
- $\alpha = 0$：难度全推给权重；
- $\alpha = 0.5$：**对半劈**，最常用的默认值。

```text
迁移前：激活某列最大 100、中位数 0.5      → per-tensor 量化：0.5 → 0
迁移后：激活 / s（s ≈ 10），最大 10、整体压平 → per-tensor 量化：大部分值都有格可占
        同时权重的对应行 × 10 —— 权重的精度富裕，背得动
```

直觉：权重分布本来就均匀，让它多承担一点动态范围不亏；激活被压平之后，per-tensor 静态量化就成立了。

### 5.3 效果：解锁 W8A8 全对称

激活变均匀之后，链式反应是：

$$
\text{激活可 per-tensor 静态量化} \Rightarrow \text{W8A8 且全对称} \Rightarrow \text{原生 INT8 GEMM，零修正项}
$$

这正是[上篇](/blog/2026-09-27-quantization-compute-cost-notes/)算过的那笔最省的账——GEMM 之外什么都不用做，还吃到 INT8 TensorCore 约 2 倍于 FP16 的吞吐。如果还想要更高精度，可以退一步用 per-token 动态量化（多一次归约的代价）。

### 5.4 工程落地与局限

- 折法同 AWQ：$s$ 乘进权重、$1/s$ 折进前一层（LayerNorm 的 $\gamma$ 或前一个 Linear 的输出通道），**运行时零额外开销**。
- 离线成本：跑校准集统计 $\max|X_j|$ 与 $\max|W_j|$，再调 $\alpha$（论文给 0.5 通用，per-layer 搜更好）。
- 局限：它解决的是 **8-bit 这一档**；4-bit 激活是另一类难题（格点只剩 15 个，SmoothQuant 式的迁移不够用）。另外 $\alpha$ 和校准集要调，本身没有 outlier 的场景收益有限。

白话：SmoothQuant 不是让激活「自己变好」，而是把「难」按比例分给权重一部分——因为权重有本钱。

## 6. 三者放一起

| 方法 | 量化什么 | 核心思路 | 精度从哪来 | 离线成本 | 运行时 |
|---|---|---|---|---|---|
| GPTQ | 权重 4-bit（W4A16） | 逐列量化 + 补偿误差 | 把误差摊给未量化的列 | 最重：校准集 + Hessian + Cholesky（175B 约 4 GPU 小时） | 融合反量化 kernel（Marlin） |
| AWQ | 权重 4-bit（W4A16） | 按激活幅度缩放，保护显著通道 | 让重要通道的相对误差变小 | 轻：前向统计 + α 网格搜索 | 融合反量化 kernel（scale 已折进权重） |
| SmoothQuant | 权重 + 激活 8-bit（W8A8） | 把激活 outlier 搬进权重 | 让激活可 per-tensor 量化 | 轻：校准统计 + 调 α | 原生 INT8 GEMM（零额外开销） |

三个记忆锚：

```text
GPTQ        → 补偿（数学最优化视角：误差产生了，把它摊掉）
AWQ         → 保护（观察视角：谁重要，就放大谁）
SmoothQuant → 搬移（迁移视角：难点堵在激活，就把难点搬走）
```

两个视角总结：

- GPTQ 与 AWQ 是**同一条战线**（W4A16）的两种打法：一个靠数学修正误差，一个靠观察避开误差。
- SmoothQuant 站在**另一条战线**（W8A8）：它不跟 4-bit 的精度较劲，而是把 8-bit 这条本可以更快、更省的路打通。

## 7. 组合与边界

- **可以叠加**：权重用 GPTQ、激活用 SmoothQuant 的分位裁剪/平滑，就是 W4A8 一类的做法；AWQ 的缩放也和 group-wise 量化兼容（AWQ 本身就用 group）。
- **都依赖校准集**：三者都是 PTQ（Post-Training Quantization，训练后量化）——分布漂移（比如换了领域的数据）会掉点，这也是 QAT（量化感知训练）存在的原因。
- **不同战线的邻居**：QLoRA/NF4 面向的是**训练**（省显存，不图推理速度）；FP8 / MXFP4 是**硬件原生**路线（Hopper / Blackwell），跟这三个「软件方法」是互补而非替代。
- **选型**：显存优先、能接受离线成本 → GPTQ；量化要快、要跨领域稳 → AWQ；想同时压激活、要 INT8 GEMM 吞吐 → SmoothQuant。

## 8. 总结

- 一切的起点是一个式子：输出误差 $= \Delta W\,x$——误差被激活幅度加权。GPTQ 从这里走向「用 Hessian 补偿」，AWQ 从这里走向「按激活保护」。
- GPTQ 的工程三招（共用 Hessian、预 Cholesky、批量更新）把 OBQ 的昂贵逐列更新变成了可用的 175B 方案。
- AWQ 用一次等价缩放，把精度花在 1% 的显著通道上；SmoothQuant 用同样的缩放，把难度从激活搬给权重。
- 一句话：**GPTQ 补偿误差、AWQ 保护重点、SmoothQuant 搬移难点**——三者都是「用离线的一点聪明，换运行时的一点精度」。

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
