---
title: 量化开销对比：各家方案的矩阵乘账本与省账手段
description: 量化省显存是明账，但量化之后的矩阵乘是怎么算的？两条路线——原生整数 GEMM 的修正项账、4-bit 反量化的融合 kernel——逐个拆 W8A8 / W4A16 / FP8 / LLM.int8 的额外开销，最后把优化手段归成三个动作：折叠、融合、离线化。
pubDate: 2026-09-27T00:30:00+08:00
tags: [大模型, 训练工程]
---

## 1. 先问一个问题：int4 权重，矩阵乘怎么算

量化省显存是明账：FP16 权重每个参数 2 字节，int4 只剩 0.5 字节（存储账见[上篇](/blog/2026-09-26-quantization-granularity-notes/)）。但 GEMM（General Matrix Multiply，通用矩阵乘）算的是浮点数——量化后的整数是怎么参与计算的？

只有两条路线：

```text
路线 A：整数 GEMM（8-bit 时代的主干道）
   int8 W × int8 X → int32 累加 → 乘 scale → 输出
   TensorCore（GPU 矩阵乘专用硬件）原生支持，吞吐约为 FP16 的 2 倍

路线 B：反量化回浮点（4-bit 时代的绕行路）
   int4 W → 解包 × scale → FP16 → 正常 FP16 GEMM
```

所有量化方式的「计算开销」，都是这两条路线上的额外工作。逐个拆。

## 2. 路线 A 的账：非对称的修正项

把 $W = s_w(W^q - z_w)$、$x = s_x(x^q - z_x)$ 代进 $y_i = \sum_j W_{ij} x_j$（推导见上篇 §2）：

$$
y_i = s_w s_x \left( \sum_j W^q_{ij} x^q_j \;-\; z_x \sum_j W^q_{ij} \;-\; z_w \sum_j x^q_j \;+\; z_w z_x \right)
$$

四项的贵贱：

| 项 | 含义 | 开销 |
|---|---|---|
| $\sum_j W^q_{ij} x^q_j$ | 整数 GEMM 本体 | 硬件原生，吞吐 ≈ FP16 的 2 倍 |
| $-z_x \sum_j W^q_{ij}$ | 每行常数 | 行和可离线预计算，并进 bias——免费 |
| $-z_w \sum_j x^q_j$ | 全列求和 | 激活每次推理都变，多一次归约——贵 |
| $z_w z_x$ | 常数 | 免费 |

账算完，工程分工就出来了（就是上篇的结论）：**权重对称（$z_w = 0$）+ 激活非对称**——只多一个并进 bias 的行常数；**双非对称**多一次全列归约；**全对称**只剩 GEMM 一条。

SmoothQuant 的价值正在于此：W8A8（权重、激活各 8-bit）且**全对称**——把激活的 outlier 搬进权重后，激活可以静态 per-tensor 对称量化——GEMM 之外零修正项，还吃到 INT8 TensorCore 约 2 倍于 FP16 的吞吐。

## 3. 路线 B 的账：反量化

int4 的整数 TensorCore 其实一直有（Turing 起就有，吞吐是 int8 的 2 倍），但它是「裸 int4」：**scale 只能按张量或按行给，表达不了 group-wise「每 128 个元素一个 scale」**——scale 在规约维度中间变，硬件不支持。软件栈也没有通用入口（cuBLAS 的 GEMM 最小到 int8）。所以 4-bit 权重量化在实践中一律走路线 B。

**朴素做法（反面教材）**：把整个权重矩阵反量化成 FP16 存回显存，再跑普通 FP16 GEMM：

```text
int4 (0.5B/参数) → [反量化] → FP16 (2B/参数) 存回显存 → FP16 GEMM
权重流量放大 4 倍——省下的带宽全还回去了
```

**融合做法（标准姿势）**：反量化只在 GEMM kernel 内部发生，结果只待在寄存器里，不写回显存：

```text
int4 (0.5B/参数) → [kernel 内：解包 + 乘 scale] → 寄存器 FP16 → 立即乘加
显存里只有 int4 进、结果出，中间不物化
```

解包不是免费的——位操作、按组查 scale、寄存器压力——但比 GEMM 本体便宜得多，所以「融合反量化」是 4-bit 推理的标准姿势。GPTQ 的 **Marlin** kernel、AWQ 的 GEMM kernel 干的都是这件事（Marlin 还专门重排权重的内存布局，让 scale/zero 按组对齐、异步预取）。

代价的另一面：**kernel 难写**。粒度越细（group 32~128）、打包格式越花（GPTQ 的行交错），融合 kernel 越难写、越难调优——4-bit 生态的「格式之争」就来源于此。

直到 Blackwell：MXFP4（Microscaling FP4）/ NVFP4（NVIDIA FP4）第一次把「块内 scale」做进硬件（块 32 的 E8M0 指数 scale），4-bit 才重新回到路线 A（见 §5）。

## 4. 动态量化的账：max 不是白来的

激活的 scale 从哪来？两种：

- **静态（离线）**：部署前用校准集定好 $s$（min-max、分位裁剪、MSE 搜索），运行时零开销。
- **动态（在线）**：per-token（每个 token 现算一把尺子）现算 $\max|x|$，每次推理多一次归约（多读一遍激活），再把 per-token scale 逐行乘到 GEMM 结果上。

LLM.int8() 就是动态路线：激活按 token 现算 scale，权重按输出通道；激活幅度超阈值的隐藏维度（经验上只占极少数）拆出来单独走 FP16：

```text
LLM.int8 的一次前向：
   X → [拆列] → 正常列 → int8 GEMM（动态 per-token scale）
              → outlier（离群值）列 → FP16 GEMM
   两条结果 → [合并] → 输出
   额外账：一次 max 归约 + 两次 GEMM + 一次拆/合
```

所以 LLM.int8 **不比 FP16 快多少**——它买的是「内存减半 + 精度不掉」，不是速度。

SmoothQuant 是反着解这道题：把 outlier 从激活搬进权重（上篇 §8），激活变得均匀 → 静态 per-tensor → **max 归约、per-token scale、非对称修正项三条账一次全清**，还解锁全对称 W8A8。一次离线搬移，换运行时零额外开销。

## 5. 各家开销对比总表

（记号：W8A8 = 权重、激活各 8-bit；W4A16 = 权重 4-bit、激活 16-bit；FP8 = 8 位浮点，E4M3/E5M2 见速查表。）

| 方案 | W / A 精度 | 矩阵乘怎么算 | 主要额外开销 | 优化手段 |
|---|---|---|---|---|
| W8A8 全对称（SmoothQuant） | int8 / int8 | 原生 INT8 GEMM | 仅 scale 后乘 | 离线搬 outlier |
| W8A8 权重对称 + 激活非对称 | int8 / int8 | INT8 GEMM + 修正项 | 行常数并入 bias | bias 折叠 |
| LLM.int8 | int8 + FP16 / int8（动态） | 两次 GEMM | max 归约、拆列合并 | outlier 拆列 |
| W4A16（GPTQ） | int4 g128 / FP16 | 融合反量化 → FP16 GEMM | kernel 内解包 + scale | Marlin kernel、Hessian 补偿 |
| W4A16（AWQ） | int4 / FP16 | 融合反量化 → FP16 GEMM | kernel 内解包 + scale | 通道缩放保护、epilogue 折叠 |
| FP8（E4M3） | fp8 / fp8 | 原生 FP8 GEMM | 几乎为零 | Hopper 起原生支持 |
| MXFP4 / NVFP4 | fp4 / fp8 | 原生 FP4 GEMM | 几乎为零（块 32 scale 在硬件内） | Blackwell 原生支持 |
| NF4（QLoRA） | nf4 / BF16 | 反量化回 BF16 | 双重量化解包 | 训练专用：省显存不图快 |

三条规律：

1. **8-bit 走路线 A**：开销只剩修正项，对称化就是免费的优化。
2. **4-bit（Blackwell 前）走路线 B**：融合 kernel 是唯一正确姿势；Blackwell 起路线 A 重开。
3. **动态统计是隐藏的大头**：LLM.int8 不快的账全在这一条。

## 6. 优化手段：折叠、融合、离线化

所有手段可以归成三个动作。

**折叠：把额外账并入已有步骤**

- 修正项折叠：$z_x$ 的行和离线算好，并进 bias；$z_w$ 直接砍掉（权重对称化）。
- scale 折叠：per-channel 的 scale 后乘放在 GEMM 的 epilogue（输出后处理）里逐行乘，不单独开 kernel；AWQ 连通道保护缩放也折进去。
- SmoothQuant 的「搬」本质也是折叠：把运行时的不均匀，折成离线的一次性调整。

**融合：让中间结果不落地**

- 反量化融合：int4 解包 + 乘 scale 全留在 kernel 内（Marlin / AWQ GEMM / CUTLASS 混合 kernel）。
- 拆合融合：LLM.int8 的拆列、合并尽量在 kernel 边界内完成，不做显存往返。

**离线化：把运行时开销挪到部署前**

- 校准与分位裁剪：min-max / 99.9% 分位 / MSE 搜索，换一个静态 scale 的零运行时开销。
- 双重量化：scale 本身再量化（QLoRA：FP32 块 64 的 scale → 8-bit 块 256，元数据 0.127 bit/参数），运行时只多一次解包。
- 权重的行和、group scale 的排布预取，全部离线排好。

选型框架（两条轴）：

```text
先看硬件代数：Ampere    → int8 原生，int4 走融合 kernel
              Hopper    → FP8 原生
              Blackwell → FP4（MXFP4/NVFP4）原生
再看瓶颈类型：decode（逐 token，访存受限）→ 量化主要赚带宽，W4A16 融合就够
              prefill（长序列，算力受限）→ 原生 INT8/FP8 GEMM 的 2 倍吞吐才发威
```

**判断顺序：先问卡在访存还是算力，再问硬件有没有原生单元，最后才选格式和 kernel。**

边界：本文算的是「推理时矩阵乘」的账。QAT（Quantization-Aware Training，量化感知训练）与 PTQ（Post-Training Quantization，训练后量化）的计算开销相同，差别在校准/训练成本（离线）；训练侧的量化是另一套账——QLoRA 训练比 16-bit 慢约 4 成，换显存降到约 1/4，见 [LoRA 那篇](/blog/2026-09-26-lora-notes/)与[训练显存账本](/blog/2026-09-26-training-memory-notes/)。

## 7. 白话总结

- 量化的计算开销只有三个来源：**修正项（$z$ 的账）、反量化（4-bit 的账）、动态统计（max 的账）**。
- 优化手段只有三个动作：**折叠、融合、离线化**。
- 时代甜点：8-bit 是全对称 W8A8（SmoothQuant）；4-bit 是融合反量化 W4A16（GPTQ/AWQ + Marlin）；再往后是硬件原生 FP8/FP4。
- 一句话：**量化省的是访存；计算上要么把账折进现有步骤，要么把账留到离线，要么等硬件原生。**

## 全称速查

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| GEMM | General Matrix Multiply | 通用矩阵乘 |
| PTQ | Post-Training Quantization | 训练后量化，部署前离线量化 |
| QAT | Quantization-Aware Training | 量化感知训练，训练时就模拟量化 |
| FP16 / BF16 | Half Precision / Brain Floating Point 16 | 16 位浮点 |
| FP8 | 8-bit Floating Point（E4M3 / E5M2） | 8 位浮点，E4M3 算、E5M2 存 |
| int8 / int4 | 8-bit / 4-bit Integer | 8/4 位整数 |
| TensorCore | — | GPU 上的矩阵乘专用硬件单元 |
| epilogue | — | GEMM 收尾阶段：输出后的逐元素处理（乘 scale、加 bias） |
| memory-bound | — | 访存受限：速度卡在显存带宽 |
| compute-bound | — | 算力受限：速度卡在计算吞吐 |
| 归约（reduction） | — | 把一批数合成为一个数（如 max、求和） |
| Marlin | — | GPTQ 4-bit 推理的融合 kernel |
| NF4 | NormalFloat 4-bit | QLoRA 的信息论最优 4-bit 格式 |
| MXFP4 / NVFP4 | Microscaling FP4 / NVIDIA FP4 | 硬件原生的块缩放 4-bit 格式（块 32） |

## 参考文献

1. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* arXiv:1712.05877.
2. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* arXiv:2208.07339.
3. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438.
4. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
5. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
6. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
7. Frantar et al. *Marlin: Nearly Ideal Inference Speed for 4-bit Large Language Models.* arXiv:2408.11743.
8. Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433.
9. Rouhani et al. *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537.
