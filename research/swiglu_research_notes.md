---
title: SwiGLU 调研笔记（内部）
description: 写 SwiGLU 博客前的调研：网上博客的讲法谱系、可借鉴角度、差异化空间。
pubDate: 2026-08-26T12:38:48+08:00
tags: [模型结构]
draft: true
---

# SwiGLU 博客调研笔记（写文章前参考）

> 调研日期：2026-08-26。目的：看网上博客怎么讲 SwiGLU / GLU 变体，找出可借鉴角度与本博可差异化的点。

## 一、调研来源与各自讲法

| 来源 | 风格 | 核心讲法 | 可借鉴 | 弱点（我们的机会） |
|---|---|---|---|---|
| [Naoki Shibuya](https://naokishibuya.github.io/blog/2023-04-30-swiglu-2020/) | 论文精读式 | 按 ReLU→GELU→Swish→GLU→变体→实验表 顺序，公式齐全，每步配 matplotlib 曲线代码 | 谱系讲法清晰；论文实验数字（T5 Base, d_ff 3072→2048）直接可用 | 纯转述论文，无自己的数字，无代码验证 |
| [Brenndoerfer](https://mbrenndoerfer.com/writing/gated-linear-units-swiglu-transformer-ffn) | 交互式教科书 | 先讲"门"的直觉（LSTM 血统）：σ→0 全关 / →1 全通 / 中间半开；两条路一个输出；ReLU vs SwiGLU 激活分布直方图对比；有 d_model=4→d_ff=6 的手算小例子 | "门=可学习阀门"的直觉铺垫；分布直方图对比的展示方式 | 手算例子埋在交互单元里，不突出；无参数账 |
| [Sebastian Raschka FAQ](https://sebastianraschka.com/faq/docs/swiglu-modern-llms.html) | FAQ 问答式 | 定义极简 + 对照表（GLU/GEGLU/SwiGLU × 门函数 × 矩阵数）+ 参数匹配账：3d·m_g vs 2d·m，m_g≈2m/3≈8d/3；大量"务实提醒"：gate/up 名字可互换、SwiGLU 不减 KV cache、SiLU vs GELU 差异在标准误内（Swish 只是略便宜） | 对照表形式；2/3 规则的参数推导写法；诚实的 caveat 段落 | 太短，不展开机制 |
| [罗西的思考·博客园](https://www.cnblogs.com/rossiXYZ/p/18765884) | 中文长文系列（最全） | FFN 的 position-wise 本质（kernel=1 卷积视角、token 内维度混合 vs 注意力的 token 间混合）；中间层比率 2022-24 趋势（标准4x → 门控2~8x 灵活）；激活函数年代趋势（2022 ReLU→2023 GELU→2024 SiLU）；**后半篇独有：FFN=知识存储**（KV 记忆网络解释、ROME、知识神经元、字典学习/SAE） | "注意力管 token 间、FFN 管 token 内"的分工讲法；知识存储视角可做延伸节；哈佛代码+llama3 代码对照 | 太长太杂（4.5 万字），数学散；手算缺失 |
| [dev.to M Shojaei](https://dev.to/mshojaei77/...) | 工程升级指南 | "把 FFN 升级成 SwiGLU = 免费性能"；代码先行；Pitfalls 节：① 不缩 2/3 参数多 50% ② 门控乘法放大激活尖峰→伤 FP8 量化（Smooth-SwiGLU 研究）③ ReLU² 是挑战者（Nemotron-4 340B、Grok） | Pitfalls 观点值钱：FP8 尖峰问题、对齐 256 整数倍的工程细节 | 直觉讲解浅 |
| 知乎若干（详解SwiGLU / 从ReLU到SwiGLU / 凭什么成为标配） | 中文科普 | 门控机制过滤信息、Swish 平滑可导缓解梯度消失、β 参数（β→0 线性 / β→∞ 退化 ReLU）；"标配"文有个好数字：100 层×11000 维 = 每 token 110 万次激活调用 | β 参数两极退化的讲法；调用次数的数量级感知 | 同质化严重，基本互相转抄 |

补充素材：
- **Shazeer 原论文金句**："We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."（我们无法解释为什么有效，归于神的恩典）——绝佳的行文素材：连作者本人都不解释，各博客的"为什么"全是事后合理化
- **ReLU² Wins 论文** (arXiv 2402.03804)：1B 同预算对照，ReLU² 在 稀疏度-性能 trade-off 全胜：~90% 稀疏下性能损失 <0.1%，FFN I/O 减少 92%；SwiGLU 与 ReGLU 性能几乎相同，裸 ReLU 稳定垫底
- **Masked GLU 论文** (arXiv 2506.23225)：推理带宽视角——SwiGLU 每 token 要读两个权重矩阵（gate+up），GELU 只读一个；门控在 memory-bound 场景反而更贵
- **Raschka PolyNorm 页**：前沿续集——Motif 3 Beta 用 PolyNorm（可学习的 x/x²/x³ 混合）替换 SiLU；"激活函数之争已死，剩下的是工程取舍"的判断
- **Raschka LinkedIn 演进链**：Norm: LayerNorm→RMSNorm→DyT；Attn: GQA/MLA/…；FFN: GeLU→SiLU→SwiGLU→MoE —— 正好把我们已写的 RMSNorm、GQA 和这篇串成系列

## 二、共性骨架（几乎所有博客都走这条路）

1. 从标准 FFN（升维→激活→降维，两矩阵）讲起
2. ReLU 的问题（死区/不平滑）→ GELU/Swish 曲线对比
3. 引入门控：GLU = σ(xW)⊙(xV)，sigmoid 当阀门
4. 变体家族表：Bilinear/ReGLU/GEGLU/SwiGLU
5. 三矩阵 → 参数多 50% → 2/3 规则
6. 论文实验结果引用 + 谁在用（Llama/PaLM/Qwen...）

## 三、全网都没做好的（我们的差异化空间）

1. **2/3 规则没人认真算过**：都引用 8d/3，没人用 Llama2-7B 真实数字走一遍（4096→11008 = 2/3×16384 再向上取整到 256 的倍数；11008×3 = 2×16384−512 的细节）
2. **手算案例缺位**：Brenndoerfer 有个 d=4 的小例子但埋在交互里。我们的招牌是"一组能手算的数字走通全流程"（RoPE d=4 案例、GQA 2×2 槽位案例同款）
3. **激活曲线图没有对比关键点**：各博客画 ReLU/GELU/Swish 曲线，但没标注 Swish 的关键特征（最小值 x≈-1.28 处 f≈-0.278、负区间非零、非单调性）
4. **"为什么有效"的诚实讨论**：Reddit r/ML 有讨论但博客都回避——Shazeer 自己都说 divine benevolence；乘性交互带来 x·x 交叉项（可拟合二次型）是最常见的假说，可以正面展开
5. **FP8/量化坑**：只有 dev.to 提了一句，可结合"超级权重"（苹果 2024 论文，罗西文末引用）讲 SwiGLU 尖峰问题
6. **verify 脚本**：没人做"文中数字全部可复跑"的验证脚本（我们 RMSNorm/GQA 的惯例）

## 四、本博文章结构草案（对齐系列三板斧）

标题候选（沿用「技术名：一句反直觉」格式）：
- SwiGLU：激活函数不是用来加非线性的，是用来当阀门的
- SwiGLU：一个 FFN 从两个矩阵变成三个的理由
- 从 ReLU 到 SwiGLU：FFN 的门是怎么装上的（演进史型，对齐 GQA 篇）

结构（先说结论表 → 机制 → 参数账 → 手算 → 代码 → 总结/延伸）：
1. 开篇结论表：FFN_ReLU / FFN_SwiSh / FFN_SwiGLU 三行对照（矩阵数、门、参数）
2. 标准 FFN 与 ReLU 的问题（死区；token 内混合的 position-wise 本质——借鉴罗西）
3. Swish/SiLU：x·σ(x)，β 两极退化，最小值 -0.278@-1.28，非单调
4. 门控：GLU 家族表（σ/ReLU/GELU/Swish 做门）
5. 参数账：2→3 矩阵，2/3 规则推导 + Llama 真实数字 11008
6. 手算案例：d=2→d_ff=4 的 SwiGLU 全流程数字
7. 代码：Llama MLP 三行（gate/up/down）+ 名字可互换的 caveat
8. 为什么有效：诚实讨论（Shazeer 金句 + 乘性交叉项假说 + 各家说法）
9. 延伸：ReLU²/稀疏派（90% 稀疏、I/O -92%）、PolyNorm 可学习激活、FP8 尖峰/超级权重

配图（drawio 三张，沿用系列风格）：
- 图1 ReLU/GELU/SiLU 曲线 + GLU 门控结构对比
- 图2 参数账：2 矩阵 vs 3 矩阵 + 2/3 规则几何
- 图3 Llama MLP 代码流程（gate/up/down 形状流转）
