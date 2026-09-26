---
title: FlashAttention：快不是靠少算，是靠少搬
description: FlashAttention 的全部机制只有三条：分块做 online softmax（前向不造 N×N 矩阵）、反向重算（不存 S 和 P）、单 kernel 融合（一次读写）。这篇把三条逐一拆开，重点讲透分块合并时「打折」这一步从哪来、为什么不能省，再给一段可跑的 PyTorch 代码逐行认领，最后算清 HBM 读写为什么能从 Θ(N²) 降到 Θ(N)。
pubDate: 2026-09-26T22:00:00+08:00
tags: [模型结构]
---

先说结论：**FlashAttention 的全部机制只有三条**——

| | 机制 | 一句话 |
|---|---|---|
| ① 分块 | 不一次算完整行 softmax，而是逐块算，用 $(m,\ell)$ 两个统计量在线合并 | 前向从头到尾不出现 $N\times N$ 的矩阵 |
| ② 重计算 | 反向不保存 $S$、$P$，用 $Q,K,V$ 的块现场重算 | 多花 13% FLOPs，少搬 90% 的数据 |
| ③ 融合 | mask → softmax → dropout → 矩阵乘装进一个 kernel | 数据进一次 SRAM，算完才回 HBM |

需要的背景只有一个：**注意力不是算得慢，是搬得慢**。标准注意力的三步

$$
S = QK^\top,\qquad P = \mathrm{softmax}(S),\qquad O = PV
$$

里，两个矩阵乘是 compute-bound，但中间的 softmax（以及 mask、dropout）是逐元素操作——每读一个数只做几次运算，属于 memory-bound。看 A100 的内存层级：

![GPU 内存层级：SRAM 比 HBM 快约 12.7 倍，容量小约 2000 倍](/images/flashattention/flash_1_hierarchy.svg)

片上 SRAM 比 HBM 快约 12.7 倍，容量却小约 2000 倍。标准实现把 $S$ 和 $P$ 两张 $N\times N$ 矩阵写进 HBM——$N$ 翻一倍，搬运量翻四倍，而搬运走的正是那条 1.5 TB/s 的窄路。FlashAttention 的全部设计围绕一件事：**让中间矩阵只活在 SRAM 里，HBM 上只保留 $Q,K,V,O$ 和两个统计量**。

## ① 分块：online softmax

问题出在 softmax 上：要把一行归一化，就得先看全这一行——$P_i = \exp(S_i - \max S_i) / \sum_j \exp(S_{ij} - \max S_i)$。要算完整行，就得先有整行 $S$；要有整行 $S$，就得先把 $Q$ 和所有 $K$ 乘完。这就是标准实现必须落地 $N\times N$ 矩阵的原因。

关键观察：**softmax 的归一化只需要两个数——行最大值 $m$ 和指数和 $\ell$**。数值稳定版定义：

$$
m(x) = \max_i x_i,\qquad f(x)_i = e^{x_i - m(x)},\qquad \ell(x) = \sum_i f(x)_i,\qquad \mathrm{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

减 $m$ 不影响结果：分子分母同时除以 $e^m$，约掉了，$m$ 取什么值 softmax 都成立。**这个自由度就是分块的全部机会**——每块先按自己的局部最大值记账，合并时再换算。

把一行拆成两段 $x = [x^{(1)},\ x^{(2)}]$ 时，两段的统计量可以直接合并：

$$
m = \max\big(m^{(1)},\, m^{(2)}\big),\qquad \ell = e^{m^{(1)}-m}\,\ell^{(1)} + e^{m^{(2)}-m}\,\ell^{(2)}
$$

**合并式里的 $e^{m^{(1)}-m}$ 就是「打折」，这是全篇最关键的一步。** 第一段算 $\ell^{(1)}$ 时，手里只有它的局部最大值 $m^{(1)}$，所以指数是按基准 $m^{(1)}$ 记的账，每一项是 $e^{x_i - m^{(1)}}$。第二段带来了更大的 $m$，基准变了，旧账必须换算到新基准——每一项乘上

$$
e^{x_i - m^{(1)}} \times e^{m^{(1)} - m} = e^{x_i - m}
$$

因为 $m^{(1)} \le m$，这个换算因子 $e^{m^{(1)}-m} \le 1$，所以叫打折：**旧段的峰值没新的高，旧账整体缩水，差距越大折得越狠**。$\ell^{(1)}$ 的每一项都乘同一个因子，整个和直接乘这个因子。第二段本来就按新基准 $m$ 记账，它的因子 $e^{m^{(2)}-m}=1$，不打折——**公式里两项各带一个因子，赢家恰好是 1，输家按差距打折**（若第一段最大值反而更大，角色对调）。

为什么不能省掉、直接 $\ell^{(1)}+\ell^{(2)}$？两个和减的是不同的基准，相加没有意义；而且折扣只作用于输掉的那一段，不是两边共同因子，最后 $O/\ell$ 约不掉。**打折不是数值技巧，是把两本不同基准的账对齐。**

输出 $O$ 的合并是同一套逻辑，多乘一个 $V$：

$$
O_{\text{new}} = \mathrm{diag}(\ell_{\text{new}})^{-1}\Big(\underbrace{\mathrm{diag}(\ell)\,e^{m-m_{\text{new}}}\,O}_{\text{旧块整体打折}} + \underbrace{e^{\tilde m - m_{\text{new}}}\,\tilde P\, V_j}_{\text{新块}}\Big)
$$

$O$ 是加权和 $\sum e^{x_i-m}\,v_i$，每一项与 $\ell$ 的每一项乘同一个因子，所以 $O$ 和 $\ell$ **同时打折**；最后统一除以 $\ell$ 就是加权平均。

于是可以一次只搬一块 $K_j, V_j$（大小 $B_c\times d$）进 SRAM，与所有 $Q_i$ 块算出 $S_{ij}$、$\tilde P_{ij}$，更新统计量后**立刻把 $S_{ij}$ 丢掉**：

![分块流程图：外层循环沿 K、V 块（红），内层循环沿 Q 块（蓝），S 从不写回 HBM](/images/flashattention/flash_2_tiling.svg)

**为什么这招能成**：全程 HBM 上只有 $Q,K,V,O,m,\ell$，大小全是 $O(N)$；$S_{ij}$、$\tilde P_{ij}$ 是 SRAM 里的临时块，用完即弃。FLOPs 一个没省——还是 $\Theta(N^2d)$——省的是那两张 $N\times N$ 矩阵的落地。

## ② 重计算：反向不存 S、P

反向需要什么？把 $O=PV$ 对输入求导（记 $D_i = \mathrm{rowsum}(dO_i \odot O_i)$）：

$$
dV = P^\top dO,\qquad dP = dO\,V^\top,\qquad dS = P \odot (dP - D),\qquad dQ = dS\,K,\qquad dK = dS^\top Q
$$

所有项都含 $P$。标准实现把前向算出的 $P$ 存进 HBM，反向再读出来——又是 $N^2$ 个元素的一读一写。

FlashAttention 的选择：前向只存 $O$ 和 $(m,\ell)$（$O(N)$），反向把 $Q,K,V$ 的块重新搬进 SRAM，**现场重算 $S_{ij}$、$P_{ij}$**，用完再丢。本质是选择性梯度检查点，但别人用检查点是拿时间换显存，FlashAttention 反而更快——因为**重算一个块的 FLOPs 比从 HBM 读同样大小的数据便宜**。论文 Figure 2 的实测：GPT-2 medium（$N{=}1024$，$d{=}64$，16 头，batch 64），标准注意力 66.6 GFLOPs、40.3 GB HBM 读写、41.7 ms；FlashAttention 75.2 GFLOPs（+13%）、4.4 GB（−89%）、7.3 ms。多算一点点，少搬十倍，总时间反而只剩六分之一。

dropout 掩码同理：前向不存 $Z$，只存伪随机数生成器状态，反向重放一遍，$O(N^2)$ 的掩码矩阵也省了。

## ③ 融合：一个 kernel 一次读写

即使分块了，如果 mask、softmax、dropout、矩阵乘还是各自独立的 kernel，每个 kernel 都要把中间结果写回 HBM、下个 kernel 再读出来，分块省下的搬运又还回去了。所以第三步是把整条流水线**融合成一个 CUDA kernel**：数据进 SRAM，mask（把掩掉的位置设 $-\infty$）→ softmax → dropout → 乘 $V$ 全部在片上串完，只把最终结果写回 HBM。mask 和 dropout 这两个内存大头，从「两个独立的 $N\times N$ 往返」变成了「kernel 里顺手的两个步骤」。这也是为什么论文报告带 mask/dropout 时 FlashAttention 的加速比反而更大。

## 复杂度账本

| | FLOPs | HBM 访问 | 额外显存 |
|---|---|---|---|
| 标准注意力 | $\Theta(N^2 d)$ | $\Theta(Nd + N^2)$ | $\Theta(N^2)$（$S,P$） |
| FlashAttention | $\Theta(N^2 d)$（+重算） | $\Theta(N^2 d^2 / M)$ | $\Theta(N)$ |

为什么是 $N^2d^2/M$？SRAM 大小 $M$ 决定块大小 $B_c \approx M/4d$（$Q_i$、$K_j$、$S_{ij}$ 三块要同时放下），外层循环轮数 $T_c = N/B_c \approx 4Nd/M$，每轮要把所有 $Q$、$O$ 各搬一遍（共 $2Nd$ 个元素），总读写 $\Theta(Nd \cdot N/B_c) = \Theta(N^2 d^2 / M)$。代入 $N{=}1024,\ d{=}64,\ M{=}192\text{KB}$（fp16 约 $10^5$ 个元素）：$B_c \approx 384$，$T_c \approx 3$——整个训练要搬的量从 40 GB 级降到 4 GB 级，这就是 Figure 2 里 9 倍差别的来源。

更狠的是这个界**已经最优**：论文证明（命题 3）不存在精确注意力算法能在所有 SRAM 大小下都做到 $o(N^2d^2/M)$ 次 HBM 访问——FlashAttention 不是「一种」快法，是这条路上最快的。块稀疏版本（跳过全零块）把这个量再乘上稀疏比例 $s$，达到 $\Theta(Nd + sN^2d^2/M)$。

## 代码

原理结束。最小实现只要 15 行（按论文 Algorithm 1，省去 mask/dropout）：

```python
def flashattention(Q, K, V, Bc):
    N, d = Q.shape
    O = torch.zeros_like(Q)                # 输出，随块累加
    m = torch.full((N,), -float("inf"))    # 统计量：行最大值
    l = torch.zeros(N)                     # 统计量：指数和
    for j in range(0, N, Bc):              # 外层循环：K、V 一块块进 SRAM
        Kj, Vj = K[j:j+Bc], V[j:j+Bc]
        Sij = Q @ Kj.T                     # ① 这一块的分数（B_r×B_c，只在片上）
        m_new = torch.maximum(m, Sij.max(dim=1).values)   # 新最大值
        P = torch.exp(Sij - m_new[:, None])               # ① 块内概率
        l_new = torch.exp(m - m_new) * l + P.sum(dim=1)   # ① 旧和打折 + 新和
        O = torch.exp(m - m_new)[:, None] * O + P @ Vj    # ① 旧输出打折 + 新块贡献
        m, l = m_new, l_new
    return O / l[:, None]                  # 最后统一归一化
```

逐行认领：`Sij` 是 Algorithm 1 第 9 行；`m_new / l_new / O` 三行是第 11、12 行，`exp(m − m_new)` 就是「旧块整体打折」的缩放因子。注意这里 $P$ 直接用 $S_{ij} - m_{\text{new}}$，与论文的 $\exp(S_{ij} - \tilde m)$ 差一个 $\exp(\tilde m - m_{\text{new}})$ 系数——它被吸收进后面乘 $V_j$ 的系数里，两者等价（这也是论文把 $\tilde m - m^{\text{new}}$ 写进 $O$ 更新式的原因）。

随机规模对照（$N{=}64/128/100$，$d{=}16/32/24$，$B_c{=}8/16/7$ 三组，输出经 PyTorch 2.5 实测，与 `torch.softmax(Q@K.T)@V` 比较）：最大误差 $\sim 10^{-6}$，差异只是浮点求和顺序不同——**分块算法是精确算法，不是近似**。

## 总结

- **① 分块**：softmax 一行只需 $(m,\ell)$ 两个统计量即可合并；合并时旧账按 $e^{m-m_{\text{new}}}$ 打折、换算到新基准——$S$ 全程不出现在 HBM；
- **② 重计算**：反向用 $Q,K,V$ 块现场重算 $S,P$，多 13% FLOPs 换 90% 的搬运——因为搬（1.5 TB/s）比算贵；
- **③ 融合**：mask/softmax/dropout 装进一个 kernel，一次读写，省掉三个逐元素 kernel 的 $N^2$ 级往返；
- **账本**：FLOPs 不变（$\Theta(N^2d)$），HBM 访问从 $\Theta(Nd+N^2)$ 降到 $\Theta(N^2d^2/M)$，且该界最优；注意力数学一字未改，KV cache、RoPE、GQA 全部原样可用。

## 延伸

- **FlashAttention-2/3**：2 砍掉非矩阵乘的 FLOPs（归一化除法只做一次）、反向不再反复写 $dQ$、沿序列维度并行；3 用 Hopper 的 TMA/WGMMA 异步拷贝与 FP8，配合流水化把利用率拉满；
- **块稀疏**：论文同期提出的块稀疏 FlashAttention 跳过全零块，稀疏度 $s$ 直接乘进 IO 复杂度，LRA 上 2.8×；
- **落地形态**：PyTorch ≥2.0 的 `scaled_dot_product_attention` 在 Ampere+ 上默认就走 FlashAttention 后端；vLLM 的 PagedAttention 管理 KV cache 分页，底层计算同样靠它；
- **与本站其他文章联动**：RoPE 只旋转 Q、K，V 原样——分块计算与位置旋转天然不冲突（见 [RoPE 那篇](/blog/2026-08-26-rope-notes/)）；[GQA](/blog/2026-08-26-gqa-notes/) 省的是 KV 头数，FlashAttention 省的是读写，两者正交、常叠加出现。
