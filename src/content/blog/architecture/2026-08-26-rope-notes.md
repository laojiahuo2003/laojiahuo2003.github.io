---
title: RoPE：位置不是加进去的，是旋转出来的
description: 从"注意力该知道两个词隔多远"到"把位置变成旋转角度"——RoPE 的复数推导、rotate_half 的实现技巧、频率表的秒针分针直觉，以及为什么只旋转 Q 和 K。
pubDate: 2026-08-26T00:50:00+08:00
tags: [模型结构]
---

上一篇讲 RMSNorm 时说过：2023 年后的开源模型在归一化层上做出了同一个选择。位置编码也是——Llama、Qwen、DeepSeek、Gemma 全部使用 RoPE。但它和 RMSNorm 的"做减法"不同，RoPE 换了一个提问方式：不问「位置 m 的向量应该长什么样」，而问「位置 m 的 Q 和位置 n 的 K 做内积时，会发生什么」。

这一问的差别是本质性的。前者的答案里位置是一个需要**加进去**的东西（可学习位置向量、原始 Transformer 的 sin/cos 相加都是这条路）；后者的答案里位置是一个**旋转角度**——什么都不用加，把 Q、K 转个角度，内积里就自动只剩相对距离。

这篇文章回答三个问题：相对位置为什么难、旋转是怎么把它变简单的、以及三十行代码怎么写出来。

## 前置知识：公式里的每个符号

| 符号 | 是什么 | 说明 |
|---|---|---|
| $m,\ n$ | token 的位置下标 | 第几个 token，从 0 数起 |
| $d$ | 单个注意力头的维度 head_dim | RoPE 逐头作用，和整个模型的隐层宽度无关 |
| $\theta_i$ | 第 $i$ 对维度的「转速」 | $\theta_i = 10000^{-2i/d}$，直接写死的常数 |
| $R(m\theta)$ | 二维旋转矩阵 | 把平面上的向量逆时针转 $m\theta$ 弧度 |
| $\langle q, k\rangle$ | 内积（点积） | 注意力打分 $q\cdot k^\top$ 的核心运算 |

RoPE 全文没有可学习参数——位置信息全部来自 $\theta_i$ 这张写死的频率表。

## 相对位置为什么难

注意力本身对顺序是无感的：打分就是内积，把输入 token 打乱，每个位置的输出也只是跟着乱，不会变错。位置编码是补上这个缺口的外挂。

最直接的外挂是**加性绝对位置**——给每个位置学一个向量 $p_m$，加到输入上。问题在内积展开的那一刻暴露：

$$
\langle q + p_m,\ k + p_n\rangle = \underbrace{q\cdot k}_{\text{内容}} + \underbrace{q\cdot p_n + p_m\cdot k + p_m\cdot p_n}_{\text{全都绑定绝对位置}}
$$

后面三项让「同一个词对，隔 3 个 token 相遇还是隔 300 个」这件事，取决于它们出现在句子的第几位。而我们真正想要的性质是**平移不变**：上下文窗口整体滑动，注意力模式不该变——打分只该依赖 $n - m$。

直接把 $n - m$ 写进注意力公式（T5 的相对 bias、Transformer-XL）确实可行，但它们要么改注意力结构、要么引入随长度增长的 bias 表，和 KV cache、FlashAttention 这些"把标准注意力做快"的工程成果都不对付。

RoPE 的漂亮之处：**标准注意力一个字不改，只把 Q、K 预先旋转一个角度**，平移不变自动成立。

## 旋转是怎么把它变简单的

把想要的东西写成方程：找一个带位置的变换 $f$，使得内积与绝对位置脱钩——

$$
\big\langle f(q, m),\ f(k, n) \big\rangle \;=\; F\big(q,\ k,\ m-n\big)
$$

$F$ 只通过内容 $q, k$ 和相对距离 $m - n$ 决定。RoPE 的解法分两步：先在二维里找到 $f$，再拼到高维。

二维里推导。把一对维度 $(x_1, x_2)$ 看成一个复数 $x_1 + i\,x_2$，定义位置变换「乘上 $e^{im\theta}$」——复数乘法就是旋转。验证内积：

$$
\big\langle q\,e^{im\theta},\ k\,e^{in\theta} \big\rangle
= \mathrm{Re}\Big[ q\,e^{im\theta} \cdot \overline{k\,e^{in\theta}} \Big]
= \mathrm{Re}\Big[ q\bar{k}\ e^{i(m-n)\theta} \Big]
$$

绝对位置 $m$、$n$ 消失了，只剩 $m - n$。这就是全部的魔法。

高维推广是把 $d$ 维切成 $d/2$ 个二维子空间，各自用不同转速 $\theta_i$ 旋转，写成矩阵即分块对角：

$$
R(m\theta) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix},
\qquad
R_m = \mathrm{diag}\big(R(m\theta_1),\ \dots,\ R(m\theta_{d/2})\big)
$$

写开到分量上（原论文的相邻配对形式，一对维度 $(2i,\ 2i+1)$，记旋转后的向量为 $\tilde{q} = R_m q$）：

$$
\begin{aligned}
\tilde{q}_{2i}   &= q_{2i}  \cos m\theta_i - q_{2i+1} \sin m\theta_i \\
\tilde{q}_{2i+1} &= q_{2i+1}\cos m\theta_i + q_{2i}   \sin m\theta_i
\end{aligned}
$$

「旋转」二字的全部含义就在这两行：**每对维度在自己的平面上转 $m\theta_i$，不同对转速不同**。

成立的关键是旋转矩阵的正交性 $R(m\theta)^\top R(n\theta) = R\big((n-m)\theta\big)$，于是：

$$
\langle R_m q,\ R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m}\, k
$$

**左边的绝对位置进去，右边的相对位置出来。** 放进完整的注意力打分公式（softmax 在 $n$ 上做，$\sqrt{d}$ 照常缩放）：

$$
\mathrm{score}(m, n) \;=\; \frac{\tilde{q}_m \cdot \tilde{k}_n}{\sqrt{d}} \;=\; \frac{q^\top R_{n-m}\, k}{\sqrt{d}}
$$

位置信息全部内嵌在 $R_{n-m}$ 里，注意力公式没有任何附加项——KV cache、FlashAttention 原样可用。

## 几何图像

把上面的代数画出来。每个二维子空间是一张平面，位置编码就是平面上的旋转量：位置 $m$ 的 Q 转过 $m\theta_i$，位置 $n$ 的 K 转过 $n\theta_i$，两者夹角永远是 $(n-m)\theta_i$——无论这对词出现在句首还是句尾。

![RoPE 的几何图像：位置变成旋转角度，夹角只依赖相对距离](/images/rope/rope_1_rotation.png)

「不同转速」是这套设计里最值得细看的部分。转速和波长由两条公式定死：

$$
\theta_i = 10000^{-2i/d}, \qquad \lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot 10000^{\,2i/d}
$$

指数衰减：$i$ 越小的维度对转得越快、波长越短。以 Llama 的 $d = 128$ 为例——

- 最快的一对：$\theta_0 = 1$，波长 $2\pi \approx 6.3$ 个 token，相邻 token 之间就转了大半圈，负责分辨「贴着的词」；
- 最慢的一对：$\theta_{63} \approx 1.2\times10^{-4}$，波长约 5.4 万个 token，几千个 token 内几乎不动，负责分辨「隔着很远的词」。

像时钟：秒针转得快、时针转得慢，快慢针的读数组合起来唯一确定时刻；这里则是 $d/2$ 根转速不同的针，唯一确定相对距离 $n-m$。单一的 $\theta$ 做不到这件事——两根同速的针读不出两个独立的角度。

顺带一提，Llama3 把 base 从 $10^4$ 提到 $5\times10^5$，就是在「把最慢的针弄得更慢」：波长拉长，长上下文里相对距离才不容易转满一圈撞车。

## 三十行实现

实现在 `llama2/model.py` 里，三个函数。第一个把 cos/sin 表预先算好——旋转角只依赖 $(m, i)$，和内容无关，训练开始前算一次，缓存成 buffer：

```python
def precompute_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)        # (S, d/2)：角度 m·θ_i
    emb = torch.cat((freqs, freqs), dim=-1) # (S, d)
    return emb.cos(), emb.sin()
```

第二个函数是整个实现里最巧的一步。朴素做法要构造 $d\times d$ 的稀疏旋转矩阵再做矩阵乘——又大又慢。观察到 $R_m$ 分块对角、每块只有 cos/sin，可以把乘法重排成逐元素乘加：

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """x = [x1, x2] → [-x2, x1]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)     # (S,d) → (1,1,S,d) 广播
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

写开验证一下（向量前半 $x$、后半 $y$，逐对 $(x_i, y_i)$ 配对）：

$$
\begin{aligned}
\text{RoPE}(v)_{\text{前半}} &= x\cos m\theta_i - y\sin m\theta_i \\
\text{RoPE}(v)_{\text{后半}} &= y\cos m\theta_i + x\sin m\theta_i
\end{aligned}
$$

正是旋转矩阵的每一行——`rotate_half` 把「配对维度交叉」压缩成一次 `chunk + cat`，稀疏矩阵乘变成了逐元素乘加，`v * cos + rotate_half(v) * sin` 一行完成。

![RoPE 在一层注意力里的位置：只作用于 Q 和 K，V 直接通过](/images/rope/rope_2_flow.png)

两个工程细节值得单独说：

**只旋转 Q 和 K，V 原样通过。** 位置信息的唯一用途是参与打分；V 是被打分加权的内容本身，旋转它不提供任何位置语义，反而破坏内容表示。这就是图里 V 那条路绕过旋转节点的原因。

**配对方式有两种，数学等价、权重不互通。** 本文实现是 GPT-NeoX/HF-Llama 风格（前半与后半配对：$(x_i, x_{i+d/2})$）；原始论文和 GPT-J 是相邻配对（$(x_{2i}, x_{2i+1})$）。两者只差一个维度置换，forward 数学上等价——但预训练权重里的维度排列跟着各自风格走，加载第三方权重时用错了配对，模型会静悄悄地输出乱码。这是手写实现里最典型的「跑通但不对」陷阱。

## 验证三个性质

代码写完不算完，三个性质各验一遍：

```python
import torch
# rotate_half / precompute_rope_cache 来自上文，cos/sin 形状 (S, d)

q = torch.randn(8)                        # 一个 head_dim=8 的头
k = torch.randn(8)
cos, sin = precompute_rope_cache(64, 8)

def rot(x, m):                            # 把 x 放到位置 m 旋转
    return x * cos[m] + rotate_half(x) * sin[m]

# ① 相对性：绝对位置平移，打分不变
a = torch.dot(rot(q, 5),  rot(k, 15))     # 位置 (5, 15)
b = torch.dot(rot(q, 25), rot(k, 35))     # 位置 (25, 35)，相对距离都是 10
torch.allclose(a, b, atol=1e-5)           # True

# ② 保范数：旋转不改变向量长度，不干扰归一化后的尺度
torch.allclose(rot(q, 9).norm(), q.norm(), atol=1e-6)   # True

# ③ 同位置退化：m = n 时打分还原为纯内容内积
torch.allclose(torch.dot(rot(q, 7), rot(k, 7)), torch.dot(q, k), atol=1e-5)  # True
```

① 是设计目标本身；② 说明 RoPE 不破坏 RMSNorm 辛苦维持的尺度稳定——旋转是正交变换，范数天然不变；③ 是 ① 的边界情形，也是检查配对风格有没有写对的最快方法（配对错了，同位置的内积不会还原）。

## 总结

- RoPE 的提问方式：不设计「位置 m 的向量长什么样」，而是设计「带位置的 Q·K 内积长什么样」——答案要求只依赖 $n-m$；
- 二维复数乘 $e^{im\theta}$ 即旋转，内积自动消掉绝对位置；高维 = $d/2$ 个子空间各自转，分块对角矩阵；
- $\theta_i = 10000^{-2i/d}$ 指数衰减，快慢针组合唯一编码相对距离；
- 实现核心是 `rotate_half`：把稀疏旋转矩阵乘重排成逐元素乘加；
- 只旋转 Q、K；配对风格（前后半 vs 相邻）等价但权重不互通。

## 延伸

RoPE 把「相对位置」做对了，但「训练 4K、推理 128K」的外推问题它自己不解决——位置超出训练见过的范围，慢针也开始转满圈。沿这条路长出来一整族长度外推方法：位置插值（PI）把位置线性压缩回训练范围，NTK-aware 调高 base 让慢针更慢，YaRN 两者的非线性组合；Llama3 则干脆把 base 提到 $5\times10^5$ 用长数据重训。另一个方向是把这篇的验证代码扩展成可视化：画出不同 $i$ 的 $\cos(m\theta_i)$ 曲线，秒针、分针、时针的比喻会直接长在眼前。
