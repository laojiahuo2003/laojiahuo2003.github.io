---
title: RMSNorm 笔记：把 LayerNorm 拆到只剩骨头
description: 从公式到手写实现：RMSNorm 砍掉了均值和偏置，为什么效果反而没掉——附 draw.io 图解和可运行代码。
pubDate: 2026-08-25T22:30:00+08:00
tags: [模型结构]
---

读 Llama 的代码时我发现一个奇怪的事：`model.py` 里的归一化层短得离谱——没有均值，没有方差，没有偏置，就两行。这就是 RMSNorm，现代 LLM 的标配。这篇把它拆开：从哪来、为什么敢删、代码怎么写。

## 先看它删了什么

原始 Transformer 用 LayerNorm，公式长这样：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

RMSNorm 把它砍到只剩：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

删掉的三样：**均值 μ**（一次全量归约）、**方差 σ²**（依赖 μ 的第二次归约）、**偏置 β**（一份参数加一次加法）。剩下的 RMS 就是"均方根"——不用先减均值，直接对平方求平均再开方。

![LayerNorm 与 RMSNorm 的计算流程对比](/images/rmsnorm/rmsnorm_1_compare.png)

用一个 4 维向量手算一遍最直观。`x = [3.0, 1.0, -2.0, 4.0]`：

```text
LayerNorm:  μ=1.50, σ=2.29 → 先减均值再除 σ → [ 0.65, -0.22, -1.53,  1.09]
RMSNorm:    RMS=2.74       → 直接除 RMS    → [ 1.10,  0.37, -0.73,  1.46]（再乘 γ）
```

区别在输出分布：LayerNorm 强制输出零均值（正负各半），RMSNorm 只管缩放、不管中心（整体偏正）。**它能这么干的前提是：缩放比中心化重要得多。**

## 为什么敢删：一个反直觉的实验结论

RMSNorm 论文（Zhang & Sennrich, 2019）的核心发现有点反直觉：他们做了消融实验，把 LayerNorm 里"重新中心化"（re-centering，即 μ 和 β 那部分）整个去掉，各种任务上效果**基本不掉**。

这指向一个对归一化的重新理解：**归一化起作用主要靠"缩放不变性"（re-scaling），不是中心化**。直觉版本——归一化真正解决的问题是"深层网络里激活的方差会逐层漂移"，把每一层的输入强行拉回稳定尺度，梯度就稳了。至于输出中心是不是 0，模型并不在乎。

删掉之后的账很清楚：

- 少一次完整的数据依赖归约（μ），训练和推理都快一截；
- 参数少一半（只有 γ 没有 β）；
- 数值上更简单，fp16/bf16 训练更友好。

Llama 全系、Qwen、DeepSeek、Gemma……2023 年之后的开源模型几乎清一色 RMSNorm。这不是巧合，是"够用 + 更快 + 更简单"的合力。

## 手写实现：两行核心代码

公式到代码的映射几乎是直译：

![公式与代码的逐行对照](/images/rmsnorm/rmsnorm_3_code.png)

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # γ，初始化为全 1

    def forward(self, x):                             # x: (B, S, d)
        # ① ms = mean(x²)，keepdim 保持 (B,S,1) 以便广播
        # ② rsqrt = 1/√(ms+ε)，一步完成除法
        # ③ 乘回 γ
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) \
               * self.weight
```

三个实现细节值得停一下：

**`keepdim=True` 是广播的关键。** `x` 是 `(B, S, d)`，归约后在最后一维得到 `(B, S, 1)`，和原形状相乘时自动广播到每个维度——一行完成"每个 token 用自己的 RMS 归一化自己"。

**`rsqrt` 而不是 `1/sqrt`。** `torch.rsqrt` 是融合算子，一次 kernel 完成平方根求导，比先 `sqrt` 再除快（fp16 下数值也更稳）。

**γ 初始化为全 1。** 训练开始时 RMSNorm 是"恒等变换 + 缩放校正"，不扰动初始表征——这是它训练稳定的一部分。

前向的完整数据流（张量形状怎么变）：

![RMSNorm 前向数据流与形状变化](/images/rmsnorm/rmsnorm_2_forward.png)

## 验证：跑一遍确认性质

光看代码不够，三个性质跑一遍才算数（`nn.RMSNorm` 要 PyTorch 2.4+，手写版不受限制）：

```python
torch.manual_seed(0)
x = torch.randn(2, 16, 256) * 3 + 1     # 故意做成非零均值

norm = RMSNorm(256)

# ① 缩放不变性：输入整体乘 k，输出不变——归一化的本职
print(torch.allclose(norm(x), norm(x * 7), atol=1e-5))   # True

# ② 输出的 RMS ≈ γ（初始 γ=1，每个 token 的输出 RMS 被锁在 1）
rms = norm(x).pow(2).mean(-1).sqrt()
print(rms.mean().item())    # ≈1.0

# ③ 平移敏感性：加常数后输出会变——这就是"没做中心化"的代价
print(torch.allclose(norm(x), norm(x + 5), atol=1e-5))  # False
```

第 ① 条就是"缩放不变性"的直接体现；第 ② 条说明输出尺度被严格锁定，深层网络里激活不会漂；第 ③ 条诚实地记下它**没**解决的问题——整体平移会穿透到输出，好在残差流里"所有 token 同加一个常数"的情形极少，模型用 γ 和后面的层把它吸收掉了。

## 它在 Llama 里的位置：pre-norm

光有 RMSNorm 还不够，放的位置同样关键。Llama 系列全部用 **pre-norm**：归一化放在子层**前面**，外面套残差：

```python
x = x + attn(rmsnorm(x))    # 先归一化再进注意力，输出加回残差流
x = x + ffn(rmsnorm(x))
```

对比原始 Transformer 的 post-norm（`x = norm(x + sublayer(x))`）：pre-norm 的残差通路是"干净"的——梯度可以沿 `x + ...` 这条加法直通底层，不经过归一化的摩擦。这就是深模型（几十上百层）能训稳的原因。

代价是Representation 上输出等效地"没被归一化"，所以 Llama 在最后一层后还补了一个 `norm`（最终的 RMSNorm）再接 `lm_head`——手写 Llama 时很容易漏掉这个。

顺带一提 Llama 全面去掉 bias 的原因：RMSNorm 的 γ 是乘性缩放，已经覆盖了"调整每维幅度"的需求，加性 β 显得多余；而且 bias 是唯一"不需要看输入也有梯度"的参数，长训练里单调漂移，还是量化（INT8/FP8）时激活异常的惯犯。**归一化优于加偏置**——这个思想后来在 Qwen3 的 QK-Norm 上又出现了一次（[QK-Norm 那篇] 里 Qwen 删掉 QKV-bias 换 QK-Norm 是同一个故事）。

## 记忆钩子

- **RMSNorm = LayerNorm 减去中心化**：不减均值、不除标准差，只除均方根，只乘 γ；
- **敢删的依据**：归一化的价值在缩放不变性，中心化是可有可无的装饰（消融实验撑腰）；
- **pre-norm + RMSNorm** 是现代 LLM 的默认骨架，最后一层后还有个容易漏的 final norm；
- 手写就两行：`x * rsqrt(mean(x²)+ε) * γ`。

## 下一步

想顺着这条线做两件事：把 RMSNorm 换回 LayerNorm 训一个小 nanoGPT 看训练曲线差异；以及看看 FlashRMSNorm 这类 fused kernel 把这两行代码又优化了什么（归约 + 缩放的融合，省一次读写）。
