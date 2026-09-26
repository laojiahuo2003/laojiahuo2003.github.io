---
title: 多轮对话指令微调：Loss Mask 应该遮住哪些 token
description: Instruction Tuning 用多轮对话数据训练时，损失不能平均摊在每个 token 上：模型只该学 assistant 的输出。本文讲清 Loss Mask 的四个决策点（用户轮、system prompt、assistant 轮与 EOS、角色标记），以及实现上最容易翻车的对齐与逐段拼接。
pubDate: 2026-09-26T23:59:00+08:00
tags: [后训练, 训练工程]
---

## 摘要

Instruction Tuning（指令微调，SFT 的一种）用对话数据训练时，一条样本是完整的 user/assistant 多轮对话。但损失不能平均摊在每个 token 上——模型只该学 assistant 的输出。Loss Mask 就是决定"哪些 token 算损失"的那个 0/1 遮罩。本文讲清四件事：为什么必须 mask、四个决策点各怎么遮、实现上最容易翻车的对齐与逐段拼接、以及常见变体。

## 0. 问题：为什么不能整条序列都算损失

语言模型的训练目标是最大化"下一个 token"的概率。一条多轮对话样本里混着两种角色的话：user 轮是**输入**，assistant 轮是**输出**。如果对整条序列算损失，模型会被训练成"预测用户接下来要说什么"——既学了不该学的（替用户说话、自问自答），又稀释了真正要学的信号。

> **要点**：mask 的本质，是把"整条对话都预测"改写成"只预测 assistant 说过的话"。

形式上，损失只在 mask 为 1 的位置计算，归一化也只用这些位置：

$$
\mathcal{L} = -\frac{1}{\sum_t m_t}\sum_t m_t \log P_\theta(y_t \mid y_{<t}), \qquad m_t \in \{0, 1\} \tag{1}
$$

先用大白话翻译式 (1)：$m_t=1$ 的 token 才进损失，$m_t=0$ 的位置完全不参与。实现上不显式构造 $m$，而是把该遮位置的 label 置成 ignore 值（PyTorch 里是 -100），交叉熵自动跳过。

## 1. 基本原则：只学 assistant 的输出

一条两轮对话的 token 流和它对应的 mask 长这样（■ = 算损失，□ = 遮掉）：

```
 system  │  user₁  │   assistant₁     │  user₂  │   assistant₂    │ <eos>
 □□□□□□ │ □□□□□□ │ ■■■■■■■■■■■■■■ │ □□□□□□ │ ■■■■■■■■■■■■■■ │   ■
```

看一个具体场景：user 问"推荐三本 RL 入门书"，assistant 列出三本；user 追问"哪本最适合初学者"，assistant 回答。这条样本里只有两个 assistant 段和最后的 `<eos>` 参与损失——其余全部遮掉。

可以把它记成：对话数据像一份"剧本"，模型要学的是"台词"（assistant 的话），不是"对手的台词"（user 的话），更不是"舞台说明"（system prompt）。

## 2. 四个决策点

> **要点**：用户轮遮、system prompt 遮、assistant 内容留、EOS 必须留；角色标记默认遮。

### 2.1 用户轮：全遮

两个理由。第一，user 轮是输入，让模型学生成输入等于教它自问自答——推理时它可能先把用户的话"脑补"出来再回答。第二，信号稀释：损失摊到无关 token 上，等价于悄悄降低了有效学习率。主流对齐实践（InstructGPT 的 SFT、LIMA 等）都是只对 assistant 输出算损失。

### 2.2 system prompt：遮

system prompt 是给模型的"身份设定"，由数据管线在推理时注入，不靠模型自己生成。训练时让它学预测这段文字没有意义，还占损失。

### 2.3 assistant 轮：保留，且 EOS 必须保留

内容 token 全部保留，但 **EOS 是最容易被误遮的**。如果把 `<eos>` 也遮掉，模型就学不到"什么时候该结束"，推理时容易长篇大论停不下来。可以把它记成：EOS 是刹车，训练时不让模型踩刹车，推理时它就只会踩油门。

反方向的实践也存在——个别实现会故意遮 EOS 让回答更长，但那是干预模型行为的特殊手段，默认值永远是保留。

### 2.4 角色标记：两派，默认遮

chat template 会在每轮前加结构 token，比如 `<|im_start|>assistant\n`。这些"角色头"算不算 assistant 输出，有两派做法：

- **遮掉，只学内容**：模型不需要学格式——格式由推理时的 template 保证。这是多数开源 recipe 的默认路线，实现也干净。
- **保留，连格式一起学**：让模型学会自己输出结束标记，某些自定义对话格式下有益，但学格式会占用容量，也容易学到格式噪声。

建议：默认遮角色头、留内容、留 EOS，并且和推理侧用的 template 保持一致——推理时注入什么格式，训练时就只学什么之外的部分。

## 3. 实现细节：两个最容易翻车的坑

> **要点**：labels 与输入对齐后交给框架错位预测，不要手工平移；轮边界要在逐段 tokenize 时记在下标上。

### 3.1 对齐约定：mask 和"预测目标"差一个位置

标准做法（HuggingFace 系）：`labels = input_ids` 原样复制，把"该 token 不属于 assistant 输出"的位置置 -100；错位由框架内部完成（`logits[:-1]` 对 `labels[1:]`），**不需要手工平移**。看一个简化例子：

```
input_ids:  [SYS, u1, u2, a1, a2, a3, <eos>]
labels:     [-100, -100, -100, a1, a2, a3, <eos>]   ← 与输入逐位对齐；框架内部自行错位
```

位置 $t$ 的 logits 预测的是 labels 的第 $t+1$ 个 token：第一个 assistant token `a1` 恰好由最后一个 user token `u2` 的位置预测，`<eos>` 由 `a3` 的位置预测——语义上完全正确。

**坑**：如果你手写的损失自己做了平移，框架内部又平移一次，就会错位一格。症状很有辨识度：对话开头或结尾的衔接学不会，或者 user 轮的最后一个 token 漏进了损失。自查方法：数一数非 -100 的 label 数量，必须恰好等于所有 assistant 内容 token 加 EOS 的总数。

### 3.2 逐段 tokenize 再拼接：轮边界记在下标上

`apply_chat_template` 一次 tokenize 整条对话很方便，但事后想找回"哪段是 assistant"只能靠内容重新定位，容易出错。可靠的做法是**逐条 message 单独 tokenize，把边界记在 token 下标上**：

```
for msg in messages:
    tok = tokenizer.encode(msg, add_special_tokens=False)
    span = (start, start + len(tok))          # 记下这段的起止下标
    labels[span] = 保留 if msg.role == "assistant" else -100
    start += len(tok)
```

大白话：别事后靠猜，拼的时候就把每段是谁的话记下来。

### 3.3 截断与 padding

两个容易漏的边角：

- **超长对话截断**：优先保留最近几轮（与最后的问题相关性最强），截断后重算 labels 区间，别让 mask 和截断后的序列错位。
- **padding**：batch 内右 padding 的位置必须置 -100，否则 padding token 也进损失——这是教科书级常识，但也是教科书级的高频翻车点。

## 4. 常见变体

- **completion-only（只学最后一轮 assistant）**：某些单轮评测或 RL 流水线里会用。对多轮 SFT 来说信号太少——标准做法是所有 assistant 轮都学。
- **不 mask，全序列都算损失**：早期 naive SFT 的坑。会让模型学预测 user 的话，等价于让模型同时扮演两个角色。对齐工作（InstructGPT、LIMA）都是明确只对输出算损失。

## 5. 结论：一张 checklist

- user 轮：**遮**
- system prompt：**遮**
- assistant 内容：**留**
- `<eos>`：**留**（刹车不能遮）
- 角色标记头：**默认遮**，与推理 template 保持一致
- 实现：labels 与输入对齐、框架内部错位；逐段拼接记边界；截断后重算；padding 遮掉
- 自查：非 -100 的 label 数 = 所有 assistant 内容 token + EOS 数

## 参考文献

1. Ouyang et al. *Training language models to follow instructions with human feedback.* arXiv:2203.02155.
2. Wei et al. *Finetuned Language Models Are Zero-Shot Learners.* arXiv:2109.01652.
3. Zhou et al. *LIMA: Less Is More for Alignment.* arXiv:2305.11206.
4. Ding et al. *Enhancing Chat Language Models by Scaling High-quality Instructional Conversations.* arXiv:2305.14233.
