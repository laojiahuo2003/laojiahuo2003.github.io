---
title: GRPO 变体谱系：DAPO、GSPO、GFPO 各自改了什么
description: 以 GRPO 为基准，沿「奖励作用粒度、采样策略、裁剪设计」三条轴拆解 DAPO、GSPO、GFPO：DAPO 打四个工程补丁，GSPO 把比率升级到序列级，GFPO 过滤掉不值得学的响应。
pubDate: 2026-09-26T21:30:00+08:00
tags: [强化学习, 后训练]
---

## 摘要

GRPO 之后涌现了大量变体——DAPO、GSPO、GFPO、Dr. GRPO、VAPO……名字五花八门，但改动其实都围绕同一个基准留下的三个问题：

1. **奖励作用在哪一层**：序列级标量该怎么落到每个 token 上？
2. **怎么采样**：哪些组、哪些响应值得花算力去训练？
3. **怎么限制更新**：裁剪该作用在 token 上，还是整条回答上？

本文先立 GRPO 基准（流程图 + 四层粒度表 + 三轴框架），再逐一拆解 DAPO、GSPO、GFPO 的改动与动机，最后给出三轴对照总表与选型建议。

## 0. 基准：GRPO 的四层粒度与三条轴

### 0.1 核心流程

Group Relative Policy Optimization（组相对策略优化）的流程非常朴素：

```
一个 prompt
   │
   ├── response 1 → reward 8
   ├── response 2 → reward 6
   ├── response 3 → reward 2
   └── response 4 → reward 4
             │
             ▼
       组内计算 Advantage
             │
             ▼
       PPO-style Clip Loss
```

同一提示采样 $G$ 个回答，各自打分，组内算相对优势，再套 PPO 式的裁剪损失。

### 0.2 四层粒度

GRPO 的典型实现里，四个关键量分布在两个粒度上：

| 量 | 粒度 |
|---|---|
| Reward $r_i$ | sequence-level |
| Advantage $\hat{A}_i$ | sequence-level |
| Ratio $\rho_{i,t}$ | token-level |
| Loss $\ell_{i,t}$ | token-level 计算，再做 reduction |

对应的公式是

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})} \tag{1}
$$

$$
\ell_{i,t} = \min\big(\rho_{i,t}\hat{A}_i,\ \ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon,\,1+\varepsilon)\cdot\hat{A}_i\big) \tag{2}
$$

逐项说明：奖励 $r_i$ 在整条回答生成完毕后给出，一个标量；组内优势 $\hat{A}_i=(r_i-\mathrm{mean})/\mathrm{std}$ 也是纯标量运算，一条回答共享一个值；重要性比率 $\rho_{i,t}$ 在每个 token 位置单独计算；裁剪项 $\ell_{i,t}$ 逐 token 组装，最后做归一化（除以 $1/\lvert y_i\rvert$ 或整批 token 总数）。

注意这张表里的**粒度错配**：奖励和优势都是序列级的，比率却是 token 级的。这条裂缝是后面 GSPO 的立论起点。

### 0.3 三条轴

后文所有变体，本质上都在回答三个问题：

- **粒度轴**：序列级优势 × token 级比率这个错配，要不要修？怎么修？
- **采样轴**：一组采样可能全对、全错、或混着冗长重复的响应——算力花在哪？
- **裁剪轴**：裁剪上下界对称吗？作用于 token 还是整条回答？

可以把它记成：GRPO 是一间毛坯房，每个变体都在装修不同的房间——修粒度、修采样、修裁剪。带着这三个问题读下文，每个变体的动机就会自己浮出来。

## 1. DAPO：GRPO 的「大规模长 CoT 工程增强版」

DAPO = Decoupled Clip and Dynamic Sampling Policy Optimization（Yu et al., arXiv:2503.14476）。它的目标很具体：让 GRPO 能在大规模长思维链（long CoT）训练中稳定工作。

可以把它记成：

> **DAPO = GRPO + 四个改动：Clip-Higher + Dynamic Sampling + Token-level Loss + Overlong Reward Shaping**

官方论文把这四项明确列为核心技术，逐一拆开看。

### 1.1 第一招：Clip-Higher（解耦裁剪）

普通 GRPO 的裁剪上下界对称：

$$
\rho \in [\,1-\varepsilon,\ 1+\varepsilon\,]
$$

上下界共用同一个 $\varepsilon$。DAPO 把它解耦成两个独立的量：

$$
\rho \in [\,1-\varepsilon_{\mathrm{low}},\ 1+\varepsilon_{\mathrm{high}}\,],\qquad \varepsilon_{\mathrm{high}}>\varepsilon_{\mathrm{low}}
$$

```
GRPO:

        0.8        1.0        1.2
---------|----------|----------|
        下限                  上限

DAPO:

        0.8        1.0              1.28
---------|----------|----------------|
        下限                       更大的上限
```

为什么要把上限放宽？考虑一个**低概率但可能很好的 token**：

```
old policy：概率很低
       ↓
new policy：希望明显提高它的概率
```

如果上限卡得太死，一次更新最多只能把它的概率抬一点点，模型探索新行为的空间就比较小。DAPO 把裁剪上限放宽一些，希望达到三个目的：

- **增加探索**；
- **防止 entropy collapse（熵坍缩）**——所有 token 的概率过早锁死；
- 让有价值的低概率行为更容易被强化。

回顾裁剪的机制有助于理解为什么这招管用：裁剪是单边惩罚——优势为正、比率超过上限时梯度截断为零（停止鼓励）；优势为负、比率跌破下限时梯度截断为零（停止压制）。放宽上限，等于把"鼓励侧"的刹车点放得更远，好行为可以更用力地涨。

### 1.2 第二招：Dynamic Sampling（动态采样）

这是 DAPO 相当重要的一招。假设一个 prompt 的四个回答全对：

```
response 1 → 对
response 2 → 对
response 3 → 对
response 4 → 对

reward: 1, 1, 1, 1
```

那么组内均值 $\mu=1$，所有优势

$$
\hat{A}_i = \frac{r_i-\mu}{\sigma} = 0
$$

于是

$$
\nabla\mathcal{L} \approx 0
$$

**这个 prompt 对训练基本没有贡献。** 反过来全错（`0, 0, 0, 0`）也一样——优势同样全为零。可是采样这四个回答的算力已经花掉了。

DAPO 的 **Dynamic Sampling（动态采样）**在训练时把这种组过滤掉：

```
全对 → 0 梯度   ← 跳过
全错 → 0 梯度   ← 跳过
```

而去寻找

```
对、对、错、错   ← reward variance > 0，能形成有效的相对优势
```

可以把它记成：**DAPO 不想把算力浪费在"大家都会"或"大家都不会"的题上，而更关注"有人做对、有人做错"的题。**

### 1.3 第三招：Token-level Loss（token 级损失归一化）

GRPO 的损失是序列级平均：每条回答的损失先除以自己的长度 $1/\lvert y_i\rvert$。这带来一个副作用——**长回答的每个 token 分摊的梯度被稀释，模型学会"写长摊薄风险"**，训练中输出长度单调膨胀。

DAPO 的 Token-level Loss 把归一化分母换成**整批的 token 总数**：

$$
\mathcal{L}_{\mathrm{DAPO}}=\frac{1}{\sum_{i=1}^{G}\lvert y_i\rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert y_i\rvert}\min\big(\rho_{i,t}\hat{A}_i,\ \ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\cdot\hat{A}_i\big) \tag{3}
$$

每个 token 平权，长回答不再享有"风险稀释"的福利。注意式 (3) 里的裁剪同时用上了第一招的 $\varepsilon_{\mathrm{low}}/\varepsilon_{\mathrm{high}}$——四个改动不是孤立的，是相互咬合的。

### 1.4 第四招：Overlong Reward Shaping（超长奖励塑形）

长 CoT 训练里，模型会学会用冗长换取正确率：写 2000 个 token 把答案包出来。DAPO 直接在奖励上动手：**回答长度超过预设上限时，从奖励中扣除一个与超出长度成比例的惩罚项**，从源头抬高"写太长"的成本。

可以把它记成：前三招管"怎么学"，这一招管"别学歪"——写太长，直接罚。

### 1.5 小结

DAPO 没有改动 GRPO 的理论结构——优势仍是组内标准化，比率仍是 token 级，裁剪仍是 PPO 式——它做的是**把大规模训练中暴露的四个具体故障逐个修掉**。它的局限也在这里：四个改动各管一摊，互相之间没有统一的理论框架，更像一张经验修补清单。

## 2. GSPO：把比率也升级到序列级

GSPO = Group Sequence Policy Optimization（Zheng et al., arXiv:2507.18071，阿里 Qwen 团队）。如果说 DAPO 是打补丁，GSPO 就是动结构——它直指 §0.2 那张粒度表里的错配。

### 2.1 动机：粒度错配

回到粒度表：奖励是序列级，优势是序列级，**比率却是 token 级**。GSPO 的出发点：既然奖励和优势都定义在整条回答的层面，重要性比率也应该是序列级的——用整条回答的概率比，而不是每个 token 的概率比。

错配的实际后果：token 级裁剪会把"这条回答该更新多少"这个序列级决策，在每个 token 位置上分别截断一遍。一条回答里不同 token 的比率差异巨大（有的 0.5、有的 3.0），逐 token 裁剪后的总更新量，既不对应任何序列级的意图，也不对应任何序列级目标——**每一步的裁剪都是对的，合起来却不知道在优化什么**。

### 2.2 序列级比率与几何平均平滑

理论上精确的序列级比率是整条回答的概率比：

$$
R_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}=\prod_{t=1}^{\lvert y_i\rvert}\rho_{i,t} \tag{4}
$$

它的问题是**方差**。连乘在对数空间里是 $\lvert y_i\rvert$ 个随机项之和，方差随长度线性增长；取回指数后分布极度长尾——长回答的比率动辄爆炸或趋零，序列级裁剪根本无从下手。

GSPO 的解法是**几何平均平滑**：把连乘换成对数平均再取指数，即 token 比率的几何平均：

$$
s_i=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}\right)^{1/\lvert y_i\rvert}
=\exp\!\left(\frac{1}{\lvert y_i\rvert}\sum_{t=1}^{\lvert y_i\rvert}\log\rho_{i,t}\right) \tag{5}
$$

极端 token 比率被压缩回 1 附近，$s_i$ 随长度稳定，不炸不灭。

可以把它记成：连乘是"每个 token 的变化全部叠乘"，几何平均是"整条回答的平均变化幅度"——前者被一个极端 token 带飞，后者只看整体趋势。

### 2.3 序列级裁剪

比率升级后，裁剪也跟着升级到序列级：

$$
\mathcal{J}_{\mathrm{GSPO}}=\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\min\big(s_i\,\hat{A}_i,\ \ \mathrm{clip}(s_i,\,1-\varepsilon,\,1+\varepsilon)\cdot\hat{A}_i\big)\Big] \tag{6}
$$

注意裁剪对象从 $\rho_{i,t}$（每个 token 一个）变成了 $s_i$（每条回答一个标量）：**整条回答共享同一个更新幅度，裁剪只做一次。** 优势 $\hat{A}_i$ 仍沿用组内标准化。

粒度表的变化一目了然：

| | GRPO | GSPO |
|---|---|---|
| Reward | sequence | sequence |
| Advantage | sequence | sequence |
| Ratio | **token** | **sequence** |
| Loss | token 级计算再 reduction | **sequence 级** |

四层粒度全部对齐到序列级。GSPO 名字里的 **S**equence 就来自这里——它修的是三轴中的**粒度轴**。

### 2.4 代价

- **位置信息进一步丢失**：GRPO 好歹比率还是 token 级的，几何平均之后连比率都不带位置了。"哪个 token 导致了好结果"这个信用分配问题依然无解——GSPO 的态度是：既然序列级优势本来就不带位置信息，token 级比率的位置信息也只是添乱，不如一起抹平；
- **平滑压制更新幅度**：几何平均把比率拉回 1 附近，即使某条回答确实值得大幅更新，$s_i$ 也只会温和地偏离 1，更新偏保守；
- §0.2 里 GRPO 组内标准化的固有偏差（全对全错零梯度、难度偏差等）原样继承。

### 2.5 一句话

GRPO 是"每个 token 单独算比率、再和同一个序列级优势相乘"；GSPO 是"整条回答一张成绩单，比率、优势、裁剪全部序列级"——**把错配的粒度对齐**。

## 3. GFPO：过滤掉不值得学的响应

GFPO = Group Filtered Policy Optimization，论文标题 *Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning*（Shrivastava et al., arXiv:2508.09726，微软）。它在三轴中的落点是**采样轴**，但和 DAPO 的过滤粒度不同。

### 3.1 动机：采样里混着大量"不值得学"的响应

GRPO 默认组内 $G$ 个回答**全部**参与损失计算。但一组采样里往往混着：

- **冗长的回答**——正确但啰嗦，把它当正样本就是教会模型"写长"；
- **重复/高度相似的响应**——冗余信息；
- **质量差的响应**——纯噪声。

全盘学习等于把坏习惯一起学进来。GFPO 的思路反过来：**多采样，少学习**（sample more, think less）——采样数 $G$ 加大，但训练前把不值得学的响应过滤掉。

### 3.2 机制：组内过滤，只对保留者算优势

对每个 prompt：

1. 采样 $G$ 个响应（比 GRPO 更多，所以叫 sample more）；
2. 按某个指标排序——例如先按正确性、再按长度，**正确且简洁的排前面**；
3. 保留 top-$k$，其余丢弃；
4. **优势只对保留的 $k$ 个响应计算，被过滤者的贡献置零**，损失只在保留者上累积：

$$
\mathcal{J}_{\mathrm{GFPO}}=\mathbb{E}\Big[\frac{1}{\sum_{i\in\mathcal{K}}\lvert y_i\rvert}\sum_{i\in\mathcal{K}}\sum_{t=1}^{\lvert y_i\rvert}\min\big(\rho_{i,t}\hat{A}_i,\ \ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon,\,1+\varepsilon)\cdot\hat{A}_i\big)\Big] \tag{7}
$$

其中 $\mathcal{K}$ 是过滤后保留的响应集合。

### 3.3 为什么能大幅缩减冗长

机制很朴素：**模型只从被保留的简洁回答上学**。冗长的正确回答一旦被过滤掉，就永远当不了正样本——"写长"不再是安全的加分策略，模型被迫学习简洁的推理路径。论文报道的冗长响应缩减（约 80%）正是这个信号在训练中单向积累的结果。

可以把它记成：批改作业时，答案对但写得啰嗦的卷子**不拿来当范例讲评**——学生自然就学会写简练。

### 3.4 与 DAPO Dynamic Sampling 的区别

两者都在"采样"上做文章，容易混淆，但过滤的粒度不同：

| | DAPO Dynamic Sampling | GFPO 过滤 |
|---|---|---|
| 过滤对象 | **整组**（零方差组） | **组内响应**（冗长/重复/低质） |
| 过滤标准 | 组内奖励方差 | 质量 + 长度排序 |
| 目的 | 省算力，跳过无梯度组 | 净化学习信号，反冗长 |

一个丢整组，一个丢组内个体——两者并不冲突，理论上可以叠加。

### 3.5 代价

- 过滤引入**选择偏差**：被滤掉的响应并非全无信息——"长但正确"的推理路径可能包含着有价值的中间步骤，一刀切会丢失一部分探索信号；
- 排序指标本身就是强先验：按长度排就学简洁，按别的指标排就学别的方向，**指标选错等于学错方向**；
- $k$ 是新的超参数，保留多少、丢弃多少需要调。

## 4. 三轴对照总表

| | 粒度轴（奖励作用到哪） | 采样轴（学哪些数据） | 裁剪轴（怎么限更新） |
|---|---|---|---|
| GRPO | 序列级优势 × token 级比率 | 全部 $G$ 个响应 | 对称裁剪，token 级 |
| DAPO | 比率不变；损失归一化改为批次级 | **丢零方差组** | **Clip-Higher**：上界放宽 |
| GSPO | **比率升级为序列级（几何平均）** | 不变 | **序列级裁剪**：整条回答一次 |
| GFPO | 不变 | **组内过滤 top-$k$** | 不变 |

三轴不是严格互斥的分类，只是一个理解框架——比如 DAPO 的 Token-level Loss 也可以归入粒度轴。重要的是每读一个新变体时先问：它动的是哪根轴？

## 5. 谱系里的其他成员（一句话版）

| 变体 | 一句话 |
|---|---|
| Dr. GRPO（Liu et al., arXiv:2503.20783） | 指出 GRPO 的两处系统偏差——长度归一化偏差与标准差归一化偏差；修法：移除 std 归一化、用固定 $\max\_len$ 做分母 |
| VAPO（Yue et al., arXiv:2504.05118，字节 Seed） | 请回价值网络估计逐 token 优势（解耦 GAE），用 PPO 的信用分配精度换回工程复杂度 |

三轴视角下：Dr. GRPO 修的是"优势怎么算"，VAPO 则直接质疑"组内基线"这条路线本身——它认为序列级标量优势的信息量不够，选择回到逐 token 的价值估计。

## 6. 选型与结语

- 要**开箱即用的工程稳定**、训练长 CoT：DAPO 四件套，每个改动都有明确的故障对应；
- 在意**粒度一致性**、信不过 token 级裁剪：GSPO，结构最干净，代价是更新保守；
- 被**冗长输出**困扰、算力充裕愿意多采样：GFPO，用过滤把学习信号提纯；
- 对**组内标准化本身**有疑虑：先读 Dr. GRPO 的偏差分析，再决定要不要 VAPO 式地请回价值网络。

最后强调一条跨变体的优先级：**变体改的是"怎么学"，决定上限的始终是奖励信号本身。** 先确认奖励能不能被程序化验证、会不会被钻空子，再谈选哪个变体——这条判断在任何变体之上。

## 参考文献

1. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476, 2025.
2. Zheng et al. *Group Sequence Policy Optimization*. arXiv:2507.18071, 2025.
3. Shrivastava et al. *Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning*. arXiv:2508.09726, 2025.
4. Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective*. arXiv:2503.20783, 2025.
5. Yue et al. *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks*. arXiv:2504.05118, 2025.
6. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300, 2024.
