---
title: 从 Trust Region 到 PPO：策略更新到底被约束了什么
description: 从经典优化中的信赖域方法讲起，经 TRPO 的 KL 硬约束与 Fisher 信息矩阵，到 PPO 用裁剪替代约束的简化思路。术语与记号在首次出现处就地解释，关键步骤配可核对的数字算例。
pubDate: 2026-09-26T13:00:00+08:00
tags: [强化学习, 后训练]
---

## 摘要

PPO 的全称是 *Proximal Policy Optimization*——近端策略优化。"近端"二字直接来自优化理论中的**信赖域**（Trust Region）思想：每一步更新都不允许走太远，以保证局部近似仍然可信。

然而 PPO 并没有显式地求解一个约束优化问题——真正在策略优化中引入信赖域约束的是它的前身 TRPO（Trust Region Policy Optimization）。PPO 所做的，是用一个裁剪目标函数来**近似**TRPO 的约束效果，从而把求解成本从"解一个带约束的优化问题"降低到"对同一个无约束目标做梯度下降"。

本文从经典信赖域优化讲起，经 TRPO 的 KL 硬约束与 Fisher 信息矩阵，到 PPO 的裁剪替代，逐步展示这条演化链上每一步的动机与代价。

## 0. 记号与问题设定

### 0.1 记号约定

| 记号 | 含义 |
|---|---|
| $\theta$ | 策略参数 |
| $\theta_{\mathrm{old}}$ | 当前迭代开始时的参数（采样数据时的快照） |
| $\pi_\theta$ | 以 $\theta$ 为参数的策略 |
| $J(\theta)$ | 策略的期望回报，即待最大化的目标 |
| $\nabla J(\theta)$ | $J$ 对 $\theta$ 的梯度 |
| $\Delta\theta$ | 一次迭代中参数的更新量 $\theta - \theta_{\mathrm{old}}$ |
| $\rho(\theta)$ | 重要性比率 $\pi_\theta / \pi_{\theta_{\mathrm{old}}}$ |
| $A(s,a)$ | 优势函数 |
| $D_{\mathrm{KL}}(p\|q)$ | KL 散度 |
| $\mathbf{F}$ | Fisher 信息矩阵 |
| $\delta$ | 信赖域半径（KL 约束上界） |
| $\varepsilon$ | PPO 裁剪范围 |

### 0.2 策略优化的基本循环

策略梯度的核心循环只有三步：

1. **采样**：用当前策略 $\pi_{\theta_{\mathrm{old}}}$ 与环境交互，收集轨迹数据；
2. **估计**：用收集到的数据估计目标函数 $J(\theta)$ 的梯度；
3. **更新**：沿梯度方向更新参数 $\theta \leftarrow \theta_{\mathrm{old}} + \alpha\,\nabla J(\theta_{\mathrm{old}})$。

策略梯度定理给出

$$
\nabla J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla\log\pi_\theta(\tau)\cdot R(\tau)\big] \tag{1}
$$

其中 $\tau$ 是一条完整的交互轨迹，$R(\tau)$ 是其累计回报。式 (1) 的含义朴素：**采样到高回报轨迹就提高它的概率，低回报就压低。**

这个循环有一个隐含假设：**式 (1) 中的样本必须来自当前策略 $\pi_\theta$**。一旦执行了第 3 步的更新，$\theta$ 变了，之前采的数据就不再"on-policy"。这就是策略优化中所有复杂性质的根源。

## 1. 信赖域优化：经典优化中的"别走太远"

### 1.1 从牛顿法说起

在经典优化中，对目标函数 $f(x)$ 在当前点 $x_k$ 做二阶泰勒展开：

$$
f(x_k + \Delta x) \approx f(x_k) + \nabla f(x_k)^\top \Delta x + \frac{1}{2}\Delta x^\top \mathbf{H}\,\Delta x \tag{2}
$$

其中 $\mathbf{H} = \nabla^2 f(x_k)$ 是 Hessian 矩阵，描述目标函数在 $x_k$ 处的局部曲率。

如果直接令 $\nabla f(x_k) + \mathbf{H}\,\Delta x = 0$ 求解 $\Delta x$，就得到**牛顿法**的更新步 $\Delta x = -\mathbf{H}^{-1}\nabla f(x_k)$。问题在于：式 (2) 是一个**局部近似**，它只在 $x_k$ 附近可信。一旦 $\Delta x$ 过大，二阶展开与实际函数严重偏离，牛顿步可能把迭代点送到目标函数反而变差的位置。

信赖域方法的处理方式是：**给 $\Delta x$ 加一个约束，要求它不能走太远。**

$$
\max_{\Delta x}\quad f(x_k) + \nabla f(x_k)^\top \Delta x + \frac{1}{2}\Delta x^\top \mathbf{H}\,\Delta x \qquad \text{s.t.}\quad \|\Delta x\| \leq \Delta_{\max} \tag{3}
$$

$\Delta_{\max}$ 就是**信赖域半径**。式 (3) 的含义是：在二阶模型可信的范围内，找到最优的更新步。

### 1.2 信赖域的核心思想

信赖域方法的关键不在于"用什么模型"（二阶展开只是最常见的一种），而在于**用局部近似的有效性来约束更新幅度**。这个思想可以直接迁移到策略优化：

- 策略优化中，我们用**一阶泰勒展开**（即策略梯度）来近似目标函数；
- 一阶近似比二阶近似更粗糙，可信范围更小；
- 因此更需要约束更新幅度，保证"近似仍然有效"。

区别在于：经典优化中的"距离"用参数空间的欧氏距离 $\|\Delta\theta\|$ 衡量，而策略优化中需要一种能反映**策略本身变化幅度**的距离度量——因为同样的参数变化量，在不同方向上对策略行为的影响可能截然不同。

## 2. 策略梯度中的"走太远"问题

### 2.1 一阶近似的失效

回到策略优化。在第 2 步中，我们用 $\theta_{\mathrm{old}}$ 附近的一阶展开来近似 $J(\theta)$：

$$
J(\theta) \approx J(\theta_{\mathrm{old}}) + \nabla J(\theta_{\mathrm{old}})^\top (\theta - \theta_{\mathrm{old}}) \tag{4}
$$

如果直接用式 (4) 做梯度上升 $\theta = \theta_{\mathrm{old}} + \alpha\,\nabla J(\theta_{\mathrm{old}})$，步长 $\alpha$ 完全由人工设定。问题有两个：

**第一，策略变了，数据就过期了。** 式 (1) 的梯度估计要求样本来自当前策略。但重新采样数据的成本极高（对语言模型而言，等于完整跑一次推理）。实践中必须用旧策略的数据来估计新策略的梯度——这就是**重要性采样**：

$$
\nabla J(\theta)=\mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}}\Big[\underbrace{\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}}_{\textstyle \rho(\theta)}\cdot A(s,a)\Big] \tag{5}
$$

$\rho(\theta)$ 是重要性比率，衡量当前策略相对采样策略更偏好该动作多少倍。式 (5) 在 $\pi_\theta$ 与 $\pi_{\theta_{\mathrm{old}}}$ 接近时无偏且方差可控。但两者一旦偏离，$\rho(\theta)$ 的方差会迅速增大——某个动作在旧策略下概率极低、在新策略下概率很高时，比率会变成几十甚至上百，单个样本就能主导整个梯度估计。

**第二，策略空间的几何结构不是欧氏的。** 参数空间中 $\|\Delta\theta\|=0.01$ 的更新，在某些方向上可能使策略行为几乎不变（例如同时缩放某层权重矩阵的所有行），在另一些方向上可能使策略完全改变（例如修改输出层中某个关键方向）。欧氏距离无法区分这两种情况。

**数字算例**：考虑一个简化的二分类场景，策略输出 $\pi_\theta(a=1\mid s) = \sigma(\theta)$（$\sigma$ 为 Logistic 函数）。

- 当 $\theta_{\mathrm{old}}=0$ 时，$\pi(a=1)=0.5$。更新 $\Delta\theta=+0.5$ 后，$\pi(a=1)=\sigma(0.5)\approx0.622$，变化了 0.122。
- 当 $\theta_{\mathrm{old}}=5$ 时，$\pi(a=1)=\sigma(5)\approx0.993$。同样更新 $\Delta\theta=+0.5$ 后，$\pi(a=1)=\sigma(5.5)\approx0.996$，变化仅 0.003。

**同样的参数步长，策略行为的变化量差了 40 倍。** 这说明用参数空间的欧氏距离来约束更新幅度是不合理的——需要一种直接度量"策略行为变化了多少"的距离。

### 2.2 需要一个"策略行为距离"

KL 散度 $D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta)$ 恰好满足这个需求：它直接衡量两个策略在输出分布上的差异，与参数空间的几何无关。

但直接约束每一步的 KL 散度仍然不够——我们需要理解这个约束在参数空间中意味着什么，以及如何高效求解。这就是 TRPO 的核心贡献。

## 3. TRPO：带 KL 硬约束的策略优化

### 3.1 目标函数

TRPO（Trust Region Policy Optimization, Schulman et al., 2015）的思路是：在重要性采样的目标函数上，直接施加 KL 散度约束。

$$
\max_{\theta}\quad \mathbb{E}\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}\cdot A(s,a)\Big] \qquad \text{s.t.}\quad \mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta)\big]\leq\delta \tag{6}
$$

式 (6) 中，**目标函数**是重要性采样下的策略梯度目标（即允许用旧数据估计新梯度），**约束条件**要求新旧策略的平均 KL 散度不超过阈值 $\delta$。

这里的期望均对 $(s,a)$ 的采样分布取。约束的含义是：**在策略行为变化不超过 $\delta$ 的前提下，最大化重要性采样估计的回报改进。**

### 3.2 从 KL 约束到 Fisher 信息矩阵

式 (6) 的约束是非线性的，直接求解困难。TRPO 的关键技巧是**对约束做二阶泰勒展开**。

将 $D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta)$ 在 $\theta=\theta_{\mathrm{old}}$ 处展开。一阶项为零（KL 散度在 $\theta=\theta_{\mathrm{old}}$ 处取最小值 0，梯度为零），二阶项为：

$$
D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta) \approx \frac{1}{2}(\theta-\theta_{\mathrm{old}})^\top \mathbf{F}\,(\theta-\theta_{\mathrm{old}}) \tag{7}
$$

其中 $\mathbf{F}$ 是 KL 散度关于 $\theta$ 的 Hessian，即 **Fisher 信息矩阵**：

$$
\mathbf{F} = \mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}}\Big[\nabla\log\pi_{\theta_{\mathrm{old}}}(a\mid s)\,\nabla\log\pi_{\theta_{\mathrm{old}}}(a\mid s)^\top\Big] \tag{8}
$$

Fisher 信息矩阵描述的是**参数空间中策略分布的局部曲率**：在 $\mathbf{F}$ 特征值大的方向上，参数的微小变化就会导致策略分布的显著改变；在特征值小的方向上，参数可以走得更远而不影响策略行为。

**数字算例**（接 §2.1）：继续用 $\pi_\theta(a=1\mid s)=\sigma(\theta)$ 的例子。Fisher 信息矩阵退化为标量：

$$
F = \mathbb{E}\big[(\nabla\log\pi_\theta)^2\big] = \mathbb{E}\big[(a-\sigma(\theta))^2\big] = \sigma(\theta)(1-\sigma(\theta)) \tag{9}
$$

当 $\theta_{\mathrm{old}}=0$ 时，$F=0.25$；当 $\theta_{\mathrm{old}}=5$ 时，$F\approx0.007$。约束 $\frac{1}{2}F\,(\Delta\theta)^2\leq\delta$ 给出的允许步长分别为 $|\Delta\theta|\leq\sqrt{2\delta/0.25}$ 和 $|\Delta\theta|\leq\sqrt{2\delta/0.007}$——**后者是前者的约 5.3 倍**。这正是 §2.1 中观察到的现象的数学解释：策略已经接近确定时（$\theta=5$），参数空间中可以走得更远。

### 3.3 求解：共轭梯度法

将式 (7) 代入式 (6)，约束变为二次型：

$$
\max_{\Delta\theta}\quad \mathbf{g}^\top\Delta\theta \qquad \text{s.t.}\quad \frac{1}{2}\Delta\theta^\top\mathbf{F}\,\Delta\theta\leq\delta \tag{10}
$$

其中 $\mathbf{g}=\nabla J(\theta_{\mathrm{old}})$ 是策略梯度。这是一个带椭球约束的线性规划，有解析解：

$$
\Delta\theta = \sqrt{\frac{2\delta}{\mathbf{g}^\top\mathbf{F}^{-1}\mathbf{g}}}\;\mathbf{F}^{-1}\mathbf{g} \tag{11}
$$

式 (11) 的含义清晰：**更新方向是 $\mathbf{F}^{-1}\mathbf{g}$（自然梯度方向），步长由约束 $\delta$ 确定。**

问题在于 $\mathbf{F}$ 的规模。对语言模型而言，$\theta$ 的维度可达数十亿，$\mathbf{F}$ 是一个 $|\theta|\times|\theta|$ 的矩阵，无法显式存储或求逆。TRPO 的解决方案是**共轭梯度法**（Conjugate Gradient, CG）：

- 不需要显式构造 $\mathbf{F}$，只需要能计算**矩阵向量乘积** $\mathbf{F}\mathbf{v}$；
- $\mathbf{F}\mathbf{v}$ 可以通过**自动微分**的两次反向传播（即 Hessian-vector product）来计算；
- 共轭梯度法经过 $k$ 次迭代后给出 $\mathbf{F}^{-1}\mathbf{g}$ 的近似解，$k$ 远小于 $|\theta|$。

此外，由于式 (7) 是二阶近似，实际更新后还需做**线搜索**（line search）：沿 $\Delta\theta$ 方向逐步缩小步长，直到实际目标函数值确实改进，以确保近似的有效性。

### 3.4 TRPO 的完整流程

1. 用 $\pi_{\theta_{\mathrm{old}}}$ 采样一批轨迹数据；
2. 计算策略梯度 $\mathbf{g}$；
3. 用共轭梯度法求解 $\mathbf{F}^{-1}\mathbf{g}$（每次迭代做一次 Hessian-vector product）；
4. 按式 (11) 计算更新步 $\Delta\theta$；
5. 沿 $\Delta\theta$ 做线搜索，确保目标函数改进；
6. 更新 $\theta \leftarrow \theta_{\mathrm{old}} + \Delta\theta$。

### 3.5 TRPO 的代价

TRPO 在理论上优雅，工程上却有三个主要代价：

| 代价 | 说明 |
|---|---|
| 共轭梯度的额外计算 | 每步迭代需要多次 Hessian-vector product，每次涉及两次反向传播 |
| 线搜索 | 需要反复评估目标函数，进一步增加计算量 |
| 实现复杂度 | 需要手动实现共轭梯度、Hessian-vector product、线搜索，难以直接复用标准优化器 |

更关键的是，在语言模型的规模下，策略网络与价值网络均有数十亿参数，TRPO 的额外计算开销变得难以承受。这就引出了 PPO 的动机：**能否在保留信赖域思想的稳定性的同时，去掉共轭梯度和线搜索？**

## 4. PPO：用裁剪替代约束

### 4.1 核心思路

PPO（Proximal Policy Optimization, Schulman et al., 2017）的做法是：放弃显式的 KL 约束，转而修改目标函数本身，使其对策略的大幅偏离**自动不敏感**。

具体而言，PPO 对重要性比率 $\rho(\theta)$ 做裁剪：

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\Big[\min\big(\rho(\theta)\,A,\;\;\mathrm{clip}(\rho(\theta),\,1-\varepsilon,\,1+\varepsilon)\cdot A\big)\Big] \tag{12}
$$

其中 $\mathrm{clip}(z,\,1-\varepsilon,\,1+\varepsilon)$ 把 $z$ 截断到 $[1-\varepsilon,\,1+\varepsilon]$ 区间，$\varepsilon$ 常取 0.2。

式 (12) 中的 $\min$ 使裁剪表现为**单边惩罚**。按优势符号与比率位置展开，共四种情形：

| 优势 | 比率区间 | 无裁剪时的行为 | 含 $\min$ 后的行为 |
|---|---|---|---|
| $A>0$ | $\rho>1+\varepsilon$ | 持续增大该动作概率，单步更新过大 | 梯度为零，停止鼓励 |
| $A>0$ | $\rho<1-\varepsilon$ | — | 取 $\rho A$ 项，仍产生梯度 |
| $A<0$ | $\rho<1-\varepsilon$ | 持续压低该动作概率 | 梯度为零，停止压低 |
| $A<0$ | $\rho>1+\varepsilon$ | — | 取较悲观项，仍产生梯度 |

**当 $\rho$ 落在 $[1-\varepsilon,\,1+\varepsilon]$ 内时，式 (12) 退化为普通策略梯度；超出该区间后，仅保留使目标变差方向的梯度。**

### 4.2 裁剪如何近似信赖域

将 TRPO 的约束与 PPO 的裁剪放在一起比较：

| | TRPO | PPO |
|---|---|---|
| 约束方式 | 硬约束：$D_{\mathrm{KL}}\leq\delta$ | 软约束：裁剪 $\rho\in[1-\varepsilon,1+\varepsilon]$ |
| 约束对象 | 策略分布的 KL 散度 | 重要性比率 |
| 求解方式 | 共轭梯度 + 线搜索 | 标准梯度下降（Adam 等） |
| 每步额外开销 | 多次 Hessian-vector product | 无 |
| 约束的精确性 | 精确（在二阶近似意义下） | 近似 |

两者的共同点是：**都限制了策略的单步更新幅度，以保证局部近似的有效性。** 区别在于实现方式：

- TRPO 在参数空间上施加了一个**全局的、耦合的**约束——$\mathbf{F}$ 捕捉了参数间的曲率关系，约束是各参数维度联合生效的；
- PPO 的裁剪作用在**每个样本的重要性比率**上——它是逐样本、逐 token 的独立约束，不涉及参数间的耦合。

**数字算例**：取 $\varepsilon=0.2$。

- 若 $\rho=1.5$、$A=+1$：$\rho>1+\varepsilon$，$\min$ 取裁剪项，$\min(1.5,\;1.2)=1.2$，该 token 不再产生梯度；
- 若 $\rho=0.5$、$A=+1$：$\rho<1-\varepsilon$，$\min$ 取未裁剪项，$\min(0.5,\;0.8)=0.5$，梯度正常流动，把该动作的概率往回拉。

裁剪的效果是：当策略相对采样策略偏离超过 20% 时，梯度被截断——这与 TRPO 中"策略行为变化不超过 $\delta$"的约束在精神上是一致的，只是 PPO 用重要性比率的变化（而非 KL 散度）来度量"偏离了多少"。

### 4.3 裁剪 vs. KL 惩罚

在 PPO 之前，已有另一种简化 TRPO 的方案：**直接以 KL 散度作为惩罚项加入目标函数**。

$$
L^{\mathrm{KPEN}}(\theta) = \mathbb{E}\big[\rho(\theta)\,A\big] - \beta\,\mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta)\big] \tag{13}
$$

$\beta$ 是惩罚系数。这种做法的问题在于：

- $\beta$ 需要自适应调节：太小起不到约束效果，太大则过度保守。原论文指出，自适应调节 $\beta$ 的效果不稳定；
- KL 惩罚是**事后**约束——先更新参数，再检查 KL 是否过大，过大就增大 $\beta$ 重新来。这相当于用试错法来满足约束，效率低；
- 在深度网络中，KL 散度关于 $\theta$ 不是凸的，惩罚项的梯度可能不可靠。

PPO 的裁剪是**事前**约束——目标函数本身就在 $\rho$ 偏离时自动截断梯度，不需要事后调节。这使得 PPO 可以直接使用 Adam 等标准优化器，无需任何额外的超参数调节逻辑。

### 4.4 PPO 的完整流程

1. 用 $\pi_{\theta_{\mathrm{old}}}$ 采样一批轨迹数据；
2. 计算优势估计 $\hat{A}_t$（通常用 GAE）；
3. 在同一批数据上执行 $K$ 个 epoch 的梯度上升，优化式 (12)；
4. 更新 $\theta_{\mathrm{old}} \leftarrow \theta$，回到第 1 步。

注意第 3 步：PPO 可以在同一批数据上执行**多个 epoch**——这正是裁剪带来的核心收益。裁剪保证了即使多次复用同一批数据，策略也不会偏离采样策略太远。TRPO 由于每步都做精确的约束求解，反而不需要多 epoch（每步已经是最优的）。

### 4.5 在 RLHF 中的实现

在大语言模型的 RLHF 流程中，PPO 需要同时维护四个模型：

| 模型 | 作用 | 是否训练 |
|---|---|---|
| Actor $\pi_\theta$ | 待优化策略 | 是 |
| Critic $V_\psi$ | 估计状态价值，供优势估计使用 | 是（与 actor 同规模） |
| Reward Model $r_\phi$ | 输出序列级奖励 | 否（预先训练） |
| Reference $\pi_{\mathrm{ref}}$ | 计算 KL 惩罚 | 否 |

逐 token 奖励的构造为：

$$
r_t=-\beta\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}+r_\phi(x,y)\cdot\mathbb{1}[t=T] \tag{14}
$$

式中第一项是逐 token 的 KL 惩罚（与参考模型的偏离），第二项是奖励模型的序列级打分（$\mathbb{1}[\cdot]$ 为指示函数，仅在最后一个 token 处计入）。

注意这里出现了**两层"信赖域"**：

- **PPO 裁剪**（式 12）：约束 $\pi_\theta$ 相对 $\pi_{\theta_{\mathrm{old}}}$ 的偏离——保证重要性采样有效；
- **KL 惩罚**（式 14）：约束 $\pi_\theta$ 相对 $\pi_{\mathrm{ref}}$ 的偏离——防止 reward hacking。

两者的作用对象不同，但思想一致：**限制策略的更新幅度，以保证近似的有效性。**

## 5. 从 TRPO 到 PPO：一条清晰的简化链

### 5.1 演化路径

把从信赖域优化到 PPO 的演化画成一条链：

$$
\text{信赖域优化（经典）} \xrightarrow{\text{策略空间}} \text{TRPO} \xrightarrow{\text{去约束}} \text{PPO-clip}
$$

每一步的简化逻辑：

| 步骤 | 简化了什么 | 保留了什么 | 代价 |
|---|---|---|---|
| 信赖域 → TRPO | 将欧氏距离替换为 KL 散度 | 硬约束 + 二阶信息 | 需要共轭梯度 |
| TRPO → PPO-clip | 将硬约束替换为目标函数裁剪 | 限制单步偏离 | 约束变为近似的 |

### 5.2 两种近似的对比

TRPO 的近似：对 KL 约束做二阶泰勒展开（式 7），在 $\theta$ 接近 $\theta_{\mathrm{old}}$ 时精确。

PPO 的近似：用裁剪替代约束。裁剪在重要性比率维度上划定了一个固定宽度的"安全区间"，但它忽略了参数空间的曲率结构——即 Fisher 信息矩阵所描述的各方向上的不同敏感度。

**这意味着 PPO 的约束是"各向同性"的**：无论参数更新方向对策略行为的影响有多大，裁剪的阈值 $\varepsilon$ 都相同。而 TRPO 的约束是"各向异性"的：Fisher 信息矩阵自动在敏感方向上施加更紧的约束。

这一差异在实践中表现为：PPO 的 $\varepsilon$ 通常需要比 TRPO 的 $\delta$ 更仔细地调节。$\varepsilon$ 过大时，PPO 的行为接近无约束的策略梯度，稳定性下降；$\varepsilon$ 过小时，训练过于保守，收敛缓慢。

### 5.3 为什么 PPO 在实践中胜出

尽管 TRPO 的约束更精确，PPO 在大语言模型的 RLHF 中几乎完全取代了 TRPO。原因有三：

**第一，工程简单。** PPO 只需要一个标准的梯度下降优化器（如 Adam），不需要实现共轭梯度、Hessian-vector product 或线搜索。在分布式训练环境中，每减少一个自定义组件，就减少一个潜在的故障点。

**第二，计算高效。** TRPO 每步迭代需要多次 Hessian-vector product（每次涉及两次完整的反向传播），而 PPO 的裁剪只需要在计算图中插入一个 `clip` 操作，几乎不增加额外开销。

**第三，多 epoch 复用。** PPO 的裁剪天然支持在同一批数据上做多个 epoch 的训练，进一步摊薄了采样成本。TRPO 的精确约束求解反而使多 epoch 变得不必要（每步已经是最优的），但采样成本并未减少。

## 6. 总结

1. **信赖域思想**的核心是：用局部近似的有效性来约束更新幅度。在策略优化中，这意味着限制策略的单步变化；

2. **TRPO** 将信赖域思想具体化为 KL 散度硬约束，并通过 Fisher 信息矩阵将约束转化为参数空间中的二次型，用共轭梯度法求解。约束精确但计算代价高；

3. **PPO** 用裁剪重要性比率来近似 TRPO 的约束效果。裁剪是逐样本的、各向同性的软约束，忽略了参数空间的曲率结构，但使优化问题退化为标准的无约束梯度下降；

4. 从 TRPO 到 PPO 的演化链是：**精确约束 + 二阶信息 → 近似约束 + 一阶信息**。这一简化在工程上带来了巨大的收益，代价是约束的精确性下降；

5. 在 RLHF 中存在两层信赖域：PPO 裁剪约束 $\pi_\theta$ 相对 $\pi_{\theta_{\mathrm{old}}}$ 的偏离（保证重要性采样有效），KL 惩罚约束 $\pi_\theta$ 相对 $\pi_{\mathrm{ref}}$ 的偏离（防止 reward hacking）。两者的思想一致，作用对象不同。

## 参考文献

1. Nocedal & Wright. *Numerical Optimization*, 2nd Edition. Springer, 2006. （信赖域方法的经典教材）
2. Schulman et al. *Trust Region Policy Optimization*. arXiv:1502.05477, 2015.
3. Schulman et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
4. Schulman et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438, 2015.
5. Ouyang et al. *Training language models to follow instructions with human feedback*. arXiv:2203.02155, 2022.
6. Kakade & Langford. *Approximately Optimal Approximate Reinforcement Learning*. ICML 2002. （策略更新的 KL 约束的理论基础）
