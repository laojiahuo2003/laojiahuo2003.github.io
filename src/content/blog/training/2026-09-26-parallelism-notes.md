---
title: 并行训练：切数据、切参数、切时间、切序列、切专家
description: 五种并行（DP/TP/PP/SP/EP）的本质是同一个问题——单卡装不下、算不动——的五种切法：切 batch、切权重矩阵、切层、切序列、切专家。每种切法换来不同的通信形态（all-reduce、all-gather、点对点、all-to-all）与通信账本；文末附全称速查表，并按「先 DP，再 TP/PP，最后 SP/EP」给出组合选型。
pubDate: 2026-09-26T23:40:00+08:00
tags: [训练工程]
---

先说结论：**五种并行回答的是同一个问题**——模型太大、太慢，一张卡装不下、算不动。差别只在**刀落在哪、换来什么通信**：

| | 缩写 | 全称 | 切的对象 | 一句话 | 通信形态 | 用在哪 |
|---|---|---|---|---|---|---|
| ① | DP | Data Parallelism（数据并行） | batch | 模型复印 N 份，各算各的样本 | all-reduce（梯度） | 最普适，首选 |
| ② | TP | Tensor Parallelism（张量并行） | 权重矩阵 | 矩阵按列撕开，各算各的切片 | all-gather / all-reduce（每层） | 节点内 NVLink |
| ③ | PP | Pipeline Parallelism（流水线并行） | 层 | 模型切几段，microbatch 流水过 | 点对点（激活） | 跨节点 |
| ④ | SP | Sequence Parallelism（序列并行） | 序列 | 沿序列维切开 | all-gather/reduce-scatter 或 all-to-all | 省激活 / 长序列 |
| ⑤ | EP | Expert Parallelism（专家并行） | 专家 | token 按路由流动到专家卡 | all-to-all（token） | MoE 专属 |

记法：**切数据、切参数、切时间、切序列、切专家**。

五种并行的关系看这张分类树：

```text
        一个模型
           │
  ┌────────┼────────┐
  ↓        ↓        ↓
Batch    Model   Sequence
  │        │        │
  ↓        ↓        ↓
  DP       MP       SP
(数据并行)  │    (序列并行)
        ┌──┴──┐
        ↓     ↓
       TP    PP
   (张量并行)(流水线并行)

—— MoE 模型另加一路：EP（专家并行，切专家），与前几路正交
```

先摆一张总账，后面每节都会回到这张表：

| 并行 | 切分对象 | 通信原语 | 通信量级 | 通信频率 | 带宽需求 | 显存收益 |
|---|---|---|---|---|---|---|
| DP | batch | all-reduce | ≈2Φ（Φ=总参数量，每步） | 每步一次 | 中 | 无（模型全量复制） |
| TP | 权重矩阵 | all-gather + all-reduce | batch×seq×hidden | 每层 2 次 | 极高 | 权重÷p |
| PP | 层 | P2P send/recv | microbatch×seq×hidden | 每 microbatch 每边界一次 | 低 | 权重÷p |
| SP-A | 序列（LN/Dropout） | all-gather + reduce-scatter | 2×batch×seq×hidden | 每层 1 次 | 中 | 激活÷p |
| SP-B | 序列（attention） | all-to-all / 环形 P2P | ≈2~4×N×h | 每层 1 次 | 高 | 注意力中间量÷p |
| EP | 专家 | all-to-all | 2×k×token×h | 每 MoE 层 2 次 | 高（突发） | 专家权重÷p |

## ① DP：Data Parallelism（数据并行）——切 batch

**含义**：每张卡放一份**完整模型**，batch 切成 N 份，各算各的。

**为什么需要**：单卡 batch 太小，梯度估计噪声大、收敛慢，GPU 也喂不满；大 batch 还能摊薄每步的启动开销。样本之间在前向/反向里互不相干（Loss 是各样本求和），天然可切。

**流程**：

1. batch 切 N 份，每卡一份；
2. 各卡独立前向 + 反向，得到**各自的**梯度；
3. all-reduce（环形实现）把所有卡的梯度求和平均；
4. 每卡拿到**同一个**平均梯度，各自更新参数。

```text
batch 切 N 份，每卡一份完整模型：

GPU0 ── mb 1/N ──▶ 模型副本 ──▶ 梯度 g0 ──┐
GPU1 ── mb 2/N ──▶ 模型副本 ──▶ 梯度 g1 ──┼──▶ all-reduce ──▶ 平均梯度
GPU2 ── mb 3/N ──▶ 模型副本 ──▶ 梯度 g2 ──┘        │
                                           每卡拿到同一个平均梯度，各自更新
                                           → N 份模型永远一致
```

因为梯度一致、初始参数一致，N 份模型永远一模一样——训练严格等价于一张卡跑大 batch，这是 DP 区别于其他并行最舒服的一点：**结果和数学上完全等价，没有近似**。

**通信账**：环形 all-reduce 里每张卡要收发 $2(N-1)/N \times \Phi$ 的数据（$\Phi$ = 总参数量），N 大时 ≈ $2\Phi$——每步把全部梯度「出去一趟、回来一趟」。通信量随卡数线性增长，所以 DP 不是无限扩展的：卡多了同步时间上去了，global batch 太大还得降学习率。工程上常用**梯度累积**（本地先攒几个 microbatch 的梯度再同步一次）和**混合精度**（fp16/bf16 前向 + fp32 master weights）与 DP 搭配，前者降通信频率，后者省一半显存带宽。

**白话**：全班复印同一本教材，各做各的卷子，考完把错题订正汇总，大家改同一版。

**DP 的软肋与 ZeRO 三刀**：DP 不省显存——每卡仍要装参数、梯度、优化器状态三座大山（这正是[显存账本](/blog/2026-09-26-training-memory-notes/)里的三块大头）。**ZeRO（Zero Redundancy Optimizer，零冗余优化器）** 把这三样也切片：

| 阶段 | 切掉什么 | 每卡显存（三座大山） | 通信量 |
|---|---|---|---|
| 普通 DP（DDP） | 无 | 参数 + 梯度 + 优化器状态 | ≈2Φ |
| ZeRO-1 | 优化器状态 | 参数 + 梯度 + 状态÷N | ≈2Φ（与 DP 相同） |
| ZeRO-2 | + 梯度 | 参数 +（梯度、状态）÷N | ≈2Φ |
| ZeRO-3 | + 参数 | 全部÷N | ≈3Φ（1.5×DP） |

```text
每卡显存的三块：■ = 全量，□ = 切 1/N 分片

普通 DP : [■■■ 参数][■■■ 梯度][■■■ 优化器状态]
ZeRO-1 : [■■■ 参数][■■■ 梯度][□ 优化器状态 1/N]
ZeRO-2 : [■■■ 参数][□ 梯度 1/N][□ 优化器状态 1/N]
ZeRO-3 : [□ 参数 1/N][□ 梯度 1/N][□ 优化器状态 1/N]
```

机制：ZeRO-1/2 用 reduce-scatter 替代 all-reduce（每卡只拿自己分片对应的梯度，Φ），更新完再 all-gather 参数（Φ）；ZeRO-3 连参数都分片，前向、反向各要一次 all-gather（2Φ）+ 一次 reduce-scatter（Φ）。**FSDP（Fully Sharded Data Parallel，全分片数据并行）** 就是 ZeRO-3 的 PyTorch 实现，FSDP2 用 DTensor（每参数分片）实现。DDP 全称 **DistributedDataParallel**，是 PyTorch 里普通 DP 的实现。

## ② TP：Tensor Parallelism（张量并行）——切权重矩阵

**含义**：把**单个权重矩阵**切开分到多卡，每卡算矩阵的一部分。TP 属于模型并行（Model Parallelism，MP）的一种，MP 是 TP + PP 的统称。

**为什么需要**：大模型的单层矩阵就放不下。Llama-70B：hidden 8192、FFN 中间维 28672，一个权重矩阵就是 8192×28672 ≈ 2.35 亿参数 ≈ 0.47 GB（fp16），一层三个。DP 帮不了这个忙——DP 要求每卡装**整份**模型。

**思路（数学）**：矩阵乘天然可分块。$Y = XA$，把 $A$ 按列切 $A = [A_1\ A_2]$：

$$
Y = X\,[A_1\ A_2] = [XA_1 \ \ XA_2]
$$

各卡算各自的列切片，输出**天然按列分片**。第二层按行切 $B = \begin{bmatrix}B_1 \\ B_2\end{bmatrix}$：

$$
Y = B_1(XA_1) + B_2(XA_2)
$$

各卡算自己的部分和，最后 all-reduce 加起来——两层组合起来严丝合缝，不需要任何近似。

```text
X（复制到每张卡）
│
├─▶ GPU0：X·A₁（列并行）──▶ B₁·(X·A₁)（行并行）──┐
├─▶ GPU1：X·A₂（列并行）──▶ B₂·(X·A₂)（行并行）──┼──▶ all-reduce(g) ──▶ Y
│                                                │
（每块 MLP 两次通信：f = 输入对齐，g = 部分和汇总）
```

**流程**（Megatron-LM 风格，一个 MLP 块）：

1. 输入 $X$ 复制到每张卡（或按需 all-gather，记作 `f`）；
2. 第一层**列并行**（column parallel）：各卡算 $A_i X$，输出按列分片；
3. 第二层**行并行**（row parallel）：各卡算 $B_i(A_iX)$，得到部分和；
4. all-reduce 把部分和加起来（记作 `g`）——**每块两次通信（f/g），每层都有**。

attention 的切法同理：QKV 按 head 切，各卡算自己那份 head 的 attention，输出投影后 all-reduce 汇总（要求 head 数能被 TP 度整除）。

**白话**：一堵墙四个人砌，每人一条竖条，砌完拼起来——墙是完整的，但拼缝（通信）每层都有。

**通信账**：通信每层发生、量 = batch×seq×hidden（每块两次），频率是所有并行里最高的，对带宽极敏感。NVLink（H100 约 900 GB/s）比跨节点 InfiniBand（约 25~50 GB/s）快一个量级以上——**这就是 TP 不出节点（通常 ≤8 卡）的原因**：跨节点跑 TP，通信会反客为主，GPU 大部分时间在等数据。

## ③ PP：Pipeline Parallelism（流水线并行）——切层

**含义**：模型按**层**切成 p 段，每卡一段；数据切成 M 个 microbatch，像流水线一样流过。

**为什么需要**：大模型有几十上百层，层与层是严格串行依赖；但**不同数据之间没有依赖**——用数据的并行去填层的串行，所以 PP 切的其实是「时间」。

**流程（GPipe，Google 提出）**：

1. batch 切 M 个 microbatch；
2. mb1 在 GPU0 算完前向交给 GPU1，GPU0 立刻算 mb2……前向波依次往后传；
3. **全部前向做完才开始反向**（F-then-B），反向波倒着传回来；
4. 攒齐全部梯度，统一更新一次参数。

```text
p=4，32 层切成 4 段：

GPU 0: Layer 0~7      GPU 1: Layer 8~15
GPU 2: Layer 16~23    GPU 3: Layer 24~31

前向波（→）：
GPU0:  mb1 ─▶ mb2 ─▶ mb3 ─▶ mb4
GPU1:      mb1 ─▶ mb2 ─▶ mb3 ─▶ mb4
GPU2:          mb1 ─▶ mb2 ─▶ mb3 ─▶ mb4
GPU3:              mb1 ─▶ mb2 ─▶ mb3 ─▶ mb4
        ↑bubble↑

反向波（←）沿原路传回；GPipe 攒齐全部梯度后统一更新
```

**Bubble 的账**：理想情况下 M 个 microbatch 总耗时 $M(t_f + t_b)$；GPipe 实际耗时 $(M + p - 1)(t_f + t_b)$——流水线要「充满」和「排空」，开头结尾有空转。空闲比例：

$$
\text{bubble} = \frac{p-1}{M+p-1}
$$

M 越大 bubble 越小，但途中要暂存的激活也越多（显存换时间）。**1F1B（One Forward, One Backward，一前向一反向，Megatron 提出）**：先用 p−1 个前向把流水线充满，之后每卡前向/反向交替做，bubble 压到约 $(p-1)/M$，接近减半。Megatron 的**交错 1F1B**（interleaved）让每卡轮流持多个不连续的层段，bubble 再降一截。另一条路线是**异步流水**（如 PipeDream）：允许各卡用稍旧的权重继续算、不等同步，代价是收敛变慢、权重版本管理复杂。

**白话**：汽车流水线，每道工序一台机器；刚开工和快收工时后面的机器在等料——等的这段就是 bubble。

**通信账**：只传层间激活，点对点（P2P send/recv），量 = microbatch×seq×hidden，**频率低、量小、能容忍延迟** → 适合跨节点。局限：首尾卡负载不均（embedding 层小）、层数要远大于 p、激活要暂存。

## ④ SP：Sequence Parallelism（序列并行）——切序列

「序列并行」有两位，面试常混，必须分开。

### SP-A：Megatron 的序列并行（省激活）

**为什么**：TP 里每张卡都存整份输入 $X$（batch×seq×hidden），激活显存重复了 p 份。观察：LayerNorm、Dropout 沿序列维本来就是**逐 token 独立**的（LN 没有 BN 那种跨 batch 统计量，每个 token 自己归一化），可以放心切。

**流程**：各卡只算自己那 1/p 段序列的 LN/Dropout → 进 attention 前 all-gather 拼回 → 出来后 reduce-scatter 切回。效果：激活显存 ÷p，代价只是 LN 附近两次通信（量 ≈ 2×batch×seq×hidden）。SP-A 必须和 TP 一起用才有意义（单独用省不了权重），且只省激活、不省权重。

```text
序列: [t1 t2 t3 | t4 t5 t6]（切 2 段）

GPU0: LN(t1..t3) ──all-gather──▶ attention（看全行）──reduce-scatter──▶ 拿回 t1..t3
GPU1: LN(t4..t6) ──all-gather──▶ attention（看全行）──reduce-scatter──▶ 拿回 t4..t6
```

### SP-B：长序列的 Context Parallelism（上下文并行）

**为什么**：注意力矩阵是 N×N，序列太长（如百万 token）单卡连注意力中间量都放不下。把序列切 chunk 分给多卡，但 attention 要「看全行」——两种解法：

- **Ring Attention（环形注意力）**：KV 块沿环传一圈，每卡轮流看过所有 KV（通信 ≈ 2×N×h，点对点）；
- **Ulysses**（DeepSpeed 提出）：按 head 切，QKV 各按 head 分片，all-to-all 交换（通信 ≈ 4×N×h/P）。

```text
Ring Attention（KV 沿环传阅，每卡轮流看全所有 KV）：

  ┌────────────────────────────────────┐
  ▼                                    │
GPU0 ──KV块──▶ GPU1 ──KV块──▶ GPU2 ──KV块──▶ GPU3
(seq 1/4)     (seq 2/4)     (seq 3/4)     (seq 4/4)
```

FlashAttention 的分块恰好让「传阅 KV 块」成为可能，两者是天然搭档（见 [FlashAttention 那篇](/blog/2026-09-26-flashattention-notes/)）。

**白话**：一长条纸裁几段分人读，但答题要看整行——要么大家传阅（ring），要么按题号分工互相抄（Ulysses）。

**通信账**：SP-B 通信量与序列长度成正比，序列不够长时不如直接上 TP；序列超长、注意力显存 N² 爆炸时它才登场。

## ⑤ EP：Expert Parallelism（专家并行）——切专家

**含义**：MoE（Mixture of Experts，混合专家）的专家分散在不同卡，token 按路由结果送到对应专家所在的卡。

**为什么需要**：MoE 每层有几百个专家，专家总参数量远超单卡。以 DeepSeek-V3 为例：每层 256 个路由专家 + 1 个共享专家，专家 hidden 2048，每层专家权重约 110 亿参数——单卡放不下。而且每步只有 top-k（V3 是 top-8）个专家被激活：**货（专家）不动，人（token）动**。

**流程**：

1. 各卡算出自己 token 的路由结果（router/gating：token → 专家映射）；
2. all-to-all **dispatch**：按目标专家把 token 发到对应卡；
3. 各卡算自己专家的 FFN；
4. all-to-all **combine**：结果送回原卡；
5. 继续后面的层（残差连接在 combine 之后）。

```text
token → 路由 → all-to-all dispatch → 专家所在卡计算 → all-to-all combine → 回原卡

GPU0: token a（路由→专家100）  token b（路由→专家3）
          │ dispatch                    │ dispatch
          ▼                             ▼
    专家100所在的卡：算 E100(a)    专家3所在的卡：算 E3(b)
          └────────────┬──────────────┘
                       ▼ combine
                 结果送回 GPU0，继续下一层
```

**白话**：医院分诊台，按病症把病人送到对应科室，看完送回大厅继续排队。

**通信账**：每 MoE 层两次 all-to-all，量 = 2×k×token×h；all-to-all 是全网状通信，对网络拓扑敏感，所以 EP 的卡通常部署在同一节点/机架内。更大的麻烦是**负载不均**：路由天然偏爱少数热门专家，忙的卡没算完、闲的卡干等。解法有三：**容量因子**（capacity factor，限制每个专家接收的 token 数，超了丢弃）、**辅助负载均衡 loss**、以及 DeepSeek 的 **EPLB**（Expert Parallelism Load Balancer，专家并行负载均衡器：用可学习 bias 动态调路由，不加辅助 loss，还支持在线换专家分组）。V3 的专家设计本身就是均衡手段：**细粒度专家**（更多、更小的专家，单专家计算粒度小、好摊平）+ **共享专家**（人人路过都算，稳定兜底）。EP 与 TP 也正交可叠加：MoE 层内的专家权重可以再按 TP 切。

## 组合与选型

**3D 并行（DP × TP × PP）**：Megatron-LM 的标准配置。DP 管横（跨机扩展）、TP 管单节点内、PP 管纵（跨节点流水），三者正交相乘，世界规模 = d × t × p。

```text
世界规模 = DP × TP × PP（正交相乘，如 16 DP × 4 TP × 4 PP = 256 卡）

┌───────── DP 组 1 ─────────┐   ┌───────── DP 组 2 ─────────┐
│ PP0: [TP0][TP1][TP2][TP3] │   │ PP0: [TP0][TP1][TP2][TP3] │
│ PP1: [TP0][TP1][TP2][TP3] │   │ PP1: [TP0][TP1][TP2][TP3] │
└───────────────────────────┘   └───────────────────────────┘
  TP 走节点内 NVLink               组间 all-reduce 梯度（DP 跨机）
```

**DeepSeek-V3 的配置**（训练 2048 张 H800）：**EP 64 × PP 16（DualPipe）× DP 2（ZeRO-1）× TP 1 × SP 1**——EP 挑大梁解决 MoE 显存，PP 走 DualPipe（把前向、反向、通信三条流水线叠起来，bubble 压到近零，计算与通信互相藏延迟），TP/SP 干脆不用（EP 已解决显存问题，TP 每层高频通信不值得）。模型 671B 总参数、每 token 激活 37B。

**选型口诀**：**先 DP（ZeRO 扛），单卡放不下模型再上 TP（节点内）/ PP（跨节点），序列超长加 SP/CP，MoE 必上 EP**。

## 全称速查表

| 缩写 | 全称 | 中文 |
|---|---|---|
| DP | Data Parallelism | 数据并行 |
| DDP | DistributedDataParallel | PyTorch 的分布式数据并行实现 |
| TP | Tensor Parallelism | 张量并行 |
| PP | Pipeline Parallelism | 流水线并行 |
| SP / CP | Sequence / Context Parallelism | 序列并行 / 上下文并行 |
| EP | Expert Parallelism | 专家并行 |
| MP | Model Parallelism | 模型并行（TP/PP 的统称） |
| ZeRO | Zero Redundancy Optimizer | 零冗余优化器 |
| FSDP | Fully Sharded Data Parallel | 全分片数据并行 |
| GPipe | — | Google 的流水线并行方案（F-then-B） |
| 1F1B | One Forward, One Backward | 一前向一反向（Megatron 流水调度） |
| MoE | Mixture of Experts | 混合专家 |
| EPLB | Expert Parallelism Load Balancer | 专家并行负载均衡器（DeepSeek） |
| DualPipe | — | DeepSeek 的双向流水线（计算通信重叠） |
| USP | Unified Sequence Parallelism | 统一序列并行框架 |
| P2P | Point-to-Point | 点对点通信 |
| NCCL | NVIDIA Collective Communications Library | NVIDIA 集合通信库 |
| microbatch | — | 微批次（流水线的最小调度单元） |
| bubble | — | 流水线气泡（GPU 空转时间） |
| capacity factor | — | 专家容量因子（负载均衡参数） |

## 总结

- **DP 切 batch**：通信 all-reduce（梯度，≈2Φ）；显存不省 → ZeRO/FSDP 把优化器状态、梯度、参数也切了（ZeRO-3 通信 1.5×，显存 ÷N）；
- **TP 切矩阵**：列并行 + 行并行组合，每块 f/g 两次通信，频率最高 → 限节点内 NVLink；
- **PP 切层**：点对点传激活、量小可跨节点；代价是 bubble，GPipe 公式 (p−1)/(p−1+M)，1F1B 压半，DualPipe 压到近零；
- **SP 切序列**：Megatron 版切 LN/Dropout 省激活（配 TP）；长序列版（ring/Ulysses）撑超长上下文；
- **EP 切专家**：两次 all-to-all 搬 token；负载均衡是工程核心（capacity factor / 辅助 loss / EPLB）；
- **组合**：五种并行全部正交，实际系统按「DP 打底 → TP/PP 撑模型 → SP/EP 补短板」层层叠加。

## 延伸

- **ZeRO/FSDP2**：DP 的分片路线，FSDP2 用 DTensor（每参数分片）实现，与 TP 可以共存；
- **DualPipe + EPLB**（DeepSeek 开源周）：前向/反向/通信三流水线重叠 + 无辅助 loss 的专家均衡，是 EP/PP 组合的当前标杆；
- **USP**（Unified Sequence Parallelism）：把 Ulysses、Ring、Megatron SP 统一起来的混合框架，按序列长度自适应切换；
- **与本站联动**：[显存账本](/blog/2026-09-26-training-memory-notes/)讲单卡显存的三块大头（ZeRO 三刀的对象）；[FlashAttention](/blog/2026-09-26-flashattention-notes/)的分块让 KV 传阅（ring attention）成为可能。
