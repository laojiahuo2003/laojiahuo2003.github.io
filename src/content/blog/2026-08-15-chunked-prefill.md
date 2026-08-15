---
title: Chunked Prefill 学习笔记：不拆机器的中间路线
description: 把 prefill 切片给 decode 让路：token 预算怎么设、和 PD 分离怎么选、隐藏的缓存代价。
pubDate: 2026-08-15T13:00:00+08:00
tags: [推理, 笔记]
---

[PD 分离笔记](/blog/2026-08-15-pd-disaggregation-notes/)留的另一个坑。那次说的是"拆成两个池"，但业界还有一条温和得多的路线：**机器不拆，把 prefill 的活切碎**。Sarathi-Serve 论文是代表作，vLLM 的默认模式之一。

## 问题重述：一步大活挡住一路小活

混部时，TTFT（首 token 延迟）和 ITL（打字机流畅度）打架的根源：一个 8k prompt 的 prefill 是一整块几百毫秒的计算，期间 decode 全停。长请求一来，所有正在 streaming 的用户集体卡顿。

PD 分离的答案是物理隔离。但拆机器有代价：两套集群、KV 要跨机搬家、负载配比难调。**中小规模根本玩不起**。

## Chunked prefill 的答案：把大活切片

把长 prompt 的 prefill 切成固定大小的块（比如每块 512 token），分散到多个 step 里：

```text
没有切片：  [====== 8k prefill ======][decode decode decode]
                                            ↑ 此前全部干等
切了之后：  [chunk][chunk][decode][chunk][decode][chunk]...
```

每个 step 有个 **token 预算**：比如 4096。切好的 prefill 块和 decode 步装进同一个 step 一起算——GPU 从"要么忙死要么闲死"变成持续有活干，decode 再也不用等一整块 prefill。

## 为什么"一起算"反而是赚的

我一开始的疑惑：把 prefill 和 decode 混在一个 batch 里，不是互相拖累吗？想通靠这个观察：

- prefill 是**大而方**的矩阵乘（算力密集）；
- decode 是**细而长**的搬运（带宽密集）。

**同一个 step 里，一个喂饱算力、一个喂饱带宽——两种资源互补而不是竞争。**混合 step 的总利用率比纯 prefill 或纯 decode 都高。这是整篇笔记里我最喜欢的一个"想通了"时刻。

## 代价与旋钮

**延迟的重新分配。** TTFT 会略涨（prefill 拖长了），换 ITL 稳定（decode 不再被大块阻塞）。块切多小是旋钮：切小了 decode 丝滑但 prefill 尾巴长；切大了反之。Sarathi 的建议是块大小以"单个 step 不超过一个 decode 步的几倍"为准，让卡顿低于人的感知阈值（~100ms 级）。

**注意力计算的重复账。** 每个 chunk 算 attention 时要看到之前所有 chunk 的 KV——总计算量比整块 prefill 略多一点点（渐进还是线性，常数变大）。小钱，但不是零。

## 和 PD 分离怎么选

| | Chunked prefill | PD 分离 |
|---|---|---|
| 机器 | 同一套 | 两个池 |
| KV 传输 | 不用 | 必须（RDMA/缓存池） |
| 谁赚 | 中小规模、负载温和 | 超大规模、长上下文高并发 |
| 复杂度 | 单机调度 | 集群级系统工程 |

不是二选一：大厂玩法是**两者叠加**（分了池之后，池内部照样 chunk）。再叠上 [continuous batching](/blog/2026-08-15-continuous-batching/) 和 [GQA/MLA 的缓存压缩](/blog/2026-08-15-gqa-mla-notes/)，推理服务这栋楼的图纸就齐了。

## 一个后知后觉

本地跑开源模型的人其实天天在用这个：装个 vLLM 起服务，默认参数里就有 chunked prefill 的预算值。**框架把一个研究热点默默变成了默认配置**——读论文的意义之一就是认出你每天路过的东西。
