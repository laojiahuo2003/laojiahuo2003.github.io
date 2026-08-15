---
title: PD 分离学习笔记：一台机器伺候不好两种脾气
description: prefill 吃算力、decode 吃带宽，拆开部署为什么赢了，拆完又欠下什么新债。
pubDate: 2026-08-15
tags: [推理, 笔记]
---

[上一篇 KV Cache 笔记](/blog/2026-08-15-kv-cache-notes/)结尾留了个坑：prefill 和 decode 为什么"一个算力密集一个带宽密集，是 PD 分离部署的由来"。这篇就是填坑的。读完最大的感受：**这是一次微服务拆分在 GPU 集群上的完整重演**。

## 先把两种负载的性格摆出来

LLM 推理其实是两段完全不同的活：

**Prefill（预填充）**：把整个 prompt 一次性喂进去，算出所有位置的 KV。几百上千个 token 一起算，矩阵乘又大又方——**算力密集**，GPU 干得热火朝天，卡的就是 FLOPS。

**Decode（解码）**：一个 token 一个 token 往外蹦。每步只算一个新 token，但要搬全部权重和 KV Cache——上篇推过的结论：**显存带宽瓶颈**，算力大量闲置。

一个像举重运动员，一个像长跑运动员。混部在同一张卡上，就是让一个人同时练两种。

## 混部的真实代价：互相制造延迟尖刺

关键冲突在两个 SLO 指标天然打架：

- **TTFT**（Time To First Token）：用户发出请求到第一个字出来的时间——prefill 决定；
- **ITL/TPOT**（每 token 间隔）：打字机效果稳不稳——decode 决定。

混部时，一个长 prompt 的 prefill 进来，是一次几百毫秒到几秒的**大计算块**，排在它后面的所有 decode 步全部干等——正在 streaming 的用户那边，打字机突然卡住。反过来，为了保 decode 流畅把 prefill 切碎，TTFT 又崩了。**在单一资源上，这两个指标是跷跷板**。

PD 分离的答案很直接：prefill 池和 decode 池物理分开，各自按自己的负载选硬件、配比例、做调度。跷跷板两头各自落地。

## 拆完欠的新债：KV Cache 要搬家

麻烦立刻出现：prefill 算出的 KV Cache，decode 机器上没有。**必须传过去**。

传多少？上篇手算过 70B 模型是 2.6 MB/token——一个 8k prompt 的请求就是 **20 GB 出头**。用普通 TCP 走以太网，传一次的时间比算一次还长，分离的收益全填了这条路。

所以这个方向的所有工程都在围绕同一件事：**让 KV 搬家便宜**。我看到的几层手法：

1. **RDMA/InfiniBand 直传**：绕过内核协议栈，GPU 显存到网卡的高速公路；
2. **分层缓存池**：KV 不只存在于 GPU——CPU DRAM、SSD 都是它的家，热的在显存、温的在内存、冷的在盘上（Mooncake 把这叫 KVCache-centric 架构：先想清楚缓存住哪，再谈计算谁做）；
3. **前缀复用**：多轮对话第二轮的 prompt 和第一轮高度重叠，重叠部分的 KV 根本不用重算也不用传——找个地方存着就行。

## 2026 年的现状：从赌注到标配

最有意思的读物是 DistServe 团队自己写的[《Disaggregated Inference: 18 Months Later》](https://haoailab.com/blogs/distserve-retro/)——当年论文是把"拆开"当成研究赌注，18 个月回看，它已经成了大规模 serving 的事实标准：

- **Mooncake**（Kimi 的 serving 平台）是最大规模的生产案例：以 KV Cache 池为中心组织整个集群，把 CPU/DRAM/SSD/RDMA 的闲置资源全部利用起来（[GitHub](https://github.com/kvcache-ai/Mooncake)）；
- **DeepSeek** 自家生产环境也在跑 PD 分离（配合专家并行）；
- **Meta** 上了分离架构；**AWS SageMaker HyperPod** 做成了托管能力，长上下文高并发收益最大；
- 开源框架 **vLLM、SGLang、llm-d** 都把 PD 分离写成一等公民的部署模式。

研究前沿也往前走了：多轮对话在分离集群上的 KV 复用（arXiv 2602.14516）、流式感知的分离调度（Stream2LLM）、甚至**跨数据中心**传 KV——把"缓存住哪"这个问题问到了广域网上。

## 我的理解钩子：微服务剧本逐字重演

把时间线摆开看特别眼熟：

```text
单体应用（混合部署）
  → 按负载特征拆服务（prefill 池 / decode 池）
    → 引入服务间通信成本（KV 传输）
      → 通信成本催生基础设施（RDMA、缓存池）
        → 基础设施反过来改变架构（以缓存为中心组网）
```

分布式系统的课在 GPU 集群上一节节重上。学 LLM Infra 的正确姿势可能确实是两门课一起上。

## 下一个坑

- chunked prefill：不彻底分离、但把 prefill 切片让路的中间路线，和 PD 分离怎么选？
- Mooncake 的全局调度器具体怎么决定"这个请求的 KV 放哪层"——缓存分层和调度是个联合优化问题。

继续挖，挖到再写。

## 出处

- [Disaggregated Inference: 18 Months Later — Hao AI Lab @ UCSD](https://haoailab.com/blogs/distserve-retro/)
- [Mooncake — GitHub (kvcache-ai)](https://github.com/kvcache-ai/Mooncake)
- [Efficient Multi-round LLM Inference over Disaggregated Serving — arXiv](https://arxiv.org/html/2602.14516v1)
- [Disaggregated Prefill and Decode on SageMaker HyperPod — AWS](https://aws.amazon.com/blogs/machine-learning/disaggregated-prefill-and-decode-for-llm-inference-on-sagemaker-hyperpod/)
