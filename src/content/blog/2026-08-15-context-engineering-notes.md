---
title: Context Engineering 学习笔记：从"写好提示词"到"管理信息环境"
description: 2026 年 Agent 圈的方法论共识：Write / Select / Compress / Isolate 四个动词，一张我的 cheatsheet。
pubDate: 2026-08-15
tags: [Agent]
---

这两年看 Agent 相关的讨论，"prompt engineering"这个词出现得越来越少，取而代之的是 **context engineering**。这不止是换个时髦词。这篇笔记记录我的理解：它到底新在哪，以及那套被反复引用的四动词框架。

## 为什么不叫 Prompt Engineering 了

提示词工程的隐含图景是：**一次调用**，把一段话打磨到极致。但真实的 Agent 是**长循环**——几十上百次工具调用，每次调用时模型看到的上下文都是动态拼出来的：系统提示 + 工具结果 + 历史消息 + 检索回来的文档 + 记忆。

问题从"这句话怎么写"变成了"**这一步该让模型看见什么**"。提示词只是众多上下文来源里的一种。词变了，是因为对象从句子变成了系统。

## 四个动词：Write / Select / Compress / Isolate

这套框架出自 [LangChain 的博客](https://blog.langchain.com/context-engineering-for-agents)，现在是事实上的通用语言。我的理解版：

| 动词 | 管什么 | 例子 |
|------|--------|------|
| **Write** | 往长期存储写什么 | 把这次任务的关键决定存进记忆，下轮可取 |
| **Select** | 每次调用往上下文里装什么 | 检索、裁剪工具输出、只带相关的历史 |
| **Compress** | 太长了怎么压 | 对话摘要、把旧的工具结果换成一句话结论 |
| **Isolate** | 不同上下文怎么隔离 | 子任务开子 agent，干完只把结论带回来 |

我自己的记忆钩子：这是**图书馆学**——写卡片（Write）、上架检索（Select）、做摘要（Compress）、分阅览室（Isolate）。

## 几个我认为被低估的点

**上下文是零和的。** 窗口再大，注意力也会被无关内容稀释——塞进去的每样东西都在给别的东西抢位置。所以 Select 的本质不是"多找点资料"而是**狠心扔东西**。工具输出动辄几万 token，原样进历史是最常见的浪费。

**Isolate 是子 agent 存在的真正理由。** 不是"多个 agent 更聪明"，而是**上下文污染需要隔离舱**：探索性任务会拉回大量垃圾，关在子 agent 里消化完，只回传结论，主循环保持干净。

**失败大多在 Select，而不是模型。** Agent 效果差时，第一反应不该是换更大的模型，而是打开日志看它当时到底看见了什么。很多"幻觉"其实是"上下文里没有相关信息，模型只能编"。

## 一张给自己的 cheatsheet

写 Agent 循环时按这个顺序自查：

1. 这一轮模型需要完成什么决策？（决定 Select 的靶子）
2. 为这个决策，上下文里少了什么？多了什么？
3. 工具输出进历史前，压缩了吗？
4. 这个任务会污染主循环吗？要不要 Isolate 出去？
5. 有什么这轮学到的东西值得 Write 下来？

## 出处

- [Context Engineering for Agents — LangChain Blog](https://blog.langchain.com/context-engineering-for-agents)（四动词框架出处）
- [Context Engineering Guide — Prompting Guide](https://www.promptingguide.ai/guides/context-engineering-guide)
- [Context Engineering: A Practical Guide for AI Agents — Sourcegraph](https://sourcegraph.com/blog/context-engineering)
