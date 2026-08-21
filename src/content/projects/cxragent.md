---
title: CXRAgent：胸部 X 光智能诊断框架
role: 第一作者 · IEEE TMI（SCI 1 区）已发表
period: "2025.03 – 2026.03"
stack:
  - LLM Agent
  - 多智能体协作
  - 工具调用与验证
  - 医学影像 VQA
links:
  - label: 论文
    url: https://ieeexplore.ieee.org/abstract/document/11611226
order: 4
---

- 针对胸部 X 光诊断中专业模型适应性差、多工具协同不可靠和系统流程僵化的问题，提出三阶段动态分析机制（工具调用与验证 → 诊断策略规划 → 专家团队协同决策），实现自适应胸片诊断
- 提出证据驱动的验证器：分析工具输出与影像特征的关联证据及置信度评估，修正工具结论，提高诊断可信度
- 提出 LLM 驱动的自适应团队协作机制：支持 Relay / Dispatch / Probe / Skip 四种协作策略动态切换，智能组建多学科专家团队（数量 / 角色 / 任务自主配置），实现临床诊断流程的自主优化
- 实验设计：在 MIMIC-CXR 与 CheXBench 上分别验证报告生成和多种视觉问答任务的能力
