---
title: GitHub Profile 自动化主页
role: 独立开发
period: "2026.03 - 至今"
stack:
  - Python
  - SVG
  - GitHub API
  - Actions
links:
  - label: 效果
    url: https://github.com/laojiahuo2003
order: 1
---

- 用脚本生成整套 Profile 主页卡片：报头、打字机自述、足迹统计、趋势雷达、天气等，全部以 SVG 内嵌 CSS 动画实现，支持明暗双主题自适应
- 基于 GitHub REST/GraphQL API 聚合贡献数据（年度提交、连续天数、星标总量、最近活动），Actions 定时刷新无需服务器
- 趋势雷达卡片直接消费自建日报系统的结构化数据，跨仓库数据联动
