---
title: GitHub 趋势日报与黑马分析系统
role: 独立开发
period: "2026.04 - 至今"
stack:
  - Python
  - GitHub Actions
  - BeautifulSoup
  - Astro
links:
  - label: 日报仓库
    url: https://github.com/laojiahuo2003/github-daily-report
  - label: 线上日报
    url: https://laojiahuo2003.github.io/daily/
order: 2
---

- 构建 GitHub Trending 全自动日报系统：多路采集（趋势榜解析 / Search API 按语言与主题策略探索 / 新建仓库追踪），每日两次定时生成报告并推送微信通知
- 设计 30 天星标历史追踪器，逐日记录仓库星标轨迹，计算日/周增长，自动识别「持续黑马」与「一日游」项目
- 输出结构化 JSON feed 驱动个人网站的日报与周报页面，markdown / RSS / JSON 多格式分发
- 数据生命周期自动化：过期报告与陈旧仓库记录定时清理，仓库体积可控
