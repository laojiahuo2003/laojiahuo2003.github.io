---
title: 用 GitHub Actions 养了一个会自己更新的主页
description: 每小时刷新的天气、作息钟和贪吃蛇：一整套卡片的设计与实现。
pubDate: 2026-03-15
tags: [自动化]
---

我的 GitHub 个人主页上没有一张静态图：天气卡每小时抓 open-meteo 的实时数据重新生成，作息钟统计 GitHub Events API 里最近 90 天的活动时间，贡献贪吃蛇每天爬一遍新的热力图。全部由一个 cron 每小时触发的 Actions 工作流驱动，最后由 bot 提交回仓库。

## 为什么是 SVG 而不是徽章

 shields.io 徽章是个人主页的"默认皮肤"，信息密度低，长得也都一样。自己生成 SVG 有三个好处：

1. **双主题原生支持**——GitHub README 的 `<picture>` 标签配 `prefers-color-scheme` 媒体查询，浅色深色各出一张图；
2. **完全的排版控制**——中文字体、栅格、留白都按自己的规范来；
3. **动效可以藏在 SVG 里**——CSS keyframes 写进 SVG，README 里直接能动。

## 整条链路

```yaml
on:
  schedule:
    - cron: "23 * * * *"
```

每小时 23 分触发（避开整点的调度高峰），Python 脚本分别调 GitHub API、open-meteo、Traffic API，渲染出日夜各一张 SVG，bot 提交推送。README 里的 `<picture>` 引用 `raw.githubusercontent.com` 的地址，推送完两分钟内页面就是新的。

## 踩过的三个坑

**时区。** CI 跑在 UTC 上，作息钟要按北京时间统计"几点在写代码"，不显式 `+8` 的话所有时间段偏移 8 小时，凌晨型直接变早起型。

**缓存。** camo 代理会缓存图片，卡上写"实时"是撒谎——改成标注"更新于 HH:MM"（北京时间），承诺只承诺到能做到的程度。

**机器人自触发。** bot 的提交会再次触发 workflow，`paths:` 过滤器只监听脚本文件的变更，让循环停下来。

## 价值观

个人主页应该是**活的**：它该反映"我现在在干什么"，而不是"我注册那天在干什么"。一套零成本的 cron + 脚本，就能让静态页面有生命体征——这套思路搬到任何静态站上都成立。
