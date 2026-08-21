---
title: 西湖论剑数字安全大会平台
stack:
  - SpringBoot
  - Spring Cloud
  - Redis
  - XXL-Job
  - Elasticsearch
  - Kafka
  - MongoDB
  - Docker
order: -1
---

- 面向企业宣传、资讯发布和大会日程的数字平台，支持资讯发布、审核管理、用户互动等多功能模块
- 用户认证模块：利用双 token 实现无感刷新，优化用户体验并提高系统安全
- 文章渲染模块：使用 Freemarker 模板渲染引擎实现页面静态化，大幅减少数据库负担，提升平台响应速度与稳定性
- 文章发布模块：MinIO 保存图片素材及静态页面地址，Redis 完成资讯延迟发布，Kafka 异步监听通知文章上下架
- 热度定时计算：Redis 缓存热度数据保证高频访问快速响应，XXL-Job 定时任务每天凌晨 2 点自动计算文章热度，让热门文章及时被推荐
- 资讯检索模块：使用 Elasticsearch 基于倒排索引和分词机制实现搜索，MongoDB 记录搜索历史和联想词
- 实时数据处理：引入 Kafka Streams 对用户行为数据实时计算，动态调整热点资讯排名，提升资讯推荐的准确性和实时性
