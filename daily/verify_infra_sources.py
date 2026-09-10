# -*- coding: utf-8 -*-
"""端到端验证：真实抓取新数据源 → 生成报告结构 → 断言 infra 信号与去重。

不发微信推送、不写 reports/ 目录、不回写 stars_history.json。
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from fetchers.trending import fetch_all_trending
from fetchers.search import fetch_created_repos, explore_all
import main as daily_main
import history_tracker

failures = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        failures.append(msg)


print("== 1. 数据源连通性 ==")
trending = fetch_all_trending()
for k in ["daily", "weekly", "monthly", "cuda"]:
    n = len(trending.get(k) or [])
    check(n > 0, f"trending[{k}] 抓到 {n} 条")

created = fetch_created_repos()
for k in ["today", "this_week", "llm_week", "this_month"]:
    n = len(created.get(k) or [])
    check(n >= 0, f"created[{k}] 抓到 {n} 条")

print("== 2. infra 信号质量 ==")
cuda_names = [r["full_name"] for r in trending.get("cuda", [])]
print("   CUDA 榜:", ", ".join(cuda_names[:6]))
llm_week = created.get("llm_week", [])
print("   llm_week 前5:", ", ".join(f"{r['full_name']}({r['stargazers_count']}⭐)" for r in llm_week[:5]))
check(len(cuda_names) > 0, "CUDA 榜非空")
check(len(llm_week) > 0, "llm_week 非空")

print("== 3. 报告生成 + 去重回归 ==")
data = daily_main.build_report_data(trending, created, [], [], [], "2026-09-10")

occ = Counter()
secs = {}
def add(sec, items):
    for p in items or []:
        if p.get("name"):
            occ[p["name"]] += 1
            secs.setdefault(p["name"], set()).add(sec)

add("leaderboard", data.get("leaderboard"))
add("fast_growing", data.get("fast_growing"))
add("newly_discovered", data.get("newly_discovered"))
for g in data.get("by_category") or []:
    add(f"by_category/{g['category']}", g.get("projects"))
for g in data.get("new_projects") or []:
    add(f"new_projects/{g['category']}", g.get("projects"))

internal = {n: c for n, c in occ.items() if c > 1 and len(secs[n]) == 1}
check(not internal, f"板块内无重复（重复项: {internal or '无'}）")
print(f"   各板块项目总数: {sum(occ.values())}，唯一项目: {len(occ)}")

print()
if failures:
    print(f"结果: {len(failures)} 项 FAIL")
    sys.exit(1)
print("结果: 全部 PASS")
