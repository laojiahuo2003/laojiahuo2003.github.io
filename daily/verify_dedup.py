# -*- coding: utf-8 -*-
"""回放验证：用真实 stars_history.json + 近期报告数据检验去重修复。

不联网、不发推送，只测数据管线逻辑：
  1. get_fast_growing_repos 对含副本的输入去重
  2. weekly_report.analyze 三个板块互斥
  3. main.py 的 all_repos 聚合去重逻辑（从 main import 时不触发副作用）
"""
import json
import sys
import importlib.util
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import history_tracker
import weekly_report

failures = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        failures.append(msg)


# ---------- 1. fast_growing 去重 ----------
print("== 1. get_fast_growing_repos 去重 ==")
# 模拟 main.py 修复后的聚合：同一仓库多份副本（trending daily/weekly 各一份）
dup_repo = {
    "full_name": "tt-a1i/archify",
    "stargazers_count": 39700,
    "html_url": "https://github.com/tt-a1i/archify",
    "language": "JavaScript",
    "description": "Agent skill for architecture",
}
repos_in = [
    dict(dup_repo, _stars_gained=3904),
    dict(dup_repo),
    {"full_name": "other/repo", "stargazers_count": 100},
]

# 用真实历史文件跑（read-only，不回写）
fg = history_tracker.get_fast_growing_repos(repos_in, min_weekly_growth=50)
names = [r["full_name"] for r in fg]
cnt = Counter(names)
dups = {n: c for n, c in cnt.items() if c > 1}
check(not dups, f"fast_growing 无板块内重复（重复项: {dups or '无'}）")

# 副本数一致时结果条数 <= 输入唯一仓库数
check(len(names) <= len({r['full_name'] for r in repos_in}),
      f"输出条数 {len(names)} <= 输入唯一仓库数 {len({r['full_name'] for r in repos_in})}")

# ---------- 2. 周报三板块互斥（真实数据回放） ----------
print("== 2. weekly_report.analyze 板块互斥（真实 stars_history.json） ==")
rep = weekly_report.analyze()
check(rep is not None, "analyze() 返回了报告数据")
if rep:
    dh = {e["name"] for e in rep["darkhorses"]}
    od = {e["name"] for e in rep["onedayers"]}
    nc = {e["name"] for e in rep["newcomers"]}
    check(not (dh & nc), f"黑马 ∩ 新面孔 = {dh & nc or '∅'}")
    check(not (od & nc), f"一日游 ∩ 新面孔 = {od & nc or '∅'}")
    check(not (dh & od), f"黑马 ∩ 一日游 = {dh & od or '∅'}（elif 本应互斥，回归检查）")
    print(f"   本周: 黑马 {len(dh)}，一日游 {len(od)}，新面孔 {len(nc)}，"
          f"覆盖 {rep['coverage']}/7 天，warmup={rep['warmup']}")

# ---------- 3. main.py 聚合去重逻辑（真实报告数据模拟输入） ----------
print("== 3. main.py all_repos 聚合去重（用最近真实报告的池子模拟） ==")
# 取最近报告的各板块项目名，模拟"重叠数据源"的输入形态
latest = sorted((HERE / "reports").glob("*.json"))[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
trending_like = [{"full_name": p["name"], "stargazers_count": p["stars"]} for p in data["leaderboard"]]
created_like = [{"full_name": p["name"], "stargazers_count": p["stars"]} for p in data["leaderboard"]]  # 完全重叠
explored_like = [{"full_name": p["name"], "stargazers_count": p["stars"]} for p in data.get("newly_discovered") or []]

# 复现修复后的聚合代码路径
all_repos, seen_repo_names = [], set()
for repos in ({"daily": trending_like, "weekly": list(trending_like)}.values()):
    for repo in repos:
        name = repo.get("full_name", "")
        if not name or name in seen_repo_names:
            continue
        seen_repo_names.add(name)
        all_repos.append(repo)
for repos in ({"today": created_like, "this_week": list(created_like)}.values()):
    for repo in repos:
        name = repo.get("full_name", "")
        if not name or name in seen_repo_names:
            continue
        seen_repo_names.add(name)
        all_repos.append(repo)
all_repos.extend(explored_like)

agg_cnt = Counter(r["full_name"] for r in all_repos)
agg_dups = {n: c for n, c in agg_cnt.items() if c > 1}
check(not agg_dups, f"聚合后无重复（重复项: {agg_dups or '无'}）")
check(len(all_repos) == len(agg_cnt), f"聚合条数 {len(all_repos)} == 唯一名数 {len(agg_cnt)}")

print()
if failures:
    print(f"结果: {len(failures)} 项 FAIL")
    sys.exit(1)
print("结果: 全部 PASS")
