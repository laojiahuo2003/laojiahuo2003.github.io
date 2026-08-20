#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周报生成器：基于 stars_history.json 的 30 天星标轨迹，产出每周黑马分析。

产出 src/data/weekly.json（网站 /weekly/ 页数据源）：
  darkhorses  本周黑马——周增长 >= 50 且 7 天里出现 >= 4 天（持续上涨，非一日游）
  onedayers   一日游——周增长 >= 100 但只出现 <= 2 天（脉冲式上榜）
  newcomers   新面孔——本周首次进入追踪视野且已有一定星标
  langs       黑马 + 新面孔的语言分布

数据不足整周时（迁移后预热期）置 warmup=true，阈值按覆盖天数折算。
用法：python daily/weekly_report.py（由 weekly.yml 每周一 09:30 调度）
"""
import json
import os
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "history", "stars_history.json")
REPORTS_DIR = os.path.join(HERE, "reports")
OUT_FILE = os.path.join(HERE, "..", "src", "data", "weekly.json")
KEEP_WEEKS = 12

DARKHORSE_MIN_GROWTH = 50      # 周增长门槛
DARKHORSE_MIN_DAYS = 4         # 7 天中至少出现的天数（持续度）
ONEDAYER_MIN_GROWTH = 100      # 一日游的增长门槛
ONEDAYER_MAX_DAYS = 2
NEWCOMER_MIN_STARS = 50


def load_history():
    with open(HISTORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_language_map():
    """从近期日报 JSON 聚合 name -> language（历史文件不存语言）"""
    langs = {}
    try:
        files = sorted(f for f in os.listdir(REPORTS_DIR) if f.endswith(".json"))
    except OSError:
        return langs
    for fn in files[-14:]:  # 最近 7 天（每天 2 份）
        try:
            with open(os.path.join(REPORTS_DIR, fn), encoding="utf-8") as f:
                rep = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pools = list(rep.get("leaderboard") or []) + list(rep.get("newly_discovered") or [])
        for g in (rep.get("by_category") or []) + (rep.get("new_projects") or []) + (rep.get("explored") or []):
            pools += g.get("projects") or []
        for p in pools:
            if p.get("name") and p.get("language"):
                langs[p["name"]] = p["language"]
    return langs


def analyze():
    history = load_history()
    lang_map = build_language_map()

    all_dates = sorted({d for r in history.values() for d in r.get("stars_history", {})})
    if not all_dates:
        return None
    week_end = date.fromisoformat(all_dates[-1])
    week_start = week_end - timedelta(days=6)
    window = {(week_start + timedelta(days=i)).isoformat() for i in range(7)}
    coverage = len(window & set(all_dates))
    warmup = coverage < 5
    # 预热期按覆盖天数折算持续度门槛（覆盖 2 天时要求 2 天都在）
    min_days = DARKHORSE_MIN_DAYS if not warmup else max(2, coverage - 1)

    darkhorses, onedayers, newcomers = [], [], []
    for name, rec in history.items():
        sh = rec.get("stars_history", {})
        cur = rec.get("current_stars", 0)
        seen = sorted(d for d in sh if d in window)
        if not seen:
            continue
        days_seen = len(seen)
        # 基线：窗口开始前最近一次记录；没有则用窗口内最早一次（首见即基线，预热期可接受）
        before = [d for d in sh if d < week_start.isoformat()]
        base_date = max(before) if before else seen[0]
        if base_date == seen[-1] and days_seen == 1 and not before:
            # 只有一天记录且无历史：增长无从谈起，跳过
            continue
        growth = cur - sh[base_date]
        first_seen = rec.get("first_seen", "")

        entry = {
            "name": name,
            "url": f"https://github.com/{name}",
            "stars": cur,
            "growth": growth,
            "days_seen": days_seen,
            "language": lang_map.get(name, ""),
            "first_seen": first_seen,
        }

        if growth >= DARKHORSE_MIN_GROWTH and days_seen >= min_days:
            darkhorses.append(entry)
        elif growth >= ONEDAYER_MIN_GROWTH and days_seen <= ONEDAYER_MAX_DAYS and not warmup:
            onedayers.append(entry)

        if week_start.isoformat() <= first_seen <= week_end.isoformat() and cur >= NEWCOMER_MIN_STARS:
            newcomers.append(entry)

    darkhorses.sort(key=lambda x: -x["growth"])
    onedayers.sort(key=lambda x: -x["growth"])
    newcomers.sort(key=lambda x: -x["stars"])

    langs = {}
    for e in darkhorses[:15] + newcomers[:10]:
        if e["language"]:
            langs[e["language"]] = langs.get(e["language"], 0) + 1

    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "coverage": coverage,
        "warmup": warmup,
        "darkhorses": darkhorses[:15],
        "onedayers": onedayers[:10],
        "newcomers": newcomers[:10],
        "langs": dict(sorted(langs.items(), key=lambda kv: -kv[1])),
    }


def main():
    report = analyze()
    if not report:
        print("无历史数据，跳过周报生成")
        return

    # 归档：同周覆盖更新，跨周追加，最多保留 KEEP_WEEKS 份
    reports = []
    if os.path.exists(OUT_FILE):
        try:
            with open(OUT_FILE, encoding="utf-8") as f:
                reports = json.load(f).get("reports", [])
        except (OSError, json.JSONDecodeError):
            reports = []
    reports = [r for r in reports if r["week_start"] != report["week_start"]]
    reports.insert(0, report)
    reports = reports[:KEEP_WEEKS]

    payload = {"updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), "reports": reports}
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"周报已生成: {report['week_start']} ~ {report['week_end']}"
          f"（黑马 {len(report['darkhorses'])}，一日游 {len(report['onedayers'])}，"
          f"新面孔 {len(report['newcomers'])}，覆盖 {report['coverage']}/7 天"
          f"{'，预热期' if report['warmup'] else ''}）")


if __name__ == "__main__":
    main()
