"""诊断 batch 实验日志里的"两次会话"：哪一次是无效的，以及为什么。

问题
----
`results/batch_comparison_mixed` 与 `results/batch_saturating_12ms` 的 `summary.json` 声明
``repeats: 5``，但 `logs/requests.jsonl` 里**实际有 10 次完整运行**，报告的正好是后 5 次
（逐位吻合）。运行令牌的时间间隔显示，那 10 次其实是**两个会话各 5 次**（间隔约 10 分钟）。

**数据审计最初把这件事读成"静默丢弃一半数据"。本脚本的诊断推翻了这个读法：**

前一会话的 `metadata.batch_size` **恒为 1.000**，后一会话才有正常批（1.05–3.33）。
即前一会话跑的是**组批未生效的代码**（`AGENTS.md` 第 7 节记录过这个缺陷：
"组批逻辑缺陷导致平均批大小恒为 1.00"）。**它的数据本来就无效，丢弃是对的。**

因此真正的问题**不是"丢了数据"，而是"没有留下记录"**：

1. `logs/requests.jsonl` 同时含两个会话，任何人拿它重算都会得到与 `summary.json`
   不同的数字（实测比值 0.47×–0.77×），而文件里**没有任何标记**说明该剔哪些；
2. `repeatN.jsonl` 已被后一会话覆盖，前一会话的原始数据**只存在于 logs**。

本脚本做什么
------------
1. 按运行令牌把 `logs/requests.jsonl` 切成逐次运行；
2. 按令牌时间间隔识别**会话边界**；
3. 逐会话报告均值延迟、分位数、SLO 违约率，**以及平均批大小**——后者是判定会话
   是否有效的关键：批处理变体上批大小 ≈ 1 说明组批没生效；
4. 给出每个会话的有效性判定，并对比"报告的子集"与"全部运行"。

**吞吐不在报告范围**：该负载下服务跟得上到达率，吞吐被到达率钉死
（实测 84.6 对 84.9 req/s），无区分度。延迟与 SLO 从原始日志精确可算。

用法
----
::

    python scripts/recover_batch_runs.py
    python scripts/recover_batch_runs.py --json results/batch_runs_recovered.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 需要恢复的实验。`reported` 是 summary.json 声明用了几次。
EXPERIMENTS = {
    "batch_comparison_mixed": {"workload": "comparison_mixed", "reported": 5},
    "batch_saturating_12ms": {"workload": "saturating_12ms", "reported": 5},
    "batch_saturating_4ms": {"workload": "saturating_4ms", "reported": 5},
}

VARIANTS = ("spatial_2stream", "batch4_d0", "batch4_d1", "batch4_d5", "batch4_d20", "batch8_d5")

#: 运行令牌之间的间隔超过这个值就认为换了一个会话（纳秒；实测会话间隔约 6×10^11）。
SESSION_GAP_NS = 60 * 10**9

RUN_TOKEN = re.compile(r"-run-(\d+)-")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def split_runs(path: Path) -> dict[int, list[dict]]:
    """按运行令牌切分。令牌在 ``request_id`` 里，形如 ``...-run-1789393697724403734-0``。"""
    runs: dict[int, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        match = RUN_TOKEN.search(row.get("request_id", ""))
        if not match:
            continue                      # warmup 与其它非本次运行的记录
        runs.setdefault(int(match.group(1)), []).append(row)
    return runs


def split_sessions(tokens: list[int]) -> list[list[int]]:
    """按时间间隔把运行令牌分成会话。"""
    sessions: list[list[int]] = []
    for token in sorted(tokens):
        if sessions and token - sessions[-1][-1] <= SESSION_GAP_NS:
            sessions[-1].append(token)
        else:
            sessions.append([token])
    return sessions


def describe(rows: list[dict]) -> dict:
    total = [float(r["total_ms"]) for r in rows]
    queue = [float(r["queue_ms"]) for r in rows]
    execution = [float(r["execution_ms"]) for r in rows]
    sizes = [r.get("metadata", {}).get("batch_size") for r in rows]
    sizes = [s for s in sizes if isinstance(s, int)]
    return {
        "n": len(rows),
        "total_mean_ms": statistics.fmean(total),
        "total_p50_ms": percentile(total, 0.50),
        "total_p95_ms": percentile(total, 0.95),
        "total_p99_ms": percentile(total, 0.99),
        "queue_mean_ms": statistics.fmean(queue),
        "execution_mean_ms": statistics.fmean(execution),
        "slo_violation_rate": sum(1 for r in rows if r.get("slo_violated")) / len(rows),
        "mean_batch_size": statistics.fmean(sizes) if sizes else None,
    }


def verdict(variant: str, stats: dict, other_batch_sizes: list[float]) -> str:
    """判定一个会话的数据是否有效。

    批处理变体上平均批大小 ≈ 1 说明**组批没有生效**——那样的数据不是批处理实验的结果，
    与同实验的其它会话不可比，应当剔除。
    ``spatial_2stream`` 不组批，批大小字段为 None，不适用此判据。
    """
    bs = stats["mean_batch_size"]
    if bs is None:
        return "不组批（无判据）"
    if abs(bs - 1.0) < 0.02:
        return "❌ 无效：组批未生效（批大小≈1）"
    if other_batch_sizes and bs < min(other_batch_sizes) * 0.6:
        return "⚠ 可疑：批大小显著小于同实验其它会话"
    return "✅ 有效"


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复 batch 实验里被静默丢弃的运行")
    parser.add_argument("--json", type=Path, default=REPO / "results/batch_runs_recovered.json")
    args = parser.parse_args()

    report: dict = {}
    for exp, meta in EXPERIMENTS.items():
        print(f"\n{'='*118}")
        print(f"{exp}   （负载 {meta['workload']}；summary.json 声明 repeats={meta['reported']}）")
        print(f"{'='*118}")
        report[exp] = {}
        for variant in VARIANTS:
            log = REPO / f"results/{exp}/{variant}/logs/requests.jsonl"
            if not log.exists():
                continue
            runs = split_runs(log)
            if not runs:
                continue
            sessions = split_sessions(list(runs))
            per_run = {t: describe(rows) for t, rows in runs.items()}
            session_stats = [describe([r for t in s for r in runs[t]]) for s in sessions]
            all_stats = describe([r for rows in runs.values() for r in rows])

            n_runs = len(runs)
            batch_sizes = [s["mean_batch_size"] for s in session_stats
                           if s["mean_batch_size"] is not None]
            print(f"\n  {variant}   （{n_runs} 次运行 / {len(sessions)} 个会话）")
            print(f"    {'会话':>5} {'运行数':>7} {'n':>6} {'mean':>9} {'p50':>8} {'p95':>8} "
                  f"{'p99':>8} {'SLO':>7} {'批大小':>8}   判定")
            for i, (session, st) in enumerate(zip(sessions, session_stats)):
                bs = f"{st['mean_batch_size']:.3f}" if st["mean_batch_size"] is not None else "—"
                others = [b for b in batch_sizes if b != st["mean_batch_size"]]
                print(f"    {i:>5} {len(session):>7} {st['n']:>6} {st['total_mean_ms']:>9.3f} "
                      f"{st['total_p50_ms']:>8.2f} {st['total_p95_ms']:>8.2f} "
                      f"{st['total_p99_ms']:>8.2f} {st['slo_violation_rate']:>7.3f} {bs:>8}   "
                      f"{verdict(variant, st, others)}")

            reported = session_stats[-1] if len(sessions) > 1 else all_stats
            print(f"    {'':>5} {'—':>7} 报告的值: mean={reported['total_mean_ms']:.3f} "
                  f"p99={reported['total_p99_ms']:.2f} SLO={reported['slo_violation_rate']:.3f}")
            if len(sessions) > 1:
                print(f"    {'':>5} {'—':>7} 全部运行: mean={all_stats['total_mean_ms']:.3f} "
                      f"p99={all_stats['total_p99_ms']:.2f} SLO={all_stats['slo_violation_rate']:.3f}"
                      f"   （报告/全部 = {reported['total_mean_ms']/all_stats['total_mean_ms']:.3f}×）")
                print(f"    {'':>5} ⚠ 直接拿 logs/requests.jsonl 重算会得到与 summary.json 不同的数字，"
                      f"而文件里没有任何标记")

            report[exp][variant] = {
                "n_runs": n_runs,
                "n_sessions": len(sessions),
                "sessions": [[t for t in s] for s in sessions],
                "session_verdicts": [verdict(variant, s, [b for b in batch_sizes
                                                          if b != s["mean_batch_size"]])
                                     for s in session_stats],
                "per_run": {str(t): st for t, st in per_run.items()},
                "session_stats": session_stats,
                "reported_subset": reported,
                "all_runs": all_stats,
            }

    # ---- 跨变体：结论是否翻转 ----
    print(f"\n{'='*118}")
    print("结论稳定性：把全部运行算进去后，变体排序是否变化")
    print(f"{'='*118}")
    for exp in EXPERIMENTS:
        if exp not in report:
            continue
        rows = report[exp]
        def order(key):
            items = [(v, d[key]["total_mean_ms"]) for v, d in rows.items()
                     if d["n_sessions"] > 1]
            return [v for v, _ in sorted(items, key=lambda x: x[1])]
        if not order("reported_subset"):
            print(f"  {exp}: 单会话（无被丢弃运行），无需比较")
            continue
        o_rep, o_all = order("reported_subset"), order("all_runs")
        print(f"\n  {exp}")
        print(f"    报告子集排序（延迟低→高）: {' < '.join(o_rep)}")
        print(f"    全部运行排序              : {' < '.join(o_all)}")
        moved = [(v, o_rep.index(v) + 1, o_all.index(v) + 1) for v in o_rep
                 if o_rep.index(v) != o_all.index(v)]
        for v, a, b in moved:
            print(f"      ⚠ {v}: 第 {a} 名 → 第 {b} 名")
        if not moved:
            print("      ✓ 排序不变")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
