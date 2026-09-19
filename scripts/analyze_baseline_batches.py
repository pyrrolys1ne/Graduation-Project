"""数据可信度清理：``fixed_baselines`` 与 ``overload_baselines`` 为什么差 12.7%–52.8%？

问题
----
这两个目录是**逐字节相同的配置**（除 `logging.output_dir`），跑的是同一份负载
（`saturating_4ms.jsonl`，400 请求，并发 8，`proxy` 后端），各 5 次重复。
但结果相差 12.7%–52.8%，且**显著性会翻转**——用哪一批数据决定 `dacc` 与 `multistream_fcfs`
是否可区分。`docs/实验记录.md` 的不同章节分别引用了这两批。

已定位到的唯一差异
------------------
两批的运行时间与标定表的写入时间对得上：

======  ====================  ==============================================
批次     运行时间               用的 `data/profiles/default.csv`
======  ====================  ==============================================
overload  2026-09-14 **13:06**   `2baeaff` 版本（12:24 提交）—— 预测值高
fixed     2026-09-14 **19:37**   `5477d18` 版本（**19:11 写入**）—— 预测值低
======  ====================  ==============================================

两个 config 都写 `table_path: data/profiles/default.csv`，所以**配置相同、文件内容变了**。
这是一次自然实验，但**与会话时间混淆**。本脚本做三件事：

1. 量化两批的差异并做配对检验；
2. 判断**哪些结论在两批中都成立**（只有这些能写进论文）；
3. 用"读表 / 不读表"把策略分组——若差异集中在读表的策略上，则表的嫌疑最大。
   （真正的因果判定由 `scripts/experiment_table_swap_ab.py` 在**同一会话内**换表完成。）

用法
----
::

    python scripts/analyze_baseline_batches.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BATCHES = {"overload_baselines": "09-14 13:06（旧标定表）",
           "fixed_baselines": "09-14 19:37（新标定表）"}

#: 只有这两个策略会走 ``Scheduler.choose_quota`` 的查表路径
#: （见 ``encoder_sched/scheduler.py``：非 edf_size/dacc 直接返回满配额）。
TABLE_READING = {"edf_size", "dacc"}

METRICS = ("throughput_rps", "total_latency_ms", "slo_violation_rate")


def load_runs(exp: str, policy: str) -> list[dict]:
    runs = []
    for i in range(10):
        path = REPO / f"results/{exp}/{policy}/repeat{i}.summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append(data)
    return runs


def metric_of(run: dict, metric: str) -> float:
    value = run[metric]
    return value["mean"] if isinstance(value, dict) else float(value)


def welch(a: list[float], b: list[float]) -> float:
    """Welch t 统计量的绝对值。两组样本量与方差都不等，不能用合并方差。"""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = statistics.variance(a), statistics.variance(b)
    denom = (va / len(a) + vb / len(b)) ** 0.5
    return abs(statistics.fmean(a) - statistics.fmean(b)) / denom if denom else float("nan")


def main() -> None:
    policies = sorted({p.name for p in (REPO / "results/fixed_baselines").iterdir()
                       if (p / "repeat0.summary.json").exists()})
    print(f"策略: {policies}\n")

    raw: dict[str, dict[str, list[dict]]] = {}
    for exp in BATCHES:
        raw[exp] = {p: load_runs(exp, p) for p in policies}
        counts = {p: len(raw[exp][p]) for p in policies}
        print(f"{exp:>20} ({BATCHES[exp]}): 每策略 {sorted(set(counts.values()))} 次重复")

    print(f"\n{'='*112}")
    print("吞吐率（req/s）：逐次重复值")
    print(f"{'='*112}")
    print(f"{'策略':>18} {'读写表':>7} | {'overload (旧表)':>34} | {'fixed (新表)':>34} | "
          f"{'中位比':>8} {'Welch t':>8}")
    print("-" * 112)
    rows = []
    for p in policies:
        a = [metric_of(r, "throughput_rps") for r in raw["overload_baselines"][p]]
        b = [metric_of(r, "throughput_rps") for r in raw["fixed_baselines"][p]]
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        t = welch(a, b)
        tag = "读表" if p in TABLE_READING else "不读"
        rows.append({"policy": p, "table_reading": p in TABLE_READING,
                     "old_median": ma, "new_median": mb, "ratio": mb / ma, "welch_t": t,
                     "old": a, "new": b})
        print(f"{p:>18} {tag:>7} | {str([round(x,1) for x in a]):>34} | "
              f"{str([round(x,1) for x in b]):>34} | {mb/ma:>7.3f}× {t:>8.2f}")

    reading = [r for r in rows if r["table_reading"]]
    control = [r for r in rows if not r["table_reading"]]
    print(f"\n  读表的策略 {[r['policy'] for r in reading]}：中位变化 "
          f"{[round(r['ratio'],3) for r in reading]}")
    print(f"  不读表的策略 {[r['policy'] for r in control]}：中位变化 "
          f"{[round(r['ratio'],3) for r in control]}")
    if reading and control:
        mr = statistics.fmean(r["ratio"] for r in reading)
        mc = statistics.fmean(r["ratio"] for r in control)
        print(f"  → 读表组平均变化 {mr:.3f}×，对照组 {mc:.3f}×，"
              f"相差 {(mr-1)/(mc-1):.2f} 倍" if mc != 1 else "")

    # ---- 结论稳定性：哪些排序在两批里都成立 ----
    print(f"\n{'='*112}")
    print("结论稳定性：只有两批都成立的排序才能写进论文")
    print(f"{'='*112}")
    order_old = [r["policy"] for r in sorted(rows, key=lambda r: -r["old_median"])]
    order_new = [r["policy"] for r in sorted(rows, key=lambda r: -r["new_median"])]
    print(f"  旧表批排名: {' > '.join(order_old)}")
    print(f"  新表批排名: {' > '.join(order_new)}")
    moved = [(p, order_old.index(p) + 1, order_new.index(p) + 1)
             for p in order_old if order_old.index(p) != order_new.index(p)]
    for p, i, j in moved:
        print(f"    ⚠ {p}: 第 {i} 名 → 第 {j} 名")

    # 策略两两比较：**每个批次内部**算 p 与 q 的 Welch t。
    # 注意不能拿各自"换表效应"的 t 来冒充——那是另一个问题（批次效应），
    # 会让"哪些结论稳定"整个判错。
    print(f"\n  {'对比':>34} {'旧表批 Welch t':>16} {'新表批 Welch t':>16} {'两批都>3?':>10}")
    print("  " + "-" * 84)
    stable = []
    unstable = []
    for i, p in enumerate(policies):
        for q in policies[i + 1:]:
            rp = next((r for r in rows if r["policy"] == p), None)
            rq = next((r for r in rows if r["policy"] == q), None)
            if not rp or not rq:
                continue
            za = welch(rp["old"], rq["old"])
            zb = welch(rp["new"], rq["new"])
            both = za > 3 and zb > 3
            (stable if both else unstable).append((p, q))
            print(f"  {p+' vs '+q:>34} {za:>16.2f} {zb:>16.2f} {'✅' if both else '❌':>10}")
    print(f"\n  两批都显著（可写进论文）: {len(stable)} 组")
    if unstable:
        print(f"  ⚠ 批次相关的（不可写进论文）: {unstable}")

    out = REPO / "results/baseline_batch_audit.json"
    out.write_text(json.dumps({
        "batches": BATCHES,
        "table_reading_policies": sorted(TABLE_READING),
        "policies": rows,
        "rank_old": order_old, "rank_new": order_new,
        "stable_pairs": stable,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    sys.exit(main())
