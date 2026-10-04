#!/usr/bin/env python3
"""并发度对照的负结果归因（2026-09-22 批次）。

回答三个问题，全部用已落盘的原始 JSONL，不需要 GPU：

1. 每请求耗时里有多少是 **GPU 时间**（`execution_ms`，CUDA event），多少是流水线开销？
2. GPU 时间随共驻数 N 怎么变——是"纯资源共享"（每请求时间随 N 线性增长、吞吐饱和），
   还是**超线性**恶化（每请求时间增长快于 N）？
3. `dynamic` 实际做了什么决策，代价落在哪个目标值上？

背景：`encoder_sched/encoder.py` 的 `execution_ms` 由 CUDA event 记录，
**只覆盖 GPU 上的前向**；CPU 侧的图像生成与 H2D 拷贝在事件之外。因此
`execution_ms` 是干净的 GPU 侧量，`1/吞吐 − execution/N` 是服务端流水线开销。

用法::

    .venv/bin/python scripts/probes/an5_concurrency_attribution.py \
        results/concurrency_openloop_2026-09-22
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ARMS = ("static_1", "static_2", "static_3", "static_4", "static_6", "static_8", "dynamic")
ARM_N = {"static_1": 1, "static_2": 2, "static_3": 3, "static_4": 4, "static_6": 6, "static_8": 8}


def load_runs(batch: Path, arm: str) -> list[list[dict]]:
    runs = []
    for path in sorted((batch / arm).glob("repeat*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        runs.append(rows)
    return runs


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    batch = Path(sys.argv[1] if len(sys.argv) > 1 else "results/concurrency_openloop_2026-09-22")
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    print(f"批次 {batch}：{summary['variants']['static_1']['repeats']} 轮/臂，"
          f"客户端连接上限 {summary['concurrency']}（0 = 开环）")

    print("\n" + "=" * 108)
    print("① 每请求耗时的分解：GPU 时间 vs 流水线开销")
    print("=" * 108)
    print(f"{'臂':<10}{'吞吐':>9}{'每请求ms':>10}{'GPU/N':>9}{'流水线ms':>10}"
          f"{'GPU占比':>9}{'Σexec/墙钟':>11}{'NVML%':>8}")
    detail: dict[str, dict] = {}
    for arm in ARMS:
        runs = load_runs(batch, arm)
        agg = summary["variants"][arm]["aggregate"]
        throughput = agg["throughput_rps"]["mean"]
        per_request = 1000.0 / throughput
        exec_mean = agg["server_execution_mean_ms"]["mean"]
        n_effective = ARM_N.get(arm)
        gpu_per_request = exec_mean / n_effective if n_effective else float("nan")
        pipeline = per_request - gpu_per_request if n_effective else float("nan")
        # Σexec / 墙钟 = 平均有多少个请求"正在 GPU 上执行"，即真实共驻数
        residency = []
        for rows in runs:
            wall_s = 1000.0 * len(rows) / throughput
            residency.append(sum(r["execution_ms"] or 0.0 for r in rows) / wall_s)
        detail[arm] = {
            "throughput": throughput,
            "per_request": per_request,
            "gpu_per_request": gpu_per_request,
            "pipeline": pipeline,
            "residency": statistics.fmean(residency),
        }
        print(f"{arm:<10}{throughput:>9.1f}{per_request:>10.2f}{gpu_per_request:>9.2f}"
              f"{pipeline:>10.2f}{gpu_per_request / per_request * 100:>8.0f}%"
              f"{detail[arm]['residency']:>11.2f}{agg['gpu_utilization_percent']['mean']:>8.1f}")

    print("\n读法：`流水线ms` = 每请求的 1/吞吐 减去它分摊到的 GPU 时间；")
    print("      `Σexec/墙钟` 是整个运行里平均有几个请求真的在 GPU 上执行（真实共驻数）。")
    print("      `NVML%` 是 nvidia-smi 报的利用率——若它与 `Σexec/墙钟` 矛盾，说明该指标不可用。")

    print("\n" + "=" * 108)
    print("② GPU 时间随共驻数的缩放：纯资源共享，还是超线性恶化？")
    print("=" * 108)
    sizes = sorted({(r["width"], r["height"]) for r in load_runs(batch, "static_1")[0]})
    header = "".join(f"{f'{w}x{h}':>11}" for w, h in sizes)
    print(f"{'臂':<10}{'N':>3}{'全部 中位':>11}{'每请求/N':>11}{'相对N=1':>10}{header}")
    baseline: dict[tuple[int, int], float] = {}
    baseline_overall = float("nan")
    for arm in ("static_1", "static_2", "static_3", "static_4", "static_6", "static_8"):
        n_effective = ARM_N[arm]
        per_size: dict[tuple[int, int], list[float]] = {}
        all_exec: list[float] = []
        for rows in load_runs(batch, arm):
            for row in rows:
                if row.get("execution_ms"):
                    per_size.setdefault((row["width"], row["height"]), []).append(row["execution_ms"])
                    all_exec.append(row["execution_ms"])
        medians = {size: median(values) for size, values in per_size.items()}
        if arm == "static_1":
            baseline = dict(medians)
            baseline_overall = median(all_exec)
        overall = median(all_exec)
        cells = "".join(
            f"{medians.get(size, float('nan')) / baseline.get(size, float('nan')):>10.2f}×"
            for size in sizes
        )
        print(f"{arm:<10}{n_effective:>3}{overall:>11.2f}{overall / n_effective:>11.2f}"
              f"{overall / baseline_overall:>9.2f}×{cells}")
    print("\n读法：最后一组列是 `该臂 GPU 时间 / static_1 的 GPU 时间`（按尺寸分别比）。")
    print("      若并发只是「共享同一份带宽」，每请求时间应随 N 线性增长、吞吐饱和；")
    print("      若相对比值**超过 N**，说明多出来的部分是并发本身引入的额外低效。")

    print("\n" + "=" * 108)
    print("③ dynamic 实际选了什么（准入日志）")
    print("=" * 108)
    targets: dict[int, int] = {}
    reasons: dict[str, int] = {}
    exec_by_resident: dict[int, list[float]] = {}
    for rows in load_runs(batch, "dynamic"):
        for row in rows:
            admission = (row.get("metadata") or {}).get("admission") or {}
            target = admission.get("target")
            if target is not None:
                targets[target] = targets.get(target, 0) + 1
            reason = admission.get("reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            resident = admission.get("resident_at_admit")
            if resident is not None and row.get("execution_ms"):
                exec_by_resident.setdefault(resident, []).append(row["execution_ms"])
    total = sum(targets.values()) or 1
    print("准入时的目标共驻数分布：")
    for target, count in sorted(targets.items()):
        print(f"  target={target}: {count:>5} 次 ({count / total * 100:>5.1f}%)")
    print("准入原因分布：")
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {reason:<24}{count:>5} 次 ({count / total * 100:>5.1f}%)")
    print("准入时已在执行的请求数 → 该请求的 GPU 时间中位：")
    for resident, values in sorted(exec_by_resident.items()):
        print(f"  resident={resident}: n={len(values):>4}  中位 {median(values):>7.2f} ms")

    print("\n" + "=" * 108)
    print("④ 工作负载的尺寸构成")
    print("=" * 108)
    first = load_runs(batch, "static_1")[0]
    mix: dict[tuple[int, int], int] = {}
    for row in first:
        mix[(row["width"], row["height"])] = mix.get((row["width"], row["height"]), 0) + 1
    for size, count in sorted(mix.items()):
        print(f"  {size[0]}x{size[1]}: {count:>4} ({count / len(first) * 100:>5.1f}%)")
    small = sum(count for size, count in mix.items() if (size[0] // 32) * (size[1] // 32) <= 100)
    print(f"  小请求（patch ≤ 100，即 224/336）：{small}/{len(first)} = {small / len(first) * 100:.1f}%")

    print("\n" + "=" * 108)
    print("⑤ 按尺寸推算的最优共驻数（用干净的静态臂数据：worker 与 stream 是 1:1）")
    print("=" * 108)
    exec_by_size: dict[tuple[int, int], dict[int, float]] = {}
    for arm in ("static_1", "static_2", "static_3", "static_4", "static_6", "static_8"):
        per_size: dict[tuple[int, int], list[float]] = {}
        for rows in load_runs(batch, arm):
            for row in rows:
                if row.get("execution_ms"):
                    per_size.setdefault((row["width"], row["height"]), []).append(row["execution_ms"])
        for size, values in per_size.items():
            exec_by_size.setdefault(size, {})[ARM_N[arm]] = median(values)
    # 流水线开销按①里 static_2 的残差取 3.2 ms；它对尺寸不敏感的部分主要是
    # HTTP/事件循环/线程调度，尺寸相关部分（CPU 造图）会被低估，故说明为下界估计。
    PIPELINE_MS = 3.2
    print(f"（流水线开销按 {PIPELINE_MS} ms/请求 计，属下界估计）")
    print(f"{'尺寸':<10}{'单请求GPU ms':>14}{'最优N':>7}{'预估吞吐':>10}{'N=2吞吐':>10}{'N=6吞吐':>10}")
    for size, table in sorted(exec_by_size.items()):
        solo = table[1]
        best_n, best_tp = None, 0.0
        for n in sorted(table):
            throughput = 1000.0 * n / (table[n] + PIPELINE_MS)
            if throughput > best_tp:
                best_n, best_tp = n, throughput
        tp2 = 1000.0 * 2 / (table[2] + PIPELINE_MS)
        tp6 = 1000.0 * 6 / (table[6] + PIPELINE_MS)
        print(f"{f'{size[0]}x{size[1]}':<10}{solo:>14.2f}{best_n:>7}{best_tp:>10.1f}"
              f"{tp2:>10.1f}{tp6:>10.1f}")
    print("\n注意：这是**单尺寸纯负载**下的最优值。混合负载的最优由各尺寸的权重决定，")
    print("      而控制器的判据（小请求占比是否 ≥ 0.5）与本负载的 50/50 构成完全重合。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
