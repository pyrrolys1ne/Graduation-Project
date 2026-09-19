"""TCM-Serve 式「分档老化」的本机检验。

出处
----
TCM-Serve: Modality-aware Scheduling for Multimodal Large Language Model Inference。

它的核心抽象是把请求按资源需求分三类——**视频=卡车、图像=汽车、文本=摩托车**，
放入三个独立队列，队内 FCFS，队间静态优先级 **摩托 > 汽车 > 卡车**。

**老化机制**（§Priority Regulator）：

    Score_c = −log(Priority_c),   c ∈ {Motorcycles, Cars, Trucks}

**关键设计**：老化速率**按类别分档，且与该类请求自身的推理时长成比例**——
摩托车优先级快速上升（立刻可调度），汽车中等，"卡车"极慢上升、长期低位，
但**最终能推进、不会饿死**。

论文还做了一个**纯按年龄老化**（naive aging）的消融作为对照——**这个对照设计直接搬用**。

本机映射
--------
CLIP 请求没有"视频/文本"之分，但有**明确的尺寸分档**（本课题实测带宽敏感性随尺寸单调上升）：

    摩托车 = 224 / 288（小，访存受限，最易受队头阻塞伤害）
    汽车   = 336 / 384 / 448（中）
    卡车   = 512 / 672（大，计算受限）

判据
----
本课题已实测两条**与之直接相关**的事实：
  - 过载下 `edf` 吞吐 95.11、SLO 违约 1.1%，已是很好的基线
  - **欠饱和负载无法区分任何调度策略**（差异落在 ±0.07 噪声内）

因此本 demo 的判据必须是**过载 + 混合尺寸**下才有意义：

- **通过**：分档老化在过载下 SLO 违约率优于 `edf`，**且**大请求的最大等待时间有上界
  （没有饿死）
- **死亡**：与 `edf` 打平 → 说明 CLIP 的尺寸异质性不足以产生队头阻塞
  （**与"edf 已足够"的既有结论一致，是可写的负结果**）

本 demo 用**离线模拟**（不跑 GPU）：复用 `encoder_sched.simulator` 的机制，
把排序规则换成待检验的策略，用真实标定表当代价模型。
**这样可以在写任何在线代码前先看结论。**

用法
----
    python demos/demo_aged_priority.py --workload data/workloads/saturating_4ms.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.performance import PerformanceModel

#: 尺寸分档（摩托车 / 汽车 / 卡车）
CLASSES = {
    "moto": (224, 288),
    "car": (336, 384, 448),
    "truck": (512, 672),
}

#: 各类的静态基础优先级（摩托最高）。取自 TCM-Serve 的"优先级起点"设定：
#: 起点差异按各类自身的推理时长尺度拉开。
BASE_PRIORITY = {"moto": 1.0, "car": 0.1, "truck": 0.01}

#: 老化率。TCM-Serve 的关键：老化率与"该类请求自身的推理时长"成比例，
#: 因此卡车虽然起点极低，但**最终能追上**，只是慢得多。
#: 这里用各类代表尺寸在满配额下的标定延迟作为比例基准。
def aging_rate(klass: str, model: PerformanceModel) -> float:
    rep_patches = {"moto": 49, "car": 100, "truck": 441}[klass]
    t = model.predict(rep_patches, 1.0)
    return 1.0 / max(t, 1e-6)


def classify(width: int) -> str:
    for klass, sizes in CLASSES.items():
        if width in sizes:
            return klass
    return "car"


def score(klass: str, waited_ms: float, model: PerformanceModel,
          ages: bool = True, uniform: bool = False) -> float:
    """TCM-Serve 的 Score = −log(Priority)，优先级随等待时间增长。

    `uniform=True` 复现论文的 **naive aging 消融**：忽略分档，所有类别同一老化率。
    """
    base = BASE_PRIORITY[klass]
    if not ages:
        return -math.log(base)
    rate = aging_rate("car", model) if uniform else aging_rate(klass, model)
    priority = base * (1.0 + rate * waited_ms)
    return -math.log(max(priority, 1e-12))


def simulate(rows: list[dict], model: PerformanceModel, streams: int,
             policy: str) -> dict:
    """单 GPU 多流离线模拟。`policy` ∈ {fcfs, edf, aged, aged_uniform, srtf}。

    刻意写得简单：只模拟"N 条流、每次选一个请求执行、执行耗时查标定表"。
    不做批处理、不做并发干扰——本 demo 要看的是**排序规则**的差异，
    引入更多机制只会混淆归因。
    """
    jobs = []
    for i, row in enumerate(rows):
        size = int(row["width"])
        jobs.append({
            "id": row["request_id"], "size": size,
            "patches": (size // 32) ** 2,
            "arrival_ms": float(row.get("arrival_ms", 0.0)),
            "deadline_ms": float(row["deadline_ms"]),
            "klass": classify(size),
        })
    jobs.sort(key=lambda j: j["arrival_ms"])

    now = 0.0
    pending: list[dict] = []
    running: list[tuple[float, dict]] = []      # (finish_ms, job)
    done: list[dict] = []
    idx = 0
    exec_ms = 0.0

    def admit() -> None:
        nonlocal idx
        while idx < len(jobs) and jobs[idx]["arrival_ms"] <= now:
            pending.append(jobs[idx])
            idx += 1

    while len(done) < len(jobs):
        admit()
        while len(running) < streams and pending:
            if policy == "fcfs":
                pick = min(pending, key=lambda j: j["arrival_ms"])
            elif policy == "edf":
                pick = min(pending, key=lambda j: j["arrival_ms"] + j["deadline_ms"])
            elif policy == "srtf":
                pick = min(pending, key=lambda j: model.predict(j["patches"], 1.0))
            elif policy in ("aged", "aged_uniform"):
                pick = min(pending, key=lambda j: score(
                    j["klass"], now - j["arrival_ms"], model,
                    uniform=(policy == "aged_uniform")))
            else:
                raise ValueError(policy)
            pending.remove(pick)
            cost = model.predict(pick["patches"], 1.0)
            pick["start_ms"] = now
            pick["exec_ms"] = cost
            exec_ms += cost
            running.append((now + cost, pick))
        if not running:
            # 线程空闲但还有没到的请求：跳到下一个到达时刻
            if idx < len(jobs):
                now = jobs[idx]["arrival_ms"]
                continue
            break
        running.sort(key=lambda r: r[0])
        finish, job = running.pop(0)
        now = max(now, finish)
        job["finish_ms"] = finish
        job["total_ms"] = finish - job["arrival_ms"]
        done.append(job)

    makespan = max(j["finish_ms"] for j in done)
    totals = [j["total_ms"] for j in done]
    violations = [j for j in done if j["total_ms"] > j["deadline_ms"]]
    # 大请求（卡车）的最大等待——检验"是否饿死"
    truck_waits = [j["total_ms"] for j in done if j["klass"] == "truck"]
    return {
        "policy": policy,
        "n": len(done),
        "throughput_rps": len(done) / (makespan / 1000) if makespan else float("nan"),
        "mean_ms": statistics.fmean(totals),
        "p99_ms": sorted(totals)[min(len(totals) - 1, int(0.99 * len(totals)))],
        "slo_violation_rate": len(violations) / len(done),
        "makespan_ms": makespan,
        "truck_max_wait_ms": max(truck_waits) if truck_waits else float("nan"),
        "truck_mean_wait_ms": statistics.fmean(truck_waits) if truck_waits else float("nan"),
        "mean_exec_ms": exec_ms / len(done),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TCM-Serve 式分档老化的离线检验")
    parser.add_argument("--workload", type=Path,
                        default=Path("data/workloads/saturating_4ms.jsonl"))
    parser.add_argument("--table", type=Path, default=Path("data/profiles/default.csv"))
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/demo_aged_priority.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    model = PerformanceModel(args.table)
    sizes = sorted({int(r["width"]) for r in rows})
    print(f"负载 {args.workload.name}：{len(rows)} 请求，尺寸 {sizes}，流数 {args.streams}")
    arr = [float(r["arrival_ms"]) for r in rows]
    print(f"到达跨度 {arr[-1]-arr[0]:.0f} ms → 到达率约 "
          f"{len(rows)/((arr[-1]-arr[0])/1000):.1f} req/s")
    dl = [float(r["deadline_ms"]) for r in rows]
    print(f"deadline 中位 {statistics.median(dl):.1f} ms\n")

    policies = ["fcfs", "edf", "srtf", "aged_uniform", "aged"]
    results = [simulate(rows, model, args.streams, p) for p in policies]

    print(f"{'='*104}")
    print("离线模拟（单卡多流，代价模型 = 标定表）")
    print(f"{'='*104}")
    print(f"{'策略':>14} {'吞吐(rps)':>10} {'均值(ms)':>10} {'P99(ms)':>9} {'SLO违约':>9} "
          f"{'卡车最大等待':>12} {'卡车均值':>10}")
    print("-" * 104)
    names = {"fcfs": "FCFS", "edf": "EDF", "srtf": "SRTF（最小预测优先）",
             "aged_uniform": "naive aging（消融）", "aged": "分档老化（TCM）"}
    for r in results:
        print(f"{names[r['policy']]:>14} {r['throughput_rps']:>10.1f} {r['mean_ms']:>10.2f} "
              f"{r['p99_ms']:>9.2f} {r['slo_violation_rate']*100:>8.1f}% "
              f"{r['truck_max_wait_ms']:>12.1f} {r['truck_mean_wait_ms']:>10.1f}")

    print(f"\n{'='*104}")
    print("判据")
    print(f"{'='*104}")
    by = {r["policy"]: r for r in results}
    aged, edf, naive = by["aged"], by["edf"], by["aged_uniform"]

    print(f"  分档老化 vs EDF：")
    print(f"    SLO 违约  {aged['slo_violation_rate']*100:.1f}% 对 {edf['slo_violation_rate']*100:.1f}%")
    print(f"    P99       {aged['p99_ms']:.2f} 对 {edf['p99_ms']:.2f} ms")
    print(f"    卡车最大等待 {aged['truck_max_wait_ms']:.0f} 对 {edf['truck_max_wait_ms']:.0f} ms")
    print()
    print(f"  分档老化 vs naive aging（论文的消融对照）：")
    print(f"    SLO 违约  {aged['slo_violation_rate']*100:.1f}% 对 {naive['slo_violation_rate']*100:.1f}%")
    print(f"    卡车最大等待 {aged['truck_max_wait_ms']:.0f} 对 {naive['truck_max_wait_ms']:.0f} ms")
    print()

    better = aged["slo_violation_rate"] < edf["slo_violation_rate"] - 0.005
    no_starve = aged["truck_max_wait_ms"] < edf["truck_max_wait_ms"] * 1.5
    if better and no_starve:
        print("  ✅ 通过：分档老化在 SLO 上优于 EDF，且没有饿死大请求")
        print("     → 值得实现在线版本")
    elif abs(aged["slo_violation_rate"] - edf["slo_violation_rate"]) <= 0.005:
        print("  ❌ 判死：与 EDF 打平")
        print("     → CLIP 的尺寸异质性不足以产生队头阻塞，「edf 已足够」的既有结论再次成立")
    else:
        print("  ⚠️ 未通过：分档老化未同时满足两个条件，见上表")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"workload": str(args.workload), "streams": args.streams,
                                       "results": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
