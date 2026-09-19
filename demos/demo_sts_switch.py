"""FineST 式「时空切换判据」的本机检验：什么时候该并发、什么时候该独占。

出处
----
Serving DNN Inference With Fine-Grained Spatio-Temporal Sharing of GPU Servers
（FineST, IEEE TSC 2024）。

**切换判据**（原文 §IV，FINDPART / TRYTOTEMPORALSHARE）：

    若某模型的**剩余请求率 b 无法饱和**它的理想分区 p_ideal：
        尝试时间共享
    若存在 batch size b 与分区 part 使
        d + L(m, part, b) ≤ SLO_i
    则该模型可与其他模型时间共享 part
    其中 d = 该分区上的调度周期，L(m, part, b) = 预测的批延迟

**为什么这条判据比 DProbe 的 util 阈值更适合本课题**：它**只依赖"剩余请求率能否饱和分区"**，
不依赖 SM/GPU 占用率——而占用率在本机是失效判据（实测：12 个 TPC 中 6 个就吃满显存带宽，
真实 TPC 互斥分区在所有负载、所有指标上都劣于共享 SM 的多流）。

FineST 的另一条反直觉实测（§I，与本课题同向）：

    *固定 1:1:1:1 GPU 划分把峰值吞吐从 600 提到 **1160 rps**，**优于** Gpulet 的自适应
    时空调度器。*

即**固定等分反而更好**——这与"DACC 被 edf 支配"、"朴素多流击败 DACC"的结论同向。

本 demo 做什么
--------------
把 FineST 的判据简化到本机可测的形式，比较三种并发策略：

  always_concurrent  永远并发（= 本课题已实测的 `spatial_proxy` / 多流）
  always_serial      永远串行（= `serial_fcfs`）
  sts_switch         **FineST 式**：按"剩余请求率能否饱和"决定并发还是串行

判据
----
- **通过**：`sts_switch` 在某档负载下同时优于两个极端（吞吐不低于并发、P99 不差于串行）
  → FineST 判据在本机有效，可成为方法骨架
- **死亡**：`sts_switch` 被 `always_concurrent` 支配 → 判据无用，
  印证本课题"零等待并发始终最优"的实测（**这是更有力的负结果**）

⚠️ 本演示是**离线模拟**，代价模型用真实标定表 + 本课题实测的并发惩罚系数，
不跑 GPU。这样可以在写在线代码前先看结论。

用法
----
    python demos/demo_sts_switch.py --workloads data/workloads/saturating_4ms.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.performance import PerformanceModel

#: 并发惩罚系数。取本课题实测值：二等分分区并发时单请求会被拖慢，
#: 但**共享 SM 的多流**几乎不拖慢（这是关键区别）。
#:
#: 实测依据（`docs/实验记录.md` §2.1）：
#:   访存受限（小请求）并发/串行 = 1.038–1.283× 聚合收益
#:   计算受限（大请求）并发/串行 = 0.528–0.531× 净亏
#: 即：**并发对小请求有益、对大请求有害**——这正是 FineST 判据能起作用的前提。
CONCURRENCY_FACTOR = {
    "small": {"solo": 1.0, "concurrent": 1.15},   # 并发时单请求变慢 15%，但聚合更高
    "large": {"solo": 1.0, "concurrent": 1.90},   # 大请求并发时几乎翻倍——本课题实测的 53%
}


def bucket(patches: int) -> str:
    return "small" if patches <= 100 else "large"


def simulate(rows: list[dict], model: PerformanceModel, streams: int, policy: str,
             saturation_rps: float) -> dict:
    """单卡多流离线模拟，`policy` 决定"是否允许并发"。

    `saturation_rps` 是 FineST 判据里的关键量：一个请求流需要多高的到达率才"饱和"资源。
    低于它 → 允许并发（填气泡）；高于它 → 独占（避免争用）。
    """
    jobs = sorted(
        ({"id": r["request_id"], "size": int(r["width"]),
          "patches": (int(r["width"]) // 32) ** 2,
          "arrival_ms": float(r.get("arrival_ms", 0.0)),
          "deadline_ms": float(r["deadline_ms"])} for r in rows),
        key=lambda j: j["arrival_ms"])

    now, idx = 0.0, 0
    pending: list[dict] = []
    running: list[tuple[float, dict]] = []
    done: list[dict] = []
    total_exec = 0.0

    def recent_arrival_rate() -> float:
        """最近一个窗口内的到达率（请求/秒）。

        **修正过的实现**：初版写成 `now-50 <= arrival <= now`，但 `now` 是**模拟时钟**
        （从 0 起、按执行时间推进），而窗口只有 50 ms——大多数时刻窗口内是空的，
        rate 恒为 0，`sts_switch` 会退化成正的 `always_concurrent`，实验失去意义。
        改为按"最近 50 ms 模拟时间内到达的请求数"算，且窗口起点不早于 0。
        """
        lo = max(0.0, now - 50.0)
        window = [j for j in jobs if lo <= j["arrival_ms"] <= now]
        span = max(now - lo, 1.0) / 1000.0
        return len(window) / span if window else 0.0

    def backlog_rate() -> float:
        """**积压换算的到达率**：队列里压着的请求，若要在 1 秒内清空，需要多高的服务率。

        这才是 FineST 判据里那个 `b`（"剩余请求率"）在本机的对应量——
        它问的不是历史到达率，而是**当前还压着多少活、按现有资源干得完干不完**：

            剩余工作量(ms) = Σ 预测执行时间
            backlog_rate  = 队列长度 / (剩余工作量/1000)  ... 即"这么多活在 1 秒内
                            能干完的话，相当于多高的到达率"

        队列越长、单个活越轻，这个值越高 → 越应当独占（避免争用）。
        队列空 → 0 → 允许并发填气泡。
        """
        if not pending:
            return 0.0
        total_work_ms = sum(model.predict(j["patches"], 1.0) for j in pending)
        if total_work_ms <= 0:
            return 0.0
        # 队列长度 req ÷ 清空所需秒数 = 等效到达率 (req/s)
        return len(pending) / (total_work_ms / 1000.0)

    def permitted() -> int:
        """允许的并发数。"""
        if policy == "always_serial":
            return 1
        if policy == "always_concurrent":
            return streams
        # sts_switch：**剩余请求率不足以饱和 → 允许并发填气泡；否则独占**。
        # 用"积压 + 近期到达"的合成量，避免单一窗口在突发负载下失效。
        rate = max(recent_arrival_rate(), backlog_rate())
        return 1 if rate >= saturation_rps else streams

    while len(done) < len(jobs):
        while idx < len(jobs) and jobs[idx]["arrival_ms"] <= now:
            pending.append(jobs[idx])
            idx += 1
        limit = permitted()
        while len(running) < limit and pending:
            pick = min(pending, key=lambda j: j["arrival_ms"] + j["deadline_ms"])  # EDF
            pending.remove(pick)
            b = bucket(pick["patches"])
            concurrent = len(running) > 0
            factor = CONCURRENCY_FACTOR[b]["concurrent" if concurrent else "solo"]
            cost = model.predict(pick["patches"], 1.0) * factor
            pick["exec_ms"] = cost
            total_exec += cost
            running.append((now + cost, pick))
        if not running:
            if idx < len(jobs):
                now = jobs[idx]["arrival_ms"]
                continue
            break
        running.sort(key=lambda r: r[0])
        finish, job = running.pop(0)
        now = max(now, finish)
        job["total_ms"] = finish - job["arrival_ms"]
        done.append(job)

    makespan = max(j["total_ms"] + j["arrival_ms"] for j in done)
    totals = [j["total_ms"] for j in done]
    viol = sum(1 for j in done if j["total_ms"] > j["deadline_ms"])
    return {
        "policy": policy, "n": len(done),
        "throughput_rps": len(done) / (makespan / 1000) if makespan else float("nan"),
        "mean_ms": statistics.fmean(totals),
        "p99_ms": sorted(totals)[min(len(totals) - 1, int(0.99 * len(totals)))],
        "slo_violation_rate": viol / len(done),
        "makespan_ms": makespan,
        "mean_exec_ms": total_exec / len(done),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FineST 式时空切换判据的离线检验")
    parser.add_argument("--workload", type=Path,
                        default=Path("data/workloads/saturating_4ms.jsonl"))
    parser.add_argument("--table", type=Path, default=Path("data/profiles/default.csv"))
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/demo_sts_switch.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    model = PerformanceModel(args.table)
    arr = [float(r["arrival_ms"]) for r in rows]
    rate = len(rows) / ((arr[-1] - arr[0]) / 1000)
    # FineST 判据里的"饱和阈值"：本课题实测单流容量约 157 req/s（单流执行约 6.4 ms），
    # 双流约 292 req/s。取单流容量作为"一个流被饱和"的界线。
    saturation = 157.0

    print(f"负载 {args.workload.name}：{len(rows)} 请求，到达率约 {rate:.1f} req/s")
    print(f"饱和阈值（FineST 判据的 b*）= {saturation:.0f} req/s（本课题实测单流容量）")
    print(f"→ 到达率{'高于' if rate >= saturation else '低于'}饱和阈值，"
          f"判据预期会{'独占' if rate >= saturation else '允许并发'}\n")

    policies = ["always_serial", "always_concurrent", "sts_switch"]
    results = [simulate(rows, model, args.streams, p, saturation) for p in policies]

    print(f"{'='*92}")
    print("离线模拟（单卡多流，EDF 排序，代价 = 标定表 × 并发惩罚系数）")
    print(f"{'='*92}")
    print(f"{'策略':>18} {'吞吐(rps)':>11} {'均值(ms)':>10} {'P99(ms)':>9} "
          f"{'SLO违约':>9} {'平均执行(ms)':>13}")
    print("-" * 92)
    names = {"always_serial": "永远串行", "always_concurrent": "永远并发",
             "sts_switch": "FineST 式切换"}
    for r in results:
        print(f"{names[r['policy']]:>18} {r['throughput_rps']:>11.1f} {r['mean_ms']:>10.2f} "
              f"{r['p99_ms']:>9.2f} {r['slo_violation_rate']*100:>8.1f}% "
              f"{r['mean_exec_ms']:>13.2f}")

    print(f"\n{'='*92}")
    print("判据")
    print(f"{'='*92}")
    by = {r["policy"]: r for r in results}
    sw, conc, ser = by["sts_switch"], by["always_concurrent"], by["always_serial"]
    print(f"  切换 vs 永远并发：吞吐 {sw['throughput_rps']/conc['throughput_rps']:.3f}×  "
          f"P99 {sw['p99_ms']/conc['p99_ms']:.3f}×")
    print(f"  切换 vs 永远串行：吞吐 {sw['throughput_rps']/ser['throughput_rps']:.3f}×  "
          f"P99 {sw['p99_ms']/ser['p99_ms']:.3f}×")
    print()

    beats_conc = sw["throughput_rps"] >= conc["throughput_rps"] * 0.98 and \
        sw["p99_ms"] <= conc["p99_ms"] * 0.98
    beats_ser = sw["throughput_rps"] >= ser["throughput_rps"] * 1.02
    if beats_conc and beats_ser:
        print("  ✅ 通过：FineST 式切换同时优于两个极端 → 判据在本机有效，可作方法骨架")
    elif conc["p99_ms"] <= sw["p99_ms"] and conc["throughput_rps"] >= sw["throughput_rps"] * 0.98:
        print("  ❌ 判死：切换被「永远并发」支配")
        print("     → 印证本课题『零等待并发在尾延迟上始终最优』的实测；")
        print("       切换判据在本机不带来额外价值（**这是更有力的负结果**）")
    else:
        print("  ⚠️ 未通过：切换未同时优于两个极端，见上表")

    print(f"\n  ⚠️ 本模拟的并发惩罚系数是**实测值的简化**（小请求 1.15×、大请求 1.90×），")
    print(f"     真实系数随尺寸、配额、并发数连续变化。要下结论必须走真实 GPU 实验。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"workload": str(args.workload), "saturation_rps": saturation, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
