"""Q4：被 host 开销污染的标定表，让调度器做错了多少决策？

背景（见 `docs/实验记录.md` §10 与 `AGENTS.md` §11）
----------------------------------------------------
已确认：**小请求的 `SM 配额 → 延迟` 标定值量的不是 GPU，而是 host 喂给 GPU 的能力。**
224×224 的事件计时里 **59%** 是 host 侧 kernel 发射开销，随尺寸单调降到 672×672 的 **7%**。
把 host 移出关键路径（CUDA Graph 回放）后，四个尺寸的慢模态从 9%–27% 全部降到 0%。

**但 host 开销是真实的延迟，不是假的。** 它成为问题是因为两件事：
1. 它被记在 `execution_ms` 里，于是调度器以为**加 SM 配额能改善它**——而配额改不了 host；
2. 它随尺寸强烈变化（224 被抬高 2.4×，672 只抬高 1.08×），于是**尺寸之间的相对代价被压平**。

本脚本量化第 2 条造成的后果：用**真实的 `PerformanceModel` + `Scheduler` 代码**，
分别喂两张表，比较它们做出的决策。

两张表怎么来
------------
- **污染表** `T_eager(s, q)`：常规方式实测（eager 执行，4 尺寸 × 4 配额）。
- **干净表** `T_clean(s, q)`：**这里有一个硬约束**——掩码在 CUDA Graph 回放时不生效
  （实测 0.84×，见 §10.7），所以干净值只能测到**满配额**那一列。
  配额维度用模型外推：

  ```text
  T_clean(s, q) = T_graph(s) × [ T_eager(s, q) / T_eager(s, 1.0) ]
  ```

  即"配额惩罚的相对形状沿用 eager 测得的，绝对值换成干净的 GPU 时间"。
  **这是模型不是测量**，论文里必须这样标注。

比较什么
--------
三处真实用到标定表的代码路径（`encoder_sched/scheduler.py`）：

1. ``choose_quota`` 的 ``deadline_min``：升序找第一个「预测 ≤ 剩余时间」的档位；
2. ``rank_key``（``edf_size``）：``predicted_ms`` 是排序的第三键；
3. ``EncodeJob.slack_ms``：``deadline − now − predicted_ms``，deadline guard 用。

用法
----
::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python scripts/experiment_calibration_decision_loss.py --phase measure
    python scripts/experiment_calibration_decision_loss.py --phase analyze
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import ctypes
import json
import statistics
import time
import zlib
from dataclasses import dataclass

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.libsmctrl_adapter import load_libsmctrl
from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel
from encoder_sched.resource import create_resource_backend
from encoder_sched.scheduler import Scheduler

SIZES = (224, 336, 448, 672)
QUOTAS = (0.25, 0.5, 0.75, 1.0)
EAGER_REPS = 15
GRAPH_REPS = 200
WARMUP = 3

TABLE_COLUMNS = ["width", "height", "patches", "sm_fraction", "latency_ms"]


# --------------------------------------------------------------------------- #
# 第一部分：测两张表
# --------------------------------------------------------------------------- #
def measure_tables(config_path: str, out_dir: Path) -> dict:
    config = load_config(config_path)
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    if not resource.enforces_sm_partition:
        raise SystemExit(f"资源后端 {resource.name} 不实施真实 SM 配额；本实验禁止代理回退")

    encoder = ClipEncoderBackend(config.model, resource, 1)
    model = encoder.model
    lib = load_libsmctrl()

    def job_for(size: int, quota: float, tag: str) -> EncodeJob:
        job = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                        width=size, height=size, deadline_ms=600_000)
        job.sm_fraction = quota
        return job

    # ---- 污染表：常规 eager 测量，逐个 (尺寸, 配额) 单元 ----
    eager: dict[tuple[int, float], float] = {}
    print("=== 污染表 T_eager(s, q)：常规测量 ===", flush=True)
    for size in SIZES:
        for quota in QUOTAS:
            samples = []
            for i in range(WARMUP + EAGER_REPS):
                result = encoder.encode(job_for(size, quota, f"e-{size}-{quota}-{i}"), 0)
                if i >= WARMUP:
                    samples.append(float(result.execution_ms))
            eager[(size, quota)] = statistics.median(samples)
        print(f"  {size}×{size}: " +
              "  ".join(f"q{q}={eager[(size, q)]:.2f}" for q in QUOTAS), flush=True)

    # ---- 干净表：CUDA Graph 回放，只能测满配额 ----
    # 掩码在回放时不生效，所以这里必须显式清零掩码，让"干净"确实是无约束的 GPU 时间。
    print("\n=== 干净表 T_graph(s)：CUDA Graph 回放（仅满配额可得）===", flush=True)
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
    graph_times: dict[int, float] = {}
    for size in SIZES:
        job = job_for(size, 1.0, f"g-{size}")
        with encoder.torch.inference_mode():
            pixels = encoder._input(job)

        def forward():
            with encoder.torch.inference_mode():
                return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

        side = encoder.torch.cuda.Stream()
        side.wait_stream(encoder.torch.cuda.current_stream())
        with encoder.torch.cuda.stream(side):
            for _ in range(3):
                forward()
        encoder.torch.cuda.current_stream().wait_stream(side)
        encoder.torch.cuda.synchronize()

        graph = encoder.torch.cuda.CUDAGraph()
        with encoder.torch.inference_mode():
            with encoder.torch.cuda.graph(graph):
                captured = model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
        graph.replay()
        encoder.torch.cuda.synchronize()
        ref = forward()
        encoder.torch.cuda.synchronize()
        if float((captured.detach().float() - ref.detach().float()).abs().max()) > 1e-2:
            raise SystemExit(f"{size}: CUDA Graph 与 eager 输出不一致，终止")

        def timed():
            start = encoder.torch.cuda.Event(enable_timing=True)
            end = encoder.torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            encoder.torch.cuda.synchronize()
            return start.elapsed_time(end)

        for _ in range(10):
            timed()
        graph_times[size] = statistics.median([timed() for _ in range(GRAPH_REPS)])
        print(f"  {size}×{size}: {graph_times[size]:.3f} ms "
              f"（eager 满配额 {eager[(size, 1.0)]:.3f} → host 占比 "
              f"{(1 - graph_times[size]/eager[(size, 1.0)])*100:.0f}%）", flush=True)

    # ---- 写两张表 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    eager_path = out_dir / "profile_eager.csv"
    clean_path = out_dir / "profile_clean.csv"
    with eager_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for size in SIZES:
            for quota in QUOTAS:
                writer.writerow({"width": size, "height": size, "patches": (size // 32) ** 2,
                                 "sm_fraction": quota, "latency_ms": eager[(size, quota)]})
    with clean_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for size in SIZES:
            for quota in QUOTAS:
                shape = eager[(size, quota)] / eager[(size, 1.0)]
                writer.writerow({"width": size, "height": size, "patches": (size // 32) ** 2,
                                 "sm_fraction": quota,
                                 "latency_ms": graph_times[size] * shape})

    summary = {
        "eager": {f"{s}@{q}": v for (s, q), v in eager.items()},
        "graph": {str(s): v for s, v in graph_times.items()},
        "host_share": {str(s): 1 - graph_times[s] / eager[(s, 1.0)] for s in SIZES},
        "clean_table_note": "T_clean(s,q) = T_graph(s) × [T_eager(s,q)/T_eager(s,1.0)]；模型而非测量",
        "eager_table": str(eager_path),
        "clean_table": str(clean_path),
    }
    (out_dir / "tables.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n污染表 → {eager_path}\n干净表 → {clean_path}")
    return summary


# --------------------------------------------------------------------------- #
# 第二部分：用真实调度器代码比较两套表的决策
# --------------------------------------------------------------------------- #
@dataclass
class DecisionComparison:
    workload: str
    n: int
    quota_disagreement: float
    mean_quota_eager: float
    mean_quota_clean: float
    quota_granted_per_1000: float
    predicted_ratio_eager_over_clean: float
    rank_inversions_per_1000: float
    deadline_infeasible_eager: int
    deadline_infeasible_clean: int


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decide(scheduler: Scheduler, rows: list[dict]) -> list[tuple[float, float, float]]:
    """对每条请求跑真实的 ``Scheduler.prepare``，返回 (配额, 预测延迟, slack)。

    ``now_ns`` 显式传到达时刻——调度器在**提交时**决策，用当前时间会让结果不可复现。
    """
    out = []
    for row in rows:
        job = EncodeJob(row["request_id"], int(row.get("seed", 0)),
                        int(row["width"]), int(row["height"]), float(row["deadline_ms"]),
                        int(row.get("priority", 0)),
                        arrival_ns=int(float(row.get("arrival_ms", 0.0)) * 1_000_000))
        scheduler.prepare(job, now_ns=job.arrival_ns)
        out.append((job.sm_fraction, job.predicted_ms,
                    job.slack_ms(now_ns=job.arrival_ns)))
    return out


def count_inversions(a: list[float], b: list[float]) -> int:
    """两套预测值的排序反转对数（只在预测值不相等时计入）。"""
    inversions = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                inversions += 1
    return inversions


def analyze(eager_table: Path, clean_table: Path, workloads: list[Path]) -> list[DecisionComparison]:
    quotas = QUOTAS
    results: list[DecisionComparison] = []
    for wl in workloads:
        rows = load_trace(wl)
        sched_eager = Scheduler("edf_size", PerformanceModel(eager_table), quotas,
                                quota_policy="deadline_min")
        sched_clean = Scheduler("edf_size", PerformanceModel(clean_table), quotas,
                                quota_policy="deadline_min")
        de = decide(sched_eager, rows)
        dc = decide(sched_clean, rows)

        q_e = [x[0] for x in de]
        q_c = [x[0] for x in dc]
        p_e = [x[1] for x in de]
        p_c = [x[1] for x in dc]
        disagree = sum(1 for a, b in zip(q_e, q_c) if a != b)
        inversions = count_inversions(p_e, p_c)
        n = len(rows)
        results.append(DecisionComparison(
            workload=wl.name, n=n,
            quota_disagreement=disagree / n,
            mean_quota_eager=statistics.fmean(q_e),
            mean_quota_clean=statistics.fmean(q_c),
            quota_granted_per_1000=(statistics.fmean(q_e) - statistics.fmean(q_c)) * 1000,
            predicted_ratio_eager_over_clean=statistics.fmean(p_e) / statistics.fmean(p_c),
            rank_inversions_per_1000=inversions / n * 1000,
            deadline_infeasible_eager=sum(1 for x in de if x[2] < 0),
            deadline_infeasible_clean=sum(1 for x in dc if x[2] < 0),
        ))
    return results


def print_analysis(results: list[DecisionComparison], eager_table: Path, clean_table: Path) -> None:
    me = PerformanceModel(eager_table)
    mc = PerformanceModel(clean_table)
    print(f"\n{'='*100}")
    print("两套标定表的形状对比（预测延迟，ms）")
    print(f"{'='*100}")
    print(f"{'尺寸':>6} {'配额':>6} {'污染表':>10} {'干净表':>10} {'污染/干净':>10}")
    for size in SIZES:
        for quota in QUOTAS:
            patches = (size // 32) ** 2
            a = me.predict(patches, quota)
            b = mc.predict(patches, quota)
            flag = "  ← 污染最重" if quota == 1.0 and a / b > 2 else ""
            print(f"{size:>6} {quota:>6.2f} {a:>10.2f} {b:>10.2f} {a/b:>10.2f}×{flag}")
    print("\n跨尺寸的相对代价（满配额 672 / 224）：")
    pe = me.predict(441, 1.0) / me.predict(49, 1.0)
    pc = mc.predict(441, 1.0) / mc.predict(49, 1.0)
    print(f"  污染表 {pe:.2f}×   干净表 {pc:.2f}×   → 污染把尺寸代价压平了 {pc/pe:.2f} 倍")

    print(f"\n{'='*100}")
    print("决策后果（真实 Scheduler + edf_size + deadline_min）")
    print(f"{'='*100}")
    print(f"{'负载':>26} {'请求数':>7} {'配额决策不一致':>14} {'平均配额(污染/干净)':>20} "
          f"{'预测值膨胀':>10} {'排序反转/千':>11}")
    print("-" * 100)
    for r in results:
        print(f"{r.workload:>26} {r.n:>7} {r.quota_disagreement*100:>13.1f}% "
              f"{r.mean_quota_eager:>9.3f} /{r.mean_quota_clean:>8.3f} "
              f"{r.predicted_ratio_eager_over_clean:>9.2f}× {r.rank_inversions_per_1000:>11.1f}")


def sweep_deadlines(eager_table: Path, clean_table: Path, trace: Path,
                    scales: list[float]) -> list[dict]:
    """deadline 紧度扫描：找出标定表污染**真正开始影响决策**的区间。

    为什么必须扫：现有负载的 deadline 中位数 102.5 ms，而实测执行只有 3–22 ms——
    `deadline_min` 对所有请求都直接返回最小配额，两套表不可能分歧（实测不一致率 0.0%）。
    所以「表被污染」与「决策被污染」之间隔着一个**紧度条件**，本函数把它定位出来。
    """
    base = load_trace(trace)
    out = []
    for scale in scales:
        rows = []
        for r in base:
            r = dict(r)
            r["deadline_ms"] = max(1.0, float(r["deadline_ms"]) * scale)
            rows.append(r)
        sched_eager = Scheduler("edf_size", PerformanceModel(eager_table), QUOTAS,
                                quota_policy="deadline_min")
        sched_clean = Scheduler("edf_size", PerformanceModel(clean_table), QUOTAS,
                                quota_policy="deadline_min")
        de = decide(sched_eager, rows)
        dc = decide(sched_clean, rows)
        n = len(rows)
        q_e = [x[0] for x in de]
        q_c = [x[0] for x in dc]
        # 只统计"小请求"（224/336）上的分歧——污染集中在那里
        small = [i for i, r in enumerate(rows) if r["width"] <= 336]
        disagree_small = sum(1 for i in small if q_e[i] != q_c[i])
        out.append({
            "deadline_scale": scale,
            "deadline_median_ms": statistics.median(r["deadline_ms"] for r in rows),
            "disagreement": sum(1 for a, b in zip(q_e, q_c) if a != b) / n,
            "disagreement_small": disagree_small / max(1, len(small)),
            "mean_quota_eager": statistics.fmean(q_e),
            "mean_quota_clean": statistics.fmean(q_c),
            "infeasible_eager": sum(1 for x in de if x[2] < 0),
            "infeasible_clean": sum(1 for x in dc if x[2] < 0),
        })
    return out


def print_sweep(rows: list[dict]) -> None:
    print(f"\n{'='*104}")
    print("deadline 紧度扫描：标定表污染何时开始改变决策")
    print(f"{'='*104}")
    print(f"{'deadline缩放':>12} {'中位deadline':>12} {'配额决策不一致':>14} "
          f"{'仅小请求的不一致':>16} {'平均配额 污染/干净':>20} {'预测违约 污染/干净':>20}")
    print("-" * 104)
    for r in rows:
        print(f"{r['deadline_scale']:>12.2f} {r['deadline_median_ms']:>11.1f}ms "
              f"{r['disagreement']*100:>13.1f}% {r['disagreement_small']*100:>15.1f}% "
              f"{r['mean_quota_eager']:>9.3f} /{r['mean_quota_clean']:>7.3f} "
              f"{r['infeasible_eager']:>9} /{r['infeasible_clean']:>9}")
    first = next((r for r in rows if r["disagreement"] > 0.01), None)
    if first:
        print(f"\n  → 当 deadline 缩放到 {first['deadline_scale']:.2f}"
              f"（中位 {first['deadline_median_ms']:.0f} ms）时，两套表开始分歧")
    else:
        print("\n  → 扫描范围内两套表**始终不分歧**：本负载结构下，标定表污染不改变配额决策")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4：标定表污染造成的调度决策损失")
    parser.add_argument("--phase", choices=("measure", "analyze", "sweep", "all"), default="all")
    parser.add_argument("--sweep-scales", default="4,2,1,0.5,0.3,0.2,0.15,0.1,0.07,0.05")
    parser.add_argument("--sweep-trace", default="data/workloads/saturating_4ms.jsonl")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproducibility"))
    parser.add_argument("--workloads", nargs="*", default=None)
    parser.add_argument("--table-a", default=None,
                        help="对比用的第一张表；默认 results/reproducibility/profile_eager.csv。"
                             "**跨会话复现性问题应传 data/profiles/default.csv**（09-14 会话）")
    parser.add_argument("--table-b", default=None,
                        help="对比用的第二张表；默认 results/reproducibility/profile_clean.csv")
    args = parser.parse_args()

    eager_table = Path(args.table_a) if args.table_a else args.output_dir / "profile_eager.csv"
    clean_table = Path(args.table_b) if args.table_b else args.output_dir / "profile_clean.csv"

    if args.phase in ("measure", "all"):
        measure_tables(args.config, args.output_dir)
    if args.phase in ("analyze", "all"):
        if not eager_table.exists() or not clean_table.exists():
            raise SystemExit("两张表还不存在，先跑 --phase measure")
        workload_paths = ([Path(p) for p in args.workloads] if args.workloads
                          else sorted(Path("data/workloads").glob("saturating_*.jsonl")))
        if not workload_paths:
            raise SystemExit("找不到负载轨迹")
        results = analyze(eager_table, clean_table, workload_paths)
        print_analysis(results, eager_table, clean_table)

        out = args.output_dir / "decision_loss.json"
        out.write_text(json.dumps({
            "eager_table": str(eager_table), "clean_table": str(clean_table),
            "results": [r.__dict__ for r in results],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
    if args.phase in ("sweep", "all"):
        if not eager_table.exists() or not clean_table.exists():
            raise SystemExit("两张表还不存在，先跑 --phase measure")
        scales = [float(s) for s in args.sweep_scales.split(",") if s.strip()]
        sweep = sweep_deadlines(eager_table, clean_table, Path(args.sweep_trace), scales)
        print_sweep(sweep)
        out = args.output_dir / "deadline_sweep.json"
        out.write_text(json.dumps({"trace": args.sweep_trace, "sweep": sweep},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
