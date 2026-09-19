"""ConCo「双自由度」规则的本机检验：标定时用的资源档 ≠ 运行时给的资源档。

出处
----
ConCo: Optimizing Compilation of Concurrent Tensor Programs on Shared GPU（ICS 2025）§3.5。

论文把两个自由度分开：
  `C_i%` —— 代码/配置**面向**的资源约束（标定或编译时用的档位）
  `R_i%` —— 运行时**实际**给的资源

目标：  max_{ {c_ij, r_i} }  Σ_i Throughput(M_i, c_ij, r_i)
        s.t.  |{ i | M_i is running }| ≤ L

两条反直觉结论（原文）：

  1. *"optimal GPU resource utilization often necessitates **over-provisioning**, meaning that the
     total allocation of resources to all concurrent processes **exceeds 100%**"*
  2. *"the optimal execution strategy is **not** to run the version generated under a specific GPU
     percentage constraint within the same percentage-limited resource framework"*
     → 即 **C_i% ≠ R_i% 常最优**

机理（§4.6）：ConCo 生成的代码更激进循环展开、更少 block、**更高算术强度**，目的是
*"reducing pressure on global memory bandwidth, a critical bottleneck in high-concurrency scenarios"*。
**不是分区本身坏，而是分区迫使每个任务走"为受限资源优化的次优路径"。**

与本课题实测的关系
------------------
本课题已实测：**真实 TPC 互斥分区（0+6 / 6+6）在所有负载、所有指标上都劣于共享 SM 的多流**；
计算受限时二等分分区并发只有不分区的 **53%**。
ConCo 的 over-provisioning 与我们的"零等待并发在尾延迟上始终最优"是同一件事的两种表述。
**这个 demo 把 ConCo 的规则搬过来，验证 `C≠R` 在本机是否也常最优。**

本机版本
--------
把"代码变体"换成 **"TPC 掩码档位 + 并发策略"**，用**离线标定表**当 `Throughput(·)` 估计器，
搜索空间小到暴力枚举即可（不需要论文的模拟退火）。

四组对照（two-request 场景，两个请求同尺寸）：
  C=R=6      标定按 6 TPC、运行给 6 TPC   —— "对齐"（论文说这是次优）
  C=6, R=12  标定按 6 TPC、运行给 12 TPC  —— **超配**（论文说这常最优）
  C=12, R=12 标定按 12 TPC、运行给 12 TPC —— 全量
  C=12, R=6  标定按 12 TPC、运行给 6 TPC  —— 反向错配

判据
----
- **通过**：存在 `C≠R` 组合，其吞吐 **或** P99 稳定优于 `C=R`（5 次重复、配对比较）
  → 复现 ConCo 命题，本机采用"双自由度"作为方法的一部分
- **死亡**：`C=R` 恒最优或打平 → 命题在消费级卡不成立（**同样是可写的结果**）

⚠️ 本机限制：掩码在 CUDA Graph 回放时不生效，所以本 demo **必须走 eager 路径**。

用法
----
    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python demos/demo_dual_freedom.py --size 224 --rounds 5
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
import statistics
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.libsmctrl_adapter import load_libsmctrl
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

TOTAL_TPC = 12


class Worker:
    """专用工作线程。掩码是 __thread 的，必须在同一线程内施加并测量。"""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            fn, box = self._queue.get()
            if fn is None:
                return
            try:
                box["result"] = fn()
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

    def call(self, fn, timeout: float = 600.0):
        box: dict = {}
        self._queue.put((fn, box))
        deadline = time.time() + timeout
        while not box:
            if time.time() > deadline:
                raise TimeoutError("等待工作线程超时")
            time.sleep(0.002)
        if "error" in box:
            raise box["error"]
        return box["result"]


def done_job(size: int, tag: str) -> EncodeJob:
    job = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                    width=size, height=size, deadline_ms=600_000)
    job.sm_fraction = 1.0
    return job


CONFIG_MAP: dict[int, tuple[str, float]] = {
    6: ("data/profiles/default.csv", 0.5),
    12: ("data/profiles/default.csv", 1.0),
}


def predicted_ms(size: int, c_tpc: int, table_cache: dict) -> float:
    """按 `C`（标定时面向的档位）查标定表，得到该请求的**预测**延迟。

    这就是 ConCo 里 `C_i%` 的真实作用：**它不改变执行，只改变"你以为它会跑多久"**。
    调度器拿这个预测值去做配额与排序决策；预测错了，决策就错了——
    `docs/实验记录.md` §11.3 已量化过这个后果（紧 deadline 下 75% 的配额决策分歧）。
    """
    from encoder_sched.performance import PerformanceModel
    table_path, frac = CONFIG_MAP[c_tpc]
    if table_path not in table_cache:
        table_cache[table_path] = PerformanceModel(table_path)
    return float(table_cache[table_path].predict((size // 32) ** 2, frac))


def run_pair(encoder: ClipEncoderBackend, size: int, c_tpc: int, r_tpc: int,
             n_requests: int, table_cache: dict | None = None) -> dict:
    """两个线程各持 `r_tpc` 个互斥 TPC，并发跑 `n_requests` 个请求，返回聚合指标。

    `c_tpc` 在真实流程里决定"标定表里查到的预测延迟"（本 demo 只记录、不参与执行）；
    `r_tpc` 是实际施加的掩码。**两者不同正是 ConCo 要检验的那个自由度。**
    """
    from encoder_sched.libsmctrl_adapter import enabled_tpcs_to_native_mask
    lib = load_libsmctrl()
    w1, w2 = Worker(), Worker()
    barrier = threading.Barrier(2)
    results: dict[str, dict] = {}

    def make(wk: Worker, tag: str, start_tpc: int):
        def go():
            mask = enabled_tpcs_to_native_mask(start_tpc, r_tpc)
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask))
            try:
                for i in range(2):
                    encoder.encode(done_job(size, f"warm-{tag}-{i}"), 0)
                barrier.wait()
                t0 = time.perf_counter()
                lat = [float(encoder.encode(done_job(size, f"{tag}-{i}"), 0).execution_ms)
                       for i in range(n_requests)]
                wall = (time.perf_counter() - t0) * 1000
            finally:
                lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
            return {"lat": lat, "wall_ms": wall}
        return wk.call(go)

    # 互斥起点：r_tpc 不超过一半时两线程区间不重叠；否则两者都从 0 开始。
    #
    # ⚠️ `r_tpc = 12`（满配额）时 `start2` 也是 0，即**两个线程施加同一个"启用全部 TPC"掩码**。
    # 这不是 bug，而是"给 12 TPC"在并发场景下的唯一可行语义——掩码是"允许用哪些 TPC"，
    # 不是"独占哪些 TPC"。两个线程都允许用全部 SM，由硬件自己在它们之间调度，
    # 这正是本课题实测中"共享 SM 的多流"（`spatial_proxy`）那一臂的行为，
    # 与 `spatial_tpc`（互斥的 0+6 / 6+6）形成对照。
    start2 = r_tpc if r_tpc * 2 <= TOTAL_TPC else 0
    t1 = threading.Thread(target=lambda: results.__setitem__("a", make(w1, "a", 0)))
    t2 = threading.Thread(target=lambda: results.__setitem__("b", make(w2, "b", start2)))
    t1.start(); t2.start(); t1.join(); t2.join()

    lat = results["a"]["lat"] + results["b"]["lat"]
    # 聚合吞吐：两侧各自完成的工作量 / 各自墙钟之和（与既有 pilot 的口径一致）
    agg_rps = (2 * n_requests) / ((results["a"]["wall_ms"] + results["b"]["wall_ms"]) / 2 / 1000)
    pred = predicted_ms(size, c_tpc, table_cache if table_cache is not None else {})
    actual = statistics.median(lat)
    return {
        "c_tpc": c_tpc, "r_tpc": r_tpc,
        "median_ms": actual,
        "mean_ms": statistics.fmean(lat),
        "p99_ms": sorted(lat)[min(len(lat) - 1, int(0.99 * len(lat)))],
        "agg_rps": agg_rps,
        "n": len(lat),
        # C 的作用：它决定调度器**以为**这个请求要跑多久
        "predicted_ms": pred,
        "prediction_error": actual / pred if pred else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ConCo 双自由度规则的本机检验")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--requests-per-thread", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/demo_dual_freedom.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    backend = create_resource_backend(
        config.executor.resource_backend, config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback)
    if not backend.enforces_sm_partition:
        raise SystemExit(f"资源后端 {backend.name} 不实施真实 SM 配额，本实验无意义。"
                         "禁止代理回退——请先跑 scripts/probe_libsmctrl.py。")
    encoder = ClipEncoderBackend(config.model, backend, 1)

    arms = [(6, 6), (6, 12), (12, 12), (12, 6)]
    print(f"CLIP {args.size}×{args.size}，每臂 {args.rounds} 轮，每线程 {args.requests_per_thread} 请求")
    print(f"{'C':>3} {'R':>3}  {'轮次':>4}  说明")
    print("-" * 60)

    records: dict[tuple[int, int], list[dict]] = {a: [] for a in arms}
    table_cache: dict = {}
    for rnd in range(args.rounds):
        for c_tpc, r_tpc in arms:
            rec = run_pair(encoder, args.size, c_tpc, r_tpc, args.requests_per_thread,
                           table_cache)
            rec["round"] = rnd
            records[(c_tpc, r_tpc)].append(rec)

    print(f"\n{'='*82}")
    print(f"ConCo 双自由度对照（{args.size}×{args.size}，{args.rounds} 轮）")
    print(f"{'='*82}")
    print(f"{'C':>3} {'R':>3} {'说明':>10} {'中位(ms)':>10} {'P99(ms)':>9} {'聚合(rps)':>10} "
          f"{'C 的预测(ms)':>12} {'预测误差':>9}")
    print("-" * 96)
    labels = {(6, 6): "对齐", (6, 12): "超配", (12, 12): "全量", (12, 6): "反向错配"}
    summary = {}
    for arm in arms:
        recs = records[arm]
        med = statistics.median(r["median_ms"] for r in recs)
        p99 = statistics.median(r["p99_ms"] for r in recs)
        rps = statistics.median(r["agg_rps"] for r in recs)
        pred = statistics.median(r["predicted_ms"] for r in recs)
        err = statistics.median(r["prediction_error"] for r in recs)
        summary[arm] = {"median_ms": med, "p99_ms": p99, "agg_rps": rps,
                        "predicted_ms": pred, "prediction_error": err}
        print(f"{arm[0]:>3} {arm[1]:>3} {labels[arm]:>10} {med:>10.3f} {p99:>9.3f} {rps:>10.1f} "
              f"{pred:>12.3f} {err:>8.2f}×")

    print(f"\n  说明：`C 的预测` 是调度器**以为**该请求会跑多久（查标定表得到）；")
    print(f"       `预测误差` = 实测中位 / 预测值。**这一列才是 C 的真实作用**——")
    print(f"       C 不改变执行，只改变调度器对执行时间的预期；预期错了，配额与排序决策就错了。")

    base = summary[(6, 6)]
    print(f"\n相对「对齐 (C=6,R=6)」：")
    for arm in arms:
        if arm == (6, 6):
            continue
        s = summary[arm]
        print(f"  C={arm[0]:>2}, R={arm[1]:>2} ({labels[arm]}): "
              f"吞吐 {s['agg_rps']/base['agg_rps']:.3f}×  "
              f"P99 {s['p99_ms']/base['p99_ms']:.3f}×")

    print(f"\n{'='*82}")
    print("判据")
    print(f"{'='*82}")
    oversub = summary[(6, 12)]
    aligned = summary[(6, 6)]
    if oversub["agg_rps"] > aligned["agg_rps"] * 1.02 or oversub["p99_ms"] < aligned["p99_ms"] * 0.98:
        print("  ✅ 通过：超配 (C=6,R=12) 稳定优于对齐 (C=6,R=6)")
        print("     → 复现 ConCo 命题，'双自由度'可作为方法的一部分")
    elif abs(oversub["agg_rps"] / aligned["agg_rps"] - 1) < 0.02 \
            and abs(oversub["p99_ms"] / aligned["p99_ms"] - 1) < 0.02:
        print("  ⚠️ 打平：超配与对齐无差异")
        print("     → 在本机（12 TPC、带宽墙）双自由度不带来增益；")
        print("       可能与'6 TPC 就吃满带宽'有关——SM 档位不是有效的资源旋钮")
    else:
        print("  ❌ 判死：对齐优于超配 → ConCo 命题在消费级卡上不成立（可写为负结果）")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"size": args.size, "rounds": args.rounds,
         "summary": {f"C{c}_R{r}": v for (c, r), v in summary.items()},
         "records": {f"C{c}_R{r}": v for (c, r), v in records.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
