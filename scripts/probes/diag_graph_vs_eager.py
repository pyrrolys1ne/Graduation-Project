"""判定实验：CLIP 前向的"慢模态"是不是 host 侧 kernel 发射造成的 GPU 气泡？

背景（见 docs/实验记录.md 第 10 节）：
- CLIP 前向有 7%–20% 的样本落在中位数的 2–3 倍上；而 matmul 无论 1 个还是 20 个 kernel
  都是 0%–1%，完全没有慢模态。所以"kernel 数多 → 间隙累积"解释不了。
- 慢样本与快样本的 SM 时钟完全相同，不是升频问题。
- 滞后自相关 +0.032，不是持久状态。

剩下的候选解释是 **host 侧 kernel 发射**：transformers 是 eager 执行，几十个 kernel 之间夹着
Python 开销；只要 CPU 一时没跟上，GPU 就出现气泡，而 CUDA Event 夹住的是整段（含气泡），
于是气泡被计入"执行时间"。

设计：2×2 对照
                   无 CPU 负载          有 CPU 负载（后台线程抢 GIL）
    eager          基线                 若气泡说成立 → 慢模态变多
    CUDA Graph     若气泡说成立 → 慢模态消失   同上

`CUDA Graph` 把整段前向录制成图，回放时由驱动一次性下发，host 侧几乎没有参与。

顺带回答一个已挂号的问题（AGENTS.md 第 8 节）：**掩码在 graph 回放时是否仍然生效**。
libsmctrl 的回调挂在 kernel 发射路径上，而 graph 回放走的是 `cuGraphLaunch`，是另一条路。
本脚本用「捕获时无掩码 / 回放时有掩码」与「捕获时有掩码 / 回放时无掩码」两个方向各测一次。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python scripts/probes/diag_graph_vs_eager.py
"""

from __future__ import annotations

import ctypes
import json
import statistics
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.libsmctrl_adapter import enabled_tpcs_to_native_mask, load_libsmctrl  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

SIZE = 224
REPS = 200
CPU_THREADS = 2


class CpuLoad:
    """后台忙循环，复现"CPU 一时没跟上"的场景。

    ``duty`` 是占空比：纯 Python 自旋会彻底饿死 GIL，实测把 3.5 ms 的前向抬到 4560 ms
    （1300 倍）——那证明的是"host 在关键路径上"，不是真实工况。真实场景是 CPU 偶尔被
    其他进程占用，所以主表用 duty=0.5 的温和负载，极端的 duty=1.0 单独作为旁证。
    """

    def __init__(self, threads: int, duty: float = 0.5):
        self._stop = threading.Event()
        self.duty = duty
        self._threads = [threading.Thread(target=self._spin, daemon=True) for _ in range(threads)]

    def _spin(self) -> None:
        on = max(1, int(20000 * self.duty))
        while not self._stop.is_set():
            for _ in range(on):
                pass
            if self.duty < 1.0:
                time.sleep(0.0005)

    def __enter__(self):
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        return False


def describe(rows: list[float]) -> dict:
    ordered = sorted(rows)
    n = len(ordered)
    median = statistics.median(ordered)
    slow = [v for v in ordered if v > 2 * median and v - median >= 2.0]
    return {
        "n": n,
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "cv": (statistics.stdev(ordered) / statistics.fmean(ordered)) if n > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "range_ratio": ordered[-1] / ordered[0] if ordered[0] > 0 else float("nan"),
        "slow_n": len(slow),
        "slow_fraction": len(slow) / n,
        "slow_median_ms": statistics.median(slow) if slow else None,
    }


def timed(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    cfg = load_config("config.libsmctrl.example.yaml")
    lib = load_libsmctrl()
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    model = enc.model
    mask_all = enabled_tpcs_to_native_mask(0, 12)
    mask_3 = enabled_tpcs_to_native_mask(0, 3)

    job = EncodeJob("graph-probe", seed=zlib.crc32(b"graph-probe") % (2**31),
                    width=SIZE, height=SIZE, deadline_ms=600_000)
    job.sm_fraction = 1.0
    with torch.inference_mode():
        pixels = enc._input(job)

    def eager():
        with torch.inference_mode():
            return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

    # ---- 捕获 CUDA Graph ----
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))   # 捕获时不施加掩码
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            captured = model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

    graph.replay()
    torch.cuda.synchronize()
    ref = eager()
    torch.cuda.synchronize()
    diff = float((captured.detach().float() - ref.detach().float()).abs().max().item())
    print(f"图回放输出与 eager 输出的最大绝对差 = {diff:.3e}  （应当接近 0，否则捕获不正确）")
    if diff > 1e-2:
        raise SystemExit("CUDA Graph 捕获结果与 eager 不一致，终止——不能拿错图做实验")

    def graph_run():
        graph.replay()

    # 两种执行方式各预热
    for _ in range(5):
        timed(eager)
        timed(graph_run)

    results: dict[str, dict] = {}
    # (名称, 执行函数, CPU 负载强度, 样本数)
    arms = [
        ("eager 无负载", eager, 0.0, REPS),
        ("eager 温和负载", eager, 0.5, REPS),
        ("eager 极端负载", eager, 1.0, 20),
        ("graph 无负载", graph_run, 0.0, REPS),
        ("graph 温和负载", graph_run, 0.5, REPS),
    ]
    for name, fn, duty, reps in arms:
        lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))   # 全部在无掩码下测，排除掩码干扰
        ctx = CpuLoad(CPU_THREADS, duty) if duty > 0 else None
        if ctx:
            ctx.__enter__()
        try:
            time.sleep(0.3)
            samples = [timed(fn) for _ in range(reps)]
        finally:
            if ctx:
                ctx.__exit__()
        results[name] = describe(samples)
        print(f"  完成 {name}", flush=True)
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))

    print(f"\n{'='*104}")
    print(f"CLIP {SIZE}×{SIZE}，无掩码，每组 {REPS} 次")
    print(f"{'='*104}")
    print(f"{'臂':>18} {'中位(ms)':>9} {'均值':>8} {'CV':>7} {'极差':>8} {'慢模态占比':>10} {'慢样本中位':>10}")
    print("-" * 104)
    for name, s in results.items():
        slow_med = f"{s['slow_median_ms']:.2f}" if s["slow_median_ms"] else "—"
        print(f"{name:>18} {s['median_ms']:>9.3f} {s['mean_ms']:>8.3f} {s['cv']:>7.3f} "
              f"{s['range_ratio']:>7.2f}× {s['slow_fraction']*100:>9.0f}% {slow_med:>10}")

    # ---- 顺带：掩码在 graph 回放时是否生效 ----
    print(f"\n{'='*104}")
    print("附加：掩码在 CUDA Graph 回放时是否生效")
    print(f"{'='*104}")
    mask_probe = {}
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
    base = statistics.median([timed(graph_run) for _ in range(20)])
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_3))
    masked = statistics.median([timed(graph_run) for _ in range(20)])
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
    back = statistics.median([timed(graph_run) for _ in range(20)])
    mask_probe = {"graph_nomask_ms": base, "graph_mask3_ms": masked, "graph_back_ms": back,
                  "ratio_masked_over_nomask": masked / base if base else float("nan")}
    print(f"  回放时无掩码 : {base:.3f} ms")
    print(f"  回放时 3 TPC : {masked:.3f} ms   （比值 {masked/base:.2f}×）")
    print(f"  再恢复无掩码 : {back:.3f} ms")
    if masked / base > 1.5:
        print("  → 掩码在回放时**生效**（捕获的是 TMD，回放走同一条发射路径）")
    else:
        print("  → 掩码在回放时**不生效**（回放走 cuGraphLaunch，绕开了回调）——"
              "这意味着 CUDA Graph 与 SM 掩码在本机无法同时使用")

    out = Path("results/reproducibility/probes/diag_graph_vs_eager.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": results, "mask_probe": mask_probe},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
