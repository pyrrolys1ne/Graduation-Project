"""判别实验：分离「host 发射开销」与「波量化」两个机制。

背景
----
本课题此前发现：CLIP 前向在小请求上不可复现、存在重尾慢模态，并用 **CUDA Graph 干预实验**
证实其主导项是 **host 侧 kernel 发射开销**（224×224 事件计时里 host 占 59%，随尺寸单调降到
672 的 7%；graph 回放把四个尺寸全部压到 0% 慢模态）。

本轮读 Bullet（ASPLOS 2026）又发现**第二个独立机制**：**波量化**——
线程块数 g 不是"SM 数 N × 每 SM 槽位 b"的整数倍时，最后一波只有部分 SM 有活干：

    waves w = ⌈ g / (b·N) ⌉
    tail    = ⌈ g/b − N·(w−1) ⌉
    SM 空闲比 idle = (N − tail) / (N · w)

手算结果（N=24、b=2、128×128 tile）：

    224×224（seq=50）：QKV 62.5% / MLP-fc1 50.0% / MLP-fc2 87.5%
    672×672（seq=442）：QKV 25.0% / MLP-fc1  0.0% / MLP-fc2 50.0%

**小请求空闲 50–87.5%、大请求 0–50%，单调下降**，与实测"尺寸越大越稳"同向。

两个机制的区别与本实验的设计
----------------------------
============  ==============================  ==============================
机制            CUDA Graph 能否消除              为什么
============  ==============================  ==============================
host 发射开销    ✅ **能**（host 已移出关键路径）   中位数 3.99→1.64 ms，慢模态 22%→0%
波量化          ❌ **不能**                       它是 GPU 自身的 occupancy 问题，
                                              该空转的 SM 还是空转
============  ==============================  ==============================

**因此：在 CUDA Graph 路径下重测四个尺寸，看"尺寸越大越稳"这个特征是否仍然保留。**

- 若 **graph 下仍保留**尺寸依赖 → 两机制**独立**，波量化是第二来源
- 若 **graph 下尺寸依赖消失** → 波量化不是独立来源，全部归 host

**两种结果都直接决定方法设计**：若是前者，调度器可以**解析地预测**"这个尺寸该不该提高并发
去填空转的 SM"，而这**不需要任何测量**。

判据
----
- **通过**：graph 路径下，`CV(延迟)` 或 `慢模态占比` 仍随尺寸单调下降 → 存在第二个机制
- **死亡**：graph 路径下尺寸依赖消失 → 波量化被内部调度抹平（或本就不显著），
  结论应写为"host 是唯一来源"

⚠️ 本机限制：**掩码在 CUDA Graph 回放时不生效**，所以本实验全程不加掩码（满配额）。

用法
----
    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python demos/demo_mechanism_split.py --reps 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

SIZES = (224, 336, 448, 672)


def describe(samples: list[float]) -> dict:
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    std = statistics.stdev(ordered) if n > 1 else 0.0
    slow = [v for v in ordered if v > 2 * median and v - median >= 2.0]
    return {
        "n": n, "median_ms": median, "mean_ms": mean,
        "cv": std / mean if mean else float("nan"),
        "min_ms": ordered[0], "max_ms": ordered[-1],
        "range_ratio": ordered[-1] / ordered[0] if ordered[0] else float("nan"),
        "p99_ms": ordered[min(n - 1, int(0.99 * n))],
        "slow_fraction": len(slow) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分离 host 开销与波量化")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/demo_mechanism_split.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    # 用 proxy 后端：本实验比较的是执行路径（eager vs graph），与配额无关，
    # 且掩码在 graph 回放时不生效，加掩码只会引入不可控变量。
    encoder = ClipEncoderBackend(config.model, ProxyResourceBackend(), 1)
    model = encoder.model
    torch_ = encoder.torch

    def timed(fn) -> float:
        s = torch_.cuda.Event(enable_timing=True)
        e = torch_.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch_.cuda.synchronize()
        return s.elapsed_time(e)

    results: dict[str, dict] = {}
    for size in SIZES:
        job = EncodeJob(f"split-{size}", seed=size, width=size, height=size,
                        deadline_ms=600_000)
        job.sm_fraction = 1.0
        with torch_.inference_mode():
            pixels = encoder._input(job)

        def forward():
            with torch_.inference_mode():
                return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

        # ---- eager ----
        for _ in range(5):
            timed(forward)
        eager_samples = [timed(forward) for _ in range(args.reps)]

        # ---- CUDA Graph ----
        side = torch_.cuda.Stream()
        side.wait_stream(torch_.cuda.current_stream())
        with torch_.cuda.stream(side):
            for _ in range(3):
                forward()
        torch_.cuda.current_stream().wait_stream(side)
        torch_.cuda.synchronize()

        graph = torch_.cuda.CUDAGraph()
        with torch_.inference_mode():
            with torch_.cuda.graph(graph):
                captured = model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
        graph.replay()
        torch_.cuda.synchronize()
        ref = forward()
        torch_.cuda.synchronize()
        diff = float((captured.detach().float() - ref.detach().float()).abs().max().item())
        if diff > 1e-2:
            raise SystemExit(f"{size}: CUDA Graph 与 eager 输出不一致（{diff:.2e}），终止")

        for _ in range(10):
            timed(graph.replay)
        graph_samples = [timed(graph.replay) for _ in range(args.reps)]

        results[str(size)] = {"eager": describe(eager_samples), "graph": describe(graph_samples),
                              "graph_maxdiff": diff}
        print(f"  {size}×{size} 完成: eager 中位 {results[str(size)]['eager']['median_ms']:.3f} "
              f"/ graph 中位 {results[str(size)]['graph']['median_ms']:.3f} ms", flush=True)

    print(f"\n{'='*104}")
    print(f"分离 host 开销与波量化（每臂 {args.reps} 次）")
    print(f"{'='*104}")
    print(f"{'尺寸':>6} | {'eager 中位':>10} {'eager CV':>9} {'慢模态':>7} | "
          f"{'graph 中位':>10} {'graph CV':>9} {'慢模态':>7} | {'host 占比':>9}")
    print("-" * 104)
    for size in SIZES:
        r = results[str(size)]
        e, g = r["eager"], r["graph"]
        host = 1 - g["median_ms"] / e["median_ms"]
        print(f"{size:>6} | {e['median_ms']:>10.3f} {e['cv']:>9.3f} {e['slow_fraction']*100:>6.0f}% | "
              f"{g['median_ms']:>10.3f} {g['cv']:>9.3f} {g['slow_fraction']*100:>6.0f}% | "
              f"{host*100:>8.0f}%")

    print(f"\n{'='*104}")
    print("判读：graph 路径下「尺寸依赖」是否仍然保留")
    print(f"{'='*104}")
    e_cv = [results[str(s)]["eager"]["cv"] for s in SIZES]
    g_cv = [results[str(s)]["graph"]["cv"] for s in SIZES]
    e_slow = [results[str(s)]["eager"]["slow_fraction"] for s in SIZES]
    g_slow = [results[str(s)]["graph"]["slow_fraction"] for s in SIZES]
    mono_e = all(a >= b - 1e-9 for a, b in zip(e_cv, e_cv[1:]))
    mono_g = all(a >= b - 1e-9 for a, b in zip(g_cv, g_cv[1:]))
    print(f"  eager CV 随尺寸: {[round(v,3) for v in e_cv]}  单调递减={mono_e}")
    print(f"  graph CV 随尺寸: {[round(v,3) for v in g_cv]}  单调递减={mono_g}")
    print(f"  eager 慢模态:    {[f'{v*100:.0f}%' for v in e_slow]}")
    print(f"  graph 慢模态:    {[f'{v*100:.0f}%' for v in g_slow]}")
    print()
    if max(g_slow) > 0.02 or (mono_g and g_cv[0] > g_cv[-1] * 1.3):
        print("  ✅ 通过：graph 路径下**仍保留**尺寸依赖")
        print("     → 存在独立于 host 的第二个机制（波量化）；")
        print("       调度器可**解析地预测**该尺寸的 SM 空闲比，不需要测量")
    else:
        print("  ❌ 判死：graph 路径下尺寸依赖基本消失")
        print("     → host 发射开销是唯一来源，波量化被内部调度抹平（或本就不显著）")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"reps": args.reps, "results": results,
         "mono_eager_cv": mono_e, "mono_graph_cv": mono_g},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
