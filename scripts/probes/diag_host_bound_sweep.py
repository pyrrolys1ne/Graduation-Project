"""尺寸扫描：CLIP 前向的"host 受限程度"如何随请求尺寸变化？

已确认（diag_graph_vs_eager.py）：
- 224×224 的 eager 前向中位 3.12 ms，其中 **1.83 ms 是 host 侧的 kernel 发射开销**
  （CUDA Graph 回放同样计算只要 1.83 ms），慢模态占比 34% → graph 下 0%。
- graph 回放对 CPU 负载几乎免疫；eager 在温和 CPU 负载下中位翻到 7.39 ms。

**prediction**：如果"host 受限"就是尺寸边界的来源，那么随着请求变大（每个 kernel 的
GPU 工作量增加），host 的发射开销占比应下降，`eager/graph` 比值应向 1 收敛，
慢模态占比也应下降——即小请求 host 受限、大请求 GPU 受限。

本脚本对 224/336/448/672 各测 eager 与 graph 两臂，给出这个过渡曲线。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python scripts/probes/diag_host_bound_sweep.py
"""

from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.libsmctrl_adapter import load_libsmctrl  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

SIZES = (224, 336, 448, 672)
REPS = 200


def describe(rows: list[float]) -> dict:
    ordered = sorted(rows)
    n = len(ordered)
    median = statistics.median(ordered)
    slow = [v for v in ordered if v > 2 * median and v - median >= 2.0]
    return {
        "n": n, "median_ms": median, "mean_ms": statistics.fmean(ordered),
        "cv": (statistics.stdev(ordered) / statistics.fmean(ordered)) if n > 1 else 0.0,
        "min_ms": ordered[0], "max_ms": ordered[-1],
        "range_ratio": ordered[-1] / ordered[0] if ordered[0] > 0 else float("nan"),
        "p95_ms": ordered[min(n - 1, int(0.95 * n))],
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
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))   # 全程无掩码，只比较 host 侧差异

    results: dict[int, dict] = {}
    for size in SIZES:
        job = EncodeJob(f"sweep-{size}", seed=zlib.crc32(f"sweep-{size}".encode()) % (2**31),
                        width=size, height=size, deadline_ms=600_000)
        job.sm_fraction = 1.0
        with torch.inference_mode():
            pixels = enc._input(job)

        def eager():
            with torch.inference_mode():
                return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

        # 捕获前在旁路流上暖机（torch 的要求）
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
        if diff > 1e-2:
            raise SystemExit(f"{size}: CUDA Graph 与 eager 输出不一致（差 {diff:.2e}），终止")

        def graph_run():
            graph.replay()

        for _ in range(5):
            timed(eager)
            timed(graph_run)
        time.sleep(0.3)

        eager_samples = [timed(eager) for _ in range(REPS)]
        graph_samples = [timed(graph_run) for _ in range(REPS)]
        results[size] = {"eager": describe(eager_samples), "graph": describe(graph_samples),
                         "graph_output_maxdiff": diff}
        e, g = results[size]["eager"], results[size]["graph"]
        print(f"  {size}×{size} 完成: eager {e['median_ms']:.2f} ms / graph {g['median_ms']:.2f} ms",
              flush=True)

    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))

    print(f"\n{'='*112}")
    print(f"CLIP 前向：eager vs CUDA Graph（无掩码，每臂 {REPS} 次）")
    print(f"{'='*112}")
    print(f"{'尺寸':>6} {'eager中位':>10} {'graph中位':>10} {'host占比':>9} {'eager慢模态':>11} "
          f"{'graph慢模态':>11} {'eager极差':>10} {'graph极差':>10}")
    print("-" * 112)
    rows = []
    for size, r in results.items():
        e, g = r["eager"], r["graph"]
        host_share = (e["median_ms"] - g["median_ms"]) / e["median_ms"]
        row = {"size": size, "eager_median": e["median_ms"], "graph_median": g["median_ms"],
               "host_share": host_share,
               "eager_slow": e["slow_fraction"], "graph_slow": g["slow_fraction"],
               "eager_range": e["range_ratio"], "graph_range": g["range_ratio"],
               "eager": e, "graph": g}
        rows.append(row)
        print(f"{size:>6} {e['median_ms']:>10.3f} {g['median_ms']:>10.3f} "
              f"{host_share*100:>8.0f}% {e['slow_fraction']*100:>10.0f}% "
              f"{g['slow_fraction']*100:>10.0f}% {e['range_ratio']:>9.2f}× "
              f"{g['range_ratio']:>9.2f}×")

    print(f"\n判据：若 host 占比随尺寸单调下降 → \"尺寸边界\"就是\"host 受限 → GPU 受限\"的过渡")
    shares = [r["host_share"] for r in rows]
    mono = all(a >= b for a, b in zip(shares, shares[1:]))
    print(f"  host 占比序列: {[round(s,3) for s in shares]}  "
          f"{'（单调下降 ✔）' if mono else '（非单调 ✘）'}")

    out = Path("results/reproducibility/probes/diag_host_bound_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "monotonic": mono},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
