"""气泡可测性探针：并发执行 CLIP 时，SM 到底有多少空闲？

背景
----
Bless（EuroSys 2025）的整个前提是"GPU 执行期间存在未被利用的**气泡**"，它的核心机制是：

    *"Bless allows each request to utilize the entire GPU whenever the resources are idle,
     and shrinks its resources instantly when other requests arrive."*

即**默认不分区，一有竞争立刻收缩**——同时拿到"零等待并发"与"隔离"两者的好处，
且**不需要预测**。这与本课题实测"分区被不分区并发全面支配"完全一致，
是本课题方法最自然的候选骨架。

**但它的前提是有气泡可挤。** 我们必须先测：本机并发执行 CLIP 时，SM 空闲率是多少？

Bless 自己给了一条对**我们有利**的趋势（§"SM number of GPU"）：

    *"as the number of SMs increases, the normalized latency reduction of Bless **decreases
     from 54.4% to 40.2%**... with a larger number of SMs, the application is **less likely to
     saturate all SMs**"*

本机只有 24 SM → 按此趋势，**气泡收益应比 A100 更大**。

判据
----
- **通过**：并发执行时实测 SM 空闲率 **> 10%** → 有气泡可挤，Bless 式机制值得做
- 死亡**：空闲率 **< 5%** → 无气泡，Bless 机制在本机无收益空间（**重要的负结果**）

测量手段
--------
1. **NVML**（`pynvml`）—— `nvmlDeviceGetUtilizationRates` 的 `gpu` 字段。
   注意 DNN-occu 的提醒：NVML utilization 是"宽松上界"（ResNet-50 训练时 NVML 到 90%
   而真实 occupancy 只有 45%）。**所以它测出"忙"不代表真的忙**；
   但它测出"闲"是可信的。
2. **ncu**（若可用）—— `sm__warps_active.avg.pct_of_peak_sustained_active`，真实 occupancy。
   ⚠️ 无 root 时驱动可能拒绝 counters，**脚本会自动探测**。

用法
----
    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python demos/demo_bubble_probe.py --concurrency 1,2,4
"""

from __future__ import annotations

import argparse
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
from encoder_sched.models import EncodeJob
from encoder_sched.resource import ProxyResourceBackend

SIZES = (224, 672)


def probe_nvml() -> dict:
    """探测 NVML 是否可用，并返回一次采样。"""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {"available": True, "handle": handle, "mod": pynvml}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


class Worker:
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


def job_for(size: int, tag: str) -> EncodeJob:
    j = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                  width=size, height=size, deadline_ms=600_000)
    j.sm_fraction = 1.0
    return j


def sample_loop(nvml: dict, stop: threading.Event, out: list) -> None:
    """后台线程按固定间隔采 NVML GPU 利用率。"""
    mod, handle = nvml["mod"], nvml["handle"]
    while not stop.is_set():
        try:
            u = mod.nvmlDeviceGetUtilizationRates(handle)
            out.append(u.gpu)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.02)


def measure(encoder: ClipEncoderBackend, size: int, concurrency: int,
            n_requests: int, nvml: dict) -> dict:
    """`concurrency` 个线程并发跑 `n_requests` 个请求，期间采样 GPU 利用率。"""
    wks = [Worker() for _ in range(concurrency)]
    barrier = threading.Barrier(concurrency)
    samples: list[int] = []
    stop = threading.Event()
    sampler = None
    if nvml["available"]:
        sampler = threading.Thread(target=sample_loop, args=(nvml, stop, samples), daemon=True)

    results: list[dict] = []

    def body(wk: Worker, tid: int):
        def go():
            for i in range(2):
                encoder.encode(job_for(size, f"warm-{tid}-{i}"), 0)
            barrier.wait()
            t0 = time.perf_counter()
            lat = [float(encoder.encode(job_for(size, f"{tid}-{i}"), 0).execution_ms)
                   for i in range(n_requests)]
            return {"lat": lat, "wall_ms": (time.perf_counter() - t0) * 1000}
        return wk.call(go)

    if sampler:
        sampler.start()
    threads = [threading.Thread(target=lambda k=k: results.append(body(wks[k], k)))
               for k in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000
    stop.set()
    if sampler:
        sampler.join(timeout=2)

    lat = [v for r in results for v in r["lat"]]
    busy = statistics.fmean(samples) if samples else float("nan")
    return {
        "size": size, "concurrency": concurrency, "wall_ms": wall,
        "median_ms": statistics.median(lat), "p99_ms": sorted(lat)[min(len(lat) - 1, int(0.99 * len(lat)))],
        "n": len(lat), "n_samples": len(samples),
        "gpu_util_mean": busy, "gpu_util_max": max(samples) if samples else None,
        "sm_idle_estimate": (1 - busy / 100) if samples else float("nan"),
    }


def probe_ncu() -> dict:
    """探测 ncu 是否可用来采真实 occupancy。"""
    import shutil
    import subprocess
    exe = shutil.which("ncu")
    if not exe:
        return {"available": False, "reason": "ncu 不在 PATH"}
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        return {"available": out.returncode == 0, "path": exe,
                "version": out.stdout.splitlines()[0] if out.stdout else ""}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="气泡可测性探针")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-thread", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/demo_bubble_probe.json"))
    args = parser.parse_args()

    nvml = probe_nvml()
    ncu = probe_ncu()
    print(f"NVML: {'可用' if nvml['available'] else '不可用 — ' + nvml.get('reason', '')}")
    print(f"ncu : {'可用 ' + ncu.get('version', '') if ncu['available'] else '不可用 — ' + ncu.get('reason', '')}")
    if not nvml["available"]:
        raise SystemExit("NVML 不可用，无法测量 GPU 利用率。请先 pip install nvidia-ml-py")

    config = load_config(args.config)
    encoder = ClipEncoderBackend(config.model, ProxyResourceBackend(), 1)
    concurrencies = [int(c) for c in args.concurrency.split(",")]

    rows: list[dict] = []
    for size in SIZES:
        for conc in concurrencies:
            r = measure(encoder, size, conc, args.requests_per_thread, nvml)
            rows.append(r)
            print(f"  {size}×{size} 并发={conc}: 墙钟 {r['wall_ms']:.0f} ms, "
                  f"中位 {r['median_ms']:.3f} ms, GPU util 均值 {r['gpu_util_mean']:.1f}% "
                  f"({r['n_samples']} 样本)", flush=True)

    print(f"\n{'='*96}")
    print("气泡可测性")
    print(f"{'='*96}")
    print(f"{'尺寸':>6} {'并发':>4} {'中位(ms)':>10} {'P99(ms)':>9} {'墙钟(ms)':>10} "
          f"{'GPU util':>9} {'SM 空闲(估)':>12} {'样本':>6}")
    print("-" * 96)
    for r in rows:
        print(f"{r['size']:>6} {r['concurrency']:>4} {r['median_ms']:>10.3f} {r['p99_ms']:>9.3f} "
              f"{r['wall_ms']:>10.0f} {r['gpu_util_mean']:>8.1f}% "
              f"{r['sm_idle_estimate']*100:>11.1f}% {r['n_samples']:>6}")

    print(f"\n{'='*96}")
    print("判据")
    print(f"{'='*96}")
    # 取"单请求"与"最高并发"两组来判：气泡 = 并发时仍未用满的部分
    for size in SIZES:
        solo = next((r for r in rows if r["size"] == size and r["concurrency"] == 1), None)
        peak = max((r for r in rows if r["size"] == size), key=lambda r: r["concurrency"])
        if not solo or not peak:
            continue
        print(f"\n  {size}×{size}")
        print(f"    单请求 GPU util {solo['gpu_util_mean']:.1f}% "
              f"→ 空闲 {solo['sm_idle_estimate']*100:.0f}%")
        print(f"    并发 {peak['concurrency']} GPU util {peak['gpu_util_mean']:.1f}% "
              f"→ 空闲 {peak['sm_idle_estimate']*100:.0f}%")
        gain = solo["gpu_util_mean"]
        if peak["sm_idle_estimate"] > 0.10:
            print(f"    ✅ 通过：并发下仍有 {peak['sm_idle_estimate']*100:.0f}% 空闲"
                  f"（>10%）→ **有气泡可挤**，Bless 式机制值得做")
        elif peak["sm_idle_estimate"] > 0.05:
            print(f"    ⚠️ 边缘：空闲 {peak['sm_idle_estimate']*100:.0f}%，"
                  f"介于 5%–10%，收益空间有限")
        else:
            print(f"    ❌ 判死：空闲仅 {peak['sm_idle_estimate']*100:.0f}%（<5%）"
                  f"→ 无气泡，Bless 机制在本机没有收益空间")

    print(f"\n  ⚠️ 注意：NVML 的 gpu util 是**宽松上界**（DNN-occu 实测：ResNet-50 训练时")
    print(f"     NVML 到 90% 而真实 occupancy 只有 45%）。它测出'忙'不代表真忙，")
    print(f"     但**测出'闲'是可信的**。要精确量化需用 ncu 采真实 occupancy。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"nvml": {"available": nvml["available"]}, "ncu": ncu, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
