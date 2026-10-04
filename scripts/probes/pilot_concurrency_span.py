#!/usr/bin/env python3
"""共驻并发下 GPU 时间为何超线性增长：GPU 争用，还是主机侧喂不饱？

背景（`docs/实验记录.md` §21.3 + `an5_concurrency_attribution.py`）：服务层实测
每请求 GPU 时间随共驻数 N 超线性增长（N=8 时 224 为 26×、672 为 21×），
于是吞吐在 N=2 见顶、之后塌陷。两种解释都符合这个现象：

  H-GPU ：GPU 侧资源（带宽/L2/SM）被争用，多出来的并发只是互相拖慢。
  H-HOST：主机侧喂不饱 GPU——每请求都要在 CPU 上造图（`torch.rand` + 归一化）
          再拷贝到 GPU，8 个线程抢 GIL 时 kernel 发射被推迟，GPU 出现空泡，
          而空泡被记进了 CUDA event 的 elapsed（事件在同一 stream 内首尾包夹）。

判别方法（单变量）：同一块 GPU、同样的线程→stream 1:1 映射，只改一件事——
输入是"每次现造"（H-HOST 会发作）还是"预生成后复用"（H-HOST 被摘除）。

  - 若两种模式的超线性增长**同样严重** → H-GPU 成立（主机侧不是主因）。
  - 若预生成模式随 N 基本线性、现造模式才塌 → H-HOST 成立。

用法::

    .venv/bin/python scripts/probes/pilot_concurrency_span.py [--seconds 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import zlib

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

CONCURRENCY = (1, 2, 3, 4, 6, 8)
SIZES = (224, 672)
MODES = ("service", "prebuilt")


def build_backend(stream_count: int) -> ClipEncoderBackend:
    config = load_config()
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    return ClipEncoderBackend(config.model, resource, stream_count)


def make_job(size: int, tag: str) -> EncodeJob:
    return EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                     width=size, height=size, deadline_ms=10_000)


class PowerSampler:
    """运行期间采样功耗与利用率。

    短 kernel 和多流并发下，单一利用率快照不足以还原整个测量区间。功耗与利用率
    联合采样可用于区分
    「GPU 资源被争用」与「主机侧发射跟不上、GPU 在空转」的判别量。
    """

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                ).stdout.strip().splitlines()[0]
                power, utilization = (float(part.strip()) for part in out.split(","))
                self.samples.append((power, utilization))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                continue

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if not self.samples:
            return {"power_w": None, "utilization_percent": None, "samples": 0}
        powers = [p for p, _ in self.samples]
        utils = [u for _, u in self.samples]
        return {
            "power_w": statistics.fmean(powers),
            "power_max_w": max(powers),
            "utilization_percent": statistics.fmean(utils),
            "samples": len(self.samples),
        }


def run_case(backend: ClipEncoderBackend, size: int, n: int, mode: str, seconds: float) -> dict:
    """N 个线程各跑同一尺寸，持续 `seconds` 秒；每请求 GPU 时间取自 CUDA event。"""
    torch = backend.torch
    spans: list[float] = []
    enqueue_ms: list[float] = []
    lock = threading.Lock()
    stop_at = time.perf_counter() + seconds
    ready = threading.Barrier(n + 1)
    pixels_cache: dict[int, object] = {}

    def body(worker_id: int) -> None:
        job = make_job(size, f"{mode}-{size}-{n}-{worker_id}")
        stream = backend.streams[worker_id % len(backend.streams)]
        local: list[float] = []
        local_enqueue: list[float] = []
        if mode == "prebuilt":
            # 只在进入循环前造一次图：循环内不再有 CPU 造图与 H2D 拷贝。
            pixels_cache[worker_id] = backend._input(job)
        ready.wait()
        while time.perf_counter() < stop_at:
            if mode == "service":
                # 与线上完全同一条路径：CPU 造图 + H2D + 前向，全部在事件之外。
                local.append(backend.encode(job, worker_id).execution_ms)
                continue
            pixels = pixels_cache[worker_id]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            enqueue_started = time.perf_counter_ns()
            with torch.inference_mode(), torch.cuda.stream(stream):
                start.record(stream)
                backend.model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
                end.record(stream)
            # 这一步只把 kernel 排进队列，不等 GPU。它测的是**主机侧发射耗时**：
            # 若它随 N 增长到与 GPU span 同量级，说明 GPU 在等主机发命令。
            enqueue = (time.perf_counter_ns() - enqueue_started) / 1e6
            end.synchronize()
            local.append(float(start.elapsed_time(end)))
            local_enqueue.append(enqueue)
        with lock:
            spans.extend(local)
            enqueue_ms.extend(local_enqueue)

    threads = [threading.Thread(target=body, args=(index,), daemon=True) for index in range(n)]
    for thread in threads:
        thread.start()
    sampler = PowerSampler()
    sampler.start()
    ready.wait()
    started = time.perf_counter()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    power = sampler.stop()
    return {
        "size": size,
        "n": n,
        "mode": mode,
        "count": len(spans),
        "wall_s": wall,
        "throughput": len(spans) / wall if wall else 0.0,
        "span_median_ms": statistics.median(spans) if spans else float("nan"),
        "span_mean_ms": statistics.fmean(spans) if spans else float("nan"),
        "enqueue_mean_ms": statistics.fmean(enqueue_ms) if enqueue_ms else None,
        **power,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/probes/pilot_concurrency_span.json"))
    args = parser.parse_args()

    # stream 池固定铺满 8 条，保证任意 N 下都是线程与 stream 1:1（消除服务层
    # "8 个 worker 抢 2 条流"那个混淆项）。
    backend = build_backend(max(CONCURRENCY))
    print(f"设备 {backend.device}，streams={len(backend.streams)}，每档 {args.seconds}s", flush=True)

    rows: list[dict] = []
    for size in SIZES:
        for mode in MODES:
            baseline = None
            tp_n1 = None
            for n in CONCURRENCY:
                row = run_case(backend, size, n, mode, args.seconds)
                if baseline is None:
                    baseline = row["span_median_ms"]
                    tp_n1 = row["throughput"]
                row["span_vs_n1"] = row["span_median_ms"] / baseline
                row["throughput_vs_n1"] = row["throughput"] / tp_n1 if tp_n1 else float("nan")
                rows.append(row)
                print(f"  {size}×{size} {mode:<9} N={n:<2} "
                      f"中位 {row['span_median_ms']:>7.2f} ms ({row['span_vs_n1']:>5.2f}×)  "
                      f"吞吐 {row['throughput']:>7.1f} req/s  n={row['count']:<4} "
                      f"功耗 {row['power_w'] or 0:>5.1f} W  利用率 {row['utilization_percent'] or 0:>5.1f}%"
                      + (f"  发射 {row['enqueue_mean_ms']:>6.2f} ms" if row["enqueue_mean_ms"] else ""),
                      flush=True)
            print(flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
