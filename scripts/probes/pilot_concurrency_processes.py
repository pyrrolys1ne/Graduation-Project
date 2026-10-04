#!/usr/bin/env python3
"""线程 vs 进程：主机侧发射瓶颈到底是不是 GIL/线程争用？

`pilot_concurrency_span.py` 已证明：N≥3 时**主机侧 kernel 发射耗时与 GPU span
几乎相等**（672 预生成：N=3 发射 22.0 / span 21.6 ms，N=8 发射 131.8 / span 128.8 ms），
即 GPU 在等主机发命令。但"主机侧"仍有两种可能：

  H-GIL  ：CPython 的 GIL + torch 的 Python 层 dispatch 在多线程下互相打断；
  H-DRV  ：CUDA 驱动/上下文在多线程并发发射时自身串行化。

判别（单变量：线程 → 进程，其余不变）：每个进程有独立的 Python 解释器与 CUDA 上下文，
GIL 不再共享；若吞吐随进程数接近线性、每请求 span 保持 ~6 ms，则 H-GIL 成立；
若进程同样塌陷，则瓶颈在驱动/GPU 侧，H-DRV 成立。

用法::

    .venv/bin/python scripts/probes/pilot_concurrency_processes.py --seconds 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend


def _worker(size: int, seconds: float, start_event, queue) -> None:
    """子进程：预生成输入后循环跑前向，回传每请求 GPU span。"""
    config = load_config()
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    backend = ClipEncoderBackend(config.model, resource, 1)
    torch = backend.torch
    job = EncodeJob("proc", seed=zlib.crc32(b"proc") % (2**31),
                    width=size, height=size, deadline_ms=10_000)
    pixels = backend._input(job)
    stream = backend.streams[0]
    start_event.wait()
    spans: list[float] = []
    stop_at = time.perf_counter() + seconds
    while time.perf_counter() < stop_at:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), torch.cuda.stream(stream):
            start.record(stream)
            backend.model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
            end.record(stream)
        end.synchronize()
        spans.append(float(start.elapsed_time(end)))
    queue.put((len(spans), spans))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--size", type=int, default=672)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path,
                        default=Path("results/probes/pilot_concurrency_processes.json"))
    args = parser.parse_args()

    context = mp.get_context("spawn")
    rows: list[dict] = []
    for n in args.counts:
        queue = context.Queue()
        start_event = context.Event()
        processes = [
            context.Process(target=_worker, args=(args.size, args.seconds, start_event, queue))
            for _ in range(n)
        ]
        for process in processes:
            process.start()
        time.sleep(6.0)  # 等子进程各自加载模型（每个进程一个 CUDA 上下文）
        start_event.set()
        started = time.perf_counter()
        for process in processes:
            process.join()
        wall = time.perf_counter() - started
        spans: list[float] = []
        count = 0
        for _ in processes:
            got_count, got_spans = queue.get()
            count += got_count
            spans.extend(got_spans)
        row = {
            "processes": n,
            "size": args.size,
            "count": count,
            "wall_s": wall,
            "throughput": count / wall if wall else 0.0,
            "span_median_ms": statistics.median(spans) if spans else float("nan"),
        }
        rows.append(row)
        print(f"  进程 N={n}  中位 span {row['span_median_ms']:>7.2f} ms  "
              f"吞吐 {row['throughput']:>7.1f} req/s  n={count}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
