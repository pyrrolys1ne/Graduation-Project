"""探针：把"CPU 图像准备"与"CUDA Graph 回放"拆成两级流水线，能否避开死锁并保住吞吐？

为什么需要这个探针
──────────────────

`pilot_graph_concurrency.py` 已经测出：**图回放把主机侧发射耗时从 3.97 ms 压到 0.04 ms，
eager 在 N≥3 的塌陷随之消失**（eager 峰值 173.6 req/s @N=2；`graph_prebuilt` 348.6 @N=3
且单调不降）。

但 `graph_prebuilt` 臂**不做 CPU 造图**。把真实预处理放回去（`graph_pinned_inplace` 臂）后，
N=2 还有 303 req/s，**N=4 就挂死了**。二分结果（`/tmp` 一次性复现脚本，结论抄录如下）：

| 变体 | N=4，4 尺寸 | 结果 |
|---|---|---|
| 只做 d2d 拷贝（无 CPU 计算、无 H2D） | 354.8 req/s | 正常 |
| 分页源 H2D，**无** CPU 计算 | 324.9 req/s | 正常 |
| 锁页源 H2D，CPU 只做一趟 `uniform_` | 363.4 req/s | 正常 |
| 分页源 H2D + CPU 造图 | — | **挂死** |
| 锁页源 H2D + CPU 造图（3 趟），同一线程 | — | **挂死**（N=4） |

挂死的形态一致：**所有线程卡在各自的 `stream.synchronize()` 上，GPU 永不完结**。
即"CPU 有持续负载"与"另一条流上有图正在回放"这两件事同时发生时，驱动/运行时进入互等。

因此本探针检验的是一个**架构假设**，而不是又一个参数：

> 把 CPU 图像准备与图回放**放进不同的线程**，两级之间用**锁页缓冲队列**衔接，
> 死锁是否消失、吞吐是否保住？

实现
────

    producer 线程池（N_PREP 个）：只做 CPU 造图，填进锁页缓冲 → 入队
    consumer 线程池（N_REPLAY 个）：出队 → 锁页→显存 → 显存→图静态输入 → 回放 → 流同步 → 还缓冲

两级各自可以独立调并行度，这正是一个**调度策略**该有的形状：并发度（回放线程数）
与预处理并行度是两个解耦的旋钮。

判据
────

1. 任一配置下**不得挂死**（跑满 `--duration` 秒并正常退出）。
2. 吞吐应接近 `graph_prebuilt` 的水平（同尺寸混合下约 350 req/s），
   即流水线把 CPU 预处理藏了起来，而不是把它变成新瓶颈。
3. `--replay 1` 时吞吐应明显低于 `--replay 2`（说明回放级并发仍是有效旋钮）；
   若 `--replay 1` 已经饱和，说明瓶颈转移到了别处，需要重新定位。

用法：
    .venv-linux/bin/python scripts/probes/pilot_graph_pipeline.py --replay 1,2,4 --prep 2,4 --duration 6

⚠️ 本脚本会独占 GPU。跑之前先确认宿主侧空闲。
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

#: 与 saturating_4ms.jsonl 一致的尺寸构成（25/25/25/25）。
SIZES = (224, 336, 448, 672)
#: 每尺寸捕获几张图。必须 ≥ 最大回放线程数：同一张图的两个实例共享静态输入缓冲。
INSTANCES = 8
#: 锁页缓冲池深度。太浅会让 producer 阻塞在取缓冲上，掩盖流水线效果。
PIN_POOL = 8


class GraphPool:
    """按尺寸捕获 CUDA Graph；每尺寸 `instances` 份互不共享静态缓冲的副本。"""

    def __init__(self, model, torch_mod, dtype, sizes=SIZES, instances=INSTANCES):
        self.torch = torch_mod
        self.entries: dict[int, list[dict]] = {}
        for size in sizes:
            start = time.perf_counter()
            self.entries[size] = [self._capture_one(model, dtype, size) for _ in range(instances)]
            print(f"  已捕获 {size}×{size} × {instances} 份，"
                  f"单份 {(time.perf_counter()-start)*1000/instances:.0f} ms", flush=True)

    def _capture_one(self, model, dtype, size: int) -> dict:
        torch_mod = self.torch
        static_in = torch_mod.empty((1, 3, size, size), dtype=dtype, device="cuda")
        static_in.normal_()
        side = torch_mod.cuda.Stream()
        side.wait_stream(torch_mod.cuda.current_stream())
        with torch_mod.cuda.stream(side):
            for _ in range(3):
                with torch_mod.inference_mode():
                    model(pixel_values=static_in, interpolate_pos_encoding=True)
        torch_mod.cuda.current_stream().wait_stream(side)
        torch_mod.cuda.synchronize()

        graph = torch_mod.cuda.CUDAGraph()
        with torch_mod.inference_mode():
            with torch_mod.cuda.graph(graph):
                captured = model(pixel_values=static_in, interpolate_pos_encoding=True).image_embeds
        graph.replay()
        torch_mod.cuda.synchronize()
        with torch_mod.inference_mode():
            reference = model(pixel_values=static_in, interpolate_pos_encoding=True).image_embeds
        torch_mod.cuda.synchronize()
        diff = float((captured.detach().float() - reference.detach().float()).abs().max().item())
        if diff > 1e-2:
            raise SystemExit(f"{size}×{size} 的 CUDA Graph 输出与 eager 不符"
                             f"（最大绝对差 {diff:.3e}），不能拿错图做实验")
        return {"graph": graph, "input": static_in, "output": captured}


def run_pipeline(pool: GraphPool, model, mean, std, torch_mod, dtype,
                 n_replay: int, n_prep: int, duration_s: float) -> dict:
    """跑一次流水线，返回吞吐与各级耗时统计。"""
    free_pins: dict[int, queue.Queue] = {s: queue.Queue() for s in SIZES}
    for size in SIZES:
        for _ in range(PIN_POOL):
            free_pins[size].put(
                torch_mod.empty((1, 3, size, size), dtype=torch_mod.float32).pin_memory())
    ready: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()
    produced = [0]
    consumed = [0]
    prep_ms: list[float] = []
    replay_ms: list[float] = []
    lock = threading.Lock()
    deadline = time.perf_counter() + duration_s

    def producer(pid: int) -> None:
        gen = torch_mod.Generator(device="cpu").manual_seed(pid + 1)
        index = pid
        while not stop.is_set():
            size = SIZES[index % len(SIZES)]
            index += n_prep
            buf = free_pins[size].get()
            t0 = time.perf_counter()
            buf.uniform_(generator=gen)
            buf.sub_(mean).div_(std)   # (3,1,1) 广播到 (1,3,H,W)
            elapsed = (time.perf_counter() - t0) * 1000
            ready.put((size, buf))
            with lock:
                produced[0] += 1
                prep_ms.append(elapsed)
            if time.perf_counter() >= deadline:
                stop.set()

    def consumer(cid: int) -> None:
        stream = torch_mod.cuda.Stream()
        staging = {s: torch_mod.empty((1, 3, s, s), dtype=dtype, device="cuda") for s in SIZES}
        while not stop.is_set():
            try:
                size, buf = ready.get(timeout=0.5)
            except queue.Empty:
                continue
            slot = pool.entries[size][cid % INSTANCES]
            t0 = time.perf_counter()
            with torch_mod.cuda.stream(stream):
                staging[size].copy_(buf, non_blocking=True)   # 锁页源 → 真异步 H2D
                slot["input"].copy_(staging[size])
                slot["graph"].replay()
            stream.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            free_pins[size].put(buf)
            with lock:
                consumed[0] += 1
                replay_ms.append(elapsed)

    consumers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(n_replay)]
    producers = [threading.Thread(target=producer, args=(i,), daemon=True) for i in range(n_prep)]
    t0 = time.perf_counter()
    for t in consumers:
        t.start()
    for t in producers:
        t.start()
    time.sleep(duration_s)
    stop.set()
    wall = time.perf_counter() - t0
    for t in consumers + producers:
        t.join(timeout=2)

    def median(rows):
        return sorted(rows)[len(rows) // 2] if rows else None

    return {
        "requested_replay": n_replay,
        "requested_prep": n_prep,
        "consumed": consumed[0],
        "produced": produced[0],
        "wall_s": wall,
        "throughput_rps": consumed[0] / wall,
        "prep_ms_median": median(prep_ms),
        "replay_ms_median": median(replay_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="1,2,4", help="回放线程数，逗号分隔")
    parser.add_argument("--prep", default="2,4", help="预处理线程数，逗号分隔")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out", default="results/probes/pilot_graph_pipeline.json")
    args = parser.parse_args()

    cfg = load_config("config.libsmctrl.example.yaml")
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), INSTANCES)
    model, dtype = enc.model, enc.dtype

    print("捕获 CUDA Graph …", flush=True)
    pool = GraphPool(model, enc.torch, dtype)
    print("正确性对照：所有图与 eager 输出一致\n", flush=True)

    results = []
    for n_replay in [int(x) for x in args.replay.split(",")]:
        for n_prep in [int(x) for x in args.prep.split(",")]:
            time.sleep(0.5)
            row = run_pipeline(pool, model, enc.mean, enc.std, enc.torch, dtype,
                               n_replay, n_prep, args.duration)
            results.append(row)
            print(f"replay={n_replay} prep={n_prep}  "
                  f"吞吐 {row['throughput_rps']:6.1f} req/s  "
                  f"消费 {row['consumed']:5d}  生产 {row['produced']:5d}  "
                  f"预处理 {row['prep_ms_median'] or 0:5.2f} ms  "
                  f"回放+同步 {row['replay_ms_median'] or 0:5.2f} ms", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
