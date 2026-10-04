"""探针：CUDA Graph 回放能否消掉 N≥3 的并发塌陷？

为什么现在才做这个
──────────────────

§10.5 已经测过 graph vs eager（224：3.99 → 1.64 ms，host 占比 59% → 0%），
但当时立刻撞上一个机制冲突：**libsmctrl 的回调挂在 kernel 发射路径上，而 graph 回放走
`cuGraphLaunch`，绕开了回调**（§10.7）。于是"CUDA Graph 方向"被挂起——因为当时并发度
研究的自变量是"每请求分多少 TPC"。

**现在这个障碍不成立了**：TPC 空间分割已被证伪（本机互斥分区劣于共享 SM 多流），
`proxy` 后端本来就是零副作用。也就是说，本课题剩下的唯一控制轴是**共驻数量**，
而它不需要掩码。

于是 §22.6 的第 2 条归因可以正面攻击了：

> N≥3 的塌陷来自 Python 线程侧的 kernel 发射（GIL / torch Python 层 dispatch 互相打断），
> 不是 GPU 资源争用。进程对照下吞吐恒定（108→117→108）。

**如果这个归因是对的，那么把发射开销从关键路径上拿掉（graph 回放），塌陷就应该消失。**
这是一个可证伪的预测，本脚本就是它的判决实验。

设计
────

统一尺寸混合（224/336/448/672 各 25%，与 `saturating_4ms.jsonl` 一致），
N 个线程 × N 条流（1:1，排除 §22.5 那种"流池不够"的管线缺陷）。

| 臂 | 每次迭代做什么 | 隔离出什么 |
|---|---|---|
| `eager` | CPU 造图 + 归一化 + H2D + eager 前向 | 服务现状（§22.2 的 service 臂） |
| `graph_prebuilt` | 预生成输入拷进锁页缓冲 + 回放 | 只去掉 **eager 的发射路径**；**不含 CPU 造图** |
| `graph_pinned_stage` | 完整 `make_input` + 拷进锁页缓冲 + 回放 | 去掉发射，完整保留预处理成本 |
| `graph_pinned_inplace` | 直接在锁页缓冲里造图 + 回放 | 同上，但省掉大块 CPU 临时分配 |

⚠️ **不要把这些臂的循环体当作服务集成模板。** 这里的 CPU 造图与图回放**同处一个线程**，
在 N 足够大、CPU 负载足够重时会**挂死**（所有线程卡在 `stream.synchronize()` 上，
GPU 永不完结）。H2D 的源用锁页可以缓解，但**不能消除**——真正的解法是把两级拆到
不同线程，见 `pilot_graph_pipeline.py` 与 `docs/实验记录.md` §23.2。

`graph_pinned_inplace` 在 N=2 可跑出 303.0 req/s，N=4 即挂死；这正是"同线程"这一条
约束的实测边界。

判据（预先写死，避免事后挑数）：

1. `eager` 的吞吐应随 N **先升后塌**（复现 §22.2：672 N=1 137.6 → N=4 78.5 req/s）。
   若不复现，说明本机状态与 §22 那批不同，本探针结论不成立。
2. 若塌陷来自主机侧发射，则 `graph*` 的吞吐应**单调不降**，且在 N≥3 处显著高于 `eager`。
3. 主机侧发射耗时（`model(...)` 或 `replay()` 返回到所有 kernel 入队完毕的时间）在
   N≥3 时应 ≈ GPU span（§22.3 的形态）；graph 臂应显著更低。

用法：
    .venv/bin/python scripts/probes/pilot_graph_concurrency.py
    .venv/bin/python scripts/probes/pilot_graph_concurrency.py --concurrencies 1,2,4 --duration 4

"""

from __future__ import annotations

import argparse
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
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

#: 与 saturating_4ms.jsonl 一致的尺寸构成（25/25/25/25）。
SIZES = (224, 336, 448, 672)
#: 每尺寸捕获几张图。必须 ≥ 最大并发数：同一张图的两个实例共享静态缓冲，
#: 并发回放会互相覆写输入并读同一个输出。
INSTANCES = 8


# ── GPU 侧采样（功耗 / 利用率）───────────────────────────────────────────────
# 复用 metrics.py 的判据：NVML 的利用率是"区间内有无 kernel"的单标量，
# 区分不了算力忙与带宽忙；功耗才是"GPU 到底在不在干活"的硬证据。
class GpuSampler:
    def __init__(self, interval_s: float = 0.2):
        import subprocess

        self.subprocess = subprocess
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = self.subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
                power, util = out.split(",")
                self.samples.append((float(power), float(util)))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    def summary(self) -> dict:
        if not self.samples:
            return {"power_mean_w": None, "util_mean_pct": None, "n_samples": 0}
        return {
            "power_mean_w": statistics.fmean(s[0] for s in self.samples),
            "util_mean_pct": statistics.fmean(s[1] for s in self.samples),
            "n_samples": len(self.samples),
        }


class _NullSampler:
    """不采样 GPU 的替身。用于排除 `nvidia-smi` 子进程对并发回放的干扰。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self) -> dict:
        return {"power_mean_w": None, "util_mean_pct": None, "n_samples": 0}


def make_input(job: EncodeJob, mean, std, torch_mod):
    """服务现状的图像预处理：CPU 上定种子造图 + 归一化。"""
    generator = torch_mod.Generator(device="cpu").manual_seed(job.seed)
    pixels = torch_mod.rand((3, job.height, job.width), generator=generator, dtype=torch_mod.float32)
    return ((pixels - mean) / std).unsqueeze(0)


class GraphPool:
    """按尺寸捕获 CUDA Graph，每尺寸 ``instances`` 份互不共享静态缓冲的副本。"""

    def __init__(self, model, torch_mod, dtype, mean, std, sizes=SIZES, instances=INSTANCES):
        self.torch = torch_mod
        self.mean, self.std = mean, std
        self.entries: dict[int, list[dict]] = {}
        self.capture_ms: dict[int, float] = {}
        for size in sizes:
            start = time.perf_counter()
            entries = [self._capture_one(model, dtype, size) for _ in range(instances)]
            self.capture_ms[size] = (time.perf_counter() - start) * 1000 / instances
            self.entries[size] = entries
            print(f"  已捕获 {size}×{size} × {instances} 份，"
                  f"单份 {self.capture_ms[size]:.0f} ms", flush=True)

    def _capture_one(self, model, dtype, size: int) -> dict:
        torch_mod = self.torch
        static_in = torch_mod.empty((1, 3, size, size), dtype=dtype, device="cuda")
        job = EncodeJob(f"cap-{size}", seed=zlib.crc32(f"cap-{size}".encode()) % (2**31),
                        width=size, height=size, deadline_ms=600_000)
        # 预生成一份真实分布输入，供捕获后做正确性对照。
        probe = make_input(job, self.mean, self.std, torch_mod).to("cuda", dtype=dtype)
        static_in.copy_(probe)

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
            raise SystemExit(f"{size}×{size} 的 CUDA Graph 输出与 eager 不符（最大绝对差 {diff:.3e}），"
                             "不能拿错图做实验")
        return {"graph": graph, "input": static_in, "output": captured, "diff": diff}


def run_arm(arm: str, concurrency: int, duration_s: float, pool: GraphPool | None,
            prebuilt: dict[int, torch.Tensor], model, mean, std, torch_mod, dtype) -> list[dict]:
    """N 个线程 × N 条流，各跑 ``duration_s``；返回每次迭代的记录。

    三个臂共用同一段计时骨架，**唯一的差别是"每次迭代要不要走 CPU 造图与 eager 发射路径"**：

    - `eager`            ：CPU 造图 → H2D → eager 前向（= 服务现状）
    - `graph`            ：CPU 造图 → 拷进静态缓冲 → 回放
    - `graph_prebuilt`   ：预生成输入拷贝 → 拷进静态缓冲 → 回放（连造图也省掉）

    ``host_launch_ms`` 是"主机的发射调用返回到所有 kernel 入队完毕"的时间：
    eager 下是 `model(...)` 的返回时刻，graph 下是 `graph.replay()` 的返回时刻。
    这正是 §22.3 用来定位机制的那个量。
    """
    records: list[dict] = []
    lock = threading.Lock()
    deadline_ns = time.perf_counter_ns() + int(duration_s * 1e9)

    def worker(worker_id: int) -> None:
        stream = torch_mod.cuda.Stream()
        local: list[dict] = []
        # staging 缓冲按 (线程, 尺寸) 预分配：迭代内不再向缓存分配器要显存。
        # 这一步是必需的，不是优化——迭代内 `cpu_pixels.to("cuda")` 会在
        # graph 回放已在飞行中时向默认池申请/释放，实测会把两个线程双双卡死在
        # `cudaEventSynchronize` 上（见 docs/实验记录.md §23 的复现记录）。
        staging = {size: torch_mod.empty((1, 3, size, size), dtype=dtype, device="cuda")
                   for size in SIZES}
        # 锁页主机缓冲，按 (线程, 尺寸) 预分配。分页源 H2D 与图回放共用会死锁（§23）。
        pinned = {size: torch_mod.empty((1, 3, size, size), dtype=torch_mod.float32).pin_memory()
                  for size in SIZES}
        index = worker_id
        while time.perf_counter_ns() < deadline_ns:
            size = SIZES[index % len(SIZES)]
            index += concurrency  # N 个线程轮转尺寸，合起来仍是均匀混合
            slot_in = None if arm == "eager" else pool.entries[size][worker_id]["input"]
            try:
                with torch_mod.cuda.stream(stream):
                    t0 = time.perf_counter_ns()
                    job = EncodeJob(f"{arm}-{worker_id}", seed=(worker_id * 7919 + size),
                                    width=size, height=size, deadline_ms=600_000)
                    # ① 主机侧图像准备：三条臂的**唯一**差别在"准备到哪种内存里"。
                    if arm == "graph_pinned_inplace":
                        # 直接在锁页缓冲里算：CPU 计算量与 `make_input` 同阶，省掉大块临时分配。
                        pin = pinned[size]
                        pin.uniform_(generator=torch_mod.Generator(device="cpu").manual_seed(job.seed))
                        pin.sub_(mean).div_(std)  # (3,1,1) 广播到 (1,3,H,W)
                    elif arm == "graph_prebuilt":
                        # 零 CPU 计算，但源仍要先落到锁页缓冲（否则 H2D 走分页路径 → §23 的死锁）
                        pin = pinned[size]
                        pin.copy_(prebuilt[size])
                    elif arm == "eager":
                        pin = None
                        cpu_pixels = make_input(job, mean, std, torch_mod)
                    else:  # graph_pinned_stage：完整保留 `make_input` 的分配与计算成本
                        pin = pinned[size]
                        pin.copy_(make_input(job, mean, std, torch_mod))
                    cpu_ready_ns = time.perf_counter_ns()
                    # ② H2D。**源必须是锁页内存**：分页源 H2D 在 CPU 有负载时会与图回放
                    #    互相等待，把所有线程卡死在流同步上（复现与二分见 §23）。
                    #    服务侧 `_input` 目前是分页源，接图回放时必须一并改成锁页。
                    if arm == "eager":
                        payload = cpu_pixels.to("cuda", dtype=dtype, non_blocking=True)
                    else:
                        staging[size].copy_(pin, non_blocking=True)
                        slot_in.copy_(staging[size])
                        payload = None
                    h2d_done_ns = time.perf_counter_ns()
                    start = torch_mod.cuda.Event(enable_timing=True)
                    end = torch_mod.cuda.Event(enable_timing=True)
                    if arm == "eager":
                        with torch_mod.inference_mode():
                            start.record(stream)
                            model(pixel_values=payload, interpolate_pos_encoding=True)
                            end.record(stream)
                    else:
                        start.record(stream)
                        pool.entries[size][worker_id]["graph"].replay()
                        end.record(stream)
                    enqueue_done_ns = time.perf_counter_ns()  # 主机侧发射到此结束
                    # 用**流**同步而不是 `end.synchronize()`：二者等待的是同一批工作，
                    # 但事件同步在并发回放下会挂起（同上）。事件此时必然已完成，
                    # `elapsed_time` 仍然有效。
                    stream.synchronize()
                    finish_ns = time.perf_counter_ns()
                local.append({
                    "size": size,
                    "cpu_input_ms": (cpu_ready_ns - t0) / 1e6,
                    "h2d_ms": (h2d_done_ns - cpu_ready_ns) / 1e6,
                    "host_launch_ms": (enqueue_done_ns - h2d_done_ns) / 1e6,
                    "gpu_span_ms": float(start.elapsed_time(end)),
                    "wall_ms": (finish_ns - t0) / 1e6,
                })
            except Exception as exc:  # noqa: BLE001 - 探针里直接报错更好排查
                local.append({"size": size, "error": f"{type(exc).__name__}: {exc}"})
                break
        with lock:
            records.extend(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return records


def describe(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]

    def pct(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    by_size: dict[int, list[float]] = {}
    for r in ok:
        by_size.setdefault(r["size"], []).append(r["gpu_span_ms"])
    return {
        "completed": len(ok),
        "errors": len(records) - len(ok),
        # 出错时务必把原文带出来：探针里最常见的失败是 graph 在流上回放时的
        # 捕获/分配器约束，只报一个计数会让人误以为是"吞吐真的为 0"。
        "first_error": next((r["error"] for r in records if "error" in r), None),
        "cpu_input_ms_median": statistics.median([r["cpu_input_ms"] for r in ok]) if ok else None,
        "h2d_ms_median": statistics.median([r["h2d_ms"] for r in ok]) if ok else None,
        "host_launch_ms_median": statistics.median([r["host_launch_ms"] for r in ok]) if ok else None,
        "gpu_span_ms_median": statistics.median([r["gpu_span_ms"] for r in ok]) if ok else None,
        "wall_ms_median": statistics.median([r["wall_ms"] for r in ok]) if ok else None,
        "wall_ms_p95": pct([r["wall_ms"] for r in ok], 0.95),
        "gpu_span_by_size": {str(k): statistics.median(v) for k, v in sorted(by_size.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrencies", default="1,2,3,4,6")
    parser.add_argument("--duration", type=float, default=6.0, help="每个臂的墙钟秒数")
    # 默认只跑不会挂死的两臂。`graph_pinned_*` 是**同线程**写法，N 大时会挂死，
    # 必须显式点名才跑（它们是"约束边界在哪"的证据，不是常规对照臂）。
    parser.add_argument("--arms", default="eager,graph_prebuilt")
    parser.add_argument("--out", default="results/probes/pilot_graph_concurrency.json")
    parser.add_argument("--gpu-sample", action="store_true",
                        help="开启 nvidia-smi 采样（功耗/利用率）。默认关闭：它的子进程会干扰图形回放。")
    args = parser.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    arms = args.arms.split(",")

    cfg = load_config("config.libsmctrl.example.yaml")
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), max(concurrencies))
    model = enc.model
    dtype = enc.dtype
    mean, std = enc.mean, enc.std

    print("捕获 CUDA Graph …", flush=True)
    pool = GraphPool(model, enc.torch, dtype, mean, std)
    print("正确性对照：所有图与 eager 输出一致\n", flush=True)

    # `graph_prebuilt` 臂用的预生成输入：尺寸固定，只造一次，迭代内不再付出造图成本。
    prebuilt = {
        size: make_input(
            EncodeJob(f"prebuilt-{size}", seed=zlib.crc32(f"pre-{size}".encode()) % (2**31),
                      width=size, height=size, deadline_ms=600_000),
            mean, std, enc.torch,
        )
        for size in SIZES
    }

    results: dict[str, dict] = {}
    for arm in arms:
        for n in concurrencies:
            time.sleep(0.5)
            sampler_ctx = GpuSampler() if args.gpu_sample else _NullSampler()
            with sampler_ctx as sampler:
                records = run_arm(arm, n, args.duration, pool, prebuilt,
                                  model, mean, std, enc.torch, dtype)
            stats = describe(records)
            stats["throughput_rps"] = stats["completed"] / args.duration
            stats["gpu"] = sampler.summary()
            results[f"{arm}@N={n}"] = stats
            print(f"{arm:>15} N={n}  吞吐 {stats['throughput_rps']:6.1f} req/s  "
                  f"完成 {stats['completed']:5d}  主机发射 {stats['host_launch_ms_median'] or 0:6.2f} ms  "
                  f"GPU span {stats['gpu_span_ms_median'] or 0:6.2f} ms  "
                  f"功耗 {stats['gpu']['power_mean_w'] or 0:5.1f} W  "
                  f"利用 {stats['gpu']['util_mean_pct'] or 0:4.1f}%", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "capture_ms_per_instance": pool.capture_ms},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
