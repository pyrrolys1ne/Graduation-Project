"""真实 SM/TPC 配额实验：单线程固定配额曲线 + 双线程共置隔离对照。

**机制**：走 libsmctrl 的回调路径 + 本项目补丁新增的粘性线程掩码
（``libsmctrl_set_thread_mask``）。掩码是 ``__thread`` 的，作用域由**调用线程**决定；
项目里 ``encode()`` 在同一线程内先调 ``apply_quota`` 再跑模型，所以线程掩码恰好
等于请求掩码。

**为什么不是双流**：``set_stream_mask`` 走硬编码结构体偏移，未知驱动版本会 ``exit(1)``
终止进程；回调路径读运行时 TMD 版本自适应，跨驱动版本可用。

用法::

    export LIBSMCTRL_PATH=/path/to/libsmctrl.so     # 由 scripts/build_libsmctrl.sh 构建（含补丁）
    python scripts/experiment_sm_quota.py --config config.libsmctrl.example.yaml --repeats 5

探针不通过时脚本会**直接拒绝运行**，不会用代理数据冒充隔离结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import queue
import statistics
import threading
import time
import zlib
from dataclasses import replace

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import UnsupportedResourceBackend, create_resource_backend


def _job(tag: str, width: int, height: int, quota: float) -> EncodeJob:
    # 不用内置 hash()：字符串哈希受 PYTHONHASHSEED 影响，会让实验不可复现。
    job = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31), width=width, height=height, deadline_ms=10_000)
    job.sm_fraction = quota
    return job


def _encode(encoder: ClipEncoderBackend, job: EncodeJob, worker_id: int) -> dict:
    result = encoder.encode(job, worker_id)
    resource = result.resource or {}
    return {
        "request_id": job.request_id,
        "width": job.width,
        "height": job.height,
        "patches": job.patches,
        "requested_sm_fraction": job.sm_fraction,
        "execution_ms": result.execution_ms,
        "enforced": resource.get("enforced"),
        "effective_sm_fraction": resource.get("effective_sm_fraction"),
        "tpc_start": resource.get("tpc_start"),
        "tpc_count": resource.get("tpc_count"),
        "total_tpcs": resource.get("total_tpcs"),
        "native_mask": resource.get("native_mask"),
        "thread_id": resource.get("thread_id"),
        # 记录本次实际占用的 TPC 集合，用于判定"独跑与并发是否落在同一组 TPC"。
        # 位置不同会让比值失去意义（不同 TPC 的 L2 slice 与访存通路并不等价）。
        "tpc_span": (
            list(range(resource["tpc_start"], resource["tpc_start"] + resource["tpc_count"]))
            if resource.get("tpc_start") is not None and resource.get("tpc_count")
            else []
        ),
    }


def quota_curve(worker: "_Worker", quotas, sizes, repeats: int) -> list[dict]:
    """Part A：单线程、固定配额、重复测量。"""
    rows: list[dict] = []
    for quota in quotas:
        for width, height in sizes:
            for repeat in range(repeats):
                job = _job(f"q{quota}-{width}x{height}-r{repeat}", width, height, quota)
                row = worker.encode(job)
                row["repeat"] = repeat
                rows.append(row)
                print(f"  [曲线] q={quota} {width}x{height} r{repeat}: {row['execution_ms']:.2f} ms, "
                      f"TPC={row['tpc_start']}+{row['tpc_count']} (eff={row['effective_sm_fraction']})", flush=True)
    return rows


class _Worker:
    """一个专用工作线程，持有自己的 TPC 分配。

    不能用 `ThreadPoolExecutor`：它会复用线程，两次提交可能落在同一个线程上，而
    分配表是按线程 id 建账的。也不能从主线程替别的线程释放分配——回调路径的掩码是
    ``__thread`` 的，``release_quota`` 忽略 handle、只释放**调用线程自己**的表项。
    这两点合起来意味着：必须让每个线程自己申请、自己释放。
    """

    def __init__(self, encoder: ClipEncoderBackend):
        self.encoder = encoder
        self._queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            fn, box = self._queue.get()
            if fn is None:
                self._queue.task_done()
                return
            try:
                box["result"] = fn()
            except Exception as exc:  # noqa: BLE001 - 回传给调用方
                box["error"] = exc
            finally:
                self._queue.task_done()

    def submit(self, fn) -> dict:
        box: dict = {}
        self._queue.put((fn, box))
        return box

    def wait(self, box: dict, timeout: float = 600.0) -> dict:
        """等待本线程处理完已提交的任务。

        用 `queue.join()` 会让死锁表现为静默挂起（已发生过一次：在工作线程内部调用
        会等自己的队列排空）。这里改成带超时的轮询，超时即报错而不是无限等待。
        """
        deadline = time.time() + timeout
        while not box:
            if time.time() > deadline:
                raise TimeoutError("等待工作线程超时（疑似死锁）")
            time.sleep(0.005)
        if "error" in box:
            raise box["error"]
        return box["result"]

    def call(self, fn):
        return self.wait(self.submit(fn))

    def release(self) -> None:
        """在本线程内释放自己的 TPC 分配。"""
        self.call(lambda: self.encoder.resource_backend.release_quota(0))

    def encode(self, job: EncodeJob) -> dict:
        return self.call(lambda: _encode(self.encoder, job, 0))


def throughput_comparison(workers: tuple[_Worker, _Worker], encoder: ClipEncoderBackend,
                          size: int, count: int, repeat: int) -> list[dict]:
    """Part B：**同样多的 TPC，串行用 vs 分区并发用，哪个更快？**

    为什么不比"单请求延迟的 slowdown"：那样做需要独跑与并发落在**同一组 TPC**上，而
    本设计下做不到——`encode()` 在 `finally` 里释放配额，所以串行跑第二个请求时分配表
    已空，它必然从 TPC 0 起始；并发时则是先到者占区间头、后到者落到后面。不同 TPC 的
    L2 slice 与访存通路并不等价，于是"位置变了"会被误读成"并发加速/拖慢"。
    （实测中同配额同尺寸的独跑曾相差 3 倍：4.37 ms 对 12.45 ms。）

    改用**完成同样多请求的总墙钟时间**：这个指标对 TPC 位置不敏感，而且它才是决策相关
    的问题——给定 12 个 TPC，是该让一个请求独占、还是分给两个请求并发？

      serial：1 线程 × 12 TPC，串行处理 count 个请求
      split ：2 线程 × 6 TPC，并发处理同样的 count 个请求

    两种配置都占用全部 12 个 TPC，因此比较的是"分区方式"而非"用了多少资源"。
    """
    worker_a, worker_b = workers
    rows: list[dict] = []
    total_tpcs = encoder.resource_backend.describe().get("total_tpcs") or 12

    # 单次测量：一次只放 1 个或 2 个请求在飞。
    #
    # **不能用"每个 worker 跑一批请求"的做法。** `encode()` 在 finally 里释放配额，
    # 批处理会让两个 worker 每处理完一个请求就交还分配，交错之下分配表经常只剩一个活跃
    # 项，后来者便从 TPC 0 重新开始——实测中两个线程拿到的是同一组 [0..5]，根本没有分区，
    # 而那轮得到的 1.26x 其实是"两个请求共用同一组 TPC"的效果，不是分区的效果。
    #
    # 只有让两个请求**同时在飞**，两份分配才会重叠持有，分配器才会给出互斥区间。
    half = 0.5

    def one_serial() -> tuple[float, dict]:
        worker_a.release()
        started = time.perf_counter()
        row = worker_a.encode(_job(f"serial-{repeat}-{time.time_ns()}", size, size, 1.0))
        return time.perf_counter() - started, row

    def one_pair() -> tuple[float, dict, dict]:
        worker_a.release()
        worker_b.release()
        # 直接调 _encode：worker.encode 会再次 submit 并 join 自己的队列，在工作线程
        # 内部调用会死锁（已踩过一次）。
        box_a = worker_a.submit(lambda: _encode(worker_a.encoder, _job(f"pair-a-{time.time_ns()}", size, size, half), 0))
        box_b = worker_b.submit(lambda: _encode(worker_b.encoder, _job(f"pair-b-{time.time_ns()}", size, size, half), 0))
        started = time.perf_counter()
        row_a = worker_a.wait(box_a)
        row_b = worker_b.wait(box_b)
        return time.perf_counter() - started, row_a, row_b

    serial_s = 0.0
    serial_exec: list[float] = []
    for _ in range(count):
        elapsed, row = one_serial()
        serial_s += elapsed
        serial_exec.append(row["execution_ms"])

    pair_s = 0.0
    pair_exec: list[float] = []
    disjoint_flags: list[bool] = []
    seen_spans: list = []
    for _ in range(count):
        elapsed, row_a, row_b = one_pair()
        pair_s += elapsed
        pair_exec += [row_a["execution_ms"], row_b["execution_ms"]]
        span_a, span_b = set(row_a["tpc_span"]), set(row_b["tpc_span"])
        disjoint_flags.append(bool(span_a) and bool(span_b) and span_a.isdisjoint(span_b))
        if len(seen_spans) < 3:
            seen_spans.append((sorted(span_a), sorted(span_b)))

    rows.append({
        "mode": "serial", "repeat": repeat, "count": count,
        "threads": 1, "quota_each": 1.0, "total_tpcs": total_tpcs,
        # 串行一次 1 个请求；配对一次 2 个 —— 吞吐统一按"完成的请求数 / 总时间"算
        "wall_s": serial_s, "throughput_rps": count / serial_s,
        "mean_exec_ms": statistics.fmean(serial_exec),
    })
    rows.append({
        "mode": "split", "repeat": repeat, "count": count,
        "threads": 2, "quota_each": half, "total_tpcs": total_tpcs,
        "wall_s": pair_s, "throughput_rps": (2 * count) / pair_s,
        "mean_exec_ms": statistics.fmean(pair_exec),
        "tpc_disjoint": all(disjoint_flags),
        "spans": seen_spans,
    })
    serial_rps, split_rps = count / serial_s, (2 * count) / pair_s
    print(f"  [总时间] r{repeat}: 串行 {serial_rps:6.1f} req/s vs 各半并发 {split_rps:6.1f} req/s"
          f" -> {(split_rps/serial_rps):.2f}x, TPC 互斥={all(disjoint_flags)} {seen_spans[:1]}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 SM/TPC 配额实验（线程掩码）")
    parser.add_argument("--config", type=Path, default=Path("config.libsmctrl.example.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sm_quota"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=20,
                        help="Part B 中串行与分区并发各自处理的请求数")
    parser.add_argument("--size", type=int, default=336, help="共置对照使用的正方形边长")
    parser.add_argument("--sizes", default="224x224,336x336,672x672", help="配额曲线用的 WxH 列表")
    args = parser.parse_args()

    config = load_config(args.config)
    # 正式实验禁止静默回退：一旦回退，产出的就不是 SM 隔离实验。
    config = replace(config, executor=replace(config.executor, allow_proxy_fallback=False))
    if config.executor.resource_backend != "libsmctrl":
        raise SystemExit(
            f"配置的 resource_backend={config.executor.resource_backend!r}，本脚本要求 libsmctrl。"
            "请使用 config.libsmctrl.example.yaml。"
        )
    try:
        backend = create_resource_backend("libsmctrl", config.executor.libsmctrl_adapter, allow_proxy_fallback=False)
    except UnsupportedResourceBackend as exc:
        raise SystemExit(
            f"\n本环境无法真实施加 TPC 掩码，拒绝继续运行（不会产出代理数据代替）。\n原因: {exc}\n\n"
            "排查顺序:\n"
            "  1) python scripts/probe_libsmctrl.py       # 探针详情\n"
            "  2) echo $LIBSMCTRL_PATH                    # 是否指向【打了补丁】的库\n"
            "     缺 libsmctrl_set_thread_mask 说明用的是未打补丁的上游库，\n"
            "     请用 scripts/build_libsmctrl.sh 重建（会自动应用 patches/ 下的补丁）。\n"
            "  3) python demos/libsmctrl_thread_mask_demo.py   # 端到端验证掩码确实生效\n"
        ) from exc
    if not backend.enforces_sm_partition:
        raise SystemExit(
            "能力探针未通过，本环境无法真实施加 TPC 掩码，拒绝继续。\n"
            f"探针结果: {json.dumps(backend.describe().get('probe'), ensure_ascii=False)}"
        )
    print(f"资源后端就绪: {json.dumps(backend.describe(), ensure_ascii=False)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoderBackend(config.model, backend, stream_count=2)
    sizes = [tuple(int(v) for v in item.split("x")) for item in args.sizes.split(",") if item]
    quotas = list(config.scheduler.quota_levels)

    # 用两个**专用线程**（不是线程池）承载并发实验：分配表按线程 id 建账，
    # 线程池会复用线程，导致两次提交落在同一线程上而互相覆盖分配。
    worker_a, worker_b = _Worker(encoder), _Worker(encoder)
    try:
        # 暖机：首个 encode 含 CUDA 上下文与 cuBLAS 句柄的首次初始化，实测可达数百毫秒，
        # 不暖机会让曲线的最低配额档位出现假的高延迟。
        print(f"=== 暖机 {args.warmup} 次 ===", flush=True)
        for index in range(args.warmup):
            worker_a.encode(_job(f"warmup-{index}", args.size, args.size, 1.0))

        print("=== Part A：单线程固定配额曲线 ===", flush=True)
        curve = quota_curve(worker_a, quotas, sizes, args.repeats)
        with (args.output_dir / "quota_curve.jsonl").open("w", encoding="utf-8") as handle:
            for row in curve:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"=== Part B：串行 vs 分区并发（各 {args.pair_count} 请求 × {args.repeats} 次）===", flush=True)
        coloc: list[dict] = []
        for repeat in range(args.repeats):
            coloc.extend(throughput_comparison((worker_a, worker_b), encoder, args.size, args.pair_count, repeat))
        with (args.output_dir / "serial_vs_split.jsonl").open("w", encoding="utf-8") as handle:
            for row in coloc:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        # 让每个线程释放自己的分配（掩码是 __thread 的，只能由本线程清除）
        worker_a.release()
        worker_b.release()

    curve_summary: dict[str, dict] = {}
    for quota in quotas:
        for width, height in sizes:
            values = [r["execution_ms"] for r in curve
                      if r["requested_sm_fraction"] == quota and r["width"] == width and r["height"] == height]
            if values:
                curve_summary[f"q{quota}-{width}x{height}"] = {
                    "mean_ms": statistics.fmean(values),
                    "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "samples": len(values),
                }
    serial = [r for r in coloc if r["mode"] == "serial"]
    split = [r for r in coloc if r["mode"] == "split"]
    serial_tp = statistics.fmean(r["throughput_rps"] for r in serial)
    split_tp = statistics.fmean(r["throughput_rps"] for r in split)
    summary = {
        "config": str(args.config),
        "repeats": args.repeats,
        "pair_count": args.pair_count,
        "mask_path": (backend.describe().get("probe") or {}).get("mask_path"),
        "resource": backend.describe(),
        "quota_curve": curve_summary,
        "serial_vs_split": {
            "serial_throughput_rps": serial_tp,
            "split_throughput_rps": split_tp,
            "speedup_of_split": split_tp / serial_tp if serial_tp else None,
            "all_tpc_disjoint": all(r.get("tpc_disjoint", False) for r in split),
            "serial_mean_exec_ms": statistics.fmean(r["mean_exec_ms"] for r in serial),
            "split_mean_exec_ms": statistics.fmean(r["mean_exec_ms"] for r in split),
        },
        "interpretation": (
            "serial 与 split 占用同样多的 TPC（全卡），差别只在'一个请求独占'还是'两个请求各半'。"
            "speedup_of_split > 1 说明分区并发更快，< 1 说明串行更快。"
            "该指标对 TPC 位置不敏感，因此不依赖'独跑与并发落在同一组 TPC'这一难以满足的条件。"
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
