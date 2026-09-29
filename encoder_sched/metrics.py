from __future__ import annotations

import csv
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .models import EncodeJob, JobState


#: 客户端排队占到端到端延迟的这个比例时，认为"这段尾延迟不是服务端造成的"。
#: 本课题已有结论混用两侧口径（服务端与客户端相差 15–34 倍），因此客户端口径
#: 必须自带一个可判定的旗标，而不是只留几列数字等读者自己发现。
CLIENT_OVERHEAD_SHARE_WARN = 0.10

#: 事件循环自身积压的告警线（ms）。1 ms 量级的迟到是 asyncio 睡眠精度与调度抖动，
#: 本机实测在 Fake 编码器（客户端绝无可能排队）上就有约 1.4 ms 的 p95 迟到。
#: 只有量级明显超出抖动的迟到才说明客户端根本没按时发出请求。
EVENT_LOOP_BACKLOG_MS = 50.0

#: 连接池等待的绝对地板（ms）。客户端与服务端在同一台机器上，``perf_counter`` 同源，
#: 但 HTTP 往返本身有毫秒级抖动：本机 Fake 编码器实测 overhead 在 3.7–6.1 ms 之间波动
#: （纯往返，客户端不可能排队）。低于这个地板的"多等"与抖动无法区分，一律不计入排队，
#: 否则服务端做得越快（total 越小）越容易被误判成客户端排队。
POOL_WAIT_FLOOR_MS = 5.0


def _stats(values: list[float]) -> dict[str, float | None]:
    """与 MetricsStore._stats 同口径分位数；无样本时返回 None 而不是 0。"""
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def client_side_summary(
    rows: list[dict[str, Any]],
    concurrency: int | None = None,
    pool_can_throttle: bool | None = None,
) -> dict[str, Any]:
    """客户端观测口径的汇总，带 ``scope`` 标注。

    为什么必须单独成口径：本课题的 ``total_ms`` 是**服务端完成时间**，从请求到达
    服务端开始计时，不含网络与客户端连接池等待；``client_latency_ms`` 是客户端
    从发出前开始计时的**端到端延迟**。两者相差 15–34 倍，混用会让"尾延迟由谁造成"
    的结论整体翻转（本项目已因此撤回过多流保护尾延迟的结论）。

    各量各自的诊断职责：

    - ``latency_ms``：客户端看到的端到端延迟，SLO 应按它判定；
    - ``overhead_ms``：``client_latency_ms - total_ms``，即服务端之外的部分
      （HTTP 往返、序列化、**连接池等待**）。它包含不可避免的往返开销，因此**不能**
      直接当作"排队"：基准取本轮 overhead 的**最小值**——总有一个请求没等过连接，
      它测到的就是纯往返；``pool_wait_ms`` 是超出该基准的部分，即连接池等待的估计。
    - ``dispatch_lateness_ms``：客户端**实际发出**请求的时刻减去负载规定的到达时刻。
      注意它在 ``client.post`` **之前**取样，所以量的是事件循环自身的积压，
      **不含**连接池等待（那发生在 post 内部、计入 overhead）。

    因此判据分两条：``client_queueing_flagged`` 看 ``pool_wait_ms`` 占客户端延迟的
    比例（超过 ``CLIENT_OVERHEAD_SHARE_WARN`` 即认为连接池人为制造了排队）；
    ``event_loop_backlog_flagged`` 看 ``dispatch_lateness_ms`` 是否出现毫秒级抖动
    解释不了的绝对迟到。

    ``pool_can_throttle`` 决定 ``pool_wait_ms`` 该不该被当成"连接池排队"。连接上限
    **不小于**请求数时（开环），客户端不可能因为等空闲连接而推迟下发，此时
    ``overhead_ms`` 里超出的部分来自连接建立、服务端接收与请求解析，是**前端开销**
    而不是客户端排队——报告里仍然给出数值，但不再据此判定本批无效。缺省按
    ``concurrency < 请求数`` 推断。

    返回的 ``client_queueing_flagged`` 为真时，本次运行的端到端延迟里有可观比例
    与服务端无关，此时用客户端口径讨论"服务端尾延迟"是不成立的。
    """
    records = [
        {
            "latency": float(row["client_latency_ms"]),
            "total": row.get("total_ms"),
            "lateness": row.get("dispatch_lateness_ms"),
            "deadline": row.get("deadline_ms"),
        }
        for row in rows
        if row.get("client_latency_ms") is not None
    ]
    latency = [record["latency"] for record in records]
    lateness = [float(record["lateness"]) for record in records if record["lateness"] is not None]
    overhead = [
        record["latency"] - float(record["total"])
        for record in records
        if record["total"] is not None
    ]
    # 无排队基准 = 本轮最小的 overhead。用它而不是 0，是为了不把纯 HTTP 往返误判成排队：
    # 在 Fake 编码器（服务端 total 不足 1 ms）上，往返本身就有数毫秒，直接用 overhead
    # 会立刻误报。
    floor = min(overhead) if overhead else None
    pool_wait = [max(0.0, value - floor) for value in overhead] if floor is not None else []
    # 客户端口径的 SLO 违约：负载里的 deadline_ms 是服务级目标，客户端看到的
    # 延迟才是用户实际承受的那个数；服务端 slo_violated 用的是 total_ms。
    slo = [
        record["latency"] > float(record["deadline"])
        for record in records
        if record["deadline"] is not None
    ]
    latency_stats = _stats(latency)
    overhead_stats = _stats(overhead)
    pool_stats = _stats(pool_wait)
    lateness_stats = _stats(lateness)
    if pool_can_throttle is None:
        pool_can_throttle = concurrency is None or concurrency < len(records)
    share = None
    excess = None
    if pool_stats["p95"] is not None:
        excess = max(0.0, pool_stats["p95"] - POOL_WAIT_FLOOR_MS)
    if latency_stats["p95"] and excess is not None:
        share = max(0.0, excess / latency_stats["p95"])
    backlog = bool(lateness_stats["p95"] is not None and lateness_stats["p95"] > EVENT_LOOP_BACKLOG_MS)
    pool_flagged = bool(
        pool_can_throttle and share is not None and share > CLIENT_OVERHEAD_SHARE_WARN
    )
    flagged = bool(pool_flagged or backlog)
    return {
        "scope": "client",
        "concurrency": concurrency,
        "samples": len(latency),
        "pool_binding": "capped_by_client_pool" if pool_can_throttle else "not_a_constraint",
        "latency_ms": latency_stats,
        "overhead_ms": overhead_stats,
        "pool_wait_ms": pool_stats,
        "dispatch_lateness_ms": lateness_stats,
        "pool_wait_excess_p95_ms": excess,
        "queueing_share_p95": share,
        # overhead 为负说明两侧时间基准不可比（例如服务端把到达时刻记早了），
        # 这是一个信号而不是噪声，必须能被看到。
        "negative_overhead_samples": sum(1 for value in overhead if value < 0),
        "pool_wait_flagged": pool_flagged,
        "event_loop_backlog_flagged": backlog,
        "client_queueing_flagged": flagged,
        "slo_violation_rate": (sum(slo) / len(slo)) if slo else None,
    }


class MetricsStore:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: list[dict[str, Any]] = []
        self._gpu_samples: list[float] = []
        self._submitted = 0
        self._first_arrival_ns: int | None = None
        self._last_finished_ns: int | None = None

    def record_submission(self) -> None:
        with self._lock:
            self._submitted += 1

    def record(self, job: EncodeJob) -> None:
        row = job.public_dict()
        with self._lock:
            self._jobs.append(row)
            if self._first_arrival_ns is None or job.arrival_ns < self._first_arrival_ns:
                self._first_arrival_ns = job.arrival_ns
            if job.finished_ns is not None:
                self._last_finished_ns = max(self._last_finished_ns or 0, job.finished_ns)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_gpu_utilization(self, value: float) -> None:
        with self._lock:
            self._gpu_samples.append(value)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "p50": None, "p95": None, "p99": None}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs)
            gpu_samples = list(self._gpu_samples)
            first_arrival_ns = self._first_arrival_ns
            last_finished_ns = self._last_finished_ns
            submitted = self._submitted
        completed = [row for row in jobs if row["status"] == JobState.COMPLETED.value]
        elapsed_s = (
            max((last_finished_ns - first_arrival_ns) / 1_000_000_000, 1e-9)
            if first_arrival_ns is not None and last_finished_ns is not None
            else 0.0
        )
        return {
            # 口径声明：本 dict 里的 queue/execution/total 全部是**服务端**观测，
            # 时间从请求到达服务端算起，不含网络与客户端连接池等待。客户端口径
            # （client_latency_ms / 连接池排队）见 client_side_summary()。
            "latency_scope": "server",
            "submitted": submitted,
            "completed": len(completed),
            "failed": sum(row["status"] == JobState.FAILED.value for row in jobs),
            "throughput_rps": len(completed) / elapsed_s if elapsed_s else 0.0,
            "queue_latency_ms": self._stats([row["queue_ms"] for row in completed]),
            "execution_latency_ms": self._stats([row["execution_ms"] for row in completed]),
            "total_latency_ms": self._stats([row["total_ms"] for row in completed]),
            "slo_violation_rate": (
                sum(bool(row["slo_violated"]) for row in completed) / len(completed) if completed else None
            ),
            "gpu_utilization_percent": statistics.fmean(gpu_samples) if gpu_samples else None,
            "gpu_samples": len(gpu_samples),
        }

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._gpu_samples.clear()
            self._first_arrival_ns = None
            self._last_finished_ns = None
            self._submitted = 0

    def export_csv(self, path: Path) -> None:
        with self._lock:
            rows = list(self._jobs)
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [key for key in rows[0] if key != "metadata"]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


class GpuUtilizationSampler:
    #: 连续失败多少次后放弃采样。此前实现遇到第一次异常就直接 `return`，
    #: 一次 nvidia-smi 抖动就会永久停止采样，且失败是静默的。
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, metrics: MetricsStore, interval_s: float = 0.1):
        self.metrics = metrics
        self.interval_s = interval_s
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-utilization-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                first = completed.stdout.strip().splitlines()[0]
                self.metrics.record_gpu_utilization(float(first))
                self.failures = 0
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                self.failures += 1
                if self.failures >= self.MAX_CONSECUTIVE_FAILURES:
                    return
