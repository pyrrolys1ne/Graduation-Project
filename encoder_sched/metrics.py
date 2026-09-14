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
