from __future__ import annotations

import time
from collections.abc import Iterable

from .models import EncodeJob
from .performance import PerformanceModel
from .dacc import DaccConfig, DaccPlanner


SUPPORTED_POLICIES = {"serial_fcfs", "multistream_fcfs", "edf", "edf_size", "dacc"}


class Scheduler:
    def __init__(
        self,
        policy: str,
        performance_model: PerformanceModel,
        quota_levels: Iterable[float],
        deadline_tie_ms: float = 5.0,
        dacc_config: DaccConfig | None = None,
    ):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"未知调度策略: {policy}")
        self.policy = policy
        self.performance_model = performance_model
        self.quota_levels = tuple(sorted(float(level) for level in quota_levels))
        self.deadline_tie_ns = int(deadline_tie_ms * 1_000_000)
        self.dacc = DaccPlanner(performance_model, self.quota_levels, dacc_config) if policy == "dacc" else None

    def rank_key(self, job: EncodeJob) -> tuple[float, ...]:
        if self.policy in {"serial_fcfs", "multistream_fcfs"}:
            return (float(job.arrival_ns),)
        if self.policy == "edf":
            return (float(job.absolute_deadline_ns), float(-job.priority), float(job.arrival_ns))
        deadline_bucket = job.absolute_deadline_ns // max(1, self.deadline_tie_ns)
        return (
            float(deadline_bucket),
            float(-job.priority),
            float(job.predicted_ms),
            float(job.arrival_ns),
        )

    def choose_quota(self, job: EncodeJob, now_ns: int | None = None) -> float:
        if self.policy not in {"edf_size", "dacc"}:
            return 1.0
        current = time.perf_counter_ns() if now_ns is None else now_ns
        remaining_ms = max(0.0, (job.absolute_deadline_ns - current) / 1_000_000)
        for quota in self.quota_levels:
            prediction = self.dacc.safe_latency(job, quota) if self.policy == "dacc" else self.performance_model.predict(job.patches, quota)
            if prediction <= remaining_ms:
                return quota
        return self.quota_levels[-1]

    def prepare(self, job: EncodeJob, now_ns: int | None = None) -> EncodeJob:
        job.sm_fraction = self.choose_quota(job, now_ns)
        job.predicted_ms = self.performance_model.predict(job.patches, job.sm_fraction)
        if self.policy == "dacc":
            job.metadata.update({"prediction_mode": "prior", "resource_demand": self.dacc.resource_demand(job), "schedule_mode": "single"})
        return job
        if self.policy == "dacc":
            slack = job.slack_ms(job.arrival_ns)
            return (float(-self.dacc._urgency(job, job.arrival_ns)), float(-job.priority), float(job.predicted_ms), float(job.arrival_ns))
