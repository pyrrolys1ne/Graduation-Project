from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

from .models import EncodeJob
from .performance import PerformanceModel


@dataclass(frozen=True)
class DaccConfig:
    window: int = 8
    beta: float = 1.0
    alpha: float = 0.2
    urgency_tau_ms: float = 20.0
    starvation_ms: float = 200.0
    age_cap: float = 2.0
    guard_ms: float = 5.0
    max_pair_conflict: float = 0.35
    w_urgency: float = 3.0
    w_priority: float = 0.15
    w_age: float = 0.5
    w_gain: float = 1.0
    w_complementarity: float = 1.0
    w_risk: float = 5.0
    w_uncertainty: float = 0.5


@dataclass(frozen=True)
class DaccPlan:
    jobs: tuple[EncodeJob, ...]
    quotas: tuple[float, ...]
    mode: str
    score: float
    complementarity: float
    risk: float


class DaccPlanner:
    """Small, deterministic planner shared by the online scheduler and simulator."""

    def __init__(self, performance_model: PerformanceModel, quota_levels: Sequence[float], config: DaccConfig | None = None):
        self.performance_model = performance_model
        self.quota_levels = tuple(sorted(float(q) for q in quota_levels))
        self.config = config or DaccConfig()
        self._correction = 1.0
        self._residual_ms = 0.0

    def resource_demand(self, job: EncodeJob) -> dict[str, float]:
        scale = min(1.0, max(0.0, job.patches / 441.0))
        return {"sm": 0.25 + 0.75 * scale, "mem": 0.10 + 0.75 * scale, "l2": 0.15 + 0.70 * scale, "dram": 0.10 + 0.80 * scale}

    def uncertainty(self, job: EncodeJob, quota: float) -> float:
        predicted = self.performance_model.predict(job.patches, quota)
        return max(self._residual_ms, predicted * 0.10)

    def safe_latency(self, job: EncodeJob, quota: float) -> float:
        predicted = self.performance_model.predict(job.patches, quota)
        return self._correction * predicted + self.config.beta * self.uncertainty(job, quota)

    def update(self, predicted_ms: float, actual_ms: float) -> None:
        if predicted_ms <= 0 or actual_ms <= 0:
            return
        ratio = actual_ms / predicted_ms
        self._correction = self.config.alpha * ratio + (1.0 - self.config.alpha) * self._correction
        residual = abs(actual_ms - self._correction * predicted_ms)
        self._residual_ms = self.config.alpha * residual + (1.0 - self.config.alpha) * self._residual_ms

    def _urgency(self, job: EncodeJob, now_ns: int) -> float:
        slack = job.absolute_deadline_ns / 1_000_000 - now_ns / 1_000_000 - self.safe_latency(job, 1.0)
        return math.exp(-max(slack, 0.0) / self.config.urgency_tau_ms)

    def _age_bonus(self, job: EncodeJob, now_ns: int) -> float:
        age_ms = max(0.0, (now_ns - job.arrival_ns) / 1_000_000)
        return min(age_ms / self.config.starvation_ms, self.config.age_cap)

    def conflict(self, left: EncodeJob, right: EncodeJob, q_left: float, q_right: float) -> float:
        a, b = self.resource_demand(left), self.resource_demand(right)
        sm = max(0.0, q_left * a["sm"] + q_right * b["sm"] - 1.0)
        overlap = sum(a[key] * b[key] for key in ("l2", "dram")) / 2.0
        memory = max(0.0, a["mem"] + b["mem"] - 1.0)
        return min(1.0, 0.45 * sm + 0.25 * memory + 0.30 * overlap)

    def _single(self, job: EncodeJob, quota: float, now_ns: int) -> DaccPlan:
        safe = self.safe_latency(job, quota)
        finish_ms = now_ns / 1_000_000 + safe
        deadline_ms = job.absolute_deadline_ns / 1_000_000
        risk = max(0.0, (finish_ms - deadline_ms) / max(job.deadline_ms, 1e-6))
        score = (
            self.config.w_urgency * self._urgency(job, now_ns)
            + self.config.w_priority * job.priority
            + self.config.w_age * self._age_bonus(job, now_ns)
            + self.config.w_gain / max(safe, 1e-6)
            - self.config.w_risk * risk
            - self.config.w_uncertainty * self.uncertainty(job, quota) / max(safe, 1e-6)
        )
        return DaccPlan((job,), (quota,), "single", score, 1.0, risk)

    def plan(self, jobs: Sequence[EncodeJob], now_ns: int, active_fraction: float = 1.0) -> DaccPlan:
        candidates = list(jobs[: self.config.window])
        if not candidates:
            raise ValueError("DACC 需要至少一个候选请求")
        plans: list[DaccPlan] = [self._single(job, q, now_ns) for job in candidates for q in self.quota_levels if q <= active_fraction]
        for left, right in itertools.combinations(candidates, 2):
            for q_left in self.quota_levels:
                for q_right in self.quota_levels:
                    if q_left + q_right > active_fraction:
                        continue
                    conflict = self.conflict(left, right, q_left, q_right)
                    if conflict > self.config.max_pair_conflict:
                        continue
                    safe_left, safe_right = self.safe_latency(left, q_left), self.safe_latency(right, q_right)
                    duration = max(safe_left, safe_right) * (1.0 + conflict)
                    risk = sum(max(0.0, (now_ns / 1_000_000 + duration - j.absolute_deadline_ns / 1_000_000) / max(j.deadline_ms, 1e-6)) for j in (left, right))
                    urgency = self._urgency(left, now_ns) + self._urgency(right, now_ns)
                    age = self._age_bonus(left, now_ns) + self._age_bonus(right, now_ns)
                    gain = 2.0 / max(duration, 1e-6)
                    score = self.config.w_urgency * urgency + self.config.w_age * age + self.config.w_priority * (left.priority + right.priority) + self.config.w_gain * gain + self.config.w_complementarity * (1.0 - conflict) - self.config.w_risk * risk
                    plans.append(DaccPlan((left, right), (q_left, q_right), "pair", score, 1.0 - conflict, risk))
        guard = [plan for plan in plans if any(self._urgency(job, now_ns) >= 0.99 for job in plan.jobs)]
        pool = guard or plans
        return max(pool, key=lambda plan: (plan.score, -len(plan.jobs), tuple(j.request_id for j in plan.jobs)))
