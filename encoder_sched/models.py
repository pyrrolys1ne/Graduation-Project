from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EncodeJob:
    request_id: str
    seed: int
    width: int
    height: int
    deadline_ms: float
    priority: int = 0
    arrival_ns: int = field(default_factory=time.perf_counter_ns)
    state: JobState = JobState.QUEUED
    started_ns: int | None = None
    finished_ns: int | None = None
    predicted_ms: float = 0.0
    sm_fraction: float = 1.0
    embedding_dim: int | None = None
    embedding_norm: float | None = None
    execution_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def patches(self) -> int:
        # CLIP ViT-B/32 的 patch 是 32×32；data/profiles/*.csv 也按此计算
        # （224×224 记作 49 = (224/32)²）。改成 16 会让性能模型查到完全错误的桶。
        return (self.width // 32) * (self.height // 32)

    @property
    def absolute_deadline_ns(self) -> int:
        return self.arrival_ns + int(self.deadline_ms * 1_000_000)

    def slack_ms(self, now_ns: int | None = None) -> float:
        current = time.perf_counter_ns() if now_ns is None else now_ns
        return (self.absolute_deadline_ns - current) / 1_000_000 - self.predicted_ms

    @property
    def queue_ms(self) -> float | None:
        if self.started_ns is None:
            return None
        return (self.started_ns - self.arrival_ns) / 1_000_000

    @property
    def total_ms(self) -> float | None:
        if self.finished_ns is None:
            return None
        return (self.finished_ns - self.arrival_ns) / 1_000_000

    @property
    def slo_violated(self) -> bool | None:
        total = self.total_ms
        return None if total is None else total > self.deadline_ms

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.state.value,
            "width": self.width,
            "height": self.height,
            "patches": self.patches,
            "deadline_ms": self.deadline_ms,
            "priority": self.priority,
            "predicted_ms": self.predicted_ms,
            "sm_fraction": self.sm_fraction,
            "queue_ms": self.queue_ms,
            "execution_ms": self.execution_ms,
            "total_ms": self.total_ms,
            "embedding_dim": self.embedding_dim,
            "embedding_norm": self.embedding_norm,
            "slo_violated": self.slo_violated,
            "error": self.error,
            "metadata": self.metadata,
        }
