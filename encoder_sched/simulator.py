from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dacc import DaccConfig, DaccPlanner
from .models import EncodeJob
from .performance import PerformanceModel


@dataclass(frozen=True)
class SimulationSummary:
    policy: str
    completed: int
    throughput_rps: float
    mean_latency_ms: float
    p99_latency_ms: float
    slo_violation_rate: float
    makespan_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def load_workload(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def simulate(rows: Iterable[dict], policy: str, table_path: str | Path, streams: int = 2, seed: int = 20260902) -> SimulationSummary:
    model = PerformanceModel(table_path)
    planner = DaccPlanner(model, (0.25, 0.5, 0.75, 1.0), DaccConfig())
    rows = list(rows)
    jobs = []
    for i, row in enumerate(rows):
        arrival_ms = float(row.get("arrival_ms", 0.0))
        job = EncodeJob(row["request_id"], int(row.get("seed", seed + i)), int(row["width"]), int(row["height"]), float(row["deadline_ms"]), int(row.get("priority", 0)), arrival_ns=int(arrival_ms * 1_000_000))
        job.metadata["arrival_ms"] = arrival_ms
        jobs.append(job)
    pending = list(jobs)
    ready: list[EncodeJob] = []
    active: list[tuple[float, tuple[EncodeJob, ...]]] = []
    completed: list[tuple[EncodeJob, float]] = []
    now_ms = 0.0
    capacity = 1 if policy == "serial_fcfs" else max(1, streams)

    def add_arrivals() -> None:
        while pending and float(pending[0].metadata["arrival_ms"]) <= now_ms:
            ready.append(pending.pop(0))

    def order_ready() -> None:
        if policy in {"serial_fcfs", "multistream_fcfs"}:
            ready.sort(key=lambda j: (j.metadata["arrival_ms"], j.request_id))
        elif policy == "edf":
            ready.sort(key=lambda j: (j.absolute_deadline_ns, -j.priority, j.metadata["arrival_ms"]))
        else:
            ready.sort(key=lambda j: (j.absolute_deadline_ns // 5_000_000, -j.priority, model.predict(j.patches, 1.0), j.metadata["arrival_ms"]))

    while pending or ready or active:
        add_arrivals()
        active_slots = sum(len(batch) for _, batch in active)
        free = capacity - active_slots
        if ready and free > 0:
            if policy == "dacc":
                plan = planner.plan(ready, int(now_ms * 1_000_000), active_fraction=1.0)
                selected = tuple(plan.jobs)
                if len(selected) > free or any(job not in ready for job in selected):
                    selected = (min(ready, key=lambda j: j.metadata["arrival_ms"]),)
                for job in selected:
                    ready.remove(job)
                duration = max(planner.safe_latency(job, plan.quotas[index]) for index, job in enumerate(selected))
            else:
                order_ready()
                selected = (ready.pop(0),)
                duration = model.predict(selected[0].patches, 1.0)
            finish = now_ms + duration
            for job in selected:
                job.started_ns = int(now_ms * 1_000_000)
                job.finished_ns = int(finish * 1_000_000)
                job.execution_ms = duration
                completed.append((job, finish))
            active.append((finish, selected))
            continue
        next_arrival = float(pending[0].metadata["arrival_ms"]) if pending else float("inf")
        next_finish = min((finish for finish, _ in active), default=float("inf"))
        now_ms = min(next_arrival, next_finish)
        if next_finish <= next_arrival:
            active = [(finish, batch) for finish, batch in active if finish > now_ms + 1e-9]
    latencies = [end - job.arrival_ns / 1_000_000 for job, end in completed]
    violations = [lat > job.deadline_ms for (job, _), lat in zip(completed, latencies)]
    makespan = max((end for _, end in completed), default=0.0)
    return SimulationSummary(policy, len(completed), len(completed) / max(makespan / 1000.0, 1e-9), sum(latencies) / max(1, len(latencies)), float(sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)]) if latencies else 0.0, sum(violations) / max(1, len(violations)), makespan)
