from __future__ import annotations

import asyncio
import itertools
import threading
import time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .encoder import EncoderBackend
from .metrics import GpuUtilizationSampler, MetricsStore
from .models import EncodeJob, JobState
from .resource import ResourceBackend
from .scheduler import Scheduler


class DuplicateRequestError(ValueError):
    pass


class JobNotFoundError(KeyError):
    pass


class EncoderService:
    def __init__(
        self,
        config: AppConfig,
        scheduler: Scheduler,
        encoder: EncoderBackend,
        metrics: MetricsStore,
        resource_backend: ResourceBackend | None = None,
    ):
        self.config = config
        self.scheduler = scheduler
        self.encoder = encoder
        self.metrics = metrics
        self.resource_backend = resource_backend
        self.queue: asyncio.PriorityQueue[tuple[tuple[float, ...], int, EncodeJob]] = asyncio.PriorityQueue()
        self.jobs: dict[str, EncodeJob] = {}
        self.futures: dict[str, asyncio.Future[EncodeJob]] = {}
        self._jobs_lock = threading.Lock()
        self._counter = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self.gpu_sampler = GpuUtilizationSampler(metrics)

    async def start(self) -> None:
        worker_count = 1 if self.scheduler.policy in {"serial_fcfs", "dacc"} else self.config.executor.streams
        self._workers = [asyncio.create_task(self._worker(index), name=f"encoder-worker-{index}") for index in range(worker_count)]
        self.gpu_sampler.start()

    async def stop(self) -> None:
        await self.queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.gpu_sampler.stop()
        output_dir = self.config.resolve(self.config.logging.output_dir)
        self.metrics.export_csv(output_dir / "requests.csv")

    async def submit(self, job: EncodeJob) -> EncodeJob:
        loop = asyncio.get_running_loop()
        with self._jobs_lock:
            if job.request_id in self.jobs:
                raise DuplicateRequestError(job.request_id)
            self.scheduler.prepare(job, job.arrival_ns)
            self.jobs[job.request_id] = job
            future = loop.create_future()
            self.futures[job.request_id] = future
            self.metrics.record_submission()
        await self.queue.put((self.scheduler.rank_key(job), next(self._counter), job))
        return await asyncio.shield(future)

    def get_job(self, request_id: str) -> EncodeJob:
        with self._jobs_lock:
            try:
                return self.jobs[request_id]
            except KeyError as exc:
                raise JobNotFoundError(request_id) from exc

    async def _worker(self, worker_id: int) -> None:
        while True:
            _, _, job = await self.queue.get()
            if self.scheduler.policy == "dacc":
                await self._run_dacc_batch(job, worker_id)
                continue
            try:
                job.state = JobState.RUNNING
                job.started_ns = time.perf_counter_ns()
                result = await asyncio.to_thread(self.encoder.encode, job, worker_id)
                job.embedding_dim = result.embedding_dim
                job.embedding_norm = result.embedding_norm
                job.execution_ms = result.execution_ms
                job.metadata["resource"] = result.resource
                job.state = JobState.COMPLETED
            except Exception as exc:
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_ns = time.perf_counter_ns()
                self.metrics.record(job)
                future = self.futures.get(job.request_id)
                if future is not None and not future.done():
                    future.set_result(job)
                self.queue.task_done()

    async def _run_dacc_batch(self, first: EncodeJob, worker_id: int) -> None:
        jobs = [first]
        fetched = 1
        while fetched < self.scheduler.dacc.config.window:
            try:
                _, _, queued = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            jobs.append(queued)
            fetched += 1
        now_ns = time.perf_counter_ns()
        plan = self.scheduler.dacc.plan(jobs, now_ns)
        selected_ids = {job.request_id for job in plan.jobs}
        for queued in jobs:
            if queued.request_id not in selected_ids:
                await self.queue.put((self.scheduler.rank_key(queued), next(self._counter), queued))
                self.queue.task_done()
        for selected, quota in zip(plan.jobs, plan.quotas):
            selected.sm_fraction = quota
            selected.predicted_ms = self.scheduler.dacc.safe_latency(selected, quota)
            selected.metadata.update({"schedule_mode": plan.mode, "complementarity": plan.complementarity, "schedule_score": plan.score})
            selected.state = JobState.RUNNING
            selected.started_ns = time.perf_counter_ns()
        results = await asyncio.gather(*(asyncio.to_thread(self.encoder.encode, selected, worker_id + index) for index, selected in enumerate(plan.jobs)), return_exceptions=True)
        for selected, result in zip(plan.jobs, results):
            try:
                if isinstance(result, Exception):
                    raise result
                selected.embedding_dim = result.embedding_dim
                selected.embedding_norm = result.embedding_norm
                selected.execution_ms = result.execution_ms
                selected.metadata["resource"] = result.resource
                selected.state = JobState.COMPLETED
                self.scheduler.dacc.update(selected.predicted_ms, result.execution_ms)
            except Exception as exc:
                selected.state = JobState.FAILED
                selected.error = f"{type(exc).__name__}: {exc}"
            finally:
                selected.finished_ns = time.perf_counter_ns()
                self.metrics.record(selected)
                future = self.futures.get(selected.request_id)
                if future is not None and not future.done():
                    future.set_result(selected)
                self.queue.task_done()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "queue_depth": self.queue.qsize(),
            "scheduler": self.scheduler.policy,
            "streams": len(self._workers),
            "environment": self.encoder.environment(),
            "resource": self.resource_backend.describe() if self.resource_backend else None,
        }
