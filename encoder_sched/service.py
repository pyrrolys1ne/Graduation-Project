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
        self.gpu_sampler = GpuUtilizationSampler(metrics, interval_s=config.metrics.gpu_sample_interval_s)

    async def start(self) -> None:
        batching = self.config.batching.max_batch > 1
        if batching and self.scheduler.policy == "dacc":
            raise ValueError(
                "批处理与 dacc 策略不能同时启用：dacc 自己就要在同一窗口内挑选 2 个请求并发执行，"
                "再叠加批处理会让'一次前向处理几个请求'这件事被两套逻辑同时决定。"
            )
        if batching:
            # 批处理模式下由单个 worker 组批，并发度来自 batch 而不是多 worker
            worker_count = 1
            coro = self._batch_worker
        else:
            worker_count = 1 if self.scheduler.policy in {"serial_fcfs", "dacc"} else self.config.executor.streams
            coro = self._worker
        self._workers = [
            asyncio.create_task(coro(index), name=f"encoder-worker-{index}") for index in range(worker_count)
        ]
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

    async def _batch_worker(self, worker_id: int) -> None:
        """批处理 worker：从队列收集**同尺寸**请求凑成一批，一次前向完成。

        与空间分割的区别是本课题的核心对照：批处理把多个请求合并成一次前向，让硬件
        自己在同一批 SM 上交错调度各请求的 block（吞吐高，但请求要等组批）；空间分割
        给每个请求独立的 TPC 区间（无需等待，但划分后 SM 内部不再有跨请求填空的机会）。

        "愿意等多久"由 ``batching.max_delay_ms`` 直接控制——这正是取舍的核心旋钮：
        设为 0 表示绝不等待、只取队列里已有的请求，那时它与逐请求执行几乎没有区别。
        """
        max_batch = self.config.batching.max_batch
        delay_s = self.config.batching.max_delay_ms / 1000.0
        # 扫描窗口。批处理要求张量同形状，而请求的尺寸在队列里是交错的
        # （例如 224/336/448/672 循环到达），只看队首的下一个必然凑不成批——
        # 实测中曾因此让所有批大小都是 1，整轮实验等于在测空气。
        # 取 max_batch × 4 是为了让每种尺寸至少出现 max_batch 次。
        scan = max(max_batch * 4, 8)
        while True:
            _, _, first = await self.queue.get()
            held = [first]
            # 先把队列里**已经存在**的请求全部取出（不等新到达）。
            # 少了这一步，delay_ms=0 会退化成纯串行——实测中 batch4_d0 的平均批大小
            # 因此恒为 1.00，与 serial_fcfs 毫无区别。
            while len(held) < scan:
                try:
                    _, _, candidate = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                held.append(candidate)
            # 再按 delay_ms 等待新到达，直到凑够扫描窗口或超时。
            deadline = time.perf_counter() + delay_s
            while len(held) < scan:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    _, _, candidate = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                held.append(candidate)

            # 按尺寸分组，取最大的一组凑成这一批；其余原样放回队列。
            groups: dict[tuple[int, int], list[EncodeJob]] = {}
            for job in held:
                groups.setdefault((job.width, job.height), []).append(job)
            _, jobs = max(groups.items(), key=lambda item: len(item[1]))
            jobs = jobs[:max_batch]
            chosen = {id(job) for job in jobs}
            for job in held:
                if id(job) not in chosen:
                    await self.queue.put((self.scheduler.rank_key(job), next(self._counter), job))
                    self.queue.task_done()
            await self._run_batch(jobs, worker_id)

    async def _run_batch(self, jobs: list[EncodeJob], worker_id: int) -> None:
        for job in jobs:
            job.state = JobState.RUNNING
            job.started_ns = time.perf_counter_ns()
        results = None
        error: str | None = None
        try:
            results = await asyncio.to_thread(self.encoder.encode_batch, jobs, worker_id)
        except Exception as exc:  # noqa: BLE001 - 记录到 job 上供 API 返回
            error = f"{type(exc).__name__}: {exc}"
        finished_ns = time.perf_counter_ns()
        for index, job in enumerate(jobs):
            try:
                if results is None:
                    job.state = JobState.FAILED
                    job.error = error
                else:
                    result = results[index]
                    job.embedding_dim = result.embedding_dim
                    job.embedding_norm = result.embedding_norm
                    job.execution_ms = result.execution_ms
                    job.metadata["resource"] = result.resource
                    job.metadata["batch_size"] = len(jobs)
                    job.metadata["batch_index"] = index
                    job.state = JobState.COMPLETED
            except Exception as exc:  # noqa: BLE001
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_ns = finished_ns
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
