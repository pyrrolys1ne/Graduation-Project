from __future__ import annotations

import asyncio
import itertools
import threading
import time
from pathlib import Path
from typing import Any

from .concurrency import ConcurrencyConfig, ConcurrencyController
from .config import AppConfig
from .encoder import EncoderBackend
from .graph_runtime import GraphReplayRuntime, GraphRuntimeConfig
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
        sample_gpu: bool = True,
    ):
        self.config = config
        self.scheduler = scheduler
        self.encoder = encoder
        self.metrics = metrics
        self.resource_backend = resource_backend
        self.runtime_report: dict[str, Any] | None = None
        self.queue: asyncio.PriorityQueue[tuple[tuple[float, ...], int, EncodeJob]] = asyncio.PriorityQueue()
        self.jobs: dict[str, EncodeJob] = {}
        self.futures: dict[str, asyncio.Future[EncodeJob]] = {}
        self._jobs_lock = threading.Lock()
        self._counter = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self.gpu_sampler = (
            GpuUtilizationSampler(metrics, interval_s=config.metrics.gpu_sample_interval_s)
            if sample_gpu
            else None
        )
        #: 运行时并发控制器。仅在 ``executor.adaptive_concurrency`` 打开时非 None，
        #: 否则并发度仍是启动时定死的 ``executor.streams``（历史行为，保持可复现）。
        self.controller: ConcurrencyController | None = None
        #: 图回放流水线运行时。仅在 ``executor.graph_pipeline`` 打开时非 None。
        #: 它同时是"并发度的来源"（``graph.slots``）与"两级流水线的执行体"。
        self.graph: GraphReplayRuntime | None = None
        #: 图回放流水线的**预处理级 → 回放级**队列。两级拆成两个协程是这一级的全部意义：
        #: 合并成一个协程会让"准备下一个"与"回放当前"串行，实测把 `slots=1` 的服务吞吐
        #: 从应当的约 300 req/s 压到 **141 req/s**（几乎等于 eager 基线 129.3），
        #: 等于把图回放的收益整个吃掉。
        self._prepared: asyncio.Queue[tuple[EncodeJob, Any]] = asyncio.Queue()
        self._preparers: list[asyncio.Task[None]] = []
        # 已提交但尚未通过准入闸门的请求。worker 会先从 PriorityQueue 取走任务再等待
        # 准入，因此只看 queue 无法得到真实等待集合；必须在服务层单独建账。
        self._awaiting_admission: dict[str, int] = {}
        # 只有整个执行区间始终没有共驻者的请求，才可与单请求剖析表比较。
        self._drift_solo_candidates: set[str] = set()

    async def start(self) -> None:
        batching = self.config.batching.max_batch > 1
        if batching and self.scheduler.policy == "dacc":
            raise ValueError(
                "批处理与 dacc 策略不能同时启用：dacc 自己就要在同一窗口内挑选 2 个请求并发执行，"
                "再叠加批处理会让'一次前向处理几个请求'这件事被两套逻辑同时决定。"
            )
        if self.config.executor.graph_pipeline:
            if self.scheduler.policy in {"serial_fcfs", "dacc"}:
                raise ValueError(
                    f"graph_pipeline 不支持 scheduler.policy={self.scheduler.policy}："
                    "该策略自带单 worker 的串行编排，与'并发度来自 graph.slots'冲突。"
                )
            # 图回放流水线：并发度来自 graph.slots，两级线程池由 runtime 内部持有。
            self.graph = GraphReplayRuntime(
                self.encoder, self.config.executor.graph, self.resource_backend
            )
            worker_count = self.graph.config.slots
            coro = self._graph_worker
            # 预处理级：与回放 lane 数量解耦，由 graph.prep_threads 给出。
            self._preparers = [
                asyncio.create_task(self._graph_preparer(index),
                                    name=f"graph-prep-{index}")
                for index in range(max(1, self.graph.config.prep_threads))
            ]
        elif batching:
            # 批处理模式下由单个 worker 组批，并发度来自 batch 而不是多 worker
            worker_count = 1
            coro = self._batch_worker
        else:
            worker_count = 1 if self.scheduler.policy in {"serial_fcfs", "dacc"} else self.config.executor.streams
            coro = self._worker
        if (self.config.executor.adaptive_concurrency and not batching and self.graph is None
                and self.scheduler.policy not in {"serial_fcfs", "dacc"}):
            # 弹性池：并发度**不再**在启动时定死，而是由 controller 按队列状态给出。
            # 池子先按上界铺满，多出的 worker 阻塞在准入闸门上（见 _worker）。
            self.controller = ConcurrencyController(self.config.executor.concurrency)
            worker_count = self.controller.config.max_concurrency
        self._workers = [
            asyncio.create_task(coro(index), name=f"encoder-worker-{index}") for index in range(worker_count)
        ]
        if self.gpu_sampler is not None:
            self.gpu_sampler.start()

    async def stop(self) -> None:
        await self.queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        for preparer in self._preparers:
            preparer.cancel()
        await asyncio.gather(*self._preparers, return_exceptions=True)
        if self.graph is not None:
            self.graph.close()
        if self.gpu_sampler is not None:
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
            if self.controller is not None:
                self.controller.note_arrival()
                self._awaiting_admission[job.request_id] = job.patches
        await self.queue.put((self.scheduler.rank_key(job), next(self._counter), job))
        return await asyncio.shield(future)

    def get_job(self, request_id: str) -> EncodeJob:
        with self._jobs_lock:
            try:
                return self.jobs[request_id]
            except KeyError as exc:
                raise JobNotFoundError(request_id) from exc

    async def _await_admission(self, job: EncodeJob) -> None:
        """准入闸门：并发度自适应时，请求在此等待直到 controller 放行。

        闸门的意义是把"此刻允许几个请求共驻"变成**逐请求**的运行时决策，
        而不是启动时的常量。等待期间**不阻塞事件循环**——CLIP 前向走
        ``asyncio.to_thread``，闸门只是让多余的 worker 在此让出。

        队列为空时 controller 返回上界，因此只要没有竞争就不会有人在这里等，
        与"零等待"原则一致（本课题实测：愿意等待在任何负载下都是纯损失）。
        """
        controller = self.controller
        assert controller is not None
        while True:
            # 所有 worker 共享同一等待快照。旧实现只传 [job.patches]，使“混合比例”
            # 退化为 0%/100%，并随恰好抢到 CPU 的 worker 改变，实际上没有观察队列。
            awaiting = list(self._awaiting_admission.values())
            # 只有当前请求且 GPU 上无人执行时，确实没有竞争者，走 Bless 的空闲路径。
            # 已有驻留请求时，即使只剩当前任务，它也会与驻留任务竞争，仍按尺寸分类。
            pending = [] if controller.resident() == 0 and len(awaiting) == 1 else awaiting
            target = controller.target_concurrency(pending)
            if controller.resident() < target:
                resident_before = controller.resident()
                controller.note_start()
                if resident_before == 0:
                    self._drift_solo_candidates.add(job.request_id)
                else:
                    # 第二个请求进入后，当前所有活跃请求都受过共驻影响，不能再作为
                    # 单请求标定样本；后续即使只剩一个，也不能恢复资格。
                    self._drift_solo_candidates.clear()
                self._awaiting_admission.pop(job.request_id, None)
                job.metadata["admission"] = {
                    "target": target,
                    "reason": controller.last_reason(),
                    "resident_at_admit": controller.resident(),
                    "waiting_at_decision": len(awaiting),
                }
                return
            await asyncio.sleep(0.001)

    async def _worker(self, worker_id: int) -> None:
        while True:
            _, _, job = await self.queue.get()
            if self.scheduler.policy == "dacc":
                await self._run_dacc_batch(job, worker_id)
                continue
            admitted = False
            try:
                if self.controller is not None:
                    await self._await_admission(job)
                    admitted = True
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
                self._awaiting_admission.pop(job.request_id, None)
                job.finished_ns = time.perf_counter_ns()
                if admitted and self.controller is not None:
                    drift_eligible = job.request_id in self._drift_solo_candidates
                    self._drift_solo_candidates.discard(job.request_id)
                    job.metadata["admission"]["drift_sample_eligible"] = drift_eligible
                    self.controller.note_finish(
                        job.patches,
                        observed_ms=job.execution_ms or 0.0,
                        # 标定表来自单请求执行。共驻墙钟包含策略自身的争用，不能用来
                        # 判断环境漂移；传 0 让控制器跳过非独占样本。
                        predicted_ms=job.predicted_ms if drift_eligible else 0.0,
                    )
                self.metrics.record(job)
                future = self.futures.get(job.request_id)
                if future is not None and not future.done():
                    future.set_result(job)
                self.queue.task_done()

    async def _collect_batch(self, first: EncodeJob, cap: int) -> list[EncodeJob]:
        """从队列里凑一批**同尺寸**的请求，**绝不为凑批等待**。

        §7 的教训：「需要等待才能组批的时刻，恰恰是不值得组批的时刻」——
        等待在任何负载下都是纯损失（欠饱和时只抬高 SLO 违约率，过载时队列本就有积压、
        不等也凑得到）。因此这里只做 `get_nowait`，一次都不 sleep。

        批处理要求同形状，而请求尺寸在队列里是交错的（224/336/448/672 循环到达），
        只看队首的下一个必然凑不成批——所以要向后扫一段，把非同尺寸的原样放回。
        这与既有 `_batch_worker` 的扫描逻辑是同一个道理。
        """
        runtime = self.graph
        assert runtime is not None
        if cap <= 1:
            return [first]
        group = [first]
        others: list[EncodeJob] = []
        scanned = 0
        limit = cap * runtime.config.batch_scan_factor
        while len(group) < cap and scanned < limit:
            try:
                _, _, candidate = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            scanned += 1
            if (candidate.width, candidate.height) == (first.width, first.height):
                group.append(candidate)
            else:
                others.append(candidate)
        for job in others:
            # put 之后必须补一次 task_done：`put` 会让 unfinished_tasks 加一，
            # 而这一条已经在 `submit` 时计过一次了。少了它 `queue.join()` 会永远等下去。
            await self.queue.put((self.scheduler.rank_key(job), next(self._counter), job))
            self.queue.task_done()
        return group

    async def _graph_preparer(self, index: int) -> None:
        """预处理级：从请求队列取件、按尺寸凑批、在预处理线程池上造图，交给回放级。

        **与回放级分成两个协程是这一级的全部意义。** 合并成一个协程会让
        "准备下一个"与"回放当前"串行——实测把 `slots=1` 的服务吞吐从应当的
        约 300 req/s 压到 **141 req/s**（几乎等于 eager 基线 129.3），
        等于把图回放的收益整个吃掉。

        **凑批是这一级的第二个职责**（`graph.batch_by_size` 非空时生效）：
        批在预处理级按尺寸组好，回放级只负责"一次前向处理这一批"。
        两个并行度（`graph.prep_threads` / `graph.slots`）与批上限因此是三个解耦的旋钮。
        """
        runtime = self.graph
        assert runtime is not None
        while True:
            _, _, first = await self.queue.get()
            if not runtime.supports(first):
                self._finish_graph_job(first, error=(
                    f"尺寸 {first.width}×{first.height} 未在图池中。图必须在启动阶段捕获"
                    "（中途捕获要求显存上没有在飞的图），请把它加进 executor.graph.sizes。"
                ))
                continue
            jobs = await self._collect_batch(first, runtime.batch_capacity(first))
            try:
                prepared = await runtime.aprepare(jobs)
            except Exception as exc:  # noqa: BLE001 - 预处理失败也要把这一批收尾
                error = f"{type(exc).__name__}: {exc}"
                for job in jobs:
                    self._finish_graph_job(job, error=error)
                continue
            await self._prepared.put((jobs, prepared))

    async def _graph_worker(self, worker_id: int) -> None:
        """回放级（一条 lane）：一条流、一份图实例、一个并发槽位。

        只做两件事：从预处理级取件、调 ``runtime.replay``。**不做 CPU 造图**——
        把两者放在同一个线程里会挂死（``graph_runtime`` 模块 docstring 的二分表）：
        CPU 有持续负载、且另一条流上有图在回放时，所有线程会卡在各自的
        ``stream.synchronize()`` 上互等。

        ``worker_id`` 同时是**槽位号**：``graph.slots`` == 回放 lane 数，
        ``slots <= streams`` 由配置校验保证，因此每个槽位独占一条流与一份图实例。
        """
        runtime = self.graph
        assert runtime is not None
        while True:
            jobs, prepared = await self._prepared.get()
            started_ns = time.perf_counter_ns()
            for job in jobs:
                job.state = JobState.RUNNING
                # 批内各请求同时开始、同时结束，因此共享同一个 started_ns
                job.started_ns = started_ns
            try:
                results = await asyncio.to_thread(runtime.replay, jobs, worker_id, prepared)
            except Exception as exc:  # noqa: BLE001 - 整批一起失败
                error = f"{type(exc).__name__}: {exc}"
                for job in jobs:
                    self._finish_graph_job(job, error=error)
                continue
            for job, result in zip(jobs, results):
                job.embedding_dim = result["embedding_dim"]
                job.embedding_norm = result["embedding_norm"]
                job.execution_ms = result["execution_ms"]
                job.metadata["resource"] = result["resource"]
                # prepare_ms 与 execution_ms 一起构成"两级各花了多少"的原始记录；
                # batch 记下这一批实际几条——**成批臂的判据全靠它**：
                # 没有它就无法区分"成批没生效（batch 恒为 1）"与"成批生效但没收益"，
                # 而本项目历史上真的发生过平均批大小恒为 1.00 的无效批次（§14）。
                job.metadata["graph_pipeline"] = {
                    "slot": worker_id,
                    "prepare_ms": prepared.cpu_ms,
                    "size": f"{job.width}x{job.height}",
                    "batch": len(jobs),
                }
                job.state = JobState.COMPLETED
                self._finish_graph_job(job)

    def _finish_graph_job(self, job: EncodeJob, error: str | None = None) -> None:
        """图回放路径的统一收尾：**只有这里会调 task_done**。

        预处理级失败与回放级失败都走这一个出口，否则 `queue.join()` 会漏计。
        """
        if error is not None:
            job.state = JobState.FAILED
            job.error = error
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
            "server_runtime": self.runtime_report,
            "resource": self.resource_backend.describe() if self.resource_backend else None,
            "graph_pipeline": self.graph.describe() if self.graph is not None else None,
        }
