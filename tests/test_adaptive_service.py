"""弹性并发池的集成测试（FakeEncoder，无需 GPU）。

这里要保护的是**准入闸门不会死锁**，以及**关闭自适应时行为与历史完全一致**。

死锁是这类机制最现实的风险：闸门的放行条件是
``controller.resident() < controller.target_concurrency(...)``，
若 ``target_concurrency`` 可能返回 0、或 ``resident`` 只增不减，
worker 会永远等不到许可——而这类 bug 在 GPU 实验中表现为"吞吐骤降"，
很容易被误读成"策略更差"。因此必须在无 GPU 的确定性测试里先钉死。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from encoder_sched.concurrency import ConcurrencyConfig
from encoder_sched.config import AppConfig, ExecutorConfig, MetricsConfig, SchedulerConfig
from encoder_sched.encoder import FakeEncoderBackend
from encoder_sched.metrics import MetricsStore
from encoder_sched.models import EncodeJob, JobState
from encoder_sched.performance import PerformanceModel
from encoder_sched.resource import ProxyResourceBackend
from encoder_sched.scheduler import Scheduler
from encoder_sched.service import EncoderService

TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def make_service(
    adaptive: bool, max_concurrency: int = 4, policy: str = "multistream_fcfs", tmp_path: Path | None = None
) -> EncoderService:
    log_path = (tmp_path or Path(".")) / "requests.jsonl"
    config = AppConfig(
        scheduler=SchedulerConfig(policy=policy),
        executor=ExecutorConfig(
            streams=2,
            adaptive_concurrency=adaptive,
            concurrency=ConcurrencyConfig(max_concurrency=max_concurrency),
        ),
        metrics=MetricsConfig(gpu_sample_interval_s=0.1),
        base_dir=Path("."),
    )
    scheduler = Scheduler(policy, PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), 5.0)
    backend = ProxyResourceBackend()
    metrics = MetricsStore(log_path)
    return EncoderService(config, scheduler, FakeEncoderBackend(backend), metrics, backend)


async def run_jobs(service: EncoderService, jobs: list[EncodeJob], timeout_s: float = 30.0):
    async def one(job: EncodeJob):
        try:
            return await asyncio.wait_for(service.submit(job), timeout=timeout_s)
        except asyncio.TimeoutError:
            return "TIMEOUT"

    await service.start()
    try:
        return await asyncio.gather(*(one(job) for job in jobs))
    finally:
        await service.stop()


def mixed_jobs() -> list[EncodeJob]:
    """**顺序本身就是这个测试的一部分**，不能随便排。

    准入闸门观察的是"已提交但尚未准入"的积压，而 worker 按队列优先级
    （此处 FIFO）取走任务，因此每次决策看到的 pending 集合都是提交序列的一个
    **后缀**。于是"混合比例"只能沿后缀演变：只有先排小请求、后排大请求，
    比例才会先从高走低、跨过 ``small_mix_ratio``。

    旧版把序列排成"10 小 + 10 大 + 10 小"，尾部那 10 个小请求会一直挂在
    pending 里把比例托在 0.5 以上，``large_mix`` 分支在数学上不可达——
    测试因此恒失败，而控制器本身是好的（真实负载下 400 次准入里
    large_mix 148 次、small_mix 248 次）。
    """
    sizes = [(224, 224)] * 20 + [(672, 672)] * 10
    return [EncodeJob(f"r{i}", 1, w, h, 500.0, arrival_ns=0) for i, (w, h) in enumerate(sizes)]


def test_adaptive_pool_completes_without_deadlock(tmp_path):
    """核心安全性：自适应开启时全部请求必须完成，不得有超时。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    assert "TIMEOUT" not in results, "准入闸门死锁：有请求永远等不到许可"
    assert all(job.state is JobState.COMPLETED for job in results)
    assert all(job.error is None for job in results)


def test_adaptive_pool_records_admission_metadata(tmp_path):
    """准入决策必须落进 job.metadata，否则事后无法归因"为什么这一刻是并发 2"。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    admitted = [job for job in results if "admission" in job.metadata]
    assert admitted, "没有任何请求记录准入信息"
    for job in admitted:
        info = job.metadata["admission"]
        assert info["target"] >= 1
        assert isinstance(info["reason"], str) and info["reason"]
        assert info["resident_at_admit"] >= 1
        assert info["waiting_at_decision"] >= 1


def test_admission_uses_complete_waiting_mix(tmp_path):
    """准入必须观察服务级等待集合，不能退化成只看当前 worker 的一个任务。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)

    async def scenario() -> tuple[int, str]:
        await service.start()
        assert service.controller is not None
        # 模拟三个 worker 已从 PriorityQueue 取走任务、都在闸门前等待：两小一大。
        service._awaiting_admission = {"small-1": 49, "small-2": 100, "large": 441}
        job = EncodeJob("large", 1, 672, 672, 500.0, arrival_ns=0)
        try:
            await service._await_admission(job)
            return job.metadata["admission"]["waiting_at_decision"], job.metadata["admission"]["reason"]
        finally:
            service.controller.note_finish(job.patches, observed_ms=0.0, predicted_ms=0.0)
            await service.stop()

    waiting, reason = asyncio.run(scenario())
    assert waiting == 3
    assert reason == "small_mix(0.67)"


def test_single_request_without_resident_uses_idle_path(tmp_path):
    """单个请求且无驻留任务时没有竞争者，应走空闲整卡路径。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)

    async def scenario() -> str:
        await service.start()
        assert service.controller is not None
        service._awaiting_admission = {"only": 441}
        job = EncodeJob("only", 1, 672, 672, 500.0, arrival_ns=0)
        try:
            await service._await_admission(job)
            return job.metadata["admission"]["reason"]
        finally:
            service.controller.note_finish(job.patches, observed_ms=0.0, predicted_ms=0.0)
            await service.stop()

    assert asyncio.run(scenario()) == "idle_use_full_gpu"


def test_concurrent_requests_are_not_used_as_drift_samples(tmp_path):
    """共驻墙钟包含策略自身争用，不得拿去和单请求剖析表比较。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)

    async def scenario() -> tuple[bool, bool, int]:
        await service.start()
        assert service.controller is not None
        jobs = [
            EncodeJob("first", 1, 224, 224, 500.0, arrival_ns=0),
            EncodeJob("second", 1, 224, 224, 500.0, arrival_ns=0),
        ]
        service._awaiting_admission = {job.request_id: job.patches for job in jobs}
        try:
            await service._await_admission(jobs[0])
            await service._await_admission(jobs[1])
            eligible = tuple(job.request_id in service._drift_solo_candidates for job in jobs)
            return eligible[0], eligible[1], service.controller.resident()
        finally:
            for job in jobs:
                service.controller.note_finish(job.patches, observed_ms=0.0, predicted_ms=0.0)
            await service.stop()

    first, second, resident = asyncio.run(scenario())
    assert resident == 2
    assert not first and not second


def test_controller_sees_mixed_reasons(tmp_path):
    """混合负载下控制器应当真的改变过主意——否则它只是个常数。"""
    service = make_service(adaptive=True, tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    reasons = {job.metadata["admission"]["reason"] for job in results if "admission" in job.metadata}
    assert any("small_mix" in r for r in reasons), f"从未进入小混合分支: {reasons}"
    assert any("large_mix" in r for r in reasons), f"从未进入大混合分支: {reasons}"


def test_disabled_path_matches_historical_behaviour(tmp_path):
    """关闭自适应时不得引入任何准入元数据——历史结果必须逐字节可复现。"""
    service = make_service(adaptive=False, tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    assert all(job.state is JobState.COMPLETED for job in results)
    assert all("admission" not in job.metadata for job in results)
    assert service.controller is None


def test_adaptive_ignored_for_serial_policy(tmp_path):
    """serial_fcfs 是单流基线；给它开自适应会把基线本身改掉。"""
    service = make_service(adaptive=True, policy="serial_fcfs", tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    assert service.controller is None
    assert all(job.state is JobState.COMPLETED for job in results)


@pytest.mark.parametrize("max_conc", [1, 2, 3, 8])
def test_all_concurrency_bounds_complete(tmp_path, max_conc: int):
    """任何上界都必须能跑完；max=1 退化为串行也不能死锁。"""
    service = make_service(adaptive=True, max_concurrency=max_conc, tmp_path=tmp_path)
    results = asyncio.run(run_jobs(service, mixed_jobs()))
    assert all(job.state is JobState.COMPLETED for job in results), f"max_concurrency={max_conc} 未跑完"
