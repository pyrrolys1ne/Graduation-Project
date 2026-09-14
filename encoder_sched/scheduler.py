from __future__ import annotations

import time
from collections.abc import Iterable

from .models import EncodeJob
from .performance import PerformanceModel
from .dacc import DaccConfig, DaccPlanner


SUPPORTED_POLICIES = {"serial_fcfs", "multistream_fcfs", "edf", "edf_size", "dacc"}

#: 配额选择策略。见 ``Scheduler.choose_quota`` 的说明。
QUOTA_POLICIES = {"deadline_min", "full"}


class Scheduler:
    def __init__(
        self,
        policy: str,
        performance_model: PerformanceModel,
        quota_levels: Iterable[float],
        deadline_tie_ms: float = 5.0,
        dacc_config: DaccConfig | None = None,
        quota_policy: str = "deadline_min",
    ):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"未知调度策略: {policy}")
        if quota_policy not in QUOTA_POLICIES:
            raise ValueError(f"未知的 quota_policy: {quota_policy}；可选 {sorted(QUOTA_POLICIES)}")
        self.policy = policy
        self.quota_policy = quota_policy
        self.performance_model = performance_model
        self.quota_levels = tuple(sorted(float(level) for level in quota_levels))
        self.deadline_tie_ns = int(deadline_tie_ms * 1_000_000)
        self.dacc = DaccPlanner(performance_model, self.quota_levels, dacc_config) if policy == "dacc" else None

    def rank_key(self, job: EncodeJob) -> tuple[float, ...]:
        """优先队列的排序键（最小堆）。

        注意这是**入队时**计算的静态键。DACC 真正的决策在 ``DaccPlanner.plan()`` 里，
        那里会拿到执行时刻的 ``now_ns``；本函数只决定哪些请求进入候选窗口。
        """
        if self.policy in {"serial_fcfs", "multistream_fcfs"}:
            return (float(job.arrival_ns),)
        if self.policy == "edf":
            return (float(job.absolute_deadline_ns), float(-job.priority), float(job.arrival_ns))
        if self.policy == "dacc":
            # 用到达时刻的 urgency 作键：取负号以适配最小堆（urgency 越大越先出队）。
            # 刻意使用 job.arrival_ns 而非当前时间——本函数会在非选中请求重新入队时
            # 被再次调用，用当前时间会让排序随等待时间漂移，破坏可复现性。
            return (
                float(-self.dacc.urgency(job, job.arrival_ns)),
                float(-job.priority),
                float(job.predicted_ms),
                float(job.arrival_ns),
            )
        deadline_bucket = job.absolute_deadline_ns // max(1, self.deadline_tie_ns)
        return (
            float(deadline_bucket),
            float(-job.priority),
            float(job.predicted_ms),
            float(job.arrival_ns),
        )

    def choose_quota(self, job: EncodeJob, now_ns: int | None = None) -> float:
        """选择该请求要占用的 SM 配额。

        两种策略，由配置项 ``scheduler.quota_policy`` 决定：

        ``deadline_min``（默认，历史行为）
            升序遍历配额档位，返回第一个「预测延迟 ≤ 剩余时间」的档位，即"占用最少
            资源仍能满足截止期"。**这个目标函数值得质疑**：它把资源守恒当成收益，
            但本机 GPU 利用率仅 25–33%，省下的 TPC 若无人使用就毫无价值，而请求本身
            会慢数倍（672×672 用 0.25 档比 1.0 档慢约 4 倍）。另外它在**提交时**用
            `deadline − now` 估算，既没有扣除排队时间，也没有随执行时刻更新。

        ``full``
            始终使用最大配额。把"省资源"的动机交给并发调度去处理，而不是让单个请求
            主动变慢。哪种更好必须由实验决定，不要凭直觉选。

        另外：本函数只对 ``edf_size`` 与 ``dacc`` 生效——其余策略是固定单流或多流
        FCFS 基线，各自用满配额，不应引入配额选择这一变量。
        """
        if self.policy not in {"edf_size", "dacc"}:
            return self.quota_levels[-1]
        if self.quota_policy == "full":
            return self.quota_levels[-1]
        if self.quota_policy != "deadline_min":
            raise ValueError(f"未知的 quota_policy: {self.quota_policy}")
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
            job.metadata.update(
                {
                    "prediction_mode": "prior",
                    "resource_demand": self.dacc.resource_demand(job),
                    "schedule_mode": "single",
                    "quota_policy": self.quota_policy,
                }
            )
        return job
