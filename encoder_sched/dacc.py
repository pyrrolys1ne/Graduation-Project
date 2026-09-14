"""DACC：截止期—资源互补感知闭环调度。

本文件在 2026-09-14 依据实测结果做过一轮修复。修复前的实现存在五个已定位缺陷，
它们共同导致 DACC 在所有负载级别上被简单基线支配（见 ``docs/实验记录.md`` 第 4、5 节）：

1. **deadline guard 是死代码**：`urgency = exp(-max(slack,0)/tau)`，而 guard 条件写作
   `urgency >= 0.99`，解得 `slack <= 0.2 ms`——只有请求差 0.2 毫秒就违约时才触发。
   现在直接用 slack 判断。
2. **配对在结构上被无条件偏向**：配对方案独享 ``w_complementarity * (1-conflict)``
   （小请求对约 +0.99），而单请求没有这一项，两者在 ``gain`` 上的差异只有约 0.067。
   因此几乎总是选配对。现在单请求也拿到基准值 1.0，该项退化为只反映冲突程度。
3. **配对评分缺不确定性项**：``w_uncertainty`` 只出现在单请求评分里，对占主导的
   配对方案不起作用。现在两边都有。
4. **``future_queue_risk`` 根本没实现**：设计文档宣称的三个机制只落地了两个。
   现在补上。
5. **``resource_demand`` 是 patch 数的线性猜测**，不是测量值。现在改为由剖析表
   的配额敏感性推导：能大幅受益于更多 TPC 的请求是计算受限，几乎不受益的是访存受限。
"""

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
    #: 未来队列风险权重。修复前该项在实现中完全缺失（设计文档里有）。
    w_future: float = 2.0


@dataclass(frozen=True)
class DaccPlan:
    jobs: tuple[EncodeJob, ...]
    quotas: tuple[float, ...]
    mode: str
    score: float
    complementarity: float
    risk: float
    future_risk: float = 0.0
    duration_ms: float = 0.0


class DaccPlanner:
    """Small, deterministic planner shared by the online scheduler and simulator."""

    def __init__(self, performance_model: PerformanceModel, quota_levels: Sequence[float], config: DaccConfig | None = None):
        self.performance_model = performance_model
        self.quota_levels = tuple(sorted(float(q) for q in quota_levels))
        self.config = config or DaccConfig()
        self._correction = 1.0
        self._residual_ms = 0.0
        self._demand_cache: dict[int, dict[str, float]] = {}

    # ---------- 资源需求：由测量推导，而非 patch 数线性猜测 ----------

    def resource_demand(self, job: EncodeJob) -> dict[str, float]:
        """从剖析曲线推导请求的资源需求。

        ``sm`` 反映"多给 TPC 能挽回多少延迟"：该值高说明请求计算受限，值得占用
        SM；该值低说明请求访存/固定开销受限，多给 TPC 也没用，反而更可能与别的
        请求争夺缓存与带宽。因此 ``mem`` 取其补。

        取值范围对齐修复前的量纲（``sm ∈ [0.25, 1.0]``），使 ``conflict`` 的阈值
        ``max_pair_conflict`` 仍具可比性。
        """
        cached = self._demand_cache.get(job.patches)
        if cached is not None:
            return cached
        low = self.performance_model.predict(job.patches, self.quota_levels[0])
        high = self.performance_model.predict(job.patches, self.quota_levels[-1])
        sensitivity = 0.0 if low <= 0 else max(0.0, min(1.0, (low - high) / low))
        sm = 0.25 + 0.75 * sensitivity
        mem = 0.10 + 0.75 * (1.0 - sensitivity)
        demand = {"sm": sm, "mem": mem, "l2": mem, "dram": mem}
        self._demand_cache[job.patches] = demand
        return demand

    # ---------- 性能模型与不确定性 ----------

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

    # ---------- 排序信号 ----------

    def slack_ms(self, job: EncodeJob, now_ns: int, quota: float = 1.0) -> float:
        return job.absolute_deadline_ns / 1_000_000 - now_ns / 1_000_000 - self.safe_latency(job, quota)

    def urgency(self, job: EncodeJob, now_ns: int) -> float:
        """越接近违约越接近 1。"""
        return math.exp(-max(self.slack_ms(job, now_ns), 0.0) / max(self.config.urgency_tau_ms, 1e-6))

    def is_guarded(self, job: EncodeJob, now_ns: int) -> bool:
        """是否进入紧急保护。

        **修复点**：原实现写作 ``urgency >= 0.99``，而 urgency 是指数衰减，该条件等价于
        ``slack <= 0.2 ms``——guard 永远不会在真正需要时触发。现在直接比较 slack。
        """
        return self.slack_ms(job, now_ns) <= self.config.guard_ms

    def _age_bonus(self, job: EncodeJob, now_ns: int) -> float:
        age_ms = max(0.0, (now_ns - job.arrival_ns) / 1_000_000)
        return min(age_ms / self.config.starvation_ms, self.config.age_cap)

    def _future_risk(self, chosen_ids: set[str], pre: dict, now_ns: int, duration_ms: float) -> float:
        """执行本方案所需的时间里，窗口内未被选中的请求会被推迟到什么程度。

        **修复点**：设计文档把 ``future_queue_risk`` 列为三个机制之一，但原实现里
        完全没有这一项。这里给出一个可计算的版本：为执行本方案等待 ``duration_ms``
        之后，窗口内其余候选请求有多少会越过截止期。

        **性能**：逐请求的量全部来自 ``pre`` 预计算表。原实现对每个方案都重新调用
        `safe_latency`（内部走 numpy 插值），窗口为 8 时是 448 个方案 × 8 个候选
        ≈ 3600 次调用，单次 `plan()` 要 6.7 ms——与中位执行时间（6.87 ms）相当，
        成为 DACC 的主要排队来源。改查表后只剩算术。
        """
        finish_base_ms = now_ns / 1_000_000 + duration_ms
        risk = 0.0
        for job_id, entry in pre.items():
            if job_id in chosen_ids:
                continue
            finish_ms = finish_base_ms + entry["risk_base_ms"]
            risk += max(0.0, (finish_ms - entry["deadline_ms"]) / entry["deadline_span"])
        return risk

    def conflict(self, left: EncodeJob, right: EncodeJob, q_left: float, q_right: float) -> float:
        """两个请求共置时的资源冲突估计。

        **修复点（第六处）**：SM 项原写作 ``max(0, q_left*sm_left + q_right*sm_right - 1)``，
        但配对约束是 ``q_left + q_right <= 1`` 而 ``sm <= 1``，因此
        ``q_left*sm_left + q_right*sm_right <= 1`` **恒成立**——这一项永远为 0，是死代码。

        现在改为**与配额无关的 SM 需求竞争**：``max(0, sm_left + sm_right - 1)``。
        语义是"两个请求各自想用的 SM 份额之和是否超过整卡"，这是请求本身的属性；
        配额造成的减速已由方案评分里的 ``duration`` 项体现，不该在冲突里重复计入。

        参数 ``q_left`` / ``q_right`` 保留在签名里以维持调用方契约，当前不参与计算。
        """
        a, b = self.resource_demand(left), self.resource_demand(right)
        sm = max(0.0, a["sm"] + b["sm"] - 1.0)
        overlap = sum(a[key] * b[key] for key in ("l2", "dram")) / 2.0
        memory = max(0.0, a["mem"] + b["mem"] - 1.0)
        return min(1.0, 0.45 * sm + 0.25 * memory + 0.30 * overlap)

    # ---------- 方案评分 ----------

    def _job_value(self, job: EncodeJob, now_ns: int) -> float:
        """单个请求的"值得先服务"程度（urgency / 优先级 / 等待年龄）。"""
        return (
            self.config.w_urgency * self.urgency(job, now_ns)
            + self.config.w_priority * job.priority
            + self.config.w_age * self._age_bonus(job, now_ns)
        )

    def _precompute(self, candidates: Sequence[EncodeJob], now_ns: int) -> dict[str, dict]:
        """把逐请求量算一次，供所有候选方案查表。

        `plan()` 会枚举数百个方案；若每个方案都重新调用 `safe_latency`（内部走
        numpy 插值），单次规划的开销会与被执行的任务本身相当——实测 window=8 时
        为 6.7 ms，而中位执行时间只有 6.87 ms。这里统一预计算，把调用次数从
        O(方案数 × 候选数) 降到 O(候选数 × 配额档数)。
        """
        pre: dict[str, dict] = {}
        for job in candidates:
            safe = {}
            unc = {}
            for quota in self.quota_levels:
                value = self.safe_latency(job, quota)
                safe[quota] = value
                unc[quota] = self.uncertainty(job, quota) / max(value, 1e-6)
            pre[job.request_id] = {
                "safe": safe,
                "unc": unc,
                "value": self._job_value(job, now_ns),
                # 未来队列风险里假设该请求稍后以满配额执行
                "risk_base_ms": self.safe_latency(job, self.quota_levels[-1]),
                "deadline_ms": job.absolute_deadline_ns / 1_000_000,
                "deadline_span": max(job.deadline_ms, 1e-6),
            }
        return pre

    @staticmethod
    def _risk_of(entry: dict, now_ns: int, duration_ms: float) -> float:
        finish_ms = now_ns / 1_000_000 + duration_ms
        return max(0.0, (finish_ms - entry["deadline_ms"]) / entry["deadline_span"])

    def _single(self, job: EncodeJob, quota: float, now_ns: int, pre: dict) -> DaccPlan:
        entry = pre[job.request_id]
        safe = entry["safe"][quota]
        # safe_latency 内部已经按 quota 预测过延迟，这里不能再除以配额（会双重计算）。
        duration = safe
        risk = self._risk_of(entry, now_ns, duration)
        future = self._future_risk({job.request_id}, pre, now_ns, duration)
        score = (
            entry["value"]
            + self.config.w_gain / max(safe, 1e-6)
            # 单请求没有冲突，互补性取满值——与配对方案使用同一量纲，
            # 否则配对该项独享约 +1.0 的加成，会被无条件偏向。
            + self.config.w_complementarity * 1.0
            - self.config.w_risk * risk
            - self.config.w_uncertainty * entry["unc"][quota]
            - self.config.w_future * future
        )
        return DaccPlan((job,), (quota,), "single", score, 1.0, risk,
                        future_risk=future, duration_ms=duration)

    def plan(self, jobs: Sequence[EncodeJob], now_ns: int, active_fraction: float = 1.0) -> DaccPlan:
        candidates = list(jobs[: self.config.window])
        if not candidates:
            raise ValueError("DACC 需要至少一个候选请求")
        pre = self._precompute(candidates, now_ns)
        guarded = [job for job in candidates if self.is_guarded(job, now_ns)]

        plans: list[DaccPlan] = [
            self._single(job, q, now_ns, pre)
            for job in candidates
            for q in self.quota_levels
            if q <= active_fraction
        ]
        for left, right in itertools.combinations(candidates, 2):
            left_entry, right_entry = pre[left.request_id], pre[right.request_id]
            for q_left in self.quota_levels:
                for q_right in self.quota_levels:
                    if q_left + q_right > active_fraction:
                        continue
                    conflict = self.conflict(left, right, q_left, q_right)
                    if conflict > self.config.max_pair_conflict:
                        continue
                    safe_left, safe_right = left_entry["safe"][q_left], right_entry["safe"][q_right]
                    duration = max(safe_left, safe_right) * (1.0 + conflict)
                    # 修复点（第七处）：urgency / 优先级 / 等待年龄原为**对配对内两个请求求和**，
                    # 单请求只有一份，配对因此拿到近乎 2× 的系统性加成（等待越久越明显，
                    # age_bonus 最多给到 +0.5）。而且它与 gain 项重复奖励"一次服务更多请求"。
                    #
                    # 统一约定：**所有逐请求项在配对内取均值**，使单请求与配对的分数量纲可比。
                    # 该约定回答"这些请求值不值得服务"；"一次服务两个"的吞吐收益由 gain 项
                    # 单独表达，不再被其他项重复计入。risk 与 uncertainty 同理（它们是惩罚项，
                    # 求和会让配对承受双倍惩罚，方向相反但同样破坏可比性）。
                    value = (left_entry["value"] + right_entry["value"]) / 2.0
                    risk = (self._risk_of(left_entry, now_ns, duration)
                            + self._risk_of(right_entry, now_ns, duration)) / 2.0
                    gain = 2.0 / max(duration, 1e-6)
                    # 配对评分原先还缺不确定性项，而它占据绝大多数决策。
                    uncertainty = (left_entry["unc"][q_left] + right_entry["unc"][q_right]) / 2.0
                    future = self._future_risk({left.request_id, right.request_id}, pre, now_ns, duration)
                    score = (
                        value
                        + self.config.w_gain * gain
                        + self.config.w_complementarity * (1.0 - conflict)
                        - self.config.w_risk * risk
                        - self.config.w_uncertainty * uncertainty
                        - self.config.w_future * future
                    )
                    plans.append(DaccPlan((left, right), (q_left, q_right), "pair", score,
                                          1.0 - conflict, risk, future_risk=future, duration_ms=duration))

        if guarded:
            # 紧急保护：只在"必然包含紧急请求"的方案里挑，且不允许为了让位给配对
            # 而推迟它——配对方案仅在与紧急请求冲突足够小、且不增加其违约风险时保留。
            guarded_ids = {job.request_id for job in guarded}
            pool = [p for p in plans if any(j.request_id in guarded_ids for j in p.jobs)]
            if pool:
                # 优先单请求执行紧急请求（最小可行安全配额），配对仅作兜底。
                singles = [p for p in pool if p.mode == "single"]
                pool = singles or pool
        else:
            pool = plans
        return max(pool, key=lambda plan: (plan.score, -len(plan.jobs), tuple(j.request_id for j in plan.jobs)))
