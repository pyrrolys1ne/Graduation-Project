from pathlib import Path

import pytest

from encoder_sched.dacc import DaccConfig, DaccPlanner
from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel


TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def planner() -> DaccPlanner:
    return DaccPlanner(PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), DaccConfig(window=8))


def test_plan_is_drawn_from_the_candidate_window():
    p = planner()
    left = EncodeJob("left", 1, 224, 224, 1000, arrival_ns=0)
    right = EncodeJob("right", 1, 224, 224, 1000, arrival_ns=0)
    plan = p.plan([left, right], 0)
    assert plan.mode in {"single", "pair"}
    assert 1 <= len(plan.jobs) <= 2
    assert {j.request_id for j in plan.jobs} <= {"left", "right"}
    assert len(plan.quotas) == len(plan.jobs)


def test_pairing_is_not_structurally_forced_by_the_complementarity_term():
    """互补性项不能变成"配对独享的固定加成"。

    修复前的实现里，配对独享 ``w_complementarity * (1-conflict)``（小请求对约 +0.99），
    而单请求没有这一项，两者在 ``gain`` 上的差距只有约 0.067——配对因此被**无条件**
    选中，与请求是否真的互补无关。现在单请求也取基准值 1.0，该差异消失。

    这里验证的是行为对冲突的实际敏感度：把允许的冲突上限压到 0（禁止任何配对）后，
    规划器必须退化为单请求方案，而不是仍然返回配对。
    """
    p = planner()
    left = EncodeJob("left", 1, 224, 224, 1000, arrival_ns=0)
    right = EncodeJob("right", 1, 224, 224, 1000, arrival_ns=0)

    relaxed = DaccPlanner(PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0),
                          DaccConfig(window=8, max_pair_conflict=0.0))
    assert relaxed.plan([left, right], 0).mode == "single"


def test_conflict_reflects_measured_resource_demand():
    """冲突度必须由实测剖析表推导，并且是活的量（不能恒为 0）。

    这里断言的是性质而非具体阈值：**计算受限的大请求之间 SM 竞争更强**，
    而访存受限的小请求之间几乎没有 SM 竞争。
    """
    p = planner()
    big_a = EncodeJob("big-a", 1, 672, 672, 1000, arrival_ns=0)
    big_b = EncodeJob("big-b", 1, 672, 672, 1000, arrival_ns=0)
    small_a = EncodeJob("small-a", 1, 224, 224, 1000, arrival_ns=0)
    small_b = EncodeJob("small-b", 1, 224, 224, 1000, arrival_ns=0)

    big = p.conflict(big_a, big_b, 0.5, 0.5)
    small = p.conflict(small_a, small_b, 0.5, 0.5)
    assert 0.0 < big <= 1.0, f"两个计算受限的大请求应有可测的 SM 竞争，实际 {big}"
    assert 0.0 < small <= 1.0, f"两个小请求应有访存层面的重叠，实际 {small}"
    assert big != small, "冲突度对请求规模不敏感，说明需求推导没起作用"


def test_conflict_does_not_depend_on_quota():
    """配额造成的减速由方案评分里的 duration 体现，冲突度里不该重复计入。

    这条同时防止 SM 项退回成 ``max(0, q_a*sm_a + q_b*sm_b - 1)``——在
    ``q_a + q_b <= 1`` 且 ``sm <= 1`` 的约束下那一项恒为 0，是死代码。
    """
    p = planner()
    a = EncodeJob("a", 1, 672, 672, 1000, arrival_ns=0)
    b = EncodeJob("b", 1, 672, 672, 1000, arrival_ns=0)
    assert p.conflict(a, b, 0.25, 0.25) == pytest.approx(p.conflict(a, b, 0.75, 0.25))


def test_dacc_guard_prefers_urgent_request_over_pair():
    p = planner()
    urgent = EncodeJob("urgent", 1, 672, 672, 1, arrival_ns=0)
    relaxed = EncodeJob("relaxed", 1, 224, 224, 1000, arrival_ns=0)
    plan = p.plan([relaxed, urgent], 0)
    assert plan.jobs[0].request_id == "urgent"


def test_guard_triggers_on_slack_not_on_urgency_threshold():
    """deadline guard 必须由 slack 直接判定。

    修复前的实现写作 ``urgency >= 0.99``，而 urgency 是指数衰减，
    该条件等价于 ``slack <= tau * ln(1/0.99) ≈ 0.2 ms``——只有请求差 0.2 毫秒
    就违约时才触发，实际是死代码。
    """
    p = planner()
    # 距截止期 5 ms、安全执行时间约 6 ms 量级 -> 应判定为紧急
    nearly_late = EncodeJob("nearly-late", 1, 224, 224, 5, arrival_ns=0)
    assert p.is_guarded(nearly_late, 0) is True
    comfortable = EncodeJob("comfortable", 1, 224, 224, 5000, arrival_ns=0)
    assert p.is_guarded(comfortable, 0) is False


def test_dacc_update_changes_safe_prediction():
    p = planner()
    job = EncodeJob("job", 1, 224, 224, 1000, arrival_ns=0)
    before = p.safe_latency(job, 1.0)
    p.update(p.performance_model.predict(job.patches, 1.0), before * 2)
    assert p.safe_latency(job, 1.0) > before
