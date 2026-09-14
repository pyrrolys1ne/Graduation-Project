from pathlib import Path

import pytest

from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel
from encoder_sched.scheduler import Scheduler


TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def make_scheduler(policy: str) -> Scheduler:
    return Scheduler(policy, PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), 5.0)


def test_fcfs_orders_by_arrival():
    scheduler = make_scheduler("serial_fcfs")
    late = EncodeJob("late", 1, 224, 224, 100, arrival_ns=20)
    early = EncodeJob("early", 1, 224, 224, 100, arrival_ns=10)
    assert sorted([late, early], key=scheduler.rank_key) == [early, late]


def test_edf_orders_by_deadline_then_priority():
    scheduler = make_scheduler("edf")
    normal = EncodeJob("normal", 1, 224, 224, 100, priority=0, arrival_ns=1_000_000)
    urgent = EncodeJob("urgent", 1, 224, 224, 50, priority=0, arrival_ns=1_000_000)
    assert sorted([normal, urgent], key=scheduler.rank_key) == [urgent, normal]


def test_edf_size_prefers_short_job_in_same_deadline_bucket():
    scheduler = make_scheduler("edf_size")
    small = EncodeJob("small", 1, 224, 224, 100, arrival_ns=1_000_000)
    large = EncodeJob("large", 1, 672, 672, 100, arrival_ns=1_000_000)
    scheduler.prepare(small, small.arrival_ns)
    scheduler.prepare(large, large.arrival_ns)
    assert sorted([large, small], key=scheduler.rank_key) == [small, large]


def test_edf_size_honors_priority_within_deadline_bucket():
    scheduler = make_scheduler("edf_size")
    small = EncodeJob("small", 1, 224, 224, 100, priority=0, arrival_ns=1_000_000)
    important = EncodeJob("important", 1, 672, 672, 100, priority=10, arrival_ns=1_000_000)
    scheduler.prepare(small, small.arrival_ns)
    scheduler.prepare(important, important.arrival_ns)
    assert sorted([small, important], key=scheduler.rank_key) == [important, small]


def test_quota_is_smallest_level_predicted_to_meet_deadline():
    """deadline_min（默认）：取满足截止期的最小配额；都不满足时退回最大配额。"""
    scheduler = make_scheduler("edf_size")
    model = PerformanceModel(TABLE)
    # 用表里的实际值构造边界，避免写死数值（剖析表会被重新生成）。
    small_patches = 49
    cheapest = model.predict(small_patches, 0.25)
    dearest = model.predict(small_patches, 1.0)

    generous = EncodeJob("generous", 1, 224, 224, cheapest * 10, arrival_ns=0)
    assert scheduler.choose_quota(generous, generous.arrival_ns) == 0.25

    impossible = EncodeJob("impossible", 1, 224, 224, dearest / 10, arrival_ns=0)
    assert scheduler.choose_quota(impossible, impossible.arrival_ns) == 1.0


def test_quota_policy_full_always_uses_max():
    """full 策略：始终用满配额，不因截止期宽松而主动变慢。

    动机是 deadline_min 的目标函数值得质疑——它把资源守恒当成收益，但 GPU 利用率
    只有 25–33%，省下的 TPC 若无人使用就毫无价值，请求本身却会慢数倍。
    """
    scheduler = Scheduler("edf_size", PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), 5.0,
                          quota_policy="full")
    generous = EncodeJob("generous", 1, 672, 672, 100_000, arrival_ns=0)
    assert scheduler.choose_quota(generous, generous.arrival_ns) == 1.0


def test_unknown_quota_policy_is_rejected():
    with pytest.raises(ValueError, match="quota_policy"):
        Scheduler("edf_size", PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), 5.0,
                  quota_policy="whatever")


def test_dacc_rank_key_differs_from_edf_size():
    """dacc 的排队键必须走自己的 urgency 分支。

    修复前 `rank_key()` 里没有 dacc 分支，而对应的代码被误粘到了 `prepare()`
    末尾的 `return` 之后（死代码），导致 dacc 实际使用 edf_size 的排序键——
    urgency 排序从未生效。
    """
    edf_size = make_scheduler("edf_size")
    dacc = make_scheduler("dacc")
    jobs = [
        EncodeJob("a", 1, 224, 224, 100, arrival_ns=1_000_000),
        EncodeJob("b", 1, 672, 672, 300, arrival_ns=1_000_000),
        EncodeJob("c", 1, 336, 336, 80, arrival_ns=1_000_000),
    ]
    for job in jobs:
        edf_size.prepare(job, job.arrival_ns)
        dacc.prepare(job, job.arrival_ns)
    assert [dacc.rank_key(j) for j in jobs] != [edf_size.rank_key(j) for j in jobs]


def test_prepare_has_no_unreachable_statements():
    """`prepare()` 曾有一条 `return job` 之后的死代码（误粘的 rank key）。

    这类错误不会报错，只会让功能静默失效，所以用 AST 直接钉住。
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(Scheduler.prepare))
    body = ast.parse(source).body[0].body
    for index, statement in enumerate(body):
        if any(isinstance(previous, ast.Return) for previous in body[:index]):
            raise AssertionError(f"prepare() 第 {index + 1} 条语句在 return 之后，是不可达代码")
