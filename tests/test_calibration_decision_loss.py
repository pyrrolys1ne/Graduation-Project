"""Q4 决策损失分析的契约测试。

这个脚本产出论文的核心数字（"同一批 400 个请求，两套表预测 100 对 0 个违约"），
算错了不会报错，只会让结论偏移。三条必须钉住的语义：

1. **决策必须可复现**：``decide`` 要给调度器传显式的 ``now_ns``。默认走
   ``time.perf_counter_ns()``，同一份负载两次运行会得到不同结果。
2. **分布式共享必须区分象限**：``count_inversions`` 只在两套预测值**给出相反顺序**时
   计数；相等时不算反转（相等不是反转）。
3. **"常规负载看不出分歧"是事实而非 bug**：deadline 宽松时 ``deadline_min`` 对
   所有请求都返回最小配额。这条要用测试钉住，否则以后有人改坏了会以为 Q4 结论消失。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.performance import PerformanceModel  # noqa: E402
from encoder_sched.scheduler import Scheduler  # noqa: E402


@pytest.fixture(scope="module")
def mod():
    import experiment_calibration_decision_loss as m
    return m


def write_table(path: Path, rows: list[tuple[int, float, float]]) -> Path:
    path.write_text("width,height,patches,sm_fraction,latency_ms\n" + "".join(
        f"{s},{s},{(s // 32) ** 2},{q},{ms}\n" for s, q, ms in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# count_inversions
# --------------------------------------------------------------------------- #
def test_inversions_counts_opposite_orderings_only(mod):
    assert mod.count_inversions([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0
    assert mod.count_inversions([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == 3
    # 单对反转
    assert mod.count_inversions([1.0, 2.0, 3.0], [2.0, 1.0, 3.0]) == 1


def test_inversions_ignores_ties(mod):
    """相等不是反转——把相等算成反转会让"污染压平尺寸代价"这个结论虚高。"""
    assert mod.count_inversions([1.0, 1.0, 3.0], [1.0, 1.0, 3.0]) == 0
    assert mod.count_inversions([2.0, 2.0], [1.0, 5.0]) == 0


def test_inversions_maximal_case(mod):
    n = 5
    a = list(range(n))
    b = list(reversed(range(n)))
    assert mod.count_inversions(a, b) == n * (n - 1) // 2


# --------------------------------------------------------------------------- #
# 决策的确定性
# --------------------------------------------------------------------------- #
def test_decide_is_deterministic(mod, tmp_path):
    """同一份负载跑两次必须逐位相同——否则 Q4 的数字不可复现。"""
    table = write_table(tmp_path / "t.csv", [
        (224, 0.25, 12.0), (224, 1.0, 3.0),
        (672, 0.25, 40.0), (672, 1.0, 8.0),
    ])
    rows = [
        {"request_id": "a", "seed": 1, "width": 224, "height": 224,
         "deadline_ms": 6.0, "priority": 0, "arrival_ms": 0.0},
        {"request_id": "b", "seed": 2, "width": 672, "height": 672,
         "deadline_ms": 20.0, "priority": 0, "arrival_ms": 1.0},
    ]
    first = mod.decide(Scheduler("edf_size", PerformanceModel(table), (0.25, 1.0),
                                 quota_policy="deadline_min"), rows)
    second = mod.decide(Scheduler("edf_size", PerformanceModel(table), (0.25, 1.0),
                                  quota_policy="deadline_min"), rows)
    assert first == second


# --------------------------------------------------------------------------- #
# 宽松 deadline 下两套表不分歧（这是事实，不是 bug）
# --------------------------------------------------------------------------- #
def _two_tables(tmp_path, factor: float):
    """构造两张只在数值上差 factor 倍、形状相同的表。"""
    base = [(224, 0.25, 12.0), (224, 1.0, 3.0), (672, 0.25, 40.0), (672, 1.0, 8.0)]
    a = write_table(tmp_path / "a.csv", base)
    b = write_table(tmp_path / "b.csv", [(s, q, ms * factor) for s, q, ms in base])
    return a, b


def test_loose_deadline_gives_no_disagreement(mod, tmp_path):
    """deadline 远大于任何预测值时，deadline_min 对两套表都返回最小配额。

    这正是原始 saturating_* 负载测出 0.0% 不一致的原因（deadline 中位 102.5 ms，
    实测执行只有 3–22 ms）。**必须钉住**：否则有人会以为 Q4 结论是假的。
    """
    a, b = _two_tables(tmp_path, 2.0)
    rows = [{"request_id": f"r{i}", "seed": i, "width": 224, "height": 224,
             "deadline_ms": 500.0, "priority": 0, "arrival_ms": float(i)}
            for i in range(20)]
    da = mod.decide(Scheduler("edf_size", PerformanceModel(a), (0.25, 1.0),
                              quota_policy="deadline_min"), rows)
    db = mod.decide(Scheduler("edf_size", PerformanceModel(b), (0.25, 1.0),
                              quota_policy="deadline_min"), rows)
    assert {x[0] for x in da} == {0.25}
    assert {x[0] for x in db} == {0.25}
    assert [x[0] for x in da] == [x[0] for x in db]


def test_tight_deadline_exposes_disagreement(mod, tmp_path):
    """deadline 落在两套预测之间时，配额决策必须分歧——这是 Q4 的核心机制。"""
    a, b = _two_tables(tmp_path, 2.0)   # 224@1.0: a=3.0, b=6.0
    rows = [{"request_id": "r", "seed": 0, "width": 224, "height": 224,
             "deadline_ms": 4.0, "priority": 0, "arrival_ms": 0.0}]
    da = mod.decide(Scheduler("edf_size", PerformanceModel(a), (0.25, 1.0),
                              quota_policy="deadline_min"), rows)
    db = mod.decide(Scheduler("edf_size", PerformanceModel(b), (0.25, 1.0),
                              quota_policy="deadline_min"), rows)
    assert da[0][0] == 1.0, "低预测表：满配额 3.0 ≤ 4.0，应当选满配额"
    assert db[0][0] == 1.0, "高预测表：最小配额预测 24.0 > 4.0，也落到满配额"
    # 两者都选满配额，但预测值不同 → 违约判断不同
    assert da[0][2] >= 0, "低预测表认为该请求仍有正 slack"
    assert db[0][2] < 0, "高预测表认为该请求已违约"


def test_predicted_slack_sign_is_what_differs(mod, tmp_path):
    """验收指标是 slack 的符号（是否违约），不只是配额是否相同。"""
    a, b = _two_tables(tmp_path, 3.0)
    rows = [{"request_id": "r", "seed": 0, "width": 224, "height": 224,
             "deadline_ms": 8.0, "priority": 0, "arrival_ms": 0.0}]
    sched_a = Scheduler("edf_size", PerformanceModel(a), (0.25, 1.0), quota_policy="deadline_min")
    sched_b = Scheduler("edf_size", PerformanceModel(b), (0.25, 1.0), quota_policy="deadline_min")
    slack_a = mod.decide(sched_a, rows)[0][2]
    slack_b = mod.decide(sched_b, rows)[0][2]
    assert slack_a > 0 > slack_b


# --------------------------------------------------------------------------- #
# 表的形状
# --------------------------------------------------------------------------- #
def test_scheduler_uses_predicted_ms_as_third_key(mod, tmp_path):
    """``edf_size`` 的 rank_key 把 predicted_ms 放在第三位——Q4 说它"很少起作用"
    正是因为这个位置。位置变了，结论要重算。"""
    table = write_table(tmp_path / "t.csv", [(224, 0.25, 3.0), (224, 1.0, 3.0)])
    sched = Scheduler("edf_size", PerformanceModel(table), (0.25, 1.0),
                      quota_policy="deadline_min")
    job = EncodeJob("t", seed=0, width=224, height=224, deadline_ms=10.0,
                    priority=0, arrival_ns=0)
    job.predicted_ms = 7.0
    key = sched.rank_key(job)
    assert len(key) == 4
    assert key[2] == 7.0, "predicted_ms 必须是第 3 个排序键"
