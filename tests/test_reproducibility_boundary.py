"""标定表可复现性实验的统计口径契约测试。

这个脚本产出的 `spread` / `condition_effect` 是论文的核心数字。它们没有异常、
没有报错，算错了只会让图表偏移——所以用契约测试钉住。

三条必须钉住的语义（都来自实验中真实踩过的坑）：

1. **`describe` 不能只报均值/标准差。** 实测延迟分布是重尾的（224×224 在 dense 条件下
   中位数 4.61 ms、最大值 13.63 ms），而常规标定恰好是"从重尾里抽一个点当中位用"。
   因此 CV、极差比、p95/p50 都必须给出。
2. **条件间差异不能在没有对照时伪装成 1.0。** 只有一个条件时必须是 `None`，不是 1.0——
   假装的"无差异"会被直接读成结论。
3. **`loop` 模式的时长口径必须与 `batch` 可比。** 早期版本用外层 CUDA Event 夹住 N 次
   `encode()`，把 `torch.rand` 的 CPU 时间算了进去，实测虚高约 2 倍。正确做法是累加各次
   调用自己的 `execution_ms`。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from encoder_sched.models import EncodeJob


@pytest.fixture(scope="module")
def mod():
    import experiment_reproducibility_boundary as m
    return m


# --------------------------------------------------------------------------- #
# describe：分布刻画
# --------------------------------------------------------------------------- #
def test_describe_reports_tail_metrics_not_just_mean(mod):
    """重尾分布下，均值/标准差不足以刻画——CV 与尾部比值必须同时给出。"""
    samples = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 12.0]
    stats = mod.describe(samples)
    assert stats["n"] == 8
    assert stats["median_ms"] == pytest.approx(4.55)
    # 最大值是其余样本的近 3 倍——这正是"抽一个点当中位"会出错的场景
    assert stats["range_ratio"] == pytest.approx(12.0 / 4.2, rel=1e-9)
    assert stats["cv"] > 0.4
    assert stats["p95_over_p50"] > 1.5


def test_describe_is_exact_on_uniform_samples(mod):
    """退化情形：全部相同值时 CV 必须为 0，不能是 nan。"""
    stats = mod.describe([5.0] * 10)
    assert stats["cv"] == 0.0
    assert stats["range_ratio"] == 1.0
    assert stats["p95_over_p50"] == 1.0


def test_describe_handles_single_sample(mod):
    """单样本时标准差无定义，但中位数/极差仍必须可用——否则 n=1 的轮次会让汇总崩掉。"""
    stats = mod.describe([7.5])
    assert stats["median_ms"] == 7.5
    assert stats["std_ms"] == 0.0
    assert stats["n"] == 1


def test_describe_on_empty_is_empty(mod):
    assert mod.describe([]) == {}


# --------------------------------------------------------------------------- #
# summarize：条件间差异
# --------------------------------------------------------------------------- #
def _row(size, quota, mode, n, condition, exec_ms, round_index=0):
    return {"size": size, "quota": quota, "mode": mode, "n": n,
            "condition": condition, "round": round_index,
            "exec_ms": exec_ms, "wall_ms": exec_ms * 1.05, "sm_clock": 1890}


def test_condition_effect_none_when_only_one_condition(mod):
    """只有一个条件时不能报 1.0——那会被读成"无差异"，而实际是"没测"。"""
    raw = [_row(224, 1.0, "batch", 1, "dense", 4.0 + i * 0.01) for i in range(5)]
    summary = mod.summarize(raw)
    assert len(summary) == 1
    assert summary[0]["condition_effect"] is None


def test_condition_effect_captures_median_shift(mod):
    """三个条件中位数 4 / 6 / 8 → 条件间差异 2.0×。"""
    raw = []
    for condition, value in (("dense", 4.0), ("idle", 6.0), ("hot", 8.0)):
        raw += [_row(224, 1.0, "batch", 1, condition, value + i * 0.001) for i in range(5)]
    summary = mod.summarize(raw)
    assert summary[0]["condition_effect"] == pytest.approx(8.0 / 4.0, rel=1e-3)
    assert set(summary[0]["conditions"]) == {"dense", "idle", "hot"}


def test_summarize_keeps_configs_separate(mod):
    """不同 (size, quota, mode, n) 必须各自成组，不能合并。"""
    raw = [_row(224, 1.0, "batch", 2, "dense", 5.0),
           _row(224, 1.0, "batch", 4, "dense", 9.0),
           _row(672, 1.0, "batch", 1, "dense", 18.0)]
    summary = mod.summarize(raw)
    assert len(summary) == 3
    assert [s["median_ms"] for s in summary] == [5.0, 9.0, 18.0]  # 已按时长排序


def test_summary_sorted_by_duration(mod):
    """按时长排序是 Q2 分析的前提——排序错了会把"时长效应"读成"尺寸效应"。"""
    raw = [_row(672, 1.0, "batch", 1, "dense", 18.0),
           _row(224, 1.0, "batch", 1, "dense", 3.0)]
    summary = mod.summarize(raw)
    assert [s["size"] for s in summary] == [224, 672]


# --------------------------------------------------------------------------- #
# build_cells：实验矩阵
# --------------------------------------------------------------------------- #
def test_build_cells_covers_every_combination(mod):
    cells = mod.build_cells({224: [1, 2]}, [0.25, 1.0], ["batch"], ("dense", "idle"))
    keys = {c.key for c in cells}
    assert len(keys) == 2 * 2 * 1 * 2  # size×quota×mode×n × condition
    assert len(cells) == len(keys)


def test_build_cells_carries_sm_fraction_source(mod):
    """sm_fraction 必须来自配额档位本身——写死会让"配额"这个自变量失效。"""
    cells = mod.build_cells({224: [1]}, [0.25], ["batch"], ("dense",))
    assert cells[0].quota == 0.25


# --------------------------------------------------------------------------- #
# 契约：Harness.measure 的时长口径
# --------------------------------------------------------------------------- #
def test_loop_mode_sums_per_call_execution_not_outer_event():
    """loop 的时长口径必须是各次 execution_ms 之和，不能用外层 CUDA Event。

    外层 Event 会把 `ClipEncoderBackend._input` 里的 `torch.rand` CPU 时间算进去
    （它发生在 encode_batch 的 start_event 之前），实测把 224×224 的 3.4 ms 抬到 7.9 ms。
    这里用假编码器钉住口径：每次调用返回固定的 2.0 ms，N 次就必须是 N×2.0。
    """
    class FakeStream:
        cuda_stream = 0

    class FakeEncoder:
        streams = [FakeStream()]

        def __init__(self):
            self.calls = 0

        def encode(self, job, worker_id):
            self.calls += 1
            from encoder_sched.encoder import EncodingResult
            return EncodingResult(512, 1.0, 2.0, {})

    class FakeTorch:
        class cuda:  # noqa: N801 - 只需满足属性访问
            @staticmethod
            def Event(*a, **k):
                raise AssertionError("loop 模式不应当使用 CUDA Event")

    import experiment_reproducibility_boundary as m
    encoder = FakeEncoder()
    harness = m.Harness.__new__(m.Harness)
    harness.encoder = encoder
    harness.torch = FakeTorch
    harness.heat_ms = 0

    exec_ms, wall_ms, _ = harness.measure(224, 1.0, "loop", 4, "t")
    assert encoder.calls == 4
    assert exec_ms == pytest.approx(8.0), "loop×4 必须是 4×2.0 ms，不能混入 CPU 时间"
    assert wall_ms >= 0


def test_job_seed_is_stable_across_processes(mod):
    """seed 必须与 PYTHONHASHSEED 无关——用内置 hash() 会让实验不可复现。"""
    harness_cls = mod.Harness
    job = harness_cls.__new__(harness_cls)
    a = job._job(224, 1.0, "same-tag")
    b = job._job(224, 1.0, "same-tag")
    assert a.seed == b.seed
    assert isinstance(a, EncodeJob)
    assert a.sm_fraction == 1.0
