"""换表 A/B 与批次审计两个脚本的契约测试。

这两个脚本产出的是"两条结论撤回"与"19.2% 吞吐变化"这类判断，口径错了不会报错、
只会让结论站不住。三条必须钉住的语义：

1. **Welch t 必须用不等方差公式**。两个批次的样本量与方差都不等（例如
   `overload_baselines/serial_fcfs` sd=4.5 对 `fixed_baselines/dacc` sd=2.4），
   用合并方差会系统性高估显著性。
2. **两两比较的 t 必须"每个批次内部"算**，不能用两个策略各自"换表效应"的 t 冒充——
   那是另一个问题，会让"哪些结论稳定"整个判错（**本脚本的初版就犯了这个错**）。
3. **deadline 缩放必须真的把中位数压到目标值**。原负载 deadline 中位 102.5 ms，
   而实测执行只有 3–22 ms；不压下来，`deadline_min` 对所有请求都返回最小配额，
   两套表根本不可能分歧，整个 Q4 实验就是空的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="module")
def audit():
    import analyze_baseline_batches as m
    return m


@pytest.fixture(scope="module")
def swap():
    import experiment_table_swap_ab as m
    return m


# --------------------------------------------------------------------------- #
# welch
# --------------------------------------------------------------------------- #
def test_welch_zero_for_identical_samples(audit):
    a = [10.0, 11.0, 12.0, 10.5]
    assert audit.welch(a, list(a)) == pytest.approx(0.0)


def test_welch_grows_with_separation(audit):
    a = [10.0, 10.2, 9.8, 10.1]
    near = audit.welch(a, [10.5, 10.6, 10.4, 10.7])
    far = audit.welch(a, [20.0, 20.2, 19.8, 20.1])
    assert far > near


def test_welch_matches_textbook_formula(audit):
    """直接对照 Welch 的定义式。"""
    import statistics as st
    a = [100.0, 102.0, 98.0, 101.0]
    b = [110.0, 105.0, 115.0, 108.0]
    expected = (abs(st.fmean(a) - st.fmean(b))
                / (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5)
    assert audit.welch(a, b) == pytest.approx(expected, rel=1e-12)


def test_welch_equals_pooled_when_sample_sizes_are_equal(audit):
    """**样本量相等时，Welch t 与合并方差 t 在代数上恒等。**

    本项目的批次都是 n=5 对 n=5，所以这个区别在这里**不影响任何结论**——
    记录成测试是为了避免以后误以为"换了检验就能改变显著性"。

    推导：Welch 分母 = sqrt(va/n + vb/n) = sqrt((va+vb)/n)；
    合并分母 = sp·sqrt(2/n)，而 sp² = (va+vb)/2，故 = sqrt((va+vb)/n)。两者相同。
    """
    import statistics as st
    a = [100.0, 100.1, 99.9, 100.0]
    b = [90.0, 110.0, 85.0, 120.0]
    n = len(a)
    assert len(b) == n
    va, vb = st.variance(a), st.variance(b)
    pooled_sd = (((n - 1) * va + (n - 1) * vb) / (2 * n - 2)) ** 0.5
    pooled = abs(st.fmean(a) - st.fmean(b)) / (pooled_sd * (2 / n) ** 0.5)
    assert audit.welch(a, b) == pytest.approx(pooled, rel=1e-9)


def test_welch_differs_from_pooled_when_sample_sizes_differ(audit):
    """样本量不等时两者才分开——这才是必须用 Welch 的场景。"""
    import statistics as st
    a = [100.0, 100.1, 99.9, 100.0]                    # n=4
    b = [90.0, 110.0, 85.0, 120.0, 95.0, 105.0, 88.0, 112.0]   # n=8
    na, nb = len(a), len(b)
    va, vb = st.variance(a), st.variance(b)
    pooled_sd = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    pooled = abs(st.fmean(a) - st.fmean(b)) / (pooled_sd * (1 / na + 1 / nb) ** 0.5)
    assert not abs(audit.welch(a, b) - pooled) < 1e-3, "样本量不等时两种口径必须分开"


def test_welch_handles_degenerate_input(audit):
    import math
    assert math.isnan(audit.welch([1.0], [2.0]))       # 单样本无法估方差
    assert math.isnan(audit.welch([], [1.0, 2.0]))


# --------------------------------------------------------------------------- #
# metric_of
# --------------------------------------------------------------------------- #
def test_metric_of_unwraps_summary_dict(audit):
    assert audit.metric_of({"throughput_rps": 12.5}, "throughput_rps") == 12.5
    assert audit.metric_of({"total_latency_ms": {"mean": 7.5, "p99": 9.0}},
                           "total_latency_ms") == 7.5


# --------------------------------------------------------------------------- #
# make_tight_workload
# --------------------------------------------------------------------------- #
def test_tight_workload_hits_target_median(swap, tmp_path):
    """缩放后中位数必须落在目标值上——否则 deadline_min 不会触发，实验是空的。"""
    out = tmp_path / "tight.jsonl"
    meta = swap.make_tight_workload(12.0, out)
    assert meta["median_after_ms"] == pytest.approx(12.0, rel=1e-6)
    assert meta["median_before_ms"] > 90.0, "原始负载的 deadline 应当很松"
    assert meta["scale"] < 0.2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == meta["count"]
    import statistics as st
    assert st.median(r["deadline_ms"] for r in rows) == pytest.approx(12.0, rel=1e-6)


def test_tight_workload_never_goes_non_positive(swap, tmp_path):
    """缩放到极端值时 must not 产生 0 或负 deadline（那会让 choose_quota 拿到负的剩余时间）。"""
    out = tmp_path / "tiny.jsonl"
    swap.make_tight_workload(0.01, out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(r["deadline_ms"] >= 1.0 for r in rows)


def test_tight_workload_preserves_arrivals_and_sizes(swap, tmp_path):
    """只改 deadline，到达时刻与尺寸必须原样保留——否则负载形态变了，对照就不成立。"""
    out = tmp_path / "tight.jsonl"
    swap.make_tight_workload(12.0, out)
    src = [json.loads(line) for line in
           swap.WORKLOAD_SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["arrival_ms"] for r in src] == [r["arrival_ms"] for r in dst]
    assert [r["width"] for r in src] == [r["width"] for r in dst]


# --------------------------------------------------------------------------- #
# write_config
# --------------------------------------------------------------------------- #
def test_write_config_only_swaps_table_and_enforces_no_fallback(swap, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "model": {"name": "openai/clip-vit-base-patch32"},
        "scheduler": {"policy": "edf_size"},
        "executor": {"streams": 2, "resource_backend": "proxy", "allow_proxy_fallback": True},
        "profiling": {"table_path": "data/profiles/default.csv"},
    }, sort_keys=False), encoding="utf-8")
    out = swap.write_config(base, Path("/tmp/other.csv"), tmp_path / "sub" / "c.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["profiling"]["table_path"] == "/tmp/other.csv"
    # 真实掩码实验禁止静默回退——这是 project 纪律，脚本必须强制
    assert data["executor"]["allow_proxy_fallback"] is False
    assert data["executor"]["resource_backend"] == "libsmctrl"
    # 其余字段不得被改动
    assert data["scheduler"]["policy"] == "edf_size"
    assert data["executor"]["streams"] == 2
