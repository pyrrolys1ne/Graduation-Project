"""性能模型测试。

刻意**不写死延迟数值**：剖析表会被重新生成（换机器、换掩码后端、重新测量都会
改变数值）。写死数字的测试每次重测都会碎，而碎的测试很快会被人随手改掉，
最终失去意义。这里断言的是模型的契约。
"""

import csv
from pathlib import Path

import pytest

from encoder_sched.performance import PerformanceModel

TABLE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "default.csv"


@pytest.fixture(scope="module")
def rows():
    with TABLE.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def model():
    return PerformanceModel(TABLE)


def test_quota_levels_are_sorted_and_complete(model):
    assert model.quota_levels == (0.25, 0.5, 0.75, 1.0)


def test_predict_reproduces_table_grid_points(model, rows):
    """表里已有的点必须被原样复现——这是模型最基本的契约。"""
    for row in rows:
        patches, fraction, latency = int(row["patches"]), float(row["sm_fraction"]), float(row["latency_ms"])
        assert model.predict(patches, fraction) == pytest.approx(latency, abs=1e-6), (
            f"patches={patches} sm={fraction} 应等于表值 {latency}"
        )


def test_interpolation_stays_between_neighbours(model, rows):
    """插值结果必须落在相邻两个已知点之间，不能外插出离谱的值。"""
    by_size: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        by_size.setdefault(int(row["patches"]), []).append((float(row["sm_fraction"]), float(row["latency_ms"])))
    for patches, points in by_size.items():
        points.sort()
        (q_lo, t_lo), (q_hi, t_hi) = points[0], points[1]
        mid = model.predict(patches, (q_lo + q_hi) / 2)
        assert min(t_lo, t_hi) - 1e-6 <= mid <= max(t_lo, t_hi) + 1e-6


def test_more_quota_never_predicts_slower(model, rows):
    """配额越高，预测延迟不应变大。

    剖析脚本会做单调化（见 scripts/profile_encoder.py 的 enforce_monotonic），
    因为实测中位数会因噪声违反单调性，而非单调的表会让调度决策静默失真。
    """
    by_size: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        by_size.setdefault(int(row["patches"]), []).append((float(row["sm_fraction"]), float(row["latency_ms"])))
    for patches, points in by_size.items():
        points.sort()
        values = [t for _, t in points]
        assert values == sorted(values, reverse=True), f"patches={patches} 的配额—延迟非单调递减: {points}"


def test_quota_outside_range_is_clipped(model):
    """配额超出表范围时按边界裁剪，不应外插或抛错。"""
    patches = 49
    assert model.predict(patches, 0.01) == pytest.approx(model.predict(patches, 0.25))
    assert model.predict(patches, 5.0) == pytest.approx(model.predict(patches, 1.0))


def test_table_without_required_columns_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("patches,latency_ms\n49,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少列"):
        PerformanceModel(bad)


def test_table_with_non_positive_latency_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("patches,sm_fraction,latency_ms\n49,1.0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="正延迟"):
        PerformanceModel(bad)
