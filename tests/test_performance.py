from pathlib import Path

import pytest

from encoder_sched.performance import PerformanceModel


TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def test_predict_exact_and_interpolated_values():
    model = PerformanceModel(TABLE)
    assert model.predict(49, 1.0) == pytest.approx(6.0)
    assert model.predict(49, 0.625) == pytest.approx(8.0)
    assert 13.0 < model.predict(150, 1.0) < 22.0


def test_quota_levels_are_sorted():
    assert PerformanceModel(TABLE).quota_levels == (0.25, 0.5, 0.75, 1.0)

