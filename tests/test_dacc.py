from pathlib import Path

from encoder_sched.dacc import DaccConfig, DaccPlanner
from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel


TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def planner() -> DaccPlanner:
    return DaccPlanner(PerformanceModel(TABLE), (0.25, 0.5, 0.75, 1.0), DaccConfig(window=8))


def test_dacc_prefers_pair_for_small_complementary_requests():
    p = planner()
    left = EncodeJob("left", 1, 224, 224, 1000, arrival_ns=0)
    right = EncodeJob("right", 1, 224, 224, 1000, arrival_ns=0)
    plan = p.plan([left, right], 0)
    assert plan.mode == "pair"
    assert len(plan.jobs) == 2


def test_dacc_guard_prefers_urgent_request_over_pair():
    p = planner()
    urgent = EncodeJob("urgent", 1, 672, 672, 1, arrival_ns=0)
    relaxed = EncodeJob("relaxed", 1, 224, 224, 1000, arrival_ns=0)
    plan = p.plan([relaxed, urgent], 0)
    assert plan.jobs[0].request_id == "urgent"


def test_dacc_update_changes_safe_prediction():
    p = planner()
    job = EncodeJob("job", 1, 224, 224, 1000, arrival_ns=0)
    before = p.safe_latency(job, 1.0)
    p.update(p.performance_model.predict(job.patches, 1.0), before * 2)
    assert p.safe_latency(job, 1.0) > before
