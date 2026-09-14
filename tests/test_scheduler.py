from pathlib import Path

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
    scheduler = make_scheduler("edf_size")
    relaxed = EncodeJob("relaxed", 1, 224, 224, 100, arrival_ns=1_000_000)
    tight = EncodeJob("tight", 1, 224, 224, 5, arrival_ns=1_000_000)
    assert scheduler.choose_quota(relaxed, relaxed.arrival_ns) == 0.25
    assert scheduler.choose_quota(tight, tight.arrival_ns) == 1.0
