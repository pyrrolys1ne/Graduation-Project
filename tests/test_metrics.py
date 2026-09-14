import time

from encoder_sched.metrics import MetricsStore
from encoder_sched.models import EncodeJob, JobState


def test_metrics_summary_and_slo(tmp_path):
    metrics = MetricsStore(tmp_path / "requests.jsonl")
    now = time.perf_counter_ns()
    job = EncodeJob("one", 1, 224, 224, 10, arrival_ns=now)
    job.started_ns = now + 2_000_000
    job.finished_ns = now + 12_000_000
    job.execution_ms = 10.0
    job.state = JobState.COMPLETED
    metrics.record_submission()
    metrics.record(job)
    summary = metrics.summary()
    assert summary["completed"] == 1
    assert summary["throughput_rps"] > 0
    assert summary["total_latency_ms"]["p99"] == 12.0
    assert summary["slo_violation_rate"] == 1.0
    assert (tmp_path / "requests.jsonl").read_text(encoding="utf-8").count("\n") == 1
    metrics.reset()
    assert metrics.summary()["completed"] == 0
