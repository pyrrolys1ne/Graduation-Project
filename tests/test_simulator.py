from pathlib import Path

from encoder_sched.simulator import simulate


TABLE = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"


def test_simulator_replays_arrivals_and_returns_metrics():
    rows = [
        {"arrival_ms": 0, "request_id": "a", "seed": 1, "width": 224, "height": 224, "deadline_ms": 1000, "priority": 0},
        {"arrival_ms": 1, "request_id": "b", "seed": 2, "width": 224, "height": 224, "deadline_ms": 1000, "priority": 0},
    ]
    result = simulate(rows, "dacc", TABLE, streams=2)
    assert result.completed == 2
    assert result.throughput_rps > 0
    assert result.mean_latency_ms >= 0
