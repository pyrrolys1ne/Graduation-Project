import time

from encoder_sched.metrics import MetricsStore, client_side_summary
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


def test_server_summary_declares_its_scope(tmp_path):
    """服务端汇总必须自报口径，否则下游无法判断手里的数字是哪一侧。"""
    metrics = MetricsStore(tmp_path / "requests.jsonl")
    assert metrics.summary()["latency_scope"] == "server"


def test_client_side_summary_separates_scopes_and_flags_pool_queueing():
    """两侧口径必须分别命名，且连接池排队要能被判出来。

    数据取自本课题的真实形态：客户端延迟约服务端 total 的 15–34 倍，
    其中一部分是固定往返（第一条请求没等过连接），一部分是连接池等待。
    """
    rows = [
        # 服务端 10 ms、客户端 200 ms：往返 + 无排队
        {"client_latency_ms": 200.0, "total_ms": 10.0, "dispatch_lateness_ms": 1.0, "deadline_ms": 100},
        # 服务端仍只 12 ms，客户端却 900 ms → 多出来的 700 ms 是连接池等待
        {"client_latency_ms": 900.0, "total_ms": 12.0, "dispatch_lateness_ms": 2.0, "deadline_ms": 100},
    ]
    # 连接上限 8 < 需求：客户端可能被池卡住，此时 overhead 超出基准的部分算池等待
    result = client_side_summary(rows, concurrency=8, pool_can_throttle=True)
    assert result["scope"] == "client"
    assert result["concurrency"] == 8
    assert result["pool_binding"] == "capped_by_client_pool"
    assert result["latency_ms"]["mean"] == 550.0
    assert result["overhead_ms"]["mean"] == 539.0  # (190 + 888) / 2
    assert result["pool_wait_ms"]["p50"] == 349.0  # 基线 190，第二条多等 698/2
    assert result["pool_wait_ms"]["p50"] > 0
    assert result["pool_wait_flagged"] is True
    # 客户端看到的超时比例按 client_latency_ms 判定：两条都已违约
    assert result["slo_violation_rate"] == 1.0
    assert result["client_queueing_flagged"] is True


def test_client_side_summary_not_flagged_when_server_is_the_bottleneck():
    """服务端自己排队（客户端按时发出）时不应误判为连接池问题。"""
    rows = [
        {"client_latency_ms": 100.0, "total_ms": 99.0, "dispatch_lateness_ms": 0.5, "deadline_ms": 500},
        {"client_latency_ms": 120.0, "total_ms": 118.0, "dispatch_lateness_ms": 0.8, "deadline_ms": 500},
    ]
    result = client_side_summary(rows, concurrency=8, pool_can_throttle=True)
    assert result["overhead_ms"]["mean"] < 2.0
    assert result["client_queueing_flagged"] is False
    assert result["slo_violation_rate"] == 0.0


def test_client_side_summary_does_not_flag_plain_http_round_trip():
    """纯 HTTP 往返不能被当成排队。

    这来自一次真实冒烟：Fake 编码器的服务端 total 不足 1 ms，而客户端看到约 4.4 ms
    （全是往返），若直接用 overhead 占比判定就会立刻误报。
    """
    rows = [
        {"client_latency_ms": 4.4, "total_ms": 0.7, "dispatch_lateness_ms": 0.9, "deadline_ms": 1000},
        {"client_latency_ms": 6.9, "total_ms": 0.8, "dispatch_lateness_ms": 1.4, "deadline_ms": 1000},
    ]
    # 开环：连接上限 ≥ 请求数，客户端不可能因等连接而推迟下发
    result = client_side_summary(rows, concurrency=4, pool_can_throttle=False)
    assert result["pool_binding"] == "not_a_constraint"
    # 往返约 3.7–6.1 ms，全部是网络与序列化，没有连接池等待的成分
    assert result["pool_wait_ms"]["p50"] < 1.5
    assert result["client_queueing_flagged"] is False
    assert result["event_loop_backlog_flagged"] is False


def test_open_loop_overhead_is_not_attributed_to_the_client_pool():
    """开环下 overhead 再大也不算客户端排队，但数值仍然保留。

    这来自 2026-09-22 的开环实测：连接上限 = 请求数（400）后，服务端排队达到
    1.4–3.2 s，客户端与服务端之间仍有几百毫秒差距——那来自连接建立、服务端接收
    与请求解析，是前端开销，不是客户端在等空闲连接。若照旧判成"客户端排队"，
    这批本来有效的对照会被闸门误杀。
    """
    rows = [
        {"client_latency_ms": 1500.0, "total_ms": 1386.0, "dispatch_lateness_ms": 1.0, "deadline_ms": 500},
        {"client_latency_ms": 2000.0, "total_ms": 1400.0, "dispatch_lateness_ms": 1.2, "deadline_ms": 500},
    ]
    result = client_side_summary(rows, concurrency=400, pool_can_throttle=False)
    assert result["pool_wait_ms"]["p95"] > 5.0  # 前端开销确实存在，如实报告
    assert result["pool_wait_flagged"] is False
    assert result["client_queueing_flagged"] is False


def test_open_loop_still_flags_event_loop_backlog():
    """开环也拦不住"客户端根本没按时发出"——这一条与连接上限无关。"""
    rows = [
        {"client_latency_ms": 100.0, "total_ms": 90.0, "dispatch_lateness_ms": 200.0, "deadline_ms": 500},
        {"client_latency_ms": 110.0, "total_ms": 95.0, "dispatch_lateness_ms": 250.0, "deadline_ms": 500},
    ]
    result = client_side_summary(rows, concurrency=400, pool_can_throttle=False)
    assert result["pool_wait_flagged"] is False
    assert result["event_loop_backlog_flagged"] is True
    assert result["client_queueing_flagged"] is True
