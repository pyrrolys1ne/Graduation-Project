"""负载生成侧的契约：分片合并与"连接池是否成为瓶颈"的判定。

这两个判定的共同点是**它们错的时候都不会报错**——只会让一整批实验读起来像"客户端排队"，
从而抑制策略判定、或者更糟：把一个客户端瓶颈当成服务端结论。本项目已经因此作废过
一整批实验（§21.2），所以这两条要有测试钉住。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows(n: int, latency_ms: float = 5.0) -> list[dict]:
    return [
        {"client_latency_ms": latency_ms, "total_ms": latency_ms - 1.0,
         "dispatch_lateness_ms": 1.0, "slo_violated": False}
        for _ in range(n)
    ]


def test_shard_merge_counts_pool_capacity_per_shard_not_per_batch():
    """开环判据必须按**每片**算，不能按合并后的总记录数。

    4 个分片、每片 225 个请求、每片连接上限 225：每个分片都不可能因为等空闲连接而
    推迟下发，因此连接池**不是**约束。若按总数算（225 < 900）就会误判成
    "capped_by_client_pool"，进而把这一批判定为客户端瓶颈、抑制策略结论。
    """
    summary = MODULE.merge_shard_summary({}, _rows(900), concurrency=225, shards=4)
    assert summary["client_scope"]["pool_binding"] == "not_a_constraint"
    assert summary["client_scope"]["client_queueing_flagged"] is False
    assert summary["shards"] == 4


def test_shard_merge_still_flags_a_genuinely_capped_pool():
    """反向：每片连接上限低于该片请求数时，必须判成受限。"""
    summary = MODULE.merge_shard_summary({}, _rows(900), concurrency=100, shards=4)
    assert summary["client_scope"]["pool_binding"] == "capped_by_client_pool"


def test_shard_merge_keeps_concurrency_for_the_record():
    summary = MODULE.merge_shard_summary({}, _rows(80), concurrency=20, shards=4)
    assert summary["client_scope"]["concurrency"] == 20


def test_shard_merge_handles_empty_rows_without_dividing_by_zero():
    summary = MODULE.merge_shard_summary({}, [], concurrency=100, shards=4)
    assert summary["shards"] == 4
