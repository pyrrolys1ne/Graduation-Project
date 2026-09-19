"""batch 运行恢复脚本的契约测试。

这个脚本的结论是"被丢弃的那批数据无效，丢弃是对的"——判定错了会让整个批处理实验的
可信度被误判。三条必须钉住的语义：

1. **运行令牌解析**：`request_id` 形如 `mixed-20260902-00000-run-1789393697724403734-0`。
   漏掉正则会把一次运行切成很多段，或者把 warmup 也算进来。
2. **会话切分按时间间隔**：两个会话之间隔约 10 分钟，阈值定小了会把一个会话切碎。
3. **有效性判据是"批大小 ≈ 1"**：批处理变体上恒为 1.000 说明组批没生效。
   这条判据不能误伤 `spatial_2stream`（它不组批，字段为空）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="module")
def mod():
    import recover_batch_runs as m
    return m


def write_log(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def make_record(run_token: int, index: int, batch_size: int = 1, total_ms: float = 10.0,
                slo: bool = False) -> dict:
    return {
        "request_id": f"mixed-20260902-{index:05d}-run-{run_token}-{index}",
        "queue_ms": 1.0, "execution_ms": total_ms - 1.0, "total_ms": total_ms,
        "slo_violated": slo, "metadata": {"batch_size": batch_size},
    }


# --------------------------------------------------------------------------- #
# 运行令牌解析
# --------------------------------------------------------------------------- #
def test_split_runs_groups_by_token(mod, tmp_path):
    records = []
    for token in (1000, 2000, 3000):
        records += [make_record(token, i) for i in range(4)]
    runs = mod.split_runs(write_log(tmp_path / "l.jsonl", records))
    assert sorted(runs) == [1000, 2000, 3000]
    assert all(len(v) == 4 for v in runs.values())


def test_split_runs_ignores_warmup_and_foreign_ids(mod, tmp_path):
    """warmup 与没有运行令牌的记录必须被排除——否则每次运行会多出记录，
    算出的均值与 summary.json 对不上（那正是本脚本要复现的量）。"""
    records = [
        {"request_id": "warmup-1789393697771181928-0", "queue_ms": 1.0,
         "execution_ms": 1.0, "total_ms": 2.0, "slo_violated": False, "metadata": {}},
        make_record(5000, 0),
        make_record(5000, 1),
    ]
    runs = mod.split_runs(write_log(tmp_path / "l.jsonl", records))
    assert list(runs) == [5000]
    assert len(runs[5000]) == 2


def test_split_runs_handles_empty_file(mod, tmp_path):
    assert mod.split_runs(write_log(tmp_path / "l.jsonl", [])) == {}


# --------------------------------------------------------------------------- #
# 会话切分
# --------------------------------------------------------------------------- #
def test_split_sessions_splits_on_large_gap(mod):
    """会话间隔实测约 6×10^11 ns（10 分钟），阈值 6×10^10（60 s）能正确分开。"""
    sessions = mod.split_sessions([100, 200, 300, 700_000_000_000, 700_000_000_100])
    assert len(sessions) == 2
    assert sessions[0] == [100, 200, 300]
    assert sessions[1] == [700_000_000_000, 700_000_000_100]


def test_split_sessions_keeps_close_runs_together(mod):
    """同一会话内相邻运行间隔只有几秒，不能被切开。"""
    sessions = mod.split_sessions([0, 7_000_000_000, 14_000_000_000, 21_000_000_000])
    assert len(sessions) == 1


def test_split_sessions_sorts_unsorted_input(mod):
    sessions = mod.split_sessions([300, 100, 200])
    assert sessions == [[100, 200, 300]]


# --------------------------------------------------------------------------- #
# 有效性判定
# --------------------------------------------------------------------------- #
def test_verdict_rejects_batch_size_one(mod):
    stats = {"mean_batch_size": 1.0}
    assert "无效" in mod.verdict("batch4_d0", stats, [1.2, 2.4])


def test_verdict_accepts_real_batching(mod):
    stats = {"mean_batch_size": 2.372}
    assert "有效" in mod.verdict("batch4_d20", stats, [1.0])


def test_verdict_does_not_apply_to_non_batching_variant(mod):
    """`spatial_2stream` 不组批，batch_size 字段为空——判据不能误伤它。"""
    stats = {"mean_batch_size": None}
    verdict = mod.verdict("spatial_2stream", stats, [1.0, 2.4])
    assert "不组批" in verdict
    assert "无效" not in verdict


def test_verdict_flags_anomalously_small_batch(mod):
    """批大小 1.05 不是"无效"（确实组了批），但若同实验其它会话都是 2.4，就值得标记。"""
    assert "可疑" in mod.verdict("batch4_d0", {"mean_batch_size": 1.05}, [2.4, 3.0])


def test_verdict_boundary_is_tight(mod):
    """1.02 与 1.0 必须被区分——判据太松会把真正无效的会话放过去。"""
    assert "无效" in mod.verdict("batch4_d0", {"mean_batch_size": 1.01}, [2.0])
    assert "无效" not in mod.verdict("batch4_d0", {"mean_batch_size": 1.05}, [1.1])


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def test_describe_computes_percentiles_and_slo(mod):
    rows = [make_record(1, i, total_ms=float(i), slo=(i >= 8)) for i in range(10)]
    st = mod.describe(rows)
    assert st["n"] == 10
    assert st["total_mean_ms"] == pytest.approx(4.5)
    assert st["total_p50_ms"] == pytest.approx(4.0, abs=1.0)
    assert st["slo_violation_rate"] == pytest.approx(0.2)
    assert st["mean_batch_size"] == 1.0


def test_describe_batch_size_none_when_absent(mod):
    rows = [{"total_ms": 1.0, "queue_ms": 0.1, "execution_ms": 0.9,
             "slo_violated": False, "metadata": {}} for _ in range(3)]
    assert mod.describe(rows)["mean_batch_size"] is None
