"""`scripts/profile_encoder.py` 的记录格式契约测试。

这个脚本产出的是**调度决策的输入表**。它的失败模式不是报错，而是**丢信息**：
汇总表只留中位数与标准差，而违约概率模型需要的是分布本身。因此用契约测试钉住
四条语义：

1. **原始样本必须逐条落盘**，不能只留中位数——否则尾部/违约概率永远估不出来。
2. **样本必须带会话标识与执行机制标签**。同一个 (尺寸, 配额) 单元在图回放路径与
   eager 路径上不是同一个函数（共驻代价 1.0× 对 2–3.5×），混在一起就是错的。
3. **样本文件是追加写的**。跨会话离散度只能靠累积多个会话得到；截断写会把上一个
   会话的数据静默抹掉。
4. **单调化必须无条件写 `latency_ms_measured`**。否则"这一行没被抬升"与"这一列
   不存在"无法区分，原始值就永久丢了。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

import profile_encoder as m  # noqa: E402


class _FakeResult:
    def __init__(self, execution_ms: float):
        self.execution_ms = execution_ms


class _FakeEncoder:
    """延迟随配额升高而下降，且带一点确定性噪声。不需要 GPU。"""

    def __init__(self, *_args, **_kwargs):
        self.calls = 0

    def encode(self, job, _worker_id):  # noqa: ANN001
        self.calls += 1
        base = {224: 4.0, 336: 5.0, 448: 6.0, 672: 10.0}[job.width]
        return _FakeResult(base / job.sm_fraction + 0.01 * self.calls)


class _FakeResource:
    name = "proxy"
    enforces_sm_partition = False


def _run(tmp_path: Path, monkeypatch, **overrides) -> tuple[Path, Path, list[str]]:
    csv_out = tmp_path / "measured.csv"
    jsonl_out = tmp_path / "samples.jsonl"
    argv = ["profile_encoder", "--output", str(csv_out), "--samples-out", str(jsonl_out),
            "--sizes", "224", "--repeats", "3", "--warmup", "1"]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]

    monkeypatch.setattr(m, "ClipEncoderBackend", _FakeEncoder)
    monkeypatch.setattr(m, "create_resource_backend", lambda *a, **k: _FakeResource())
    monkeypatch.setattr(sys, "argv", argv)
    m.main()

    lines = jsonl_out.read_text(encoding="utf-8").strip().split("\n")
    return csv_out, jsonl_out, lines


def test_raw_samples_are_persisted_not_just_the_median(tmp_path, monkeypatch):
    """契约 1：每个 post-warmup 样本都要有一行，不能只留中位数。"""
    _, _, lines = _run(tmp_path, monkeypatch)
    assert len(lines) == 3, f"应有 3 条样本，实际 {len(lines)}"
    records = [json.loads(line) for line in lines]
    assert [r["sample_index"] for r in records] == [0, 1, 2]
    assert all(r["execution_ms"] > 0 for r in records)


def test_samples_carry_session_and_mechanism(tmp_path, monkeypatch):
    """契约 2：会话标识与执行机制标签必须同时出现在 CSV 与 JSONL 里。"""
    csv_out, _, lines = _run(tmp_path, monkeypatch, session_id="sess-A")
    records = [json.loads(line) for line in lines]
    assert all(r["session_id"] == "sess-A" for r in records)
    assert all(r["mechanism"].startswith(("eager/", "graph/")) for r in records)

    with csv_out.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["session_id"] == "sess-A"
    assert rows[0]["mechanism"] == records[0]["mechanism"]


def test_samples_file_is_appended_across_sessions(tmp_path, monkeypatch):
    """契约 3：第二次运行是追加，不能把上一会话的数据截断掉。"""
    _, jsonl_out, first = _run(tmp_path, monkeypatch, session_id="sess-A")
    _, _, second = _run(tmp_path, monkeypatch, session_id="sess-B")
    assert len(second) == len(first) * 2, "第二次运行应追加而不是覆盖"
    ids = [json.loads(line)["session_id"] for line in second]
    assert ids[: len(first)] == ["sess-A"] * len(first)
    assert ids[len(first):] == ["sess-B"] * len(first)


def test_mechanism_distinguishes_graph_from_eager():
    """契约 2 的补充：两条路径不能折叠成同一个标签。"""

    class _Exec:
        def __init__(self, graph: bool, streams: int, slots: int):
            self.graph_pipeline = graph
            self.streams = streams
            self.graph = type("G", (), {"slots": slots})()

    class _Cfg:
        def __init__(self, exec_):
            self.executor = exec_

    assert m.describe_mechanism(_Cfg(_Exec(False, 2, 1))) == "eager/par=2"
    assert m.describe_mechanism(_Cfg(_Exec(True, 2, 1))) == "graph/par=1"
    assert m.describe_mechanism(_Cfg(_Exec(False, 2, 1))) != m.describe_mechanism(_Cfg(_Exec(True, 2, 1)))


def test_monotonic_writes_measured_column_for_every_row():
    """契约 4：无论是否被抬升，原始中位数都要留下。"""
    rows = [
        {"width": 224, "height": 224, "sm_fraction": 1.0, "latency_ms": 4.0},
        {"width": 224, "height": 224, "sm_fraction": 0.5, "latency_ms": 3.0},  # 非单调
        {"width": 224, "height": 224, "sm_fraction": 0.25, "latency_ms": 9.0},
    ]
    out = {float(r["sm_fraction"]): r for r in m.enforce_monotonic(rows)}
    assert all("latency_ms_measured" in r and "monotonic_adjusted" in r for r in out.values())
    assert out[0.5]["latency_ms_measured"] == 3.0    # 原始值被保留
    assert out[0.5]["latency_ms"] == 4.0             # 但预测值被抬到相邻高位
    assert out[0.5]["monotonic_adjusted"] is True
    assert out[1.0]["monotonic_adjusted"] is False
    assert out[1.0]["latency_ms_measured"] == 4.0


def test_tail_floor_constant_is_the_distribution_free_bound():
    """`TAIL_SAMPLE_FLOOR` 是序统计量界：n=30 时 ⌈31×0.9⌉=28，可估 10% 分位。"""
    import math
    assert math.ceil((m.TAIL_SAMPLE_FLOOR + 1) * 0.9) <= m.TAIL_SAMPLE_FLOOR


def test_warns_when_repeats_below_tail_floor(tmp_path, monkeypatch, capsys):
    """样本不足时必须发声——静默产出不可用于尾部的表是本项目已经吃过的亏。"""
    _run(tmp_path, monkeypatch)
    captured = capsys.readouterr().out
    assert "低于估计 10% 分位数所需" in captured


@pytest.mark.parametrize("graph,expected", [(False, "eager/par=7"), (True, "graph/par=7")])
def test_graph_slots_uses_slot_count_as_parallelism(graph, expected):
    class _Exec:
        def __init__(self):
            self.graph_pipeline = graph
            self.streams = 7
            self.graph = type("G", (), {"slots": 7})()

    class _Cfg:
        executor = _Exec()

    assert m.describe_mechanism(_Cfg()) == expected
