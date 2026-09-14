"""工作负载生成测试。

背景：`generate` 里累加到达时间用的局部变量一度被改名为 `current_ms`，但写入每行
的仍是函数参数 `arrival_ms`，导致所有请求的到达时间都等于常量。这类错误不会报错，
只会让整个工作负载的到达结构失效——所以用不变量把它钉住。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_workload import generate  # noqa: E402


def rate(rows):
    return len(rows) / (rows[-1]["arrival_ms"] / 1000)


def test_arrivals_are_monotonic_and_distinct():
    rows = generate(50, "mixed", seed=1, slo="normal")
    arrivals = [r["arrival_ms"] for r in rows]
    assert arrivals == sorted(arrivals), "到达时间必须单调不减"
    assert len(set(arrivals)) > len(arrivals) * 0.9, "到达时间不应退化成同一个常量"


@pytest.mark.parametrize("pattern", ["mixed", "uniform", "burst"])
def test_arrival_ms_controls_rate(pattern):
    """显式到达间隔必须真正改变到达率，否则无法构造饱和负载。"""
    slow = generate(200, pattern, seed=7, slo="normal", arrival_ms=20.0)
    fast = generate(200, pattern, seed=7, slo="normal", arrival_ms=5.0)
    if pattern == "burst":
        # burst 保持自身突发结构，不随 arrival_ms 缩放。
        assert slow[-1]["arrival_ms"] == fast[-1]["arrival_ms"]
    else:
        assert rate(fast) > rate(slow) * 2, f"{pattern}: 到达率未随 arrival_ms 变化"


def test_default_rate_matches_documented_value():
    """默认 20ms 约等于 50 req/s，低于本机服务率，GPU 不会饱和。"""
    rows = generate(300, "mixed", seed=20260902, slo="normal")
    assert 40 < rate(rows) < 65, f"默认到达率偏离预期: {rate(rows):.1f} req/s"


def test_slo_factor_changes_deadlines():
    tight = generate(10, "uniform", seed=3, slo="tight")
    loose = generate(10, "uniform", seed=3, slo="loose")
    assert loose[0]["deadline_ms"] > tight[0]["deadline_ms"]


def test_cli_reports_rate(tmp_path):
    """CLI 必须把实际到达率打印出来——这是判断负载是否会饱和的唯一线索。"""
    out = tmp_path / "w.jsonl"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_workload.py"),
         "--count", "100", "--arrival-ms", "10", "--output", str(out)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    assert "req/s" in result.stdout
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 100
    assert rows[-1]["arrival_ms"] > 100, "到达跨度不应塌缩"
