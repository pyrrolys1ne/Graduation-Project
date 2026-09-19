from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_repeated_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_repeated_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sentinel_variant_forces_single_request_execution():
    source = {
        "policy": "edf_size",
        "streams": 8,
        "adaptive": True,
        "batch": 4,
        "delay_ms": 5.0,
        "backend": "proxy",
    }
    sentinel = MODULE.sentinel_variant(source)
    assert sentinel["policy"] == "multistream_fcfs"
    assert sentinel["streams"] == 1
    assert sentinel["adaptive"] is False
    assert "batch" not in sentinel
    assert "delay_ms" not in sentinel
    assert sentinel["backend"] == "proxy"
    assert source["adaptive"] is True


def test_contamination_check_uses_per_size_worst_median(tmp_path):
    path = tmp_path / "sentinel.jsonl"
    rows = [
        {"width": 224, "height": 224, "execution_ms": 5.0, "predicted_ms": 5.0},
        {"width": 224, "height": 224, "execution_ms": 6.0, "predicted_ms": 5.0},
        {"width": 672, "height": 672, "execution_ms": 20.0, "predicted_ms": 10.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = MODULE.contamination_check(path)
    assert result["per_size"] == {"224x224": 1.1, "672x672": 2.0}
    assert result["worst_ratio"] == 2.0
    assert result["contaminated"] is True
