from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_repeated_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_repeated_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _summary(**overrides) -> dict:
    """构造一份真实形态的实验汇总：服务端口径 + 客户端口径并存。"""
    data = {
        "throughput_rps": 100.0,
        "total_latency_ms": {"mean": 50.0, "p50": 45.0, "p95": 90.0, "p99": 120.0},
        "queue_latency_ms": {"mean": 40.0},
        "execution_latency_ms": {"mean": 10.0},
        "slo_violation_rate": 0.2,
        "gpu_utilization_percent": 30.0,
        "gpu_samples": 20,
        "completed": 400,
        "failed": 0,
        "client_scope": {
            "scope": "client",
            "concurrency": 8,
            "latency_ms": {"mean": 200.0, "p50": 190.0, "p95": 400.0, "p99": 500.0},
            "overhead_ms": {"mean": 150.0, "p50": 145.0, "p95": 350.0, "p99": 400.0},
            "pool_wait_ms": {"mean": 10.0, "p50": 5.0, "p95": 60.0, "p99": 90.0},
            "dispatch_lateness_ms": {"mean": 40.0, "p50": 30.0, "p95": 150.0, "p99": 200.0},
            "pool_wait_excess_p95_ms": 55.0,
            "queueing_share_p95": 0.11,
            "client_queueing_flagged": True,
            "slo_violation_rate": 0.6,
        },
    }
    data.update(overrides)
    return data


def test_flatten_keeps_server_and_client_calibers_separate():
    """两个口径必须各有各的字段，且不能再用含混的 total_* 命名。"""
    row = MODULE.flatten(_summary())
    assert row["server_total_p99_ms"] == 120.0
    assert row["client_total_p99_ms"] == 500.0
    assert row["client_overhead_p95_ms"] == 350.0
    assert row["client_pool_wait_p95_ms"] == 60.0
    assert row["client_dispatch_lateness_p95_ms"] == 150.0
    assert row["client_slo_violation_rate"] == 0.6
    assert row["client_queueing_flagged"] is True
    # 含混命名一旦回归，混用就会重新出现在下游结论里
    for ambiguous in ("total_p99_ms", "total_mean_ms", "queue_mean_ms", "execution_mean_ms"):
        assert ambiguous not in row


def test_flatten_refuses_summary_without_client_scope():
    """缺客户端块时必须报错，而不是静默退回单口径。"""
    data = _summary()
    del data["client_scope"]
    with pytest.raises(KeyError, match="client_scope"):
        MODULE.flatten(data)


def test_metrics_tuple_is_explicitly_scoped():
    """所有延迟指标都必须带 server_ / client_ 前缀。"""
    latency_metrics = [m for m in MODULE.METRICS if m.endswith("_ms")]
    assert latency_metrics, "延迟指标不能为空"
    assert all(m.startswith(("server_", "client_")) for m in latency_metrics), latency_metrics


def test_resolve_client_concurrency_zero_means_open_loop(tmp_path):
    """0 = 开环：连接上限取负载条数，避免客户端把吞吐钉住。"""
    workload = tmp_path / "w.jsonl"
    workload.write_text("\n".join(json.dumps({"request_id": str(i)}) for i in range(7)) + "\n",
                        encoding="utf-8")
    assert MODULE.resolve_client_concurrency(0, workload) == 7
    # 显式给了正数就照用，但那样一旦被客户端卡住会抑制判定
    assert MODULE.resolve_client_concurrency(8, workload) == 8


# ── 方法对照组（`--group graph`）的契约 ────────────────────────────────────────
#
# 这一组的跑道必须一次搭对：一批正式对照是 5 轮 × 3 臂 = 15 次服务运行，
# 配置写错会在每一次启动时以同样的方式失败，而"跑完再看"的代价是一个多小时。


def _base_config() -> dict:
    import yaml

    return yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))


def test_build_config_enables_graph_pipeline_and_validates(tmp_path):
    from encoder_sched.config import load_config

    variant = MODULE.GROUPS["graph"]["graph_slots1"]
    path = MODULE.build_config(_base_config(), variant, tmp_path)
    # 生成出来的配置必须能被真实加载器接受——这是这一组唯一不会骗人的契约。
    config = load_config(path)
    assert config.executor.graph_pipeline is True
    assert config.executor.graph.slots == 1
    assert config.executor.graph.prep_threads == 2


def test_build_config_rejects_graph_with_libsmctrl(tmp_path):
    variant = {"policy": "multistream_fcfs", "backend": "libsmctrl",
               "graph": {"slots": 1}}
    with pytest.raises(ValueError, match="libsmctrl"):
        MODULE.build_config(_base_config(), variant, tmp_path)


def test_build_config_rejects_graph_with_batching(tmp_path):
    variant = {"policy": "multistream_fcfs", "batch": 4, "graph": {"slots": 1}}
    with pytest.raises(ValueError, match="批处理"):
        MODULE.build_config(_base_config(), variant, tmp_path)


def test_build_config_rejects_graph_with_adaptive_concurrency(tmp_path):
    variant = {"policy": "multistream_fcfs", "adaptive": True, "graph": {"slots": 1}}
    with pytest.raises(ValueError, match="adaptive_concurrency"):
        MODULE.build_config(_base_config(), variant, tmp_path)


def test_build_config_rejects_slots_above_streams(tmp_path):
    """槽位数大于流数时两个槽位会挤在同一条流上，并发被静默压回流数。"""
    variant = {"policy": "multistream_fcfs", "streams": 1, "graph": {"slots": 2}}
    with pytest.raises(ValueError, match="slots"):
        MODULE.build_config(_base_config(), variant, tmp_path)


def test_graph_group_arms_are_well_formed(tmp_path):
    """各臂只差执行机制：策略必须完全一致，否则差异无法归因。"""
    from encoder_sched.config import load_config

    variants = MODULE.GROUPS["graph"]
    assert set(variants) == {"eager_serial", "eager_2stream", "graph_slots1", "graph_batch"}
    assert {v["policy"] for v in variants.values()} == {"multistream_fcfs"}
    assert variants["eager_serial"]["streams"] == 1
    assert variants["eager_2stream"]["streams"] == 2
    # 两个参照系里不能混入任何组批或自适应并发——那会让"vs 固定资源分配"这个对照失效
    for name in ("eager_serial", "eager_2stream"):
        assert "batch" not in variants[name] and "adaptive" not in variants[name]
        assert "graph" not in variants[name]
    # 每一臂都要真的能生成配置，并且被**真实加载器**接受
    for name, arm in variants.items():
        path = MODULE.build_config(_base_config(), arm, tmp_path)
        config = load_config(path)
        assert config.executor.graph_pipeline is ("graph" in arm)


def test_graph_batch_arm_batches_small_sizes_only(tmp_path):
    """批上限必须按尺寸给，且只对小尺寸开。

    §24.2 实测单流 B=4 相对 B=1 的收益是 224 2.42× / 336 1.79× / 448 1.38× / 672 1.16×，
    而批越大单批墙钟越长（672 的 B=4 是 19.2 ms 对 5.4 ms）。因此大尺寸不该批；
    若有人把 672 也开成批，这个测试会失败并提醒他重读那一节。
    """
    from encoder_sched.config import load_config

    arm = MODULE.GROUPS["graph"]["graph_batch"]
    config = load_config(MODULE.build_config(_base_config(), arm, tmp_path))
    batch = {f"{w}x{h}": count for w, h, count in config.executor.graph.batch_by_size}
    assert batch["224x224"] > 1 and batch["336x336"] > 1
    assert batch.get("672x672", 1) == 1


def test_graph_group_default_workload_is_an_overload_tier():
    """`graph` 组的默认负载必须是**过载档**，不能是接近容量档。

    两条独立理由，任何一条都足够：
    ① §3 的教训：到达率低于服务容量时所有臂的吞吐都被到达率钉死，**无法区分任何策略**；
    ② §28.3 的实测：`graph_batch` 臂只在有积压时才凑得成批——档位 1 下平均批大小
       恒为 1.00，成批臂与不成批臂完全一样，那个"两臂相同"会被误读成"成批无效"。
    因此默认负载必须显式指定，且不能退回历史上那个"到达率等于服务率"的负载。
    """
    default = MODULE.GROUP_DEFAULT_WORKLOAD["graph"]
    assert default != MODULE.DEFAULT_WORKLOAD
    assert "280rps" in default, "graph 组默认负载应取过载档（见 §28.3）"
    assert (Path(__file__).parents[1] / default).is_file()
