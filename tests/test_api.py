from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from encoder_sched.api import create_app


def write_config(tmp_path: Path) -> Path:
    source_table = Path(__file__).parents[1] / "data" / "profiles" / "default.csv"
    table = tmp_path / "profiles.csv"
    table.write_text(source_table.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
model:
  name: openai/clip-vit-base-patch32
  dtype: float32
scheduler:
  policy: edf_size
  deadline_tie_ms: 5
  quota_levels: [0.25, 0.5, 0.75, 1.0]
executor:
  streams: 2
  resource_backend: proxy
  allow_proxy_fallback: true
  libsmctrl_adapter: ""
profiling:
  table_path: "{table.as_posix()}"
logging:
  output_dir: "{(tmp_path / 'results').as_posix()}"
  request_log: requests.jsonl
seed: 1
""".strip(),
        encoding="utf-8",
    )
    return config


def test_encode_job_lookup_metrics_and_duplicate(tmp_path):
    app = create_app(write_config(tmp_path), fake=True)
    payload = {
        "request_id": "api-1",
        "seed": 42,
        "width": 224,
        "height": 224,
        "deadline_ms": 500,
        "priority": 1,
    }
    with TestClient(app) as client:
        response = client.post("/v1/encode", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert result["embedding_dim"] == 512
        assert result["metadata"]["resource"]["enforced"] is False
        assert client.get("/v1/jobs/api-1").status_code == 200
        assert client.get("/v1/jobs/missing").status_code == 404
        assert client.post("/v1/encode", json=payload).status_code == 409
        summary = client.get("/metrics/summary").json()
        assert summary["completed"] == 1
        assert client.post("/metrics/reset").status_code == 204
        assert client.get("/metrics/summary").json()["completed"] == 0


def test_invalid_request_is_rejected(tmp_path):
    app = create_app(write_config(tmp_path), fake=True)
    with TestClient(app) as client:
        response = client.post(
            "/v1/encode",
            json={"request_id": "bad", "width": 32, "height": 224, "deadline_ms": 100},
        )
        assert response.status_code == 422


def _make_libsmctrl_config(tmp_path):
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "resource_backend: proxy", "resource_backend: libsmctrl"
    ).replace(
        'libsmctrl_adapter: ""', 'libsmctrl_adapter: "encoder_sched.libsmctrl_adapter:apply_quota"'
    )
    config.write_text(text, encoding="utf-8")
    return config


def test_libsmctrl_with_fake_encoder_is_rejected_at_startup(tmp_path):
    """FakeEncoder 传的是 worker id，把它当 CUDA stream 句柄会写坏内存，必须启动即拒绝。"""
    app = create_app(_make_libsmctrl_config(tmp_path), fake=True)
    with pytest.raises(ValueError, match="FakeEncoder"):
        with TestClient(app):
            pass


def test_config_with_removed_device_field_gives_migration_hint(tmp_path):
    """`model.device` 已移除。旧配置必须得到明确提示，而不是难懂的 TypeError。

    真实 CLIP 后端只支持 CUDA；曾支持的 `device: cpu` 分支从未被真正执行过
    （测试夹具写过它，但所有调用都走 fake=True）。
    """
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "  name: openai/clip-vit-base-patch32\n", "  name: openai/clip-vit-base-patch32\n  device: cpu\n"
    )
    config.write_text(text, encoding="utf-8")
    app = create_app(config, fake=True)
    with pytest.raises(ValueError, match="model.device 已移除"):
        with TestClient(app):
            pass


def test_model_config_has_no_device_field():
    """CPU 推理路径已删除：`ModelConfig` 不应再有 `device` 字段。

    这条是契约检查而非源码扫描——扫描源码会把 docstring 里解释性的字样也当成
    "残留分支"，测试会碎得毫无价值。
    """
    from dataclasses import fields

    from encoder_sched.config import ModelConfig

    names = {item.name for item in fields(ModelConfig)}
    assert "device" not in names, f"ModelConfig 仍暴露 device 字段: {sorted(names)}"


def test_health_reports_resource_capability(tmp_path):
    app = create_app(write_config(tmp_path), fake=True)
    with TestClient(app) as client:
        resource = client.get("/health").json()["resource"]
        assert resource["effective_backend"] == "proxy"
        assert resource["fallback_active"] is False
        assert resource["enforces_sm_partition"] is False


def test_dacc_policy_runs_through_fastapi(tmp_path):
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8").replace("policy: edf_size", "policy: dacc")
    config.write_text(text, encoding="utf-8")
    app = create_app(config, fake=True)
    with TestClient(app) as client:
        responses = [
            client.post("/v1/encode", json={"request_id": "dacc-a", "width": 224, "height": 224, "deadline_ms": 1000}),
            client.post("/v1/encode", json={"request_id": "dacc-b", "width": 224, "height": 224, "deadline_ms": 1000}),
        ]
        assert all(response.status_code == 200 for response in responses)
        assert all(response.json()["metadata"]["schedule_mode"] in {"single", "pair"} for response in responses)
