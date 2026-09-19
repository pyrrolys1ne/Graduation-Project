"""配置加载与校验测试。

重点保护 dacc_overrides：消融实验全靠它，字段名拼错必须报错，
否则会静默地"什么都没改"，产出一批看似有效实则无效的消融结果。
"""

from pathlib import Path

import pytest
import yaml

from encoder_sched.config import load_config

REPO_ROOT = Path(__file__).parents[1]


def write(tmp_path, **scheduler):
    data = {"scheduler": scheduler} if scheduler else {}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_defaults_are_usable(tmp_path):
    config = load_config(write(tmp_path))
    assert config.scheduler.policy == "edf_size"
    assert config.metrics.gpu_sample_interval_s > 0


def test_valid_dacc_overrides_accepted(tmp_path):
    config = load_config(write(tmp_path, policy="dacc", dacc_overrides={"w_complementarity": 0.0, "window": 16}))
    assert config.scheduler.dacc_overrides == {"w_complementarity": 0.0, "window": 16}


def test_unknown_override_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="未知字段"):
        load_config(write(tmp_path, policy="dacc", dacc_overrides={"w_typo": 1.0}))


def test_override_value_type_is_rejected(tmp_path):
    """dataclasses.replace 不校验类型；值写错必须在加载配置时就报错。"""
    with pytest.raises(ValueError, match="应为数值"):
        load_config(write(tmp_path, policy="dacc", dacc_overrides={"window": "many"}))


@pytest.mark.parametrize("value", [0, -0.5])
def test_non_positive_gpu_interval_rejected(tmp_path, value):
    path = write(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["metrics"] = {"gpu_sample_interval_s": value}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="gpu_sample_interval_s"):
        load_config(path)


def test_streams_and_quota_validation(tmp_path):
    with pytest.raises(ValueError, match="streams"):
        path = write(tmp_path)
        data = {"executor": {"streams": 0}}
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        load_config(path)
    with pytest.raises(ValueError, match="quota_levels"):
        load_config(write(tmp_path, quota_levels=[0.0, 1.5]))


def _with_executor(tmp_path, **executor):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"executor": executor}, allow_unicode=True), encoding="utf-8")
    return path


def test_adaptive_concurrency_defaults_off(tmp_path):
    """默认必须关闭——本项目此前的全部历史结果都跑在定死并发度下，改默认会让它们不可复现。"""
    config = load_config(write(tmp_path))
    assert config.executor.adaptive_concurrency is False
    assert config.executor.concurrency.min_concurrency >= 1


def test_adaptive_concurrency_parsed_from_nested_mapping(tmp_path):
    """``executor.concurrency`` 是嵌套 dataclass。

    若直接 ``ExecutorConfig(**data)``，YAML 里的 dict 会被原样塞进去，
    直到访问 ``config.executor.concurrency.max_concurrency`` 才炸——离出错地点很远。
    这里钉死"加载时就展开成 ConcurrencyConfig"。
    """
    config = load_config(
        _with_executor(
            tmp_path,
            streams=2,
            adaptive_concurrency=True,
            concurrency={"small_mix_concurrency": 5, "large_mix_concurrency": 3},
        )
    )
    assert config.executor.adaptive_concurrency is True
    assert config.executor.concurrency.small_mix_concurrency == 5
    assert config.executor.concurrency.large_mix_concurrency == 3
    # 未覆盖的字段应保留默认值，而不是丢失
    assert config.executor.concurrency.max_concurrency >= 5


def test_unknown_concurrency_key_is_rejected(tmp_path):
    """拼错字段名必须报错，否则控制器会静默地什么都不改。"""
    with pytest.raises(ValueError, match="未知字段"):
        load_config(_with_executor(tmp_path, adaptive_concurrency=True, concurrency={"max_concurr": 4}))


def test_non_mapping_concurrency_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="必须是映射"):
        load_config(_with_executor(tmp_path, adaptive_concurrency=True, concurrency=4))


def test_adaptive_config_example_loads():
    """仓库里给出的示例配置必须真的能加载——否则文档与代码会各说各话。"""
    config = load_config(REPO_ROOT / "config.adaptive.yaml")
    assert config.executor.adaptive_concurrency is True
    assert config.executor.concurrency.enable_drift_fallback is True
