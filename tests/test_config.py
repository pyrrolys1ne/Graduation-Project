"""配置加载与校验测试。

重点保护 dacc_overrides：消融实验全靠它，字段名拼错必须报错，
否则会静默地"什么都没改"，产出一批看似有效实则无效的消融结果。
"""

import pytest
import yaml

from encoder_sched.config import load_config


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
        path = write(tmp_path, quota_levels=[0.0, 1.5])
        load_config(path)
