"""图回放流水线的**配置层**契约。

这里只测配置校验与嵌套段解析，不碰 GPU：`GraphReplayRuntime.__init__` 要捕获 CUDA
Graph，在本机以外的环境跑不了。配置层的三条互斥与一条上界关系是**静默失效风险最高**
的地方——比如 slots 大于 streams 时，两个回放槽位会挤在同一条流上，回放并发被压回
流数，而吞吐只是"看起来低一点"，不看日志根本发现不了。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from encoder_sched.config import load_config


def write_config(tmp_path: Path, executor: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"name": "openai/clip-vit-base-patch32", "dtype": "float16"},
        "scheduler": {"policy": "edf_size"},
        "executor": executor,
        "profiling": {"table_path": "data/profiles/default.csv"},
    }, allow_unicode=True), encoding="utf-8")
    return path


def base_executor(**overrides) -> dict:
    data = {
        "streams": 2,
        "resource_backend": "proxy",
        "graph_pipeline": True,
        "graph": {"slots": 2, "prep_threads": 2},
    }
    data.update(overrides)
    return data


def test_slots_defaults_to_one():
    """`slots` 的默认值必须是 1——**这是安全默认，不是性能默认**。

    `slots > 1` 时"多张 CUDA Graph 同时在 GPU 上执行"会偶发挂死（所有回放线程卡在
    `stream.synchronize()`、GPU 永不完结）：预取结构下 `slots=2` 实测 6 次运行 4 次挂，
    而 `slots=1` 27 次运行 0 次挂。把默认值改回 2 会让每一次正式对照都有丢数据的风险，
    而且失败形态是"某一轮没有数据"而不是报错。详见 `docs/实验记录.md` §25。
    """
    from encoder_sched.graph_runtime import GraphRuntimeConfig

    assert GraphRuntimeConfig().slots == 1


def test_defaults_are_off_and_graph_config_has_sizes():
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert config.executor.graph_pipeline is False
    # 默认尺寸集合必须非空：留空表示"从剖析表推导"，那是 api.build_service 的职责，
    # 而 dataclass 的默认值不能依赖运行时文件。
    assert config.executor.graph.sizes == ((224, 224), (336, 336), (448, 448), (672, 672))


def test_graph_section_parses_and_coerces_sizes(tmp_path: Path):
    path = write_config(tmp_path, base_executor(
        graph={"slots": 2, "prep_threads": 3, "sizes": [[224, 224], [672, 672]]},
    ))
    config = load_config(path)
    assert config.executor.graph.slots == 2
    assert config.executor.graph.prep_threads == 3
    # YAML 给的是嵌套 list，必须变成 tuple of tuple，否则 dataclass 的 frozen 契约形同虚设
    assert config.executor.graph.sizes == ((224, 224), (672, 672))


def test_unknown_field_in_graph_section_is_rejected(tmp_path: Path):
    path = write_config(tmp_path, base_executor(
        graph={"slots": 2, "slots_typo": 4},
    ))
    with pytest.raises(ValueError, match="executor.graph 含未知字段"):
        load_config(path)


def test_empty_sizes_means_derive_from_table(tmp_path: Path):
    path = write_config(tmp_path, base_executor(graph={"slots": 2, "sizes": []}))
    assert load_config(path).executor.graph.sizes == ()


def test_graph_pipeline_rejects_libsmctrl(tmp_path: Path):
    """掩码回调在 cuGraphLaunch 上不生效（§10.7），同用会得到假的"已限 SM"结果。"""
    path = write_config(tmp_path, base_executor(resource_backend="libsmctrl"))
    with pytest.raises(ValueError, match="graph_pipeline 与 resource_backend=libsmctrl 互斥"):
        load_config(path)


def test_graph_pipeline_rejects_adaptive_concurrency(tmp_path: Path):
    path = write_config(tmp_path, base_executor(adaptive_concurrency=True))
    with pytest.raises(ValueError, match="graph_pipeline 与 adaptive_concurrency 不能同时启用"):
        load_config(path)


def test_graph_pipeline_rejects_batching(tmp_path: Path):
    path = write_config(tmp_path, base_executor())
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["batching"] = {"max_batch": 4}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="batching 段不能同时启用"):
        load_config(path)


def test_batch_by_size_parses_and_validates(tmp_path: Path):
    path = write_config(tmp_path, base_executor(
        graph={"slots": 1, "batch_by_size": {224: 4, 336: 1}},
    ))
    config = load_config(path)
    assert config.executor.graph.batch_by_size == ((224, 224, 4), (336, 336, 1))


def test_batch_by_size_size_must_be_captured(tmp_path: Path):
    """表里出现一个不在 `sizes` 里的尺寸，只会静默地永远用不上——必须拦下。"""
    path = write_config(tmp_path, base_executor(
        graph={"slots": 1, "sizes": [[224, 224]], "batch_by_size": {672: 2}},
    ))
    with pytest.raises(ValueError, match="不在 executor.graph.sizes 里"):
        load_config(path)


def test_batch_by_size_must_be_positive(tmp_path: Path):
    path = write_config(tmp_path, base_executor(graph={"slots": 1, "batch_by_size": {224: 0}}))
    with pytest.raises(ValueError, match="批上限必须"):
        load_config(path)


def test_slots_may_not_exceed_streams(tmp_path: Path):
    """slots > streams 会让两个回放槽位挤在同一条流上，回放并发被静默压回流数。"""
    path = write_config(tmp_path, base_executor(streams=2, graph={"slots": 4}))
    with pytest.raises(ValueError, match="不能大于"):
        load_config(path)


def test_slots_must_be_positive(tmp_path: Path):
    path = write_config(tmp_path, base_executor(graph={"slots": 0}))
    with pytest.raises(ValueError, match="executor.graph.slots 必须大于 0"):
        load_config(path)


def test_graph_pipeline_off_ignores_slots_upper_bound(tmp_path: Path):
    """关闭时不做这些校验：历史配置不该因为引入新段而变得无法加载。"""
    path = write_config(tmp_path, base_executor(graph_pipeline=False, streams=1,
                                                graph={"slots": 8}))
    config = load_config(path)
    assert config.executor.graph_pipeline is False
    assert config.executor.graph.slots == 8
