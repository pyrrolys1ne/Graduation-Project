from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .concurrency import ConcurrencyConfig


@dataclass(frozen=True)
class ModelConfig:
    """视觉编码器模型配置。

    **没有 `device` 字段**：真实 CLIP 后端只支持 CUDA。CPU 分支已删除，因为它在整个
    测试与实验里从未被真正执行过（测试夹具写 `device: cpu` 但全部走 `fake=True`），
    而 CPU 推理对延迟与配额研究没有意义。需要无 GPU 验证链路时用 `FakeEncoderBackend`。

    旧配置里若仍写着 `device:`，`load_config` 会给出明确的迁移提示，而不是抛出
    难懂的 `unexpected keyword argument`。
    """

    name: str = "openai/clip-vit-base-patch32"
    dtype: str = "float16"
    local_files_only: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "edf_size"
    deadline_tie_ms: float = 5.0
    quota_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    #: 配额选择策略：deadline_min（历史行为，"最小可行配额"）或 full（始终用满）。
    #: 详见 encoder_sched/scheduler.py 的 choose_quota。
    quota_policy: str = "deadline_min"
    dacc_window: int = 8
    dacc_beta: float = 1.0
    dacc_guard_ms: float = 5.0
    #: 覆盖 DaccConfig 中的任意字段，用于消融实验，例如 {"w_complementarity": 0.0}。
    #: 键必须在 DaccConfig 中存在，否则 load_config 直接报错，避免拼错字段名后
    #: 消融实验静默地什么都没改。
    dacc_overrides: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorConfig:
    streams: int = 2
    resource_backend: str = "proxy"
    allow_proxy_fallback: bool = True
    libsmctrl_adapter: str = ""
    #: 是否启用**运行时并发度控制**。
    #:
    #: 关闭时（默认）并发度是启动时定死的 ``streams``——这是本项目此前的全部
    #: 历史行为，保持默认关闭以保证既有结果可复现。
    #: 打开时由 ``ConcurrencyController`` 按队列状态逐请求决定共驻数，
    #: 见 ``encoder_sched/concurrency.py`` 的模块说明。
    adaptive_concurrency: bool = False
    #: 并发度控制器的参数。仅当 ``adaptive_concurrency`` 为真时生效。
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


@dataclass(frozen=True)
class ProfilingConfig:
    table_path: str = "data/profiles/default.csv"


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "results"
    request_log: str = "requests.jsonl"


@dataclass(frozen=True)
class BatchingConfig:
    """批处理配置。

    ``max_batch = 1`` 表示**关闭批处理**（逐请求执行），即本项目此前的行为。
    大于 1 时，worker 会从队列里收集**同尺寸**请求凑成一批。

    批处理与空间分割是两种不同的并发手段：批处理把多个请求合并成一次前向，让硬件
    自己在同一批 SM 上交错调度（吞吐高、但要等组批）；空间分割给每个请求独立的 TPC
    区间（无需等待、但划分后 SM 内部不再有跨请求填空的机会）。两者的取舍由
    ``max_delay_ms`` 这个"愿意等多久"直接控制。
    """

    max_batch: int = 1
    #: 组批时最多等待多少毫秒。0 表示只取队列里已有的请求，绝不为凑批而等待。
    max_delay_ms: float = 2.0


@dataclass(frozen=True)
class MetricsConfig:
    #: GPU 利用率采样间隔（秒）。`nvidia-smi` 返回的是瞬时快照而非区间均值，
    #: 间隔过大时短实验只能采到个位数样本，均值不具代表性。默认 0.1s 是为
    #: 秒级实验准备的；长时间实验可适当调大以降低采样开销。
    gpu_sample_interval_s: float = 0.1


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    seed: int = 20260902
    base_dir: Path = Path(".")

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_dir / path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {name} 必须是映射")
    return value


def _make_executor_config(data: dict[str, Any]) -> ExecutorConfig:
    """从 YAML 段构造 ExecutorConfig。

    需要单独处理是因为 ``concurrency`` 是**嵌套 dataclass**：直接
    ``ExecutorConfig(**data)`` 会把 YAML 里的 dict 原样塞进去，之后访问
    ``config.executor.concurrency.max_concurrency`` 才在运行时炸——
    而那时离出错地点已经很远。这里显式展开，并把未知键拦在加载阶段。
    """
    payload = dict(data)
    concurrency_data = payload.pop("concurrency", None)
    if concurrency_data is None:
        return ExecutorConfig(**payload)
    if not isinstance(concurrency_data, dict):
        raise ValueError("executor.concurrency 必须是映射")
    known = {item.name for item in fields(ConcurrencyConfig)}
    unknown = sorted(set(concurrency_data) - known)
    if unknown:
        raise ValueError(f"executor.concurrency 含未知字段: {unknown}；可用字段: {sorted(known)}")
    return ExecutorConfig(concurrency=ConcurrencyConfig(**concurrency_data), **payload)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model_data = dict(_section(data, "model"))
    if "device" in model_data:
        raise ValueError(
            "model.device 已移除：真实 CLIP 后端只支持 CUDA。请从配置中删除该字段；"
            "若需要在无 GPU 环境验证链路，请改用 ENCODER_SCHED_FAKE=1（FakeEncoderBackend）。"
        )
    scheduler_data = _section(data, "scheduler")
    if "quota_levels" in scheduler_data:
        scheduler_data = {**scheduler_data, "quota_levels": tuple(scheduler_data["quota_levels"])}
    config = AppConfig(
        model=ModelConfig(**model_data),
        scheduler=SchedulerConfig(**scheduler_data),
        executor=_make_executor_config(_section(data, "executor")),
        profiling=ProfilingConfig(**_section(data, "profiling")),
        logging=LoggingConfig(**_section(data, "logging")),
        batching=BatchingConfig(**_section(data, "batching")),
        metrics=MetricsConfig(**_section(data, "metrics")),
        seed=int(data.get("seed", 20260902)),
        base_dir=config_path.parent,
    )
    if config.executor.streams < 1:
        raise ValueError("executor.streams 必须大于 0")
    if config.metrics.gpu_sample_interval_s <= 0:
        raise ValueError("metrics.gpu_sample_interval_s 必须大于 0")
    if config.batching.max_batch < 1:
        raise ValueError("batching.max_batch 必须大于等于 1（1 表示关闭批处理）")
    if config.batching.max_delay_ms < 0:
        raise ValueError("batching.max_delay_ms 不能为负")
    if not config.scheduler.quota_levels or any(not 0 < value <= 1 for value in config.scheduler.quota_levels):
        raise ValueError("quota_levels 必须位于 (0, 1]")
    if config.scheduler.dacc_overrides:
        from .dacc import DaccConfig

        # 局部名刻意不叫 fields——那会遮蔽从 dataclasses 导入的同名函数。
        dacc_fields = {item.name: item.type for item in fields(DaccConfig)}
        unknown = sorted(set(config.scheduler.dacc_overrides) - set(dacc_fields))
        if unknown:
            raise ValueError(f"dacc_overrides 含未知字段: {unknown}；可用字段: {sorted(dacc_fields)}")
        # dataclasses.replace 不做类型检查，值写错会一路带到运行时才崩。
        # DaccConfig 的字段全是数值，这里直接拦住。
        for key, value in config.scheduler.dacc_overrides.items():
            if dacc_fields[key] in {"int", "float"} and not isinstance(value, (int, float)):
                raise ValueError(f"dacc_overrides.{key} 应为数值，实际为 {type(value).__name__}: {value!r}")
    return config
