from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "edf_size"
    deadline_tie_ms: float = 5.0
    quota_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    dacc_window: int = 8
    dacc_beta: float = 1.0
    dacc_guard_ms: float = 5.0


@dataclass(frozen=True)
class ExecutorConfig:
    streams: int = 2
    resource_backend: str = "proxy"
    allow_proxy_fallback: bool = True
    libsmctrl_adapter: str = ""


@dataclass(frozen=True)
class ProfilingConfig:
    table_path: str = "data/profiles/default.csv"


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "results"
    request_log: str = "requests.jsonl"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
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


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    scheduler_data = _section(data, "scheduler")
    if "quota_levels" in scheduler_data:
        scheduler_data = {**scheduler_data, "quota_levels": tuple(scheduler_data["quota_levels"])}
    config = AppConfig(
        model=ModelConfig(**_section(data, "model")),
        scheduler=SchedulerConfig(**scheduler_data),
        executor=ExecutorConfig(**_section(data, "executor")),
        profiling=ProfilingConfig(**_section(data, "profiling")),
        logging=LoggingConfig(**_section(data, "logging")),
        seed=int(data.get("seed", 20260902)),
        base_dir=config_path.parent,
    )
    if config.executor.streams < 1:
        raise ValueError("executor.streams 必须大于 0")
    if not config.scheduler.quota_levels or any(not 0 < value <= 1 for value in config.scheduler.quota_levels):
        raise ValueError("quota_levels 必须位于 (0, 1]")
    return config
