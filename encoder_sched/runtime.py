from __future__ import annotations

import platform
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .config import AppConfig


class RuntimeValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeReport:
    system: str
    python: str
    torch: str
    torch_cuda: str
    gpu: str
    compute_capability: str
    device_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError as exc:
        raise RuntimeValidationError(f"无法解析版本号: {value!r}") from exc


def validate_server_runtime(config: AppConfig) -> dict[str, Any]:
    """在加载模型前拒绝不满足正式服务器约束的环境。"""

    runtime = config.runtime
    system = platform.system()
    if runtime.require_linux and system != "Linux":
        raise RuntimeValidationError(f"正式服务仅支持 Linux，当前系统为 {system}")
    if sys.version_info[:2] < runtime.min_python:
        required = ".".join(map(str, runtime.min_python))
        raise RuntimeValidationError(f"需要 Python >= {required}，当前为 {platform.python_version()}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeValidationError("未安装 PyTorch；请安装与服务器 NVIDIA 驱动兼容的 CUDA wheel") from exc

    torch_cuda = torch.version.cuda
    if runtime.require_cuda and not torch.cuda.is_available():
        raise RuntimeValidationError("torch.cuda.is_available() 为 False，无法启动 GPU 服务")
    if runtime.require_cuda and not torch_cuda:
        raise RuntimeValidationError("当前 PyTorch 是 CPU 构建，未包含 CUDA runtime")
    if torch_cuda and _version_tuple(torch_cuda)[0] > runtime.max_cuda_major:
        raise RuntimeValidationError(
            f"PyTorch CUDA runtime {torch_cuda} 超过配置允许的 CUDA {runtime.max_cuda_major}.x"
        )

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if runtime.require_cuda and device_count < 1:
        raise RuntimeValidationError("没有可见 CUDA GPU；请检查 NVIDIA 驱动和 CUDA_VISIBLE_DEVICES")
    device = torch.cuda.current_device() if device_count else 0
    capability = torch.cuda.get_device_capability(device) if device_count else (0, 0)
    report = RuntimeReport(
        system=system,
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch_cuda or "none",
        gpu=torch.cuda.get_device_name(device) if device_count else "none",
        compute_capability=f"{capability[0]}.{capability[1]}",
        device_count=device_count,
    )
    return report.as_dict()


def main() -> None:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="验证 Linux CUDA 服务器运行环境")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        report = validate_server_runtime(load_config(args.config))
    except (OSError, ValueError, RuntimeValidationError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps({"status": "ready", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
