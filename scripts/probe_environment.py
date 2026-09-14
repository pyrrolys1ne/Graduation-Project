from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util
import json
import platform
import shutil
import subprocess
from typing import Any


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_config_values(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        executor = data.get("executor", {})
        if not isinstance(executor, dict):
            return {"config_error": "executor 必须是映射"}
        return {
            "configured_backend": executor.get("resource_backend", "proxy"),
            "adapter": executor.get("libsmctrl_adapter", ""),
            "allow_proxy_fallback": executor.get("allow_proxy_fallback", True),
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"config_error": str(exc)}


def _adapter_status(adapter: str) -> tuple[bool, str | None, dict[str, Any] | None]:
    """检查适配器能否加载，并调用其 probe() 获取真实能力。

    仅"能导入"不等于"能生效"：libsmctrl_set_stream_mask 返回 void，驱动版本不
    匹配时会静默 no-op，所以必须由 probe() 给出结论。
    """
    if not adapter:
        return False, "未配置 libsmctrl_adapter", None
    if ":" not in adapter:
        return False, "适配器必须使用 模块:对象 格式", None
    module_name, object_name = adapter.split(":", 1)
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, f"找不到适配器模块 {module_name}", None
        module = __import__(module_name, fromlist=[object_name])
        target = getattr(module, object_name)
        if not callable(target):
            return False, f"适配器对象 {adapter} 不可调用", None
    except (ImportError, AttributeError, ValueError) as exc:
        return False, f"无法加载适配器 {adapter}: {exc}", None

    capability = None
    probe = getattr(module, "probe", None)
    if callable(probe):
        try:
            capability = probe()
        except Exception as exc:  # 探针失败不能让整个环境报告崩掉
            return True, f"能力探针执行失败: {exc}", None
    return True, None, capability


def build_report(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config = _load_config_values(config_path)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "nvcc": command_output(["nvcc", "--version"]) if shutil.which("nvcc") else None,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"]
        ),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        **config,
    }
    if report["torch_installed"]:
        try:
            import torch

            report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        except Exception as exc:  # environment probe must still emit JSON
            report.update(torch_error=f"无法读取 torch CUDA 状态: {exc}")
    if report.get("configured_backend") == "libsmctrl":
        loadable, reason, capability = _adapter_status(str(report.get("adapter", "")))
        report["adapter_loadable"] = loadable
        report["capability"] = capability
        available = bool(isinstance(capability, dict) and capability.get("available"))
        if not loadable:
            report["probe_status"] = "unavailable"
            report["reason"] = reason
        elif not available:
            report["probe_status"] = "capability_unavailable"
            report["reason"] = (capability or {}).get("reason") or "适配器未提供可用的能力探针结论"
        else:
            report["probe_status"] = "ready"
            report["reason"] = None
        report["libsmctrl_possible"] = report["system"] == "Linux" and available
    else:
        report["adapter_loadable"] = None
        report["probe_status"] = "proxy_only"
        report["reason"] = "当前配置使用 proxy，不执行 libsmctrl 控制调用。"
        report["libsmctrl_possible"] = report["system"] == "Linux"
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="检查编码器资源控制运行环境")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    print(json.dumps(build_report(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
