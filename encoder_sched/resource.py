from __future__ import annotations

import importlib
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


class UnsupportedResourceBackend(RuntimeError):
    pass


class ResourceApplyError(RuntimeError):
    pass


class ResourceBackend(ABC):
    name: str
    enforces_sm_partition: bool = False

    @abstractmethod
    def apply_quota(self, stream_or_worker: Any, sm_fraction: float) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "configured_backend": self.name,
            "effective_backend": self.name,
            "enforces_sm_partition": self.enforces_sm_partition,
            "fallback_active": False,
        }


@dataclass
class ProxyResourceBackend(ResourceBackend):
    name: str = "proxy"
    enforces_sm_partition: bool = False
    fallback_reason: str | None = None

    def apply_quota(self, stream_or_worker: Any, sm_fraction: float) -> dict[str, Any]:
        _validate_fraction(sm_fraction)
        return {
            "backend": self.name,
            "configured_backend": "proxy",
            "effective_backend": self.name,
            "requested_sm_fraction": sm_fraction,
            "enforced": False,
            "fallback": self.fallback_reason is not None,
            "reason": self.fallback_reason
            or "代理后端只记录配额并控制并发，不实现 SM 硬件隔离。",
        }

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update(
            configured_backend="libsmctrl" if self.fallback_reason else "proxy",
            fallback_active=self.fallback_reason is not None,
            fallback_reason=self.fallback_reason,
            capability="proxy_only",
        )
        return result


class LibSmCtrlBackend(ResourceBackend):
    name = "libsmctrl"
    #: 只有在能力探针确认驱动版本受支持且 TPC 可查询后，实例才会把它置为 True。
    #: 绝不能是类级常量：libsmctrl_set_stream_mask 返回 void，版本不匹配时它只向
    #: stderr 打印一行就返回，调用方无法察觉，把这种情况记为"已隔离"就是伪造结果。
    enforces_sm_partition = False

    def __init__(self, adapter: str):
        if platform.system() != "Linux":
            raise UnsupportedResourceBackend("libsmctrl 仅在 Linux 阶段启用")
        if not adapter or ":" not in adapter:
            raise UnsupportedResourceBackend("需要配置 libsmctrl_adapter=模块:函数")
        module_name, function_name = adapter.split(":", 1)
        if not module_name or not function_name:
            raise UnsupportedResourceBackend("libsmctrl_adapter 的模块和函数不能为空")
        try:
            module = importlib.import_module(module_name)
            self._apply: Callable[[Any, float], Any] = getattr(module, function_name)
        except (ImportError, AttributeError) as exc:
            raise UnsupportedResourceBackend(f"无法加载 libsmctrl 适配器 {adapter}: {exc}") from exc
        if not callable(self._apply):
            raise UnsupportedResourceBackend(f"libsmctrl 适配器 {adapter} 不可调用")
        self.adapter = adapter
        self._last_apply: dict[str, Any] | None = None
        self.capability: dict[str, Any] = self._run_probe(module)

    def _run_probe(self, module: Any) -> dict[str, Any]:
        probe = getattr(module, "probe", None)
        if not callable(probe):
            return {
                "available": None,
                "capability": "unverified",
                "reason": "适配器未提供 probe()，无法确认掩码是否真正生效；enforces_sm_partition 保持为 False",
            }
        try:
            info = probe()
        except Exception as exc:
            raise UnsupportedResourceBackend(f"libsmctrl 能力探针执行失败: {exc}") from exc
        if not isinstance(info, dict):
            raise UnsupportedResourceBackend("libsmctrl 能力探针必须返回字典")
        if not info.get("available", False):
            raise UnsupportedResourceBackend(f"libsmctrl 能力探针未通过: {info.get('reason', '原因未知')}")
        self.enforces_sm_partition = True
        return info

    def apply_quota(self, stream_or_worker: Any, sm_fraction: float) -> dict[str, Any]:
        _validate_fraction(sm_fraction)
        try:
            raw_result = self._apply(stream_or_worker, sm_fraction)
        except Exception as exc:
            raise ResourceApplyError(f"libsmctrl 适配器执行失败: {exc}") from exc
        result = _normalize_adapter_result(raw_result)
        result.update(
            backend=self.name,
            configured_backend=self.name,
            effective_backend=self.name,
            requested_sm_fraction=sm_fraction,
        )
        self._last_apply = result
        return result

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update(
            capability=self.capability.get("capability", "probe_passed"),
            probe=self.capability,
            adapter=self.adapter,
            last_apply=self._last_apply,
        )
        return result


def _validate_fraction(sm_fraction: float) -> None:
    if not isinstance(sm_fraction, (int, float)) or not 0 < sm_fraction <= 1:
        raise ValueError("sm_fraction 必须位于 (0, 1]")


def _normalize_adapter_result(raw_result: Any) -> dict[str, Any]:
    """Require the adapter to explicitly confirm whether control was enforced."""
    if isinstance(raw_result, bool):
        if not raw_result:
            return {"enforced": False, "reason": "适配器未确认配额已生效"}
        return {"enforced": True}
    if not isinstance(raw_result, dict):
        raise ResourceApplyError("适配器必须返回 bool 或包含 enforced 字段的字典")
    if "enforced" not in raw_result or not isinstance(raw_result["enforced"], bool):
        raise ResourceApplyError("适配器返回结果必须明确包含布尔字段 enforced")
    return dict(raw_result)


def create_resource_backend(name: str, adapter: str, allow_proxy_fallback: bool) -> ResourceBackend:
    if name == "proxy":
        return ProxyResourceBackend()
    if name != "libsmctrl":
        raise UnsupportedResourceBackend(f"未知资源控制后端: {name}")
    try:
        return LibSmCtrlBackend(adapter)
    except UnsupportedResourceBackend as exc:
        if not allow_proxy_fallback:
            raise
        return ProxyResourceBackend(fallback_reason=str(exc))
