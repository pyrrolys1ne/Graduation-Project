import types

import pytest

import encoder_sched.resource as resource_module
from encoder_sched.resource import (
    ProxyResourceBackend,
    ResourceApplyError,
    UnsupportedResourceBackend,
    create_resource_backend,
)


def test_proxy_does_not_claim_enforcement():
    result = ProxyResourceBackend().apply_quota(0, 0.5)
    assert result["backend"] == "proxy"
    assert result["enforced"] is False
    assert result["effective_backend"] == "proxy"


def test_libsmctrl_requires_explicit_success(monkeypatch):
    module = types.SimpleNamespace(apply=lambda handle, fraction: {"enforced": True, "native_status": 0})
    monkeypatch.setattr(resource_module.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(resource_module.platform, "system", lambda: "Linux")
    backend = create_resource_backend("libsmctrl", "adapter:apply", False)
    result = backend.apply_quota(123, 0.5)
    assert result["enforced"] is True
    assert result["native_status"] == 0


def test_libsmctrl_rejects_unknown_result(monkeypatch):
    module = types.SimpleNamespace(apply=lambda handle, fraction: object())
    monkeypatch.setattr(resource_module.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(resource_module.platform, "system", lambda: "Linux")
    backend = create_resource_backend("libsmctrl", "adapter:apply", False)
    with pytest.raises(ResourceApplyError):
        backend.apply_quota(123, 0.5)


def test_libsmctrl_initialization_can_fallback(monkeypatch):
    monkeypatch.setattr(resource_module.platform, "system", lambda: "Linux")
    backend = create_resource_backend("libsmctrl", "missing:apply", True)
    assert isinstance(backend, ProxyResourceBackend)
    assert backend.describe()["fallback_active"] is True
    assert backend.apply_quota(0, 0.5)["fallback"] is True


def test_libsmctrl_rejects_non_linux_without_fallback(monkeypatch):
    monkeypatch.setattr(resource_module.platform, "system", lambda: "Windows")
    with pytest.raises(UnsupportedResourceBackend):
        create_resource_backend("libsmctrl", "adapter:apply", False)


def test_invalid_fraction_is_rejected():
    with pytest.raises(ValueError):
        ProxyResourceBackend().apply_quota(0, 0)


def _install(monkeypatch, module):
    monkeypatch.setattr(resource_module.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(resource_module.platform, "system", lambda: "Linux")


def test_adapter_without_probe_stays_unverified(monkeypatch):
    """没有 probe() 的适配器无法证明掩码生效，能力标志必须保持 False。"""
    _install(monkeypatch, types.SimpleNamespace(apply=lambda h, f: {"enforced": True}))
    backend = create_resource_backend("libsmctrl", "adapter:apply", False)
    assert backend.enforces_sm_partition is False
    assert backend.describe()["capability"] == "unverified"


def test_probe_failure_raises_without_fallback(monkeypatch):
    module = types.SimpleNamespace(
        apply=lambda h, f: {"enforced": True},
        probe=lambda: {"available": False, "reason": "驱动报告的 CUDA 版本不受支持"},
    )
    _install(monkeypatch, module)
    with pytest.raises(UnsupportedResourceBackend, match="能力探针未通过"):
        create_resource_backend("libsmctrl", "adapter:apply", False)


def test_probe_failure_falls_back_explicitly(monkeypatch):
    module = types.SimpleNamespace(
        apply=lambda h, f: {"enforced": True},
        probe=lambda: {"available": False, "reason": "驱动报告的 CUDA 版本不受支持"},
    )
    _install(monkeypatch, module)
    backend = create_resource_backend("libsmctrl", "adapter:apply", True)
    assert isinstance(backend, ProxyResourceBackend)
    assert "驱动报告的 CUDA 版本不受支持" in backend.fallback_reason
    assert backend.describe()["fallback_active"] is True


def test_probe_success_enables_partition_flag(monkeypatch):
    module = types.SimpleNamespace(
        apply=lambda h, f: {"enforced": True},
        probe=lambda: {"available": True, "total_tpcs": 12},
    )
    _install(monkeypatch, module)
    backend = create_resource_backend("libsmctrl", "adapter:apply", False)
    assert backend.enforces_sm_partition is True
    assert backend.describe()["probe"]["total_tpcs"] == 12
