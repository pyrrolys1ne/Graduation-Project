"""libsmctrl 适配器测试。

默认不依赖真实 libsmctrl.so 或 GPU：ctypes 边界通过 monkeypatch 注入。
真实环境验证由 scripts/probe_libsmctrl.py 承担。
"""

import pytest

import encoder_sched.libsmctrl_adapter as adapter_module
from encoder_sched.libsmctrl_adapter import (
    SUPPORTED_DRIVER_VERSIONS,
    LibSmCtrlAdapter,
    LibSmCtrlUnavailable,
    enabled_tpcs_to_native_mask,
    fraction_to_enabled_tpcs,
)


class FakeLib:
    """记录 set_stream_mask 调用的假共享库。"""

    def __init__(self):
        self.calls = []

    def libsmctrl_set_stream_mask(self, stream, mask):
        self.calls.append((stream.value, mask.value))


@pytest.fixture
def supported_env(monkeypatch):
    """伪装成驱动版本受支持、TPC 数为 12 的环境。"""
    monkeypatch.setenv("LIBSMCTRL_DRIVER_VERSION", "12080")
    monkeypatch.setattr(adapter_module, "load_libcuda", lambda *a, **k: object())
    monkeypatch.setattr(adapter_module, "tpc_count", lambda device=0: 12)
    fake = FakeLib()
    monkeypatch.setattr(adapter_module, "load_libsmctrl", lambda *a, **k: fake)
    return fake


# --- 掩码语义：必须与上游 libsmctrl.h 的约定一致 ---


def test_set_bit_means_disabled():
    """上游约定：掩码中置位的 TPC 被**禁用**。"""
    mask = enabled_tpcs_to_native_mask(0, 3)
    for tpc in range(3):
        assert not (mask >> tpc) & 1, f"TPC {tpc} 应被启用（该位应为 0）"
    for tpc in range(3, 64):
        assert (mask >> tpc) & 1, f"TPC {tpc} 应被禁用（该位应为 1）"


def test_matches_upstream_documented_example():
    """上游示例：只允许 TPC 2,3,4,5 -> ~0b00111100ull。"""
    assert enabled_tpcs_to_native_mask(2, 4) == (~0b00111100) & ((1 << 64) - 1)


def test_contiguous_range_is_shifted():
    mask = enabled_tpcs_to_native_mask(6, 6)
    enabled = [i for i in range(64) if not (mask >> i) & 1]
    assert enabled == [6, 7, 8, 9, 10, 11]


# --- 配额量化 ---


@pytest.mark.parametrize(
    "fraction,expected",
    [(0.25, 3), (0.5, 6), (0.75, 9), (1.0, 12), (0.01, 1), (0.99, 12)],
)
def test_fraction_quantises_to_tpc_granularity(fraction, expected):
    assert fraction_to_enabled_tpcs(fraction, 12) == expected


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_invalid_fraction_rejected(bad):
    with pytest.raises(ValueError):
        fraction_to_enabled_tpcs(bad, 12)


# --- 探针：版本门控是防伪造的核心 ---


def test_probe_reports_unsupported_driver(monkeypatch):
    monkeypatch.setenv("LIBSMCTRL_DRIVER_VERSION", "13030")
    monkeypatch.setattr(adapter_module, "load_libcuda", lambda *a, **k: object())
    result = LibSmCtrlAdapter().probe()
    assert result["available"] is False
    assert result["stage"] == "driver_version"
    assert "exit(1)" in result["reason"]


def test_all_supported_versions_are_pre_cuda13():
    assert 13030 not in SUPPORTED_DRIVER_VERSIONS
    assert max(SUPPORTED_DRIVER_VERSIONS) == 12080


# --- 下发与不重叠分配 ---


def test_apply_quota_programs_stream_mask(supported_env):
    result = LibSmCtrlAdapter().apply_quota(0x1000, 0.5)
    assert result["enforced"] is True
    assert result["effective_sm_fraction"] == 0.5
    assert result["tpc_count"] == 6
    assert result["verified_by_measurement"] is False
    assert len(supported_env.calls) == 1
    handle, mask = supported_env.calls[0]
    assert handle == 0x1000
    assert mask == enabled_tpcs_to_native_mask(0, 6)


def test_concurrent_streams_get_disjoint_tpc_ranges(supported_env):
    adapter = LibSmCtrlAdapter()
    a = adapter.apply_quota(0x10, 0.5)
    b = adapter.apply_quota(0x20, 0.5)
    assert (a["tpc_start"], a["tpc_count"]) == (0, 6)
    assert (b["tpc_start"], b["tpc_count"]) == (6, 6)
    range_a = set(range(a["tpc_start"], a["tpc_start"] + a["tpc_count"]))
    range_b = set(range(b["tpc_start"], b["tpc_start"] + b["tpc_count"]))
    assert not (range_a & range_b), "两个并发流拿到了重叠的 TPC 集合，这不是空间隔离"


def test_over_subscription_fails_loudly(supported_env):
    adapter = LibSmCtrlAdapter()
    assert adapter.apply_quota(0x10, 0.75)["enforced"] is True
    result = adapter.apply_quota(0x20, 0.75)
    assert result["enforced"] is False
    assert "TPC 池不足" in result["reason"]
    assert len(supported_env.calls) == 1, "分配失败时不应下发掩码"


def test_reapply_releases_previous_allocation(supported_env):
    adapter = LibSmCtrlAdapter()
    adapter.apply_quota(0x10, 0.75)
    result = adapter.apply_quota(0x10, 0.25)
    assert result["enforced"] is True
    assert result["tpc_count"] == 3


@pytest.mark.parametrize("handle", [0, None, "stream"])
def test_invalid_stream_handle_rejected(supported_env, handle):
    result = LibSmCtrlAdapter().apply_quota(handle, 0.5)
    assert result["enforced"] is False
    assert not supported_env.calls, "无效句柄绝不能进入原生调用"


def test_apply_quota_refuses_when_probe_failed(monkeypatch):
    monkeypatch.setenv("LIBSMCTRL_DRIVER_VERSION", "13030")
    monkeypatch.setattr(adapter_module, "load_libcuda", lambda *a, **k: object())
    result = LibSmCtrlAdapter().apply_quota(0x1000, 0.5)
    assert result["enforced"] is False
    assert result["probe"]["stage"] == "driver_version"


def test_missing_library_raises(monkeypatch):
    monkeypatch.delenv("LIBSMCTRL_PATH", raising=False)
    monkeypatch.setattr(adapter_module.ctypes.util, "find_library", lambda name: None)
    with pytest.raises(LibSmCtrlUnavailable):
        adapter_module.load_libsmctrl()
