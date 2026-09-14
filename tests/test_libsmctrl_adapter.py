"""libsmctrl 适配器测试（回调路径 + 粘性线程掩码）。

默认不依赖真实 libsmctrl.so 或 GPU：ctypes 边界用 monkeypatch 注入。
真实环境验证由 scripts/probe_libsmctrl.py 与 demos/libsmctrl_thread_mask_demo.py 承担。
"""

import threading

import pytest

import encoder_sched.libsmctrl_adapter as adapter_module
from encoder_sched.libsmctrl_adapter import (
    LibSmCtrlAdapter,
    LibSmCtrlUnavailable,
    enabled_tpcs_to_native_mask,
    fraction_to_enabled_tpcs,
)


class FakeLib:
    """记录 set_thread_mask 调用的假共享库。"""

    def __init__(self, total_tpcs: int = 12):
        self.calls = []
        self.total_tpcs = total_tpcs

    def libsmctrl_set_thread_mask(self, mask):
        self.calls.append(mask.value)

    def libsmctrl_get_tpc_info_cuda(self, out, device):
        out._obj.value = self.total_tpcs
        return 0


@pytest.fixture
def ready_env(monkeypatch):
    """伪装成"补丁库已就绪、回调注册成功、设备 12 TPC"的环境。"""
    fake = FakeLib()
    monkeypatch.setattr(adapter_module, "resolve_library_path", lambda *a, **k: "/fake/libsmctrl.so")
    monkeypatch.setattr(adapter_module, "load_libsmctrl", lambda *a, **k: fake)
    monkeypatch.setattr(adapter_module, "tpc_count", lambda device=0, library=None: 12)
    monkeypatch.setattr(adapter_module, "_callback_path_works", lambda *a, **k: (True, None))
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


def test_zero_count_means_no_restriction():
    """count<=0 必须是"不加限制"(0)，绝不能是 ~0——那会禁用全部 TPC 并挂死进程。"""
    assert enabled_tpcs_to_native_mask(0, 0) == 0
    assert enabled_tpcs_to_native_mask(0, -1) == 0


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


# --- 探针门控：防伪造成功的核心 ---


def test_probe_requires_patched_library(monkeypatch):
    """未打补丁的上游库没有 set_thread_mask，必须明确报告。

    用局部子类模拟，**不要 del 共享类的属性**——那会污染同一进程内的其他测试。
    """

    class UnpatchedLib:
        def libsmctrl_get_tpc_info_cuda(self, out, device):
            out._obj.value = 12
            return 0

    monkeypatch.setattr(adapter_module, "resolve_library_path", lambda *a, **k: "/fake/libsmctrl.so")
    monkeypatch.setattr(adapter_module, "load_libsmctrl", adapter_module.load_libsmctrl)
    monkeypatch.setattr(adapter_module.ctypes, "CDLL", lambda path: UnpatchedLib())
    result = LibSmCtrlAdapter().probe()
    assert result["available"] is False
    assert result["stage"] == "library"
    assert "set_thread_mask" in result["reason"]


def test_probe_fails_when_callback_setup_dies(monkeypatch, ready_env):
    """回调注册失败（子进程被 abort 掉）时必须报告不可用。"""
    monkeypatch.setattr(adapter_module, "_callback_path_works", lambda *a, **k: (False, "exit(1)"))
    result = LibSmCtrlAdapter().probe()
    assert result["available"] is False
    assert result["stage"] == "callback_setup"


def test_failed_probe_never_applies_mask(monkeypatch, ready_env):
    """探针不可用时下发的调用次数必须为 0。

    曾经的缺陷：probe() 在失败前已缓存了库与 TPC 数，apply_quota 只看这两个字段
    就绕过了门控，于是"探针报告不可用"却仍下发掩码并返回 enforced=True。
    """
    monkeypatch.setattr(adapter_module, "_callback_path_works", lambda *a, **k: (False, "模拟失败"))
    adapter = LibSmCtrlAdapter()
    result = adapter.apply_quota(0x1000, 0.5)
    assert result["enforced"] is False
    assert ready_env.calls == [], "探针失败时绝不能调用原生掩码函数"


def test_apply_quota_programs_thread_mask(ready_env):
    result = LibSmCtrlAdapter().apply_quota(0x1000, 0.5)
    assert result["enforced"] is True
    assert result["effective_sm_fraction"] == 0.5
    assert result["tpc_count"] == 6
    assert result["mask_path"] == "callback"
    assert result["verified_by_measurement"] is False
    assert ready_env.calls == [enabled_tpcs_to_native_mask(0, 6)]


# --- 并发分配 ---


def test_concurrent_threads_get_disjoint_tpc_ranges(ready_env):
    """两个并发线程应拿到互不重叠的 TPC 区间。

    用 barrier 强制两份分配**同时存在**。不能用线程池：任务很快时线程会被复用，
    两次提交落到同一线程上就变成"替换自己的分配"，测试会随机失败（已发生过）。
    """
    adapter = LibSmCtrlAdapter()
    results: list[dict] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=15)

    def worker(handle: int, fraction: float) -> None:
        row = adapter.apply_quota(handle, fraction)
        with lock:
            results.append(row)
        barrier.wait()

    threads = [threading.Thread(target=worker, args=(0x10, 0.25)), threading.Thread(target=worker, args=(0x20, 0.5))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2 and all(r["enforced"] for r in results), results
    spans = [set(range(r["tpc_start"], r["tpc_start"] + r["tpc_count"])) for r in results]
    assert spans[0].isdisjoint(spans[1]), "并发线程拿到了重叠的 TPC 集合，这不是空间隔离"
    assert spans[0] | spans[1] <= set(range(12))


def test_many_threads_cover_all_tpcs_without_overlap(ready_env):
    """4 个线程各持 0.25（3 个 TPC），应恰好覆盖全部 12 个且互不重叠。

    用 barrier 强制四个分配**同时存在**。若改用线程池，任务很快完成时线程会被
    复用，落到同一线程上就会变成"替换自己的分配"，测试随之变得不确定。
    """
    adapter = LibSmCtrlAdapter()
    results: list[dict] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=15)

    def worker(handle: int) -> None:
        result = adapter.apply_quota(handle, 0.25)
        with lock:
            results.append(result)
        barrier.wait()

    threads = [threading.Thread(target=worker, args=(0x1000 + i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 4 and all(r["enforced"] for r in results), results
    spans = [set(range(r["tpc_start"], r["tpc_start"] + r["tpc_count"])) for r in results]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert spans[i].isdisjoint(spans[j]), f"线程 {i} 与 {j} 的 TPC 区间重叠"
    assert set().union(*spans) == set(range(12))


def test_over_subscription_reports_shortfall(ready_env):
    """超额申请只在**不同线程**之间成立。

    分配表按线程 id 建账：同一线程重新申请会替换自己的分配，永远成功；
    要触发"池不足"必须是多个线程同时各持一份分配。12 TPC 上三个 0.5 配额
    只能满足两个。
    """
    adapter = LibSmCtrlAdapter()
    results: dict[int, dict] = {}
    barrier = threading.Barrier(3, timeout=15)

    def worker(index: int) -> None:
        results[index] = adapter.apply_quota(0x2000 + index, 0.5)
        barrier.wait()          # 强制三份分配同时存在

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    enforced = sorted(r["enforced"] for r in results.values())
    assert enforced == [False, True, True], f"12 TPC 上三个 0.5 配额应有且仅有一个失败: {results}"
    failed = next(r for r in results.values() if not r["enforced"])
    assert "TPC 池不足" in failed["reason"]


def test_same_thread_reapply_releases_previous(ready_env):
    adapter = LibSmCtrlAdapter()
    adapter.apply_quota(0x10, 0.75)
    result = adapter.apply_quota(0x10, 0.25)
    assert result["enforced"] is True and result["tpc_count"] == 3


@pytest.mark.parametrize("handle", [0, None, "stream", True])
def test_invalid_handle_rejected(ready_env, handle):
    result = LibSmCtrlAdapter().apply_quota(handle, 0.5)
    assert result["enforced"] is False
    assert ready_env.calls == [], "无效句柄绝不能进入原生调用"


def test_release_clears_mask_and_frees_allocation(ready_env):
    adapter = LibSmCtrlAdapter()
    adapter.apply_quota(0x10, 0.5)
    adapter.release()
    assert adapter.describe()["active_threads"] == 0
    assert ready_env.calls[-1] == 0, "释放时应把掩码清为 0（不加限制）"


def test_missing_library_raises(monkeypatch):
    monkeypatch.delenv("LIBSMCTRL_PATH", raising=False)
    monkeypatch.setattr(adapter_module.ctypes.util, "find_library", lambda name: None)
    with pytest.raises(LibSmCtrlUnavailable):
        adapter_module.resolve_library_path(None)
