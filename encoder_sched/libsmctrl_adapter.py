"""基于真实 libsmctrl 的 TPC 掩码适配器。

上游来源：``http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git``（原仓库）。
注意 ``github.com/atomicapple0/libsmctrl`` 只是 RTAS23 论文的冻结快照 fork，
其 readme 明确建议改用原仓库；两者支持的 CUDA 版本范围不同。

掩码语义（逐字取自上游 ``libsmctrl.h``）::

    A set bit in the mask indicates that the respective Thread Processing
    Cluster (TPC) is to be __disabled__.

即**置位表示禁用**。若要启用 TPC ``0..n-1``，必须传入 ``~((1 << n) - 1)``。

本适配器存在的最重要理由是下面这个陷阱：
``libsmctrl_set_stream_mask()`` 的返回类型是 ``void``，其实现是按**驱动版本
硬编码的字节偏移**直接写入驱动内部 stream 结构。一旦驱动版本不在它的
``switch`` 白名单里，上游实现会走到::

    abort(1, 0, "Stream masking unsupported on this CUDA version (%d), ...")

而 ``abort`` 在上游被定义为 ``error_at_line``（见源码注释："we favor
terminating with an error rather than merely printing a warning and
continuing"）。``error_at_line`` 在第一个参数非零时会 **``exit(1)`` 终止整个
进程**——不是返回错误码，不是抛异常，而是直接杀掉调用进程。

因此对这个库而言，"先检查驱动版本"不是保守起见，而是防止服务进程被终止的
必要护栏。本模块先读 ``cuDriverGetVersion`` 与白名单比对，版本不受支持时返回
``enforced=False`` 并给出原因，**绝不把控制权交给原生函数**。

注意上游与 ``atomicapple0/libsmctrl``（RTAS23 冻结快照 fork）在此处行为不同：
fork 版本是静默 ``return``（什么都没做且不报错），上游版本是 ``exit(1)``。
两者都会让"调用成功"不等于"已生效"，所以本适配器对二者都采取同样的拒绝策略。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SUPPORTED_DRIVER_VERSIONS",
    "DriverInfo",
    "LibSmCtrlAdapter",
    "apply_quota",
    "driver_info",
    "enabled_tpcs_to_native_mask",
    "fraction_to_enabled_tpcs",
    "tpc_count",
]

#: ``libsmctrl_set_stream_mask`` 支持的非 Jetson 驱动版本，逐字取自上游
#: ``libsmctrl.c`` 中该函数 ``switch (ver)`` 的 case 标签。CUDA 13.x 不在其中，
#: 会落入 ``default:`` 分支变成静默 no-op。
SUPPORTED_DRIVER_VERSIONS = frozenset(
    {
        8000,
        9000,
        9010,
        9020,
        10000,
        10010,
        10020,
        11000,
        11010,
        11020,
        11030,
        11040,
        11050,
        11060,
        11070,
        11080,
        12000,
        12010,
        12020,
        12030,
        12040,
        12050,
        12060,
        12070,
        12080,
    }
)

#: 覆盖驱动版本检测的测试钩子；正式实验不应设置。
DRIVER_VERSION_ENV = "LIBSMCTRL_DRIVER_VERSION"
#: 显式指定 libsmctrl 共享库路径。
LIBRARY_PATH_ENV = "LIBSMCTRL_PATH"

_LIBCUDA_CANDIDATES = (
    "/usr/lib/wsl/lib/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
)


class LibSmCtrlUnavailable(RuntimeError):
    """真实库或驱动前置条件不满足。"""


def _find_library(names: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise LibSmCtrlUnavailable(f"指定的库不存在: {explicit}")
        return explicit
    for name in names:
        if name.startswith("/"):
            if Path(name).exists():
                return name
            continue
        found = ctypes.util.find_library(name)
        if found:
            return found
    raise LibSmCtrlUnavailable(f"找不到共享库，候选: {names}")


def load_libcuda(explicit: str | None = None) -> ctypes.CDLL:
    """加载 libcuda（WSL2 上位于 /usr/lib/wsl/lib/）。"""
    names = (_LIBCUDA_CANDIDATES if explicit is None else (explicit,)) + ("cuda",)
    path = _find_library(names, explicit)
    return ctypes.CDLL(path)


def load_libsmctrl(explicit: str | None = None) -> ctypes.CDLL:
    """加载真实 libsmctrl 共享库。"""
    explicit = explicit or os.environ.get(LIBRARY_PATH_ENV) or None
    candidates = (explicit,) if explicit else ("smctrl", "libsmctrl.so")
    path = _find_library(tuple(c for c in candidates if c), explicit)
    lib = ctypes.CDLL(path)
    if not hasattr(lib, "libsmctrl_set_stream_mask"):
        raise LibSmCtrlUnavailable(f"{path} 未导出 libsmctrl_set_stream_mask")
    # 返回类型是 void；显式声明避免 ctypes 把返回值当 int 处理。
    lib.libsmctrl_set_stream_mask.restype = None
    lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    return lib


@dataclass(frozen=True)
class DriverInfo:
    version: int
    supported: bool
    source: str

    @property
    def label(self) -> str:
        return f"{self.version // 1000}.{(self.version % 1000) // 10}"


def driver_info(libcuda: ctypes.CDLL | None = None) -> DriverInfo:
    """读取驱动支持的 CUDA 版本，用于判断 stream 掩码是否可能生效。"""
    override = os.environ.get(DRIVER_VERSION_ENV)
    if override:
        version = int(override)
        return DriverInfo(version, version in SUPPORTED_DRIVER_VERSIONS, "env-override")
    lib = libcuda or load_libcuda()
    lib.cuDriverGetVersion.restype = ctypes.c_int
    lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int(0)
    rc = lib.cuDriverGetVersion(ctypes.byref(version))
    if rc != 0:
        raise LibSmCtrlUnavailable(f"cuDriverGetVersion 失败，rc={rc}")
    return DriverInfo(version.value, version.value in SUPPORTED_DRIVER_VERSIONS, "cuDriverGetVersion")


def tpc_count(cuda_device: int = 0) -> int:
    """查询设备的 TPC 总数（上游按 SM 数除以每 TPC 的 SM 数换算）。"""
    lib = load_libsmctrl()
    count = ctypes.c_uint32(0)
    rc = lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(count), ctypes.c_int(cuda_device))
    if rc != 0:
        raise LibSmCtrlUnavailable(f"libsmctrl_get_tpc_info_cuda 失败，rc={rc}")
    if count.value == 0:
        raise LibSmCtrlUnavailable("TPC 数量为 0，设备信息不可用")
    return int(count.value)


def fraction_to_enabled_tpcs(fraction: float, total_tpcs: int) -> int:
    """把 SM 配额比例换算成要启用的 TPC 个数。

    粒度是 1 个 TPC，因此实际配额会被量化；调用方应使用返回的
    ``effective_sm_fraction`` 而不是请求值来记录实验结果。
    """
    if not 0 < fraction <= 1:
        raise ValueError("sm_fraction 必须位于 (0, 1]")
    if total_tpcs < 1:
        raise ValueError("total_tpcs 必须为正")
    enabled = int(round(fraction * total_tpcs))
    return max(1, min(total_tpcs, enabled))


def enabled_tpcs_to_native_mask(start_tpc: int, count: int) -> int:
    """把"启用从 start_tpc 开始的 count 个 TPC"翻译成 libsmctrl 的禁用掩码。

    上游约定置位为**禁用**，所以要取反。超出设备实际 TPC 数的高位被置为禁用，
    与上游示例 ``libsmctrl_set_stream_mask(stream, ~0b00111100ull)`` 一致。
    """
    if count < 0 or start_tpc < 0:
        raise ValueError("start_tpc 与 count 不能为负")
    enabled_bits = ((1 << count) - 1) << start_tpc
    return (~enabled_bits) & ((1 << 64) - 1)


@dataclass
class LibSmCtrlAdapter:
    """把项目契约 ``(stream_handle, sm_fraction)`` 映射到真实 TPC 掩码。

    两个流各自调用本适配器时会拿到**互不重叠**的 TPC 区间，否则"两个请求各占
    一半资源"只是掩码重叠的假象。配额之和超过设备 TPC 总数时显式失败，而不是
    悄悄给出重叠分配。
    """

    cuda_device: int = 0
    library_path: str | None = None
    _libcuda: ctypes.CDLL | None = field(default=None, repr=False)
    _total_tpcs: int | None = field(default=None, repr=False)
    _allocations: dict[int, tuple[int, int]] = field(default_factory=dict, repr=False)
    _last_error: str | None = field(default=None, repr=False)

    def _ensure_total_tpcs(self) -> int:
        if self._total_tpcs is None:
            self._total_tpcs = tpc_count(self.cuda_device)
        return self._total_tpcs

    def probe(self) -> dict:
        """启动前能力探针：库能否加载、驱动版本是否受支持、TPC 能否查询。"""
        result: dict = {"adapter": type(self).__name__, "cuda_device": self.cuda_device}
        try:
            libcuda = self._libcuda or load_libcuda()
            self._libcuda = libcuda
        except (LibSmCtrlUnavailable, OSError) as exc:
            result.update(available=False, stage="libcuda", reason=str(exc))
            return result
        try:
            info = driver_info(libcuda)
        except (LibSmCtrlUnavailable, AttributeError, OSError) as exc:
            result.update(available=False, stage="driver_query", reason=str(exc))
            return result
        result.update(driver_cuda_version=info.version, driver_label=info.label, driver_source=info.source)
        if not info.supported:
            result.update(
                available=False,
                stage="driver_version",
                reason=(
                    f"驱动报告的 CUDA 版本为 {info.label}（{info.version}），不在 libsmctrl "
                    "set_stream_mask 的受支持白名单内；上游实现在该分支会调用 exit(1) 终止进程，"
                    "因此拒绝下发真实掩码"
                ),
            )
            return result
        try:
            tpcs = self._ensure_total_tpcs()
        except (LibSmCtrlUnavailable, OSError) as exc:
            result.update(available=False, stage="tpc_query", reason=str(exc))
            return result
        result.update(available=True, stage="ready", total_tpcs=tpcs)
        return result

    def _allocate(self, handle: int, count: int) -> int | None:
        """为 handle 分配 count 个连续且与其他流不重叠的 TPC，返回起始编号。"""
        total = self._ensure_total_tpcs()
        others = sorted((start, start + size) for h, (start, size) in self._allocations.items() if h != handle)
        cursor = 0
        for start, end in others:
            if start - cursor >= count:
                return cursor
            cursor = max(cursor, end)
        return cursor if total - cursor >= count else None

    def apply_quota(self, stream_handle: int, sm_fraction: float) -> dict:
        """项目契约入口。stream_handle 必须是真实 CUDA stream（CUstream 指针值）。"""
        if not isinstance(stream_handle, int) or stream_handle == 0:
            return {
                "enforced": False,
                "reason": "stream_handle 必须是有效的 CUDA stream 句柄；worker id 不是 stream，拒绝下发掩码",
            }
        if self._libcuda is None or self._total_tpcs is None:
            probe = self.probe()
            if not probe.get("available"):
                return {"enforced": False, "reason": probe.get("reason", "适配器未就绪"), "probe": probe}
        total = self._ensure_total_tpcs()
        want = fraction_to_enabled_tpcs(sm_fraction, total)
        start = self._allocate(stream_handle, want)
        if start is None:
            used = sum(size for _, size in self._allocations.values())
            return {
                "enforced": False,
                "reason": f"TPC 池不足：已分配 {used}/{total}，本次请求 {want}；配额之和不得超过设备 TPC 总数",
                "total_tpcs": total,
                "allocated_tpcs": used,
            }
        native_mask = enabled_tpcs_to_native_mask(start, want)
        lib = load_libsmctrl(self.library_path)
        # 返回 void：这里只能确认"已下发"，无法据此确认硬件已生效。
        lib.libsmctrl_set_stream_mask(ctypes.c_void_p(stream_handle), ctypes.c_uint64(native_mask))
        self._allocations[stream_handle] = (start, want)
        return {
            "enforced": True,
            "applied": True,
            "verified_by_measurement": False,
            "requested_sm_fraction": sm_fraction,
            "effective_sm_fraction": want / total,
            "tpc_start": start,
            "tpc_count": want,
            "total_tpcs": total,
            "native_mask": native_mask,
            "stream_handle": stream_handle,
            "note": "libsmctrl_set_stream_mask 返回 void，此处仅表示已按受支持驱动版本下发；是否真正生效需由独立测量验证",
        }

    def describe(self) -> dict:
        return {
            "adapter": type(self).__name__,
            "cuda_device": self.cuda_device,
            "total_tpcs": self._total_tpcs,
            "active_streams": len(self._allocations),
        }


_default_adapter: LibSmCtrlAdapter | None = None


def _adapter() -> LibSmCtrlAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = LibSmCtrlAdapter()
    return _default_adapter


def apply_quota(stream_handle: int, sm_fraction: float) -> dict:
    """模块级入口，供 ``libsmctrl_adapter: "encoder_sched.libsmctrl_adapter:apply_quota"`` 使用。"""
    return _adapter().apply_quota(stream_handle, sm_fraction)


def probe() -> dict:
    """模块级能力探针入口。

    资源后端会调用它来判断本环境是否真的能下发 TPC 掩码。探针未通过时后端
    必须拒绝启用 ``enforces_sm_partition``，否则会把静默 no-op 当成硬件隔离。
    """
    return _adapter().probe()


def describe() -> dict:
    return _adapter().describe()
