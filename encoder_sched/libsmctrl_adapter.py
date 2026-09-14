"""基于真实 libsmctrl 的 TPC 掩码适配器（回调路径 + 粘性线程掩码）。

上游来源：``http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git``（原仓库）。
``github.com/atomicapple0/libsmctrl`` 只是 RTAS23 论文的冻结快照 fork。

掩码语义（逐字取自上游 ``libsmctrl.h``）::

    A set bit in the mask indicates that the respective Thread Processing
    Cluster (TPC) is to be __disabled__.

即**置位表示禁用**。启用 TPC ``0..n-1`` 需传 ``~((1 << n) - 1)``。

为什么走回调路径而不是 ``set_stream_mask``
------------------------------------------------

libsmctrl 提供两套互不相关的机制：

======================  ==========================  ==============================
机制                     代表函数                     版本门控
======================  ==========================  ==============================
硬编码结构体偏移          ``set_stream_mask``          按驱动版本白名单，未知版本
                                                      ``exit(1)`` **终止进程**
启动回调                  ``set_global_mask`` /        只有下界（<6.5），
                        ``set_next_mask``             运行时读 TMD 版本自适应
======================  ==========================  ==============================

后者能跨驱动版本工作，**恰恰因为它不硬编码版本**——它读运行时 TMD 版本字段决定写入偏移。
实测在 CUDA 13.3 + Ada（CC 8.9）上可用，而前者在任何 CUDA 13.x 驱动上都会终止进程。

上游的 ``set_next_mask`` 只覆盖**一次**启动，无法表达"一次请求（一次前向的几十个
kernel）用某个配额"。因此本项目对上游打了一个约 6 行的补丁
（``patches/libsmctrl-thread-mask.patch``），新增**粘性线程掩码**
``libsmctrl_set_thread_mask()``：设一次即对该线程后续所有启动生效。

与项目结构的对应关系
------------------------

``service.py`` 用 ``asyncio.to_thread(self.encoder.encode, ...)`` 把每个请求丢进独立
线程，``encode()`` 在**同一线程内**先调 ``apply_quota`` 再跑模型。因此**线程掩码就等于
请求掩码**，不需要额外的数据通路改造。

注意 ``apply_quota`` 的第一个参数（原为 CUDA stream 句柄）在回调路径下**不参与掩码**——
作用域由**调用线程**决定。该参数仍被校验并原样记录，以便与既有契约兼容。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LibSmCtrlAdapter",
    "LibSmCtrlUnavailable",
    "apply_quota",
    "describe",
    "enabled_tpcs_to_native_mask",
    "fraction_to_enabled_tpcs",
    "probe",
    "tpc_count",
]

#: 环境变量：显式指定 libsmctrl 共享库路径。
LIBRARY_PATH_ENV = "LIBSMCTRL_PATH"

_LIBCUDA_CANDIDATES = (
    "/usr/lib/wsl/lib/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
)

#: 回调路径所需的符号。缺失说明用的是未打补丁的上游库。
REQUIRED_SYMBOLS = ("libsmctrl_set_thread_mask", "libsmctrl_get_tpc_info_cuda")


class LibSmCtrlUnavailable(RuntimeError):
    """真实库或前置条件不满足。"""


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
    return ctypes.CDLL(_find_library(names, explicit))


def resolve_library_path(explicit: str | None = None) -> str:
    """解析补丁版 libsmctrl 共享库的实际路径（不加载）。

    探针需要在**子进程**中加载同一个库，因此必须能把解析结果传出去；
    仅传 ``None`` 会让子进程拿到空串，``ctypes.CDLL("")`` 会解析到 Python
    主程序本身并报出误导性的 undefined symbol。
    """
    explicit = explicit or os.environ.get(LIBRARY_PATH_ENV) or None
    candidates = (explicit,) if explicit else ("smctrl", "libsmctrl.so")
    return _find_library(tuple(c for c in candidates if c), explicit)


def load_libsmctrl(explicit: str | None = None) -> ctypes.CDLL:
    """加载本项目补丁版 libsmctrl 共享库。"""
    path = resolve_library_path(explicit)
    lib = ctypes.CDLL(path)
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        raise LibSmCtrlUnavailable(
            f"{path} 缺少符号 {missing}。libsmctrl_set_thread_mask 是本项目补丁新增的，"
            "请用 scripts/build_libsmctrl.sh 重新构建（会应用 patches/libsmctrl-thread-mask.patch）。"
        )
    # 这些函数都返回 void；显式声明避免 ctypes 把返回值当 int 处理。
    lib.libsmctrl_set_thread_mask.restype = None
    lib.libsmctrl_set_thread_mask.argtypes = [ctypes.c_uint64]
    lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
    lib.libsmctrl_get_tpc_info_cuda.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
    return lib


def driver_cuda_version(libcuda: ctypes.CDLL | None = None) -> int:
    """读取驱动报告的 CUDA 版本。仅作记录；回调路径不按它设限。"""
    lib = libcuda or load_libcuda()
    lib.cuDriverGetVersion.restype = ctypes.c_int
    lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int(0)
    rc = lib.cuDriverGetVersion(ctypes.byref(version))
    if rc != 0:
        raise LibSmCtrlUnavailable(f"cuDriverGetVersion 失败，rc={rc}")
    return version.value


def tpc_count(cuda_device: int = 0, library: ctypes.CDLL | None = None) -> int:
    """查询设备的 TPC 总数（上游按 SM 数除以每 TPC 的 SM 数换算）。

    ``library`` 可传入已加载的句柄以复用缓存的 ``dlopen`` 结果。
    """
    lib = library or load_libsmctrl()
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
    return max(1, min(total_tpcs, int(round(fraction * total_tpcs))))


def enabled_tpcs_to_native_mask(start_tpc: int, count: int) -> int:
    """把"启用从 start_tpc 开始的 count 个 TPC"翻译成 libsmctrl 的禁用掩码。

    上游约定置位为**禁用**，所以要取反。超出设备实际 TPC 数的高位被置为禁用，
    与上游示例 ``libsmctrl_set_stream_mask(stream, ~0b00111100ull)`` 一致。

    **注意**：``count <= 0`` 表示"不加限制"，返回 0。切勿传入 ``~0``——那会禁用
    全部 TPC，kernel 永远发不出去，进程直接挂死。
    """
    if start_tpc < 0:
        raise ValueError("start_tpc 不能为负")
    if count <= 0:
        return 0
    enabled_bits = ((1 << count) - 1) << start_tpc
    return (~enabled_bits) & ((1 << 64) - 1)


def _callback_path_works(library_path: str | None) -> tuple[bool, str | None]:
    """在**子进程**中验证回调注册是否会终止进程。

    ``setup_sm_control_callback()`` 在订阅/启用失败时会 ``abort()``，而 ``abort`` 被
    上游定义为 ``error_at_line``——第一个参数非零即 ``exit(1)``。这发生在**首次调用**
    时，若在服务进程内发生会直接把服务杀掉。所以用子进程试一次，把"可能杀进程"
    变成"报告不可用"。
    """
    code = (
        "import ctypes,sys;"
        "lib=ctypes.CDLL(sys.argv[1]);"
        "lib.libsmctrl_set_thread_mask.restype=None;"
        "lib.libsmctrl_set_thread_mask.argtypes=[ctypes.c_uint64];"
        "lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0));"  # 0 = 不加限制
        "print('OK')"
    )
    concrete = library_path or resolve_library_path(None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, concrete],
            capture_output=True, text=True, timeout=60, env={**os.environ},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"回调路径探针无法执行: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else "无输出"
        return False, f"回调注册在子进程中失败（退出码 {result.returncode}）: {detail}"
    if "OK" not in result.stdout:
        return False, "回调路径探针未返回预期结果"
    return True, None


@dataclass
class LibSmCtrlAdapter:
    """把项目契约 ``(handle, sm_fraction)`` 映射到**当前线程**的 TPC 掩码。

    作用域说明：回调路径的掩码是 ``__thread`` 的，因此**调用线程**决定了受影响的
    范围。项目里 ``encode()`` 在同一线程内先调本适配器再跑模型，所以线程掩码恰好
    等于请求掩码。第一个参数（原 CUDA stream 句柄）不参与掩码，仅被校验与记录。

    分配表按**线程 id** 建账：并发线程各自拿到互不重叠的 TPC 区间，否则"两个请求
    各占一半"只是掩码重叠的假象。配额之和超过设备 TPC 总数时显式失败。
    """

    cuda_device: int = 0
    library_path: str | None = None
    _total_tpcs: int | None = field(default=None, repr=False)
    _allocations: dict[int, tuple[int, int]] = field(default_factory=dict, repr=False)
    _library: ctypes.CDLL | None = field(default=None, repr=False)
    #: 服务中两个请求经 asyncio.to_thread 真正并发执行，会同时进入 apply_quota。
    #: 分配表是共享可变状态，必须串行化。
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: 最近一次探针结果。**apply_quota 以此为准，而不是看 `_library` 是否为 None**——
    #: 探针失败前已经加载了库与 TPC 数，若只看这两个字段，失败的探针会被绕过，
    #: 于是"探针报告不可用"却仍然下发掩码并返回 enforced=True。
    _probe_result: dict | None = field(default=None, repr=False)

    def _ensure_total_tpcs(self) -> int:
        if self._total_tpcs is None:
            self._total_tpcs = tpc_count(self.cuda_device, library=self._ensure_library())
        return self._total_tpcs

    def _ensure_library(self) -> ctypes.CDLL:
        if self._library is None:
            self._library = load_libsmctrl(self.library_path)
        return self._library

    def probe(self) -> dict:
        """启动前能力探针，并记录结果供 ``apply_quota`` 门控使用。"""
        result = self._probe_impl()
        self._probe_result = result
        return result

    def _probe_impl(self) -> dict:
        """库是否打了补丁、TPC 是否可查、回调能否安全注册。"""
        result: dict = {
            "adapter": type(self).__name__,
            "cuda_device": self.cuda_device,
            "mask_path": "callback",
        }
        try:
            self._ensure_library()
        except (LibSmCtrlUnavailable, OSError) as exc:
            result.update(available=False, stage="library", reason=str(exc))
            return result
        try:
            result["driver_cuda_version"] = driver_cuda_version()
        except (LibSmCtrlUnavailable, OSError, AttributeError) as exc:
            result.update(available=False, stage="driver_query", reason=str(exc))
            return result
        try:
            result["total_tpcs"] = self._ensure_total_tpcs()
        except (LibSmCtrlUnavailable, OSError) as exc:
            result.update(available=False, stage="tpc_query", reason=str(exc))
            return result
        ok, reason = _callback_path_works(self.library_path)
        if not ok:
            result.update(available=False, stage="callback_setup", reason=reason)
            return result
        result.update(available=True, stage="ready")
        return result

    def _allocate(self, key: int, count: int) -> int | None:
        """为 key（线程 id）分配 count 个连续且互不重叠的 TPC，返回起始编号。"""
        total = self._ensure_total_tpcs()
        others = sorted((s, s + n) for k, (s, n) in self._allocations.items() if k != key)
        cursor = 0
        for start, end in others:
            if start - cursor >= count:
                return cursor
            cursor = max(cursor, end)
        return cursor if total - cursor >= count else None

    def apply_quota(self, handle: int, sm_fraction: float) -> dict:
        """为**当前调用线程**设定粘性 TPC 掩码。

        ``handle`` 在回调路径下不参与掩码，但必须是一个有效标识（非零整数）。
        """
        if not isinstance(handle, int) or isinstance(handle, bool) or handle == 0:
            return {
                "enforced": False,
                "reason": "handle 必须是有效标识（非零整数）；拒绝设定掩码",
            }
        if not (self._probe_result and self._probe_result.get("available")):
            probe = self.probe()
            if not probe.get("available"):
                return {"enforced": False, "reason": probe.get("reason", "适配器未就绪"), "probe": probe}
        total = self._ensure_total_tpcs()
        want = fraction_to_enabled_tpcs(sm_fraction, total)
        lib = self._ensure_library()
        thread_id = threading.get_ident()
        with self._lock:
            start = self._allocate(thread_id, want)
            if start is None:
                used = sum(n for _, n in self._allocations.values())
                return {
                    "enforced": False,
                    "reason": f"TPC 池不足：已分配 {used}/{total}，本次请求 {want}；"
                              "配额之和不得超过设备 TPC 总数",
                    "total_tpcs": total,
                    "allocated_tpcs": used,
                }
            native_mask = enabled_tpcs_to_native_mask(start, want)
            # 返回 void：这里只能确认"已下发"，无法据此确认硬件已生效。
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(native_mask))
            self._allocations[thread_id] = (start, want)
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
            "thread_id": thread_id,
            "handle": handle,
            "mask_path": "callback",
            "note": "libsmctrl_set_thread_mask 返回 void，此处仅表示已下发；"
                    "是否真正生效需由独立测量验证",
        }

    def release(self) -> None:
        """清除**当前线程**的掩码并释放其分配。线程结束前调用可避免占坑。"""
        thread_id = threading.get_ident()
        with self._lock:
            self._allocations.pop(thread_id, None)
        if self._library is not None:
            self._library.libsmctrl_set_thread_mask(ctypes.c_uint64(0))

    def describe(self) -> dict:
        return {
            "adapter": type(self).__name__,
            "cuda_device": self.cuda_device,
            "mask_path": "callback",
            "total_tpcs": self._total_tpcs,
            "active_threads": len(self._allocations),
        }


_default_adapter: LibSmCtrlAdapter | None = None


def _adapter() -> LibSmCtrlAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = LibSmCtrlAdapter()
    return _default_adapter


def apply_quota(handle: int, sm_fraction: float) -> dict:
    """模块级入口，供 ``libsmctrl_adapter: "encoder_sched.libsmctrl_adapter:apply_quota"`` 使用。"""
    return _adapter().apply_quota(handle, sm_fraction)


def probe() -> dict:
    """模块级能力探针入口。

    资源后端会调用它判断本环境能否真实施加 TPC 掩码；探针未通过时后端必须拒绝
    启用 ``enforces_sm_partition``。
    """
    return _adapter().probe()


def release() -> None:
    """模块级释放入口：清除当前线程的掩码并释放其 TPC 分配。

    必须在每个请求结束时调用。否则空闲线程会继续占着旧分配，后续请求即使
    在别的线程上也会因为"池不足"而失败。
    """
    _adapter().release()


def describe() -> dict:
    return _adapter().describe()
