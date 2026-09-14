"""最小验证：回调路径 + 粘性线程掩码在 CUDA 13.x 上可用。

**要证明的两件事：**

1. **粘性**：`libsmctrl_set_thread_mask()` 设一次就对该线程后续**所有**启动生效，
   而上游的 `libsmctrl_set_next_mask()` 只影响**一次**启动。项目的 CLIP 前向一次
   要发几十个 kernel，所以只有前者能表达"按请求配额"。
2. **线程局部**：并发线程各自持不同掩码，互不干扰——这正是服务里
   `asyncio.to_thread` 每请求一线程的模型。

跑法::

    CUDA_HOME=/path/to/cuda scripts/build_libsmctrl.sh   # 会打上 patches/ 里的补丁
    LIBSMCTRL_PATH=<...>/libsmctrl.so python demos/libsmctrl_thread_mask_demo.py

注意：未打补丁的上游库没有 `libsmctrl_set_thread_mask`，脚本会明确报错退出。
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOTAL_TPCS = 12          # RTX 4060 Laptop：24 SM ÷ 2
LAUNCHES = 20            # 模拟一次前向的多次 kernel 启动


def load_library() -> ctypes.CDLL:
    path = os.environ.get("LIBSMCTRL_PATH")
    if not path:
        raise SystemExit("请先设置 LIBSMCTRL_PATH 指向构建好的 libsmctrl.so")
    lib = ctypes.CDLL(path)
    missing = [name for name in ("libsmctrl_set_thread_mask", "libsmctrl_set_next_mask", "libsmctrl_set_global_mask")
               if not hasattr(lib, name)]
    if missing:
        raise SystemExit(
            f"{path} 缺少符号 {missing}。\n"
            "libsmctrl_set_thread_mask 是本项目补丁新增的；请用 scripts/build_libsmctrl.sh 重新构建"
            "（该脚本会自动应用 patches/libsmctrl-thread-mask.patch）。"
        )
    for name in ("libsmctrl_set_thread_mask", "libsmctrl_set_next_mask", "libsmctrl_set_global_mask"):
        getattr(lib, name).restype = None
        getattr(lib, name).argtypes = [ctypes.c_uint64]
    return lib


def mask_for(enabled_tpcs: int) -> int:
    """启用前 n 个 TPC 对应的原生掩码（上游约定：置位为**禁用**）。

    ``enabled_tpcs <= 0`` 表示**不加限制**，返回 0（没有任何位被置为禁用）。
    注意不要传 ``~0``：那会禁用全部 TPC，kernel 永远发不出去，进程直接挂死。
    """
    if enabled_tpcs <= 0:
        return 0
    enabled = (1 << enabled_tpcs) - 1
    return (~enabled) & ((1 << 64) - 1)


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA 设备")

    lib = load_library()
    print(f"库: {os.environ['LIBSMCTRL_PATH']}")
    print(f"设备: {torch.cuda.get_device_name(0)} | TPC 总数: {TOTAL_TPCS}")
    print()

    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    lib.libsmctrl_set_global_mask(0)          # 清掉全局掩码，避免干扰本 Demo
    torch.cuda.synchronize()

    def per_launch_times(setter, mask: int) -> list[float]:
        """按 `setter` 的方式布防，记录每次启动的耗时。"""
        y = x @ x
        torch.cuda.synchronize()
        setter(ctypes.c_uint64(mask))
        times = []
        for _ in range(LAUNCHES):
            if setter is lib.libsmctrl_set_thread_mask:
                pass          # 粘性：只在开始时设一次
            t0 = time.perf_counter()
            y = x @ x
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return times

    # ---------- 阶段 1：粘性 vs 仅一次 ----------
    print("=" * 74)
    print(f"阶段 1：三种方式都**只布防一次**，然后连续 {LAUNCHES} 次启动（启用 1 个 TPC）")
    print("=" * 74)

    def measure(arm_once) -> list[float]:
        y = x @ x
        torch.cuda.synchronize()
        arm_once()                                   # 只在循环开始前布防一次
        times = []
        for _ in range(LAUNCHES):
            t0 = time.perf_counter()
            y = x @ x
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return times

    mask_1 = mask_for(1)
    t_base = measure(lambda: None)
    t_next = measure(lambda: lib.libsmctrl_set_next_mask(ctypes.c_uint64(mask_1)))
    t_thread = measure(lambda: lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_1)))
    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))   # 清除粘性掩码

    def covered(times: list[float]) -> int:
        """有多少次启动被掩码覆盖（显著慢于基线）。"""
        threshold = statistics.median(t_base) * 3
        return sum(1 for t in times if t > threshold)

    print(f"  无掩码基线           中位 {statistics.median(t_base):6.2f} ms")
    print(f"  next_mask 布防一次   第1次 {t_next[0]:6.2f} ms，其余 {statistics.median(t_next[1:]):5.2f} ms"
          f"  -> 覆盖 {covered(t_next)}/{LAUNCHES} 次")
    print(f"  thread_mask 布防一次 第1次 {t_thread[0]:6.2f} ms，其余 {statistics.median(t_thread[1:]):5.2f} ms"
          f"  -> 覆盖 {covered(t_thread)}/{LAUNCHES} 次")
    print()
    print("  next_mask 只覆盖 1 次启动；thread_mask 覆盖全部——这正是补丁要解决的问题。")
    sticky_ok = covered(t_thread) == LAUNCHES and covered(t_next) <= 2

    # ---------- 阶段 2：线程局部性 ----------
    print()
    print("=" * 74)
    print("阶段 2：一个线程持有的掩码不会影响其他线程")
    print("=" * 74)
    # 不并发测量：并发时三个 matmul 争用同一块 GPU，会把手持掩码的线程和
    # 不持掩码的线程一起拖慢，掩码效应被争用淹没。要证明的是"掩码是否跨线程
    # 泄漏"，所以让持掩码的线程存活等待，由主线程与它分别测量。
    results: dict[str, float] = {}
    armed = threading.Event()
    measured_worker = threading.Event()

    def worker() -> None:
        lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(1)))   # 只在本线程生效
        y = x @ x
        torch.cuda.synchronize()
        armed.set()                       # 掩码已就位，通知主线程
        measured_worker.wait(60)          # 保持存活，掩码随线程存在
        t0 = time.perf_counter()
        for _ in range(10):
            y = x @ x
        torch.cuda.synchronize()
        results["持掩码线程(1 TPC)"] = (time.perf_counter() - t0) / 10 * 1000

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    armed.wait(60)

    y = x @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    results["主线程(无掩码)"] = (time.perf_counter() - t0) / 10 * 1000

    measured_worker.set()
    thread.join(60)

    print(f"  持掩码线程(1 TPC) {results['持掩码线程(1 TPC)']:7.2f} ms/次  ← 受掩码限制")
    print(f"  主线程(无掩码)    {results['主线程(无掩码)']:7.2f} ms/次  ← 不受影响")
    ratio = results["持掩码线程(1 TPC)"] / results["主线程(无掩码)"]
    print()
    print(f"  比值 = {ratio:.2f}x。若掩码跨线程泄漏，主线程也会同样变慢。")
    isolated_ok = ratio > 3

    # ---------- 结论 ----------
    summary = {
        "baseline_ms": round(statistics.median(t_base), 3),
        "thread_mask_median_ms": round(statistics.median(t_thread), 3),
        "sticky_mask_demonstrated": sticky_ok,
        "per_thread_isolation_demonstrated": isolated_ok,
        "note": "本 Demo 只证明掩码机制可用；真实配额曲线与双流隔离对照见 scripts/experiment_sm_quota.py",
    }
    print()
    print("=" * 74)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 74)
    return 0 if (sticky_ok and isolated_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
