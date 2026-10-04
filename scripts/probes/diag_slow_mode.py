"""诊断：CLIP 的"慢模态"是环境噪声，还是多 kernel 前向自身的性质？

已确认（/tmp/repro_tail）：224×224 的 CLIP 前向有 22% 的样本落在中位数的 3.3 倍上，
672×672 只有 2%；且慢样本与快样本的 SM 时钟完全相同（都是 1890 / 2145），
滞后自相关仅 +0.03（不是持久状态，是独立事件）。

两种解释：
  E1 外部环境噪声——那所有负载都应该有差不多的慢样本比例
  E2 多 kernel 前向的固有性质（kernel 之间可加性失效）——单 kernel 负载应该没有慢模态

四组负载，同一条件（紧挨着连测）、每组 300 次，比较慢模态占比与慢样本的超额时长。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/diag_slow_mode.py
"""

import ctypes
import json
import queue
import statistics
import sys
import threading
import time
import zlib

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.libsmctrl_adapter import enabled_tpcs_to_native_mask, load_libsmctrl  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

REPS = 300


class Worker:
    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            fn, box = self.q.get()
            if fn is None:
                return
            try:
                box["r"] = fn()
            except Exception as e:  # noqa: BLE001
                box["e"] = e

    def call(self, fn):
        box = {}
        self.q.put((fn, box))
        while not box:
            time.sleep(0.001)
        if "e" in box:
            raise box["e"]
        return box["r"]


def main():
    cfg = load_config("config.libsmctrl.example.yaml")
    lib = load_libsmctrl()
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    w = Worker()
    mask12 = enabled_tpcs_to_native_mask(0, 12)

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    m2a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    m2b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    def clip_job(size, tag):
        j = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                      width=size, height=size, deadline_ms=600_000)
        j.sm_fraction = 1.0
        return j

    def timed(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    loads = {
        "CLIP 224×1 (~3-4ms, 几十个kernel)": lambda i: enc.encode(clip_job(224, f"c224-{i}"), 0).execution_ms,
        "CLIP 672×1 (~10ms, 几十个kernel)": lambda i: enc.encode(clip_job(672, f"c672-{i}"), 0).execution_ms,
        "matmul 4096³ ×1 (~5ms, 1个kernel)": lambda i: timed(lambda: torch.mm(a, b)),
        "matmul 1024³ ×20 (~5ms, 20个kernel)": lambda i: timed(
            lambda: [torch.mm(m2a, m2b) for _ in range(20)]),
    }

    results = {}
    for name, fn in loads.items():
        def go(name=name, fn=fn):
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask12))
            for i in range(20):
                fn(-1 - i)          # 暖机（用负索引避免与正式样本共用 seed）
            torch.cuda.synchronize()
            out = []
            for i in range(REPS):
                clk = torch.cuda.clock_rate()
                t0 = time.perf_counter()
                ms = fn(i)
                out.append({"exec_ms": ms, "wall_ms": (time.perf_counter() - t0) * 1000,
                            "sm_clock": clk, "t": time.time(), "i": i})
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
            return out
        rows = w.call(go)
        results[name] = rows

    print(f"{'负载':>42} {'中位(ms)':>9} {'慢模态占比':>10} {'慢样本中位':>11} {'超额':>8} {'时钟差':>8}")
    print("-" * 100)
    for name, rows in results.items():
        x = [r["exec_ms"] for r in rows]
        med = statistics.median(x)
        slow = [r for r in rows if r["exec_ms"] > 2 * med and r["exec_ms"] - med >= 2.0]
        fast = [r for r in rows if r not in slow]
        sc = [r["sm_clock"] for r in slow] or [0]
        fc = [r["sm_clock"] for r in fast] or [0]
        slow_med = statistics.median([r["exec_ms"] for r in slow]) if slow else float("nan")
        print(f"{name:>42} {med:>9.3f} {100*len(slow)/len(rows):>9.0f}% "
              f"{slow_med:>11.3f} {slow_med-med:>8.2f} "
              f"{statistics.median(sc)-statistics.median(fc):>8.0f}")

    with open("/tmp/diag_slow_mode.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n已写入 /tmp/diag_slow_mode.json")


if __name__ == "__main__":
    main()
