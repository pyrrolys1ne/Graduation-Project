"""诊断：掩码"设置一次"与"每轮设置+释放"之间的开销差异。

背景：审计脚本 `audit_clip_reproducibility.py` 每轮只 set_thread_mask（不释放），
      224@3TPC 测得 3.40 ms；而新实验脚本每轮走适配器的 apply_quota + release_quota
      （设掩码 → 清零 → 再设），同一配置测得 7.5-7.9 ms。差 2.2 倍。

三种口径：
  A 只设一次，之后不再动掩码
  B 每轮重复设成同一个值（无释放）
  C 每轮 设掩码 → 释放(清零) → 再设（即服务实际走的路径）

若 C 显著慢于 B，则"每请求释放配额"本身有代价——这会直接意味着
**标定表（通常静态设掩码）与真实服务（每请求切换）测的不是同一个量**。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/diag_mask_toggle.py
"""

import ctypes
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
from encoder_sched.libsmctrl_adapter import load_libsmctrl  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

TPCS = 3
TOTAL = 12
SIZE = 224
REPS = 12


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


def mask_for(tpcs):
    e = 0
    for t in tpcs:
        e |= 1 << t
    return (~e) & ((1 << 64) - 1)


def main():
    cfg = load_config("config.libsmctrl.example.yaml")
    lib = load_libsmctrl()
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    w = Worker()

    def job(tag):
        j = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31), width=SIZE, height=SIZE, deadline_ms=600_000)
        j.sm_fraction = TPCS / TOTAL
        return j

    def forward(tag):
        return enc.encode(job(tag), 0).execution_ms

    # 暖机
    for i in range(4):
        forward(f"warm-{i}")
    torch.cuda.synchronize()

    def run(kind):
        def go():
            ts = []
            if kind == "A":
                lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(range(TPCS))))
            for i in range(REPS):
                if kind == "B":
                    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(range(TPCS))))
                elif kind == "C":
                    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(range(TPCS))))
                    lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
                ts.append(forward(f"{kind}-{i}"))
            if kind in ("B", "C"):
                lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
            return ts
        return w.call(go)

    print(f"CLIP {SIZE}x{SIZE} @ {TPCS} TPC，每档 {REPS} 次")
    print(f"{'口径':>4} {'说明':>34} {'中位数':>10} {'均值':>10} {'极差':>8}")
    for kind, desc in (("A", "只设一次，之后不动掩码"),
                       ("B", "每轮重复设成同一个值（不释放）"),
                       ("C", "每轮 设置→清零→再设置")):
        ts = run(kind)
        print(f"{kind:>4} {desc:>34} {statistics.median(ts):>10.3f} {statistics.fmean(ts):>10.3f} "
              f"{max(ts)/min(ts):>7.2f}×")
        print(f"     样本: {[round(t,2) for t in ts]}")


if __name__ == "__main__":
    main()
