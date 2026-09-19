"""审核实验 3：把审核实验 2 的"严格协议"搬到真实 CLIP 前向上，看曲线是否仍然可复现。

已确立（审核实验 2）：单个 kernel 的配额曲线在严格协议下可复现到 ±1–2%，
matmul 3/6/12 TPC = 17.5/8.9/4.56 ms（4× 配额给出 3.84× 加速，接近理想）。
因此 `profile_encoder.py` 与 `experiment_sm_quota.py` 之间 88% 的差异**不可能**来自掩码本身。

本实验要定位它到底藏在哪里。三种口径测同一件事（CLIP 前向 224/672 × 3/6/12 TPC）：
  A 严格协议   单线程、紧循环、每档测 30 次取中位数，交错顺序
  B 复刻现状   完全照 profile_encoder.py 的顺序（尺寸外层、配额内层、warmup 2 + 5 次）
  C 冷启动     每档切换后立刻测第一次（不暖机），看首次测量偏多少

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/audit_clip_reproducibility.py
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
from encoder_sched.libsmctrl_adapter import load_libsmctrl  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402

TOTAL = 12
SIZES = [224, 672]
TPCS = [3, 6, 12]


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

    # 直接用 CLIP 编码器，但绕开项目的资源后端——掩码由本脚本按 TPC 集合精确指定，
    # 以便测"非前缀"集合而不受分配器影响。
    from encoder_sched.resource import ProxyResourceBackend
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    w = Worker()
    out = {}

    def forward(size, tpcs):
        def go():
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(list(range(tpcs)))))
            job = EncodeJob(f"a-{size}-{tpcs}-{zlib.crc32(str(time.time_ns()).encode())}",
                            seed=1, width=size, height=size, deadline_ms=60_000)
            return enc.encode(job, 0).execution_ms
        return w.call(go)

    def warmup_all():
        for size in SIZES:
            for t in TPCS:
                for _ in range(2):
                    forward(size, t)

    print("预热中……", flush=True)
    warmup_all()
    print("完成\n", flush=True)

    # ---------- A 严格协议：交错，每格 30 次 ----------
    print("=== A 严格协议（交错随机顺序，每格 30 次中位数）===")
    cells = {(s, t): [] for s in SIZES for t in TPCS}
    plan = [(s, t) for s in SIZES for t in TPCS] * 30
    import random
    random.Random(11).shuffle(plan)
    for s, t in plan:
        cells[(s, t)].append(forward(s, t))
    strict = {f"{s}x{s}@{t}TPC": statistics.median(v) for (s, t), v in cells.items()}
    for s in SIZES:
        print(f"  {s}x{s}: " + "  ".join(f"{t}TPC={strict[f'{s}x{s}@{t}TPC']:7.3f}ms" for t in TPCS))
    out["strict"] = strict

    # ---------- C 冷启动：切档后立刻测第一次 ----------
    print("\n=== C 冷启动：每档切换后第一次测量 ===")
    cold = {}
    for size in SIZES:
        vals = []
        for t in TPCS:
            forward(size, 12 if t != 12 else 3)   # 先把掩码切到别的档，制造"刚切换"
            v = forward(size, t)
            vals.append(v)
        cold[f"{size}x{size}"] = vals
        print(f"  {size}x{size}: " + "  ".join(f"{t}TPC={v:8.3f}ms" for t, v in zip(TPCS, vals)))
    out["cold_first"] = cold

    # ---------- B 复刻 profile_encoder 的顺序，重复 3 轮 ----------
    print("\n=== B 复刻 profile_encoder.py 顺序（尺寸外层/配额内层，warmup2+5 次中位数）===")
    rounds = []
    for rd in range(3):
        row = {}
        for size in SIZES:
            for t in TPCS:
                samples = []
                for i in range(2 + 5):
                    v = forward(size, t)
                    if i >= 2:
                        samples.append(v)
                row[f"{size}x{size}@{t}TPC"] = statistics.median(samples)
        rounds.append(row)
        print(f"  轮{rd}: " + "  ".join(f"{k.split('@')[0]}@{k.split('@')[1]}={v:.2f}" for k, v in list(row.items())[:3]))
    out["profile_order_rounds"] = rounds

    print("\n=== 汇总：跨轮/跨协议的离散度 ===")
    print(f"  {'档位':>14} {'严格协议':>10} {'复刻现状 三轮':>34} {'极差':>8}")
    for size in SIZES:
        for t in TPCS:
            k = f"{size}x{size}@{t}TPC"
            vals = [r[k] for r in rounds]
            spread = max(vals) / min(vals)
            print(f"  {k:>14} {strict[k]:>10.3f} {str([round(v,2) for v in vals]):>34} {spread:>7.3f}×")

    with open("/tmp/audit_clip_reproducibility.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 /tmp/audit_clip_reproducibility.json")


if __name__ == "__main__":
    main()
