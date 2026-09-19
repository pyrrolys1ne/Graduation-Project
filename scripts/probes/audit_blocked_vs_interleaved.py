"""审核实验 2：同一测量，"分块顺序" vs "交错随机" 的复现性对比。

动机：本项目 #1 待办是"配额—延迟曲线不可复现"。审核实验中发现一个线索——
  · 分块式测量（profile_encoder.py 的做法：一档跑完再跑下一档）在两次相隔约 20 分钟的
    窗口之间，同一档位（不加掩码！）差 27%（173.5 vs 219.8 GB/s）；
  · 而交错随机测量（审核实验第二部分）在不同 TPC 子集之间的极差只有 <1%。
若成立，则"不可复现"不是掩码或硬件的性质，而是**测量协议**的性质——这是一个可直接交付的
方法学结论，且是 SpaceServe 等按算子顺序剖析的工作共有的隐患。

设计：3 个配额档 × 3 轮，A/B/A/B/A/B 交替（控制时间漂移）
  blocked       每轮内：一档连续测 30 次 → 中位数；再下一档
  interleaved   每轮内：30 个打乱顺序的轮次，每轮依次测 3 档
判据：比较**跨轮**的同档位离散度（同一协议重复三次的一致性）。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/audit_blocked_vs_interleaved.py
"""

import ctypes
import json
import queue
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.libsmctrl_adapter import load_libsmctrl  # noqa: E402

LEVELS = [3, 6, 12]
TOTAL = 12
ROUNDS = 3           # 每个协议重复几轮
INNER = 30           # 每档每轮内测几次
MIB = 1024 * 1024


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
    lib = load_libsmctrl()
    dev = torch.device("cuda")
    ma = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    mb = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    n = 64 * MIB
    x = torch.randn(n, device=dev); y = torch.empty_like(x)
    w = Worker()

    def one(lv, kind):
        def go():
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask_for(list(range(lv)))))
            body = (lambda: torch.mm(ma, mb)) if kind == "mm" else (lambda: y.copy_(x))
            body(); torch.cuda.synchronize()
            t0 = time.perf_counter(); body(); torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000
        return w.call(go)

    rng = random.Random(7)
    out = {}
    for kind in ("mm", "mem"):
        print(f"\n===== {'计算受限 matmul' if kind=='mm' else '访存受限 copy'} =====")
        # 暖机
        for lv in LEVELS:
            for _ in range(3):
                one(lv, kind)

        results = {"blocked": [], "interleaved": []}
        for rd in range(ROUNDS):
            for proto in ("blocked", "interleaved"):
                if proto == "blocked":
                    cell = {}
                    for lv in LEVELS:
                        cell[lv] = statistics.median([one(lv, kind) for _ in range(INNER)])
                else:
                    bucket = {lv: [] for lv in LEVELS}
                    plan = LEVELS * INNER
                    rng.shuffle(plan)
                    for lv in plan:
                        bucket[lv].append(one(lv, kind))
                    cell = {lv: statistics.median(v) for lv, v in bucket.items()}
                results[proto].append(cell)
                print(f"  轮{rd} {proto:>12}: " +
                      "  ".join(f"{lv}TPC={cell[lv]:7.3f}ms" for lv in LEVELS), flush=True)

        print(f"\n  {'档位':>6} | {'分块 三轮值':>30} {'极差':>7} | {'交错 三轮值':>30} {'极差':>7}")
        rec = {}
        for lv in LEVELS:
            b = [r[lv] for r in results["blocked"]]
            i = [r[lv] for r in results["interleaved"]]
            rb = max(b) / min(b); ri = max(i) / min(i)
            rec[lv] = {"blocked": b, "interleaved": i, "spread_blocked": rb, "spread_interleaved": ri}
            print(f"  {lv:>4}TPC | {str([round(v,2) for v in b]):>30} {rb:>6.3f}× | "
                  f"{str([round(v,2) for v in i]):>30} {ri:>6.3f}×")
        out[kind] = rec

    with open("/tmp/audit_blocked_vs_interleaved.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 /tmp/audit_blocked_vs_interleaved.json")


if __name__ == "__main__":
    main()
