"""诊断：GPU 时钟状态是不是由"测量占空比"决定的，而延迟又随之改变。

已知：负载下时钟 2 秒内可从 1890 升到 2460+，但实验全程停在 1890。
怀疑：测量是突发式的（每次几毫秒、间隔几十毫秒），占空比太低，调速器不触发升频。

三种排布，跑同一组请求，记录**每次测量前**的 SM 时钟与执行时间：
  DENSE   连续测，无间隔          —— 占空比高
  SPARSE  每次之间睡 150 ms       —— 占空比低（复刻常规标定扫描）
  BURST   先跑 2 s 连续负载预热，再单次测 —— 验证"历史"能否决定单次测量的快慢

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/diag_duty_cycle.py
"""

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
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402
from encoder_sched.libsmctrl_adapter import enabled_tpcs_to_native_mask, load_libsmctrl  # noqa: E402
import ctypes  # noqa: E402

TPCS = 3
SIZE = 224
REPS = 30


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
    mask = enabled_tpcs_to_native_mask(0, TPCS)

    def job(tag, size=SIZE):
        j = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                      width=size, height=size, deadline_ms=600_000)
        j.sm_fraction = TPCS / 12
        return j

    def forward(tag, size=SIZE):
        return enc.encode(job(tag, size), 0).execution_ms

    def set_mask():
        lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask))

    # 大矩阵，用于"预热/占空比"负载
    big_a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    big_b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    def heat(seconds):
        # torch.mm 是异步的：不逐次同步的话，2 s 内能入队上千个 matmul，
        # 之后 synchronize() 要等它们全部跑完（实测约 14 s），完全不是"预热 2 s"。
        end = time.time() + seconds
        while time.time() < end:
            torch.mm(big_a, big_b)
            torch.cuda.synchronize()

    def run(phase):
        def go():
            set_mask()
            for _ in range(3):
                forward(f"warm-{phase}")
            torch.cuda.synchronize()
            rows = []
            for i in range(REPS):
                if phase == "DENSE":
                    pass
                elif phase == "SPARSE":
                    time.sleep(0.150)
                elif phase == "BURST":
                    heat(2.0)
                clk = torch.cuda.clock_rate()
                ts = time.perf_counter()
                ms = forward(f"{phase}-{i}")
                wall = (time.perf_counter() - ts) * 1000
                rows.append({"i": i, "clk_before": clk, "exec_ms": ms, "wall_ms": wall})
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(0))
            return rows
        return w.call(go)

    print(f"CLIP {SIZE}x{SIZE} @ {TPCS} TPC，{REPS} 次/相位\n")
    results = {}
    # 每个相位前先空载 3 s，保证起点一致
    for phase in ("DENSE", "SPARSE", "BURST"):
        time.sleep(3.0)
        rows = run(phase)
        results[phase] = rows
        ms = [r["exec_ms"] for r in rows]
        clk = [r["clk_before"] for r in rows]
        print(f"[{phase}]")
        print(f"  时钟(MHz) : min={min(clk)} max={max(clk)} 中位={statistics.median(clk)}")
        print(f"  执行(ms)  : 中位={statistics.median(ms):.3f} 极差={max(ms)/min(ms):.2f}×")
        print(f"  前5次     : {[round(r['clk_before']) for r in rows[:5]]} MHz / "
              f"{[round(r['exec_ms'],2) for r in rows[:5]]} ms")
        print(f"  后5次     : {[round(r['clk_before']) for r in rows[-5:]]} MHz / "
              f"{[round(r['exec_ms'],2) for r in rows[-5:]]} ms")
        print()

    print("=" * 76)
    print(f"{'相位':>8} {'中位时钟':>10} {'中位执行':>10} {'执行极差':>10}")
    for phase, rows in results.items():
        ms = [r["exec_ms"] for r in rows]
        clk = [r["clk_before"] for r in rows]
        print(f"{phase:>8} {statistics.median(clk):>10.0f} {statistics.median(ms):>10.3f} "
              f"{max(ms)/min(ms):>9.2f}×")

    import json
    with open("/tmp/diag_duty_cycle.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
