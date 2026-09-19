"""试点实验：启用 TPC 数 -> 有效显存带宽 / 算力，同时记录 SM 时钟。

目的：为「空间分区为什么在小分辨率视觉编码器上失效」提供机制证据，
并顺带检验「配额-延迟曲线 88% 不一致」是否由 DVFS 时钟状态导致。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_tpc_bandwidth.py
"""

import ctypes
import json
import os
import queue
import statistics
import subprocess
import sys
import threading

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter  # noqa: E402

MIB = 1024 * 1024


def sm_clock() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    return out


class Worker:
    """专用线程：掩码是 __thread 的，必须在同一线程内施加并测量。"""

    def __init__(self):
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

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
            pass
        if "e" in box:
            raise box["e"]
        return box["r"]


def main():
    adapter = LibSmCtrlAdapter()
    probe = adapter.probe()
    print("probe:", json.dumps(probe, ensure_ascii=False))
    if not probe.get("available"):
        raise SystemExit("掩码不可用，终止")

    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0), "| L2:", torch.cuda.get_device_properties(0).L2_cache_size // MIB, "MiB")

    # 256 MiB 工作集，远大于 L2，确保打到 HBM
    n = 64 * MIB  # float32 元素数 -> 256 MiB
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    # 计算受限参照：4096^3 fp16 matmul
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)

    def timed(fn, warmup=3, reps=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    w = Worker()
    total = adapter._ensure_total_tpcs()
    rows = []
    for count in range(1, total + 1):
        frac = count / total

        def setmask(c=count):
            r = adapter.apply_quota(1, c / total)
            if not r.get("enforced"):
                raise RuntimeError(r)
            return r

        info = w.call(setmask)

        def run():
            clk0 = sm_clock()
            t_copy = timed(lambda: y.copy_(x))
            t_read = timed(lambda: x.sum())
            t_mm = timed(lambda: torch.mm(a, b))
            clk1 = sm_clock()
            return t_copy, t_read, t_mm, clk0, clk1

        t_copy, t_read, t_mm, clk0, clk1 = w.call(run)
        bw_copy = (2 * 256 * MIB) / (t_copy * 1e-3) / 1e9  # GB/s，读+写
        bw_read = (256 * MIB) / (t_read * 1e-3) / 1e9
        tflops = (2 * 4096**3) / (t_mm * 1e-3) / 1e12
        row = {
            "tpcs": count, "frac": round(frac, 3), "tpc_span": f"{info['tpc_start']}+{info['tpc_count']}",
            "copy_ms": round(t_copy, 3), "bw_copy_GBs": round(bw_copy, 1),
            "read_ms": round(t_read, 3), "bw_read_GBs": round(bw_read, 1),
            "mm_ms": round(t_mm, 3), "tflops": round(tflops, 2),
            "clk_before": clk0, "clk_after": clk1,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with open("/tmp/pilot_tpc_bandwidth.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n已写入 /tmp/pilot_tpc_bandwidth.json")


if __name__ == "__main__":
    main()
