"""试点 2：把「掩码效应」与「DVFS 时钟漂移」分离开。

设计：
  阶段 1  空载时钟采样 12 s —— 看时钟自己会不会漂、漂多少
  阶段 2  交错 / 随机顺序重复测量 3、6、12 TPC 三档，每次测量前后各采一次时钟
  分析    用测量窗口内的时钟把墙钟时间归一化成「SM-周期数」，看归一化后掩码效应是否还在

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_clock_vs_mask.py
"""

import json
import queue
import random
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter  # noqa: E402

MIB = 1024 * 1024
QUERY = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"


def sample_gpu() -> dict:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    sm, mem, pw, temp, util = [p.strip() for p in out.split(",")]
    return {"sm": int(sm), "mem": int(mem), "power": float(pw), "temp": int(temp), "util": int(util)}


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
    adapter = LibSmCtrlAdapter()
    assert adapter.probe().get("available")
    total = adapter._ensure_total_tpcs()
    dev = torch.device("cuda")

    n = 64 * MIB
    x = torch.randn(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    w = Worker()

    print("=== 阶段 1：空载时钟采样 12 s ===", flush=True)
    idle = []
    for _ in range(24):
        idle.append(sample_gpu())
        time.sleep(0.5)
    sm_idle = [s["sm"] for s in idle]
    print(f"空载 SM 时钟: min={min(sm_idle)} max={max(sm_idle)} "
          f"mean={statistics.fmean(sm_idle):.0f} 跨度={max(sm_idle)-min(sm_idle)} MHz", flush=True)

    def measure_once(tpcs):
        def go():
            r = adapter.apply_quota(1, tpcs / total)
            assert r.get("enforced"), r
            # 暖机，避免把冷启动算进去
            for _ in range(3):
                torch.mm(a, b); y.copy_(x); x.sum()
            torch.cuda.synchronize()
            g0 = sample_gpu()
            t0 = time.perf_counter()
            for _ in range(5):
                torch.mm(a, b); y.copy_(x); x.sum()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000 / 5
            g1 = sample_gpu()
            return {"tpcs": tpcs, "wall_ms": wall, "sm_before": g0["sm"], "sm_after": g1["sm"],
                    "mem_before": g0["mem"], "mem_after": g1["mem"],
                    "pwr": g1["power"], "temp": g1["temp"]}
        return w.call(go)

    print("\n=== 阶段 2：交错随机顺序，3/6/12 TPC 各 15 轮 ===", flush=True)
    levels = [3, 6, 12]
    plan = levels * 15
    random.Random(20260915).shuffle(plan)
    rows = []
    for i, lv in enumerate(plan):
        row = measure_once(lv)
        row["round"] = i
        rows.append(row)
        print(f"  #{i:02d} tpc={lv:2d} wall={row['wall_ms']:7.3f} ms  sm={row['sm_before']}->{row['sm_after']} "
              f"mem={row['mem_before']} pwr={row['pwr']:.1f}W", flush=True)

    print("\n=== 分析 ===", flush=True)
    print(f"{'TPC':>4} {'n':>3} {'wall均值':>10} {'wall标准差':>11} {'SM时钟均值':>11} {'归一化(ms·MHz/2610)':>21}")
    summary = {}
    for lv in levels:
        sub = [r for r in rows if r["tpcs"] == lv]
        walls = [r["wall_ms"] for r in sub]
        clocks = [(r["sm_before"] + r["sm_after"]) / 2 for r in sub]
        # 归一化到 2610 MHz：墙钟 × (clk/2610)，即把不同时钟折算到同一时钟下的等效时间
        norm = [r["wall_ms"] * c / 2610 for r, c in zip(sub, clocks)]
        summary[lv] = {"n": len(sub), "wall_mean": statistics.fmean(walls), "wall_sd": statistics.stdev(walls),
                       "clk_mean": statistics.fmean(clocks),
                       "norm_mean": statistics.fmean(norm), "norm_sd": statistics.stdev(norm)}
        print(f"{lv:>4} {len(sub):>3} {statistics.fmean(walls):>10.3f} {statistics.stdev(walls):>11.3f} "
              f"{statistics.fmean(clocks):>11.0f} {statistics.fmean(norm):>21.3f}")

    base = summary[12]
    print(f"\n相对 12 TPC：")
    for lv in levels:
        s = summary[lv]
        print(f"  {lv:>2} TPC  原始比值={s['wall_mean']/base['wall_mean']:.3f}   "
              f"时钟归一化后比值={s['norm_mean']/base['norm_mean']:.3f}")

    # 墙钟与时钟的相关性
    walls = [r["wall_ms"] for r in rows]
    clks = [(r["sm_before"] + r["sm_after"]) / 2 for r in rows]
    mw, mc = statistics.fmean(walls), statistics.fmean(clks)
    cov = sum((a - mw) * (b - mc) for a, b in zip(walls, clks))
    denom = (sum((a - mw) ** 2 for a in walls) * sum((b - mc) ** 2 for b in clks)) ** 0.5
    print(f"\n墙钟时间与 SM 时钟的皮尔逊相关系数: {cov/denom:+.3f}")

    with open("/tmp/pilot_clock_vs_mask.json", "w") as f:
        json.dump({"idle": idle, "rows": rows, "summary": summary}, f, indent=2)
    print("已写入 /tmp/pilot_clock_vs_mask.json")


if __name__ == "__main__":
    main()
