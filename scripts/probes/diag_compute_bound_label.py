"""判定：`audit_two_directions.py` 的"计算受限"臂，真的计算受限吗？

【为什么需要这一条】
§10.2 的四臂表里，"计算受限"一行给出的 `C66/S12 = 0.528×` 是**任务书里支撑
"分区被共享并发支配"最强的一个数字**。但换个算法它不成立：

    S12 的单作业 = 12 × mm(2048², fp16) = 206.2 GFLOP
    实测单作业 ≈ 6.85 ms  →  **约 30 TFLOP/s**

而本卡（RTX 4060 Laptop，24 SM，max SM 3105 MHz，130 W 上限）的 fp16 稠密峰值
在 120 TFLOP/s 量级。**30 只有峰值的约 1/4。** 一个只跑到峰值 1/4 的 kernel，
按定义就不是计算受限——它要么受发射/占用限制，要么受功耗限制。

【两种解读对应两种结论，必须分开】
  (a) 本卡在真实功耗下的**持续**峰值就在 30–40 TFLOP/s → S12 已到顶 →
      C66 掉到一半是真损失，"计算受限时分区只有 53%" 成立；
  (b) 换大矩阵能跑到 100+ → 2048² 只是没喂饱 → 该臂不是计算受限臂，
      §10.2 的结论应改述为"**共享并发填满了独占时留下的空转**"。

【判据】把 2048² 的实测 TFLOP/s 除以**同一进程内测到的本卡实测峰值**。
  ≥ 0.8 → 是计算受限（解读 a）
  ≤ 0.4 → 不是（解读 b）

【做法】
  1. 峰值扫描：单线程跑 mm(N², fp16)，N ∈ {2048, 3072, 4096, 6144, 8192}，
     取各档 TFLOP/s 的最大值作为"本卡实测峰值"（8192² 一定喂得饱）。
  2. 复刻 §10.2 的三臂（同一 `Worker` + `apply_quota(1.0)`、默认流、ITERS=12）：
     `S12` / `C66`（两线程各 0.5 配额）/ `C12`（两线程无掩码）。
  3. 全程用 NVML 采 SM 时钟 / 功耗 / GPU 占用，判断是"到顶"还是"没喂饱"。

⚠️ §10.2 的脚本里 `per_job` 对 mm 写成 `2*4096³*ITERS` 而矩阵是 2048²（差 8 倍），
本脚本一律按 `2*N³` 实算，并同时给出"换成 §10.2 口径"的数值供对照。

用法::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    .venv/bin/python scripts/probes/diag_compute_bound_label.py
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter  # noqa: E402

OUT_DEFAULT = "results/reproducibility/probes/diag_compute_bound_label.json"


class Worker:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
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
            time.sleep(0.0005)
        if "e" in box:
            raise box["e"]
        return box["r"]


class NvmlSampler:
    """后台按固定间隔采 SM 时钟 / 功耗 / 占用。约 1.3 ms 一次（§11.1：不能用 nvidia-smi）。"""

    def __init__(self, interval_s: float = 0.01):
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self._h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self._pynvml = None

    def start(self):
        if self._pynvml is None:
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        p = self._pynvml
        while not self._stop.is_set():
            try:
                u = p.nvmlDeviceGetUtilizationRates(self._h)
                self.samples.append({
                    "sm_clk": p.nvmlDeviceGetClockInfo(self._h, p.NVML_CLOCK_SM),
                    "power_w": round(p.nvmlDeviceGetPowerUsage(self._h) / 1000, 1),
                    "util_gpu": u.gpu,
                    "util_mem": u.memory,
                })
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        s = self.samples
        self.samples = []
        if not s:
            return {"samples": 0}
        med = lambda k: statistics.median(x[k] for x in s)  # noqa: E731
        return {
            "samples": len(s),
            "sm_clk_med": med("sm_clk"), "sm_clk_max": max(x["sm_clk"] for x in s),
            "power_med": med("power_w"), "power_max": max(x["power_w"] for x in s),
            "util_gpu_med": med("util_gpu"), "util_mem_med": med("util_mem"),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--iters", type=int, default=12, help="与 §10.2 一致")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sweep", default="2048,3072,4096,6144,8192")
    ap.add_argument("--sweep-iters", type=int, default=4)
    args = ap.parse_args()

    adapter = LibSmCtrlAdapter()
    probe = adapter.probe()
    report: dict = {
        "script": "scripts/probes/diag_compute_bound_label.py",
        "note": "判定 §10.2 的『计算受限』臂是否真的是计算受限；全部按 2*N^3 实算 FLOP",
        "libsmctrl": probe,
        "iters": args.iters,
        "reps": args.reps,
        "env_before": _env(),
    }
    print(f"libsmctrl: available={probe.get('available')} total_tpcs={probe.get('total_tpcs')}")

    sampler = NvmlSampler()

    # ---------- 1. 实测峰值扫描（单线程，全卡） ----------
    w = Worker()
    w.call(lambda: adapter.apply_quota(11, 1.0))
    mats: dict[int, tuple] = {}
    sweep = []
    print("\n---- 实测峰值扫描（单线程全卡，fp16 GEMM）----")
    for n in [int(x) for x in args.sweep.split(",")]:
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        mats[n] = (a, b)

        def body(n=n):
            for _ in range(args.sweep_iters):
                torch.mm(mats[n][0], mats[n][1])

        w.call(lambda: (body(), torch.cuda.synchronize()))          # 暖机
        sampler.start()
        times = []
        for _ in range(args.reps):
            times.append(w.call(lambda: _timed(body)))
        env = sampler.stop()
        ms = statistics.median(times)
        tflops = args.sweep_iters * 2 * n ** 3 / (ms / 1000) / 1e12
        sweep.append({"n": n, "ms": ms, "tflops": tflops, "env": env})
        print(f"  N={n:>5}  {ms:8.2f} ms  → {tflops:6.1f} TFLOP/s   "
              f"clk {env.get('sm_clk_med')} MHz  {env.get('power_med')} W  "
              f"util {env.get('util_gpu_med')}%")

    peak = max(x["tflops"] for x in sweep)
    peak_n = max(sweep, key=lambda x: x["tflops"])["n"]
    report["peak"] = {"tflops": peak, "n": peak_n, "sweep": sweep}
    print(f"  >>> 本卡实测峰值 = {peak:.1f} TFLOP/s（N={peak_n}）")

    # ---------- 2. 复刻 §10.2 的三臂 ----------
    n = 2048
    a = mats.get(n) or (torch.randn(n, n, device="cuda", dtype=torch.float16),
                        torch.randn(n, n, device="cuda", dtype=torch.float16))

    def body_small():
        for _ in range(args.iters):
            torch.mm(a[0], a[1])

    print(f"\n---- 复刻 §10.2 三臂（mm {n}² fp16 × {args.iters}）----")
    arms: dict = {}

    # S12：单 worker 全卡，A 跑完再跑 B（各自计时）
    def solo():
        return w.call(lambda: _timed(body_small))

    sampler.start()
    s12 = []
    for _ in range(args.reps):
        body_small(), torch.cuda.synchronize()
        s12.append(solo())
        s12.append(solo())
    env_s12 = sampler.stop()
    arms["S12"] = {"times": s12, "ms": statistics.median(s12), "env": env_s12}

    # C66 / C12：两线程，barrier 对齐，各报自己的时间。
    # 与 §10.2 一致：配额按**调用线程**记账（`_allocate(thread_id, want)`），
    # 所以 (a) 必须先把 S12 那个线程的整卡分配**释放**，否则 0.5+0.5 会报
    # "TPC 池不足"；(b) 两个 0.5 必须打在两个各自独立的 worker 上。
    w.call(lambda: adapter.release())
    wa2, wb = Worker(), Worker()
    ra = wa2.call(lambda: adapter.apply_quota(13, 0.5))
    rb = wb.call(lambda: adapter.apply_quota(17, 0.5))
    if not (ra.get("enforced") and rb.get("enforced")):
        raise SystemExit(f"配额未生效，拒绝继续：ra={ra} rb={rb}")

    def concurrent(wa, wbk, mask_on):
        if not mask_on:
            wa.call(lambda: adapter.release())
            wbk.call(lambda: adapter.release())

        def pair():
            box, gate = {}, threading.Barrier(3)

            def go(wk, tag):
                gate.wait()
                box[tag] = wk.call(lambda: _timed(body_small))
            t1 = threading.Thread(target=go, args=(wa, "A"))
            t2 = threading.Thread(target=go, args=(wbk, "B"))
            t1.start(); t2.start(); gate.wait(); t1.join(); t2.join()
            return box["A"], box["B"]

        sampler.start()
        pairs = []
        for _ in range(args.reps):
            body_small(), torch.cuda.synchronize()
            pairs.append(pair())
        env = sampler.stop()
        return pairs, env

    pairs66, env66 = concurrent(wa2, wb, True)
    c66 = [x for p in pairs66 for x in p]
    arms["C66"] = {"pairs": pairs66, "ms": statistics.median(c66), "env": env66,
                   "disjoint": _disjoint(ra, rb)}

    pairs12, env12 = concurrent(wa2, wb, False)
    c12 = [x for p in pairs12 for x in p]
    arms["C12"] = {"pairs": pairs12, "ms": statistics.median(c12), "env": env12}

    per_job = 2 * n ** 3 * args.iters
    for k, v in arms.items():
        ms = v["ms"]
        v["tflops_correct"] = per_job / (ms / 1000) / 1e12
        v["tflops_agentsmd_units"] = per_job / (ms / 1000) / 1e12 * 8   # 换 §10.2 口径
        v["pct_of_peak"] = v["tflops_correct"] / peak
        print(f"  {k:>4}  单侧 {ms:8.2f} ms  → {v['tflops_correct']:6.1f} TFLOP/s "
              f"= 峰值的 {v['pct_of_peak'] * 100:5.1f}%   "
              f"clk {v['env'].get('sm_clk_med')} MHz  {v['env'].get('power_med')} W")

    # §10.2 的聚合口径：
    #   agg_S12 = 2*per_job/(tA+tB) = per_job/t_s12        （A、B 同尺寸）
    #   agg_Cxx = per_job/cA + per_job/cB = 2*per_job/c     （c 为单侧中位）
    # 故 agg_Cxx/agg_S12 = 2*t_s12/c。**因子 2 不能丢**——
    # 丢了会得到 0.31/0.55，而 §10.2 报的是 0.528/0.997。
    s12_ms = arms["S12"]["ms"]
    for k in ("C66", "C12"):
        arms[k]["agg_over_s12"] = 2 * s12_ms / arms[k]["ms"]

    pct = arms["S12"]["pct_of_peak"]
    report["arms"] = arms
    report["verdict"] = {
        "s12_pct_of_peak": pct,
        "label": ("计算受限（已到峰值 80% 以上）" if pct >= 0.8
                  else "**不是**计算受限（低于峰值 40%）" if pct <= 0.4
                  else "介于两者之间，需人工判断"),
        "ratio_c66_over_s12": arms["C66"]["agg_over_s12"],
        "ratio_c12_over_s12": arms["C12"]["agg_over_s12"],
    }
    report["env_after"] = _env()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n判定：{report['verdict']['label']}（S12 = 峰值的 "
          f"{arms['S12']['pct_of_peak'] * 100:.1f}%）")
    print(f"[写出] {out}")
    return 0


def _timed(fn):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def _disjoint(ra, rb) -> bool:
    return set(range(ra["tpc_start"], ra["tpc_start"] + ra["tpc_count"])).isdisjoint(
        range(rb["tpc_start"], rb["tpc_start"] + rb["tpc_count"]))


def _env() -> dict:
    snap = {"at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        snap["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        snap["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
        pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001
        snap["error"] = str(e)
    return snap


if __name__ == "__main__":
    raise SystemExit(main())
