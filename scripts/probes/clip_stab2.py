"""同会话交替：S12 串行 / S6 半卡串行 / C66 分区并发 / C12 共享并发。

【出处】本脚本 2026-09-29 从 `/tmp/clip_stab2.py` 抢救入库（`/tmp` 会被清理）。
原始版本**只 print、不落盘**，因此 `任务书.txt:261-264` 里那组
"1024：分区 1.032× / 共享 1.154×；2048：分区 0.963× / 共享 1.075×"
在仓库与 /tmp 里都没有可核对的产物。

【本次唯一的改动】加 `--out`，把逐次原始值、配对比值与环境读数写成 JSON。
**测量逻辑一个字未改**（打印格式也保持原样），所以重跑结果可与原数字直接对照。

与 `clip_arms2.py` 的区别（重要）：
  - 本脚本的比值是**逐次配对**的（第 i 轮的 S12 配第 i 轮的 C66），不是臂级中位数相除；
  - 本脚本每尺寸交替 8 轮，`clip_arms2` 只采 2 个并发样本、并用 `2/max(t_a,t_b)` 聚合。

用法::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    .venv/bin/python scripts/probes/clip_stab2.py --out results/reproducibility/probes/clip_stab2.json
"""

import argparse
import json
import queue
import statistics
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import create_resource_backend  # noqa: E402

OUT_DEFAULT = "results/reproducibility/probes/clip_stab2.json"


class Worker:
    def __init__(self, wid, enc):
        self.wid, self.enc = wid, enc
        self.q = queue.Queue(); threading.Thread(target=self._run, daemon=True).start()

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


def env_snapshot() -> dict:
    """记录时钟、功耗和利用率快照，不参与自动判定。"""
    snap = {"at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        snap["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        snap["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
        snap["util_gpu_pct"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001
        snap["error"] = str(e)
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--sizes", default="672,1024,2048")
    ap.add_argument("--config", default="config.libsmctrl.example.yaml")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    cfg = load_config(args.config)
    res = create_resource_backend(cfg.executor.resource_backend, cfg.executor.libsmctrl_adapter,
                                  cfg.executor.allow_proxy_fallback)
    enc = ClipEncoderBackend(cfg.model, res, 2)
    wa, wb = Worker(0, enc), Worker(1, enc)

    def mk(s, t):
        return EncodeJob(f"{s}-{t}", zlib.crc32(f"{s}-{t}".encode()) % (2 ** 31), s, s, 600_000)

    def once(w, s, q, t):
        j = mk(s, t)
        j.sm_fraction = q
        return w.call(lambda: w.enc.encode(j, w.wid).execution_ms)

    def conc(wa, wb, s, qa, qb, t):
        box = {}
        gate = threading.Barrier(3)

        def go(w, q, n):
            j = mk(s, f"{t}-{n}")
            j.sm_fraction = q
            gate.wait()
            box[n] = w.call(lambda: w.enc.encode(j, w.wid).execution_ms)
        t1 = threading.Thread(target=go, args=(wa, qa, "A"))
        t2 = threading.Thread(target=go, args=(wb, qb, "B"))
        t1.start(); t2.start(); gate.wait(); t1.join(); t2.join()
        return max(box["A"], box["B"])

    report = {
        "script": "scripts/probes/clip_stab2.py",
        "note": "从 /tmp 抢救后仅加了落盘；测量逻辑与原版一致。比值是逐次配对的。",
        "config": args.config,
        "sizes": sizes,
        "reps": args.reps,
        "env_before": env_snapshot(),
        "per_size": {},
    }

    for size in sizes:
        once(wa, size, 1.0, "warm")
        rs, r6, r66, r12 = [], [], [], []
        for i in range(args.reps):
            rs.append(once(wa, size, 1.0, f"s{i}"))
            r6.append(once(wa, size, 0.5, f"h{i}"))
            r66.append(conc(wa, wb, size, 0.5, 0.5, f"p{i}"))
            r12.append(conc(wa, wb, size, 1.0, 1.0, f"c{i}"))
        m = statistics.median
        print(f"\n=== {size}（patch {(size // 32) ** 2}）===")
        print(f"  S12 单请求 {m(rs):7.2f} ms   S6 单请求 {m(r6):7.2f} ms   "
              f"S6/S12 = {m(r6) / m(rs):.3f} (即 T6/T12)")
        print(f"  C66 墙钟 {m(r66):7.2f}   C12 墙钟 {m(r12):7.2f}")
        q66 = [2 * s / c for s, c in zip(rs, r66)]
        q12 = [2 * s / c for s, c in zip(rs, r12)]
        print(f"  C66/S12 逐次 {[round(x, 3) for x in q66]}  中位 {m(q66):.3f}")
        print(f"  C12/S12 逐次 {[round(x, 3) for x in q12]}  中位 {m(q12):.3f}")

        report["per_size"][str(size)] = {
            "patches": (size // 32) ** 2,
            "s12": rs, "s6": r6, "c66": r66, "c12": r12,
            "median": {"s12": m(rs), "s6": m(r6), "c66": m(r66), "c12": m(r12),
                       "t6_over_t12": m(r6) / m(rs),
                       "c66_over_s12": m(q66), "c12_over_s12": m(q12)},
            "paired": {"c66_over_s12": q66, "c12_over_s12": q12},
            "env_after": env_snapshot(),
        }

    report["env_after"] = env_snapshot()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
