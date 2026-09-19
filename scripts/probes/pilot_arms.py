"""试点 4：补齐关键对照臂——「全卡串行」vs「二等分并发」。

这是空间分区论文真正该做的比较：同样两个请求，
  臂 S12  全卡串行：A 独占 12 TPC 跑完，再 B 独占 12 TPC
  臂 C66  二等分并发：A 拿 0+6、B 拿 6+6，同时跑
若 C66 的聚合吞吐 <= S12，则"空间分区"相对"直接串行用满全卡"没有任何收益。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_arms.py
"""

import json
import queue
import statistics
import sys
import threading
import time

sys.path.insert(0, "/home/lyon/projects/bishe")
import torch  # noqa: E402

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter  # noqa: E402

MIB = 1024 * 1024
ITERS = 12
REP = 3


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
    dev = torch.device("cuda")
    n = 64 * MIB
    bufs = {k: (torch.randn(n, device=dev), torch.empty(n, device=dev)) for k in ("A", "B")}
    ma = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    mb = torch.randn(4096, 4096, device=dev, dtype=torch.float16)

    def make_body(tag, kind):
        x, y = bufs[tag]
        if kind == "mem":
            def body():
                for _ in range(ITERS):
                    y.copy_(x)
        else:
            def body():
                for _ in range(ITERS):
                    torch.mm(ma, mb)
        return body

    results = {}
    for kind, unit, per_job in (("mem", "GB", 2 * 256 * MIB * ITERS / 1e9),
                                ("mm", "TFLOP", 2 * 4096**3 * ITERS / 1e12)):
        print(f"\n########## {'访存受限' if kind=='mem' else '计算受限'} ##########")
        arm = {}

        # --- 臂 S12：全卡串行（每个任务独占 12 TPC）---
        w = Worker()

        def set12():
            r = adapter.apply_quota(11, 1.0)
            assert r.get("enforced"), r
            return r
        info12 = w.call(set12)
        print(f"S12 掩码: TPC {info12['tpc_start']}+{info12['tpc_count']}")

        def run_solo(tag):
            body = make_body(tag, kind)
            def go():
                body(); torch.cuda.synchronize()
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            return w.call(go)
        s12 = {t: statistics.median([run_solo(t) for _ in range(REP)]) for t in ("A", "B")}
        w.call(lambda: adapter.release())
        print(f"  A={s12['A']:.1f} ms  B={s12['B']:.1f} ms   串行总墙钟={(s12['A']+s12['B']):.1f} ms")
        arm["S12"] = s12

        # --- 臂 C66：二等分并发 ---
        wa, wb = Worker(), Worker()

        def setq(w, frac):
            r = w.call(lambda: adapter.apply_quota(13, frac))
            assert r.get("enforced"), r
            return r
        ra = setq(wa, 0.5); rb = setq(wb, 0.5)
        disj = set(range(ra["tpc_start"], ra["tpc_start"] + ra["tpc_count"])).isdisjoint(
            range(rb["tpc_start"], rb["tpc_start"] + rb["tpc_count"]))
        print(f"C66 掩码: A=TPC {ra['tpc_start']}+{ra['tpc_count']}  B=TPC {rb['tpc_start']}+{rb['tpc_count']}  互斥={disj}")

        def run_solo_q(w, tag):
            body = make_body(tag, kind)
            def go():
                body(); torch.cuda.synchronize()
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            return w.call(go)
        q6 = {t: statistics.median([run_solo_q(w, t) for _ in range(REP)]) for w, t in ((wa, "A"), (wb, "B"))}
        print(f"  各自单独(6TPC): A={q6['A']:.1f} ms  B={q6['B']:.1f} ms")

        walls = []
        for _ in range(REP):
            barrier = threading.Barrier(2)

            def go(tag, w):
                body = make_body(tag, kind)
                def inner():
                    body(); torch.cuda.synchronize()
                    barrier.wait()
                    t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                    return (time.perf_counter() - t0) * 1000
                w.call(inner)
            t0 = time.perf_counter()
            ta = threading.Thread(target=go, args=("A", wa)); tb = threading.Thread(target=go, args=("B", wb))
            ta.start(); tb.start(); ta.join(); tb.join()
            walls.append((time.perf_counter() - t0) * 1000)
        W = statistics.median(walls)
        wa.call(lambda: adapter.release()); wb.call(lambda: adapter.release())
        print(f"  并发总墙钟={W:.1f} ms")
        arm["C66"] = {"solo_A": q6["A"], "solo_B": q6["B"], "conc_wall": W}

        # --- 结论 ---
        seq12 = s12["A"] + s12["B"]
        agg12 = 2 * per_job / (seq12 / 1000)
        agg66 = 2 * per_job / (W / 1000)
        print(f"\n  >>> 全卡串行聚合 = {agg12:.1f} {unit}/s")
        print(f"  >>> 二等分并发聚合 = {agg66:.1f} {unit}/s")
        print(f"  >>> 分区并发 / 全卡串行 = {agg66/agg12:.3f}×   "
              f"({'并发更好' if agg66 > agg12 else '串行更好'})")
        results[kind] = {"unit": unit, "s12": s12, "seq12_ms": seq12, "agg_seq12": agg12,
                         "c66": arm["C66"], "agg_c66": agg66, "ratio_c66_over_s12": agg66 / agg12,
                         "concurrency_gain": agg66 / agg12}

    with open("/tmp/pilot_arms.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n已写入 /tmp/pilot_arms.json")


if __name__ == "__main__":
    main()
