"""试点 5：修正并发臂计时口径，给出可靠的三臂对照。

试点 4 的缺陷：并发臂的总墙钟从「启动线程之前」开始计，把线程创建、暖机与 barrier
同步都算进了聚合吞吐，导致并发被系统性低估。本版改为**只由各自线程报告自己的执行时间**，
聚合吞吐按两侧执行时间求和计算。

三臂（每个请求各自完整跑 12 次大 kernel）：
  S12  全卡串行      A 独占 12 TPC → B 独占 12 TPC
  S6   六卡串行对照  A 独占 6 TPC  → B 独占 6 TPC（用于分离"配额档位"与"并发与否"）
  C66  二等分并发    A 拿 0+6、B 拿 6+6，同时跑

判据  并发/串行 = 聚合吞吐之比。<1 表示"一个接一个用满全卡"优于"各分一半同时跑"。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_arms2.py
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
REP = 5


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

    out = {}
    for kind, unit, per_job in (("mem", "GB", 2 * 256 * MIB * ITERS / 1e9),
                                ("mm", "TFLOP", 2 * 4096**3 * ITERS / 1e12)):
        name = "访存受限 (copy 256MiB ×12)" if kind == "mem" else "计算受限 (4096³ fp16 matmul ×12)"
        print(f"\n{'#'*8} {name} {'#'*8}")
        rec = {"unit": unit, "per_job": per_job}

        # ---------- S12: 全卡串行 ----------
        w = Worker()
        r12 = w.call(lambda: adapter.apply_quota(11, 1.0))
        assert r12.get("enforced"), r12

        def solo(wk, tag, kind=kind):
            body = make_body(tag, kind)
            def go():
                body(); torch.cuda.synchronize()
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            return wk.call(go)

        s12 = {t: statistics.median([solo(w, t) for _ in range(REP)]) for t in ("A", "B")}
        w.call(lambda: adapter.release())
        print(f"S12 全卡串行 (TPC {r12['tpc_start']}+{r12['tpc_count']}): "
              f"A={s12['A']:.1f} B={s12['B']:.1f} ms，合计 {s12['A']+s12['B']:.1f} ms")
        rec["S12"] = {"A": s12["A"], "B": s12["B"], "total_ms": s12["A"] + s12["B"]}
        rec["S12"]["agg"] = 2 * per_job / (rec["S12"]["total_ms"] / 1000)

        # ---------- 两线程建立互斥分配 ----------
        wa, wb = Worker(), Worker()
        ra = wa.call(lambda: adapter.apply_quota(13, 0.5))
        rb = wb.call(lambda: adapter.apply_quota(17, 0.5))
        assert ra.get("enforced") and rb.get("enforced")
        sa = set(range(ra["tpc_start"], ra["tpc_start"] + ra["tpc_count"]))
        sb = set(range(rb["tpc_start"], rb["tpc_start"] + rb["tpc_count"]))
        print(f"C66 二等分并发: A=TPC {ra['tpc_start']}+{ra['tpc_count']}  "
              f"B=TPC {rb['tpc_start']}+{rb['tpc_count']}  互斥={sa.isdisjoint(sb)}")

        # ---------- S6: 各自独占 6 TPC 串行 ----------
        s6 = {t: statistics.median([solo(wk, t) for _ in range(REP)]) for wk, t in ((wa, "A"), (wb, "B"))}
        print(f"S6  六卡串行: A={s6['A']:.1f} B={s6['B']:.1f} ms，合计 {s6['A']+s6['B']:.1f} ms")
        rec["S6"] = {"A": s6["A"], "B": s6["B"], "total_ms": s6["A"] + s6["B"]}
        rec["S6"]["agg"] = 2 * per_job / (rec["S6"]["total_ms"] / 1000)

        # ---------- C66: 并发（记录各线程自己的执行时间） ----------
        conc = {"A": [], "B": []}
        for _ in range(REP):
            barrier = threading.Barrier(2)
            got = {}

            def go(tag, wk):
                body = make_body(tag, kind)
                def inner():
                    body(); torch.cuda.synchronize()
                    barrier.wait()
                    t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                    return (time.perf_counter() - t0) * 1000
                got[tag] = wk.call(inner)
            t1 = threading.Thread(target=go, args=("A", wa)); t2 = threading.Thread(target=go, args=("B", wb))
            t1.start(); t2.start(); t1.join(); t2.join()
            conc["A"].append(got["A"]); conc["B"].append(got["B"])
        ca, cb = statistics.median(conc["A"]), statistics.median(conc["B"])
        # 关键：聚合吞吐按"两个任务各自完成所做的工作 / 各自的耗时"求和，
        # 不能用墙钟——墙钟不含并发度信息，且会被线程启动污染。
        agg_c = per_job / (ca / 1000) + per_job / (cb / 1000)
        print(f"C66 并发执行: A={ca:.1f} B={cb:.1f} ms  "
              f"(相对各自独跑的 slowdown: A={ca/s6['A']:.2f}× B={cb/s6['B']:.2f}×)")
        rec["C66"] = {"A": ca, "B": cb, "agg": agg_c,
                      "slowdown_A": ca / s6["A"], "slowdown_B": cb / s6["B"]}
        wa.call(lambda: adapter.release()); wb.call(lambda: adapter.release())

        # ---------- 结论 ----------
        print(f"\n  聚合吞吐   S12 全卡串行 : {rec['S12']['agg']:6.1f} {unit}/s")
        print(f"             S6  半卡串行 : {rec['S6']['agg']:6.1f} {unit}/s   "
              f"(相对 S12: {rec['S6']['agg']/rec['S12']['agg']:.3f}×)")
        print(f"             C66 二等分并发: {rec['C66']['agg']:6.1f} {unit}/s   "
              f"(相对 S12: {rec['C66']['agg']/rec['S12']['agg']:.3f}×)")
        rec["ratio_C66_over_S12"] = rec["C66"]["agg"] / rec["S12"]["agg"]
        rec["ratio_S6_over_S12"] = rec["S6"]["agg"] / rec["S12"]["agg"]
        verd = "串行全卡更好" if rec["ratio_C66_over_S12"] < 1 else "分区并发更好"
        print(f"  >>> 分区并发 / 全卡串行 = {rec['ratio_C66_over_S12']:.3f}×  → {verd}")
        out[kind] = rec

    with open("/tmp/pilot_arms2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n已写入 /tmp/pilot_arms2.json")


if __name__ == "__main__":
    main()
