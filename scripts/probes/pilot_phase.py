"""试点 6：把「分区收益」扫成一条曲线——算术强度 → 分区/串行 收益比。

试点 5 给出两个端点：纯访存负载分区并发赢 1.28×，纯计算负载输到 0.53×。
本试点在两端之间连续调节「每字节内存流量对应的计算量」，看收益比是否单调过渡，
从而把判据从两个点变成一条可预测的曲线。

做法：请求体 = copy×c + matmul×m，扫 m/(c+m)。算术强度随之从"访存受限"滑向"计算受限"。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_phase.py
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
REP = 3
# (copy 次数, matmul 次数) —— 从纯访存扫到纯计算
MIXES = [(8, 0), (6, 1), (4, 2), (2, 4), (1, 6), (0, 8)]


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
    ma = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    mb = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    MM_FLOP = 2 * 2048**3
    COPY_BYTES = 2 * 256 * MIB

    rows = []
    for c, m in MIXES:
        def make_body(tag):
            x, y = bufs[tag]
            def body():
                for _ in range(c):
                    y.copy_(x)
                for _ in range(m):
                    torch.mm(ma, mb)
            return body
        # 算术强度 = FLOP / 字节
        ai = (m * MM_FLOP) / max(c * COPY_BYTES, 1) if c else float("inf")

        w = Worker()
        r12 = w.call(lambda: adapter.apply_quota(11, 1.0))
        assert r12.get("enforced")

        def solo(wk, tag):
            body = make_body(tag)
            def go():
                body(); torch.cuda.synchronize()
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            return wk.call(go)
        s12 = statistics.median([solo(w, "A") for _ in range(REP)])
        w.call(lambda: adapter.release())

        wa, wb = Worker(), Worker()
        ra = wa.call(lambda: adapter.apply_quota(13, 0.5))
        rb = wb.call(lambda: adapter.apply_quota(17, 0.5))
        assert ra.get("enforced") and rb.get("enforced")
        s6 = statistics.median([solo(wa, "A") for _ in range(REP)])

        cc = []
        for _ in range(REP):
            barrier = threading.Barrier(2)
            got = {}
            def go(tag, wk):
                body = make_body(tag)
                def inner():
                    body(); torch.cuda.synchronize()
                    barrier.wait()
                    t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                    return (time.perf_counter() - t0) * 1000
                got[tag] = wk.call(inner)
            t1 = threading.Thread(target=go, args=("A", wa)); t2 = threading.Thread(target=go, args=("B", wb))
            t1.start(); t2.start(); t1.join(); t2.join()
            cc.append(got["A"])
        conc = statistics.median(cc)
        wa.call(lambda: adapter.release()); wb.call(lambda: adapter.release())

        # 两臂做同样的总工作量：串行臂 = 2 个任务前后各跑一次 s12
        serial_total = 2 * s12
        conc_total = conc          # 两任务并发，各自耗时约等于 conc
        ratio = serial_total / conc_total
        row = {"copy": c, "mm": m, "ai_flop_per_byte": (round(ai, 1) if ai != float("inf") else "inf"),
               "solo_12tpc_ms": round(s12, 2), "solo_6tpc_ms": round(s6, 2),
               "conc_6tpc_ms": round(conc, 2),
               "agg_serial_rps": round(2 / (serial_total / 1000), 2),
               "agg_conc_rps": round(2 / (conc_total / 1000), 2),
               "conc_over_serial": round(ratio, 3)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("\n=== 结论：并发的两任务总墙钟 / 串行的两任务总墙钟 ===")
    print(f"{'copy':>5}{'mm':>4}{'算术强度':>12}{'并发/串行':>12}   判定")
    for r in rows:
        v = r["conc_over_serial"]
        print(f"{r['copy']:>5}{r['mm']:>4}{str(r['ai_flop_per_byte']):>12}{v:>12.3f}   "
              f"{'并发胜' if v > 1 else '串行胜'}")
    with open("/tmp/pilot_phase.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\n已写入 /tmp/pilot_phase.json")


if __name__ == "__main__":
    main()
