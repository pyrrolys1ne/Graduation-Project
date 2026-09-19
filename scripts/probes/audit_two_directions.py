"""审核实验：针对两个方向各自最脆弱的一环，做一次判定性测量。

【针对方向一】嫌疑：我的三臂对照漏了第三个选项——**不分区的并发**（两条流共享全部 12 TPC）。
  服务级数据（AGENTS.md 第 7 节）显示 spatial_proxy（共享 SM）在所有指标上都优于
  spatial_tpc（互斥分区）。若共享并发的确支配分区并发，则"该不该分区"这个问题的答案
  可能是"永远不该"，判据的重要性随之下降。
  → 补第四臂 C12：两条流都不加掩码，同时跑。

【针对方向二】嫌疑：我全程只用「从 TPC 0 开始的连续掩码」，因此无法区分"6 个 TPC"与
  "TPC 0–5"。若同一 k 的不同子集给出不同结果，则方向二成立，且方向一的预测器 T6/T12
  本身不是良定义的。
  → 固定 k=6，测 6 种不同子集（前缀/后缀/跨步/奇偶/随机×2）的执行时间差异。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/audit_two_directions.py
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

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter, load_libsmctrl  # noqa: E402

MIB = 1024 * 1024
ITERS = 12
REP = 5
TOTAL_TPC = 12


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
    """TPC 集合 -> libsmctrl 禁用掩码（置位=禁用）。空集合返回 0（=不加限制）。"""
    if not tpcs:
        return 0
    enabled = 0
    for t in tpcs:
        enabled |= 1 << t
    return (~enabled) & ((1 << 64) - 1)


def main():
    adapter = LibSmCtrlAdapter()
    assert adapter.probe().get("available")
    lib = load_libsmctrl()
    dev = torch.device("cuda")

    n = 64 * MIB
    bufs = {k: (torch.randn(n, device=dev), torch.empty(n, device=dev)) for k in ("A", "B")}
    ma = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    mb = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    MM_FLOP = 2 * 2048**3
    COPY_BYTES = 2 * 256 * MIB

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

    report = {}

    # ==================== 第一部分：四臂对照，补上 C12 ====================
    print("=" * 72)
    print("第一部分：全卡串行 / 半卡串行 / 二等分并发 / 共享全卡并发")
    print("=" * 72)
    arms_out = {}
    for kind, unit, per_job in (("mem", "GB", 2 * 256 * MIB * ITERS / 1e9),
                                ("mm", "TFLOP", 2 * 4096**3 * ITERS / 1e12)):
        print(f"\n---- {'访存受限' if kind == 'mem' else '计算受限'} ----")
        # 计算受限用大矩阵，访存受限用小矩阵不参与
        mm_globals = None
        rec = {"unit": unit}

        def solo(wk, tag, kind=kind):
            body = make_body(tag, kind)
            def go():
                body(); torch.cuda.synchronize()
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            return wk.call(go)

        def concurrent_run(wa, wb, kind):
            got = {}
            barrier = threading.Barrier(2)

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
            return got["A"], got["B"]

        # --- S12 全卡串行 ---
        w = Worker()
        assert w.call(lambda: adapter.apply_quota(11, 1.0)).get("enforced")
        s12 = {t: statistics.median([solo(w, t) for _ in range(REP)]) for t in ("A", "B")}
        w.call(lambda: adapter.release())
        rec["S12"] = {"A": s12["A"], "B": s12["B"], "agg": 2 * per_job / ((s12["A"] + s12["B"]) / 1000)}

        # --- C66 二等分并发（互斥）---
        wa, wb = Worker(), Worker()
        ra = wa.call(lambda: adapter.apply_quota(13, 0.5))
        rb = wb.call(lambda: adapter.apply_quota(17, 0.5))
        assert ra.get("enforced") and rb.get("enforced")
        disj = set(range(ra["tpc_start"], ra["tpc_start"] + ra["tpc_count"])).isdisjoint(
            range(rb["tpc_start"], rb["tpc_start"] + rb["tpc_count"]))
        c66 = [concurrent_run(wa, wb, kind) for _ in range(REP)]
        ca = statistics.median(x[0] for x in c66); cb = statistics.median(x[1] for x in c66)
        rec["C66"] = {"A": ca, "B": cb, "agg": per_job / (ca / 1000) + per_job / (cb / 1000),
                      "disjoint": disj}

        # --- C12 共享全卡并发（不加掩码）---
        wa.call(lambda: adapter.release()); wb.call(lambda: adapter.release())
        c12 = [concurrent_run(wa, wb, kind) for _ in range(REP)]
        da = statistics.median(x[0] for x in c12); db = statistics.median(x[1] for x in c12)
        rec["C12"] = {"A": da, "B": db, "agg": per_job / (da / 1000) + per_job / (db / 1000)}
        wa.call(lambda: adapter.release()); wb.call(lambda: adapter.release())

        for k in ("S12", "C66", "C12"):
            print(f"  {k:>4} 聚合 = {rec[k]['agg']:7.1f} {unit}/s"
                  f"   (A={rec[k]['A']:.1f} B={rec[k]['B']:.1f} ms)"
                  f"   相对 S12 = {rec[k]['agg']/rec['S12']['agg']:.3f}×")
        print(f"  C66 TPC 互斥 = {rec['C66']['disjoint']}")
        best = max(("S12", "C66", "C12"), key=lambda k: rec[k]["agg"])
        print(f"  >>> 最优臂：{best}")
        rec["best_arm"] = best
        arms_out[kind] = rec
    report["arms"] = arms_out

    # ==================== 第二部分：同一 k、不同 TPC 子集 ====================
    print("\n" + "=" * 72)
    print("第二部分：固定 k=6，不同 TPC 子集的身份效应")
    print("=" * 72)

    rng = random.Random(20260915)
    subsets = {
        "0-5 (前缀)": [0, 1, 2, 3, 4, 5],
        "6-11 (后缀)": [6, 7, 8, 9, 10, 11],
        "交错 A": [0, 1, 2, 6, 7, 8],
        "交错 B": [3, 4, 5, 9, 10, 11],
        "奇偶": [1, 3, 5, 7, 9, 11],
        "随机": sorted(rng.sample(range(TOTAL_TPC), 6)),
        "12 全开(参照)": list(range(TOTAL_TPC)),
    }
    for name, tpcs in subsets.items():
        print(f"  {name:>14}: TPC {tpcs}")

    w = Worker()

    def measure(tpcs, kind, reps=10):
        def go():
            mask = mask_for(tpcs) if len(tpcs) < TOTAL_TPC else 0
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask))
            body = make_body("A", kind)
            for _ in range(3):
                body()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); body(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            return statistics.median(ts)
        return w.call(go)

    # 交错顺序测量，避免顺序漂移被误读成子集效应
    plan = [(name, kind) for kind in ("mem", "mm") for name in subsets] * 3
    rng.shuffle(plan)
    samples = {}
    for name, kind in plan:
        samples.setdefault((name, kind), []).append(measure(subsets[name], kind))
    w.call(lambda: adapter.release())

    print(f"\n  {'子集':>14} {'拷贝(ms)':>10} {'矩阵乘(ms)':>11} {'k=6 内极差':>11}")
    k6 = [n for n in subsets if n != "12 全开(参照)"]
    iden = {}
    for kind in ("mem", "mm"):
        vals = {n: statistics.median(samples[(n, kind)]) for n in k6}
        iden[kind] = vals
    for name in subsets:
        cm = statistics.median(samples[(name, "mem")])
        mm_ = statistics.median(samples[(name, "mm")])
        tag = ""
        if name in k6:
            spread_mem = max(iden["mem"].values()) / min(iden["mem"].values())
            tag = f"{spread_mem:.3f}×" if True else ""
        print(f"  {name:>14} {cm:>10.2f} {mm_:>11.2f} {tag:>11}")

    for kind in ("mem", "mm"):
        v = list(iden[kind].values())
        print(f"\n  k=6 六种子集的 {kind} 极差 = {max(v)/min(v):.3f}×  "
              f"(max {max(v):.2f} / min {min(v):.2f} ms)")
    report["identity"] = iden
    report["subsets"] = {k: v for k, v in subsets.items()}

    with open("/tmp/audit_two_directions.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n已写入 /tmp/audit_two_directions.json")


if __name__ == "__main__":
    main()
