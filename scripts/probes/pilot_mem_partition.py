"""试点 3（决定性）：互斥 TPC 分区能否隔离显存带宽争用？

假设：掩码切的是**计算**，切不动**显存**。若一个访存受限负载在 6 TPC 上就已把 HBM 带宽
吃满，那么两个各拿 6 TPC 的访存负载并发时必然互相拖慢约 2×——**互斥分区挡不住**。

对照组：计算受限负载（matmul）在同样设置下应能接近线性并发（各自变慢不多）。

设计：两个专用线程各持自己的粘性线程掩码（0+6 / 6+6，由适配器分配器保证互斥）。
  阶段 A  各自单独跑（掩码已施加），测单侧耗时
  阶段 B  同时跑，用 barrier 对齐起跑，测并发墙钟总耗时
判据：slowdown = 并发时单侧耗时 / 单独时单侧耗时。≈1 为无干扰，≈2 为完全争用。

用法：
    LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so python /tmp/pilot_mem_partition.py
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

    # 每个线程一份独立缓冲，避免 L2 复用造成的假象；工作集远大于 32 MiB L2
    n = 64 * MIB  # float32 -> 256 MiB
    bufs = {k: (torch.randn(n, device=dev), torch.empty(n, device=dev)) for k in ("A", "B")}
    ma = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    mb = torch.randn(4096, 4096, device=dev, dtype=torch.float16)

    wa, wb = Worker(), Worker()

    def setup(w, tag):
        r = w.call(lambda: adapter.apply_quota(7, 0.5))
        assert r.get("enforced"), r
        return r

    ra = setup(wa, "A")
    rb = setup(wb, "B")
    print(f"线程 A 掩码: TPC {ra['tpc_start']}+{ra['tpc_count']}   "
          f"线程 B 掩码: TPC {rb['tpc_start']}+{rb['tpc_count']}   "
          f"互斥={set(range(ra['tpc_start'], ra['tpc_start']+ra['tpc_count'])).isdisjoint(range(rb['tpc_start'], rb['tpc_start']+rb['tpc_count']))}")

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

    def solo(w, tag, kind):
        body = make_body(tag, kind)

        def go():
            body()          # 暖机
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            body()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000
        return w.call(go)

    def concurrent(kind):
        """两线程 barrier 对齐后同时开跑，返回 (A 耗时, B 耗时, 并发墙钟)。"""
        barrier = threading.Barrier(2)
        boxes = {}

        def go(tag, w):
            body = make_body(tag, kind)

            def inner():
                body()                      # 暖机
                torch.cuda.synchronize()
                barrier.wait()              # 对齐起跑
                t0 = time.perf_counter()
                body()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1000
            boxes[tag] = w.call(inner)

        t0 = time.perf_counter()
        ta = threading.Thread(target=go, args=("A", wa))
        tb = threading.Thread(target=go, args=("B", wb))
        ta.start(); tb.start(); ta.join(); tb.join()
        wall = (time.perf_counter() - t0) * 1000
        return boxes["A"], boxes["B"], wall

    out = {}
    for kind, unit, total_gb in (("mem", "GB", 2 * 256 * MIB * ITERS / 1e9),
                                 ("mm", "TFLOP", 2 * 4096**3 * ITERS / 1e12)):
        print(f"\n===== {'访存受限 (copy 256MiB)' if kind=='mem' else '计算受限 (4096^3 fp16 matmul)'} =====")
        sa = [solo(wa, "A", kind) for _ in range(3)]
        sb = [solo(wb, "B", kind) for _ in range(3)]
        A = statistics.median(sa); B = statistics.median(sb)
        ca, cb, wall = [], [], []
        for _ in range(3):
            a, b, w = concurrent(kind)
            ca.append(a); cb.append(b); wall.append(w)
        CA = statistics.median(ca); CB = statistics.median(cb); W = statistics.median(wall)

        print(f"  单独: A={A:.1f} ms  B={B:.1f} ms")
        print(f"  并发: A={CA:.1f} ms  B={CB:.1f} ms  (墙钟 {W:.1f} ms)")
        print(f"  slowdown: A={CA/A:.2f}×  B={CB/B:.2f}×")
        print(f"  聚合吞吐: 单独={2*total_gb/((A+B)/1000):.1f} {unit}/s   "
              f"并发={2*total_gb/(W/1000):.1f} {unit}/s   "
              f"并发/单独={ (2*total_gb/(W/1000)) / (2*total_gb/((A+B)/1000)):.2f}×")
        out[kind] = {
            "solo_A_ms": A, "solo_B_ms": B, "conc_A_ms": CA, "conc_B_ms": CB, "conc_wall_ms": W,
            "slowdown_A": CA / A, "slowdown_B": CB / B,
            "agg_solo_per_s": 2 * total_gb / ((A + B) / 1000),
            "agg_conc_per_s": 2 * total_gb / (W / 1000), "unit": unit,
        }

    with open("/tmp/pilot_mem_partition.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n已写入 /tmp/pilot_mem_partition.json")


if __name__ == "__main__":
    main()
