"""稳定性检查：同一会话内交替测 S12 与 C12，看并发收益的分布。"""
import queue, statistics, sys, threading, time, zlib, json
sys.path.insert(0, "/home/lyon/projects/bishe")
import torch
from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

class Worker:
    def __init__(self, wid, enc):
        self.wid, self.enc = wid, enc
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            fn, box = self.q.get()
            if fn is None: return
            try: box["r"] = fn()
            except Exception as e: box["e"] = e
    def call(self, fn):
        box = {}; self.q.put((fn, box))
        while not box: time.sleep(0.0005)
        if "e" in box: raise box["e"]
        return box["r"]

cfg = load_config("config.libsmctrl.example.yaml")
res = create_resource_backend(cfg.executor.resource_backend, cfg.executor.libsmctrl_adapter,
                              cfg.executor.allow_proxy_fallback)
enc = ClipEncoderBackend(cfg.model, res, 2)
wa, wb = Worker(0, enc), Worker(1, enc)

def mk(size, tag):
    return EncodeJob(f"{size}-{tag}", zlib.crc32(f"{size}-{tag}".encode()) % (2**31), size, size, 600_000)

def once(w, size, q, tag):
    j = mk(size, tag); j.sm_fraction = q
    return w.call(lambda: w.enc.encode(j, w.wid).execution_ms)

def conc(wa, wb, size, qa, qb, tag):
    box = {}; gate = threading.Barrier(3)
    def go(w, q, n):
        j = mk(size, f"{tag}-{n}"); j.sm_fraction = q
        gate.wait()
        box[n] = w.call(lambda: w.enc.encode(j, w.wid).execution_ms)
    t1 = threading.Thread(target=go, args=(wa, qa, "A")); t2 = threading.Thread(target=go, args=(wb, qb, "B"))
    t1.start(); t2.start(); gate.wait(); t1.join(); t2.join()
    return max(box["A"], box["B"])

for size in (672, 1024, 2048):
    once(wa, size, 1.0, "warm")
    rows = []
    for i in range(8):
        s = once(wa, size, 1.0, f"s{i}")          # 全卡串行（每请求）
        c = conc(wa, wb, size, 1.0, 1.0, f"c{i}") # 共享并发（墙钟）
        rows.append((s, c, c / s))                 # c/s = 并发单请求相对串行的膨胀
    s12 = [r[0] for r in rows]; wall = [r[1] for r in rows]
    print(f"\n=== {size}（patch {(size//32)**2}）===")
    print(f"  S12 单请求 : 中位 {statistics.median(s12):7.2f}  极差 {min(s12):6.2f}–{max(s12):6.2f}")
    print(f"  C12 墙钟   : 中位 {statistics.median(wall):7.2f}  极差 {min(wall):6.2f}–{max(wall):6.2f}")
    ratios = [2/(c/ (s)) if False else (2*s)/c for s, c, _ in rows]   # 聚合比 = 2*s/c
    print(f"  聚合比C12/S12: 中位 {statistics.median(ratios):.3f}  "
          f"全部 {[round(x,3) for x in ratios]}")
