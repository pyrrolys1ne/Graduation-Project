"""CLIP 前向四臂对照（真实前向，非合成 kernel）。
 S12 全卡串行 / S6 半卡串行 / C66 二等分分区并发 / C12 不分区共享并发
聚合口径：S12 = 2/(2t)；并发 = 2/max(t_a,t_b)（同一 barrier 起跑）。
"""
import json, queue, statistics, sys, threading, time, zlib
sys.path.insert(0, "/home/lyon/projects/bishe")
import torch
from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

SIZES = (672, 1024, 1344, 1536, 1792, 2048)
REP = 5

class Worker:
    def __init__(self, wid, enc):
        self.wid, self.enc = wid, enc
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
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

def mkjob(size, tag):
    return EncodeJob(f"{size}-{tag}", zlib.crc32(f"{size}-{tag}".encode()) % (2**31),
                     size, size, 600_000)

def once(w, size, quota, tag):
    j = mkjob(size, tag); j.sm_fraction = quota
    return w.call(lambda: w.enc.encode(j, w.wid).execution_ms)

def concurrent(wa, wb, size, qa, qb, tag):
    box = {}; gate = threading.Barrier(3)
    def go(w, quota, name):
        j = mkjob(size, f"{tag}-{name}"); j.sm_fraction = quota
        gate.wait()
        box[name] = w.call(lambda: w.enc.encode(j, w.wid).execution_ms)
    t1 = threading.Thread(target=go, args=(wa, qa, "A"))
    t2 = threading.Thread(target=go, args=(wb, qb, "B"))
    t1.start(); t2.start(); gate.wait(); t1.join(); t2.join()
    return box["A"], box["B"]

out = {}
for size in SIZES:
    wa, wb = Worker(0, enc), Worker(1, enc)
    for i in range(2):                       # 暖机
        once(wa, size, 1.0, f"warm{i}")
    s12 = statistics.median([once(wa, size, 1.0, f"s12-{i}") for i in range(REP)])
    s6  = statistics.median([once(wa, size, 0.5, f"s6-{i}")  for i in range(REP)])
    c66 = [concurrent(wa, wb, size, 0.5, 0.5, f"c66-{i}") for i in range(REP)]
    c12 = [concurrent(wa, wb, size, 1.0, 1.0, f"c12-{i}") for i in range(REP)]
    a66 = statistics.median(x[0] for x in c66); b66 = statistics.median(x[1] for x in c66)
    a12 = statistics.median(x[0] for x in c12); b12 = statistics.median(x[1] for x in c12)
    agg_s12 = 2 / (2 * s12); agg_s6 = 2 / (2 * s6)
    agg_c66 = 2 / max(a66, b66); agg_c12 = 2 / max(a12, b12)
    out[size] = {"s12": s12, "s6": s6, "c66": [a66, b66], "c12": [a12, b12],
                 "r_c66": agg_c66 / agg_s12, "r_c12": agg_c12 / agg_s12,
                 "r_s6": agg_s6 / agg_s12}
    print(f"{size:>5} 完成  S12 {s12:7.2f}  S6 {s6:7.2f}  "
          f"C66 {a66:7.2f}/{b66:7.2f}  C12 {a12:7.2f}/{b12:7.2f}", flush=True)

print(f"\n{'尺寸':>6} {'patch':>6} {'S12串行':>8} {'S6串行':>8} {'C66分区':>8} {'C12共享':>8} "
      f"{'C66/S12':>8} {'C12/S12':>8} {'S6/S12':>8}")
for size, r in out.items():
    print(f"{size:>6} {(size//32)**2:>6} {r['s12']:8.2f} {r['s6']:8.2f} "
          f"{r['c66'][0]:8.2f} {r['c12'][0]:8.2f} "
          f"{r['r_c66']:8.3f} {r['r_c12']:8.3f} {r['r_s6']:8.3f}")
json.dump(out, open("/tmp/clip_arms2.json", "w"), indent=2)
