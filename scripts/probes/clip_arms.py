"""CLIP 前向的四臂对照：全卡串行 / 二等分分区并发 / 不分区共享并发。
用真实 CLIP 前向，不是合成 kernel。"""
import json, queue, statistics, sys, threading, time, zlib
sys.path.insert(0, "/home/lyon/projects/bishe")
import torch
from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

REP = 5
SIZES = (672, 1024)

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
        while not box: time.sleep(0.001)
        if "e" in box: raise box["e"]
        return box["r"]

cfg = load_config("config.libsmctrl.example.yaml")
res = create_resource_backend(cfg.executor.resource_backend, cfg.executor.libsmctrl_adapter,
                              cfg.executor.allow_proxy_fallback)
enc = ClipEncoderBackend(cfg.model, res, 2)
out = {}
for size in SIZES:
    def job(i, size=size):
        return EncodeJob(f"arm-{size}-{i}", zlib.crc32(f"{size}-{i}".encode()) % (2**31),
                         size, size, 600_000)
    def run(w, wj, quota, i):
        wj.sm_fraction = quota
        return w.call(lambda: w.enc.encode(wj, w.wid).execution_ms).real if False else \
               w.call(lambda: w.enc.encode(wj, w.wid).execution_ms)

    for _ in range(2):                      # 暖机
        run(Worker(0, enc), job(-1), 1.0, 0); run(Worker(1, enc), job(-2), 1.0, 0)

    rec = {}
    # --- S12 全卡串行 ---
    w0 = Worker(0, enc)
    s = [statistics.median([run(w0, job(i01), 1.0, i01) for i01 in range(3)]) for _ in range(REP)]
    rec["S12"] = statistics.median(s)
    # --- C66 二等分分区并发 ---
    wa, wb = Worker(0, enc), Worker(1, enc)
    got = []; 
    for _ in range(REP):
        barrier = threading.Barrier(2); box = {}
        def go(w, tag, quota):
            def inner():
                r = run(w, job(hash(tag) % 1000), quota, 0); barrier.wait(); return r
            box[tag] = w.call(inner)
        t1 = threading.Thread(target=go, args=(wa, "A", 0.5))
        t2 = threading.Thread(target=go, args=(wb, "B", 0.5))
        # 重新写：并发必须两个线程同时进 encode
        start = threading.Barrier(3)
        def worker(w, tag, quota):
            jj = job(abs(hash(tag)) % 10000); jj.sm_fraction = quota
            start.wait()
            box[tag] = w.call(lambda: w.enc.encode(jj, w.wid).execution_ms)
        t1 = threading.Thread(target=worker, args=(wa, "A", 0.5))
        t2 = threading.Thread(target=worker, args=(wb, "B", 0.5))
        t1.start(); t2.start(); start.wait(); t1.join(); t2.join()
        got.append((box["A"], box["B"]))
    ca = statistics.median(x[0] for x in got); cb = statistics.median(x[1] for x in got)
    rec["C66"] = {"a": ca, "b": cb}
    # --- C12 不分区共享并发 ---
    got2 = []
    for _ in range(REP):
        box = {}; start = threading.Barrier(3)
        def worker2(w, tag):
            jj = job(abs(hash(tag)) % 10000); jj.sm_fraction = 1.0
            start.wait()
            box[tag] = w.call(lambda: w.enc.encode(jj, w.wid).execution_ms)
        t1 = threading.Thread(target=worker2, args=(wa, "A"))
        t2 = threading.Thread(target=worker2, args=(wb, "B"))
        t1.start(); t2.start(); start.wait(); t1.join(); t2.join()
        got2.append((box["A"], box["B"]))
    da = statistics.median(x[0] for x in got2); db = statistics.median(x[1] for x in got2)
    rec["C12"] = {"a": da, "b": db}
    out[size] = rec
    print(f"{size}: S12 serial {rec['S12']:.2f} | C66 {ca:.2f}/{cb:.2f} | C12 {da:.2f}/{db:.2f}", flush=True)

print(f"\n{'尺寸':>6} {'S12串行':>9} {'C66分区并发':>26} {'C12共享并发':>26} {'C66/S12':>9} {'C12/S12':>9}")
for size, r in out.items():
    s = r["S12"]; c = r["C66"]; d = r["C12"]
    agg66 = 2/s if False else (2/(c["a"]/s + 1) )  # placeholder
    # 聚合吞吐：两个作业各自的执行时间
    aggS = 2 / (s/s)      # S12 两个作业串行：每个 s
    aggS = 2 / (s + s)
    agg66 = 1/(c["a"]/1000)/2 + 1/(c["b"]/1000)/2
    agg12 = 1/(d["a"]/1000)/2 + 1/(d["b"]/1000)/2
    print(f"{size:>6} {s:>9.2f} {c['a']:>12.2f}/{c['b']:<12.2f} {d['a']:>12.2f}/{d['b']:<12.2f} "
          f"{agg66/aggS:>9.3f} {agg12/aggS:>9.3f}")
json.dump(out, open("/tmp/clip_arms.json","w"), indent=2)
