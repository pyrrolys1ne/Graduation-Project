#!/usr/bin/env python3
"""Deep analysis of raw JSONL request logs."""
import os, json, glob, statistics as st, math

ROOT = "/home/lyon/projects/bishe/results"

def pct(xs, q):
    if not xs: return float('nan')
    xs = sorted(xs); i = min(len(xs)-1, max(0, int(round(q*(len(xs)-1)))))
    return xs[i]

def find_jsonl():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for f in filenames:
            if f.endswith(".jsonl"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)

# ---------- 1. GPU samples audit across all summary json ----------
print("=" * 110)
print("GPU SAMPLE AUDIT - every summary file that reports gpu_samples")
print("=" * 110)
def walk_summaries():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for f in filenames:
            if f.endswith(".json") and "summary" in f:
                yield os.path.join(dirpath, f)

low = []
allg = []
for fp in sorted(walk_summaries()):
    try: S = json.load(open(fp))
    except Exception as e: continue
    recs = []
    if isinstance(S, list):
        for r in S:
            if isinstance(r, dict) and "gpu_samples" in r:
                recs.append((r.get("policy","?"), r.get("gpu_samples"), r.get("gpu_utilization_percent")))
    elif isinstance(S, dict):
        if "variants" in S:
            for vn, vb in S["variants"].items():
                a = (vb.get("aggregate") or {})
                gs = a.get("gpu_samples"); gu = a.get("gpu_utilization_percent")
                if isinstance(gs, dict):
                    recs.append((vn, gs.get("mean"), gu.get("mean") if isinstance(gu,dict) else gu))
                elif gs is not None:
                    recs.append((vn, gs, gu))
        elif "gpu_samples" in S:
            recs.append((S.get("policy", os.path.basename(fp)), S["gpu_samples"], S.get("gpu_utilization_percent")))
    for name, gs, gu in recs:
        rel = os.path.relpath(fp, ROOT)
        allg.append((rel, name, gs, gu))
        if gs is not None and gs < 10:
            low.append((rel, name, gs, gu))

print(f"total (file,variant) gpu_samples records: {len(allg)}")
print(f"\n*** ENTRIES WITH gpu_samples < 10 : {len(low)} ***")
for rel, name, gs, gu in sorted(low):
    print(f"  {rel:<60} {str(name):<22} samples={gs:<6} util%={gu}")

print("\n--- gpu_samples distribution (all variants) ---")
from collections import Counter
c = Counter(int(g) for _,_,g,_ in allg if g is not None and float(g)==int(g))
for k in sorted(c): print(f"  samples={k:<5} count={c[k]}")

# ---------- 2. client vs server latency ----------
print()
print("=" * 110)
print("CLIENT vs SERVER LATENCY (client_latency_ms / total_ms)")
print("=" * 110)
rows = []
for fp in find_jsonl():
    tot, cli, st_, q, e = [], [], [], [], []
    n = 0
    try:
        with open(fp, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except Exception: continue
                n += 1
                if r.get("total_ms") is not None: tot.append(r["total_ms"])
                if r.get("client_latency_ms") is not None: cli.append(r["client_latency_ms"])
                if r.get("status"): st_.append(r["status"])
                if r.get("queue_ms") is not None: q.append(r["queue_ms"])
                if r.get("execution_ms") is not None: e.append(r["execution_ms"])
    except Exception as ex:
        print(f"  !! {fp}: {ex}"); continue
    if not tot and not cli:
        continue
    rel = os.path.relpath(fp, ROOT)
    ratio = (st.mean(cli)/st.mean(tot)) if (cli and tot and st.mean(tot)>0) else float('nan')
    rows.append(dict(path=rel, n=n, ntot=len(tot), ncli=len(cli),
                     srv_mean=st.mean(tot) if tot else float('nan'),
                     cli_mean=st.mean(cli) if cli else float('nan'),
                     srv_std=st.pstdev(tot) if tot else float('nan'),
                     cli_std=st.pstdev(cli) if cli else float('nan'),
                     ratio=ratio,
                     q_mean=st.mean(q) if q else float('nan'),
                     e_mean=st.mean(e) if e else float('nan'),
                     statuses=dict((s, st_.count(s)) for s in set(st_))))

print(f"{'file':<58}{'n':>6}{'srv_mean':>10}{'cli_mean':>10}{'ratio':>8}{'q_mean':>9}{'e_mean':>9}  status")
print("-" * 125)
for r in rows:
    stt = ",".join(f"{k}:{v}" for k,v in list(r["statuses"].items())[:3])
    print(f"{r['path']:<58}{r['n']:>6}{r['srv_mean']:>10.2f}{r['cli_mean']:>10.2f}{r['ratio']:>8.2f}"
          f"{r['q_mean']:>9.2f}{r['e_mean']:>9.2f}  {stt}")

json.dump(rows, open("/tmp/jsonl_latency.json","w"), indent=1)

# ---------- 3. queue/exec ratio ----------
print()
print("=" * 110)
print("QUEUE vs EXECUTION SHARE")
print("=" * 110)
print(f"{'file':<58}{'q/tot':>9}{'e/tot':>9}{'q_std/q':>10}{'e_std/e':>10}")
print("-"*110)
for r in rows:
    if math.isnan(r["srv_mean"]) or r["srv_mean"]==0: continue
    qs = r["q_mean"]/r["srv_mean"]; es = r["e_mean"]/r["srv_mean"]
    print(f"{r['path']:<58}{qs:>9.3f}{es:>9.3f}{r['q_std'] if False else '':>10}{'':>10}")
