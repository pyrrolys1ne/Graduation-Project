#!/usr/bin/env python3
"""Overview of results/ directory structure + summary.json harvesting."""
import os, json, glob, statistics as st

ROOT = "/home/lyon/projects/bishe/results"

print("=" * 100)
print("SECTION A: DIRECTORY OVERVIEW")
print("=" * 100)

rows = []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    subdirs = sorted([x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))])
    files = sorted([x for x in os.listdir(p) if os.path.isfile(os.path.join(p, x))])
    # count repeats
    n_rep = 0
    n_jsonl = 0
    for sd in subdirs:
        sp = os.path.join(p, sd)
        for f in os.listdir(sp):
            if f.endswith(".jsonl"):
                n_jsonl += 1
                if f.startswith("repeat"):
                    n_rep += 1
    rows.append((d, len(subdirs), subdirs[:8], [f for f in files if f.endswith(('.json','.csv'))][:6], n_jsonl, n_rep))

print(f"{'dir':<32}{'#sub':>5}  {'rep files':>10} {'jsonl':>7}  summary files")
print("-" * 100)
for d, ns, sd, fs, nj, nr in rows:
    print(f"{d:<32}{ns:>5}  {nr:>10} {nj:>7}  {','.join(fs)}")

print()
print("=" * 100)
print("SECTION B: ROOT-LEVEL FILES")
print("=" * 100)
for f in sorted(os.listdir(ROOT)):
    fp = os.path.join(ROOT, f)
    if os.path.isfile(fp):
        print(f"  {f:<45} {os.path.getsize(fp):>10,} bytes")

# ---- Harvest all repeat-level summary.json ----
print()
print("=" * 100)
print("SECTION C: ALL repeat*.summary.json  (top-level aggregates)")
print("=" * 100)

HDR = f"{'experiment/variant/rep':<48}{'sub':>5}{'done':>6}{'thr':>9}{'t_mean':>9}{'t_p95':>9}{'t_p99':>9}{'slo':>7}{'gpu%':>7}{'gsmp':>5}{'q_mean':>9}{'e_mean':>8}"
print(HDR)
print("-" * len(HDR))

all_summaries = []
for exp in sorted(os.listdir(ROOT)):
    ep = os.path.join(ROOT, exp)
    if not os.path.isdir(ep):
        continue
    for var in sorted(os.listdir(ep)):
        vp = os.path.join(ep, var)
        if not os.path.isdir(vp):
            continue
        for f in sorted(os.listdir(vp)):
            if not f.endswith(".summary.json"):
                continue
            fp = os.path.join(vp, f)
            try:
                s = json.load(open(fp))
            except Exception as e:
                print(f"  !! parse error {fp}: {e}")
                continue
            rec = dict(exp=exp, var=var, file=f, path=fp, raw=s)
            all_summaries.append(rec)

# print grouped
cur = None
for r in all_summaries:
    s = r["raw"]
    key = f"{r['exp']}/{r['var']}"
    if key != cur:
        cur = key
    tl = s.get("total_latency_ms", {}) or {}
    ql = s.get("queue_latency_ms", {}) or {}
    el = s.get("execution_latency_ms", {}) or {}
    print(f"{key + '/' + r['file'].replace('.summary.json',''):<48}"
          f"{s.get('submitted',-1):>5}{s.get('completed',-1):>6}"
          f"{s.get('throughput_rps',float('nan')):>9.2f}"
          f"{tl.get('mean',float('nan')):>9.2f}"
          f"{tl.get('p95',float('nan')):>9.2f}"
          f"{tl.get('p99',float('nan')):>9.2f}"
          f"{s.get('slo_violation_rate',float('nan')):>7.3f}"
          f"{s.get('gpu_utilization_percent',float('nan')):>7.2f}"
          f"{s.get('gpu_samples',-1):>5}"
          f"{ql.get('mean',float('nan')):>9.2f}"
          f"{el.get('mean',float('nan')):>8.2f}")

json.dump([{k: v for k, v in r.items() if k != 'raw'} for r in all_summaries],
          open("/tmp/all_summaries_index.json", "w"), indent=1)
print(f"\nTotal repeat summaries found: {len(all_summaries)}")
