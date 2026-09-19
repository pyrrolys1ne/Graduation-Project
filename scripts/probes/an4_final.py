#!/usr/bin/env python3
import json, os, statistics as st, itertools

ROOT = "/home/lyon/projects/bishe/results"

def S(exp):
    p = os.path.join(ROOT, exp, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None

def agg(exp):
    d = S(exp)
    out = {}
    if not d or not isinstance(d, dict) or "variants" not in d: return out
    for vn, vb in d["variants"].items():
        a = vb.get("aggregate", {}) or {}
        def m(f):
            x = a.get(f, {}); return x.get("mean") if isinstance(x, dict) else None
        def s(f):
            x = a.get(f, {}); return x.get("std") if isinstance(x, dict) else None
        out[vn] = dict(thr=m("throughput_rps"), thr_s=s("throughput_rps"),
                       tmean=m("total_mean_ms"), tmean_s=s("total_mean_ms"),
                       tp99=m("total_p99_ms"), tp99_s=s("total_p99_ms"),
                       slo=m("slo_violation_rate"), slo_s=s("slo_violation_rate"),
                       gpu=m("gpu_utilization_percent"), gsmp=m("gpu_samples"))
    return out

print("#" * 110)
print("# A. RANK INVERSION ACROSS LOAD LEVELS (same 6 batch variants, 3 workloads)")
print("#" * 110)
LOADS = [("comparison_mixed (arrival 57.3 rps)", "batch_comparison_mixed"),
         ("saturating_12ms (arrival 84.6 rps)", "batch_saturating_12ms"),
         ("saturating_4ms (arrival 256.1 rps)", "batch_saturating_4ms")]
data = {nm: agg(e) for nm, e in LOADS}
vars_ = sorted(data[LOADS[0][0]].keys())
for metric, lo in (("tmean", "mean total_ms (lower=better)"), ("thr", "throughput rps (higher=better)")):
    print(f"\n--- ranking by {lo} ---")
    rk = {}
    for nm, e in LOADS:
        d = data[nm]
        order = sorted(vars_, key=lambda v: d[v][metric] if d[v][metric] is not None else 9e9,
                       reverse=(metric == "thr"))
        rk[nm] = order
        print(f"  {nm:<38} " + " < ".join(f"{v}({d[v][metric]:.2f})" for v in order))
    print("  --- rank change per variant ---")
    for v in vars_:
        ranks = [rk[nm].index(v) + 1 for nm, _ in LOADS]
        if max(ranks) - min(ranks) >= 2:
            print(f"   !! {v:<18} ranks {ranks}  (span {max(ranks)-min(ranks)})")

print()
print("#" * 110)
print("# B. STATISTICAL SEPARABILITY of headline pairs (|Δ| / pooled sd)")
print("#" * 110)
def sep(exp, va, vb, field):
    d = agg(exp)
    if va not in d or vb not in d: return None
    a, b = d[va], d[vb]
    if a[field] is None or b[field] is None: return None
    sa, sb = a[field + "_s"], b[field + "_s"]
    delta = a[field] - b[field]
    pooled = ((sa or 0) ** 2 + (sb or 0) ** 2) ** 0.5
    return delta, pooled, (abs(delta) / pooled if pooled else float("inf")), a[field], b[field]

PAIRS = [
    ("overload_baselines", "dacc", "multistream_fcfs", "thr"),
    ("overload_baselines", "edf", "multistream_fcfs", "thr"),
    ("overload_baselines", "edf", "serial_fcfs", "thr"),
    ("fixed_baselines", "dacc", "multistream_fcfs", "thr"),
    ("fixed_baselines", "edf", "multistream_fcfs", "thr"),
    ("fixed_baselines", "dacc", "edf", "tmean"),
    ("fixed_baselines", "edf", "multistream_fcfs", "tmean"),
    ("fixed_baselines", "serial_fcfs", "edf_size", "tmean"),
    ("saturated_baselines", "dacc", "multistream_fcfs", "tmean"),
    ("saturated_baselines", "dacc", "edf", "tmean"),
    ("saturated_baselines", "edf", "multistream_fcfs", "tmean"),
    ("batch_saturating_4ms", "batch4_d0", "spatial_2stream", "thr"),
    ("batch_saturating_4ms", "batch4_d0", "spatial_2stream", "tmean"),
    ("batch_saturating_4ms", "batch4_d0", "spatial_2stream", "tp99"),
    ("batch_saturating_12ms", "batch4_d0", "spatial_2stream", "tmean"),
    ("spatial_saturating_4ms", "spatial_tpc", "spatial_proxy", "thr"),
    ("spatial_saturating_4ms", "spatial_tpc", "spatial_proxy", "tmean"),
    ("spatial_saturating_4ms", "batch4_d0_tpc", "spatial_tpc", "thr"),
    ("spatial_saturating_12ms", "spatial_tpc", "spatial_proxy", "tmean"),
    ("spatial_comparison_mixed", "spatial_tpc", "spatial_proxy", "tmean"),
    ("fixed_ablations", "dacc_no_complementarity", "dacc_no_pairing", "thr"),
    ("fixed_window", "dacc_k1", "dacc_k16", "thr"),
    ("fixed_window", "dacc_k1", "dacc_k16", "tmean"),
    ("repeated_baselines", "dacc", "edf", "tmean"),
    ("repeated_window", "dacc_k1", "dacc_k16", "tmean"),
]
print(f"{'experiment':<28}{'A vs B':<44}{'metric':<7}{'A':>9}{'B':>9}{'|d|/sd':>8}  verdict")
print("-" * 136)
for exp, va, vb, f in PAIRS:
    r = sep(exp, va, vb, f)
    if r is None: continue
    d, pooled, z, av, bv = r
    verdict = "SEPARATED" if z > 3 else ("marginal" if z > 1.5 else "INDISTINGUISHABLE")
    print(f"{exp:<28}{va+' vs '+vb:<44}{f:<7}{av:>9.2f}{bv:>9.2f}{z:>8.2f}  {verdict}")

print()
print("#" * 110)
print("# C. THROUGHPUT PINNED BY ARRIVAL RATE")
print("#" * 110)
ARR = {"batch_comparison_mixed": 57.3, "repeated_baselines": 57.3, "repeated_ablations": 57.3,
       "repeated_window": 57.3, "spatial_comparison_mixed": 57.3,
       "batch_saturating_12ms": 84.6, "saturated_baselines": 84.6, "saturated_ablations": 84.6,
       "spatial_saturating_12ms": 84.6,
       "batch_saturating_4ms": 256.1, "fixed_baselines": 256.1, "fixed_ablations": 256.1,
       "fixed_window": 256.1, "fixed_pairing": 256.1, "fixed_quota": 256.1,
       "overload_baselines": 256.1, "overload_ablations": 256.1,
       "spatial_saturating_4ms": 256.1}
print(f"{'experiment':<26}{'arrival':>9}{'thr min':>9}{'thr max':>9}{'spread':>8}{'max/arr':>9}  pinned?")
print("-" * 92)
for exp, arr in ARR.items():
    d = agg(exp)
    if not d: continue
    thr = [v["thr"] for v in d.values() if v["thr"]]
    if not thr: continue
    spread = max(thr) - min(thr)
    print(f"{exp:<26}{arr:>9.1f}{min(thr):>9.2f}{max(thr):>9.2f}{spread:>8.2f}{max(thr)/arr:>9.3f}  "
          f"{'YES (<=arrival)' if max(thr) <= arr*1.005 else 'no (server-limited)'}")

print()
print("#" * 110)
print("# D. GPU UTILIZATION vs THROUGHPUT (across all variants with n>=10 samples)")
print("#" * 110)
pts = []
for exp in sorted(os.listdir(ROOT)):
    for vn, v in agg(exp).items():
        if v["gpu"] and v["gsmp"] and v["gsmp"] >= 10 and v["thr"]:
            pts.append((exp, vn, v["gpu"], v["thr"], v["gsmp"]))
xs = [p[2] for p in pts]; ys = [p[3] for p in pts]
mx, my = st.mean(xs), st.mean(ys)
num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
den = (sum((a-mx)**2 for a in xs) ** 0.5) * (sum((b-my)**2 for b in ys) ** 0.5)
print(f"n={len(pts)}  corr(gpu_util%, throughput) = {num/den:.3f}")
print("\ntop 8 by gpu util:")
for p in sorted(pts, key=lambda x: -x[2])[:8]:
    print(f"   {p[0]+'/'+p[1]:<46} gpu={p[2]:6.2f}%  thr={p[3]:7.2f}  n={p[4]}")
print("bottom 5 by gpu util:")
for p in sorted(pts, key=lambda x: x[2])[:5]:
    print(f"   {p[0]+'/'+p[1]:<46} gpu={p[2]:6.2f}%  thr={p[3]:7.2f}  n={p[4]}")

print()
print("#" * 110)
print("# E. SAME-CONFIG REPLICATES ACROSS DIRECTORY FAMILIES")
print("#" * 110)
FAMS = [("repeated_baselines", "comparison_mixed"), ("saturated_baselines", "saturating_12ms"),
        ("overload_baselines", "saturating_4ms"), ("fixed_baselines", "saturating_4ms")]
POLS = ["serial_fcfs", "multistream_fcfs", "edf", "edf_size", "dacc"]
print(f"{'policy':<18}" + "".join(f"{f[0][:14]:>16}" for f in FAMS))
print("-" * (18 + 16 * len(FAMS)))
for p in POLS:
    row = f"{p:<18}"
    for f, _ in FAMS:
        d = agg(f)
        row += f"{d.get(p,{}).get('thr',float('nan')):>16.2f}" if p in d else f"{'-':>16}"
    print(row)
print("\n  same config families that disagree (identical config.yaml apart from logging.output_dir):")
for i, j in ((2, 3),):
    for p in POLS:
        a = agg(FAMS[i][0]).get(p, {}).get("thr")
        b = agg(FAMS[j][0]).get(p, {}).get("thr")
        if a and b:
            print(f"    {p:<18} overload={a:7.2f}  fixed={b:7.2f}  ratio={a/b:.2f}x  delta={100*(a-b)/b:+.1f}%")
