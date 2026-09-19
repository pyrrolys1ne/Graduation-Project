#!/usr/bin/env python3
"""Aggregate tables from per-experiment summary.json, with CV (std/mean) flags."""
import os, json, math

ROOT = "/home/lyon/projects/bishe/results"

EXPS = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT, d))]

def fnum(x, w=8, p=2):
    if x is None: return f"{'-':>{w}}"
    try: return f"{float(x):>{w}.{p}f}"
    except: return f"{str(x):>{w}}"

for exp in EXPS:
    sp = os.path.join(ROOT, exp, "summary.json")
    if not os.path.exists(sp):
        continue
    try:
        S = json.load(open(sp))
    except Exception as e:
        print(f"!! {exp}: {e}"); continue
    vars_ = S.get("variants")
    if not vars_:
        continue
    meta = {k: v for k, v in S.items() if k not in ("variants",)}
    print("=" * 118)
    print(f"EXPERIMENT: {exp}")
    print(f"  meta: {json.dumps({k: v for k, v in meta.items() if not isinstance(v, (dict, list))}, ensure_ascii=False)}")
    for k in ("group", "repeats", "workload", "concurrency", "sm_isolation_verified", "note", "resource_backend", "config"):
        if k in meta:
            print(f"    {k} = {meta[k]}")
    hdr = (f"{'variant':<24}{'rep':>4}{'thr':>9}{'thr_cv':>8}{'tmean':>9}{'tmean_cv':>9}"
           f"{'tp95':>9}{'tp99':>9}{'tp99_cv':>9}{'slo':>8}{'slo_cv':>8}{'gpu%':>7}{'gsmp':>6}{'done':>6}")
    print(hdr); print("-" * len(hdr))
    for vname, vb in vars_.items():
        agg = vb.get("aggregate", {}) or {}
        def m(f):
            x = agg.get(f, {})
            return x.get("mean") if isinstance(x, dict) else x
        def s(f):
            x = agg.get(f, {})
            return x.get("std") if isinstance(x, dict) else None
        def cv(f):
            mm, ss = m(f), s(f)
            if mm in (None, 0) or ss is None: return None
            return ss / abs(mm)
        print(f"{vname:<24}{vb.get('repeats',''):>4}"
              f"{fnum(m('throughput_rps'),9)}{fnum(cv('throughput_rps'),8,2)}"
              f"{fnum(m('total_mean_ms'),9)}{fnum(cv('total_mean_ms'),9,2)}"
              f"{fnum(m('total_p95_ms'),9)}{fnum(m('total_p99_ms'),9)}{fnum(cv('total_p99_ms'),9,2)}"
              f"{fnum(m('slo_violation_rate'),8,4)}{fnum(cv('slo_violation_rate'),8,2)}"
              f"{fnum(m('gpu_utilization_percent'),7)}{fnum(m('gpu_samples'),6,1)}"
              f"{fnum(m('completed'),6,0)}")
    print()
