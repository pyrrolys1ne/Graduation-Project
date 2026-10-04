"""判定：SM 掩码能不能"烧进" CUDA Graph —— 在捕获阶段施加，回放时是否仍然生效？

【为什么这是前置条件】
已知（`diag_graph_vs_eager.py`，本机 CUDA UMD 13.3）：回调路径挂在内核发射上，
而图回放走 `cuGraphLaunch`，绕开那条路 —— 回放时设掩码**不生效**（实测 0.84×）。
于是"请求级 SM 配额"与"图回放主路径"在当前实现下无法并存。

剩下唯一在原理上不依赖"驱动何时读掩码"的方案是：**让掩码成为图的一部分**——
捕获时施加，则图里录下的就是被掩码的发射。代价是图的数量变成 尺寸 × 配额档位。

【三条臂】

  A. 无掩码捕获 → 无掩码回放     基准
  B. 3 TPC 掩码下捕获 → 无掩码回放   ← 本脚本要判定的就是这一条
  C. 无掩码捕获 → 3 TPC 掩码下回放  已知无效，作为阴性对照一并复测

另测一次 eager 在 3 TPC 下的耗时，作为"掩码生效长什么样"的标尺
（已有标尺：1 个 TPC 使矩阵乘慢约 11–12 倍）。

【事先写定的判定规则】

  · B ≈ A（比值 < 1.2×）      → 掩码烧不进图，该路径作废。
  · B ≈ eager@3TPC（比值 > 1.5×）→ 掩码烧进了图，该路径可用。
  · 介于两者之间              → 部分生效，需人工判断，不据此下结论。
  · C ≈ A                     → 复现"回放时设掩码无效"，与已知一致；
                                若 C 显著慢于 A，说明已知结论在本配置下不成立，
                                必须先解决这个矛盾。

【测量顺序】三条臂**逐次轮转、每轮换起点**。固定轮内顺序会制造随位置单调的假
梯度（本项目在 Q1 边界扫描时吃过这个亏）。

用法::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    .venv/bin/python scripts/probes/diag_mask_in_graph_capture.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.libsmctrl_adapter import (  # noqa: E402
    LibSmCtrlAdapter,
    enabled_tpcs_to_native_mask,
    load_libsmctrl,
)
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

OUT_DEFAULT = "results/reproducibility/probes/diag_mask_in_graph_capture.json"


def span_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--masked-tpcs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    adapter = LibSmCtrlAdapter()
    probe = adapter.probe()
    if not probe.get("available"):
        raise SystemExit(f"libsmctrl 探针未通过，拒绝继续：{probe}")
    lib = load_libsmctrl()

    cfg = load_config("config.libsmctrl.example.yaml")
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    model = enc.model

    size = args.size
    job = EncodeJob(f"maskcap-{size}", zlib.crc32(b"maskcap") % (2 ** 31), size, size, 600_000)
    job.sm_fraction = 1.0
    with torch.inference_mode():
        pixels = enc._input(job)

    no_mask = enabled_tpcs_to_native_mask(0, 0)          # 0 个启用即"不限制"
    mask_3 = enabled_tpcs_to_native_mask(0, args.masked_tpcs)

    def set_mask(value: int) -> None:
        lib.libsmctrl_set_thread_mask(ctypes.c_uint64(value))

    def eager():
        with torch.inference_mode():
            return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

    def capture() -> tuple["torch.cuda.CUDAGraph", "torch.Tensor"]:
        """在**当前线程的粘性掩码**下捕获一张图，返回 (图, 静态输出张量)。"""
        g = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.inference_mode():
            with torch.cuda.graph(g):
                out = model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
        return g, out

    # ---- 捕获两张图 ----
    set_mask(no_mask)
    g_nomask, out_nomask = capture()          # A 基准；C 的阴性对照也用它
    set_mask(mask_3)
    g_mask_captured, out_masked = capture()   # B：掩码下捕获
    set_mask(no_mask)

    # ---- 正确性：每张图重放后的静态输出必须与 eager 逐元素一致 ----
    ref = eager()
    torch.cuda.synchronize()
    diffs = {}
    for name, (g, out) in (("nomask", (g_nomask, out_nomask)),
                           ("mask_captured", (g_mask_captured, out_masked))):
        g.replay()
        torch.cuda.synchronize()
        diffs[name] = float((out.detach().float() - ref.detach().float()).abs().max().item())
    print(f"图与 eager 的逐元素最大差：{diffs}")
    if max(diffs.values()) > 1e-2:
        raise SystemExit("图捕获结果与 eager 不一致，终止——不能拿错图做实验")

    # ---- eager 标尺 ----
    set_mask(no_mask)
    eager_free = statistics.median(span_ms(eager) for _ in range(5))
    set_mask(mask_3)
    eager_masked = statistics.median(span_ms(eager) for _ in range(5))
    set_mask(no_mask)

    # ---- 三条臂逐次轮转 ----
    arms = {
        "A_nomask_capture_nomask_replay": lambda: _replay(g_nomask, no_mask, set_mask),
        "B_mask_capture_nomask_replay": lambda: _replay(g_mask_captured, no_mask, set_mask),
        "C_nomask_capture_mask_replay": lambda: _replay(g_nomask, mask_3, set_mask),
    }
    samples: dict[str, list[float]] = {k: [] for k in arms}
    order = list(arms)
    for round_index in range(args.reps):
        rotated = order[round_index % len(order):] + order[: round_index % len(order)]
        for key in rotated:
            samples[key].append(arms[key]())

    set_mask(no_mask)

    med = {k: statistics.median(v) for k, v in samples.items()}
    report = {
        "script": "scripts/probes/diag_mask_in_graph_capture.py",
        "note": "掩码能否在捕获阶段烧进 CUDA Graph",
        "size": size,
        "masked_tpcs": args.masked_tpcs,
        "total_tpcs": probe.get("total_tpcs"),
        "reps": args.reps,
        "libsmctrl": probe,
        "eager_no_mask_ms": eager_free,
        "eager_masked_ms": eager_masked,
        "eager_mask_effect": eager_masked / eager_free,
        "arms_ms": med,
        "arms_samples": samples,
        "output_max_diff": diffs,
        "ratios": {
            "B_over_A": med["B_mask_capture_nomask_replay"] / med["A_nomask_capture_nomask_replay"],
            "C_over_A": med["C_nomask_capture_mask_replay"] / med["A_nomask_capture_nomask_replay"],
            "B_over_eager_masked": med["B_mask_capture_nomask_replay"] / eager_masked,
        },
        "criteria": [
            "B/A < 1.2  → 掩码烧不进图，该路径作废",
            "B/A > 1.5  → 掩码烧进了图，该路径可用",
            "两者之间   → 部分生效，不下结论",
            "C/A ≈ 1    → 复现回放时设掩码无效（与已知一致）",
        ],
    }
    report["verdict"] = _verdict(report)

    print(f"\n=== {size}×{size}，{args.masked_tpcs}/{probe.get('total_tpcs')} TPC ===")
    print(f"  eager 标尺：无掩码 {eager_free:8.3f} ms，{args.masked_tpcs} TPC {eager_masked:8.3f} ms "
          f"（{report['eager_mask_effect']:.2f}×）")
    for k in order:
        print(f"  {k:34s} {med[k]:8.3f} ms")
    print(f"  B/A = {report['ratios']['B_over_A']:.3f}   C/A = {report['ratios']['C_over_A']:.3f}")
    print(f"  >>> {report['verdict']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[写出] {out}")
    return 0


def _replay(graph, mask: int, set_mask) -> float:
    set_mask(mask)
    return span_ms(graph.replay)


def _verdict(report: dict) -> str:
    r = report["ratios"]
    if r["B_over_A"] < 1.2:
        base = "掩码烧不进图，该路径作废"
    elif r["B_over_A"] > 1.5:
        base = "掩码烧进了图，该路径可用"
    else:
        base = "介于两者之间，部分生效，不据此下结论"
    c = "" if abs(r["C_over_A"] - 1.0) < 0.2 else "；⚠️ C/A 偏离 1，与已知结论不一致，须先查清"
    return f"B/A = {r['B_over_A']:.3f} → {base}{c}"


if __name__ == "__main__":
    raise SystemExit(main())
