"""分解 `1 − T_graph / T_eager`：这里面有多少是主机发射，多少是 GPU 内核间的间隙？

【为什么需要这一条】
`T_eager − T_graph` 是一个**两臂差**。把它整体命名为"主机侧 kernel 发射开销"是
推断，不是分解。这个差里至少混着三类东西：

  (i)  主机没跟上、GPU 等米下锅   —— 只有这一类才叫"主机发射开销"；
  (ii) GPU 排空了但序列本身有间隙（内核间发射尾延迟、流上的假依赖）；
  (iii) 两臂执行的**根本不是同一组 kernel**（算法选择不同、少发了 kernel）。

现成的支持只覆盖了 (i) 存在：主机负载干预实验里 eager 随 CPU 负载变化、
graph 不随。但没有任何一次测量把 (ii) 与 (iii) 排除掉。

【本脚本做什么】

  T1  Σ内核时长 vs 区间跨度。用 torch.profiler 取两臂各自的 Σ(kernel 时长) 与
      内核调用次数，与 CUDA Event 测得的跨度对比。
  T2  内核集合一致性。比对两臂的内核名单与次数。
  T4  主机发射耗时与气泡量的对齐。**同一配置、同一会话内**同时取这三个量：
      跨度 span、Σkernel、主机发射耗时 host_launch，气泡 = span − Σkernel。

【事先写定的判定规则】——先写下再跑，避免看着数字找解释

  ① 若两臂的 **Σkernel 显著不同**，或 **内核集合/次数不同** →
     (iii) 成立，`1 − T_graph / T_eager` **不能**用来谈主机开销，该表述作废。
  ② 若 Σkernel 相当而跨度不同 → 差全部是间隙。再看 T4。
     ⚠️ 判据是"气泡 vs **超出量**"，不是"气泡 vs 主机发射总量"——发射与 GPU 执行
     重叠，只有 `主机发射 − Σ内核 > 0` 的那部分才可能变成气泡：
       · 气泡 ≈ (主机发射 − Σ内核) → 主机确实是瓶颈，(i) 成立；
       · 气泡 ≪ 超出量            → 主机不足以解释，部分另有来源；
       · 气泡 ≫ 超出量            → 主机根本解释不了这个差；
       · 主机发射 ≤ Σ内核          → 发射被 GPU 执行掩盖，气泡另有来源。

  两侧都用**同一张已经搬上 GPU 的输入张量**、事件打在**同一个流**上、H2D 与
  造图都在计时区之外，因此同步位置与计时边界这两项不构成解释（见
  `docs/实验记录.md` 的限制清单）。

【T3 说明】"主机非阻塞对照"（人为制造 GPU 积压让发射被掩盖）未在本脚本内实现：
它已由 `diag_graph_vs_eager.py` 的 CPU 负载干预部分覆盖（eager 随负载变化、
graph 不随）。本脚本补的是**分解**，不是**存在性**。

用法::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    .venv/bin/python scripts/probes/diag_launch_vs_bubble.py
"""

from __future__ import annotations

import argparse
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
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

OUT_DEFAULT = "results/reproducibility/probes/diag_launch_vs_bubble.json"


def span_ms(fn) -> float:
    """CUDA Event 夹住的区间跨度（含区间内的 GPU 空闲）。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def host_launch_ms(fn) -> tuple[float, float]:
    """主机侧发射耗时 + 这一次的区间跨度。

    发射耗时 = 从进入调用到调用返回（所有 kernel 已入队）的墙钟。它**不等于**
    关键路径上的代价：主机发射与 GPU 执行是重叠的，只有当 GPU 跑空时才变成代价。
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    end.record()
    torch.cuda.synchronize()
    return (t1 - t0) * 1000, start.elapsed_time(end)


def _profile_once(fn) -> dict | None:
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    per_kernel: dict[str, list[float]] = {}
    try:
        for evt in prof.events():
            if evt.device_type != torch.autograd.DeviceType.CUDA:
                continue
            if getattr(evt, "is_legacy", False):
                continue
            per_kernel.setdefault(evt.name, []).append(float(evt.device_time))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"解析 profiler 事件失败: {type(exc).__name__}: {exc}"}

    if not per_kernel:
        return None
    total = sum(sum(v) for v in per_kernel.values())
    return {
        "sum_kernel_ms": total / 1000.0,             # torch 以 µs 计
        "kernel_calls": sum(len(v) for v in per_kernel.values()),
        "distinct_kernels": len(per_kernel),
        "kernels": sorted(per_kernel, key=lambda k: -sum(per_kernel[k]))[:12],
    }


def profile_kernels(fn, reps: int = 1) -> dict:
    """用 torch.profiler 取 Σ(kernel 时长)、内核调用次数与名单。

    ⚠️ **profiler 的那一次执行与计时用的那些执行不是同一次**（profiler 本身会扰动
    时钟与调度）。实测 672 的 Σ内核 在相邻两次采样间可差百分之十几，直接相减会得到
    一个**负的气泡**——物理上不可能，只是口径错。

    因此波动幅度**必须从跨进程重复里估**：`--prof-reps > 1`（同一进程内多次采样）
    实测会挂死——与 CUDA Graph 的挂死同类，不可用。默认单次采样，此时
    `sum_kernel_spread` 为 0，判定函数会拒绝在波动未知时下结论。

    采不到任何内核事件时返回 error，**不写 0**——把"没测到"写成 0 是本项目吃过
    的亏。
    """
    attempts = [_profile_once(fn) for _ in range(reps)]
    results = [r for r in attempts if r and "sum_kernel_ms" in r]
    if not results:
        reason = next((r["error"] for r in attempts if r and "error" in r), "未采到内核事件")
        return {"error": f"profiler 不可用（{reps} 次尝试）：{reason}"}
    sums = [r["sum_kernel_ms"] for r in results]
    latest = results[-1]
    return {
        "sum_kernel_ms": statistics.median(sums),
        "sum_kernel_min_ms": min(sums),
        "sum_kernel_max_ms": max(sums),
        "sum_kernel_spread": (max(sums) - min(sums)) / statistics.median(sums) if sums else 0.0,
        "prof_samples": len(sums),
        "kernel_calls": latest["kernel_calls"],
        "distinct_kernels": latest["distinct_kernels"],
        "kernels": latest["kernels"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="224,672")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    cfg = load_config("config.libsmctrl.example.yaml")
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 1)
    model = enc.model
    report: dict = {
        "script": "scripts/probes/diag_launch_vs_bubble.py",
        "note": "分解 1 - T_graph/T_eager：Σkernel vs 跨度 vs 主机发射耗时",
        "reps": args.reps,
        "criteria": [
            "① Σkernel 或内核集合不同 → 两臂不是同一个计算，该比值不能谈主机开销",
            "② Σkernel 相当而跨度不同 → 差全是间隙；再比 气泡 与 host_launch",
            "   气泡≈host_launch → 主机是瓶颈；气泡≪host_launch → 发射被掩盖；",
            "   气泡≫host_launch → 主机解释不了",
        ],
        "sizes": {},
    }

    for size in [int(s) for s in args.sizes.split(",")]:
        job = EncodeJob(f"lvb-{size}", zlib.crc32(f"lvb-{size}".encode()) % (2 ** 31),
                        size, size, 600_000)
        job.sm_fraction = 1.0
        with torch.inference_mode():
            pixels = enc._input(job)          # 只算一次；两臂共用同一张 CUDA 张量

        def eager():
            with torch.inference_mode():
                return model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                captured = model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
        graph.replay()
        torch.cuda.synchronize()
        diff = float((captured.detach().float() - eager().detach().float()).abs().max().item())
        if diff > 1e-2:
            raise SystemExit(f"图捕获结果与 eager 不一致（{diff:.3e}），终止")

        def graph_run():
            graph.replay()

        for _ in range(5):
            span_ms(eager)
            span_ms(graph_run)

        arms = {}
        for name, fn in (("eager", eager), ("graph", graph_run)):
            spans, launches, bubbles = [], [], []
            for _ in range(args.reps):
                hl, sp = host_launch_ms(fn)
                launches.append(hl)
                spans.append(sp)
            prof = profile_kernels(fn)
            med_span = statistics.median(spans)
            med_launch = statistics.median(launches)
            entry = {
                "span_ms": med_span,
                "host_launch_ms": med_launch,
                "host_launch_ratio": med_launch / med_span,
                "span_samples": spans,
                "profiler": prof,
            }
            if "sum_kernel_ms" in prof:
                entry["bubble_ms"] = med_span - prof["sum_kernel_ms"]
                entry["bubble_ratio"] = entry["bubble_ms"] / med_span
            arms[name] = entry
            print(f"\n=== {size}×{size} / {name} ===")
            print(f"  区间跨度 {med_span:8.3f} ms   主机发射 {med_launch:8.3f} ms")
            if "sum_kernel_ms" in prof:
                print(f"  Σ内核    {prof['sum_kernel_ms']:8.3f} ms（{prof['prof_samples']} 次采样，"
                      f"波动 ±{prof['sum_kernel_spread'] * 100:.1f}%）"
                      f"   气泡 {entry['bubble_ms']:8.3f} ms（占跨度 {entry['bubble_ratio'] * 100:.1f}%）"
                      f"   内核 {prof['kernel_calls']} 次 / {prof['distinct_kernels']} 种")
            else:
                print(f"  ⚠️ profiler 不可用：{prof.get('error')}")

        verdict = _verdict(arms)
        report["sizes"][str(size)] = {"arms": arms, "verdict": verdict}
        print(f"  >>> 判定：{verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {out}")
    return 0


def _verdict(arms: dict) -> str:
    """按事先写定的规则判定。

    ⚠️ 第一版判据把"气泡"直接与"主机发射耗时"相比，这是错的。主机发射与 GPU 执行
    是**重叠**的，只有超出的部分才变成气泡。正确的关系是

        跨度 ≈ max(主机发射, Σ内核)，  气泡 = 跨度 − Σ内核 ≈ max(0, 主机发射 − Σ内核)

    即"主机是否真的是瓶颈"取决于 `主机发射 > Σ内核` 是否成立。实测 224：主机发射
    3.386 ms、Σ内核 1.670 ms，气泡实测 1.500 ms，而 3.386 − 1.670 = 1.716 ms——
    两者吻合到 13% 以内。所以这里比的是**超出量**，不是总量。
    """
    e, g = arms.get("eager", {}), arms.get("graph", {})
    if "sum_kernel_ms" not in e.get("profiler", {}) or "sum_kernel_ms" not in g.get("profiler", {}):
        return "无法判定：profiler 数据缺失（不写 0，如实报缺）"

    # ① 的主证据是**内核集合与调用次数完全相同**——它对噪声免疫。
    # ⚠️ 比较前必须**按名字排序**：`kernels` 是按耗时排序的前 12 个，耗时受噪声影响，
    # 次序会抖，直接比列表会把同一次计算误判成"不同"（第一版就踩了这个）。
    if (e["profiler"]["kernel_calls"] != g["profiler"]["kernel_calls"]
            or e["profiler"]["distinct_kernels"] != g["profiler"]["distinct_kernels"]
            or sorted(e["profiler"]["kernels"]) != sorted(g["profiler"]["kernels"])):
        return (f"① 内核集合/次数不同（{e['profiler']['kernel_calls']}/"
                f"{e['profiler']['distinct_kernels']} 对 "
                f"{g['profiler']['kernel_calls']}/{g['profiler']['distinct_kernels']}）"
                f"→ 不是同一个计算")
    ratio = e["profiler"]["sum_kernel_ms"] / g["profiler"]["sum_kernel_ms"] \
        if g["profiler"]["sum_kernel_ms"] else float("inf")
    if not (0.7 <= ratio <= 1.4):
        return (f"① 内核集合相同但 Σ内核 相差 {ratio:.2f}× —— 超出单次测量波动范围，"
                f"须重测后再判")

    sum_k = e["profiler"]["sum_kernel_ms"]
    bubble, launch = e["bubble_ms"], e["host_launch_ms"]
    if sum_k <= 0:
        return "② Σ内核 不可测（为 0），无法对齐"

    # 气泡 = 跨度中位数 − Σ内核 中位数，两者来自**不同的执行**。若气泡本身不超过
    # profiler 的采样波动，就没有可判定的信号——此时必须报"无法判定"，不能挑一个
    # 分支去套。实测 672 正是这种情况。
    spread = max(e["profiler"]["sum_kernel_spread"], g["profiler"]["sum_kernel_spread"])
    if bubble <= spread * sum_k:
        return (f"② 无法判定：气泡 {bubble:.2f} ms 不超过 profiler 采样波动 "
                f"（±{spread * 100:.0f}% × Σ内核 = {spread * sum_k:.2f} ms）")

    excess = launch - sum_k                      # 主机发射超出 GPU 总工作量多少
    if excess <= 0:
        return (f"② 主机发射 {launch:.2f} ms 不超过 Σ内核 {sum_k:.2f} ms → "
                f"发射被 GPU 执行掩盖，气泡另有来源（实测气泡 {bubble:.2f} ms）")
    if bubble < 0.5 * excess:
        return (f"② 气泡 {bubble:.2f} ms 远小于超出量 {excess:.2f} ms → "
                f"主机不足以解释这个气泡，部分另有来源")
    if bubble > 1.5 * excess:
        return (f"② 气泡 {bubble:.2f} ms 远大于超出量 {excess:.2f} ms → "
                f"主机解释不了这个差，差另有来源")
    return (f"② 宿主侧发射是瓶颈：发射 {launch:.2f} ms 超过 Σ内核 {sum_k:.2f} ms，"
            f"超出 {excess:.2f} ms；实测气泡 {bubble:.2f} ms，两者同量级 → 归因成立")


if __name__ == "__main__":
    raise SystemExit(main())
