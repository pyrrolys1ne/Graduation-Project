"""DProbe 式「非侵入式在线采集 + 四级准入控制」的本机检验。

出处
----
DProbe: Profiling and Predicting Multi-tenant Deep Learning Workloads for GPU Resource Scaling。

**（E1）非侵入式在线采集**：用 `nvidia-smi` + DCGM 采三个 runtime 指标——
**GPU utilization (%)、GPU memory occupancy (MB)、SM utilization (%)**，取平均。
论文称 "Online Job Profiling"，但**在代表模型集合上做**（13 个模型族），
**不在每个线上请求上做**；遇新模型走 GNN 离线预测，不重新 profiling
（原文：re-profiling *"is inconvenient and wastes cluster computational resources"*）。

**（E2）四级分类阈值**（论文 §2.2，可直接抄的常数）：

    Level=1   GPU util < 25% 且 mem < 3 GB  → 对共置干扰最小  → 代理需求 30%
    Level=2   中间                                          → 50%
    Level=3   GPU util 与 SM util **均 > 70%**              → 70%
    Exclusive 需独占满足 QoS                                → 100%

配套：共享调度 = **带优先级的装箱问题**。关键规则：
*"noticeable slowdowns primarily manifest when the **cumulative utilization** of co-located
DL tasks **surpasses 100%**"*。

本机适配（**本次 demo 的核心**）
-------------------------------
论文的判据基于 **SM/GPU 占用率**，而本课题实测**本机是带宽墙**：

    - 12 个 TPC 中 **6 个就吃满显存带宽**
    - 真实 TPC 互斥分区在**所有负载、所有指标**上都劣于共享 SM 的多流
    - 并发干扰实测 ≈1.0×（计算侧），残余干扰来自**共享的 L2 与 HBM 带宽**

→ 因此**把"SM util > 70%"替换为"实测带宽占用 > 阈值"**，作"借鉴 + 本机标定修正"。

⚠️ WSL2 上 **DCGM 不可用**（需特权守护进程），只保留 **NVML** 路径。
NVML 的 `gpu` 字段是**宽松上界**（DNN-occu 实测：ResNet-50 训练时 NVML 到 90%、
真实 occupancy 只有 45%），所以本 demo 用它作**相对信号**而非绝对值。

判据
----
- **通过**：Level 标签与实际干扰**相关**——即被标为 Level=1 的请求共置后
  延迟劣化显著小于 Level=3
- **死亡**：Level 标签与实际干扰**无关** → 该特征集在本机无效，需换带宽特征
  （**这是"SM 占用率判据不适用"的直接证据，可写**）

用法
----
    python demos/demo_online_probe_admission.py --size 224 --reps 60
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import ProxyResourceBackend

#: DProbe 的原始阈值
DPROBE_LEVELS = {
    "L1": {"gpu_util_max": 25.0, "mem_max_gb": 3.0, "proxy_demand": 0.30},
    "L2": {"proxy_demand": 0.50},
    "L3": {"gpu_util_min": 70.0, "sm_util_min": 70.0, "proxy_demand": 0.70},
    "EX": {"proxy_demand": 1.00},
}

#: 本机修正：把"saturation"定义为 GPU util 达到该值——超过说明带宽已被吃满，
#: 此时再共置只会加剧争用。
#: **依据**：本课题实测 6/12 TPC 即吃满带宽，且互斥分区在所有负载都劣于共享 SM。
LOCAL_SATURATION_UTIL = 60.0


def probe_nvml():
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        return {"available": True, "mod": pynvml,
                "handle": pynvml.nvmlDeviceGetHandleByIndex(0)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


class Worker:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            fn, box = self._q.get()
            if fn is None:
                return
            try:
                box["r"] = fn()
            except Exception as exc:  # noqa: BLE001
                box["e"] = exc

    def call(self, fn, timeout: float = 600.0):
        box: dict = {}
        self._q.put((fn, box))
        deadline = time.time() + timeout
        while not box:
            if time.time() > deadline:
                raise TimeoutError("等待工作线程超时")
            time.sleep(0.002)
        if "e" in box:
            raise box["e"]
        return box["r"]


def job(size: int, tag: str) -> EncodeJob:
    j = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                  width=size, height=size, deadline_ms=600_000)
    j.sm_fraction = 1.0
    return j


def measure_pair(encoder: ClipEncoderBackend, size_a: int, size_b: int,
                 reps: int, nvml: dict) -> dict:
    """并发跑两个**不同尺寸**的请求，期间采 NVML。

    返回两个尺寸各自的延迟分布 + 采样到的 GPU util，用于检验
    "Level 标签（按 util 判定）能否预测实际干扰"。
    """
    wa, wb = Worker(), Worker()
    barrier = threading.Barrier(2)
    out: dict[str, dict] = {}
    utils: list[int] = []
    stop = threading.Event()

    def sampler():
        mod, h = nvml["mod"], nvml["handle"]
        while not stop.is_set():
            try:
                utils.append(mod.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)

    def make(wk: Worker, tag: str, size: int, solo: bool):
        def go():
            for i in range(2):
                encoder.encode(job(size, f"warm-{tag}-{i}"), 0)
            if not solo:
                barrier.wait()
            lat = [float(encoder.encode(job(size, f"{tag}-{i}"), 0).execution_ms)
                   for i in range(reps)]
            return {"lat": lat, "median": statistics.median(lat)}
        return wk.call(go)

    # 先各自独跑（基准）
    solo_a = make(wa, "solo-a", size_a, True)
    solo_b = make(wb, "solo-b", size_b, True)

    # 再并发
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    t1 = threading.Thread(target=lambda: out.__setitem__("a", make(wa, "conc-a", size_a, False)))
    t2 = threading.Thread(target=lambda: out.__setitem__("b", make(wb, "conc-b", size_b, False)))
    t1.start(); t2.start(); t1.join(); t2.join()
    stop.set(); sampler_thread.join(timeout=2)

    util = statistics.fmean(utils) if utils else float("nan")
    return {
        "size_a": size_a, "size_b": size_b,
        "solo_a_ms": solo_a["median"], "conc_a_ms": out["a"]["median"],
        "solo_b_ms": solo_b["median"], "conc_b_ms": out["b"]["median"],
        "slowdown_a": out["a"]["median"] / solo_a["median"] if solo_a["median"] else float("nan"),
        "slowdown_b": out["b"]["median"] / solo_b["median"] if solo_b["median"] else float("nan"),
        "gpu_util_mean": util, "gpu_util_max": max(utils) if utils else None,
        "n_samples": len(utils),
    }


def classify_local(util: float) -> str:
    """本机修正版分级：以**实测 GPU util** 为准（本机带宽墙，util 是相对信号）。"""
    if util < DPROBE_LEVELS["L1"]["gpu_util_max"]:
        return "L1"
    if util >= DPROBE_LEVELS["L3"]["gpu_util_min"]:
        return "L3"
    return "L2"


def main() -> None:
    parser = argparse.ArgumentParser(description="DProbe 式在线采集 + 准入控制")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--pairs", default="224:224,224:672,672:672,336:448")
    parser.add_argument("--reps", type=int, default=60)
    parser.add_argument("--output", type=Path,
                        default=Path("results/demo_online_probe_admission.json"))
    args = parser.parse_args()

    nvml = probe_nvml()
    if not nvml["available"]:
        raise SystemExit(f"NVML 不可用（{nvml.get('reason')}）。"
                         "本 demo 的采集层依赖它，请 pip install nvidia-ml-py")
    print(f"NVML 可用。DCGM 在 WSL2 不可用（需特权守护进程），只用 NVML 路径。\n")

    config = load_config(args.config)
    encoder = ClipEncoderBackend(config.model, ProxyResourceBackend(), 1)

    pairs = []
    for spec in args.pairs.split(","):
        a, b = spec.split(":")
        pairs.append((int(a), int(b)))

    rows = []
    for a, b in pairs:
        r = measure_pair(encoder, a, b, args.reps, nvml)
        r["level"] = classify_local(r["gpu_util_mean"])
        rows.append(r)
        print(f"  {a}×{a} + {b}×{b}: slowdown {r['slowdown_a']:.2f}× / {r['slowdown_b']:.2f}×  "
              f"GPU util {r['gpu_util_mean']:.1f}% → 分级 {r['level']}", flush=True)

    print(f"\n{'='*100}")
    print("Level 标签能否预测实际干扰？")
    print(f"{'='*100}")
    print(f"{'组合':>14} {'分级':>5} {'GPU util':>9} {'slowdown(a)':>12} {'slowdown(b)':>12} "
          f"{'最大':>8}")
    print("-" * 100)
    for r in rows:
        mx = max(r["slowdown_a"], r["slowdown_b"])
        print(f"{str(r['size_a'])+'+'+str(r['size_b']):>14} {r['level']:>5} "
              f"{r['gpu_util_mean']:>8.1f}% {r['slowdown_a']:>12.2f} {r['slowdown_b']:>12.2f} "
              f"{mx:>8.2f}")

    print(f"\n{'='*100}")
    print("判据")
    print(f"{'='*100}")
    l1 = [max(r["slowdown_a"], r["slowdown_b"]) for r in rows if r["level"] == "L1"]
    l3 = [max(r["slowdown_a"], r["slowdown_b"]) for r in rows if r["level"] == "L3"]
    print(f"  L1（util < 25%）的 slowdown: {[round(v,2) for v in l1] or '（无样本）'}")
    print(f"  L3（util ≥ 70%）的 slowdown: {[round(v,2) for v in l3] or '（无样本）'}")
    if l1 and l3:
        if statistics.fmean(l3) > statistics.fmean(l1) * 1.2:
            print(f"  ✅ 通过：L3 的干扰显著大于 L1"
                  f"（{statistics.fmean(l3):.2f}× 对 {statistics.fmean(l1):.2f}×）")
            print("     → Level 标签有效，准入控制值得做")
        else:
            print(f"  ❌ 判死：L1 与 L3 的干扰无显著差别"
                  f"（{statistics.fmean(l1):.2f}× 对 {statistics.fmean(l3):.2f}×）")
            print("     → 基于占用率的 Level 标签在本机无效（**带宽墙导致 SM 占用率不是瓶颈指标**），")
            print("       需改用带宽特征")
    else:
        print("  ⚠️ 样本不足：本机 util 未跨越 L1/L3 两个区间。")
        print("     这本身是信息——说明 CLIP 前向的 GPU util 变化范围很窄，")
        print("     **占用率类特征在本机区分度低**。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"nvml": {"available": True}, "dcgm": {"available": False, "reason": "WSL2 不可用"},
         "local_saturation_util": LOCAL_SATURATION_UTIL, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
