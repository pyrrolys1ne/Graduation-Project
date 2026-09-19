"""Bullet 式「算术强度饱和组批」的本机检验。

出处
----
Bullet: Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing（ASPLOS 2026）§2.3。

它的组批终止条件是**算术强度达到峰值**，而不是批大小达到上限（Algorithm 1 line 12）：

    while ARITHINTEN(next_tasks, S.ES) < peak do
        next_tasks.append(Q.pop())

即：**批增长的停止条件是 AI 拐点，不是人为设的 `max_batch`。**

为什么本课题该试这个
--------------------
本课题已实测（`docs/实验记录.md` §7）：

  - 批处理**只在过载下吞吐更高**（+15.4%），欠饱和下打平（瓶颈是到达率）
  - **零等待的并发流在尾延迟上始终占优**（P99 差 44%）
  - **「愿意等待」在任何负载下都是纯损失**——"需要等待才能组批"的时刻，
    恰恰是"不值得组批"的时刻
  - **批不是越大越好**：`batch8_d5` 与 `batch4_d5` 几乎相同，瓶颈在等待而非批大小

最后一条正指向 Bullet 的动机：**固定 `max_batch` 是错的旋钮**。
若改用 AI 拐点作终止条件，就能**在不引入额外等待的前提下**让批长到它该长的大小。

本机版本
--------
CLIP 的算术强度不随批线性变化——同尺寸拼批时 M 维增长、K/N 不变，
FLOPs 线性增长而访存也线性增长，**AI 基本不变**。
真正让 AI 变化的是**批内混合不同尺寸**：

    小图（224，访存受限，AI 低） + 大图（672，计算受限，AI 高）
    → 混合批的 AI 介于两者之间，**可能更接近峰值**

这正是"算术强度饱和式组批"在本机有意义的形态：**用混合尺寸批把 AI 拉到拐点**。

判据
----
- **通过**：AI 饱和式组批在同等吞吐下 P99 优于固定 `max_batch`，
  **或**在同等 P99 下吞吐更高
- **死亡**：打平或更差 → CLIP 的 AI 拐点不显著（很可能因为**6 TPC 就带宽饱和**，
  已经过了拐点，再加批只增延迟）

本 demo 是**离线计算 + 已有数据的再分析**，不跑 GPU：
先用标定表算各尺寸的 AI，再判断"混合批的 AI 是否真的更接近峰值"。
**若这一步就不成立，后面的 GPU 实验不必做。**

用法
----
    python demos/demo_ai_saturating_batch.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.performance import PerformanceModel

#: CLIP ViT-B/32 的结构常数
HIDDEN, LAYERS, HEADS = 768, 12, 12
INTERMEDIATE = 3072
PATCH = 32


def flops_for(size: int, batch: int = 1) -> float:
    """一次前向的浮点运算量（只算 GEMM/attention，忽略 LayerNorm 等）。

    ViT-B/32 单个 block：
      QKV 投影  2·M·768·2304
      attention 2·M·M·768（这里取 M=seq）
      输出投影  2·M·768·768
      MLP       2·M·768·3072 + 2·M·3072·768
    """
    seq = (size // PATCH) ** 2 + 1
    m = seq * batch
    per_block = (
        2 * m * HIDDEN * 3 * HIDDEN
        + 2 * m * m * HIDDEN
        + 2 * m * HIDDEN * HIDDEN
        + 2 * m * HIDDEN * INTERMEDIATE
        + 2 * m * INTERMEDIATE * HIDDEN
    )
    return LAYERS * per_block


def bytes_for(size: int, batch: int = 1, dtype_bytes: int = 2) -> float:
    """访存量估算：输入图 + 每层激活的读写。

    粗粒度估算——本 demo 只关心**AI 随尺寸/批的变化方向**，不追求绝对精度。
    """
    seq = (size // PATCH) ** 2 + 1
    pixels = 3 * size * size * batch
    # 每层激活：读入 + 写出（含中间 MLP 扩展）
    act = batch * seq * (HIDDEN + INTERMEDIATE) * 2
    return (pixels + LAYERS * act) * dtype_bytes


def arithmetic_intensity(size: int, batch: int = 1) -> float:
    return flops_for(size, batch) / max(bytes_for(size, batch), 1.0)


def mixed_ai(sizes: list[int]) -> float:
    """混合尺寸批的算术强度（按总量加权）。"""
    f = sum(flops_for(s, 1) for s in sizes)
    b = sum(bytes_for(s, 1) for s in sizes)
    return f / max(b, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 饱和式组批的离线可行性检验")
    parser.add_argument("--table", type=Path, default=Path("data/profiles/default.csv"))
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[224, 288, 336, 384, 448, 512, 672])
    parser.add_argument("--output", type=Path, default=Path("results/demo_ai_batch.json"))
    args = parser.parse_args()

    model = PerformanceModel(args.table)

    print(f"{'='*96}")
    print("各尺寸的算术强度（AI = FLOPs / Bytes）")
    print(f"{'='*96}")
    print(f"{'尺寸':>6} {'seq':>5} {'FLOPs(M)':>10} {'Bytes(MB)':>11} {'AI':>9} "
          f"{'满配额延迟(ms)':>15} {'受限类型':>10}")
    print("-" * 96)
    ai_by_size = {}
    lat_by_size = {}
    for size in args.sizes:
        seq = (size // PATCH) ** 2 + 1
        f = flops_for(size)
        b = bytes_for(size)
        ai = f / b
        lat = model.predict((size // PATCH) ** 2, 1.0)
        # RTX 4060 Laptop 的机器平衡点估算：约 15 TFLOP/s FP16 ÷ 256 GB/s ≈ 58 FLOP/byte
        kind = "计算受限" if ai > 58 else "访存受限"
        ai_by_size[size] = ai
        lat_by_size[size] = lat
        print(f"{size:>6} {seq:>5} {f/1e6:>10.1f} {b/2**20:>11.2f} {ai:>9.1f} "
              f"{lat:>15.3f} {kind:>10}")
    print()
    print("  （机器平衡点 ≈ 58 FLOP/byte，按 4060 Laptop 约 15 TFLOP/s FP16 ÷ 256 GB/s 估）")

    print(f"\n{'='*96}")
    print("混合尺寸批能否把 AI 拉近平衡点？")
    print(f"{'='*96}")
    print(f"{'组合':>26} {'混合AI':>9} {'延迟(ms)':>10} {'AI 是否更接近 58':>18}")
    print("-" * 96)
    combos = [
        [224], [672], [224, 672], [224, 224, 672], [224, 224, 224, 672],
        [336, 448], [224, 448, 672], [224, 288, 336, 384, 448, 512, 672],
    ]
    balance = 58.0
    rows = []
    for combo in combos:
        ai = mixed_ai(combo)
        # 延迟：同尺寸批用标定表查，混合批取各成员之和（近似，因为要分桶执行）
        lat = sum(lat_by_size.get(s, model.predict((s // PATCH) ** 2, 1.0)) for s in combo)
        single_best = min(abs(mixed_ai([s]) - balance) for s in args.sizes)
        closer = abs(ai - balance) < single_best
        rows.append({"combo": combo, "ai": ai, "latency_ms": lat, "closer_to_balance": closer})
        label = "+".join(str(s) for s in combo)
        print(f"{label:>26} {ai:>9.1f} {lat:>10.2f} {'✅ 是' if closer else '否':>18}")

    print(f"\n{'='*96}")
    print("判据")
    print(f"{'='*96}")
    gain = [r for r in rows if r["closer_to_balance"] and len(r["combo"]) > 1]
    if gain:
        print(f"  ✅ 通过：{len(gain)} 个混合组合的 AI 比任何单一尺寸都更接近平衡点")
        print(f"     例：{'+'.join(str(s) for s in gain[0]['combo'])} → AI {gain[0]['ai']:.1f}")
        print("     → **混合尺寸批值得做 GPU 实验验证**")
    else:
        print("  ❌ 判死：没有任何混合组合比单一尺寸更接近平衡点")
        print("     → CLIP 的 AI 随尺寸变化有限，AI 饱和式组批在本机无意义")

    print(f"\n  ⚠️ 本估算有明确局限，下结论前必须注意：")
    print(f"     ① FLOPs/Bytes 是**粗粒度估算**（忽略 LayerNorm、激活函数、attention 的真实实现）")
    print(f"     ② 机器平衡点 58 FLOP/byte 是**理论估算**，真实值取决于 cuBLAS 实现与显存效率")
    print(f"     ③ **最关键**：本课题实测『6 TPC 就吃满显存带宽』——")
    print(f"        若带宽早已饱和，则 AI 再高也换不来吞吐，加批只增延迟")
    print(f"     → 因此本 demo 只决定『**值不值得做 GPU 实验**』，不能替代实验。")

    print(f"\n  与既有实测的关系（`docs/实验记录.md` §7）：")
    print(f"     『批不是越大越好』『愿意等待是纯损失』——")
    print(f"     若 AI 饱和式组批能**在不引入等待的前提下**改变批大小，它才有价值；")
    print(f"     若它仍需要等组批，则已被实测否定。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"ai_by_size": ai_by_size, "lat_by_size": lat_by_size,
         "balance_flop_per_byte": balance, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
