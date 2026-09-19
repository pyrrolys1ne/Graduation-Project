"""波量化（wave quantization）的解析估算 —— 对 CLIP ViT-B/32 的每一个 GEMM/attention 算子。

出处
----
Bullet: Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing（ASPLOS 2026）§2.2.1。

原文给出的解析式——给定 kernel 有 g 个线程块、N 个 SM、每 SM 能驻留 b 个 TB：

    waves  w = ⌈ g / (b·N) ⌉
    tail   = ⌈ g/b − N·(w−1) ⌉
    SM 空闲比  idle = (N − tail) / (N · w)

直觉：线程块数不是"SM 数 × 每 SM 槽位"的整数倍时，最后一波只有部分 SM 有活干，
其余空转到结束。

原文的关键判词（直接指向本课题）
--------------------------------
*"This inefficiency is particularly pronounced in **Transformer's self-attention** and
**small-shaped GEMMs for short input sequences or small chunked prefill-sizes**."*

**CLIP ViT-B/32 正是"短序列 + 小 GEMM"**：hidden=768、12 层、12 头（head_dim=64）、
intermediate=3072、patch 7×7。224×224 时 seq_len=50，672×672 时 seq_len=442。

为什么这个 demo 重要
--------------------
本课题此前把"小请求不稳"归因于 **host 发射开销**（已用 CUDA Graph 干预证实：host 占比
59%→7%）。波量化给的是**另一个独立的、纯 GPU 侧的机制**，而且**完全不需要测量**——
只要有 kernel 的线程块数就能算。

若它成立，调度器就能**解析地预测**"这个尺寸该不该提高并发度去填空转的 SM"。
若不成立，说明 cuBLAS/TileLang 的内部 swizzle 已把波量化抹平——**那也是可写的结论**。

通过判据
--------
- 解析算出的 `idle` 与实测延迟**单调相关**（尺寸越大 idle 越小、延迟越高）→ 波量化可观
- 反向算 `idle` 与实测**不相关** → CLIP 的 kernel 已被内部调度抹平（负结果）

本节只做**解析计算**，与实测的对照需要另跑一段 GPU 代码（见 `--emit-csv`）。

用法
----
    python demos/demo_wave_quant.py
    python demos/demo_wave_quant.py --emit-csv /tmp/wave.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# CLIP ViT-B/32 的结构参数（取自 HuggingFace config）
# --------------------------------------------------------------------------- #
HIDDEN = 768
LAYERS = 12
HEADS = 12
HEAD_DIM = HIDDEN // HEADS          # 64
INTERMEDIATE = 3072
PATCH = 32

#: 本机 SM 数。RTX 4060 Laptop = 24 SM（12 TPC × 2）
SM_COUNT = 24

#: cuBLAS 类 kernel 常用的 tile 尺寸。**这是估算里最大的不确定项**，
#: 真实 tile 由 cublasLt 在运行时挑选。因此本脚本给的是**量级**而非精确值，
#: 并在输出里显式标注敏感度。
TILE_M = 128
TILE_N = 128
TILE_K = 32

#: 每 SM 能同时驻留的线程块数。同样依赖寄存器和 shared memory 占用，
#: 这里给一个保守值 2（cuBLAS 的大 tile kernel 常为 1–2）。
TB_PER_SM = 2

#: 单个线程块的线程数（用于 attention kernel 的估算）
ATTN_THREADS = 128


def waves(blocks: int, sm: int = SM_COUNT, tb_per_sm: int = TB_PER_SM) -> tuple[int, int, float]:
    """按 Bullet 的解析式算 (waves, tail, idle_ratio)。"""
    if blocks <= 0:
        return 0, 0, 0.0
    slots = tb_per_sm * sm
    w = math.ceil(blocks / slots)
    tail = math.ceil(blocks / tb_per_sm - sm * (w - 1))
    idle = (sm - tail) / (sm * w) if w > 0 else 0.0
    return w, tail, idle


def gemm_blocks(m: int, n: int, k: int, tm: int = TILE_M, tn: int = TILE_N) -> int:
    """GEMM [m,k]×[k,n] 的线程块数：按 (m/tm) × (n/tn) 的 tile 网格。"""
    return math.ceil(m / tm) * math.ceil(n / tn)


def attention_blocks(seq: int, heads: int = HEADS, threads: int = ATTN_THREADS) -> int:
    """Attention 的线程块数：每个 (head, query-block) 一个块。

    FlashAttention 类的实现按查询维分块，每块处理 `threads` 个 query。
    """
    return heads * math.ceil(seq / threads)


def fmt_ops(patches: int) -> list[tuple[str, int, int, int]]:
    """返回该分辨率下 CLIP 单个 transformer block 的主要算子 [(名字, M, N, K)]。

    一个 block 的主要 GEMM：
      - QKV 投影   [seq,768] × [768,2304]
      - attention  [heads, seq,64] × [heads,64,seq]（视为 batched GEMM，另有专门估算）
      - 输出投影   [seq,768] × [768,768]
      - MLP fc1    [seq,768] × [768,3072]
      - MLP fc2    [seq,3072] × [3072,768]
    """
    seq = patches + 1                                   # +1 是 cls token
    return [
        ("QKV proj", seq, 3 * HIDDEN, HIDDEN),
        ("out proj", seq, HIDDEN, HIDDEN),
        ("MLP fc1", seq, INTERMEDIATE, HIDDEN),
        ("MLP fc2", seq, HIDDEN, INTERMEDIATE),
    ]


def analyze(sizes: list[int], batch: int = 1, tb_per_sm: int = TB_PER_SM) -> list[dict]:
    rows: list[dict] = []
    for size in sizes:
        patches = (size // PATCH) ** 2
        seq = patches + 1
        # 批处理：M 维乘 batch（同尺寸图拼批）
        for name, m, n, k in fmt_ops(patches):
            g = gemm_blocks(m * batch, n, k)
            w, tail, idle = waves(g, SM_COUNT, tb_per_sm)
            rows.append({"size": size, "patches": patches, "batch": batch, "op": name,
                         "kind": "gemm", "grid": g, "waves": w, "tail": tail, "idle": idle})
        # attention 单独算（它的分块逻辑不同）
        g_a = attention_blocks(seq)
        w_a, t_a, i_a = waves(g_a, SM_COUNT, tb_per_sm)
        rows.append({"size": size, "patches": patches, "batch": batch, "op": "attention",
                     "kind": "attn", "grid": g_a, "waves": w_a, "tail": t_a, "idle": i_a})
    return rows


def print_table(rows: list[dict], batch: int) -> None:
    print(f"\n{'='*94}")
    print(f"波量化估算（N={SM_COUNT} SM，每 SM {TB_PER_SM} 个 TB，batch={batch}）")
    print(f"{'='*94}")
    print(f"{'尺寸':>6} {'seq':>5} {'算子':>12} {'线程块 g':>10} {'waves':>6} {'tail':>5} {'SM 空闲比':>10}")
    print("-" * 94)
    for r in rows:
        if r["batch"] != batch:
            continue
        print(f"{r['size']:>6} {r['patches']+1:>5} {r['op']:>12} {r['grid']:>10} "
              f"{r['waves']:>6} {r['tail']:>5} {r['idle']:>9.1%}")
    print()


def print_scaling(rows: list[dict]) -> None:
    """跨尺寸：空闲比是否随尺寸下降？"""
    print(f"{'='*94}")
    print("跨尺寸趋势（同一算子，不同请求尺寸）")
    print(f"{'='*94}")
    ops = []
    for r in rows:
        if r["op"] not in ops:
            ops.append(r["op"])
    print(f"{'尺寸':>6}" + "".join(f"{op:>12}" for op in ops))
    by = {}
    for r in rows:
        by[(r["size"], r["op"])] = r["idle"]
    for size in sorted({r["size"] for r in rows}):
        print(f"{size:>6}" + "".join(
            f"{by.get((size, op), float('nan')):>11.1%} " for op in ops))
    print()

    # 平均空闲比随尺寸的变化
    print("平均 SM 空闲比（跨算子）：")
    prev = None
    trend = []
    for size in sorted({r["size"] for r in rows}):
        vals = [v for (s, _), v in by.items() if s == size]
        avg = sum(vals) / len(vals)
        trend.append((size, avg))
        print(f"  {size:>5}: {avg:>6.1%}")
    print()
    mono = all(a[1] >= b[1] - 1e-9 for a, b in zip(trend, trend[1:]))
    label = "是 ✔（与实测「尺寸越大越稳」同向）" if mono else "否"
    print(f"  单调递减？ {label}")
    print()
    print("  ⚠️ 注意：本脚本的 tile 尺寸与 TB/SM 是**估算值**。")
    print("     真实 tile 由 cublasLt 运行时挑选，未公开。因此只应使用**趋势与量级**，")
    print("     不应把具体 idle 数值当作事实引用。要确认，需用 ncu 采真实 grid 尺寸。")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 波量化解析估算")
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 288, 336, 384, 448, 512, 672])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4],
                        help="同尺寸拼批的批大小")
    parser.add_argument("--tb-per-sm", type=int, default=TB_PER_SM,
                        help="每 SM 驻留的线程块数（敏感度分析用）")
    parser.add_argument("--emit-csv", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for b in args.batches:
        all_rows += analyze(args.sizes, batch=b, tb_per_sm=args.tb_per_sm)

    for b in args.batches:
        print_table(all_rows, b)
    print_scaling([r for r in all_rows if r["batch"] == 1])

    # 敏感度：TB/SM 取值对结论的影响
    print(f"{'='*94}")
    print("敏感度分析：每 SM 的 TB 数对平均空闲比的影响（batch=1）")
    print(f"{'='*94}")
    print(f"{'尺寸':>6}" + "".join(f"{'TB/SM=' + str(t):>12}" for t in (1, 2, 4)))
    for size in args.sizes:
        cells = []
        for t in (1, 2, 4):
            rs = [r for r in analyze([size], batch=1, tb_per_sm=t) if r["batch"] == 1]
            cells.append(sum(r["idle"] for r in rs) / len(rs))
        print(f"{size:>6}" + "".join(f"{c:>11.1%} " for c in cells))
    print()
    print("  若不同 TB/SM 下趋势一致 → 结论对参数不敏感，可信度较高。")

    if args.emit_csv:
        args.emit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.emit_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n已写出 {args.emit_csv}")


if __name__ == "__main__":
    main()
