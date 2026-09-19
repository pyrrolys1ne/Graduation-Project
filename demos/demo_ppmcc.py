"""验证 ConCo 的跨资源档位相关性假设在本机是否成立。

出处
----
ConCo: Optimizing Compilation of Concurrent Tensor Programs on Shared GPU（ICS 2025）§3.4。

论文的观察：**同一实现在不同 SM 资源档位下的归一化延迟是高度相关的**，
用 PPMCC（Pearson product-moment correlation coefficient）度量：

    PPMCC(X,Y) = Σᵢ[(Xᵢ−X̄)(Yᵢ−Ȳ)] / ( sqrt(Σᵢ(Xᵢ−X̄)²) · sqrt(Σᵢ(Yᵢ−Ȳ)²) )

其中 Xᵢ、Yᵢ 是同一实现在两个档位下的归一化执行时间。
原文结论：*"As the resource allocation percentages become more similar, the PPMCC increases."*
ConCo 据此把"其他档位筛出的 top 代码"拿到当前档位实测，把编译时间降到 Ansor 的 14.3%–32.3%。

为什么值得在本机验一次
----------------------
若假设成立，我们就能**用少量标定点插值出其他 TPC 档位的性能**，大幅缩短标定时间——
这对本课题直接有用（标定表的构建成本是主要开销之一）。

若假设**不成立**，我们就得到一个反例：ConCo 的共享假设建立在"SM 是主要瓶颈"之上；
而本机实测 12 个 TPC 中 **6 个就吃满显存带宽**，档位之间的相关性会被内存墙截断。
**两种结果都能写进论文**，所以这个 demo 没有"失败"的可能。

通过判据
--------
- 相邻 TPC 档位（如 6↔9、9↔12）的 PPMCC **> 0.9** → 假设在本机成立，可做插值。
- 相邻档位 PPMCC **< 0.9** → 假设被带宽墙截断，得到一个可写的反例。

用法
----
    python demos/demo_pp mcc.py
    python demos/demo_pp mcc.py --table data/profiles/default.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import statistics


def load_table(path: Path) -> dict[tuple[int, float], float]:
    """读标定表 -> {(patches, sm_fraction): latency_ms}。"""
    rows: dict[tuple[int, float], float] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["patches"]), float(row["sm_fraction"]))] = float(row["latency_ms"])
    return rows


def pearson(x: list[float], y: list[float]) -> float:
    """PPMCC（即 Pearson 相关系数）。样本量 <2 或方差为 0 时返回 nan。"""
    n = len(x)
    if n < 2 or len(y) != n:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = sum(v * v for v in dx) ** 0.5
    sy = sum(v * v for v in dy) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def main() -> None:
    parser = argparse.ArgumentParser(description="ConCo 跨档相关性假设的本机检验")
    parser.add_argument("--table", type=Path, default=Path("data/profiles/default.csv"))
    args = parser.parse_args()

    table = load_table(args.table)
    if not table:
        raise SystemExit(f"标定表为空: {args.table}")

    patches_list = sorted({p for p, _ in table})
    quotas = sorted({q for _, q in table})
    print(f"标定表 {args.table}")
    print(f"  请求规模（patches）: {patches_list}")
    print(f"  配额档位（sm_fraction）: {quotas}")
    print()

    # 每个档位一个向量：跨请求规模的延迟
    vectors: dict[float, list[float]] = {}
    for q in quotas:
        series = [table.get((p, q)) for p in patches_list]
        if any(v is None for v in series):
            print(f"  ⚠ 档位 {q} 缺样本，跳过")
            continue
        vectors[q] = [float(v) for v in series]

    print("原值矩阵（行 = 配额，列 = patches）")
    header = "  " + " ".join(f"{p:>9}" for p in patches_list)
    print(f"  {'配额':>6} {header.strip()}")
    for q, vec in vectors.items():
        print(f"  {q:>6.2f} " + " ".join(f"{v:>9.3f}" for v in vec))
    print()

    # ---- 相关性：原文用"同一实现在两个档位下的归一化执行时间" ----
    # 归一化的意义是去掉档位的整体快慢，只看"跨请求规模的形状"是否一致。
    def normalize(vec: list[float]) -> list[float]:
        base = vec[-1]  # 以最大请求（最后一个）为基准
        return [v / base for v in vec]

    norm = {q: normalize(v) for q, v in vectors.items()}

    print("归一化矩阵（以最大请求为基准）")
    print(f"  {'配额':>6} {header.strip()}")
    for q, vec in norm.items():
        print(f"  {q:>6.2f} " + " ".join(f"{v:>9.3f}" for v in vec))
    print()

    print("=" * 78)
    print("PPMCC 矩阵（原值 / 归一化后）")
    print("=" * 78)
    print(f"  {'':>6}" + "".join(f"{q:>16}" for q in quotas))
    pairs: list[tuple[float, float, float, float]] = []
    for i, qa in enumerate(quotas):
        if qa not in vectors:
            continue
        cells = []
        for qb in quotas:
            if qb not in vectors:
                cells.append(f"{'—':>16}")
                continue
            raw = pearson(vectors[qa], vectors[qb])
            nrm = pearson(norm[qa], norm[qb])
            cells.append(f"{raw:>7.3f}/{nrm:<8.3f}")
            if i < quotas.index(qb):
                pairs.append((qa, qb, raw, nrm))
        print(f"  {qa:>6.2f}" + "".join(cells))
    print()
    print("  （格式：原值 PPMCC / 归一化后 PPMCC）")
    print()

    # ---- 判读 ----
    print("=" * 78)
    print("判读")
    print("=" * 78)

    # 相邻档位 vs 远距离档位：论文说"档位越接近，PPMCC 越高"
    gap_stats: dict[int, list[float]] = {}
    for i, qa in enumerate(quotas):
        for qb in quotas[i + 1:]:
            if qa not in vectors or qb not in vectors:
                continue
            gap = round(abs(qb - qa) * 100)
            gap_stats.setdefault(gap, []).append(pearson(norm[qa], norm[qb]))

    print(f"  {'档位间距':>10} {'对数':>5} {'PPMCC 中位':>12} {'最小':>8} {'最大':>8}")
    for gap in sorted(gap_stats):
        vals = [v for v in gap_stats[gap] if v == v]
        if not vals:
            continue
        print(f"  {gap:>8}% {len(vals):>5} {statistics.median(vals):>12.3f} "
              f"{min(vals):>8.3f} {max(vals):>8.3f}")
    print()

    close = [v for gap, vals in gap_stats.items() if gap <= 25 for v in vals if v == v]
    far = [v for gap, vals in gap_stats.items() if gap >= 50 for v in vals if v == v]
    if close and far:
        mc, mf = statistics.median(close), statistics.median(far)
        print(f"  相邻档位（间距 ≤25%）PPMCC 中位 = {mc:.3f}")
        print(f"  远档位  （间距 ≥50%）PPMCC 中位 = {mf:.3f}")
        if mc > mf:
            print("  → 与 ConCo 的 '档位越接近 PPMCC 越高' **一致** ✔")
        else:
            print("  → 与 ConCo 的该论断**不一致**，需记录为反例 ✘")
        print()

    adjacent = [v for gap, vals in gap_stats.items() if gap <= 25 for v in vals if v == v]
    if adjacent:
        med = statistics.median(adjacent)
        print(f"  相邻档位 PPMCC 中位 = {med:.3f}")
        if med > 0.9:
            print("  ✅ 通过：可尝试用少量标定点插值出其他 TPC 档位，缩短标定时间")
        else:
            print("  ❌ 判死：跨档相关性不足，不能用插值省标定——"
                  "ConCo 的共享假设在本机（带宽墙）不成立")
            print("     这本身是可写进论文的反例：该假设以 'SM 是主要瓶颈' 为前提，")
            print("     而本机 12 个 TPC 中 6 个就吃满显存带宽。")


if __name__ == "__main__":
    main()
