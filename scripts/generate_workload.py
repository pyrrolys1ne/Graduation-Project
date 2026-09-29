from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIZES = (224, 336, 448, 672)

#: deadline 的固定下限（毫秒）。这一项覆盖请求到达服务端到开始执行之间的固定开销
#: （HTTP 往返、事件循环、线程调度）；不含它会让 deadline 比实际可达的端到端延迟还短。
SLO_FLOOR_MS = 10.0

#: 单请求的**串行执行**服务时间（毫秒），即"什么都不做"这个参照系的服务时间。
#: deadline 由它乘以 ``slo_factor`` 得到。
#:
#: ⚠️ **这是一组实测值，必须随测量口径更新。** 取值来自 2026-09-26 在本机对
#: `multistream_fcfs` + `streams=1` 的测量（`execution_ms` 中位，每尺寸 10 次）。
#: 上一版用的是图回放之前的旧值（224/336/448/672 = 6/13/22/50 ms），
#: 其中 336/448/672 **偏高 2–4 倍**——用旧值生成的 deadline 对今天的快路径
#: 过松，SLO 违约率会恒为 0、失去区分度（§21.3 遇到的正是这个问题的另一个极端）。
#:
#: 换硬件、换模型或换推理栈之后，必须重新测这组数，否则 SLO 这一项没有意义。
BASE_LATENCY_MS = {224: 4.73, 336: 4.08, 448: 4.97, 672: 7.83}


def generate(count: int, pattern: str, seed: int, slo: str, arrival_ms: float = 20.0) -> list[dict]:
    """`arrival_ms` 是平均到达间隔（毫秒），直接决定到达率。

    这一点对评估调度策略至关重要：单个 CLIP ViT-B/32 在这台机器上的服务率约为
    57 req/s（见 `docs/实验记录.md`）。若到达率低于服务率，GPU 不饱和，所有策略的
    吞吐都会被到达率钉在同一个值上，**无法区分优劣**。要检验"配对并发填充空闲
    SM"这类吞吐导向的假设，必须让到达率高于服务率。
    """
    rng = random.Random(seed)
    slo_factor = {"tight": 1.5, "normal": 3.0, "loose": 6.0}[slo]
    rows = []
    current_ms = 0.0
    for index in range(count):
        if pattern == "uniform":
            current_ms += arrival_ms
        elif pattern == "burst":
            # burst 保持自身的突发结构（每 10 个请求一次突发），不随 arrival_ms 缩放。
            current_ms += 100.0 if index % 10 == 0 else 0.5
        else:
            current_ms += rng.expovariate(1 / arrival_ms)
        size = SIZES[index % len(SIZES)] if pattern == "mixed" else rng.choice(SIZES)
        rows.append(
            {
                "arrival_ms": round(current_ms, 3),
                "request_id": f"{pattern}-{seed}-{index:05d}",
                "seed": seed + index,
                "width": size,
                "height": size,
                "deadline_ms": round(BASE_LATENCY_MS[size] * slo_factor + SLO_FLOOR_MS, 3),
                "priority": rng.choice([0, 0, 0, 1]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--pattern", choices=["uniform", "burst", "mixed"], default="mixed")
    parser.add_argument("--slo", choices=["tight", "normal", "loose"], default="normal")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--arrival-ms",
        type=float,
        default=20.0,
        help="平均到达间隔（毫秒）。20ms≈50 req/s，低于本机服务率(≈57 req/s)，GPU 不会饱和；"
             "要区分吞吐导向的调度策略，请取小于 1000/57≈17.5 的值。",
    )
    parser.add_argument("--output", type=Path, default=Path("data/workloads/mixed.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.count, args.pattern, args.seed, args.slo, args.arrival_ms)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    span = rows[-1]["arrival_ms"] if rows else 0.0
    rate = len(rows) / (span / 1000) if span else 0.0
    print(f"生成 {args.count} 个请求: {args.output}（跨度 {span:.0f} ms，到达率约 {rate:.1f} req/s）")


if __name__ == "__main__":
    main()

