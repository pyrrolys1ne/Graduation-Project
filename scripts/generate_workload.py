from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIZES = (224, 336, 448, 672)


def generate(count: int, pattern: str, seed: int, slo: str, arrival_ms: float = 20.0) -> list[dict]:
    """`arrival_ms` 是平均到达间隔（毫秒），直接决定到达率。

    这一点对评估调度策略至关重要：单个 CLIP ViT-B/32 在这台机器上的服务率约为
    57 req/s（见 `docs/实验记录.md`）。若到达率低于服务率，GPU 不饱和，所有策略的
    吞吐都会被到达率钉在同一个值上，**无法区分优劣**。要检验"配对并发填充空闲
    SM"这类吞吐导向的假设，必须让到达率高于服务率。
    """
    rng = random.Random(seed)
    slo_factor = {"tight": 1.5, "normal": 3.0, "loose": 6.0}[slo]
    base_latency = {224: 6.0, 336: 13.0, 448: 22.0, 672: 50.0}
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
                "deadline_ms": round(base_latency[size] * slo_factor + 50.0, 3),
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

