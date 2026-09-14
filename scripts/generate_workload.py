from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIZES = (224, 336, 448, 672)


def generate(count: int, pattern: str, seed: int, slo: str) -> list[dict]:
    rng = random.Random(seed)
    slo_factor = {"tight": 1.5, "normal": 3.0, "loose": 6.0}[slo]
    base_latency = {224: 6.0, 336: 13.0, 448: 22.0, 672: 50.0}
    rows = []
    arrival_ms = 0.0
    for index in range(count):
        if pattern == "uniform":
            arrival_ms += 20.0
        elif pattern == "burst":
            arrival_ms += 100.0 if index % 10 == 0 else 0.5
        else:
            arrival_ms += rng.expovariate(1 / 20.0)
        size = SIZES[index % len(SIZES)] if pattern == "mixed" else rng.choice(SIZES)
        rows.append(
            {
                "arrival_ms": round(arrival_ms, 3),
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
    parser.add_argument("--output", type=Path, default=Path("data/workloads/mixed.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in generate(args.count, args.pattern, args.seed, args.slo):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"生成 {args.count} 个请求: {args.output}")


if __name__ == "__main__":
    main()

