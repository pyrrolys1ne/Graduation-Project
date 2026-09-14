from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import statistics

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 336, 448, 672])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/profiles/measured.csv"))
    parser.add_argument("--no-enforce-monotonic", dest="enforce_monotonic", action="store_false",
                        help="保留原始中位数，不做单调化")
    args = parser.parse_args()

    config = load_config(args.config)
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    encoder = ClipEncoderBackend(config.model, resource, 1)
    quotas = config.scheduler.quota_levels if resource.enforces_sm_partition else (1.0,)
    rows = []
    for size in args.sizes:
        for quota in quotas:
            samples = []
            total = args.warmup + args.repeats
            for index in range(total):
                job = EncodeJob(f"profile-{size}-{quota}-{index}", config.seed + index, size, size, 60_000)
                job.sm_fraction = quota
                result = encoder.encode(job, 0)
                if index >= args.warmup:
                    samples.append(result.execution_ms)
            rows.append(
                {
                    "width": size,
                    "height": size,
                    "patches": (size // 32) ** 2,
                    "sm_fraction": quota,
                    "latency_ms": statistics.median(samples),
                    "mean_ms": statistics.fmean(samples),
                    "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "samples": len(samples),
                    "backend": resource.name,
                    "sm_enforced": resource.enforces_sm_partition,
                }
            )
    if args.enforce_monotonic:
        rows = enforce_monotonic(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"剖析完成: {args.output}；资源后端={resource.name}；真实 SM 配额={resource.enforces_sm_partition}")
    if args.enforce_monotonic:
        print("  已做单调化：同一尺寸下，配额升高时延迟不会变大（见 enforce_monotonic 说明）")


def enforce_monotonic(rows: list[dict]) -> list[dict]:
    """强制"配额越高、延迟不增"。

    实测中位数会因噪声违反单调性（例如 224×224 测出 0.5 档 4.80 ms 快于 0.75 档
    5.68 ms——该尺寸标准差就有 1.2–1.5 ms）。但性能表是**调度决策的输入**：
    非单调会让"多给资源反而预测更慢"，插值与排序都会失真，且这类失真不会报错。

    做法：对每个尺寸，按配额从高到低扫描，取运行最大值。即
    ``latency[q] = max(latency[q], latency[q'])``，其中 ``q'`` 是所有比 ``q`` 高的档位。
    这只把异常偏低的值抬到相邻更高配额的水平，不会凭空降低任何预测。

    代价是可能掩盖真实的非单调效应（例如某个尺寸在某个配额下确实更快）。
    因此**原始值仍保留在 mean_ms/std_ms 列里**，需要时可比对。
    """
    from collections import defaultdict

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["width"]), int(row["height"]))].append(row)

    adjusted: list[dict] = []
    for key, group in grouped.items():
        order = sorted(group, key=lambda r: float(r["sm_fraction"]), reverse=True)
        ceiling = float("-inf")
        for row in order:
            value = float(row["latency_ms"])
            if value < ceiling:
                row["latency_ms_measured"] = value
                row["latency_ms"] = ceiling
                row["monotonic_adjusted"] = True
            else:
                ceiling = value
                row["monotonic_adjusted"] = False
            adjusted.append(row)
    adjusted.sort(key=lambda r: (int(r["width"]), float(r["sm_fraction"])))
    return adjusted


if __name__ == "__main__":
    main()
