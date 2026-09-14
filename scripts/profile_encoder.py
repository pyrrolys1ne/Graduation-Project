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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"剖析完成: {args.output}；资源后端={resource.name}；真实 SM 配额={resource.enforces_sm_partition}")


if __name__ == "__main__":
    main()
