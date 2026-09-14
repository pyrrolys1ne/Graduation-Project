from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import concurrent.futures
import json

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel
from encoder_sched.resource import create_resource_backend
from encoder_sched.scheduler import Scheduler


def main() -> None:
    config = load_config()
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    performance = PerformanceModel(config.resolve(config.profiling.table_path))
    scheduler = Scheduler(
        config.scheduler.policy,
        performance,
        config.scheduler.quota_levels,
        config.scheduler.deadline_tie_ms,
    )
    encoder = ClipEncoderBackend(config.model, resource, config.executor.streams)
    jobs = [
        EncodeJob("smoke-224", config.seed, 224, 224, 1000),
        EncodeJob("smoke-336", config.seed + 1, 336, 336, 1000),
    ]
    for job in jobs:
        scheduler.prepare(job, job.arrival_ns)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(encoder.encode, job, worker_id) for worker_id, job in enumerate(jobs)]
        results = [future.result() for future in futures]
    print(json.dumps({"environment": encoder.environment(), "results": [result.__dict__ for result in results]}, indent=2))


if __name__ == "__main__":
    main()
