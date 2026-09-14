from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import statistics

from encoder_sched.simulator import load_workload, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="使用同一工作负载离线比较五种调度策略")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=Path("data/profiles/default.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/offline_comparison.json"))
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rows = load_workload(args.workload)
    results = []
    for policy in ("serial_fcfs", "multistream_fcfs", "edf", "edf_size", "dacc"):
        runs = [simulate(rows, policy, args.table, args.streams).as_dict() for _ in range(max(1, args.repeats))]
        aggregate = {"policy": policy, "repeats": len(runs)}
        for key in ("completed", "throughput_rps", "mean_latency_ms", "p99_latency_ms", "slo_violation_rate", "makespan_ms"):
            values = [float(run[key]) for run in runs]
            aggregate[key] = statistics.fmean(values)
            aggregate[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        results.append(aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
