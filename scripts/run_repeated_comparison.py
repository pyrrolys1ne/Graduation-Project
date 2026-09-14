"""重复对照实验：基线与 DACC 消融，每组重复 N 次并报告标准差。

与 ``run_real_comparison.py`` 的区别：
- 每个变体重启服务、独立运行 N 次，得到可计算标准差的重复样本；
- 支持通过 ``dacc_overrides`` 做消融，不需要改代码；
- 输出含均值与标准差，便于判断策略之间是否真的可区分。

用法::

    python scripts/run_repeated_comparison.py --group baselines --repeats 5
    python scripts/run_repeated_comparison.py --group ablations --repeats 5
    python scripts/run_repeated_comparison.py --group window --repeats 5

注意：本脚本只验证软件调度闭环。资源后端为 ``proxy`` 时结果不含真实 SM 隔离效果，
输出中的 ``resource_backend`` 与 ``sm_isolation_verified`` 字段会如实标注。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
import statistics
import subprocess
import time

import httpx
import yaml

#: 每个变体 = 基础配置上的一处改动。消融项只改 DACC 评分权重。
GROUPS: dict[str, dict[str, dict]] = {
    "baselines": {
        "serial_fcfs": {"policy": "serial_fcfs"},
        "multistream_fcfs": {"policy": "multistream_fcfs"},
        "edf": {"policy": "edf"},
        "edf_size": {"policy": "edf_size"},
        "dacc": {"policy": "dacc"},
    },
    "ablations": {
        # 最有诊断价值的一项：禁止配对，DACC 退化为"只选单请求"。
        # 若它与 edf 表现相近，即证明损害来自配对本身而非评分函数。
        "dacc_no_pairing": {"policy": "dacc", "overrides": {"max_pair_conflict": 0.0}},
        "dacc_no_complementarity": {"policy": "dacc", "overrides": {"w_complementarity": 0.0}},
        "dacc_no_uncertainty": {"policy": "dacc", "overrides": {"w_uncertainty": 0.0, "beta": 0.0}},
        "dacc_no_risk": {"policy": "dacc", "overrides": {"w_risk": 0.0}},
        "dacc_no_urgency": {"policy": "dacc", "overrides": {"w_urgency": 0.0}},
    },
    "window": {
        "dacc_k1": {"policy": "dacc", "overrides": {"window": 1}},
        "dacc_k4": {"policy": "dacc", "overrides": {"window": 4}},
        "dacc_k8": {"policy": "dacc"},
        "dacc_k16": {"policy": "dacc", "overrides": {"window": 16}},
    },
    # 配对倾向扫描：消除结构性偏向之后，配对能否胜出完全由权重决定。
    # w_complementarity 越小越愿意配对；w_gain 越大越看重"一次服务两个"的吞吐收益。
    # 这组实验直接回答"配对机制是否值得保留"，不靠直觉调参。
    "pairing": {
        "dacc_wc1.0": {"policy": "dacc"},
        "dacc_wc0.5": {"policy": "dacc", "overrides": {"w_complementarity": 0.5}},
        "dacc_wc0.0": {"policy": "dacc", "overrides": {"w_complementarity": 0.0}},
        "dacc_gain2": {"policy": "dacc", "overrides": {"w_gain": 2.0}},
    },
    # 并发方式对照：空间分割（多 worker/多 stream 各跑一个请求）vs 批处理（合并成一次前向）。
    # 两臂都是 FCFS 排序、同一资源后端，只改并发方式。
    #
    # 主轴是 `delay_ms`——"愿意等多久来凑批"。它直接决定这条取舍曲线：
    #   delay=0  绝不等待，只取队列里已有的（等价于逐请求，拿不到批处理的好处）
    #   delay 增大 → batch 变大 → 吞吐上升，但每个请求的首字节延迟也随之上升
    # 产品文档给的参考是交互式场景下队列延迟压到 100–300 µs（见 Triton 的
    # max_queue_delay_microseconds），本组从 0 扫到 20 ms 是为了把整条曲线画出来。
    "batch": {
        "spatial_2stream": {"policy": "multistream_fcfs", "batch": 1},
        "batch4_d0": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 0.0},
        "batch4_d1": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 1.0},
        "batch4_d5": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 5.0},
        "batch4_d20": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 20.0},
        "batch8_d5": {"policy": "serial_fcfs", "batch": 8, "delay_ms": 5.0},
    },
    # 配额选择策略对照：deadline_min（取满足截止期的最小配额）vs full（始终用满）。
    # 动机见 encoder_sched/scheduler.py 的 choose_quota 说明。
    "quota": {
        "dacc_deadline_min": {"policy": "dacc", "quota_policy": "deadline_min"},
        "dacc_full": {"policy": "dacc", "quota_policy": "full"},
        "edf_size_deadline_min": {"policy": "edf_size", "quota_policy": "deadline_min"},
        "edf_size_full": {"policy": "edf_size", "quota_policy": "full"},
    },
}

METRICS = ("throughput_rps", "total_mean_ms", "total_p50_ms", "total_p95_ms", "total_p99_ms", "slo_violation_rate", "gpu_utilization_percent", "gpu_samples")


def wait_ready(url: str, process: subprocess.Popen, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"服务进程提前退出，返回码 {process.returncode}")
        try:
            response = httpx.get(f"{url}/health", timeout=2)
            if response.is_success:
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError("等待服务启动超时")


def build_config(base: dict, variant: dict, workdir: Path) -> Path:
    data = json.loads(json.dumps(base))
    scheduler = data.setdefault("scheduler", {})
    scheduler["policy"] = variant["policy"]
    if variant.get("quota_policy"):
        scheduler["quota_policy"] = variant["quota_policy"]
    if variant.get("overrides"):
        scheduler["dacc_overrides"] = variant["overrides"]
    if variant.get("batch") is not None:
        data["batching"] = {
            "max_batch": variant["batch"],
            "max_delay_ms": variant.get("delay_ms", 2.0),
        }
    log_dir = workdir / "logs"
    data.setdefault("logging", {})["output_dir"] = str(log_dir.resolve())
    profile = Path(data.get("profiling", {}).get("table_path", "data/profiles/default.csv"))
    if not profile.is_absolute():
        profile = (Path.cwd() / profile).resolve()
    data.setdefault("profiling", {})["table_path"] = str(profile)
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_once(config_path: Path, workload: Path, output: Path, concurrency: int, warmup: int, port: int) -> dict:
    env = {**os.environ, "ENCODER_SCHED_CONFIG": str(config_path.resolve())}
    log = config_path.parent / "server.log"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "encoder_sched.api", "--port", str(port)],
            stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
    try:
        health = wait_ready(f"http://127.0.0.1:{port}", process)
        subprocess.run(
            [sys.executable, "scripts/run_experiment.py", "--url", f"http://127.0.0.1:{port}",
             "--workload", str(workload), "--output", str(output),
             "--concurrency", str(concurrency), "--warmup", str(warmup)],
            check=True, env=env,
        )
        summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
        summary["gpu"] = health.get("environment", {}).get("gpu")
        summary["resource"] = health.get("resource")
        return summary
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def flatten(summary: dict) -> dict:
    return {
        "throughput_rps": summary["throughput_rps"],
        "total_mean_ms": summary["total_latency_ms"]["mean"],
        "total_p50_ms": summary["total_latency_ms"]["p50"],
        "total_p95_ms": summary["total_latency_ms"]["p95"],
        "total_p99_ms": summary["total_latency_ms"]["p99"],
        "slo_violation_rate": summary["slo_violation_rate"],
        "gpu_utilization_percent": summary["gpu_utilization_percent"],
        "gpu_samples": summary.get("gpu_samples"),
        "completed": summary["completed"],
        "failed": summary["failed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重复对照实验（含 DACC 消融）")
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--workload", type=Path, default=Path("data/workloads/comparison_mixed.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    out_dir = args.output_dir or Path(f"results/repeated_{args.group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    variants = GROUPS[args.group]

    results: dict[str, dict] = {}
    for name, variant in variants.items():
        workdir = out_dir / name
        workdir.mkdir(parents=True, exist_ok=True)
        config_path = build_config(base, variant, workdir)
        samples: list[dict] = []
        for repeat in range(args.repeats):
            output = workdir / f"repeat{repeat}.jsonl"
            print(f"[{name}] repeat {repeat + 1}/{args.repeats} ...", flush=True)
            summary = run_once(config_path, args.workload, output, args.concurrency, args.warmup, args.port)
            samples.append(flatten(summary))
        aggregate = {}
        for metric in METRICS:
            values = [s[metric] for s in samples]
            aggregate[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        aggregate["completed"] = sum(s["completed"] for s in samples)
        aggregate["failed"] = sum(s["failed"] for s in samples)
        backend = samples[0].get("resource") or {}
        results[name] = {
            "variant": variant,
            "repeats": args.repeats,
            "aggregate": aggregate,
            "samples": samples,
            "resource_backend": backend.get("effective_backend"),
            "enforces_sm_partition": backend.get("enforces_sm_partition"),
        }
        print(f"  -> 吞吐 {aggregate['throughput_rps']['mean']:.2f}±{aggregate['throughput_rps']['std']:.2f} req/s, "
              f"P99 {aggregate['total_p99_ms']['mean']:.1f}±{aggregate['total_p99_ms']['std']:.1f} ms, "
              f"SLO 违约 {aggregate['slo_violation_rate']['mean'] * 100:.1f}%", flush=True)

    payload = {
        "group": args.group,
        "repeats": args.repeats,
        "workload": str(args.workload),
        "concurrency": args.concurrency,
        "sm_isolation_verified": bool(results and results[next(iter(results))]["enforces_sm_partition"]),
        "note": "resource_backend=proxy 时不含真实 SM 隔离效果；这些数字只反映软件调度层。",
        "variants": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["variant", "resource_backend", "enforces_sm_partition"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, item in results.items():
            row = {"variant": name, "resource_backend": item["resource_backend"], "enforces_sm_partition": item["enforces_sm_partition"]}
            for metric in METRICS:
                row[f"{metric}_mean"] = round(item["aggregate"][metric]["mean"], 4)
                row[f"{metric}_std"] = round(item["aggregate"][metric]["std"], 4)
            writer.writerow(row)
    print(f"\n结果写入 {out_dir}/summary.json 与 summary.csv")


if __name__ == "__main__":
    main()
