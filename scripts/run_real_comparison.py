from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
import subprocess
import time

import httpx
import yaml


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
        time.sleep(1)
    raise TimeoutError("等待真实 CLIP 服务启动超时")


def main() -> None:
    parser = argparse.ArgumentParser(description="依次运行五种真实 CLIP 调度策略并汇总")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--workload", type=Path, default=Path("data/workloads/comparison_mixed.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_comparison"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for policy in ("serial_fcfs", "multistream_fcfs", "edf", "edf_size", "dacc"):
        data = json.loads(json.dumps(base))
        data.setdefault("scheduler", {})["policy"] = policy
        data.setdefault("logging", {})["output_dir"] = str((args.output_dir / policy).resolve())
        data["logging"]["request_log"] = "requests.jsonl"
        profile_path = Path(data.get("profiling", {}).get("table_path", "data/profiles/default.csv"))
        if not profile_path.is_absolute():
            profile_path = (args.config.parent / profile_path).resolve()
        data.setdefault("profiling", {})["table_path"] = str(profile_path)
        config_path = args.output_dir / f"config_{policy}.yaml"
        config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        log_path = args.output_dir / f"server_{policy}.log"
        with log_path.open("w", encoding="utf-8") as log:
            env = os.environ.copy()
            env["ENCODER_SCHED_CONFIG"] = str(config_path.resolve())
            process = subprocess.Popen([sys.executable, "-m", "encoder_sched.api"], stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            health = wait_ready("http://127.0.0.1:8000", process)
            output = args.output_dir / f"{policy}.jsonl"
            command = [sys.executable, "scripts/run_experiment.py", "--workload", str(args.workload), "--output", str(output), "--concurrency", str(args.concurrency), "--warmup", str(args.warmup)]
            subprocess.run(command, check=True, env={**os.environ, "ENCODER_SCHED_CONFIG": str(config_path.resolve())})
            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
            summary["policy"] = policy
            summary["gpu"] = health.get("environment", {}).get("gpu")
            summary["resource_backend"] = data.get("executor", {}).get("resource_backend")
            summaries.append(summary)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["policy", "submitted", "completed", "failed", "throughput_rps", "queue_mean_ms", "total_mean_ms", "total_p99_ms", "slo_violation_rate", "gpu_utilization_percent", "gpu"]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({"policy": item["policy"], "submitted": item["submitted"], "completed": item["completed"], "failed": item["failed"], "throughput_rps": item["throughput_rps"], "queue_mean_ms": item["queue_latency_ms"]["mean"], "total_mean_ms": item["total_latency_ms"]["mean"], "total_p99_ms": item["total_latency_ms"]["p99"], "slo_violation_rate": item["slo_violation_rate"], "gpu_utilization_percent": item["gpu_utilization_percent"], "gpu": item["gpu"]})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
