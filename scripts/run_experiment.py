from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def run_one(client: httpx.AsyncClient, row: dict, origin_ns: int) -> dict:
    target_ns = origin_ns + int(row.pop("arrival_ms") * 1_000_000)
    delay = (target_ns - time.perf_counter_ns()) / 1_000_000_000
    if delay > 0:
        await asyncio.sleep(delay)
    started = time.perf_counter_ns()
    response = await client.post("/v1/encode", json=row, timeout=180)
    client_ms = (time.perf_counter_ns() - started) / 1_000_000
    data = response.json()
    if response.is_error:
        return {"request_id": row["request_id"], "http_status": response.status_code, "error": data}
    data["http_status"] = response.status_code
    data["client_latency_ms"] = client_ms
    return data


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines() if line]
    # 每次回放生成独立 request_id，避免服务端把重复实验误判为重复提交。
    run_tag = f"run-{time.time_ns()}"
    for index, row in enumerate(rows):
        row["request_id"] = f"{row['request_id']}-{run_tag}-{index}"
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, limits=limits) as client:
        for index in range(args.warmup):
            warmup = {
                "request_id": f"warmup-{time.time_ns()}-{index}",
                "seed": index,
                "width": 224,
                "height": 224,
                "deadline_ms": 10_000,
                "priority": 0,
            }
            response = await client.post("/v1/encode", json=warmup, timeout=180)
            response.raise_for_status()
        reset = await client.post("/metrics/reset")
        reset.raise_for_status()
        origin_ns = time.perf_counter_ns()
        results = await asyncio.gather(*(run_one(client, dict(row), origin_ns) for row in rows))
        summary = (await client.get("/metrics/summary")).json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", type=Path, default=Path("data/workloads/mixed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/experiment.jsonl"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
