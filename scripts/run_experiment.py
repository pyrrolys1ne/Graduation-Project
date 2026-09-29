from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from encoder_sched.metrics import client_side_summary


async def run_one(client: httpx.AsyncClient, row: dict, origin_ns: int) -> dict:
    arrival_ms = row.pop("arrival_ms", 0.0)
    target_ns = origin_ns + int(arrival_ms * 1_000_000)
    delay = (target_ns - time.perf_counter_ns()) / 1_000_000_000
    if delay > 0:
        await asyncio.sleep(delay)
    # 迟到量必须在 await client.post 之前取：它衡量的是"负载规定的到达时刻"与
    # "请求真正发出"之间的差，也就是客户端自身（事件循环 / 连接池上限）造成的排队。
    # 若在 post 之后取，这个差值会退化成响应时间，失去区分能力。
    dispatch_lateness_ms = (time.perf_counter_ns() - target_ns) / 1_000_000
    started = time.perf_counter_ns()
    response = await client.post("/v1/encode", json=row, timeout=180)
    client_ms = (time.perf_counter_ns() - started) / 1_000_000
    data = response.json()
    if response.is_error:
        return {
            "request_id": row["request_id"],
            "http_status": response.status_code,
            "error": data,
            "dispatch_lateness_ms": dispatch_lateness_ms,
        }
    data["http_status"] = response.status_code
    data["client_latency_ms"] = client_ms
    data["dispatch_lateness_ms"] = dispatch_lateness_ms
    return data


async def warmup_and_reset(url: str, warmup: int) -> None:
    """暖机 + 重置服务端内存指标。**必须在任何分片开始发送之前只做一次**：
    `/metrics/reset` 会把服务端计数清零，若某个分片在别的分片已开始之后才重置，
    先到的请求就从统计里消失了。"""
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    async with httpx.AsyncClient(base_url=url, limits=limits) as client:
        for index in range(warmup):
            response = await client.post("/v1/encode", json={
                "request_id": f"warmup-{time.time_ns()}-{index}",
                "seed": index, "width": 224, "height": 224,
                "deadline_ms": 10_000, "priority": 0,
            }, timeout=180)
            response.raise_for_status()
        reset = await client.post("/metrics/reset")
        reset.raise_for_status()


def merge_shard_summary(server_summary: dict, merged_rows: list[dict],
                        concurrency: int, shards: int) -> dict:
    """把分片客户端的原始记录并进服务端汇总。**纯函数，可离线测。**

    `pool_can_throttle` 必须按**每片**判断，不能按合并后的总记录数：
    每个分片各自的连接上限 ≥ 它自己的请求数时，该分片不可能因为等空闲连接而推迟下发，
    超出的 overhead 是前端开销而不是客户端排队。用 `concurrency < len(merged)` 会得到
    225 < 900 = True，把一个开环的批次误判成"连接池成为瓶颈"并抑制整批判定
    （这正是本项目历史上让一整批实验结论作废的那个陷阱，§21.2）。
    """
    per_shard = -(-len(merged_rows) // max(shards, 1))
    server_summary["client_scope"] = client_side_summary(
        merged_rows, concurrency=concurrency,
        pool_can_throttle=concurrency < per_shard,
    )
    server_summary["shards"] = shards
    return server_summary


async def run_sharded(args: argparse.Namespace, rows: list[dict]) -> None:
    """把一个负载拆成 N 个**独立进程**同时回放，父进程只负责合并。

    为什么需要它（`docs/实验记录.md` §27）：真机过载档上单进程客户端会先垮——
    在飞请求数 = 到达率 × 延迟，过载时延迟随队列增长，在飞数随之增长，
    单个 Python 事件循环扛不住。用 CPU 侧替身复现后确认，**根因是整机 CPU 负载
    把客户端的事件循环饿死**（加 3 个忙循环进程即可复现：发出迟到 P99 从 5 ms
    涨到 522 ms），而**服务端自己的预处理线程正是那个负载**。
    把发送拆到多个进程后，每个事件循环只扛 1/N 的流量，实测 4 个分片
    在同样负载下发出迟到 P99 都是 ~2 ms。

    分片按 `index % N` 划分，**所有分片共用同一个起始时刻**，因此合起来仍是
    同一个到达过程（`time.perf_counter_ns` 是系统级单调时钟，跨进程可比）。
    """
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [output.with_suffix(f".shard{k}.jsonl") for k in range(args.shards)]
    await warmup_and_reset(args.url, args.warmup)
    # 留出一点余量让所有子进程都起来，避免第一个分片独自吃掉开头一段负载。
    start_ns = time.perf_counter_ns() + int(args.shard_lead_ms * 1e6)
    children = []
    for index, shard_path in enumerate(shard_paths):
        children.append(subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()),
            "--url", args.url, "--workload", str(args.workload),
            "--output", str(shard_path), "--concurrency", str(args.concurrency),
            "--warmup", "0", "--shards", str(args.shards), "--shard-index", str(index),
            "--start-at-ns", str(start_ns), "--jsonl-only",
        ]))
    codes = [child.wait() for child in children]
    if any(code != 0 for code in codes):
        raise SystemExit(f"分片客户端退出码非零: {codes}")

    merged = []
    for shard_path in shard_paths:
        merged += [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line]
    async with httpx.AsyncClient(base_url=args.url, limits=httpx.Limits(max_connections=8)) as client:
        summary = (await client.get("/metrics/summary")).json()
    summary = merge_shard_summary(summary, merged, args.concurrency, args.shards)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("throughput_rps", "completed", "failed")}, ensure_ascii=False))


async def run(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines() if line]
    # 每次回放生成独立 request_id，避免服务端把重复实验误判为重复提交。
    run_tag = f"run-{time.time_ns()}"
    for index, row in enumerate(rows):
        row["request_id"] = f"{row['request_id']}-{run_tag}-{index}"

    if args.shards > 1 and args.shard_index is None:
        await run_sharded(args, rows)
        return

    if args.shard_index is not None:
        rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard_index]

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, limits=limits) as client:
        if args.shard_index is None:
            # 单进程路径才做暖机与重置；分片路径由父进程统一做（见 run_sharded）。
            await warmup_and_reset(args.url, args.warmup)
        origin_ns = args.start_at_ns or time.perf_counter_ns()
        results = await asyncio.gather(*(run_one(client, dict(row), origin_ns) for row in rows))
        if args.jsonl_only:
            # 分片子进程只写原始 JSONL，汇总由父进程在所有分片结束后统一做——
            # 每个分片各自抓一次 /metrics/summary 只会拿到彼此的中间快照。
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            return
        summary = (await client.get("/metrics/summary")).json()

    # 服务端口径（queue/execution/total，自请求到达服务端起算）与客户端口径
    # （client_latency_ms，含连接池等待与 HTTP 往返）**同时写入汇总但分别命名**：
    # 两者相差 15–34 倍，混用会让"尾延迟由谁造成"的结论整体翻转。
    summary["client_scope"] = client_side_summary(results, concurrency=args.concurrency)
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
    parser.add_argument(
        "--shards", type=int, default=1,
        help="把发送拆到几个独立进程。**过载档必须 >1**：单进程客户端会被整机 CPU 负载"
             "饿死（服务端自己的预处理线程就是那个负载），见 docs/实验记录.md §27。",
    )
    parser.add_argument("--shard-index", type=int, default=None,
                        help="内部使用：本进程负责第几个分片。给了一般用户不要传。")
    parser.add_argument("--start-at-ns", type=int, default=None,
                        help="内部使用：所有分片共用的起始时刻（perf_counter_ns，系统级单调时钟）。")
    parser.add_argument("--jsonl-only", action="store_true",
                        help="内部使用：分片子进程只写 JSONL，不做汇总。")
    parser.add_argument("--shard-lead-ms", type=float, default=1500.0,
                        help="分片模式下，从重置指标到开始发送之间留出的余量（毫秒），"
                             "让所有子进程都起来。")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
