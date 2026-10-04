"""CPU 侧替身：一个**饱和的排队服务**，用来标定负载生成能力——不占 GPU。

为什么需要它
────────────

§26.4 记录了一个硬约束：单进程 Python 客户端在 288 req/s 的过载档上**自己先垮**
（`client_queueing_flagged`、事件循环积压、连接池等待 P95 5979 ms）。根因是开环下
**在飞请求数 = 到达率 × 延迟**，过载时服务端延迟随队列增长，在飞数随之增长。

但那个结论是在真机上、占着 GPU 得到的，**每验证一次客户端改动都要重新占一次 GPU**。
本脚本把"饱和"这件事抽象出来：一个单服务台 + 无界 FIFO 队列，
服务时间固定，于是延迟随积压线性增长——与真实饱和服务的形状一致，
而它**完全不碰 GPU**。于是客户端的任何改动都可以先在 CPU 上标定，
再上真机确认。

模型
────

    POST /v1/encode → 入队 → 单个 worker 协程逐个取出、sleep(service_ms)、兑现 future

单服务台的容量 = 1000 / service_ms（req/s）。到达率高于它时队列无界增长，
响应延迟 = service_ms × (该请求在队列中的位置 + 1)，这正是真机过载时的形态。

判据（本脚本的用途就是让这些量在**无 GPU** 下可测）：

- 客户端在某个到达率下是否仍能按时发出请求（`dispatch_lateness_ms`）；
- 连接池等待是否开始占据端到端延迟（`client_queueing_flagged`）。

用法：

    .venv/bin/python scripts/stub_saturated_server.py --service-ms 4.5 --port 8099
    .venv/bin/python scripts/run_experiment.py \\
        --url http://127.0.0.1:8099 --workload data/workloads/saturating_280rps.jsonl \\
        --output /tmp/stub.jsonl --concurrency 900 --warmup 5

⚠️ 它**不**模拟 GPU、不模拟真实的编码延迟分布，也**不**用于产出任何课题结论。
它唯一的用途是把"客户端的容量"这件事从 GPU 上解耦出来单独测量。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class EncodeRequest(BaseModel):
    request_id: str
    seed: int = 0
    width: int = 224
    height: int = 224
    deadline_ms: float = 1000.0
    priority: int = 0


class Server:
    """单服务台 + 无界 FIFO 队列。"""

    def __init__(self, service_ms: float):
        self.service_ms = service_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self.completed = 0
        self.started_ns = time.perf_counter_ns()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    async def _run(self) -> None:
        while True:
            request, future = await self.queue.get()
            await asyncio.sleep(self.service_ms / 1000.0)
            if not future.done():
                future.set_result(request)
            self.completed += 1

    async def submit(self, request: EncodeRequest) -> dict:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((request, future))
        await future
        return {
            "request_id": request.request_id,
            "status": "completed",
            "width": request.width,
            "height": request.height,
            "patches": (request.width // 32) * (request.height // 32),
            "deadline_ms": request.deadline_ms,
            "priority": request.priority,
            "predicted_ms": self.service_ms,
            "sm_fraction": 1.0,
            "queue_ms": 0.0,
            "execution_ms": self.service_ms,
            "total_ms": self.service_ms,
            "embedding_dim": 512,
            "embedding_norm": 1.0,
            "slo_violated": False,
            "error": None,
            "metadata": {},
        }

    def reset(self) -> None:
        self.completed = 0
        self.started_ns = time.perf_counter_ns()

    def summary(self) -> dict:
        span = max((time.perf_counter_ns() - self.started_ns) / 1e9, 1e-9)
        return {
            "throughput_rps": self.completed / span,
            "completed": self.completed,
            "failed": 0,
            "mean_ms": self.service_ms,
            "p99_ms": self.service_ms,
            "slo_violation_rate": 0.0,
            "queue_depth": self.queue.qsize(),
        }


def create_app(service_ms: float) -> FastAPI:
    server = Server(service_ms)
    app = FastAPI(title="Stub Saturated Server")

    @app.on_event("startup")
    async def _startup() -> None:
        await server.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await server.stop()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "kind": "stub_saturated_server", "service_ms": service_ms}

    @app.post("/v1/encode")
    async def encode(payload: EncodeRequest) -> dict:
        return await server.submit(payload)

    @app.post("/metrics/reset", status_code=204)
    async def reset() -> None:
        server.reset()

    @app.get("/metrics/summary")
    async def metrics() -> dict:
        return server.summary()

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="饱和排队替身服务（CPU，不占 GPU）")
    parser.add_argument("--service-ms", type=float, default=4.5,
                        help="单请求服务时间（毫秒）。容量 = 1000/它。图回放安全档实测容量"
                             "约 222 req/s，对应 4.5 ms。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--cpu-load", type=int, default=0,
        help="额外起几个忙循环线程（纯 Python 自旋）。用于验证一个假设：真机过载档上"
             "客户端失效，是因为**服务端自己的 CPU 负载**（预处理线程跑 torch.rand）"
             "把客户端的事件循环饿死，而不是客户端本身弱。",
    )
    args = parser.parse_args()
    if args.cpu_load:
        import threading
        stop = threading.Event()
        def spin() -> None:
            while not stop.is_set():
                for _ in range(50_000):
                    pass
                time.sleep(0.0002)
        for _ in range(args.cpu_load):
            threading.Thread(target=spin, daemon=True).start()
    print(f"替身服务：service_ms={args.service_ms}，单服务台容量 "
          f"{1000/args.service_ms:.1f} req/s；CPU 忙循环线程 {args.cpu_load} 个；不占 GPU",
          flush=True)
    uvicorn.run(create_app(args.service_ms), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
