from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, model_validator

from .config import AppConfig, load_config
from .dacc import DaccConfig
from .encoder import ClipEncoderBackend, FakeEncoderBackend
from .metrics import MetricsStore
from .models import EncodeJob
from .performance import PerformanceModel
from .resource import create_resource_backend
from .scheduler import Scheduler
from .service import DuplicateRequestError, EncoderService, JobNotFoundError


class EncodeRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(default=0, ge=0)
    width: int = Field(default=224, ge=64, le=672)
    height: int = Field(default=224, ge=64, le=672)
    deadline_ms: float = Field(default=1000, gt=0, le=120_000)
    priority: int = Field(default=0, ge=-100, le=100)

    @model_validator(mode="after")
    def validate_patch_grid(self) -> "EncodeRequest":
        if self.width < 32 or self.height < 32:
            raise ValueError("图像尺寸必须至少包含一个 32x32 patch")
        return self


class EncodeResponse(BaseModel):
    request_id: str
    status: str
    width: int
    height: int
    patches: int
    deadline_ms: float
    priority: int
    predicted_ms: float
    sm_fraction: float
    queue_ms: float | None
    execution_ms: float | None
    total_ms: float | None
    embedding_dim: int | None
    embedding_norm: float | None
    slo_violated: bool | None
    error: str | None
    metadata: dict[str, Any]


def _ensure_libsmctrl_target(config: AppConfig, fake: bool) -> None:
    """libsmctrl 只能作用于真实 CUDA stream，启动阶段就拒绝错误组合。

    ``libsmctrl_set_stream_mask`` 会把传入句柄当作 ``CUstream*`` **解引用**后按
    偏移改写驱动内部结构。FakeEncoder 和 CPU 路径传的是 worker id，把它当成
    stream 传入等于向任意地址写入。这里选择直接报错而不是自动改成 proxy，
    因为静默降级会让配置错误一直隐藏到实验结果里。
    """
    if config.executor.resource_backend != "libsmctrl":
        return
    if fake:
        raise ValueError(
            "resource_backend=libsmctrl 不能与 FakeEncoder 同用：FakeEncoder 传的是 worker id，"
            "不是 CUDA stream 句柄。如需验证软件链路，请显式改用 resource_backend=proxy。"
        )
    if config.model.device != "cuda":
        raise ValueError(
            f"resource_backend=libsmctrl 要求 model.device=cuda，当前为 {config.model.device!r}；"
            "CPU 没有 CUDA stream，无法承载 TPC 掩码。"
        )


def build_service(config: AppConfig, fake: bool = False) -> EncoderService:
    performance = PerformanceModel(config.resolve(config.profiling.table_path))
    scheduler = Scheduler(
        config.scheduler.policy,
        performance,
        config.scheduler.quota_levels,
        config.scheduler.deadline_tie_ms,
        DaccConfig(window=config.scheduler.dacc_window, beta=config.scheduler.dacc_beta, guard_ms=config.scheduler.dacc_guard_ms),
    )
    _ensure_libsmctrl_target(config, fake)
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    encoder = (
        FakeEncoderBackend(resource)
        if fake
        else ClipEncoderBackend(config.model, resource, config.executor.streams)
    )
    log_dir = config.resolve(config.logging.output_dir)
    metrics = MetricsStore(log_dir / config.logging.request_log)
    return EncoderService(config, scheduler, encoder, metrics, resource)


def create_app(config_path: str | Path | None = None, fake: bool | None = None) -> FastAPI:
    path = config_path or os.getenv("ENCODER_SCHED_CONFIG", "config.yaml")
    use_fake = os.getenv("ENCODER_SCHED_FAKE", "0") == "1" if fake is None else fake

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_config(path)
        service = build_service(config, use_fake)
        app.state.service = service
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    application = FastAPI(title="Encoder Scheduler", version="0.1.0", lifespan=lifespan)

    @application.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return request.app.state.service.health()

    @application.post("/v1/encode", response_model=EncodeResponse)
    async def encode(payload: EncodeRequest, request: Request) -> dict[str, Any]:
        job = EncodeJob(**payload.model_dump())
        try:
            completed = await request.app.state.service.submit(job)
        except DuplicateRequestError:
            raise HTTPException(status.HTTP_409_CONFLICT, "request_id 已存在")
        if completed.error:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, completed.public_dict())
        return completed.public_dict()

    @application.get("/v1/jobs/{request_id}", response_model=EncodeResponse)
    async def get_job(request_id: str, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.service.get_job(request_id).public_dict()
        except JobNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

    @application.get("/metrics/summary")
    async def metrics(request: Request) -> dict[str, Any]:
        return request.app.state.service.metrics.summary()

    @application.post("/metrics/reset", status_code=status.HTTP_204_NO_CONTENT)
    async def reset_metrics(request: Request) -> None:
        request.app.state.service.metrics.reset()

    return application


app = create_app()


def main() -> None:
    uvicorn.run("encoder_sched.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
