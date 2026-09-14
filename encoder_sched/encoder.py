from __future__ import annotations

import inspect
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from .config import ModelConfig
from .models import EncodeJob
from .resource import ResourceBackend


@dataclass(frozen=True)
class EncodingResult:
    embedding_dim: int
    embedding_norm: float
    execution_ms: float
    resource: dict[str, Any]


class EncoderBackend(ABC):
    @abstractmethod
    def encode(self, job: EncodeJob, worker_id: int) -> EncodingResult:
        raise NotImplementedError

    def encode_batch(self, jobs: Sequence[EncodeJob], worker_id: int) -> list[EncodingResult]:
        """一次处理多个请求。

        默认退化为逐请求执行；真正支持批处理的后端覆盖它。调用方不应假设
        ``encode_batch`` 一定比逐个 ``encode`` 快——这正是本课题要测的东西。
        """
        return [self.encode(job, worker_id) for job in jobs]

    def environment(self) -> dict[str, Any]:
        return {}


class FakeEncoderBackend(EncoderBackend):
    def __init__(self, resource_backend: ResourceBackend, latency_scale: float = 0.0001):
        self.resource_backend = resource_backend
        self.latency_scale = latency_scale

    #: 批处理效率：batch 从 1 增到 N 时，总时长按 1+(N-1)*this 增长。
    #: 0.3 表示第二个请求只多花 30% 时间（硬件填空），远优于串行的 2 倍。
    BATCH_MARGINAL = 0.3

    def encode(self, job: EncodeJob, worker_id: int) -> EncodingResult:
        return self.encode_batch([job], worker_id)[0]

    def encode_batch(self, jobs: Sequence[EncodeJob], worker_id: int) -> list[EncodingResult]:
        if not jobs:
            return []
        resource = self.resource_backend.apply_quota(worker_id, jobs[0].sm_fraction)
        try:
            base_ms = max(0.05, max(job.predicted_ms for job in jobs) * self.latency_scale)
            delay_ms = base_ms * (1.0 + (len(jobs) - 1) * self.BATCH_MARGINAL)
            time.sleep(delay_ms / 1000)
        finally:
            self.resource_backend.release_quota(worker_id)
        return [EncodingResult(512, math.sqrt(512), delay_ms, resource) for _ in jobs]


class ClipEncoderBackend(EncoderBackend):
    """真实 CLIP 视觉编码后端。**仅支持 CUDA**。

    曾支持 `device: cpu`，但该分支从未被真正执行过——测试夹具虽然写过 `device: cpu`，
    所有调用却都走 `fake=True`，`ClipEncoderBackend` 根本不会被构造；也没有任何脚本用它
    跑真实推理。而且 CPU 推理对延迟与配额研究毫无意义（本课题测量的是 GPU 调度）。
    保留一条永不被验证的代码路径只会增加维护面，因此删除。

    需要无 GPU 环境验证队列/调度/指标链路时，用 `FakeEncoderBackend`（`ENCODER_SCHED_FAKE=1`）。
    """

    def __init__(self, config: ModelConfig, resource_backend: ResourceBackend, stream_count: int):
        try:
            import torch
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise RuntimeError("真实 CLIP 后端需要安装 torch 和 transformers") from exc

        self.torch = torch
        self.config = config
        self.resource_backend = resource_backend
        if not torch.cuda.is_available():
            raise RuntimeError("真实 CLIP 后端需要 CUDA，但 torch.cuda.is_available() 为 False")
        self.device = torch.device("cuda")
        dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(config.dtype)
        if dtype is None:
            raise ValueError(f"不支持的 dtype: {config.dtype}")
        self.dtype = dtype
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            config.name,
            local_files_only=config.local_files_only,
            dtype=dtype,
        ).eval().to(self.device)
        signature = inspect.signature(self.model.forward)
        self.supports_interpolation = "interpolate_pos_encoding" in signature.parameters
        if not self.supports_interpolation:
            raise RuntimeError("当前 transformers 的 CLIP 模型不支持可变分辨率位置编码插值")
        self.streams = [torch.cuda.Stream() for _ in range(stream_count)]
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)

    def _input(self, job: EncodeJob):
        # 输入在 CPU 上用固定种子生成再搬上 GPU——这是所有路径共用的做法，不是"CPU 分支"。
        generator = self.torch.Generator(device="cpu").manual_seed(job.seed)
        pixels = self.torch.rand((3, job.height, job.width), generator=generator, dtype=self.torch.float32)
        pixels = ((pixels - self.mean) / self.std).unsqueeze(0)
        return pixels.to(self.device, dtype=self.dtype, non_blocking=True)

    def encode(self, job: EncodeJob, worker_id: int) -> EncodingResult:
        return self.encode_batch([job], worker_id)[0]

    def encode_batch(self, jobs: Sequence[EncodeJob], worker_id: int) -> list[EncodingResult]:
        """把若干**同尺寸**请求拼成一个 batch，一次前向完成。

        与逐请求执行（batch=1）的区别是本课题的核心对照之一：

        - 批处理让硬件自己在同一批 SM 上交错调度各请求的 block，能填掉单个 SM 内部的
          空隙，因此**吞吐更高**；代价是请求要等组批，**首字节延迟变高**。
        - 空间分割（每请求一个线程 + 独立 TPC 区间）不需要等待，但划分之后 SM 内部不再有
          跨请求填空的机会，**吞吐低一些**。

        两者是"用等待换效率"与"用效率换即时性"的关系，而 SLO/尾延迟正是衡量"等待"的指标。

        要求所有请求尺寸一致——批处理必须是同形状张量。异构分辨率需要按尺寸分桶，
        这一点由上层调度器负责（见 EncoderService 的批处理 worker）。
        """
        if not jobs:
            return []
        sizes = {(job.width, job.height) for job in jobs}
        if len(sizes) != 1:
            raise ValueError(f"批处理要求所有请求同尺寸，收到 {sorted(sizes)}")

        torch = self.torch
        stream = self.streams[worker_id % len(self.streams)]
        stream_handle = int(stream.cuda_stream)
        # 同一批共享一次配额：批内各请求不被分别限制 SM
        resource = self.resource_backend.apply_quota(stream_handle, jobs[0].sm_fraction)
        try:
            pixels = torch.cat([self._input(job) for job in jobs], dim=0)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode(), torch.cuda.stream(stream):
                start_event.record(stream)
                output = self.model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
                end_event.record(stream)
            end_event.synchronize()
            execution_ms = float(start_event.elapsed_time(end_event))
        finally:
            # 请求结束就释放配额；否则空闲线程会继续占着 TPC 分配，挡住后续请求。
            self.resource_backend.release_quota(stream_handle)

        vector = output.detach().float()
        results: list[EncodingResult] = []
        for index in range(len(jobs)):
            row = vector[index]
            results.append(
                EncodingResult(
                    embedding_dim=int(row.shape[-1]),
                    embedding_norm=float(torch.linalg.vector_norm(row).item()),
                    # 批内各请求同时完成，因此共享同一个执行时长
                    execution_ms=execution_ms,
                    resource=resource,
                )
            )
        return results

    def environment(self) -> dict[str, Any]:
        torch = self.torch
        result = {
            "model": self.config.name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        if self.device.type == "cuda":
            result.update(
                gpu=torch.cuda.get_device_name(self.device),
                compute_capability=".".join(map(str, torch.cuda.get_device_capability(self.device))),
            )
        return result
