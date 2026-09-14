from __future__ import annotations

import inspect
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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

    def environment(self) -> dict[str, Any]:
        return {}


class FakeEncoderBackend(EncoderBackend):
    def __init__(self, resource_backend: ResourceBackend, latency_scale: float = 0.0001):
        self.resource_backend = resource_backend
        self.latency_scale = latency_scale

    def encode(self, job: EncodeJob, worker_id: int) -> EncodingResult:
        resource = self.resource_backend.apply_quota(worker_id, job.sm_fraction)
        delay_ms = max(0.05, job.predicted_ms * self.latency_scale)
        time.sleep(delay_ms / 1000)
        return EncodingResult(512, math.sqrt(512), delay_ms, resource)


class ClipEncoderBackend(EncoderBackend):
    def __init__(self, config: ModelConfig, resource_backend: ResourceBackend, stream_count: int):
        try:
            import torch
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise RuntimeError("真实 CLIP 后端需要安装 torch 和 transformers") from exc

        self.torch = torch
        self.config = config
        self.resource_backend = resource_backend
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但 torch.cuda.is_available() 为 False")
        self.device = torch.device(config.device)
        dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}.get(config.dtype)
        if dtype is None:
            raise ValueError(f"不支持的 dtype: {config.dtype}")
        if self.device.type == "cpu" and dtype != torch.float32:
            dtype = torch.float32
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
        self.streams = [torch.cuda.Stream() for _ in range(stream_count)] if self.device.type == "cuda" else [None] * stream_count
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)

    def _input(self, job: EncodeJob):
        generator = self.torch.Generator(device="cpu").manual_seed(job.seed)
        pixels = self.torch.rand((3, job.height, job.width), generator=generator, dtype=self.torch.float32)
        pixels = ((pixels - self.mean) / self.std).unsqueeze(0)
        return pixels.to(self.device, dtype=self.dtype, non_blocking=self.device.type == "cuda")

    def encode(self, job: EncodeJob, worker_id: int) -> EncodingResult:
        torch = self.torch
        stream = self.streams[worker_id % len(self.streams)]
        stream_handle = int(stream.cuda_stream) if stream is not None else worker_id
        resource = self.resource_backend.apply_quota(stream_handle, job.sm_fraction)
        pixels = self._input(job)
        with torch.inference_mode():
            if stream is None:
                started = time.perf_counter_ns()
                output = self.model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
                execution_ms = (time.perf_counter_ns() - started) / 1_000_000
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start_event.record(stream)
                    output = self.model(pixel_values=pixels, interpolate_pos_encoding=True).image_embeds
                    end_event.record(stream)
                end_event.synchronize()
                execution_ms = float(start_event.elapsed_time(end_event))
        vector = output.detach().float()
        return EncodingResult(
            embedding_dim=int(vector.shape[-1]),
            embedding_norm=float(torch.linalg.vector_norm(vector).item()),
            execution_ms=execution_ms,
            resource=resource,
        )

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
