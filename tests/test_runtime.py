from __future__ import annotations

import sys
import types

import pytest

import encoder_sched.runtime as runtime_module
from encoder_sched.config import AppConfig, RuntimeConfig
from encoder_sched.runtime import RuntimeValidationError, validate_server_runtime


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_capability(device):
        return (8, 9)

    @staticmethod
    def get_device_name(device):
        return "Test GPU"


def install_torch(monkeypatch, cuda_version="12.8", cuda=FakeCuda):
    torch = types.SimpleNamespace(
        __version__="2.8.0",
        version=types.SimpleNamespace(cuda=cuda_version),
        cuda=cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_linux_cuda_runtime_is_accepted(monkeypatch):
    install_torch(monkeypatch)
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    report = validate_server_runtime(AppConfig())
    assert report["gpu"] == "Test GPU"
    assert report["torch_cuda"] == "12.8"


def test_non_linux_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeValidationError, match="仅支持 Linux"):
        validate_server_runtime(AppConfig())


def test_cuda_runtime_above_server_limit_is_rejected(monkeypatch):
    install_torch(monkeypatch, cuda_version="14.0")
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    config = AppConfig(runtime=RuntimeConfig(max_cuda_major=13))
    with pytest.raises(RuntimeValidationError, match="超过配置允许"):
        validate_server_runtime(config)
