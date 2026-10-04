# 多模态小请求并发推理原型

本项目面向原生 Linux + NVIDIA GPU 服务器，研究 VLM/TTS 小请求在共享 GPU 上的并发、
零等待组批和 SLO 感知调度。当前主实现是 CLIP ViT-B/32 视觉编码服务。

## 服务器要求

- 64-bit Linux，不支持 Windows/WSL 作为正式实验环境
- Python 3.10--3.13
- NVIDIA 驱动可支持 CUDA 13.0；PyTorch wheel 自带的 CUDA runtime 不得高于驱动能力
- 至少一张对当前进程可见的 NVIDIA GPU
- 正式实验前将模型权重放入服务器 Hugging Face 缓存

CUDA Toolkit、NVIDIA 驱动和 PyTorch CUDA runtime 是三件不同的东西。普通推理只要求
驱动与 PyTorch wheel 兼容；只有构建历史 libsmctrl 实验库时才需要 CUDA 头文件。

## 安装

推荐使用自动脚本。若虚拟环境尚无 PyTorch，需要显式指定官方 wheel 索引：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  scripts/bootstrap_server.sh
```

服务器已有合适的 PyTorch 环境时，也可以手工安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install torch --index-url <official-pytorch-index>
.venv/bin/python -m pip install -e '.[gpu,test,plot]'
```

不要根据系统显示的“CUDA 13.0”盲目安装同名 wheel。应按 PyTorch 官方支持矩阵选择版本，
再由严格预检确认 CUDA 可用。

## 配置

默认 `config.yaml` 是服务器配置：

- `runtime.require_linux: true`
- `runtime.require_cuda: true`
- `executor.resource_backend: none`
- `executor.allow_proxy_fallback: false`
- `model.local_files_only: true`

`none` 表示请求共享 GPU，不实施 SM 分区。这是当前小请求并发主实验的真实语义。`proxy`
只为读取历史实验配置保留，新配置不得使用它。`libsmctrl` 仅用于复现历史 TPC 掩码实验，
且禁止失败后回退。

上线前执行：

```bash
.venv/bin/encoder-sched-check --config config.yaml
```

检查失败会返回非零退出码。它在模型加载前验证 Linux、Python、PyTorch CUDA runtime、
GPU 可见性和 CUDA 主版本上界。

## 启动

```bash
.venv/bin/encoder-sched-server --config config.yaml --host 0.0.0.0 --port 8000
```

验证：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -X POST http://127.0.0.1:8000/v1/encode \
  -H 'content-type: application/json' \
  -d '{"request_id":"smoke-1","width":224,"height":224,"deadline_ms":1000}'
```

`/health` 会报告服务器运行时、模型环境、有效资源后端和图回放状态。API 接受的最大图像
尺寸为 1024×1024；是否真正支持某个尺寸仍取决于模型位置编码和所选执行后端。

## systemd 部署

仓库提供 `deploy/encoder-sched.service` 和 `deploy/encoder-sched.env.example`。示例安装路径为
`/opt/encoder-sched`，运行用户为 `encoder-sched`：

```bash
sudo install -d /etc/encoder-sched
sudo install -m 0644 deploy/encoder-sched.env.example /etc/encoder-sched/encoder-sched.env
sudo install -m 0644 deploy/encoder-sched.service /etc/systemd/system/encoder-sched.service
sudo systemctl daemon-reload
sudo systemctl enable --now encoder-sched
sudo systemctl status encoder-sched
```

安装前确认环境文件中的 `CUDA_VISIBLE_DEVICES`、`HF_HOME` 和配置路径。systemd 每次启动前
都会执行严格预检；模型缓存和 `results/` 必须对服务用户可读写。

## Slurm 集群

Slurm 环境不使用 systemd，也不在登录节点启动服务。仓库提供 module 初始化、共享存储接入、
GPU 环境安装与预检、`srun` 交互调试、单次实验、重复实验和作业内服务模板。使用顺序与参数说明见
[`slurm/README.md`](slurm/README.md)。

## 实验入口

```bash
.venv/bin/python scripts/profile_encoder.py --config config.yaml
.venv/bin/python scripts/generate_workload.py --count 1000 --pattern mixed --slo normal
.venv/bin/python scripts/run_experiment.py --workload data/workloads/mixed.jsonl
.venv/bin/python scripts/run_repeated_comparison.py --group concurrency --repeats 5
```

正式实验要求开放环负载、每臂至少 5 次重复、轮次交替和原始逐请求数据落盘。
服务端与客户端延迟必须分列；至少报告吞吐率、平均延迟、P99、GPU 利用率和 SLO 违约率。

## 其他配置

- `config.graph.yaml`：固定 shape 的 CUDA Graph 回放流水线
- `config.adaptive.yaml`：历史自适应并发控制实验，仅用于复现负结果
- `config.libsmctrl.example.yaml`：历史 TPC 掩码实验，需补丁版 libsmctrl

CUDA Graph 与 libsmctrl 互斥；图回放的 `slots`、捕获尺寸和 batch 档位必须在启动前固定。

## 测试

```bash
.venv/bin/python -m pytest -q
```

无 GPU 的 CI 可以运行单元测试和 FakeEncoder 链路；真实服务启动不会绕过 Linux/CUDA 预检。
