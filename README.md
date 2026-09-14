# 多模态编码器并发调度原型

本项目实现毕业设计的最小可运行架构：CLIP ViT 视觉编码、FastAPI 请求入口、EDF-size 调度、多 CUDA Stream、可插拔 SM 资源控制以及实验指标闭环。

## 安装

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[test,plot]"
# 按 PyTorch 官方 CUDA 版本安装 torch 后：
.venv\Scripts\python -m pip install "transformers>=4.46,<5"
```

CUDA Toolkit 与 PyTorch wheel 的 CUDA 运行时不是一回事。先根据 PyTorch 官方支持矩阵选择 wheel，再安装与 `libsmctrl` 构建要求匹配的 Toolkit。Windows 仅使用代理配额；真实 SM 控制在 Linux 验证。

当前 `config.yaml` 使用 `local_files_only: true`，适合已完成首次下载的本机。新环境首次下载模型时暂时改为 `false`，下载完成后恢复为 `true`，以避免正式实验受网络状态影响。

## 运行

```powershell
.venv\Scripts\python scripts/probe_environment.py
.venv\Scripts\python scripts/profile_encoder.py
.venv\Scripts\python -m encoder_sched.api
```

另开终端生成并回放负载：

```powershell
.venv\Scripts\python scripts/generate_workload.py --count 100 --pattern mixed --slo normal
.venv\Scripts\python scripts/run_experiment.py --workload data/workloads/mixed.jsonl
.venv\Scripts\python scripts/summarize_results.py results/experiment.jsonl
```

无 GPU 或模型尚未下载时，可用假编码器验证系统链路：

```powershell
$env:ENCODER_SCHED_FAKE="1"
.venv\Scripts\python -m encoder_sched.api
```

## API

- `POST /v1/encode`：提交请求并等待编码完成。
- `GET /v1/jobs/{request_id}`：查询任务状态或结果。
- `GET /metrics/summary`：获取吞吐、延迟分位数、GPU 利用率和 SLO 违约率。
- `POST /metrics/reset`：暖机后重置当前实验的内存指标，不删除原始 JSONL 日志。
- `GET /health`：检查调度器、worker 和模型环境。

OpenAPI 页面位于 `http://127.0.0.1:8000/docs`。

## 资源控制边界

`proxy` 后端只记录调度器给出的 SM 配额并控制并发，不声称实现硬件隔离。

Linux 上的真实 SM/TPC 控制对接上游库 [libsmctrl](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git)
（Bakita & Anderson, *Hardware Compute Partitioning on NVIDIA GPUs*, RTAS 2023），适配器为
`encoder_sched/libsmctrl_adapter.py`，经 ctypes 调用 `libsmctrl.so`。注意
`github.com/atomicapple0/libsmctrl` 只是该论文的冻结快照 fork，支持的 CUDA 版本更旧，应使用上面的原仓库。

### 接入步骤

```bash
scripts/build_libsmctrl.sh     # 克隆上游并 make libsmctrl.so；库本身只需 gcc，不需要 nvcc
export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
python scripts/probe_libsmctrl.py --config config.libsmctrl.example.yaml
ENCODER_SCHED_CONFIG=config.libsmctrl.example.yaml python -m encoder_sched.api
```

### 关键限制：CUDA 13 上调用该库会终止进程

`libsmctrl_set_stream_mask()` 的返回类型是 `void`，其实现是按**驱动版本硬编码的字节偏移**直接改写
驱动内部 stream 结构。驱动版本不在其白名单（x86_64 分支止于 CUDA 12.8）时，上游实现走到：

```c
abort(1, 0, "Stream masking unsupported on this CUDA version (%d), and no fallback MASK_OFF set!", ver);
```

上游把 `abort` 定义成 `error_at_line`（源码注释：*"we favor terminating with an error rather than merely
printing a warning and continuing"*），而 `error_at_line` 在第一个参数非零时会 **`exit(1)` 终止整个进程**。
所以在这个环境里调用它，得到的是**服务进程被杀**，而不是一个错误码或异常。

*（`atomicapple0/libsmctrl` 那个 RTAS23 快照 fork 在此处是静默 `return`，什么都不做也不报错。本项目使用
上游原仓库，行为是 `exit(1)`。）*

因此适配器先读 `cuDriverGetVersion` 与白名单比对；不受支持时返回 `enforced=False` 并**不把控制权交给
原生函数**。这既是防止伪造结果，也是防止进程被终止。本机实测驱动为 CUDA 13.3（`13030`），探针输出为：

```json
{ "available": false, "stage": "driver_version",
  "reason": "驱动报告的 CUDA 版本为 13.3（13030），不在 libsmctrl set_stream_mask 的受支持白名单内；上游实现在该分支会调用 exit(1) 终止进程，因此拒绝下发真实掩码" }
```

在该环境下**不得**把任何调度结果表述为 SM 隔离效果；要获得真实 TPC 掩码需要一台驱动受支持的
Linux 机器（CUDA ≤ 12.8）。上游 README 提供了用 `MASK_OFF` 环境变量探测新版本偏移的流程，但那会按
猜测的偏移写入驱动内部结构，属于危险实验，本项目不默认启用。

### 掩码语义与配额量化

上游约定**置位表示禁用 TPC**，启用 TPC `0..n-1` 需传 `~((1 << n) - 1)`。配额按 TPC 粒度量化：
RTX 4060 为 24 SM、每 TPC 2 SM，共 **12 个 TPC**，因此 0.25/0.5/0.75/1.0 恰好对应 3/6/9/12 个 TPC。
实验中应记录 `effective_sm_fraction` 而非请求值。

两个并发流会被分配**互不重叠**的 TPC 区间；配额之和超过设备 TPC 总数时显式失败，而不是给出重叠
分配。掩码重叠的分配看起来像"各占一半"，但两个流实际使用同一批 TPC，不构成空间隔离。

### 已实现与尚未验证的边界

由于 `libsmctrl_set_stream_mask` 返回 void，适配器只能确认"已按受支持版本下发"，无法确认硬件是否
真的生效（返回值中 `verified_by_measurement` 恒为 `false`）。真实隔离结论必须来自独立测量：固定
工作负载、多档配额、双流对照实验。上游自带的 `libsmctrl_test_stream_mask` 可作交叉验证，但需要 nvcc。

`libsmctrl` 会把传入句柄当作 `CUstream*` 解引用，因此 FakeEncoder 的 worker id 绝不能进入原生调用；
`build_service()` 会在启动阶段直接拒绝 `libsmctrl` 与 Fake/CPU 的组合，而不是静默降级为 proxy。

`data/profiles/default.csv` 是启动调度器用的先验估计。`profile_encoder.py` 在代理后端只输出
`sm_fraction=1.0` 的真实测量；只有资源后端确认 `enforces_sm_partition=true`（即能力探针通过）时才会
遍历全部配额，防止把逻辑配额误当成硬件实验结果。
