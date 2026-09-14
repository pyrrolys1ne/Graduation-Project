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

## 实验脚本

| 脚本 | 用途 |
|---|---|
| `scripts/generate_workload.py` | 生成请求轨迹（uniform / burst / mixed，紧/普通/宽松 SLO） |
| `scripts/profile_encoder.py` | 单卡剖析，产出 `data/profiles/*.csv` |
| `scripts/simulate_policies.py` | 离线模拟对照（确定性，无统计效力，仅作快速检查） |
| `scripts/run_experiment.py` | 回放一份负载并保存 JSONL 与汇总 |
| `scripts/run_real_comparison.py` | 五种策略各跑一次真实 GPU 对照 |
| `scripts/run_repeated_comparison.py` | **重复对照与 DACC 消融**（每组重启服务、重复 N 次、报告标准差） |
| `scripts/experiment_sm_quota.py` | **真实 SM/TPC 配额实验**（需补丁库；本机已验证可用） |
| `scripts/summarize_results.py` | 汇总与绘图 |

重复对照支持三组：

```bash
python scripts/run_repeated_comparison.py --group baselines --repeats 5
python scripts/run_repeated_comparison.py --group ablations --repeats 5   # DACC 消融
python scripts/run_repeated_comparison.py --group window    --repeats 5   # K=1/4/8/16
```

消融通过配置覆盖实现，不需要改代码：

```yaml
scheduler:
  policy: dacc
  dacc_overrides:
    w_complementarity: 0.0   # 字段名必须在 DaccConfig 中存在，拼错会直接报错
```

`metrics.gpu_sample_interval_s` 控制 GPU 利用率采样间隔。`nvidia-smi` 返回的是瞬时快照而非区间均值，默认 0.1s；间隔过大时秒级实验只能采到个位数样本，均值不具代表性。

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
python scripts/experiment_sm_quota.py --config config.libsmctrl.example.yaml --repeats 5
```

`experiment_sm_quota.py` 做两件事：单线程固定配额曲线（`sm_fraction—latency`），以及双线程"各自单独执行 vs 同时执行"对照，并判定两个并发请求拿到的 TPC 区间是否互斥。探针不通过时它会直接拒绝运行——不会用代理数据冒充隔离结果。

### 两套机制：只有回调路径可用

libsmctrl 提供**两套互不相关的机制**，选错会直接杀掉服务进程：

| 机制 | 代表函数 | 版本门控 | CUDA 13.3 |
|---|---|---|---|
| 硬编码结构体偏移 | `set_stream_mask` | 按驱动版本白名单，未知版本 `exit(1)` | ❌ 会**终止进程** |
| **启动回调** | `set_global_mask` / `set_next_mask` | 只有下界（<6.5），**运行时读 TMD 版本自适应** | ✅ 实测生效 |

回调路径能跨驱动版本工作，**恰恰因为它不硬编码版本**——它读运行时 TMD 版本字段决定写入偏移
（`>=0x40` 走 Hopper 的 +304/+308，否则 +84/+88）。

`set_stream_mask` 则在未知驱动版本时执行 `abort(1, 0, ...)`；上游把 `abort` 定义为 `error_at_line`，
第一个参数非零即 **`exit(1)` 终止整个进程**——不是返回错误码，而是杀掉调用者。

### 为什么需要补丁

上游 `set_next_mask` 只覆盖**一次**启动，而一次 CLIP 前向要发几十个 kernel，无法表达"按请求配额"。
`patches/libsmctrl-thread-mask.patch`（约 6 行）新增**粘性线程掩码** `libsmctrl_set_thread_mask()`：
设一次即对该线程后续所有启动生效。

**与项目结构天然契合**：`service.py` 用 `asyncio.to_thread(self.encoder.encode, ...)` 把每个请求丢进
独立线程，`encode()` 在同一线程内先调 `apply_quota` 再跑模型——**线程掩码就等于请求掩码**。

`demos/libsmctrl_thread_mask_demo.py` 给出对照：三种方式都只布防一次、连跑 20 次启动，
`set_next_mask` 覆盖 1/20 次，`set_thread_mask` 覆盖 **20/20** 次。

### 探针与安全边界

`build_libsmctrl.sh` 会自动应用补丁；用未打补丁的上游库时探针会明确报错（缺 `libsmctrl_set_thread_mask`）。

回调注册失败会 `exit(1)`，因此探针在**子进程**中试调一次 `set_thread_mask(0)`，把"可能杀进程"
变成"报告不可用"。探针未通过时 `apply_quota` **不把控制权交给原生函数**——已验证此路径下原生
调用次数为 0。

### 掩码语义与配额量化

上游约定**置位表示禁用 TPC**，启用 TPC `0..n-1` 需传 `~((1 << n) - 1)`。配额按 TPC 粒度量化：
RTX 4060 为 24 SM、每 TPC 2 SM，共 **12 个 TPC**，因此 0.25/0.5/0.75/1.0 恰好对应 3/6/9/12 个 TPC。
实验中应记录 `effective_sm_fraction` 而非请求值。

两个并发流会被分配**互不重叠**的 TPC 区间；配额之和超过设备 TPC 总数时显式失败，而不是给出重叠
分配。掩码重叠的分配看起来像"各占一半"，但两个流实际使用同一批 TPC，不构成空间隔离。

### 已实现与尚未验证的边界

掩码下发函数返回 `void`，适配器只能确认"已下发"，无法确认硬件是否真的生效（返回值中
`verified_by_measurement` 恒为 `false`）。机制层面的证据来自 `demos/libsmctrl_thread_mask_demo.py`
（延迟随启用 TPC 数缩放）；效果层面的结论必须来自独立测量：固定工作负载、多档配额、双线程对照。

`build_service()` 会在启动阶段直接拒绝 `libsmctrl` 与 Fake/CPU 的组合，而不是静默降级为 proxy。

`data/profiles/default.csv` 是启动调度器用的先验估计。`profile_encoder.py` 在代理后端只输出
`sm_fraction=1.0` 的真实测量；只有资源后端确认 `enforces_sm_partition=true`（即能力探针通过）时才会
遍历全部配额，防止把逻辑配额误当成硬件实验结果。

**TPC 隔离不等于零干扰。** 实测两个并发请求即使拿到互斥的 TPC 集合，仍有约 1.8–2.3 倍的 slowdown——
L2 与显存带宽是共享的，TPC 掩码挡不住。报告中应如实区分"TPC 集合互斥"与"性能互不影响"。
