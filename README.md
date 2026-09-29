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

重复对照支持四组：

```bash
python scripts/run_repeated_comparison.py --group baselines   --repeats 5
python scripts/run_repeated_comparison.py --group ablations   --repeats 5   # DACC 消融
python scripts/run_repeated_comparison.py --group window      --repeats 5   # K=1/4/8/16
python scripts/run_repeated_comparison.py --group concurrency --repeats 5   # 并发度控制
```

消融通过配置覆盖实现，不需要改代码：

```yaml
scheduler:
  policy: dacc
  dacc_overrides:
    w_complementarity: 0.0   # 字段名必须在 DaccConfig 中存在，拼错会直接报错
```

`metrics.gpu_sample_interval_s` 控制 GPU 利用率采样间隔。`nvidia-smi` 返回的是瞬时快照而非区间均值，默认 0.1s；间隔过大时秒级实验只能采到个位数样本，均值不具代表性。

### 并发度控制（`--group concurrency`）

并发度在默认配置下是**启动时定死的**（`executor.streams`，默认 2）。开启
`executor.adaptive_concurrency` 后，改由 `ConcurrencyController`
（`encoder_sched/concurrency.py`）**逐请求**决定此刻允许多少请求共驻：

| 队列状态 | 共驻度 | 依据 |
|---|---|---|
| 空 | 用整卡 | 无竞争者（Bless 式"空闲时用整卡"） |
| 小请求（patch ≤ 100）为主 | 提高 | 小请求不随配额缩放（224 为 1.34×），SM 空转多 |
| 大请求（patch ≥ 196）为主 | 降低 | 大请求随配额显著缩放（448 为 2.45×），自己就吃满 SM |
| 实测/预测 > `drift_threshold` | 保守值 | 标定表在该尺寸上不可信，**不再据表决策**（本课题特色项） |

控制信号全部取自**队列自身**（积压、到达率、尺寸混合），不需要 root / ncu / DCGM。

> ⚠️ **上表的控制律已被本课题实测证伪（2026-09-22），保留仅为记录设计意图。**
> `results/concurrency_openloop_2026-09-22/`：7 臂 × 5 轮开环对照，`dynamic` 被最佳静态档
> 稳定支配（5 轮同向，50.6 对 114.3 req/s）。归因见 `docs/实验记录.md` §22，要点三条：
> ① 本机每种尺寸的最优共驻数都是 **2**，"小请求应上调并发"方向相反（6 档吞吐只有 2 档的 43%）；
> ② 控制信号**不是负载的函数**——同一负载下只把 CUDA 流由 2 条换成 8 条，主决策就从
> 80.9% small_mix 翻成 58.2% large_mix；且本负载小请求占比恰好 0.50，正压在阈值上；
> ③ 并发度超过 2 后的塌陷来自**主机侧 kernel 发射（Python 线程/GIL）**，不是 GPU 争用
> （去掉 CPU 造图不变；N≥3 时主机发射耗时 ≈ GPU span；换成进程后吞吐恒定）。
> 因此不要再调这两个档位；若继续并发方向，先解决每请求的 Python 层发射开销，
> 或改做批处理对照（批处理省掉 N 份发射开销，见 §7）。

**该组的对照臂**是 `static_1/2/3/4/6/8`（固定并发度扫描）与 `dynamic`，
判据为 `dynamic` 的**逐轮配对差**优于**最佳静态档**——只看"动态 vs 固定 2"
无法区分"动态控制有效"与"并发 4 恰好更好"。示例配置见 `config.adaptive.yaml`。

**客户端必须开环**。`--concurrency` 是客户端连接上限，默认 `0` = 取负载请求条数，
即客户端绝不因为上一次响应没回来而推迟下一次发送。这一条不是调参，而是结论有效性的前提：
上限固定为 8 时，某个臂的吞吐会被钉在 `8 / 服务端延迟` 上，与调度器无关，所谓"饱和负载"
从未压到服务端（本项目 2026-09-22 实测过，证据见 `results/concurrency_client_throttled_2026-09-22/`）。
命中该陷阱时脚本会**抑制策略判定**并提示提高上限。

判定规则变更后可重判已落盘批次，不必重跑 GPU：

```bash
python scripts/run_repeated_comparison.py --group concurrency \
  --reanalyze results/concurrency_openloop_2026-09-22
```

⚠️ **跑之前先确认环境**。每个正式样本前，脚本会运行一个关闭自适应、单 worker、
客户端并发为 1 的哨兵，并按尺寸计算 `execution_ms / predicted_ms`；最差尺寸的中位倍率
超过 1.5× 时会打印“环境污染”并**抑制策略判定**。正式高并发臂的墙钟不参与污染判断，
因为它包含策略自身的共驻争用。
原因：WSL2 与 Windows 共享 GPU，**宿主进程在 WSL 的 `nvidia-smi` 里看不见**，
只看利用率不足以判断整机是否空闲。

预检不必随整批实验一起跑，可以**单独执行**并在环境不合格时直接中止：

```bash
python scripts/run_repeated_comparison.py --group concurrency --sentinel-only --sentinel-repeats 3
```

它按哨兵配置**去重**（concurrency 组的 7 个臂共享 1 个哨兵），结果写
`results/sentinel_<group>/sentinel_check.json`，任一尺寸超阈值时退出码为 1。
本机实测单次预检可在 90 秒内翻转判定（1.34× → 1.58×，最差尺寸均为 448），
所以**至少跑 3 次**再下结论——单次预检是抽一次签，不是测一个状态。

重复实验按**轮次优先**执行，并在各轮循环轮转臂顺序，避免 GPU 随时间变化与某个策略名称
绑定。**项目现状与下一步见 `docs/实验历程.md`**（一页纸入口）。

### 延迟口径：`server_*` 与 `client_*` 不可混用

重复对照的汇总把两个口径**分别命名、同表并列**，因为二者在本机相差 **15–34 倍**，
历史上正是把它当成同一个数，才让"多流保护尾延迟"的结论必须撤回。

| 口径 | 字段前缀 | 起算点 | 含什么 |
|---|---|---|---|
| 服务端 | `server_*` | 请求到达服务端 | 排队 + 执行；**不含**网络与连接池等待 |
| 客户端 | `client_*` | 客户端发出请求前 | 连接池等待 + HTTP 往返 + 服务端时间 |

判定用的两个派生量：

- `client_overhead_ms` = `client_latency_ms - total_ms`，即服务端之外的部分。它包含不可避免的
  往返，所以**不能**直接当排队读；`client_pool_wait_ms` 是超出本轮最小 overhead（视为纯往返
  基准）的部分，即连接池等待的估计。
- `client_dispatch_lateness_ms` = 客户端**实际发出**时刻 − 负载规定的到达时刻。它在 `post`
  之前取样，量的是事件循环自身的积压，**不含**连接池等待。

`client_queueing_flagged` 为真表示端到端延迟里有可观比例是客户端自己造成的（连接池等待
超过客户端 P95 的 10%，或发出迟到超过抖动量级）——此时不能用这些数字论证服务端尾延迟，
应先加大 `--concurrency` 或减少宿主侧 GPU 活动后重跑。

汇总里缺少 `client_scope` 块会直接报错而不是退回单口径：静默退回正是混用的成因。
SLO 违约率同样分两侧：`slo_violation_rate` 按服务端 `total_ms` 判定，
`client_slo_violation_rate` 按客户端实际延迟判定，论文里引用哪一个必须写明。

### 图回放流水线（`executor.graph_pipeline`）

**2026-09-26 新增。** 依据与全部数字见 `docs/实验记录.md` §23。

背景：§22 已确证并发度超过 2 之后的塌陷**不是** GPU 争用，而是**每请求的主机侧
kernel 发射**（N≥3 时发射耗时 ≈ GPU span）。图回放把发射路径换掉：

| 臂（无服务探针，尺寸 224/336/448/672 均匀混合） | 主机发射 | 吞吐 |
|---|---:|---:|
| eager，N=2（峰值） | 8.64 ms | 173.6 req/s |
| eager，N=4 | 43.38 ms | 84.4 req/s |
| **图回放，N=2** | **0.04 ms** | **348.6 req/s** |
| 图回放流水线（含真实 CPU 造图） | — | 354.0 req/s |

服务级同夹具单次筛选（600 请求、开环、同种子）：eager `streams=2` **129.3 req/s**
→ 图回放 `slots=2` **289.1 req/s**。图回放与 eager 的输出**逐元素差 0.000e+00**。

```bash
ENCODER_SCHED_CONFIG=config.graph.yaml .venv-linux/bin/python -m encoder_sched.api
```

`encoder_sched/graph_runtime.py` 把处理拆成**两级线程池**，这是策略本身而不是实现细节：

| 级 | 线程 | 只做什么 | 为什么必须分开 |
|---|---|---|---|
| 预处理 | `graph.prep_threads` | CPU 造图 → 归一化 → 写**锁页**缓冲 | 与图回放同线程会**挂死** |
| 回放 | `graph.slots` | 锁页 → 显存 → 图静态输入 → 回放 | 与 CPU 造图同线程会**挂死** |

**两条硬约束**（违反时形态相同：所有线程卡在 `stream.synchronize()`，GPU 永不完结）：

1. **H2D 的源必须锁页**。分页源 H2D 与另一条流上的图回放并发时会互等。
   `ClipEncoderBackend._input` 是分页源，所以图路径不复用它，而在 `prepare` 里
   写进锁页缓冲。
2. **CPU 造图不得与图回放共处一个线程**（主要约束，锁页替代不了这一条）。
   另：回放后的后处理（模长）必须留在回放流内部，落到默认流上一样会互等。

三条互斥关系由配置加载阶段拦截，不会静默降级：`graph_pipeline` 与
`resource_backend=libsmctrl`（掩码在 `cuGraphLaunch` 上不生效，§10.7）、
与 `adaptive_concurrency`（并发度只能有一个来源）、与 `batching`（本版图不含批维）互斥；
且 `graph.slots` 不得大于 `executor.streams`。

`graph.sizes` 留空（`sizes: []`）时从剖析表推导本机跑过哪些尺寸；图**必须在启动阶段
捕获**（中途捕获要求显存上没有在飞的图），因此落在集合外的尺寸会让该请求失败并
明确报错，而不是静默回退到 eager。

## 资源控制边界

`proxy` 后端**零副作用**：它只把调度器给出的 SM 配额原样记录下来返回，**既不改变执行、也不控制并发**。并发度由执行器的 worker 数决定（`executor.streams`，或在开启 `executor.adaptive_concurrency` 后由 `ConcurrencyController` 逐请求决定），与资源后端无关。因此**任何涉及 `sm_fraction` 效力的对照都必须在 libsmctrl 后端下做**——在 `proxy` 下做等于测空气（本项目已有一次这样的无效对照被作废）。

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

**TPC 隔离不等于隔离 L2/HBM。** 早期“互斥 TPC 仍有 1.8–2.3× slowdown”的结论已撤回：
独跑基准和并发测量使用了不同 TPC 位置，把位置差异误读成争用。同一掩码位置重测后，
compute+compute / compute+memory / memory+memory 的 slowdown 约为 0.99–1.16×。
报告中仍需区分“TPC 集合互斥”和“缓存、带宽资源隔离”；libsmctrl 只保证前者。
