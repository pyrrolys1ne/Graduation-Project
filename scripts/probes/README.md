# 探针脚本归档

这些脚本原本写在 `/tmp`，随时可能被清理。它们是 `docs/实验记录.md` 与 `AGENTS.md`
里**所有数字的唯一出处**，因此移到仓库内。

**运行前必须设 `LIBSMCTRL_PATH`**（掩码实验部分）：

```bash
export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
.venv-linux/bin/python scripts/probes/<script>.py
```

原始 JSON 输出在 `results/reproducibility/probes/`（`results/` 已被 gitignore）。

## 索引

### 第一轮：方向寻找（对应候选文档 §2、§4.2–§4.3）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `pilot_tpc_bandwidth.py` | 启用 TPC 数 → 有效带宽 / 算力，含时钟记录 | 证伪 DVFS 假设 |
| `pilot_clock_vs_mask.py` | 交错随机测量，分离掩码效应与时钟漂移 | 时钟不是 88% 差异的主因 |
| `pilot_mem_partition.py` | 两线程互斥分区下的 slowdown | 分区并发被不分区并发支配 |
| `pilot_arms.py` | 三臂对照（**计时口径有缺陷，已被 v2 取代**） | — |
| `pilot_arms2.py` | 四臂对照修正版 | 候选文档 §4.3 的主表 |
| `pilot_phase.py` | 算术强度扫描 | 分区收益随算术强度单调下降 |

> ⚠️ `pilot_phase.json` 在第二轮运行时被覆盖，两轮对比值已抄录进候选文档 §2.2。

### 第二轮：方向审核（对应候选文档 §4）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `audit_two_directions.py` | 四臂对照（S12/S6/C66/C12）+ k=6 子集身份效应 | 方向二淘汰、方向一反转 |
| `audit_blocked_vs_interleaved.py` | 分块顺序 vs 交错随机的复现性 | 排除"测量顺序"假设 |
| `audit_clip_reproducibility.py` | CLIP 前向复现性：严格协议 / 复刻现状 / 冷启动 | 尺寸边界的核心证据 |

### 第三轮：机制诊断（对应实验记录第 10 节、AGENTS.md 第 11 节）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `diag_mask_toggle.py` | 掩码"设一次" vs "每轮设+释放"的开销 | 否证"掩码切换是变慢的原因" |
| `diag_duty_cycle.py` | 密集 / 稀疏 / 预热三种排布下的时钟与延迟 | 发现时钟受占空比影响（后又被部分修正） |
| `diag_slow_mode.py` | CLIP vs matmul 的慢模态占比 | **关键分界：尾部只出现在 CLIP 前向上** |
| `clock_probe.py` | 持续负载下的时钟轨迹 | 负载下 2 s 内可从 1890 升到 2460 MHz |

### 数据审计（对应候选文档 §5.3）

`an1_overview.py`、`an2_aggregates.py`、`an3_jsonl.py`、`an4_final.py`
——扫描 `results/` 全部 1326 个文件，发现"报告的是 10 次运行里的后 5 次"等问题。

### 第三轮：并发度负结果归因（对应 `docs/实验记录.md` §22）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `an5_concurrency_attribution.py` | 服务层静态臂的耗时分解、GPU 时间随 N 的缩放、控制器决策分布 | §22.1、§22.4 |
| `pilot_concurrency_span.py` | 无线程/流错配下 N 并发：现造 vs 预生成输入、发射耗时、功耗 | §22.2–22.3，机制指向主机侧发射 |
| `pilot_concurrency_processes.py` | 同并发改线程为进程（不共享 GIL） | §22.3，进程下吞吐恒定 → 塌陷来自线程侧发射 |

⚠️ 这两个 pilot 脚本会**独占 GPU**（各约 2 分钟），跑之前先确认宿主侧空闲。

### 第四轮：图回放流水线（对应 `docs/实验记录.md` §23）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `pilot_graph_concurrency.py` | eager / graph_prebuilt 三臂 × N=1…8：图回放能否消掉 N≥3 塌陷；主机发射耗时与 GPU span | §23.1 |
| `pilot_graph_pipeline.py` | 把 CPU 造图与图回放拆成两级线程池后，是否还挂死、吞吐是否接近上界 | §23.2–23.3 |
| `pilot_graph_pairing.py` | 16 个尺寸配对的成本矩阵（加性？）、成批 B=1/2/4 的缩放、并发×成批是否可乘 | §24.1–24.3 |
| `diag_fp16_nan.py` | fp16 下 `encode_batch` 是否返回 NaN 嵌入（fp32 作对照） | §23.6 |

⚠️ `pilot_graph_pairing.py` 的**多 lane 臂会挂死**（§24.5）。该探针既无 CPU 造图也无
分页 H2D，所以这不是 §23.2 那两条约束能解释的；根因仍未定位。
**挂死不是慢——是 GPU 不再推进**，`Lane.collect` 会在 lane 不返回时直接抛错，
不把 0 写成结果。

**规避手段：一个配置一个进程**（`--only <臂>`，逐臂起进程再合并）。
逐臂跑 16 个配对时 13 个成功、3 个挂死——成功率高于单进程连跑，但**不是保证**。
多 lane 臂的定位是记录约束边界，不是产生性能结论。

⚠️ `pilot_graph_*` 会**独占 GPU** 并捕获几十份 CUDA Graph（约 1 分钟启动开销）。

**两个会挂死的写法**（§23.2，形态都是"所有线程卡在 `stream.synchronize()`、GPU 永不完结"）：

1. **H2D 的源用分页内存**——必须锁页。服务侧 `ClipEncoderBackend._input` 目前是分页源。
2. **CPU 造图与图回放共处一个线程**——必须拆成两级。这是主要约束，锁页不能替代它。

还有一条：**回放后的后处理（如算模长）必须在回放流内部做**。放到 `with torch.cuda.stream`
之外会落到默认流，默认流的隐式同步语义足以让两个回放线程互等。

## 纪律

这些脚本**都不是正式实验脚本**，只在排查与探索时使用。正式实验请用：

- `scripts/experiment_reproducibility_boundary.py`（标定表可复现性，配套 12 项契约测试）
- `scripts/profile_encoder.py`、`scripts/run_repeated_comparison.py`（既有对照）
