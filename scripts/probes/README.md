# 探针脚本归档

这些脚本原本写在 `/tmp`，随时可能被清理。它们是 `docs/实验记录.md` 与 `AGENTS.md`
里**所有数字的唯一出处**，因此移到仓库内。

**运行前必须设 `LIBSMCTRL_PATH`**（掩码实验部分）：

```bash
export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
.venv/bin/python scripts/probes/<script>.py
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

### 第五轮：真 CLIP 的 S12/C66/C12 对照（2026-09-29 从 `/tmp` 抢救）

这一组脚本此前**只在 `/tmp`**，`任务书.txt` 的 E1 读数由它们产生但**没有任何落盘产物**。
2026-09-29 抢救入库，其中 `clip_stab2.py` 加了 `--out` 并重跑（测量逻辑未改）。

| 脚本 | 测什么 | 产物 |
|---|---|---|
| `clip_stab2.py` | 672 / 1024 / 2048 的 S12 / S6 / C66 / C12，**每尺寸 8 次交替、逐次配对** | `results/reproducibility/probes/clip_stab2.json` |
| `clip_arms.py` | 真 CLIP 三臂，尺寸 672 / 1024 | `clip_arms.json` |
| `clip_arms2.py` | 真 CLIP 四臂（多 S6），尺寸 672–2048 六档，聚合口径 `2/max(t_a,t_b)` | `clip_arms2.json` |
| `clip_stab.py` | `clip_stab2` 的前身 | 无（只 print） |
| `big_sizes.csv` / `big_sizes2.csv` | 1024 / 1344 的配额曲线 | 同左 |

⚠️ `clip_stab2` 与 `clip_arms2` 的**聚合口径不同**（逐次配对 vs `2/max`），
两者的比值不可直接混用。`clip_stab2` 更严谨，**引用时以它为准**。

### 第六轮：§10.2 的"计算受限"标签判定（2026-09-29）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `diag_compute_bound_label.py` | 先标定**本卡 fp16 GEMM 持续峰值**（N=2048…8192 扫描，含 NVML 时钟/功耗/占用），再复刻 §10.2 三臂并算其占峰值百分比 | §10.2 的"计算受限"标签**成立**；顺带修正 §10.2 `per_job` 的 8× 单位错 |

`kernels_probe.py` / `kernels_probe2.py` / `kernels_timing.py` / `host_share_big.py`
为同期辅助探针（打印 kernel 名、事件计时、host 占比），无独立结论。

### 第七轮：记录格式、分解与判决（2026-09-29）

| 脚本 | 测什么 | 结论去向 |
|---|---|---|
| `diag_launch_vs_bubble.py` | 把"非 GPU 时间占比"分解为 跨度 / Σ内核 / 主机发射 / 气泡 | §37.2：**224 归因成立，672 不成立**；两臂内核集合完全相同 → 排除"图回放改变 GPU 执行" |
| `diag_mask_in_graph_capture.py` | 掩码在**捕获时**施加能否烧进图（外加回放时的阴性对照） | §37.3：**三条挂载路径全部封死** |

⚠️ **这两条约束都实测撞到，写脚本时必须遵守**：

1. **一个尺寸一个进程**——同一进程内连捕两张 CUDA Graph 会挂死（与 §24.5 同类）。
2. **`torch.profiler` 在同一进程内多次会话会挂死**——因此波动幅度只能从**跨进程重复**估。
   `diag_launch_vs_bubble.py` 默认单次采样；判定函数在波动未知时拒绝下结论。

⚠️ `diag_launch_vs_bubble.py` 的判定比的是**气泡 vs (主机发射 − Σ内核)**，不是 vs 主机发射
总量——发射与 GPU 执行重叠，只有超出量才可能变成气泡。第一版比错了，已修。

## 纪律

这些脚本**都不是正式实验脚本**，只在排查与探索时使用。正式实验请用：

- `scripts/experiment_reproducibility_boundary.py`（标定表可复现性，配套 12 项契约测试）
- `scripts/profile_encoder.py`、`scripts/run_repeated_comparison.py`（既有对照）
