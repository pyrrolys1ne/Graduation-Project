# 探针脚本归档

这些脚本原本写在 `/tmp`，随时可能被清理。它们是 `docs/研究方向候选-2026-09-15.md`、
`docs/实验记录.md` 第 10 节和 `AGENTS.md` 第 9–11 节里**所有数字的唯一出处**，因此移到仓库内。

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

## 纪律

这些脚本**都不是正式实验脚本**，只在排查与探索时使用。正式实验请用：

- `scripts/experiment_reproducibility_boundary.py`（标定表可复现性，配套 12 项契约测试）
- `scripts/profile_encoder.py`、`scripts/run_repeated_comparison.py`（既有对照）
