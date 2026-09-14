# 学习与验证用 Demo

这些 Demo 只负责**验证机制**，不承担论文对照实验；正式实验在 `scripts/` 下。

## libsmctrl_thread_mask_demo.py

### 目标

验证在本机（WSL2 + 驱动 610.62 + CUDA 13.3）上，**回调路径的 TPC 掩码确实可用**，且
本项目补丁新增的粘性线程掩码能表达"按请求配额"。

### 前置条件

```bash
# 1. 构建（脚本会自动应用 patches/libsmctrl-thread-mask.patch）
CUDA_HOME=/path/to/cuda LIBSMCTRL_PREFIX=$HOME/.local/lib/libsmctrl scripts/build_libsmctrl.sh

# 2. 指向构建产物
export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
```

未打补丁的上游库没有 `libsmctrl_set_thread_mask`，Demo 会明确报错并给出重建指引。

### 运行

```bash
python demos/libsmctrl_thread_mask_demo.py
```

### 预期现象

```
阶段 1：三种方式都只布防一次，然后连续 20 次启动（启用 1 个 TPC）
  无掩码基线           中位   3.02 ms
  next_mask 布防一次   第1次  30.18 ms，其余  2.71 ms  -> 覆盖  1/20 次
  thread_mask 布防一次 第1次  31.69 ms，其余 22.19 ms  -> 覆盖 20/20 次

阶段 2：一个线程持有的掩码不会影响其他线程
  持掩码线程(1 TPC)   21.89 ms/次  ← 受掩码限制
  主线程(无掩码)       1.98 ms/次  ← 不受影响
  比值 = 11.05x
```

退出码 0 表示两项性质都成立；非 0 表示至少一项未成立。

### 为什么这在 CUDA 13 上能成立

libsmctrl 有**两套互不相关的机制**：

| 机制 | 函数 | 版本门控 | 本机（CUDA 13.3） |
|---|---|---|---|
| 硬编码结构体偏移 | `set_stream_mask` | 按驱动版本白名单，未知版本 `exit(1)` | ❌ 调用会**终止进程** |
| 启动回调 | `set_global_mask` / `set_next_mask` | 只有下界（<6.5），**运行时读 TMD 版本自适应** | ✅ 实测生效 |

回调路径能工作，恰恰因为它不硬编码驱动版本——它读运行时 TMD 版本字段决定偏移
（`>=0x40` 走 Hopper 的 +304/+308，否则 +84/+88）。

### 边界（不要说成别的）

本 Demo 证明的是**掩码机制可用**，不是"调度策略有效"。它不涉及：

- 真实配额曲线（见 `scripts/experiment_sm_quota.py`）
- 并发请求之间的吞吐/延迟对照（见 `scripts/run_repeated_comparison.py`）
- 掩码是否被硬件严格执行到 SM 级别——本 Demo 用延迟缩放作为间接证据，
  要更强的证据需要 Nsight 或 SM occupancy 计数器

### 一个会挂死进程的坑

`mask_for(n)` 在 `n <= 0` 时返回 **0**（不禁用任何 TPC，即不加限制）。
**不要传 `~0`**：那会禁用全部 TPC，kernel 永远发不出去，进程直接挂死。
Demo 早期版本就踩过这个坑。
