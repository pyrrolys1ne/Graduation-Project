"""标定表可复现性实验：边界刻画（Q1）、时长/尺寸分离（Q2）、测量条件依赖（Q3）。

背景
----
所有基于离线标定的空间分区调度器都假设 `SM 配额 → 延迟` 是一张可复现的表。本课题实测发现
该假设**在大请求上成立、在小请求上不成立**（672×672 跨轮极差 1.001×，224×224 高达 1.936×，
且换一种测量协议两者差 49–91% 对 ≤0.9%）。

**本轮进一步定位到机制**：同一个 224×224@3TPC 请求，仅仅改变**测量条件**，中位数就从
4.61 ms 变到 7.19 ms（相差 56%）：

======================  ============================================  ============
条件                    测量方式                                      实测中位数
======================  ============================================  ============
``dense``               紧挨着连测，无间隔                            4.61 ms
``idle``                每次测量前空载 150 ms（常规逐档扫描的样子）     7.19 ms
``hot``                 每次测量前先跑 1.2 s 连续负载                  5.16 ms
======================  ============================================  ============

``hot`` 条件下 GPU 时钟稳定在 2595 MHz，而 ``dense``/``idle`` 全程停在空闲频率 1890 MHz
——**224 这个请求太短，自己的工作量不足以把 GPU 推到升频状态，于是它的延迟由"前一次测量
留下了什么状态"决定**。这解释了为什么小请求的 `spread` 与 `drift` 都大：标定表的每一格
存的都是"那一次扫描恰好碰到的状态"。

本脚本要回答三个问题
--------------------
- **Q1 边界**：可复现/不可复现的分界在哪里？（``--phase boundary``）
- **Q2 时长 vs 尺寸**：决定因素是请求自己的**时长**，还是它的**尺寸/kernel 形态**？
  （``--phase duration``：把 224 用两种手段拉长，看什么时候变稳）
- **Q3 条件依赖**：同一个请求在不同测量条件下的延迟差多少？短请求是否更依赖条件？
  （``--phase condition``）

两个复现性指标（缺一不可）
--------------------------
- ``spread`` 轮间离散度：同一配置同一条件做 R 轮，每轮内 I 次取中位数，取轮间最差/最好比。
- ``condition_effect`` 条件间差异：同一配置在 ``dense`` / ``idle`` / ``hot`` 三种条件下的
  中位数之比。

只看 ``spread`` 会漏掉失效模式——实测中 224@6TPC 的 ``spread`` 只有 1.069×（看似很稳），
但换一种测量协议给出 4.506 对 2.71–2.90 ms，条件间差异高达 66%。

用法
----
::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python scripts/experiment_reproducibility_boundary.py --phase condition
    python scripts/experiment_reproducibility_boundary.py --phase duration
    python scripts/experiment_reproducibility_boundary.py --phase boundary

真实掩码实验禁止静默回退：资源后端不实施真实配额时脚本直接拒绝运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import queue
import random
import statistics
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

#: 实验设计：phase -> 尺寸 -> 该尺寸使用的 N 列表
PHASE_DESIGN: dict[str, dict[int, list[int]]] = {
    # Q2/Q3：把 224 拉长到与 672 可比；672 作为长请求参考点
    "duration": {224: [1, 2, 4, 8], 672: [1, 2]},
    "condition": {224: [1, 2, 4, 8], 672: [1, 2]},
    # Q1：在 224–672 之间补中间尺寸，定位分界
    "boundary": {224: [1], 288: [1], 336: [1], 384: [1], 448: [1], 512: [1], 672: [1]},
}

#: 测量条件 -> (测量前做什么, 每次测量前都做吗)
CONDITIONS = ("dense", "idle", "hot")


# --------------------------------------------------------------------------- #
# 工作线程：掩码是 __thread 的，必须在同一线程内施加并测量
# --------------------------------------------------------------------------- #
class Worker:
    """一个专用工作线程，持有自己的 TPC 分配。

    不能用 `ThreadPoolExecutor`：它复用线程，而分配表按线程 id 建账。也不能从主线程替
    别的线程释放——回调路径的掩码是 ``__thread`` 的，``release_quota`` 只释放调用线程自己
    的表项。两层含义合起来：每个线程自己申请、自己释放。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            fn, box = self._queue.get()
            if fn is None:
                self._queue.task_done()
                return
            try:
                box["result"] = fn()
            except Exception as exc:  # noqa: BLE001 - 回传给调用方
                box["error"] = exc
            finally:
                self._queue.task_done()

    def call(self, fn: Callable[[], Any], timeout: float = 1800.0) -> Any:
        box: dict = {}
        self._queue.put((fn, box))
        deadline = time.time() + timeout
        while not box:
            if time.time() > deadline:
                raise TimeoutError("等待工作线程超时（疑似死锁）")
            time.sleep(0.002)
        if "error" in box:
            raise box["error"]
        return box["result"]


# --------------------------------------------------------------------------- #
# 时钟快照
# --------------------------------------------------------------------------- #
def clock_now(torch_module) -> dict:
    """读取当前时钟/功耗/温度。

    用 ``torch.cuda.clock_rate()``（走 NVML）而不是 ``nvidia-smi``：后者单次约 60 ms，
    插在两次测量之间会把占空比压下去——**测量本身改变了被测对象**。这不是理论担忧：
    本课题实测到常规逐档扫描全程停在空闲频率 1890 MHz，而密集测量时 GPU 会升到 2600 MHz。
    """
    result: dict = {}
    try:
        result["sm_clock"] = torch_module.cuda.clock_rate()
    except Exception:  # noqa: BLE001
        pass
    try:
        result["power_w"] = round(torch_module.cuda.power_draw() / 1000, 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        result["temp_c"] = torch_module.cuda.temperature()
    except Exception:  # noqa: BLE001
        pass
    return result


# --------------------------------------------------------------------------- #
# 测量单元
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    size: int
    quota: float
    mode: str          # "batch" | "loop"
    n: int             # 批大小或循环次数
    condition: str     # "dense" | "idle" | "hot"

    @property
    def key(self) -> tuple:
        return (self.size, self.quota, self.mode, self.n, self.condition)

    def label(self) -> str:
        return f"{self.size} q={self.quota} {self.mode}×{self.n} [{self.condition}]"


class Harness:
    """把"一次请求"抽象成不同的时长构造方式，并统一计时。"""

    def __init__(self, encoder: ClipEncoderBackend, torch_module, heat_ms: int = 1200):
        self.encoder = encoder
        self.torch = torch_module
        self.heat_ms = heat_ms
        self.big_a = torch_module.randn(4096, 4096, device="cuda", dtype=torch_module.float16)
        self.big_b = torch_module.randn(4096, 4096, device="cuda", dtype=torch_module.float16)

    def _job(self, size: int, quota: float, tag: str) -> EncodeJob:
        # 不用内置 hash()：字符串哈希受 PYTHONHASHSEED 影响，会让实验不可复现。
        job = EncodeJob(tag, seed=zlib.crc32(tag.encode()) % (2**31),
                        width=size, height=size, deadline_ms=600_000)
        job.sm_fraction = quota
        return job

    def measure(self, size: int, quota: float, mode: str, n: int, tag: str) -> tuple[float, float, dict]:
        """返回 (GPU 执行时长 ms, 墙钟时长 ms, 资源信息)。

        两个时长口径都返回，因为它们的差本身就是信息：``loop`` 模式下 GPU 执行时间被
        每次调用的 CPU 侧输入准备切成若干段，中间有间隙；``batch`` 模式是一段连续工作。
        在"时长决定时钟状态"的假设下，这个差别正是要考察的对象。
        """
        torch = self.torch
        wall_start = time.perf_counter()
        if mode == "batch":
            results = self.encoder.encode_batch(
                [self._job(size, quota, f"{tag}-{i}") for i in range(n)], 0)
            exec_ms = float(results[0].execution_ms)
            resource = results[0].resource or {}
        elif mode == "loop":
            # 累加各次调用的 execution_ms：每次都排除自己的 CPU 输入准备，
            # 因此这个口径等于"N 次独立的前向 GPU 时间之和"，与 batch 口径可比。
            # 不能用外层 CUDA Event——那会把 torch.rand 的 CPU 时间算进去（实测虚高约 2 倍）。
            exec_ms = 0.0
            resource = {}
            for i in range(n):
                result = self.encoder.encode(self._job(size, quota, f"{tag}-{i}"), 0)
                exec_ms += float(result.execution_ms)
                resource = result.resource or resource
        else:
            raise ValueError(f"未知 mode: {mode}")
        return exec_ms, (time.perf_counter() - wall_start) * 1000, resource

    def idle(self, seconds: float) -> None:
        time.sleep(seconds)

    def heat(self, seconds: float) -> None:
        """跑一段连续负载，把 GPU 推到升频状态。

        必须逐次 ``synchronize()``：``torch.mm`` 是异步的，不同步的话 1.2 s 内能入队上千个
        matmul，之后一次性同步要等它们全部跑完（实测把"预热 2 s"变成了 14 s）。
        """
        if seconds <= 0:
            return
        torch = self.torch
        end = time.time() + seconds
        while time.time() < end:
            torch.mm(self.big_a, self.big_b)
            torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# 实验编排
# --------------------------------------------------------------------------- #
def build_cells(design: dict[int, list[int]], quotas: list[float], modes: list[str],
                conditions: tuple[str, ...]) -> list[Cell]:
    cells: list[Cell] = []
    for size, ns in design.items():
        for quota in quotas:
            for mode in modes:
                for n in ns:
                    for condition in conditions:
                        cells.append(Cell(size, quota, mode, n, condition))
    return cells


class RawLogger:
    """逐条落盘，脚本中断也不丢数据。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.count = 0

    def record(self, cell: Cell, index: int, exec_ms: float, wall_ms: float, clock: dict,
               round_index: int = 0) -> None:
        self.handle.write(json.dumps({
            "size": cell.size, "quota": cell.quota, "mode": cell.mode, "n": cell.n,
            "condition": cell.condition, "round": round_index, "inner_index": index,
            # wall_clock 是墙钟时刻。没有它就无法判断"慢样本"是均匀散布还是成簇出现——
            # 而"成簇"可能表示外部资源干扰，"均匀"更可能是被测对象自身的性质。本课题已因缺少
            # 这个字段而无法分析一次 1296 样本实验的尾部结构。
            "wall_clock": time.time(),
            "exec_ms": exec_ms, "wall_ms": wall_ms, **clock,
        }, ensure_ascii=False) + "\n")
        self.count += 1
        if self.count % 40 == 0:
            self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def describe(samples: list[float]) -> dict:
    """刻画一个分布。**不能只报均值/标准差**——实测分布是重尾的，
    常规标定恰好是"从重尾里抽一个点当中位用"。

    ``slow_fraction`` 是本课题的观测重点：实测同一个请求存在快慢两个模态
    （224×224 快模态约 3.7 ms，但约 15% 的样本落在 11–15 ms），而这个比例
    比中位数更能说明"标定表存的那一个点"有多不可靠。
    """
    if not samples:
        return {}
    ordered = sorted(samples)
    n = len(ordered)
    mean = statistics.fmean(ordered)
    std = statistics.stdev(ordered) if n > 1 else 0.0
    median = statistics.median(ordered)
    return {
        "n": n,
        "median_ms": median,
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean > 0 else float("nan"),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "range_ratio": ordered[-1] / ordered[0] if ordered[0] > 0 else float("nan"),
        "p05_ms": ordered[max(0, int(0.05 * n) - 1)],
        "p95_ms": ordered[min(n - 1, int(0.95 * n))],
        "p95_over_p50": (ordered[min(n - 1, int(0.95 * n))] / median) if median > 0 else float("nan"),
        # 慢模态占比：超过中位数 2 倍（且绝对差 ≥2 ms，避免把亚毫秒抖动算进来）的样本比例
        "slow_fraction": sum(1 for v in ordered if v > 2 * median and v - median >= 2.0) / n,
    }


def summarize(raw: list[dict]) -> list[dict]:
    """按 (size, quota, mode, n) 汇总：每个条件一个分布，再算条件间差异。"""
    grouped: dict[tuple, dict[str, list[float]]] = {}
    clk: dict[tuple, dict[str, list[int]]] = {}
    for row in raw:
        key = (row["size"], row["quota"], row["mode"], row["n"])
        grouped.setdefault(key, {}).setdefault(row["condition"], []).append(row["exec_ms"])
        # 时钟必须**按条件分别**统计。跨条件取中位数会让 dense/idle 的低频把 hot 的高频
        # 淹没，看起来"条件没生效"——本课题实际发生过，白排查了一轮。
        if "sm_clock" in row:
            clk.setdefault(key, {}).setdefault(row["condition"], []).append(row["sm_clock"])

    summary: list[dict] = []
    for key, per_condition in grouped.items():
        stats = {c: describe(v) for c, v in per_condition.items()}
        for condition, values in (clk.get(key) or {}).items():
            if condition in stats:
                stats[condition]["median_sm_clock"] = statistics.median(values)
        medians = [s["median_ms"] for s in stats.values() if s]
        condition_effect = (max(medians) / min(medians)) if len(medians) > 1 and min(medians) > 0 else None
        all_samples = [v for values in per_condition.values() for v in values]
        summary.append({
            "size": key[0], "quota": key[1], "mode": key[2], "n": key[3],
            "median_ms": statistics.median(all_samples) if all_samples else None,
            "condition_effect": condition_effect,
            "conditions": stats,
        })
    summary.sort(key=lambda r: (r["median_ms"] is None, r["median_ms"]))
    return summary


def print_summary(summary: list[dict], phase: str) -> None:
    print(f"\n{'='*128}")
    print(f"汇总（{phase}）")
    print(f"{'='*128}")
    header = (f"{'配置':>22} {'中位(ms)':>9} | "
              f"{'dense 中位/时钟/慢模态':>28} {'idle 中位/时钟/慢模态':>28} "
              f"{'hot 中位/时钟/慢模态':>28} | {'条件间差异':>10}")
    print(header)
    print("-" * 132)
    for row in summary:
        label = f"{row['size']} q={row['quota']} {row['mode']}×{row['n']}"
        parts = []
        for cond in CONDITIONS:
            s = row["conditions"].get(cond)
            if s:
                clk = s.get("median_sm_clock")
                parts.append(f"{s['median_ms']:>7.2f}/{clk if clk else '—':>5}/{s['slow_fraction']*100:>4.0f}%")
            else:
                parts.append(f"{'—':>28}")
        eff = row["condition_effect"]
        print(f"{label:>22} {row['median_ms']:>9.3f} | "
              f"{parts[0]:>28} {parts[1]:>28} {parts[2]:>28} | "
              f"{(f'{eff:.3f}×' if eff else '—'):>10}")


def print_duration_analysis(summary: list[dict]) -> None:
    """Q2 的核心输出：把每个配置按"该请求自己的时长"排序，看条件间差异如何变化。

    若条件间差异随请求时长单调下降 → **时长**是决定因素；
    若按尺寸分簇（224 一族与 672 一族各自成组，与时长无关）→ **尺寸/kernel 形态**是决定因素。
    """
    print(f"\n{'='*100}")
    print("Q2 分析：条件间差异 vs 请求自身时长（按时长排序）")
    print(f"{'='*100}")
    print(f"{'配置':>22} {'尺寸':>6} {'中位时长(ms)':>13} {'条件间差异':>11} {'最大CV':>9}")
    print("-" * 100)
    for row in summary:
        if row["condition_effect"] is None:
            continue
        max_cv = max((s["cv"] for s in row["conditions"].values() if s), default=float("nan"))
        print(f"{row['mode']+'×'+str(row['n'])+' q='+str(row['quota']):>22} "
              f"{row['size']:>6} {row['median_ms']:>13.3f} {row['condition_effect']:>10.3f}× "
              f"{max_cv:>9.3f}")


def _csv_ints(text: str) -> set[int]:
    return {int(p) for p in text.split(",") if p.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="标定表可复现性实验（Q1 边界 / Q2 时长-尺寸 / Q3 条件依赖）")
    parser.add_argument("--config", default="config.libsmctrl.example.yaml")
    parser.add_argument("--phase", choices=sorted(PHASE_DESIGN), default="condition")
    parser.add_argument("--quotas", default="0.25,1.0",
                        help="逗号分隔的 sm_fraction 档位；0.25 在 12 TPC 上即 3 个 TPC")
    parser.add_argument("--modes", default="batch,loop")
    parser.add_argument("--conditions", default="dense,idle,hot")
    parser.add_argument("--sizes", default=None, help="覆盖预设尺寸，如 224,672")
    parser.add_argument("--ns", default=None, help="覆盖预设的 N 列表，如 1,2,4")
    parser.add_argument("--rounds", type=int, default=3, help="每条件重复几轮")
    parser.add_argument("--inner", type=int, default=12, help="每轮内重复几次")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--idle-ms", type=int, default=150, help="idle 条件每次测量前的空载时长")
    parser.add_argument("--heat-ms", type=int, default=1200, help="hot 条件每次测量前的预热时长")
    parser.add_argument("--order-seed", type=int, default=20260915,
                        help="轮内单元顺序的随机种子。固定顺序会制造假梯度，见 run_all 说明")
    parser.add_argument("--settle-ms", type=int, default=1500,
                        help="每个单元开始前的静默期。没有它，上一个 hot 单元的热量会渗到下一个 "
                             "dense/idle 单元——实测整个会话的时钟会被抬到升频档，条件之间不再可比")
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproducibility"))
    args = parser.parse_args()

    design = dict(PHASE_DESIGN[args.phase])
    if args.sizes:
        wanted = _csv_ints(args.sizes)
        design = {s: v for s, v in design.items() if s in wanted}
    if args.ns:
        ns = sorted(_csv_ints(args.ns))
        design = {s: [n for n in ns if n in v] or [1] for s, v in design.items()}
    if not design:
        raise SystemExit("没有任何尺寸被选中，检查 --sizes")

    quotas = [float(q) for q in args.quotas.split(",") if q.strip()]
    modes = [m for m in args.modes.split(",") if m.strip()]
    conditions = tuple(c for c in args.conditions.split(",") if c.strip())
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"未知测量条件 {sorted(unknown)}；可选 {CONDITIONS}")

    config = load_config(args.config)
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    if not resource.enforces_sm_partition:
        raise SystemExit(
            f"资源后端 {resource.name} 不实施真实 SM 配额，本实验无意义。\n"
            "真实掩码实验禁止静默回退——请先跑 scripts/probe_libsmctrl.py 确认能力探针通过。"
        )
    print(f"资源后端 = {resource.name}，真实 SM 配额 = {resource.enforces_sm_partition}")

    encoder = ClipEncoderBackend(config.model, resource, 1)
    harness = Harness(encoder, encoder.torch, heat_ms=args.heat_ms)

    cells = build_cells(design, quotas, modes, conditions)
    per_cell_reps = args.inner
    print(f"计划 {len(cells)} 个单元 × {args.rounds} 轮 × {per_cell_reps} 次 "
          f"= {len(cells)*args.rounds*per_cell_reps} 次前向")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.phase}_raw.jsonl"
    logger = RawLogger(raw_path)

    worker = Worker()

    def cold_warmup():
        """整轮预热，避免第一个单元吃到冷启动（本课题踩过 363 ms 对 25 ms 的坑）。"""
        print("冷启动预热中……", flush=True)
        for size, ns in design.items():
            harness.measure(size, quotas[0], "loop", ns[0], "pretest")
        harness.heat(1.0)
        print("完成\n", flush=True)

    worker.call(cold_warmup)

    def run_all() -> tuple[list[dict], float]:
        """**轮次外层**：一轮 = 把全部单元各测一遍。

        为什么不是"单元外层"：会话状态在整轮实验里会漂移（实测同一台机器不同会话可差
        1.3–2×）。若一个单元的所有轮次连续测完再换下一个单元，每个单元看到的是**不同的
        会话状态**，单元之间的差异就与漂移混在一起——实测这会把 672 的跨轮极差从 1.01×
        抬到 1.66×，整个边界扫描失去分辨力。

        改成轮次外层后：**一轮就是一次完整标定**，轮内所有单元共享同一会话状态，
        因此「跨轮极差」正好是"重新标定一次会差多少"，也就是 Q1 要问的量。
        """
        raw: list[dict] = []
        started = time.time()
        rng = random.Random(args.order_seed)
        for round_index in range(args.rounds):
            # 轮间静默：让每一轮从同一状态起步（否则上一个 hot 单元的余温会污染本轮）
            harness.idle(args.settle_ms / 1000)
            if args.warmup:
                warm_cell = cells[0]
                for _ in range(args.warmup):
                    harness.measure(warm_cell.size, warm_cell.quota, warm_cell.mode,
                                    warm_cell.n, f"warm-r{round_index}")
            # **轮内顺序必须随机**：一轮要跑好几秒，若固定按尺寸顺序，则"每轮第一个测量"
            # 永远落在 224 上、"最后一个"永远落在 672 上。任何轮内的暖机或漂移都会因此
            # 变成一条**假梯度**——而梯度正是本实验要测的东西。
            # 固定种子保证可复现。
            round_cells = list(cells)
            rng.shuffle(round_cells)
            for index, cell in enumerate(round_cells):
                tag = f"r{round_index}"
                # 按条件决定测量前的动作。条件动作**每次测量前**都执行——
                # 只做一次的话，条件效应会被随后几十次测量的自加热抹掉。
                for i in range(per_cell_reps):
                    if cell.condition == "idle":
                        harness.idle(args.idle_ms / 1000)
                    elif cell.condition == "hot":
                        harness.heat(args.heat_ms / 1000)
                    clock = clock_now(encoder.torch)
                    exec_ms, wall_ms, _ = harness.measure(
                        cell.size, cell.quota, cell.mode, cell.n, f"{cell.mode}{cell.n}-{tag}-{i}")
                    logger.record(cell, i, exec_ms, wall_ms, clock, round_index)
                    raw.append({
                        "size": cell.size, "quota": cell.quota, "mode": cell.mode, "n": cell.n,
                        "condition": cell.condition, "round": round_index,
                        "exec_ms": exec_ms, "wall_ms": wall_ms, **clock,
                    })
                if (index + 1) % 8 == 0 or index + 1 == len(round_cells):
                    print(f"  轮 {round_index+1}/{args.rounds}  单元 {index+1}/{len(cells)}  "
                          f"已完成 {logger.count} 次测量", flush=True)
            print(f"  轮 {round_index+1}/{args.rounds} 完成，累计 {logger.count} 次测量", flush=True)
        return raw, time.time() - started

    raw, elapsed = worker.call(run_all)
    logger.close()

    summary = summarize(raw)
    print_summary(summary, args.phase)
    if args.phase in ("duration", "condition"):
        print_duration_analysis(summary)
    print(f"\n耗时 {elapsed:.1f} s；原始数据 {logger.count} 条 → {raw_path}")

    summary_path = args.output_dir / f"{args.phase}_summary.json"
    summary_path.write_text(json.dumps({
        "phase": args.phase,
        "config": args.config,
        "resource_backend": resource.name,
        "enforces_sm_partition": resource.enforces_sm_partition,
        "quotas": quotas, "modes": modes, "conditions": list(conditions),
        "rounds": args.rounds, "inner": args.inner, "warmup": args.warmup,
        "idle_ms": args.idle_ms, "heat_ms": args.heat_ms,
        "elapsed_s": elapsed,
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总 → {summary_path}")


if __name__ == "__main__":
    main()
