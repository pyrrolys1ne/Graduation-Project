"""图回放流水线：把"CPU 图像准备"与"CUDA Graph 回放"拆成两级。

这个模块要解决的是什么
──────────────────────

``docs/实验记录.md`` §22 把并发度负结果归因到**主机侧 kernel 发射**：N≥3 时
主机发射耗时 ≈ GPU span（672 预生成输入下 3.9 → 22.0 ms 的悬崖），换成进程后
聚合吞吐恒定。``scripts/probes/pilot_graph_concurrency.py`` 直接验证了这条归因的
反面——**把发射路径换成 CUDA Graph 回放**（主机发射 3.97 ms → 0.04 ms）之后：

    臂                 N=1     N=2     N=3     N=4
    eager             128.0   173.6   120.8    84.4     ← 塌陷复现
    graph_prebuilt    278.4   348.6   348.6    …        ← 塌陷消失，单调不降

但那个探针的控制臂**不做 CPU 造图**。把真实预处理放回去，立即撞上一个硬约束：

> **CPU 有持续负载、且另一条流上有图正在回放时，所有线程会卡死在各自的
> ``stream.synchronize()`` 上，GPU 永不完结。**

二分结果（判据是"跑满时长且正常退出"）：

| 变体（N=4，尺寸 224/336/448/672 均匀混合） | 结果 |
|---|---|
| 只做显存内拷贝（无 CPU 计算、无 H2D） | 354.8 req/s，正常 |
| 分页源 H2D，**无** CPU 计算 | 324.9 req/s，正常 |
| 锁页源 H2D，CPU 只做一趟 ``uniform_`` | 363.4 req/s，正常 |
| **分页源 H2D + CPU 造图** | **挂死** |
| **锁页源 H2D + CPU 造图（3 趟），同一线程** | **挂死** |

即"源是否锁页"不是充分条件，**"CPU 计算与图回放共处一个线程"才是**。因此本模块
的形态不是"又一个后端"，而是把两级**在结构上分开**：

    prepare 级（CPU 线程池，``slots`` 之外独立配置 ``prep_threads`` 个）
        造图 → 归一化 → 写进**锁页**缓冲
        ↑ 只做 CPU，绝不碰 CUDA
    replay 级（每个 worker 一条流、一份图实例）
        锁页 → 显存 → 图静态输入 → ``graph.replay()`` → 流同步
        ↑ 只做 CUDA，绝不做 CPU 造图

两级之间用"每个 slot 一份锁页缓冲"衔接（不是队列：slot 一次只处理一个请求，
无需额外同步原语）。拆开之后实测 351–357 req/s 且不再挂死
（``scripts/probes/pilot_graph_pipeline.py``），其中 ``replay=2`` 最优——
**本机的最优共驻数仍然是 2**，与 §21.3 的结论一致，变的是天花板的高度。

为什么保留"每个尺寸一份图"
──────────────────────────

CLIP ViT-B/32 的 patch 是 32×32，四个尺寸的 patch 数分别是 49/100/196/441，
经 ``interpolate_pos_encoding`` 后序列长度不同，**张量形状不同 → 必须分尺寸捕获**。
批处理同理（``encode_batch`` 已要求同尺寸），所以本模块与批处理是**正交**的：
将来若要叠加批，只需把"每个尺寸一份图"扩成"每个（尺寸, 批大小）一份图"。

与掩码的关系
────────────

§10.7 已记录：**libsmctrl 的回调挂在 kernel 发射路径上，而 graph 回放走
``cuGraphLaunch``，绕开了回调——掩码在回放时不生效**。因此本模块与 SM 掩码
天然互斥，构造时就拒绝 ``enforces_sm_partition`` 为真的资源后端，而不是
静默地给出一份"看起来限了 SM、实际没限"的结果。
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .models import EncodeJob
from .resource import ResourceBackend


@dataclass(frozen=True)
class GraphRuntimeConfig:
    """图回放流水线的参数。

    两个并行度是**两个解耦的旋钮**：回放级决定有几条 lane 在跑，
    预处理级决定主机侧有几份造图在并行。
    """

    #: 回放 lane 数。每个 lane 持有自己的图实例。
    #:
    #: ⚠️ **必须为 1。** 大于 1 时"多张 CUDA Graph 同时在 GPU 上执行"会**偶发挂死**
    #: （所有回放线程卡在 ``stream.synchronize()``、GPU 永不完结）：
    #: 预取结构下实测 `slots=2` **6 次运行 4 次挂**，而 `slots=1`
    #: **27 次运行 0 次挂**。把发射调用串行化**无效**；把**执行**串行化才能消除，
    #: 但那等于变回单 lane（221 对 222 req/s），所以直接用 1。
    #: 全部实测与已排除项见 ``docs/实验记录.md`` §25。
    slots: int = 1
    #: 预处理线程数。与 ``slots`` 独立——预处理不出现在 GPU 上，
    #: 它只需要"足够快到不成为新瓶颈"。
    prep_threads: int = 2
    #: 预先捕获图的尺寸集合。请求落在集合外时由调用方走 eager 回退路径：
    #: 中途捕获要求显存上没有任何在飞的图（见 ``_capture`` 的说明），
    #: 放进请求路径只会把偶发的长停顿变成常态。
    sizes: tuple[tuple[int, int], ...] = ((224, 224), (336, 336), (448, 448), (672, 672))
    #: **每个尺寸允许的批上限**，形如 ``((224, 224, 4), ...)``。空表示全部按 batch=1。
    #:
    #: 为什么批大小要**按尺寸**给：§24.2 实测批内填空转的空间随尺寸单调收紧
    #: （单流 B=4 相对 B=1：224 为 **2.42×**、336 1.79×、448 1.38×、672 仅 1.16×），
    #: 而代价（等尺寸凑齐）每个尺寸都是一样的。**小请求该大批、大请求该小批**
    #: 不是调参，是机制。
    #:
    #: ⚠️ 批上限大于 1 时**必须**同时满足：该尺寸的批图已被捕获（构造时按此表捕获），
    #: 且 ``slots``（lane 数）足以容纳——本版每个 (尺寸, 批大小) 各捕获 ``slots`` 份实例。
    batch_by_size: tuple[tuple[int, int, int], ...] = ()
    #: 凑批时最多向后扫多少条队列项。批处理要求同形状，而请求尺寸在队列里是交错的
    #: （224/336/448/672 循环到达），只看队首的下一个必然凑不成批。
    batch_scan_factor: int = 4


@dataclass
class _Slot:
    """一个 (尺寸, 槽位) 单元的图实例与缓冲。**只允许一个线程在任一时刻使用它。**"""

    graph: Any
    static_input: Any
    output: Any
    staging: Any
    start_event: Any = None
    end_event: Any = None
    #: 保护"图实例只被一个线程用"这一不变量。service 侧按 worker_id 分配 slot，
    #: 正常情况下锁永远不竞争；它是给回退路径与将来可能的负载均衡留的安全网。
    lock: threading.Lock = field(default_factory=threading.Lock)


#: 每个**尺寸**预备多少块锁页缓冲。
#:
#: ⚠️ 池子按尺寸共享，**不按槽位**。预处理级不知道也不该知道"这次由哪条 lane 回放"——
#: 一旦让 prepare 按自己的编号去选槽位，准备协程 1 可能把件交给 lane 0，
#: 于是两条 lane 共用同一份图实例与同一组 CUDA 事件，报
#: `Both events must be completed before calculating elapsed time`
#: （实测：曾一次运行 281/1200 失败）。槽位只由回放 lane 决定。
#:
#: 容量要 ≥ 同时在飞的件数（预处理协程数 + lane 数）；672 一块约 5.4 MB，取 8 余量充足。
PIN_POOL = 8


@dataclass(frozen=True)
class PreparedInput:
    """``prepare`` 的产物：一份已填好的锁页缓冲 + 它的归属。

    ``buffer`` 是**从池子里借的**，``replay`` 用完必须还回（见 ``replay`` 的 finally）。
    """

    size: tuple[int, int]
    #: 这一批里有几个请求，也即 buffer 的第 0 维。
    batch: int
    buffer: Any
    cpu_ms: float
    resource: dict[str, Any]


class GraphReplayRuntime:
    """按尺寸捕获的 CUDA Graph 池，提供 prepare / replay 两级接口。

    调用约定（**服务侧必须遵守，否则会退化成死锁配置**）：

    - ``prepare`` 只在**预处理线程**上调用，只做 CPU 工作；
    - ``replay`` 只在**回放线程**上调用，只做 CUDA 工作；
    - ``prepare`` 从锁页缓冲池取一块缓冲、填好、交给 ``replay``，``replay`` 用完还回。
      **缓冲是借来的，调用方不得跨请求持有**——服务侧的预取会让"下一请求的准备"
      与"当前请求的回放"同时存在，共用一块缓冲会直接写坏正在被回放读取的输入。
    """

    def __init__(self, backend: Any, config: GraphRuntimeConfig, resource_backend: ResourceBackend):
        self.backend = backend
        self.config = config
        self.resource_backend = resource_backend
        self.torch = backend.torch
        self.model = backend.model
        self.dtype = backend.dtype
        self.mean = backend.mean
        self.std = backend.std

        # 掩码与图回放互斥（§10.7）：宁可拒绝启动，也不要给一份"限了 SM"的假象。
        if getattr(resource_backend, "enforces_sm_partition", False):
            raise ValueError(
                f"资源后端 {type(resource_backend).__name__} 声明 enforces_sm_partition=true，"
                "而 libsmctrl 的回调在 CUDA Graph 回放时不生效（cuGraphLaunch 绕开了发射回调）。"
                "图回放流水线必须与 proxy 这类零副作用后端搭配。"
            )

        self._slots: dict[tuple[tuple[int, int], int], list[_Slot]] = {}
        #: (尺寸, 批大小) → 空闲锁页缓冲队列。借还由 prepare / replay 配对完成。
        #: **按 (尺寸, 批) 共享、不按槽位**（理由见 PIN_POOL 的说明）。
        self._pin_free: dict[tuple[tuple[int, int], int], queue.Queue] = {}
        self._capture_ms: dict[str, float] = {}
        #: (尺寸) → 批上限。来自 ``config.batch_by_size``。
        self._batch_by_size: dict[tuple[int, int], int] = {
            (w, h): batch for w, h, batch in config.batch_by_size
        }
        for size in config.sizes:
            # 捕获 1..cap **全部**批大小，而不是只捕获 1 与 cap：
            # 队列里能凑到的同尺寸条数是随负载变化的（负载低时可能只有 2 条），
            # 只捕获两端会让"凑不满 cap"的批退化成逐请求，白丢掉整批的收益。
            cap = max(1, self._batch_by_size.get(size, 1))
            for batch in range(1, cap + 1):
                self._capture(size, batch)

        self._prep_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, config.prep_threads), thread_name_prefix="graph-prep")
        self._replayed = 0
        self._counter_lock = threading.Lock()

    # ── 捕获 ──────────────────────────────────────────────────────

    def batch_capacity(self, job: EncodeJob) -> int:
        """该请求尺寸允许的最大批大小。未配置的尺寸一律 1。"""
        return self._batch_by_size.get((job.width, job.height), 1)

    def _capture(self, size: tuple[int, int], batch: int) -> None:
        """为一个 (尺寸, 批大小) 捕获 ``slots`` 份图。

        必须在**显存上没有任何在飞的图**时进行：``torch.cuda.graph`` 的捕获会让
        分配器切到一份私有池，此时若有别的线程在回放，捕获会失败或产出错图。
        因此这里的调用点只有两处——服务启动阶段，以及显式的预热。
        """
        torch = self.torch
        width, height = size
        start = time.perf_counter()
        slots: list[_Slot] = []
        for _ in range(max(1, self.config.slots)):
            static_input = torch.empty((batch, 3, height, width), dtype=self.dtype, device="cuda")
            static_input.normal_()
            # 捕获前在旁路流上预热：首次执行会做 cuDNN 算法选择与内存池扩张，
            # 那些一次性动作不能进图。
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    with torch.inference_mode():
                        self.model(pixel_values=static_input, interpolate_pos_encoding=True)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    captured = self.model(
                        pixel_values=static_input, interpolate_pos_encoding=True).image_embeds

            # 正确性对照：图输出必须与 eager 一致，否则宁可直接失败。
            graph.replay()
            torch.cuda.synchronize()
            with torch.inference_mode():
                reference = self.model(
                    pixel_values=static_input, interpolate_pos_encoding=True).image_embeds
            torch.cuda.synchronize()
            diff = float((captured.detach().float() - reference.detach().float()).abs().max().item())
            if diff > 1e-2:
                raise RuntimeError(
                    f"{width}×{height} batch{batch} 的 CUDA Graph 输出与 eager 不符"
                    f"（最大绝对差 {diff:.3e}）。不能拿错图做实验。"
                )

            slots.append(_Slot(
                graph=graph,
                static_input=static_input,
                output=captured,
                staging=torch.empty((batch, 3, height, width), dtype=self.dtype, device="cuda"),
                start_event=torch.cuda.Event(enable_timing=True),
                end_event=torch.cuda.Event(enable_timing=True),
            ))
        key = (size, batch)
        self._slots[key] = slots
        free: queue.Queue = queue.Queue()
        for _ in range(PIN_POOL):
            free.put(torch.empty((batch, 3, height, width), dtype=torch.float32).pin_memory())
        self._pin_free[key] = free
        self._capture_ms[f"{width}x{height}x{batch}"] = (
            time.perf_counter() - start) * 1000 / len(slots)

    # ── 第一级：CPU 图像准备 ──────────────────────────────────────

    def supports(self, job: EncodeJob) -> bool:
        return ((job.width, job.height), 1) in self._slots

    def prepare(self, jobs: list[EncodeJob]) -> PreparedInput:
        """造图 → 归一化 → 写进**锁页**缓冲。**只做 CPU，绝不碰 CUDA。**

        写锁页缓冲这一步不是优化，是正确性要求：分页源上的 H2D 会与另一条流上的
        图回放互相等待（模块 docstring 的二分表）。

        ``jobs`` 必须**同尺寸**且条数落在已捕获的批大小集合内（1..该尺寸的 cap）。
        批内各请求按各自的 seed 造图，落在借来的那块 ``(B,3,H,W)`` 锁页缓冲的对应行。
        """
        start = time.perf_counter()
        size = (jobs[0].width, jobs[0].height)
        batch = len(jobs)
        torch = self.torch
        buffer = self._pin_free[(size, batch)].get()
        # 与 ``ClipEncoderBackend._input`` 同口径：CPU 上定种子造图再归一化。
        # 差别只在于落点是**借来的**锁页缓冲而不是新分配的分页张量。
        for index, job in enumerate(jobs):
            row = buffer[index]                      # (3, H, W)
            row.uniform_(generator=torch.Generator(device="cpu").manual_seed(job.seed))
            row.sub_(self.mean).div_(self.std)
        # 配额在预处理级下发：回放级要保持"纯 CUDA"，不做任何额外的主机动作。
        # 图回放与掩码互斥（构造时已拒绝），所以这里的后端必然是 proxy：
        # 第一个参数（流句柄）不参与执行，只进日志。
        resource = self.resource_backend.apply_quota(0, jobs[0].sm_fraction)
        return PreparedInput(
            size=size, batch=batch, buffer=buffer,
            cpu_ms=(time.perf_counter() - start) * 1000, resource=resource,
        )

    async def aprepare(self, jobs: list[EncodeJob]) -> PreparedInput:
        """在**预处理线程池**上执行 ``prepare``。

        服务侧必须走这个入口而不是直接调 ``prepare``——直接在回放线程上造图，
        就是模块 docstring 里那张二分表最后一行会挂死的配置。
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._prep_pool, self.prepare, jobs)

    # ── 第二级：图回放 ────────────────────────────────────────────

    def replay(self, jobs: list[EncodeJob], slot_index: int,
               prepared: PreparedInput) -> list[dict[str, Any]]:
        """锁页 → 显存 → 图静态输入 → 回放 → 流同步。**只做 CUDA。**

        返回与 ``jobs`` 等长的列表，每项是
        ``{"embedding_dim", "embedding_norm", "execution_ms", "resource"}``，
        与 ``EncodingResult`` 同形，由调用方组装。**批内各请求共享同一个
        ``execution_ms``**——它们本来就是同一次前向。
        """
        torch = self.torch
        # **槽位只由 lane 决定**（见 PIN_POOL 的说明）。批大小来自这一批的实际条数，
        # 它一定是已捕获的（1..cap，见 __init__）。
        key = (prepared.size, prepared.batch)
        copies = self._slots[key]
        slot = copies[slot_index % len(copies)]
        stream = self.backend.streams[slot_index % len(self.backend.streams)]

        try:
            with slot.lock, torch.cuda.stream(stream):
                # ``non_blocking=True`` 只在源是锁页时才是真异步；这里源一定是锁页。
                slot.staging.copy_(prepared.buffer, non_blocking=True)
                slot.static_input.copy_(slot.staging)
                slot.start_event.record(stream)
                slot.graph.replay()
                slot.end_event.record(stream)
                # 模长**必须在本流上算**（且 `end_event` 之后）。放到流外会让它落到默认流上：
                # 默认流对同上下文的其它流有隐式同步语义，一次阻塞式 D2H 就会和另一槽位
                # 正在回放的图互等，两个回放线程双双卡死在各自的同步点（实测复现）。
                # dim=-1：按行取模长，得到 (B,) 而不是整个批的范数
                norm_tensor = torch.linalg.vector_norm(slot.output.detach().float(), dim=-1)
                embedding_dim = int(slot.output.shape[-1])
            stream.synchronize()
            execution_ms = float(slot.start_event.elapsed_time(slot.end_event))
            norms = [float(value) for value in norm_tensor.cpu().tolist()]
        finally:
            # 缓冲必须还回——调用方（预取）会在下一次 prepare 里再借走它。
            self._pin_free[key].put(prepared.buffer)
        self.resource_backend.release_quota(slot_index)
        with self._counter_lock:
            self._replayed += 1
        return [
            {
                "embedding_dim": embedding_dim,
                "embedding_norm": norms[index],
                "execution_ms": execution_ms,
                "resource": prepared.resource,
            }
            for index in range(len(jobs))
        ]

    # ── 观测 ──────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "cuda_graph_replay_pipeline",
            "slots": self.config.slots,
            "prep_threads": self.config.prep_threads,
            "sizes": [f"{w}x{h}" for w, h in self.config.sizes],
            "batch_by_size": {f"{w}x{h}": batch for (w, h), batch in sorted(self._batch_by_size.items())},
            "capture_ms_per_instance": {k: round(v, 1) for k, v in self._capture_ms.items()},
            "replayed": self._replayed,
        }

    def close(self) -> None:
        self._prep_pool.shutdown(wait=False, cancel_futures=True)
