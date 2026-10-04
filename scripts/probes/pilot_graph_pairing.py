"""探针：发射成本被抹平之后，"把哪些请求放在一起"是不是一个有效的调度自由度？

为什么现在问这个
────────────────

§23 已经把图回放流水线做通：主机发射从 3.97 ms 压到 0.04 ms，N≥3 的塌陷消失，
两级流水线跑到 354 req/s。此时系统的状态变了——

    均匀混合下每请求 GPU span 中位 2.84 ms，replay=2 实测 354 req/s ≈ 2 / 2.84 ms × … 的量级

也就是说**瓶颈已经从"主机喂不饱 GPU"变成"GPU 本身要花多久"**。这时"共驻几个"
不再是唯一的旋钮：既然每个尺寸的 GPU 成本是**录制时就固定下来的常量**，
"哪几个请求凑在一起跑"就成了新的、且**可以离线标定**的决策。

这正是 §22.4 判死 old 控制器时缺的东西。当时失败的信号是"等待队列的尺寸混合"——
一个**取决于实现时序**的队列快照，同一负载下换流池条数就能让它翻面。
而下面这张成本矩阵是**负载的函数**：给定尺寸组合，成本唯一，跟谁先谁后无关。

同时它回答一个文献梳理时标记为"互相排斥、可判定"的问题：

    §22.6 把批处理的历史优势归因于"省掉 N 份发射开销"。
    如果这个归因对，那么**发射被图回放抹平之后，批处理相对多流并发的优势应当显著缩小**。
    若批处理在图回放下**仍然**明显赢，说明它的收益来自 GPU 侧（SM 填空），
    与发射无关——两种结果互相排斥，都要写进论文。

测什么
──────

三族臂，全部**预生成输入**（本探针要的是纯 GPU 成本，不含 CPU 造图）：

| 臂 | 含义 | 吞吐口径 |
|---|---|---|
| `single_<s>` | 1 条流连续回放尺寸 s 的 batch-1 图 | 1 / span |
| `pair_<si>_<sj>` | 2 条流分别连续回放 si、sj 的 batch-1 图 | 2 / 墙钟 |
| `batch2_<s>` | 1 条流连续回放尺寸 s 的 batch-2 图（一次前向两个请求） | 2 / span |
| `mix4` | 4 条流各跑一个尺寸 | 4 / 墙钟 |

`pair` 与 `batch2` 的对比就是上面那个判决：同样是"2 个请求"，一个是 2 次发射
2 条流，一个是 1 次发射 1 条流。

再顺带得到：**16 个尺寸组合里有没有明显更优/更差的配对**。若有，"配对策略"
就是一个可离线标定、输入是负载而非实现时序的调度自由度。

⚠️ **多 lane 臂会挂死，而且挂死不是慢——是 GPU 不再推进。**
本探针第一次运行（2026-09-26 上午）16 个配对臂全部跑完；此后每一次重跑都在中途挂死，
且挂死的臂从 `pair_224_672` 起把**后面所有臂都变成"0 完成"**。
`run_configuration` 现在会在 lane 线程不退出时**直接抛错**，而不是把 0 写成结果。
**结论只取单 lane 臂**（`--phases singles,batches`）；多 lane 臂的用途是记录约束边界，
不是产生对照数据。§24.1 的配对矩阵来自**那一次干净运行**，这一点必须与数字一起引用。

用法：
    .venv/bin/python scripts/probes/pilot_graph_pairing.py --duration 3

"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

#: 与 saturating_4ms.jsonl 一致的尺寸构成。
SIZES = (224, 336, 448, 672)
#: 捕获哪些批大小。batch=1 用于单流与配对臂；>1 用于成批臂。
BATCHES = (1, 2, 4, 8)
#: 每个 (尺寸, 批大小) 捕获几份互不共享静态缓冲的实例。
#: 批越大越占显存，因此大批判少配几份——本机 8 GB，实测能稳住。
INSTANCES = {1: 2, 2: 2, 4: 1, 8: 1}
#: 回放用的 CUDA 流，在池子构造时建一次、全程复用。
#:
#: ⚠️ 这**不是**对挂死的修复——试过，没用（挂死从第 8 个配置推到第 13 个，属于波动）。
#: 保留它只是因为它本来就该这么写（每个配置新建/丢弃流是没有理由的）。
#:
#: 已知的事实，供后续定位：全新进程只跑**一个**配置（哪怕是 2 lane × 4 尺寸）
#: 至今都能正常跑完；而同一个进程里连续跑十几个配置必挂。即挂死是
#: **进程内累积**的，触发量尚未定位（候选：累计回放次数、图私有池、
#: `_capture` 里为每份图各建一个 `side` 流且从不释放）。见 §24.5。
LANE_STREAMS = 4


class GraphPool:
    """按 (尺寸, 批大小) 捕获 CUDA Graph。"""

    def __init__(self, model, torch_mod, dtype, batches=BATCHES):
        self.torch = torch_mod
        self.dtype = dtype
        self.entries: dict[tuple[int, int], list[dict]] = {}
        self.capture_ms: dict[str, float] = {}
        # 流与事件都在这里建一次，跨配置复用；不要在每次 run_configuration 里新建。
        self.streams = [torch_mod.cuda.Stream() for _ in range(LANE_STREAMS)]
        self.events = [(torch_mod.cuda.Event(enable_timing=True),
                        torch_mod.cuda.Event(enable_timing=True)) for _ in range(LANE_STREAMS)]
        # 常驻 lane 线程，**在所有图捕获完之后**才起：捕获要求显存上没有在飞的图。
        self.lanes = [Lane(i, self, torch_mod) for i in range(LANE_STREAMS)]
        for size in SIZES:
            for batch in batches:
                copies = INSTANCES.get(batch, 1)
                start = time.perf_counter()
                self.entries[(size, batch)] = [
                    self._capture(model, dtype, size, batch) for _ in range(copies)]
                self.capture_ms[f"{size}x{size}x{batch}"] = (
                    time.perf_counter() - start) * 1000 / copies
            print(f"  已捕获 {size}×{size}：" + "、".join(
                f"batch{b} ×{INSTANCES.get(b, 1)}" for b in batches), flush=True)

    def _capture(self, model, dtype, size: int, batch: int) -> dict:
        torch_mod = self.torch
        static_in = torch_mod.empty((batch, 3, size, size), dtype=dtype, device="cuda")
        static_in.normal_()
        side = torch_mod.cuda.Stream()
        side.wait_stream(torch_mod.cuda.current_stream())
        with torch_mod.cuda.stream(side):
            for _ in range(3):
                with torch_mod.inference_mode():
                    model(pixel_values=static_in, interpolate_pos_encoding=True)
        torch_mod.cuda.current_stream().wait_stream(side)
        torch_mod.cuda.synchronize()

        graph = torch_mod.cuda.CUDAGraph()
        with torch_mod.inference_mode():
            with torch_mod.cuda.graph(graph):
                captured = model(pixel_values=static_in, interpolate_pos_encoding=True).image_embeds
        graph.replay()
        torch_mod.cuda.synchronize()
        with torch_mod.inference_mode():
            reference = model(pixel_values=static_in, interpolate_pos_encoding=True).image_embeds
        torch_mod.cuda.synchronize()
        diff = float((captured.detach().float() - reference.detach().float()).abs().max().item())
        if diff > 1e-2:
            raise SystemExit(f"{size}×{size} batch{batch} 的图与 eager 不符（差 {diff:.3e}）")
        return {"graph": graph, "input": static_in, "output": captured}


def verify_batches(pool: GraphPool, torch_mod, batches=BATCHES) -> dict[str, float]:
    """每个 batch>1 图的每一行，都应与同一输入的 batch-1 图输出一致。

    这既是正确性检查，也顺带回答"成批会不会改变数值"——fp16 下批量不同的
    归约顺序可能不同，差异要能看出来。
    """
    result = {}
    for size in SIZES:
        single = pool.entries[(size, 1)][0]
        single["input"].normal_()   # 具体取值无所谓：这里比的是"同一份输入下批量是否改变结果"
        torch_mod.cuda.synchronize()
        single["graph"].replay()
        torch_mod.cuda.synchronize()
        for batch in batches:
            if batch == 1:
                continue
            entry = pool.entries[(size, batch)][0]
            # 每一行灌同一份输入，因此每一行都应与 batch-1 的输出一致
            entry["input"].copy_(single["input"].expand_as(entry["input"]))
            torch_mod.cuda.synchronize()
            entry["graph"].replay()
            torch_mod.cuda.synchronize()
            ref = single["output"][0].detach().float()
            diffs = [
                float((entry["output"][row].detach().float() - ref).abs().max())
                for row in range(batch)
            ]
            # 判据用**相对 RMS** 而不是绝对差：fp16 下不同批大小会让 cuBLAS 选到不同的
            # kernel，逐元素差会随批量变化，而嵌入本身的量级也在变。绝对阈值会把
            # "换了 kernel" 误判成 "算错了"。
            result[f"{size}x{batch}"] = {
                "max_abs": max(diffs),
                "rms": float(ref.pow(2).mean().sqrt()),
                "relative": max(diffs) / max(float(ref.pow(2).mean().sqrt()), 1e-9),
            }
    return result


class Lane:
    """一条**常驻**回放线程。

    线程只创建一次、全程复用——这不是优化，是正确性要求。见 §24.5：
    反复创建并销毁"发过 CUDA 图回放"的 Python 线程，会在十几轮之后
    把驱动拖进不可恢复的挂死（两条 lane 都卡在 `stream.synchronize()`，
    GPU 永不完结）。最小复现：同样的 2-lane 配置，线程常驻时 30 轮全部正常；
    每轮新建线程时第 6–15 轮必挂；而**单轮跑 30 秒（约 9300 次回放）不挂**——
    即触发条件是线程建销次数，不是回放次数。

    服务侧本来就安全：`asyncio.to_thread` 复用默认线程池，`aprepare` 用常驻的
    `ThreadPoolExecutor`，都不会反复建销 CUDA 线程。挂死只出现在探针里。
    """

    def __init__(self, index: int, pool: "GraphPool", torch_mod):
        self.index = index
        self._tasks: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(pool, torch_mod), daemon=True)
        self._thread.start()

    def _run(self, pool: "GraphPool", torch_mod) -> None:
        stream = pool.streams[self.index]
        start_event, end_event = pool.events[self.index]
        while True:
            task = self._tasks.get()
            if task is None:
                return
            size, batch, duration_s, stop = task
            copies = pool.entries[(size, batch)]
            slot = copies[self.index % len(copies)]
            local: list[float] = []
            deadline = time.perf_counter() + duration_s
            while not stop.is_set() and time.perf_counter() < deadline:
                with torch_mod.cuda.stream(stream):
                    start_event.record(stream)
                    slot["graph"].replay()
                    end_event.record(stream)
                stream.synchronize()
                local.append(float(start_event.elapsed_time(end_event)))
            self._done.put(local)

    def run(self, size: int, batch: int, duration_s: float, stop: threading.Event) -> None:
        self._tasks.put((size, batch, duration_s, stop))

    def collect(self, timeout_s: float = 15.0) -> list[float]:
        try:
            return self._done.get(timeout=timeout_s)
        except queue.Empty:
            # 挂死必须变成明确失败，不能把 0 写成"吞吐为 0 的正常结果"（§21.2 的教训）。
            raise RuntimeError(
                f"lane {self.index} 在 {timeout_s}s 内未返回：并发图回放挂死（§24.5）。"
                "本次运行的数据不可用，不要落盘。"
            ) from None


def run_configuration(pool: GraphPool, torch_mod, duration_s: float,
                      lanes: list[tuple[int, int]]) -> dict:
    """``lanes`` 是 (尺寸, 批大小) 列表，每条一项 = 一条常驻 lane 连续回放该图。

    返回稳态吞吐与每条流的中位 span。lane 线程由 `pool.lanes` 持有、跨配置复用。
    """
    stop = threading.Event()
    for index, (size, batch) in enumerate(lanes):
        pool.lanes[index].run(size, batch, duration_s, stop)
    t0 = time.perf_counter()
    time.sleep(duration_s)
    stop.set()
    wall = time.perf_counter() - t0
    spans = [pool.lanes[index].collect() for index in range(len(lanes))]
    completed = sum(len(span) * batch for span, (_, batch) in zip(spans, lanes))
    return {
        "lanes": [f"{s}x{s}x{b}" for s, b in lanes],
        "requests": completed,
        "wall_s": wall,
        "throughput_rps": completed / wall,
        "span_median_ms": [round(statistics.median(v), 3) if v else None for v in spans],
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=3.0, help="每个配置的墙钟秒数")
    parser.add_argument("--batches", default="1,2,4,8", help="要捕获并测的批大小")
    parser.add_argument("--phases", default="singles,pairs,batches,lanes",
                        help="跑哪些族：singles / pairs / batches / lanes")
    parser.add_argument("--out", default="results/probes/pilot_graph_pairing.json")
    #: 只跑一个臂并单独落盘。**多 lane 并发回放在本机不可靠**（§24.5），
    #: 一个进程只跑一个配置是目前唯一能稳定取全数据的编排；
    #: 由调用方（run_all.sh 之类）逐配置起进程再合并。
    parser.add_argument("--only", default="", help="只跑这一个臂，例如 pair_224_672")
    args = parser.parse_args()

    batches = tuple(int(x) for x in args.batches.split(","))
    phases = set(args.phases.split(","))

    cfg = load_config("config.libsmctrl.example.yaml")
    # encoder 的流数只需 ≥ 被测的最大并发 lanes 数
    enc = ClipEncoderBackend(cfg.model, ProxyResourceBackend(), 2)
    print("捕获 CUDA Graph …", flush=True)
    pool = GraphPool(enc.model, enc.torch, enc.dtype, batches)
    diffs = verify_batches(pool, enc.torch, batches)
    for key, row in diffs.items():
        print(f"  {key:<8} 与 batch-1 同行最大差 {row['max_abs']:.3e}"
              f"（相对 RMS {row['relative']*100:.3f}%）")
        # 相对 RMS 超过 5% 就不再是"换了 kernel"能解释的，必须当成算错了。
        if row["relative"] > 0.05:
            raise SystemExit(f"{key} 的成批图与 batch-1 相差 {row['relative']*100:.1f}% RMS，"
                             "不能拿它做实验")
    print("正确性对照通过\n", flush=True)

    results: dict[str, dict] = {}

    def record(name: str, lanes: list[tuple[int, int]]) -> None:
        if args.only and name != args.only:
            return
        time.sleep(0.4)
        row = run_configuration(pool, enc.torch, args.duration, lanes)
        results[name] = row
        spans = " ".join("—" if v is None else f"{v:.2f}" for v in row["span_median_ms"])
        print(f"{name:>16}  吞吐 {row['throughput_rps']:7.1f} req/s  "
              f"完成 {row['requests']:6d}  各流 span(ms) {spans}", flush=True)

    if "singles" in phases:
        for size in SIZES:
            record(f"single_{size}", [(size, 1)])
        print()
    if "pairs" in phases:
        for a in SIZES:
            for b in SIZES:
                record(f"pair_{a}_{b}", [(a, 1), (b, 1)])
        print()
    if "batches" in phases:
        # 成批：同样处理 2 个请求，一次前向 vs 两条流——本探针的核心判决
        for size in SIZES:
            for batch in batches:
                if batch > 1:
                    record(f"batch{batch}_{size}", [(size, batch)])
        print()
    if "lanes" in phases:
        # 并发 × 成批是否可乘：两条流各跑 batch-B
        for size in SIZES:
            for batch in batches:
                if batch > 1 and INSTANCES.get(batch, 1) >= 2:
                    record(f"x2lane_batch{batch}_{size}", [(size, batch), (size, batch)])
        print()
        record("mix4", [(s, 1) for s in SIZES])
        for batch in batches:
            if batch > 1:
                record(f"mix4_batch{batch}", [(s, batch) for s in SIZES])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"results": results, "capture_ms": pool.capture_ms,
         "batch_vs_single_diff": diffs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
