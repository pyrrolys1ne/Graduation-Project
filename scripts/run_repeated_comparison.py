"""重复对照实验：基线与 DACC 消融，每组重复 N 次并报告标准差。

与 ``run_real_comparison.py`` 的区别：
- 每个变体重启服务、独立运行 N 次，得到可计算标准差的重复样本；
- 支持通过 ``dacc_overrides`` 做消融，不需要改代码；
- 输出含均值与标准差，便于判断策略之间是否真的可区分。

用法::

    python scripts/run_repeated_comparison.py --group baselines --repeats 5
    python scripts/run_repeated_comparison.py --group ablations --repeats 5
    python scripts/run_repeated_comparison.py --group window --repeats 5

正式对照之前**先跑预检**（只跑单流哨兵，环境不合格时退出码为 1）::

    python scripts/run_repeated_comparison.py --group concurrency --sentinel-only

注意：本脚本只验证软件调度闭环。资源后端为 ``proxy`` 时结果不含真实 SM 隔离效果，
输出中的 ``resource_backend`` 与 ``sm_isolation_verified`` 字段会如实标注。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
import statistics
import subprocess
import time

import httpx
import yaml

from encoder_sched.metrics import CLIENT_OVERHEAD_SHARE_WARN, client_side_summary

#: 每个变体 = 基础配置上的一处改动。消融项只改 DACC 评分权重。
GROUPS: dict[str, dict[str, dict]] = {
    "baselines": {
        "serial_fcfs": {"policy": "serial_fcfs"},
        "multistream_fcfs": {"policy": "multistream_fcfs"},
        "edf": {"policy": "edf"},
        "edf_size": {"policy": "edf_size"},
        "dacc": {"policy": "dacc"},
    },
    "ablations": {
        # 最有诊断价值的一项：禁止配对，DACC 退化为"只选单请求"。
        # 若它与 edf 表现相近，即证明损害来自配对本身而非评分函数。
        "dacc_no_pairing": {"policy": "dacc", "overrides": {"max_pair_conflict": 0.0}},
        "dacc_no_complementarity": {"policy": "dacc", "overrides": {"w_complementarity": 0.0}},
        "dacc_no_uncertainty": {"policy": "dacc", "overrides": {"w_uncertainty": 0.0, "beta": 0.0}},
        "dacc_no_risk": {"policy": "dacc", "overrides": {"w_risk": 0.0}},
        "dacc_no_urgency": {"policy": "dacc", "overrides": {"w_urgency": 0.0}},
    },
    "window": {
        "dacc_k1": {"policy": "dacc", "overrides": {"window": 1}},
        "dacc_k4": {"policy": "dacc", "overrides": {"window": 4}},
        "dacc_k8": {"policy": "dacc"},
        "dacc_k16": {"policy": "dacc", "overrides": {"window": 16}},
    },
    # 并发度控制：静态扫描 vs 动态控制。
    #
    # 为什么必须有静态扫描：只看"动态 vs 固定 2"无法区分"动态控制有效"与
    # "并发 4 恰好更好"。判据是 **dynamic 必须优于最佳静态档**，这条对照臂
    # 就是为此存在的（本课题的 DACC 教训：复杂度的正当性只能靠打赢朴素对照获得）。
    #
    # 静态臂一律用 multistream_fcfs（本课题最强且最朴素的基线），只改 streams；
    # 动态臂用同一策略，改由 controller 逐请求决定共驻数。
    "concurrency": {
        "static_1": {"policy": "multistream_fcfs", "streams": 1, "adaptive": False},
        "static_2": {"policy": "multistream_fcfs", "streams": 2, "adaptive": False},
        "static_3": {"policy": "multistream_fcfs", "streams": 3, "adaptive": False},
        "static_4": {"policy": "multistream_fcfs", "streams": 4, "adaptive": False},
        "static_6": {"policy": "multistream_fcfs", "streams": 6, "adaptive": False},
        "static_8": {"policy": "multistream_fcfs", "streams": 8, "adaptive": False},
        "dynamic": {"policy": "multistream_fcfs", "adaptive": True},
    },
    # 并发度方向的**管线对照**（2026-09-22 新增）：正式批次复盘发现，dynamic 臂的
    # 控制器按 `max_concurrency=8` 铺 worker 池，但编码器只建了 `executor.streams`(=2)
    # 条 CUDA 流——"6 个 resident"其实是 6 个线程抢 2 条流。于是"dynamic 被支配"
    # 这个结论里混进了一个与控制器无关的因素。本组用三臂把它分开：
    #   fixed2_streams2  ：固定目标 2、2 条流 → 应复现 static_2（证明管线本身没问题）
    #   fixed6_streams2  ：固定目标 6、2 条流 → 单独看"流池没铺开"的代价
    #   dynamic_streams8 ：控制器照常决策、流池铺满 → 这才是控制律的公平检验
    "plumbing": {
        "fixed2_streams2": {
            "policy": "multistream_fcfs", "adaptive": True, "streams": 2,
            "concurrency": {"min_concurrency": 2, "max_concurrency": 2,
                            "small_mix_concurrency": 2, "large_mix_concurrency": 2,
                            "enable_drift_fallback": False},
        },
        "fixed6_streams2": {
            "policy": "multistream_fcfs", "adaptive": True, "streams": 2,
            "concurrency": {"min_concurrency": 6, "max_concurrency": 6,
                            "small_mix_concurrency": 6, "large_mix_concurrency": 6,
                            "enable_drift_fallback": False},
        },
        "dynamic_streams8": {
            "policy": "multistream_fcfs", "adaptive": True, "streams": 8
        },
    },
    # 配对倾向扫描：消除结构性偏向之后，配对能否胜出完全由权重决定。
    # w_complementarity 越小越愿意配对；w_gain 越大越看重"一次服务两个"的吞吐收益。
    # 这组实验直接回答"配对机制是否值得保留"，不靠直觉调参。
    "pairing": {
        "dacc_wc1.0": {"policy": "dacc"},
        "dacc_wc0.5": {"policy": "dacc", "overrides": {"w_complementarity": 0.5}},
        "dacc_wc0.0": {"policy": "dacc", "overrides": {"w_complementarity": 0.0}},
        "dacc_gain2": {"policy": "dacc", "overrides": {"w_gain": 2.0}},
    },
    # 并发方式对照：空间分割（多 worker/多 stream 各跑一个请求）vs 批处理（合并成一次前向）。
    # 两臂都是 FCFS 排序、同一资源后端，只改并发方式。
    #
    # 主轴是 `delay_ms`——"愿意等多久来凑批"。它直接决定这条取舍曲线：
    #   delay=0  绝不等待，只取队列里已有的（等价于逐请求，拿不到批处理的好处）
    #   delay 增大 → batch 变大 → 吞吐上升，但每个请求的首字节延迟也随之上升
    # 产品文档给的参考是交互式场景下队列延迟压到 100–300 µs（见 Triton 的
    # max_queue_delay_microseconds），本组从 0 扫到 20 ms 是为了把整条曲线画出来。
    "batch": {
        "spatial_2stream": {"policy": "multistream_fcfs", "batch": 1},
        "batch4_d0": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 0.0},
        "batch4_d1": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 1.0},
        "batch4_d5": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 5.0},
        "batch4_d20": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 20.0},
        "batch8_d5": {"policy": "serial_fcfs", "batch": 8, "delay_ms": 5.0},
    },
    # 方法对照（M1）：执行粒度三档。**本组是本课题的落点。**
    #
    # 参照系是任务书规定的两个——**简单的串行执行**与**固定资源分配**：
    #   eager_serial  ：单 worker（串行），即"什么都不做"这一档
    #   eager_2stream ：固定 2 条流。这是 §21.3 用 7 臂 × 5 轮开环对照确立的**最佳 eager 档**
    #   graph_slots1  ：图回放流水线（方法候选，§23/§25）
    #
    # 三臂的策略**全部是 multistream_fcfs**，只改执行机制，因此差异可归因到机制本身
    # 而不是排序规则。判据：`graph_slots1` 相对 `eager_2stream` 与 `eager_serial`
    # 的**逐轮配对差**为正、且每轮哨兵 ≤1.5×。
    #
    # ⚠️ 本组只能用 `saturating_180rps.jsonl` 这一档（见 GROUP_DEFAULT_WORKLOAD）：
    # 图回放的容量是 222 req/s，更低的到达率两个臂都跟得上、吞吐被到达率钉死；
    # 更高的档位会先把**客户端**压垮（§26.4，D3），测到的不是服务端。
    #
    # `graph_batch` 是本组的方法臂之二：在"何时并发"之外加上"**何时合并**"。
    # 批上限**按尺寸给**，取自 §24.2 的实测收益（单流 B=4 相对 B=1）：
    #     224 2.42× / 336 1.79× / 448 1.38× / 672 1.16×
    # 小请求批次内填空转的空间大、大请求几乎没有，而批越大单批墙钟越长
    # （672 的 B=4 是 19.2 ms 对 B=1 的 5.4 ms，尾延迟风险实打实）。
    # 因此 224/336 取 4、448 取 2、672 不批——**这是机制不是调参**。
    # 组批**绝不等待**（只取队列里已有的），所以这些上限只封顶、不引入等待成本。
    "graph": {
        "eager_serial": {"policy": "multistream_fcfs", "streams": 1},
        "eager_2stream": {"policy": "multistream_fcfs", "streams": 2},
        "graph_slots1": {"policy": "multistream_fcfs",
                         "graph": {"slots": 1, "prep_threads": 2}},
        "graph_batch": {"policy": "multistream_fcfs",
                        "graph": {"slots": 1, "prep_threads": 2,
                                  "batch_by_size": {224: 4, 336: 4, 448: 2, 672: 1}}},
    },
    # 真实 TPC 分区 vs 批处理（需 LIBSMCTRL_PATH 指向带粘性线程掩码补丁的库）。
    #
    # 两臂都必须占满 12 个 TPC，否则比的就不是"并发方式"而是"用了多少资源"：
    #   spatial 臂：两个 worker 各申请 0.5 → 分配器给出互斥的两个 6-TPC 区间
    #   batch   臂：单 worker 用满 1.0 → 整卡
    # libsmctrl 臂强制 allow_proxy_fallback=false，避免静默退回 proxy 后仍被当作分区结果。
    "spatial": {
        "spatial_tpc": {"policy": "multistream_fcfs", "quota_levels": [0.5], "backend": "libsmctrl"},
        "batch4_d0_tpc": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 0.0,
                          "quota_levels": [1.0], "backend": "libsmctrl"},
        "batch4_d5_tpc": {"policy": "serial_fcfs", "batch": 4, "delay_ms": 5.0,
                          "quota_levels": [1.0], "backend": "libsmctrl"},
        # 参照：同一策略在 proxy 下（无掩码、两条流共享全部 SM），用于隔离"分区"与"并发"两种效应
        "spatial_proxy": {"policy": "multistream_fcfs", "quota_levels": [0.5, 1.0], "backend": "proxy"},
    },
    # 配额选择策略对照：deadline_min（取满足截止期的最小配额）vs full（始终用满）。
    # 动机见 encoder_sched/scheduler.py 的 choose_quota 说明。
    "quota": {
        "dacc_deadline_min": {"policy": "dacc", "quota_policy": "deadline_min"},
        "dacc_full": {"policy": "dacc", "quota_policy": "full"},
        "edf_size_deadline_min": {"policy": "edf_size", "quota_policy": "deadline_min"},
        "edf_size_full": {"policy": "edf_size", "quota_policy": "full"},
    },
    # 换表 A/B 的最小臂集（配合 scripts/experiment_table_swap_ab.py）。
    #
    # 一个阳性臂 + 一个阴性对照，缺了对照就说不清"差异来自标定表"还是"来自任何两次运行
    # 之间都会有的漂移"：
    #   edf_size  **读表**：choose_quota 会查表选配额，rank_key 也用 predicted_ms
    #   edf       **不读表**：choose_quota 对非 edf_size/dacc 直接返回满配额
    "table_ab": {
        "edf_size_deadline_min": {"policy": "edf_size", "quota_policy": "deadline_min"},
        "edf": {"policy": "edf"},
    },
}

METRICS = (
    "throughput_rps",
    # ---- 服务端口径：自请求到达服务端起算，不含网络与客户端连接池等待 ----
    "server_total_mean_ms",
    "server_total_p50_ms",
    "server_total_p95_ms",
    "server_total_p99_ms",
    # 排队/执行拆分：诊断"瓶颈在 GPU 还是在服务端排队"必需，不能只看 total。
    "server_queue_mean_ms",
    "server_execution_mean_ms",
    # ---- 客户端口径：含连接池等待与 HTTP 往返，是用户实际承受的端到端延迟 ----
    # 本课题实测两侧相差 15–34 倍（见 AGENTS.md 待办第 8 条）。历史上正是把
    # 服务端 total_ms 当成"端到端延迟"引用，才让"多流保护尾延迟"的结论必须撤回。
    # 两个口径同表并列、字段名自带前缀，杜绝再次混用。
    "client_total_mean_ms",
    "client_total_p50_ms",
    "client_total_p95_ms",
    "client_total_p99_ms",
    # 服务端之外的部分：连接池人为排队只会出现在这里，不会出现在服务端队列里。
    "client_overhead_mean_ms",
    "client_overhead_p95_ms",
    # 连接池等待的估计（overhead 超出本轮纯往返基准的部分）。
    "client_pool_wait_p95_ms",
    # 客户端实际发出时刻相对负载规定到达时刻的迟到量：
    # "并发 8 的连接池是否人为制造排队"的直接判据。
    "client_dispatch_lateness_p95_ms",
    "slo_violation_rate",
    "client_slo_violation_rate",
    "gpu_utilization_percent",
    "gpu_samples",
)

#: 单流哨兵的污染阈值：实测中位 / 预测中位 超过它即判定环境不合格。
#: 与 ``ConcurrencyConfig.drift_threshold`` 一致——本课题的标定表本身有 ±35%
#: 级别的会话间波动，超过 1.5× 就超出了"噪声"能解释的范围。
#: 每组的默认负载。**不给默认会让历史上那个头号陷阱重演**：到达率低于服务容量时，
#: 所有臂的吞吐都被到达率钉在同一个值上，**无法区分任何调度策略**（§3 的教训——
#: 早期 `comparison_mixed.jsonl` 的 57.8 req/s 与服务率 57.4 几乎相等，
#: 四个基线的差异全在 ±0.07 噪声内）。
#: 组默认值随容量变化：图回放的容量是 222 req/s（§25），故 `graph` 组用 180rps 档。
GROUP_DEFAULT_WORKLOAD = {
    # graph 组默认用**过载档**，不是接近容量档。
    # 理由（§28.3 实测）：`graph_batch` 臂只在队列有积压时才凑得成批——
    # 档位 1（167.9 req/s）下回放 lane 容量 222 高于 offering，队列基本是空的，
    # **平均批大小恒为 1.00**，成批臂与不成批臂完全一样。在那一档上跑它
    # 只会得到"两臂相同"的零结果，而那个零结果**不是**"成批无效"的证据。
    "graph": "data/workloads/saturating_280rps.jsonl",
    "concurrency": "data/workloads/saturating_4ms.jsonl",
}
DEFAULT_WORKLOAD = "data/workloads/comparison_mixed.jsonl"

SENTINEL_THRESHOLD = 1.5

#: 预检必须覆盖的四种尺寸。与 ``data/profiles/default.csv`` 同一组采样点：
#: "可复现性有尺寸依赖"是本课题的核心发现，少覆盖一档，"合格"这个结论
#: 就不能外推到那一档。
SENTINEL_SIZES = ((224, 224), (336, 336), (448, 448), (672, 672))


def wait_ready(url: str, process: subprocess.Popen, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"服务进程提前退出，返回码 {process.returncode}")
        try:
            response = httpx.get(f"{url}/health", timeout=2)
            if response.is_success:
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError("等待服务启动超时")


def build_config(base: dict, variant: dict, workdir: Path) -> Path:
    data = json.loads(json.dumps(base))
    scheduler = data.setdefault("scheduler", {})
    scheduler["policy"] = variant["policy"]
    if variant.get("quota_policy"):
        scheduler["quota_policy"] = variant["quota_policy"]
    if variant.get("overrides"):
        scheduler["dacc_overrides"] = variant["overrides"]
    if variant.get("batch") is not None:
        data["batching"] = {
            "max_batch": variant["batch"],
            "max_delay_ms": variant.get("delay_ms", 2.0),
        }
    if variant.get("quota_levels"):
        scheduler["quota_levels"] = list(variant["quota_levels"])
    if variant.get("streams") is not None:
        data.setdefault("executor", {})["streams"] = variant["streams"]
    if variant.get("adaptive") is not None:
        executor = data.setdefault("executor", {})
        executor["adaptive_concurrency"] = bool(variant["adaptive"])
        if variant.get("concurrency"):
            executor["concurrency"] = variant["concurrency"]
    if variant.get("backend"):
        executor = data.setdefault("executor", {})
        executor["resource_backend"] = variant["backend"]
        if variant["backend"] == "libsmctrl":
            # 真实掩码实验禁止静默回退到 proxy：一旦回退，这一轮测的就不是 SM 分区，
            # 而结果里不会留下任何异常痕迹。宁可启动失败。
            executor["allow_proxy_fallback"] = False
            # 基础配置里 libsmctrl_adapter 为空；libsmctrl 臂必须显式指定适配器，
            # 否则 build_service 会直接报 "需要配置 libsmctrl_adapter=模块:函数" 并拒绝启动。
            if not executor.get("libsmctrl_adapter"):
                executor["libsmctrl_adapter"] = "encoder_sched.libsmctrl_adapter:apply_quota"
    if variant.get("graph"):
        executor = data.setdefault("executor", {})
        graph = dict(variant["graph"])
        # 三条互斥与一条上界关系必须在**这里**也拦一道。config.load_config 已经会拦，
        # 但那时错误发生在每个臂启动服务的时候——一批 5 轮 × 7 臂会连续失败 35 次，
        # 而这里能一次性说清楚是哪个臂配错了。
        if int(graph.get("slots", 1)) > int(executor.get("streams", 2)):
            raise ValueError(
                f"臂的 graph.slots({graph.get('slots')}) 大于 executor.streams"
                f"({executor.get('streams')})：回放槽位按 slot_index 选流，流不够时"
                "两个槽位会挤在同一条流上、并发被静默压回流数。"
            )
        if executor.get("resource_backend") == "libsmctrl":
            raise ValueError("graph 臂与 libsmctrl 互斥：掩码回调在 cuGraphLaunch 上不生效。")
        if data.get("batching", {}).get("max_batch", 1) > 1:
            raise ValueError("graph 臂与批处理互斥：本版图不含批维。")
        if executor.get("adaptive_concurrency"):
            raise ValueError("graph 臂与 adaptive_concurrency 互斥：并发度只能有一个来源。")
        executor["graph_pipeline"] = True
        executor["graph"] = graph
    log_dir = workdir / "logs"
    data.setdefault("logging", {})["output_dir"] = str(log_dir.resolve())
    profile = Path(data.get("profiling", {}).get("table_path", "data/profiles/default.csv"))
    if not profile.is_absolute():
        profile = (Path.cwd() / profile).resolve()
    data.setdefault("profiling", {})["table_path"] = str(profile)
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_once(config_path: Path, workload: Path, output: Path, concurrency: int, warmup: int,
             port: int, shards: int = 1) -> dict:
    env = {**os.environ, "ENCODER_SCHED_CONFIG": str(config_path.resolve())}
    log = config_path.parent / "server.log"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "encoder_sched.api", "--port", str(port)],
            stdout=handle, stderr=subprocess.STDOUT, env=env,
            # **降低服务端进程的调度优先级**。这不是调参，是测量台的共置缺陷补偿：
            # 负载生成器与服务端同机时，服务端自己的 CPU 负载（预处理线程跑 torch.rand）
            # 会把客户端的事件循环饿死，表现为"发出迟到"尾部越线、进而抑制整批判定。
            # 已实测该机制：另起 3 个忙循环进程即可把客户端发出迟到 P99 从 5 ms 抬到 522 ms
            # （§27）。真实部署里客户端不在服务端机器上，所以这里是**补偿而不是作弊**——
            # 它对所有臂一视同仁，且不改变被测对象的代码与配置。
            preexec_fn=lambda: os.nice(10),
        )
    try:
        health = wait_ready(f"http://127.0.0.1:{port}", process)
        subprocess.run(
            [sys.executable, "scripts/run_experiment.py", "--url", f"http://127.0.0.1:{port}",
             "--workload", str(workload), "--output", str(output),
             "--concurrency", str(concurrency), "--warmup", str(warmup),
             "--shards", str(shards)],
            check=True, env=env,
        )
        summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
        summary["gpu"] = health.get("environment", {}).get("gpu")
        summary["resource"] = health.get("resource")
        return summary
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def paired_differences(
    results: dict[str, dict],
    baseline: str,
    metrics: tuple[str, ...] = ("throughput_rps", "server_total_p99_ms", "client_total_p99_ms"),
) -> dict[str, dict]:
    """逐轮**配对差**：同一轮内两个臂相减，再看差值的均值与符号一致性。

    为什么必须配对：本课题的方法学要求是"两批都显著才写进结论"，而
    ``demo_dual_freedom.py`` 的教训是**用单次聚合值做判据会让结论在同会话重跑后翻转**
    （同一个脚本两次运行给出"打平"与"通过"两种相反判定，而三次干净重复的配对比值
    是 0.998/1.000/0.986）。配对差把"同一轮的会话漂移"消掉——两个臂在同一次
    服务启动的相邻时刻测量，共享大部分环境噪声。

    返回每臂相对 ``baseline`` 的：
      ``mean``       配对差的均值
      ``std``        配对差的标准差
      ``sign_ratio`` 差值为正的比例（1.0 表示**每一轮**都朝同一方向）
      ``consistent`` 符号是否完全一致——**这是比 p 值更硬的判据**，
                     因为在 n=5 的小样本下 t 检验的功效本来就低。
    """
    if baseline not in results:
        raise KeyError(f"配对基准 {baseline!r} 不在结果里；可选 {sorted(results)}")
    base_samples = results[baseline]["samples"]
    output: dict[str, dict] = {}
    for name, item in results.items():
        if name == baseline:
            continue
        samples = item["samples"]
        if len(samples) != len(base_samples):
            # 轮数不齐时配对没有意义——直接拒绝，而不是静默截断。
            raise ValueError(f"{name} 的轮数 {len(samples)} 与基准 {len(base_samples)} 不一致")
        per_metric: dict[str, dict] = {}
        for metric in metrics:
            # 吞吐"越高越好"，延迟"越低越好"，因此统一成"正数=更好"的方向。
            sign = -1.0 if metric.endswith("_ms") else 1.0
            diffs = [sign * (s[metric] - b[metric]) for s, b in zip(samples, base_samples)]
            positive = sum(1 for d in diffs if d > 0)
            per_metric[metric] = {
                "mean": statistics.fmean(diffs),
                "std": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                "sign_ratio": positive / len(diffs),
                "consistent": positive == len(diffs) or positive == 0,
                "diffs": diffs,
            }
        output[name] = per_metric
    return output


def contamination_check(jsonl_path: Path) -> dict:
    """用独立的**单流哨兵**判断这一轮是否被环境污染。

    为什么需要它：本课题已有一次因 Windows 宿主进程占用 GPU 而整轮作废的记录。
    WSL 内的 ``nvidia-smi`` **看不到宿主进程**，所以"开机前查一眼利用率"是
    不可靠的守门方式——2026-09-17 那一轮就是这么漏过去的（GPU 显示 1% 空闲，
    但 CLIP 单请求实测 9.8 ms 而标定表写 4.9 ms）。

    调用方必须保证日志来自 ``streams=1``、关闭自适应、客户端并发为 1 的哨兵。
    不能读取正式臂日志：高并发下 ``execution_ms`` 包含策略自身的共驻争用，而
    ``predicted_ms`` 来自单请求剖析表，直接比较会随并发度自然增大并制造假阳性。

    返回的 ``ratio`` 是**按尺寸分档统计后取最差档**的中位比值，因为本课题的
    核心发现正是可复现性有尺寸依赖，池化会把小尺寸的失真平均掉。
    """
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_size: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        observed = row.get("execution_ms")
        predicted = row.get("predicted_ms")
        if observed and predicted:
            by_size.setdefault((row.get("width", 0), row.get("height", 0)), []).append(observed / predicted)
    worst = 0.0
    per_size: dict[str, float] = {}
    for size, ratios in sorted(by_size.items()):
        median_ratio = statistics.median(ratios)
        per_size[f"{size[0]}x{size[1]}"] = median_ratio
        worst = max(worst, median_ratio)
    return {
        "worst_ratio": worst,
        "per_size": per_size,
        "samples": len(rows),
        "contaminated": worst > SENTINEL_THRESHOLD,
    }


def sentinel_variant(variant: dict) -> dict:
    """从正式臂派生单流环境哨兵，保留后端设置但关闭批处理与并发控制。"""
    sentinel = dict(variant)
    sentinel.update(policy="multistream_fcfs", streams=1, adaptive=False)
    sentinel.pop("batch", None)
    # 哨兵是**eager 单流**的环境参照系，必须剥掉图回放——否则量到的是图路径的
    # 执行时间，与单请求标定表不可比，环境污染判定会整体失真。
    sentinel.pop("graph", None)
    sentinel.pop("delay_ms", None)
    sentinel.pop("overrides", None)
    return sentinel


def workload_sizes(path: Path) -> set[tuple[int, int]]:
    """负载里出现过的 (width, height) 集合。"""
    sizes: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                sizes.add((row.get("width"), row.get("height")))
    return sizes


def workload_request_count(path: Path) -> int:
    """负载的请求条数，用于把客户端连接上限开到"永不节流"（开环）。"""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_client_concurrency(requested: int, workload: Path) -> int:
    """把 ``--concurrency`` 解析成真实的客户端连接上限。

    ``0``（默认）表示**开环**：连接上限 = 负载请求数，客户端绝不因为上一次响应还没
    回来而推迟下一次发送。这不是调参，而是这批实验有效性的前提。

    为什么必须开环：请求的到达过程是自变量。若客户端连接上限固定为 8，而某个臂的
    服务端延迟是 150 ms，那么该臂的实测吞吐会被**客户端**钉在 8/0.15 ≈ 53 req/s，
    与调度器无关。2026-09-22 那批 7 臂 × 5 次实测正是如此：每个臂的客户端延迟都是
    服务端的 10–40 倍，吞吐恰好等于"连接上限 / 服务端延迟"，所谓"饱和负载"从未真正
    压到服务端（证据保留在 ``results/concurrency_client_throttled_2026-09-22/``）。

    上限调到负载条数后，服务端看到的就是负载规定的到达过程，超出容量的部分积压在
    服务端队列里——这才是这批对照要测的东西。
    """
    if requested > 0:
        return requested
    return max(1, workload_request_count(workload))


def unique_sentinels(base: dict, variants: dict, out_dir: Path) -> list[dict]:
    """按哨兵配置内容去重，一个唯一配置 = 一次预检。

    concurrency 组的 7 个臂只派生得出少数几个不同的哨兵：静态臂之间只差
    ``streams``，而哨兵本来就固定 ``streams=1``；只有资源后端不同的臂才会
    派生出不重复的哨兵。不去重就要把同一次预检跑 7 遍，而预检的全部意义
    恰恰是"别在环境可能不合格时启动整批实验"。
    """
    seen: dict[str, dict] = {}
    for name, variant in variants.items():
        sentinel = sentinel_variant(variant)
        key = json.dumps(sentinel, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            workdir = out_dir / f"sentinel_{len(seen)}"
            workdir.mkdir(parents=True, exist_ok=True)
            seen[key] = {
                "arms": [],
                "variant": sentinel,
                "workdir": workdir,
                "config": build_config(base, sentinel, workdir),
            }
        seen[key]["arms"].append(name)
    return list(seen.values())


def run_sentinel_only(args: argparse.Namespace, base: dict, variants: dict, out_dir: Path) -> int:
    """环境预检：只跑单流哨兵，给出可判定的通过 / 不通过。

    为什么必须能单独成命令：预检的全部价值在于**在启动整批实验之前**中止。
    此前预检只有"挂在每个正式臂前面"这一种形态，环境不合格也要先跑完
    7 臂 × 5 轮才会发现——本课题已有两批对照实验正是这样整批作废的
    （``results/concurrency_fixed_2026-09-19/`` 的 70 次运行全部作废）。

    返回退出码：0 = 环境合格，1 = 有尺寸超过阈值。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [f"{w}x{h}" for (w, h) in SENTINEL_SIZES if (w, h) not in workload_sizes(args.workload)]
    if missing:
        print(f"⚠️ 负载 {args.workload} 不含尺寸 {missing}，预检不覆盖这些档位")

    specs = unique_sentinels(base, variants, out_dir)
    repeats = max(1, args.sentinel_repeats)
    print(f"环境预检：{len(specs)} 个唯一哨兵配置 × {repeats} 次"
          f"（负载 {args.workload}，客户端并发 1，阈值 {SENTINEL_THRESHOLD}×）")
    if repeats == 1:
        # 这不是保守起见的空话：2026-09-19 实测同一条命令间隔 90 秒给出
        # 1.34×（合格）与 1.58×（不合格）两种相反判定，最差尺寸都是 448。
        # 单次预检因此是"抽一次签"，而不是"测一个状态"。
        print("⚠️ 只跑 1 次：本机实测单次预检可在 90 秒内翻转判定，"
              "建议加 --sentinel-repeats 3 再下结论")

    report: list[dict] = []
    for index, spec in enumerate(specs):
        per_size_runs: dict[str, list[float]] = {}
        for repeat in range(repeats):
            output = spec["workdir"] / f"repeat{repeat}.jsonl"
            print(f"[sentinel_{index}] repeat {repeat + 1}/{repeats} "
                  f"(覆盖臂 {', '.join(spec['arms'])}) ...", flush=True)
            # 哨兵固定为**单进程、客户端并发 1**：它的语义是"逐请求、无并发"，
            # 分片会把这个语义破坏掉（N 个进程等于 N 份并发）。
            run_once(spec["config"], args.workload, output, 1, args.warmup, args.port, shards=1)
            check = contamination_check(output)
            for size, ratio in check["per_size"].items():
                per_size_runs.setdefault(size, []).append(ratio)
        # 与正式臂汇总同一纪律：跨轮取 max 而不是 mean。一轮污染就足以让整批对照
        # 失去意义，而均值会被其余干净轮次稀释到阈值以下——那等于放过污染。
        per_size = {size: max(ratios) for size, ratios in sorted(per_size_runs.items())}
        worst = max(per_size.values(), default=0.0)
        report.append({
            "arms": spec["arms"],
            "variant": spec["variant"],
            "per_size": per_size,
            "worst_ratio": worst,
            "contaminated": worst > SENTINEL_THRESHOLD,
        })

    # 覆盖臂单独列成图例而不是塞进表格列：concurrency 组有 7 个臂，名字长到会把
    # 列宽撑破，而截断恰好会隐去"哪些臂被这次预检覆盖"——那正是要读的信息。
    print("\n覆盖臂（同一哨兵配置的臂共享一次预检）：")
    for index, item in enumerate(report):
        print(f"  sentinel_{index} = {', '.join(item['arms'])}")

    columns = sorted({size for item in report for size in item["per_size"]})
    print(f"\n{'哨兵':<12}" + "".join(f"{size:>10}" for size in columns) + f"{'最差':>10}")
    for index, item in enumerate(report):
        cells = "".join(f"{item['per_size'].get(size, float('nan')):>10.2f}" for size in columns)
        print(f"{'sentinel_' + str(index):<12}{cells}{item['worst_ratio']:>10.2f}")

    (out_dir / "sentinel_check.json").write_text(
        json.dumps({
            "group": args.group,
            "workload": str(args.workload),
            "repeats": repeats,
            "threshold": SENTINEL_THRESHOLD,
            "missing_sizes": missing,
            "sentinels": report,
            "passed": not any(item["contaminated"] for item in report),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果写入 {out_dir}/sentinel_check.json")

    bad = [(index, item) for index, item in enumerate(report) if item["contaminated"]]
    if bad:
        print(f"\n❌ 环境不合格：{len(bad)}/{len(report)} 个哨兵有尺寸超过 {SENTINEL_THRESHOLD}×，整批对照不可启动")
        for index, item in bad:
            worst_size = max(item["per_size"], key=item["per_size"].get)
            print(f"      sentinel_{index}（臂 {', '.join(item['arms'])}）"
                  f"{worst_size} = {item['per_size'][worst_size]:.2f}×")
        print("     处置：确认 Windows 侧没有占用 GPU 的进程、电源与温度状态稳定后，重跑本命令。")
        print("     注意 WSL 内的 nvidia-smi 看不到宿主机进程，只看利用率不足以判断整机空闲。")
        return 1
    print(f"\n✅ 环境合格：所有尺寸的单流哨兵均未超过 {SENTINEL_THRESHOLD}×，可以启动正式对照")
    return 0


def flatten(summary: dict) -> dict:
    """把一份实验汇总摊平成单轮样本，**两个口径分别命名**。

    ``server_*`` 来自服务端 ``/metrics/summary``（自请求到达服务端起算）；
    ``client_*`` 来自客户端回放的逐请求观测（含连接池等待与 HTTP 往返）。
    客户端块缺失时直接报错，而不是退回单口径——静默退回正是本课题历史上
    "服务端与客户端混用"的成因。
    """
    client = summary.get("client_scope")
    if not isinstance(client, dict):
        raise KeyError(
            "实验汇总缺少 client_scope 块：无法区分服务端与客户端口径。"
            "请确认 scripts/run_experiment.py 是最新版本（它会写入客户端口径）。"
        )
    return {
        "throughput_rps": summary["throughput_rps"],
        "server_total_mean_ms": summary["total_latency_ms"]["mean"],
        "server_total_p50_ms": summary["total_latency_ms"]["p50"],
        "server_total_p95_ms": summary["total_latency_ms"]["p95"],
        "server_total_p99_ms": summary["total_latency_ms"]["p99"],
        # 排队与执行的拆分必须单独保留：二者的相对大小决定了"瓶颈在 GPU 还是在
        # 客户端连接池"，而本课题已有一处结论（多流保护尾延迟）正是因为只看服务端
        # 汇总、没看这个拆分而必须撤回。缺了这两列，诊断只能靠翻原始 JSONL。
        "server_queue_mean_ms": summary.get("queue_latency_ms", {}).get("mean"),
        "server_execution_mean_ms": summary.get("execution_latency_ms", {}).get("mean"),
        "client_total_mean_ms": client["latency_ms"]["mean"],
        "client_total_p50_ms": client["latency_ms"]["p50"],
        "client_total_p95_ms": client["latency_ms"]["p95"],
        "client_total_p99_ms": client["latency_ms"]["p99"],
        "client_overhead_mean_ms": client["overhead_ms"]["mean"],
        "client_overhead_p95_ms": client["overhead_ms"]["p95"],
        "client_pool_wait_p95_ms": client["pool_wait_ms"]["p95"],
        "client_dispatch_lateness_p95_ms": client["dispatch_lateness_ms"]["p95"],
        "client_slo_violation_rate": client["slo_violation_rate"],
        "client_queueing_flagged": bool(client["client_queueing_flagged"]),
        "slo_violation_rate": summary["slo_violation_rate"],
        "gpu_utilization_percent": summary["gpu_utilization_percent"],
        "gpu_samples": summary.get("gpu_samples"),
        "completed": summary["completed"],
        "failed": summary["failed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重复对照实验（含 DACC 消融）")
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--workload", type=Path, default=None,
        help="回放负载。不给则按组取默认（见 GROUP_DEFAULT_WORKLOAD）——"
             "默认值经过挑选，保证到达率高于该组服务容量，否则所有臂的吞吐都会被"
             "到达率钉死、无法区分策略。",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="客户端连接上限。0（默认）= 开环：上限取负载请求条数，客户端不因响应慢而"
             "节流下一次发送。给正数则固定上限，被客户端卡住时会自动抑制策略判定。",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--shards", type=int, default=4,
        help="把回放发送拆到几个客户端进程。默认 4——**这不是调参**：单个 Python 事件循环"
             "会被整机 CPU 负载饿死，而服务端自己的预处理线程就是那个负载，"
             "过载档上单进程客户端会先垮（§26.4/§27）。分片让客户端不成为瓶颈。",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--baseline",
        default=None,
        help="配对基准臂名。给了就额外出逐轮配对差（见 paired_differences）。"
        "concurrency 组默认用最佳静态档，其它组不启用。",
    )
    parser.add_argument(
        "--sentinel-only",
        action="store_true",
        help="只跑环境预检哨兵，不跑正式臂。任一尺寸的单流 实测/预测 超阈值时退出码为 1，"
        "用于在启动整批实验前中止。",
    )
    parser.add_argument(
        "--sentinel-repeats",
        type=int,
        default=1,
        help="预检模式下每个唯一哨兵配置的重复次数。",
    )
    parser.add_argument(
        "--reanalyze",
        type=Path,
        default=None,
        help="不跑实验，用当前判定规则重新判定一个已落盘的批次目录"
             "（读 repeat*.summary.json 与 sentinel/ 日志）。",
    )
    args = parser.parse_args()

    if args.reanalyze:
        sys.exit(reanalyze(args, args.reanalyze))

    if args.workload is None:
        args.workload = Path(GROUP_DEFAULT_WORKLOAD.get(args.group, DEFAULT_WORKLOAD))
        print(f"（未指定 --workload，{args.group} 组用默认负载 {args.workload}）")
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    variants = GROUPS[args.group]

    # 预检必须在 builder 之前分叉：它要能独立运行并带着退出码返回，
    # 而不是把环境不合格这件事推到整批实验跑完之后。
    if args.sentinel_only:
        out_dir = args.output_dir or Path(f"results/sentinel_{args.group}")
        sys.exit(run_sentinel_only(args, base, variants, out_dir))

    out_dir = args.output_dir or Path(f"results/repeated_{args.group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    client_concurrency = resolve_client_concurrency(args.concurrency, args.workload)
    print(f"客户端连接上限 = {client_concurrency}"
          f"{'（开环：不节流）' if args.concurrency <= 0 else '（固定上限）'}"
          f"，负载 {args.workload}，轮数 {args.repeats}", flush=True)

    run_specs: dict[str, dict] = {}
    samples_by_variant: dict[str, list[dict]] = {name: [] for name in variants}
    for name, variant in variants.items():
        workdir = out_dir / name
        workdir.mkdir(parents=True, exist_ok=True)
        sentinel_dir = workdir / "sentinel"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        run_specs[name] = {
            "variant": variant,
            "workdir": workdir,
            "config": build_config(base, variant, workdir),
            "sentinel_dir": sentinel_dir,
            "sentinel_config": build_config(base, sentinel_variant(variant), sentinel_dir),
        }

    # 轮次优先，且每轮循环轮转臂顺序。旧实现按臂连续跑完全部重复，使 GPU 随时间
    # 变慢时“臂名称”与“实验早晚”完全混杂，所谓逐轮配对其实相隔数分钟。
    arm_names = list(variants)
    execution_order: list[list[str]] = []
    for repeat in range(args.repeats):
        offset = repeat % len(arm_names)
        ordered_names = arm_names[offset:] + arm_names[:offset]
        execution_order.append(ordered_names)
        for name in ordered_names:
            spec = run_specs[name]
            workdir = spec["workdir"]
            output = workdir / f"repeat{repeat}.jsonl"
            sentinel_output = spec["sentinel_dir"] / f"repeat{repeat}.jsonl"
            print(f"[{name}] repeat {repeat + 1}/{args.repeats}: sentinel ...", flush=True)
            run_once(spec["sentinel_config"], args.workload, sentinel_output, 1, args.warmup,
                     args.port, shards=1)
            check = contamination_check(sentinel_output)
            print(f"[{name}] repeat {repeat + 1}/{args.repeats}: policy ...", flush=True)
            summary = run_once(spec["config"], args.workload, output, client_concurrency,
                               args.warmup, args.port, shards=args.shards)
            row = flatten(summary)
            # 环境自检只看紧邻正式臂之前的单流哨兵，不读取正式臂墙钟。
            row["speed_ratio"] = check["worst_ratio"]
            row["sentinel_per_size"] = check["per_size"]
            if check["contaminated"]:
                print(
                    f"  ⚠️ 环境污染：单流哨兵实测/预测 = {check['worst_ratio']:.2f}× "
                    f"(按尺寸 {check['per_size']})——本轮数字不可用于结论",
                    flush=True,
                )
            samples_by_variant[name].append(row)

    analyze(
        samples_by_variant,
        variants,
        args,
        out_dir,
        client_concurrency=client_concurrency,
        execution_order=execution_order,
    )


def analyze(
    samples_by_variant: dict[str, list[dict]],
    variants: dict[str, dict],
    args: argparse.Namespace,
    out_dir: Path,
    client_concurrency: int,
    execution_order: list[list[str]],
) -> dict:
    """聚合、自检与判定。

    独立成函数是为了让**已落盘的重复样本可以被重新判定**（``--reanalyze``）：
    判定规则（口径、环境闸门、客户端闸门）会随方法学演进，而 GPU 实验很贵。
    重新判定只读 ``repeat*.summary.json`` 与哨兵日志，不重跑任何服务。
    """
    results: dict[str, dict] = {}
    for name, variant in variants.items():
        samples = samples_by_variant[name]
        aggregate = {}
        for metric in METRICS:
            values = [s[metric] for s in samples]
            aggregate[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        aggregate["completed"] = sum(s["completed"] for s in samples)
        aggregate["failed"] = sum(s["failed"] for s in samples)
        # speed_ratio 不在 METRICS 里（它不是性能指标而是环境自检量），
        # 但必须单独聚合：否则下游读 aggregate["speed_ratio"] 会拿到 None，
        # 让污染检查**静默失效**——而它恰恰是唯一能发现宿主干扰的手段。
        ratios = [s["speed_ratio"] for s in samples if s.get("speed_ratio")]
        aggregate["speed_ratio"] = {
            "mean": statistics.fmean(ratios) if ratios else 0.0,
            "max": max(ratios) if ratios else 0.0,
            "min": min(ratios) if ratios else 0.0,
        }
        # 客户端连接池是否人为制造排队：任何一轮被标记就值得写进汇总，
        # 因为"端到端延迟里有可观比例与服务端无关"会直接改变结论的归属。
        aggregate["client_queueing_flagged"] = any(
            bool(s.get("client_queueing_flagged")) for s in samples
        )
        backend = samples[0].get("resource") or {}
        results[name] = {
            "variant": variant,
            "repeats": args.repeats,
            "aggregate": aggregate,
            "samples": samples,
            "resource_backend": backend.get("effective_backend"),
            "enforces_sm_partition": backend.get("enforces_sm_partition"),
        }
        print(f"  -> 吞吐 {aggregate['throughput_rps']['mean']:.2f}±{aggregate['throughput_rps']['std']:.2f} req/s, "
              f"服务端 P99 {aggregate['server_total_p99_ms']['mean']:.1f}"
              f"±{aggregate['server_total_p99_ms']['std']:.1f} ms, "
              f"客户端 P99 {aggregate['client_total_p99_ms']['mean']:.1f}"
              f"±{aggregate['client_total_p99_ms']['std']:.1f} ms, "
              f"SLO 违约 服务端 {aggregate['slo_violation_rate']['mean'] * 100:.1f}% / "
              f"客户端 {aggregate['client_slo_violation_rate']['mean'] * 100:.1f}%", flush=True)
        if aggregate["client_queueing_flagged"]:
            print(f"  ⚠️ 客户端口径告警：连接池等待估计 p95 最大 "
                  f"{aggregate['client_pool_wait_p95_ms']['max']:.1f} ms、发出迟到 p95 最大 "
                  f"{aggregate['client_dispatch_lateness_p95_ms']['max']:.1f} ms——"
                  f"（判据：连接池等待 p95 占客户端 P95 超过 "
                  f"{CLIENT_OVERHEAD_SHARE_WARN:.0%}，或发出迟到超过抖动量级）"
                  f"端到端延迟里有一部分是客户端自己造成的，"
                  f"不能用它论证服务端尾延迟", flush=True)

    payload = {
        "group": args.group,
        "repeats": args.repeats,
        "workload": str(args.workload),
        "concurrency": client_concurrency,
        "concurrency_requested": args.concurrency,
        "execution_order": execution_order,
        "sm_isolation_verified": bool(results and results[next(iter(results))]["enforces_sm_partition"]),
        "note": "resource_backend=proxy 时不含真实 SM 隔离效果；这些数字只反映软件调度层。",
        "latency_scopes": {
            "server": "自请求到达服务端起算（queue/execution/total），不含网络与连接池等待",
            "client": "客户端逐请求观测，含连接池等待与 HTTP 往返；字段前缀 client_",
        },
        "variants": results,
    }

    # 环境自检汇总。**先看这一项再看任何策略结论**：被污染的运行里，
    # 所有臂的比较都不可信（本课题已有一次整轮作废的记录）。
    def worst_ratio(item: dict) -> float:
        """该臂所有轮次里最差的环境比值。

        取 **max 而不是 mean**：一轮污染就足以让整批对照失去意义，
        而均值会被其余干净轮次稀释到阈值以下——那等于放过污染。
        """
        aggregated = item["aggregate"].get("speed_ratio")
        if aggregated:
            return aggregated["max"]
        return max((s.get("speed_ratio") or 0.0) for s in item["samples"])

    contaminated = [
        (name, worst_ratio(item)) for name, item in results.items() if worst_ratio(item) > SENTINEL_THRESHOLD
    ]
    payload["environment_contaminated"] = bool(contaminated)
    if contaminated:
        print(f"\n⚠️⚠️ 环境被污染（实测/预测 > {SENTINEL_THRESHOLD}×），以下策略对比**不成立**：")
        for name, ratio in contaminated:
            print(f"      {name}: {ratio:.2f}×")
        print("     处置：确认 Windows 侧没有占用 GPU 的进程后重跑。")
        print("     判据：nvidia-smi 在 WSL 内看不到宿主进程，因此使用紧邻正式臂的单流哨兵。")

    # 客户端自检：与环境污染同级。若某个臂的吞吐被**客户端**连接上限钉住，
    # 那这一批比的是客户端池而不是调度器——2026-09-22 的 7 臂实测正是如此，
    # 因此这里必须在给出任何策略判定之前拦下来。
    client_throttled = [
        (name, item["aggregate"]["client_pool_wait_p95_ms"]["max"])
        for name, item in results.items()
        if item["aggregate"].get("client_queueing_flagged")
    ]
    payload["client_throttled_arms"] = [name for name, _ in client_throttled]
    if client_throttled:
        print("\n⚠️⚠️ 客户端连接池成为瓶颈（端到端延迟里有可观比例不是服务端造成的），"
              "以下策略对比**不成立**：")
        for name, wait in client_throttled:
            print(f"      {name}: 连接池等待 p95 最大 {wait:.1f} ms")
        print(f"     处置：本批客户端连接上限 = {client_concurrency}"
              f"（请求数 = {workload_request_count(args.workload)}）。"
              "用 --concurrency 0（开环）或提高上限后重跑。")
        print("     判据：吞吐会被钉在 连接上限 / 服务端延迟 上，与调度策略无关。")

    # 配对差：concurrency 组默认以**最佳静态档**为基准，判据是 dynamic 优于它。
    baseline = args.baseline
    if baseline is None and args.group == "graph":
        # 基准取 incumbent——固定资源分配档（§21.3 确立的最佳 eager 档）。
        # 判据是方法臂相对它**逐轮配对差为正**，而不是"比串行快"。
        baseline = "eager_2stream" if "eager_2stream" in results else None
    if baseline is None and args.group == "concurrency":
        static_names = [name for name in results if name.startswith("static_") and name in results]
        if static_names:
            baseline = max(static_names, key=lambda n: results[n]["aggregate"]["throughput_rps"]["mean"])
    if baseline:
        if baseline not in results:
            print(f"\n⚠️ 配对基准 {baseline!r} 不在结果里，跳过配对分析")
        elif contaminated or client_throttled:
            # 污染下不给结论。数字仍写进 summary.json 供诊断，但**不打印判定**，
            # 因为"✅ 通过"这种字样一旦出现就很容易被后来的读者当成结论引用。
            reason = "环境被污染" if contaminated else "客户端连接池成为瓶颈"
            print(f"\n（{reason}：配对分析与判定已跳过，数字仅供诊断）")
        else:
            paired = paired_differences(results, baseline)
            payload["paired_baseline"] = baseline
            payload["paired"] = paired
            print(f"\n配对差（基准 = {baseline}；正数 = 该臂更好）")
            print("口径：吞吐与服务端 P99 来自服务端；客户端 P99 含连接池等待与 HTTP 往返。"
                  "两者相差 15–34 倍，禁止混用。")
            print(f"{'臂':<12}{'吞吐 Δ':>12}{'服务端P99 Δ':>14}{'客户端P99 Δ':>14}{'符号一致':>12}")
            for name, per_metric in paired.items():
                tput = per_metric["throughput_rps"]
                server_p99 = per_metric["server_total_p99_ms"]
                client_p99 = per_metric["client_total_p99_ms"]
                agree = all(
                    per_metric[m]["consistent"]
                    for m in ("throughput_rps", "server_total_p99_ms", "client_total_p99_ms")
                )
                print(f"{name:<12}{tput['mean']:>+12.2f}{server_p99['mean']:>+14.2f}"
                      f"{client_p99['mean']:>+14.2f}{'是' if agree else '否':>12}")
            dynamic = paired.get("dynamic")
            if dynamic:
                better_tput = dynamic["throughput_rps"]["mean"] > 0 and dynamic["throughput_rps"]["consistent"]
                better_server_p99 = (
                    dynamic["server_total_p99_ms"]["mean"] > 0
                    and dynamic["server_total_p99_ms"]["consistent"]
                )
                better_client_p99 = (
                    dynamic["client_total_p99_ms"]["mean"] > 0
                    and dynamic["client_total_p99_ms"]["consistent"]
                )
                # 口径冲突必须显式报出来：服务端 P99 每一轮都更好、而客户端 P99 每一轮
                # 都更差，意味着"改善"发生在服务端内部而用户并没有受益（或反之）。
                conflict = better_server_p99 != better_client_p99 and (
                    dynamic["server_total_p99_ms"]["consistent"]
                    and dynamic["client_total_p99_ms"]["consistent"]
                )
                if conflict:
                    print("\n⚠️ 两个口径给出**方向相反**的 P99 判定（服务端"
                          f"{'更好' if better_server_p99 else '更差'} / 客户端"
                          f"{'更好' if better_client_p99 else '更差'}）——"
                          "不得据单一端口径下结论，须先查 client_overhead_ms 与 "
                          "client_dispatch_lateness_p95_ms。")
                elif better_tput or better_server_p99 or better_client_p99:
                    print("\n✅ dynamic 在**每一轮**上都优于最佳静态档（至少一项指标）")
                else:
                    print("\n❌ dynamic 未能稳定优于最佳静态档 → 判据不成立，"
                          "说明'尺寸混合'不构成有效控制信号")

    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["variant", "resource_backend", "enforces_sm_partition"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, item in results.items():
            row = {"variant": name, "resource_backend": item["resource_backend"], "enforces_sm_partition": item["enforces_sm_partition"]}
            for metric in METRICS:
                row[f"{metric}_mean"] = round(item["aggregate"][metric]["mean"], 4)
                row[f"{metric}_std"] = round(item["aggregate"][metric]["std"], 4)
            writer.writerow(row)
    print(f"\n结果写入 {out_dir}/summary.json 与 summary.csv")
    return payload


def reanalyze(args: argparse.Namespace, out_dir: Path) -> int:
    """用当前判定规则重新判定一个已落盘的批次，不重跑任何服务。

    场景：实验很贵，而判定规则（延迟口径、环境闸门、客户端闸门）会随方法学演进。
    典型一例是 ``results/concurrency_client_throttled_2026-09-22/``——那批运行本身没问题，
    是**客户端连接上限**让每个臂的吞吐都变成"上限 / 服务端延迟"，必须换规则重判。
    """
    variants = GROUPS[args.group]
    samples_by_variant: dict[str, list[dict]] = {}
    execution_order: list[list[str]] = []
    client_concurrency = resolve_client_concurrency(args.concurrency, args.workload)
    for name in variants:
        workdir = out_dir / name
        samples: list[dict] = []
        for path in sorted(workdir.glob("repeat*.jsonl")):
            if path.name.startswith("repeat") and path.parent.name == name:
                summary_path = path.with_suffix(".summary.json")
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                # 客户端口径**从原始逐请求日志重算**，不复用落盘的判定：
                # 判定规则会变（2026-09-22 的批次就因连接上限语义变化必须重判），
                # 复用旧旗标等于把旧规则带进新结论。
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                summary["client_scope"] = client_side_summary(
                    rows,
                    concurrency=client_concurrency,
                    pool_can_throttle=client_concurrency < len(rows),
                )
                row = flatten(summary)
                # 环境自检重新读取该轮紧邻的哨兵日志，不复用落盘结论：
                # 阈值或判据变化时，旧结论不能自动继承。
                sentinel = workdir / "sentinel" / path.name
                if sentinel.exists():
                    check = contamination_check(sentinel)
                    row["speed_ratio"] = check["worst_ratio"]
                    row["sentinel_per_size"] = check["per_size"]
                samples.append(row)
        if not samples:
            print(f"⚠️ {workdir} 里没有可用的 repeat*.summary.json")
            return 1
        samples_by_variant[name] = samples
    repeats = {len(samples) for samples in samples_by_variant.values()}
    if len(repeats) != 1:
        print(f"⚠️ 各臂轮数不一致 {sorted(repeats)}，配对差没有意义")
        return 1
    args.repeats = repeats.pop()
    args.workload = Path(args.workload)
    print(f"重新判定 {out_dir}：{len(variants)} 臂 × {args.repeats} 轮，"
          f"客户端连接上限按 {client_concurrency} 计"
          f"（该批次若用别的上限，请加 --concurrency N）", flush=True)
    analyze(
        samples_by_variant,
        variants,
        args,
        out_dir,
        client_concurrency=client_concurrency,
        execution_order=execution_order,
    )
    return 0


if __name__ == "__main__":
    main()
