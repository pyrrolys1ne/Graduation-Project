from __future__ import annotations

import sys
from pathlib import Path

# 以脚本方式运行时 sys.path[0] 是 scripts/，此时 encoder_sched 会解析到当前
# 可编辑安装所指向的路径，而不一定是本仓库。显式把仓库根放到最前，保证
# 脚本始终运行本目录下的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import statistics
import time

from encoder_sched.config import load_config
from encoder_sched.encoder import ClipEncoderBackend
from encoder_sched.models import EncodeJob
from encoder_sched.resource import create_resource_backend

#: 用分布无关的序统计量界估计 10% 分位数所需的最小样本数（⌈(n+1)·0.9⌉）。
#: 低于它时中位数仍然可用，但任何尾部结论都不可用。
TAIL_SAMPLE_FLOOR = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 (尺寸 × 配额) 网格标定延迟。"
                    "除汇总 CSV 外，还会把**每一个原始样本**逐条写入 JSONL——"
                    "汇总表只留中位数与标准差，装不下违约概率模型需要的信息。")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 336, 448, 672])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5,
                        help=f"每单元重复次数。估计 10% 分位数需要 >= {TAIL_SAMPLE_FLOOR}")
    parser.add_argument("--output", type=Path, default=Path("data/profiles/measured.csv"))
    parser.add_argument("--samples-out", type=Path, default=Path("data/profiles/samples.jsonl"),
                        help="原始样本的落盘位置。**追加写**，故多个会话会累积在同一文件里，"
                             "这正是估计跨会话离散度所必需的。")
    parser.add_argument("--session-id", default=None,
                        help="会话标识。默认取脚本启动时刻（如 20260929T142700）。"
                             "跨会话离散度靠它区分；缺了它，两次运行的数据无法分账。")
    parser.add_argument("--no-enforce-monotonic", dest="enforce_monotonic", action="store_false",
                        help="保留原始中位数，不做单调化")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_config(args.config)
    resource = create_resource_backend(
        config.executor.resource_backend,
        config.executor.libsmctrl_adapter,
        config.executor.allow_proxy_fallback,
    )
    encoder = ClipEncoderBackend(config.model, resource, 1)

    session_id = args.session_id or time.strftime("%Y%m%dT%H%M%S")
    mechanism = describe_mechanism(config)
    quotas = config.scheduler.quota_levels if resource.enforces_sm_partition else (1.0,)

    rows: list[dict] = []
    records: list[dict] = []
    for size in args.sizes:
        for quota in quotas:
            samples: list[float] = []
            total = args.warmup + args.repeats
            for index in range(total):
                job = EncodeJob(f"profile-{size}-{quota}-{index}", config.seed + index, size, size, 60_000)
                job.sm_fraction = quota
                result = encoder.encode(job, 0)
                if index >= args.warmup:
                    samples.append(result.execution_ms)
                    records.append({
                        "session_id": session_id,
                        "mechanism": mechanism,
                        "backend": resource.name,
                        "sm_enforced": resource.enforces_sm_partition,
                        "width": size,
                        "height": size,
                        "patches": (size // 32) ** 2,
                        "sm_fraction": quota,
                        "sample_index": index - args.warmup,
                        "execution_ms": result.execution_ms,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
            rows.append(
                {
                    "width": size,
                    "height": size,
                    "patches": (size // 32) ** 2,
                    "sm_fraction": quota,
                    "latency_ms": statistics.median(samples),
                    "mean_ms": statistics.fmean(samples),
                    "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "samples": len(samples),
                    "backend": resource.name,
                    "sm_enforced": resource.enforces_sm_partition,
                    "session_id": session_id,
                    "mechanism": mechanism,
                }
            )
    if args.enforce_monotonic:
        rows = enforce_monotonic(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_out.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"剖析完成: {args.output}；资源后端={resource.name}；真实 SM 配额={resource.enforces_sm_partition}")
    print(f"  会话={session_id}  执行机制={mechanism}")
    print(f"  原始样本 {len(records)} 条 -> {args.samples_out}（追加）")
    if args.repeats < TAIL_SAMPLE_FLOOR:
        print(f"  ⚠️ 每单元仅 {args.repeats} 个样本，低于估计 10% 分位数所需的 {TAIL_SAMPLE_FLOOR}。"
              "该表只能用于中位数类决策；任何尾部/违约概率结论都不可用。")
    if args.enforce_monotonic:
        print("  已做单调化：同一尺寸下，配额升高时延迟不会变大（见 enforce_monotonic 说明）")


def describe_mechanism(config) -> str:
    """执行机制标签。

    同一个 (尺寸, 配额) 单元在**不同执行机制**下不是同一个函数：图回放路径上两个请求
    共驻的代价可加，eager 路径上是 2–3.5 倍。因此机制必须进索引，否则跨路径的样本会
    被当成同一分布。

    格式：``{eager|graph}/par={并发度}``；资源后端另有 ``backend`` 列单独记录
    （掩码会改变执行，不能只靠机制标签区分）。
    """
    if getattr(config.executor, "graph_pipeline", False):
        return f"graph/par={config.executor.graph.slots}"
    return f"eager/par={config.executor.streams}"


def enforce_monotonic(rows: list[dict]) -> list[dict]:
    """强制"配额越高、延迟不增"。

    实测中位数会因噪声违反单调性（例如 224×224 测出 0.5 档 4.80 ms 快于 0.75 档
    5.68 ms——该尺寸标准差就有 1.2–1.5 ms）。但性能表是**调度决策的输入**：
    非单调会让"多给资源反而预测更慢"，插值与排序都会失真，且这类失真不会报错。

    做法：对每个尺寸，按配额从高到低扫描，取运行最大值。即
    ``latency[q] = max(latency[q], latency[q'])``，其中 ``q'`` 是所有比 ``q`` 高的档位。
    这只把异常偏低的值抬到相邻更高配额的水平，不会凭空降低任何预测。

    代价是可能掩盖真实的非单调效应（例如某个尺寸在某个配额下确实更快）。
    因此**原始值一律保留在 ``latency_ms_measured`` 列**（未被抬升的行也写，值为原
    中位数），`monotonic_adjusted` 标记是否被抬升。
    """
    from collections import defaultdict

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["width"]), int(row["height"]))].append(row)

    adjusted: list[dict] = []
    for key, group in grouped.items():
        order = sorted(group, key=lambda r: float(r["sm_fraction"]), reverse=True)
        ceiling = float("-inf")
        for row in order:
            value = float(row["latency_ms"])
            row["latency_ms_measured"] = value  # 无条件保留原始中位数
            if value < ceiling:
                row["latency_ms"] = ceiling
                row["monotonic_adjusted"] = True
            else:
                ceiling = value
                row["monotonic_adjusted"] = False
            adjusted.append(row)
    adjusted.sort(key=lambda r: (int(r["width"]), float(r["sm_fraction"])))
    return adjusted


if __name__ == "__main__":
    main()
