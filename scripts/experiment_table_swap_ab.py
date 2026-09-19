"""换表 A/B：同一会话内只替换标定表，测服务指标怎么变。

回答两个问题
------------
1. **Q4 的硬件验证**：第 12.3 节的决策分歧是**离线**算出来的（用真实调度器代码，
   但没真跑服务）。这里把"100 对 0 个违约"放到真机上检验。
2. **数据可信度**：`results/fixed_baselines`（09-14 19:37）与 `results/overload_baselines`
   （09-14 13:06）是**逐字节相同的配置**，结果却差 12.7%–52.8%。查下来唯一的差异是
   `data/profiles/default.csv` 在这两批之间被替换过（19:11 写入），而两个 config 指向
   同一个路径。**这是一次自然实验，但与会话时间混淆。** 本脚本把混淆去掉：
   同一会话内交替换表。

为什么必须交替
--------------
标定表的效应本身是待测量；若先跑完 A 臂再跑完 B 臂，会话漂移会与表效应不可分——
本课题已经踩过一次（两份"相同配置"的结果差 12.7%–52.8%）。因此按 round 交替，
A/B/A/B…，再比较配对差异。

臂设计
------
- ``edf_size``：**读表**（`quota_policy=deadline_min` 会查表选配额，`rank_key` 用 `predicted_ms`）
- ``edf``：**不读表**（`choose_quota` 对非 edf_size/dacc 直接返回满配额）——作为对照，
  它应当在换表前后**没有系统差异**。这是本实验的阴性对照，缺了它就说不清"差异来自表"
  还是"差异来自任何两次运行之间都会有的漂移"。

用法
----
::

    export LIBSMCTRL_PATH=$HOME/.local/lib/libsmctrl/libsmctrl.so
    python scripts/experiment_table_swap_ab.py --rounds 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import shutil
import statistics
import subprocess

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKLOAD_SRC = REPO / "data/workloads/saturating_4ms.jsonl"


def make_tight_workload(target_median_ms: float, out: Path) -> dict:
    """把 ``saturating_4ms.jsonl`` 的 deadline 缩放到目标中位数。

    为什么需要：原负载 deadline 中位 **102.5 ms**，而实测执行只有 3–22 ms。
    `deadline_min` 对所有请求都直接返回最小配额，**两套表不可能分歧**（离线实测 0.0%）。
    「表被污染」与「决策被污染」之间隔着一个紧度条件，离线扫描定位到它在中位 10.2 ms
    附近出现。本函数把负载压进那个区间。
    """
    rows = [json.loads(line) for line in WORKLOAD_SRC.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    current = statistics.median(float(r["deadline_ms"]) for r in rows)
    scale = target_median_ms / current
    for r in rows:
        r["deadline_ms"] = round(max(1.0, float(r["deadline_ms"]) * scale), 3)
        r["request_id"] = r["request_id"].replace("mixed", "tight")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    after = statistics.median(float(r["deadline_ms"]) for r in rows)
    return {"source": str(WORKLOAD_SRC), "median_before_ms": current,
            "median_after_ms": after, "scale": scale, "count": len(rows)}


def write_config(base_path: Path, table: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    data.setdefault("profiling", {})["table_path"] = str(table)
    # 真实掩码实验禁止静默回退（与 project 既有纪律一致）
    data.setdefault("executor", {})["allow_proxy_fallback"] = False
    data.setdefault("executor", {})["resource_backend"] = "libsmctrl"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def run_once(cfg: Path, workload: Path, out_dir: Path, group: str, port: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts/run_repeated_comparison.py"),
         "--group", group, "--config", str(cfg), "--workload", str(workload),
         "--output-dir", str(out_dir), "--repeats", "1", "--concurrency", "8",
         "--warmup", "2", "--port", str(port)],
        cwd=REPO, check=True, capture_output=True, text=True,
    )


def collect(out_dir: Path, variants: list[str]) -> dict[str, dict]:
    """从 repeat0.summary.json 读每个变体的指标。"""
    result: dict[str, dict] = {}
    for name in variants:
        path = out_dir / name / "repeat0.summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        total = data["total_latency_ms"]
        result[name] = {
            "throughput_rps": data["throughput_rps"],
            "total_mean_ms": total["mean"],
            "total_p99_ms": total["p99"],
            "queue_mean_ms": data["queue_latency_ms"]["mean"],
            "execution_mean_ms": data["execution_latency_ms"]["mean"],
            "slo_violation_rate": data["slo_violation_rate"],
            "gpu_utilization_percent": data["gpu_utilization_percent"],
            "completed": data["completed"],
        }
    return result


def report(rounds: list[dict], variants: list[str], meta: dict, output_dir: Path) -> None:
    """打印并保存配对汇总。

    ``rounds`` 的每个条目形如 ``{"round": r, "arm": "old"|"new", <变体名>: {指标: 值}}``。
    按 ``arm`` 分组后取 ``r[v][metric]``——**不是** ``r[arm][v][metric]``。
    """
    print(f"\n{'='*104}")
    print(f"换表 A/B（同一会话内按轮交替），紧 deadline 中位 {meta['median_after_ms']:.1f} ms")
    print(f"{'='*104}")
    print(f"{'策略':>26} {'指标':>18} {'旧表':>11} {'新表':>11} {'新/旧':>9} {'配对差中位':>11} {'n配对':>6}")
    print("-" * 104)
    summary: dict = {}
    for v in variants:
        for metric in ("throughput_rps", "total_mean_ms", "total_p99_ms",
                       "queue_mean_ms", "execution_mean_ms", "slo_violation_rate"):
            by_arm = {arm: [r[v][metric] for r in rounds
                            if r["arm"] == arm and v in r]
                      for arm in ("old", "new")}
            if not by_arm["old"] or not by_arm["new"]:
                continue
            a, b = statistics.median(by_arm["old"]), statistics.median(by_arm["new"])
            pairs = {r["round"]: r for r in rounds if v in r}
            diffs = []
            for rnd in sorted({r["round"] for r in rounds}):
                old = next((r for r in rounds if r["round"] == rnd and r["arm"] == "old" and v in r), None)
                new = next((r for r in rounds if r["round"] == rnd and r["arm"] == "new" and v in r), None)
                if old and new:
                    diffs.append(new[v][metric] - old[v][metric])
            pair = statistics.median(diffs) if diffs else float("nan")
            ratio = b / a if a else float("nan")
            print(f"{v:>26} {metric:>18} {a:>11.3f} {b:>11.3f} {ratio:>8.3f}× {pair:>+11.3f} {len(diffs):>6}")
            summary.setdefault(v, {})[metric] = {
                "old_median": a, "new_median": b, "ratio": ratio,
                "paired_median_diff": pair, "n_pairs": len(diffs),
                "old_values": by_arm["old"], "new_values": by_arm["new"]}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="换表 A/B：同一会话内只替换标定表")
    parser.add_argument("--rounds", type=int, default=5, help="每个臂跑几轮，按轮交替")
    parser.add_argument("--table-old", type=Path,
                        default=REPO / "data/profiles/legacy_2026-09-14.csv")
    parser.add_argument("--table-new", type=Path, default=REPO / "data/profiles/default.csv")
    parser.add_argument("--base-config", type=Path, default=REPO / "config.libsmctrl.example.yaml")
    parser.add_argument("--target-deadline-ms", type=float, default=12.0)
    parser.add_argument("--variants", default="edf_size_deadline_min,edf")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/table_swap_ab")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--from-rounds", type=Path, default=None,
                        help="跳过采集，直接对已有 rounds.json 重算汇总")
    args = parser.parse_args()

    if args.from_rounds:
        saved = json.loads(args.from_rounds.read_text(encoding="utf-8"))
        report(saved["rounds"], saved["variants"], saved["meta"], args.from_rounds.parent)
        return

    for table in (args.table_old, args.table_new):
        if not table.exists():
            raise SystemExit(f"标定表不存在: {table}")

    workload = REPO / "data/workloads/tight_8ms.jsonl"
    meta = make_tight_workload(args.target_deadline_ms, workload)
    print(f"紧 deadline 负载: 中位 {meta['median_before_ms']:.1f} → "
          f"{meta['median_after_ms']:.1f} ms（缩放 {meta['scale']:.3f}，{meta['count']} 条）")

    cfg_old = write_config(args.base_config, args.table_old, args.output_dir / "config_old.yaml")
    cfg_new = write_config(args.base_config, args.table_new, args.output_dir / "config_new.yaml")

    variants = [v for v in args.variants.split(",") if v.strip()]
    # `table_ab` 组只含这两个臂——用 `baselines`/`quota` 组会白跑另外三个策略，
    # 每轮多花 3 倍时间且对结论无贡献。
    group = "table_ab"

    rounds: list[dict] = []
    for r in range(args.rounds):
        for arm, cfg in (("old", cfg_old), ("new", cfg_new)):
            out = args.output_dir / f"round{r}_{arm}"
            print(f"  round {r+1}/{args.rounds}  arm={arm:>3} …", flush=True)
            run_once(cfg, workload, out, group, args.port)
            rounds.append({"round": r, "arm": arm, **collect(out, variants)})
    (args.output_dir / "rounds.json").write_text(
        json.dumps({"meta": meta, "rounds": rounds, "variants": variants},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    report(rounds, variants, meta, args.output_dir)


if __name__ == "__main__":
    main()
