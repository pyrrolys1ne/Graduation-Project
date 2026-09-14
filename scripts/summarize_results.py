from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        valid = [row for row in rows if row.get("status") == "completed"]
        totals = np.asarray([row["total_ms"] for row in valid], dtype=float)
        records.append(
            {
                "experiment": path.stem,
                "count": len(valid),
                "mean_ms": float(totals.mean()) if len(totals) else None,
                "p50_ms": float(np.percentile(totals, 50)) if len(totals) else None,
                "p95_ms": float(np.percentile(totals, 95)) if len(totals) else None,
                "p99_ms": float(np.percentile(totals, 99)) if len(totals) else None,
                "slo_violation_rate": (
                    sum(bool(row["slo_violated"]) for row in valid) / len(valid) if valid else None
                ),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    try:
        import matplotlib.pyplot as plt

        axes = frame.plot(x="experiment", y=["mean_ms", "p99_ms"], kind="bar", rot=20)
        axes.set_ylabel("Latency (ms)")
        axes.figure.tight_layout()
        axes.figure.savefig(args.output_dir / "latency.png", dpi=180)
        plt.close(axes.figure)
    except ImportError:
        print("未安装 matplotlib，已跳过绘图")


if __name__ == "__main__":
    main()

