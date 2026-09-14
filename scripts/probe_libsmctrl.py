"""libsmctrl 真实能力探针。

检查库是否打了补丁、TPC 是否可查、**回调能否安全注册**（在子进程中试调，因为回调
注册失败会 `exit(1)`）。默认不会创建 GPU 负载。

退出码：0 表示具备真实掩码能力；1 表示不具备（原因写入输出）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.libsmctrl_adapter import (  # noqa: E402
    LibSmCtrlAdapter,
    enabled_tpcs_to_native_mask,
    fraction_to_enabled_tpcs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 libsmctrl 真实掩码能力")
    parser.add_argument("--config", default=None, help="项目配置（仅用于显示配置后端）")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--library", default=None, help="显式指定 libsmctrl.so 路径")
    parser.add_argument("--show-masks", action="store_true", help="打印各配额档位对应的 TPC 掩码")
    args = parser.parse_args()

    report: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "adapter": "encoder_sched.libsmctrl_adapter",
        "mask_path": "callback",
        "note": "走启动回调路径 + 粘性线程掩码；不用 set_stream_mask（未知驱动版本会 exit(1)）",
    }
    if args.library:
        report["library"] = args.library
    adapter = LibSmCtrlAdapter(cuda_device=args.cuda_device, library_path=args.library)
    probe = adapter.probe()
    report["probe"] = probe

    if probe.get("available") and args.show_masks:
        total = probe["total_tpcs"]
        report["mask_table"] = [
            {
                "requested_sm_fraction": level,
                "enabled_tpcs": fraction_to_enabled_tpcs(level, total),
                "effective_sm_fraction": fraction_to_enabled_tpcs(level, total) / total,
                "native_disable_mask": hex(enabled_tpcs_to_native_mask(0, fraction_to_enabled_tpcs(level, total))),
            }
            for level in (0.25, 0.5, 0.75, 1.0)
        ]

    available = bool(probe.get("available"))
    report["real_sm_masking_available"] = available
    report["conclusion"] = (
        "具备真实 TPC 掩码能力；机制层面的证据见 demos/libsmctrl_thread_mask_demo.py，"
        "效果层面的结论仍需 scripts/experiment_sm_quota.py 的独立测量"
        if available
        else "不具备真实掩码能力；【禁止】把该环境下的调度结果表述为 SM 隔离效果"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
