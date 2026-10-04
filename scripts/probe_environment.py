from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder_sched.config import load_config
from encoder_sched.runtime import RuntimeValidationError, validate_server_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Linux CUDA 服务器运行环境")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strict", action="store_true", help="检查失败时返回非零退出码")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        report = {
            "status": "ready",
            **validate_server_runtime(config),
            "configured_backend": config.executor.resource_backend,
            "allow_proxy_fallback": config.executor.allow_proxy_fallback,
            "model": config.model.name,
            "local_files_only": config.model.local_files_only,
        }
    except (OSError, ValueError, RuntimeValidationError) as exc:
        report = {"status": "error", "reason": str(exc)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.strict:
            raise SystemExit(1) from exc
        return
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
