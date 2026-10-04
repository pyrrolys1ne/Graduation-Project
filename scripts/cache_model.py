from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="下载配置指定的 Hugging Face 模型到共享缓存")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = config["model"]["name"]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 huggingface_hub；请先完成 bootstrap") from exc

    path = snapshot_download(repo_id=model_name, revision=args.revision)
    print(json.dumps({"model": model_name, "cache_path": path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
