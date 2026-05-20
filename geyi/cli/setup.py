"""`geyi setup` command."""

from __future__ import annotations

from pathlib import Path

import yaml

from geyi.config import DEFAULT_CONFIG_PATH


def run(args) -> int:
    path = Path(args.config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "allow_llm_draft": False,
        "phase": "-1",
        "notes": "minimal local config; LLM is disabled in Phase -1",
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print("Wrote %s" % path)
    return 0


def add_arguments(parser) -> None:
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)

