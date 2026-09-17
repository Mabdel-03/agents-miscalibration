"""``allocation validate-config`` (lead-owned)."""
from __future__ import annotations

import argparse
import json

from agent_allocation.config import config_hash, grids_from_config, load_config, validate_config


def cmd_validate_config(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation validate-config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    problems = validate_config(cfg)
    grids = grids_from_config(cfg)
    print(json.dumps({"config": args.config, "preset": cfg.get("preset"), "config_sha256": config_hash(cfg),
                      "problems": problems, "grids": sorted(grids)}, indent=1))
    return 1 if problems else 0
