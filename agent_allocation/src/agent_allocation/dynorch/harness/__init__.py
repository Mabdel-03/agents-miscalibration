"""WP-B: harness adapters (contract §6 ``HarnessAdapter`` protocol; plan §6).

``base`` holds the protocol, the launch/result records, the per-run work-directory layout and the environment
scrub; ``null`` (H0), ``fake`` (tests), ``openclaude`` (H1), ``codex`` (H3) and ``openclaw`` (fallback H3) are the
adapters. ``adapters_from_config`` builds the registry the runner uses.
"""
from __future__ import annotations

from typing import Any, Mapping

from agent_allocation.dynorch.harness.base import HarnessAdapter

__all__ = ["adapters_from_config"]


def adapters_from_config(cfg: Mapping[str, Any], *, project_root: Any = None) -> dict[str, HarnessAdapter]:
    """One adapter instance per harness name used by the config's arms (lazy imports keep the CLIs optional)."""
    from agent_allocation.dynorch.harness.codex import CodexAdapter
    from agent_allocation.dynorch.harness.fake import FakeAdapter
    from agent_allocation.dynorch.harness.null import NullAdapter
    from agent_allocation.dynorch.harness.openclaude import OpenclaudeAdapter
    from agent_allocation.dynorch.harness.openclaw import OpenclawAdapter

    block = cfg["dynorch"]
    model = str(cfg["primary_model"])
    wall = block["wall_timeout_seconds"]
    return {
        "null": NullAdapter(model=model),
        "fake": FakeAdapter(model=model),
        "openclaude": OpenclaudeAdapter(model=model),
        "codex": CodexAdapter(model=model, prefix=str(block["codex_prefix"]), wall_base=int(wall["base"]),
                              wall_per_rung=int(wall["per_rung"])),
        "openclaw": OpenclawAdapter(model=model),
    }
