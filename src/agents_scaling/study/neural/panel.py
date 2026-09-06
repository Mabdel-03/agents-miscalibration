"""The C/G/R2 confirmation panel, defined once for every neural consumer (P1-4).

Brief R1: "the first 300 main items by rank: 150 HLE + 150 BCB, i.e. the N/D/M-panel
prefix".  ``PublicTask.rank`` is the salted-hash rank inside the split (``cells.py`` builds
rank-prefix modules from it), so the panel is *defined* by ``rank < per_domain`` on the
``main`` split — never by counting items in cell order.  N1 (``capture.native_work_list``),
N2 (``forecast.run --panel-per-domain``) and N3 (``readout.select_split``: ``rank <
panel_per_domain``) all use this rule; ``panel_source_ids`` is the single implementation
N1 and N2 call, and it reads only the public task export.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_PANEL_PER_DOMAIN = 150
PANEL_RULE = "PublicTask.rank < per_domain within split 'main', per superdomain (hle, bcb); order = (domain, rank)"


def panel_from_tasks(tasks: Sequence[Any], per_domain: int, *, split: str = "main") -> dict[str, list[str]]:
    """``{domain: [source_id, ...]}`` ordered by rank, ranks ``< per_domain`` of ``split``."""
    if per_domain < 1:
        raise ValueError("per_domain must be >= 1")
    out: dict[str, list[tuple[int, str]]] = {}
    seen: set[str] = set()
    for t in tasks:
        if str(getattr(t, "split", None)) != split:
            continue
        rank = int(t.rank)
        if rank >= per_domain:
            continue
        domain = getattr(t.domain, "value", t.domain)
        sid = str(t.source_id)
        if sid in seen:
            raise ValueError(f"duplicate source_id {sid} in the public task export")
        seen.add(sid)
        out.setdefault(str(domain), []).append((rank, sid))
    return {d: [sid for _, sid in sorted(v)] for d, v in sorted(out.items())}


def panel_source_ids(run_root: str | os.PathLike, per_domain: int, *, split: str = "main") -> dict[str, list[str]]:
    """The panel from ``<run_root>/data/public/tasks.jsonl`` (raises ``FileNotFoundError``
    when the export is absent — callers decide whether a fallback is acceptable)."""
    from agents_scaling.study.data.public import load_public_tasks

    return panel_from_tasks(load_public_tasks(run_root, split), per_domain, split=split)


def flatten_panel(panel: Mapping[str, Sequence[str]]) -> list[str]:
    """Domain-major, rank-ordered item list (the shape ``forecast.run`` shards)."""
    return [sid for d in sorted(panel) for sid in panel[d]]


__all__ = ["DEFAULT_PANEL_PER_DOMAIN", "PANEL_RULE", "flatten_panel", "panel_from_tasks", "panel_source_ids"]
