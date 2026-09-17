"""Nodes the dispatcher made terminal without a candidate, read from the coordinator's state backup.

The dispatcher writes ``parsed/`` only for accepted (transport-OK) responses. A node that exhausted its transport
attempts, hit a non-retryable transport status, or was stopped by the context-limit policy (contract v2 R03-1) is
recorded only in SQLite (``nodes.state = 'terminal'``, ``outcome != 'succeeded'``, ``terminal_reason``). Such a node
will never produce a candidate, so its checkpoints must not wait for one: selection and metrics read it as
``extraction.final.empty_candidate`` (parse ``EMPTY``, no vote, state ``I`` -- operationally unsuccessful, the same
reading agent 07 gives terminal no-output nodes), while the terminal reason keeps transport failures and context
limits distinguishable from each other and from jobs that were never admitted.

The reader opens ``paths.state_db_backup`` read-only (``mode=ro&immutable=1``), exactly as ``analysis.load`` does. A
missing backup means nothing is known to be terminal (checkpoints stay pending); an unreadable one raises.

Config keys consumed: none.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent_allocation.paths import RunPaths

__all__ = ["TerminalWithoutOutput", "load_terminal_without_output"]

_TRANSPORT_PREFIXES = ("attempts_exhausted:", "non_retryable:")


@dataclass(frozen=True)
class TerminalWithoutOutput:
    node_id: str
    outcome: str | None                   # e.g. "exhausted" | "context_limit"
    reason: str | None                    # e.g. "attempts_exhausted:upstream_error", "context_limit:..."
    attempts_used: int

    @property
    def transport_failure(self) -> bool:
        return (self.reason or "").startswith(_TRANSPORT_PREFIXES)

    @property
    def context_limit(self) -> bool:
        return self.outcome == "context_limit" or (self.reason or "").startswith("context_limit")


def load_terminal_without_output(paths: RunPaths) -> dict[str, TerminalWithoutOutput]:
    path = paths.state_db_backup
    if not path.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            rows = conn.execute(
                "SELECT node_id, outcome, terminal_reason, attempts_used FROM nodes "
                "WHERE state = 'terminal' AND (outcome IS NULL OR outcome != 'succeeded')").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"state backup {path} is unreadable: {exc}") from exc
    return {str(r[0]): TerminalWithoutOutput(node_id=str(r[0]), outcome=r[1], reason=r[2], attempts_used=int(r[3] or 0))
            for r in rows}
