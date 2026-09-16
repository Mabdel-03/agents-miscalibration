"""``legacy_claim_reproduction.csv``: one row per legacy claim, with its status (spec 7.1).

The status vocabulary is the point of the file, and the five values are not interchangeable:

``pass``                    recomputed and it matched the published number
``fail``                    recomputed and it did **not** match
``unavailable``             the input is here, but no code that could have produced the
                            published number is, so there is nothing to reproduce
``BLOCKED_MISSING_SOURCE``  a named source is not on disk; the claim is not checkable
``pending_audit``           a human audit (a blinded re-grade) is outstanding

``unavailable`` and ``BLOCKED_MISSING_SOURCE`` are the two that a careless audit collapses into
``fail``, and collapsing them is how "we could not check this" gets written up as "this did not
replicate".  Every row also carries both recomputations where two exist, so a claim that
reproduces through the original code and not through an independent one is visible as such
rather than resolved in favour of whichever ran last.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.e0 import ard36, atlas, registry as R, run0

CLAIM_FILE: Final[str] = "legacy_claim_reproduction.csv"
#: The per-claim evidence that does not fit a CSV cell, one JSON object per line.
DETAIL_FILE: Final[str] = "legacy_claim_detail.jsonl"

#: The spec-7.1 columns, in order.
CLAIM_HEADER: Final[tuple[str, ...]] = (
    "claim_id", "study", "statement", "published", "recomputed_original_code",
    "recomputed_independent", "population", "denominator", "status",
)

STATUS_PASS: Final[str] = "pass"
STATUS_FAIL: Final[str] = "fail"
STATUS_UNAVAILABLE: Final[str] = "unavailable"
STATUS_PENDING_AUDIT: Final[str] = "pending_audit"
BLOCKED: Final[str] = R.BLOCKED_MISSING_SOURCE
ALLOWED_STATUSES: Final[tuple[str, ...]] = (
    STATUS_PASS, STATUS_FAIL, STATUS_UNAVAILABLE, BLOCKED, STATUS_PENDING_AUDIT, "ok",
)


def collect(sources: R.LegacySources, *, n_resamples: int = 20_000) -> list[dict[str, Any]]:
    """Every claim row from the three legs, in study order."""
    rows = [*ard36.claims(sources, n_resamples=n_resamples),
            *atlas.claims(sources),
            *run0.claims(sources)]
    bad = sorted({r["status"] for r in rows} - set(ALLOWED_STATUSES))
    if bad:
        raise ValueError(f"claim rows carry unknown status(es) {bad}; allowed: {ALLOWED_STATUSES}")
    return rows


def write(out_dir: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the CSV with the spec's columns; ``detail`` rides beside it as JSON lines."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / CLAIM_FILE
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLAIM_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CLAIM_HEADER})
    detail = folder / DETAIL_FILE
    with detail.open("w", encoding="utf-8") as handle:
        for row in rows:
            # json.dumps, never repr + quote-swapping: Python's repr quotes with apostrophes and
            # escapes nothing JSON cares about, so a detail body that is itself JSON (every one
            # of them is) produced a file that looked right and would not parse.  The body is
            # nested as an OBJECT where it parses, and kept as a string where it does not, so a
            # reader never has to guess which it got.
            body: Any = row.get("detail", {})
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    pass
            handle.write(json.dumps({"claim_id": str(row["claim_id"]),
                                     "status": str(row.get("status", "")),
                                     "detail": body}, sort_keys=True) + "\n")
    return path


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row["status"])] = out.get(str(row["status"]), 0) + 1
    return dict(sorted(out.items()))


__all__ = [
    "ALLOWED_STATUSES", "BLOCKED", "CLAIM_FILE", "CLAIM_HEADER", "STATUS_FAIL", "STATUS_PASS",
    "STATUS_PENDING_AUDIT", "STATUS_UNAVAILABLE", "collect", "summary", "write",
]
