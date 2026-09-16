"""The study_v4 atlas claims B1-B6 (spec 7.1; plan §2.6).

The atlas is the run whose design was *nominally* 59 cells and whose data turned out to hold
far fewer distinct behaviours.  B1 recomputes that collapse; B2 reproduces its AST vote
grouping from the sealed candidate pools; B3 checks its agreement column; B4 measures the
current-vs-archive selection gap it reported.

B5 and B6 are the two rows that matter most for how this study reads the atlas, and both are
**not failures**:

* **B5** is ``unavailable``.  The published 0.392 / 0.085 change-rate pair has no surviving
  generating code, so there is nothing to reproduce.  Both candidate recomputations are carried
  in the detail field and neither is declared to be "the" number, because choosing between them
  after the fact would be picking the one that matches.
* **B6** is ``BLOCKED_MISSING_SOURCE``.  No HLE judge audit exists, so the absolute HLE
  accuracies are judge-*agreement* rates and are labelled as such wherever they appear.  That is
  a statement about what was never measured, not about what failed.

Atlas rows are never reported as a communication contrast: the run has no PEER/PRIVATE axis
(see ``protocol_map.atlas_entry``).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.e0 import registry as R

STATUS_OK: Final[str] = "ok"
STATUS_UNAVAILABLE: Final[str] = "unavailable"
STATUS_FAIL: Final[str] = "fail"
BLOCKED: Final[str] = R.BLOCKED_MISSING_SOURCE

#: How far a recomputed rate may sit from the published one and still count as reproduced.
#: The atlas published its agreement rates to two digits, so anything tighter tests the printer.
B3_TOLERANCE: Final[float] = 0.02

#: The design axes that define a *nominal* atlas cell (``s1_design.py:52-77`` logic).
NOMINAL_AXES: Final[tuple[str, ...]] = ("module", "method", "checkpoint", "N", "B")

#: The published B1 figures, for comparison only.
B1_PUBLISHED: Final[Mapping[str, int]] = {
    "nominal_cells": 59, "distinct_correctness_vectors": 40, "distinct_compute_vectors": 43,
    "largest_group": 7,
}


def _read(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        keep = [c for c in (columns or available) if c in available]
        return pq.read_table(path, columns=keep).to_pylist()
    except Exception:
        return []


def _cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(str(row.get(axis)) for axis in NOMINAL_AXES)


def b1_realized_cells(sources: R.LegacySources) -> dict[str, Any]:
    """Nominal design cells vs the distinct correctness and compute vectors they produced."""
    tables_dir = R.resolve(sources, "study_v4.tables", hash_files=False)
    path = tables_dir.path / "episodes.parquet"
    if not path.exists():
        return {"status": BLOCKED, "reason": f"{path} is absent"}
    rows = _read(path, (*NOMINAL_AXES, "source_id", "native_final_correct", "spent_flops",
                        "generated_calls"))
    if not rows:
        return {"status": BLOCKED, "reason": f"{path} is unreadable"}
    correctness: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    compute: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = _cell_key(row)
        item = str(row.get("source_id"))
        correctness[key][item] = bool(row.get("native_final_correct"))
        compute[key][item] = int(row.get("generated_calls") or 0)

    def signature(table: Mapping[tuple[Any, ...], Mapping[str, Any]]) -> Counter:
        return Counter(json.dumps(sorted(values.items()), default=str) for values in table.values())

    correctness_groups = signature(correctness)
    compute_groups = signature(compute)
    return {
        "status": STATUS_OK,
        "nominal_cells": len(correctness),
        "distinct_correctness_vectors": len(correctness_groups),
        "distinct_compute_vectors": len(compute_groups),
        "largest_group": max(correctness_groups.values()) if correctness_groups else 0,
        "published": dict(B1_PUBLISHED),
        "axes": list(NOMINAL_AXES),
        "note": ("a nominal cell whose correctness vector equals another's produced no distinct "
                 "behaviour; the collapse is measured, not assumed"),
    }


def _item_candidates(cells_dir: Path, cell_id: str, source_id: str) -> dict[str, str]:
    """``candidate_id -> final answer text`` for one (cell, item), read by name.

    The pool row carries its own ``cell_id`` and ``source_id``, so the candidate text is a
    *named* read of one file rather than a scan: no directory walk, and the number of files
    opened is the number of pools examined.
    """
    path = cells_dir / cell_id / "items" / f"{source_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for entry in (data.get("episode") or {}).get("candidates") or []:
        candidate_id = str(entry.get("candidate_id") or "")
        body = entry.get("candidate") or {}
        answer = body.get("final_answer") if isinstance(body, Mapping) else None
        if candidate_id and answer:
            out[candidate_id] = str(answer)
    return out


def b2_ast_grouping(sources: R.LegacySources, *, max_seals: int = 2, max_pools: int = 60,
                    ) -> dict[str, Any]:
    """Reproduce the AST vote grouping with ``study.selection.normalize.code_key``.

    Each sealed code pool names its own cell and item, so the candidate text is resolved by
    name and the recomputed ``code_key`` is compared with the ``vote_key`` the atlas stored
    beside that candidate.  **That comparison is the reproduction**; the group-size histogram is
    a description.  Candidate ids whose text cannot be resolved are counted, because a
    candidate silently dropped changes a vote group's size without anyone noticing.

    ``max_seals`` / ``max_pools`` bound the work: the claim is that the grouping rule
    reproduces, not that the whole atlas is re-run.
    """
    seals = R.resolve(sources, "study_v4.seals", hash_files=False)
    cells = R.resolve(sources, "study_v4.cells", hash_files=False)
    if not seals.exists:
        return {"status": BLOCKED, "reason": f"{seals.path} is absent"}
    if not cells.exists:
        return {"status": BLOCKED, "reason": f"{cells.path} is absent"}
    try:
        from agents_scaling.study.selection.normalize import code_key
    except ImportError as exc:  # pragma: no cover - study is always importable here
        return {"status": STATUS_UNAVAILABLE, "reason": f"code_key unavailable ({exc})"}
    pools_seen = candidates_seen = unresolved = 0
    key_matches = key_mismatches = no_stored_key = 0
    group_sizes: Counter = Counter()
    modes: set[str] = set()
    mismatch_examples: list[dict[str, str]] = []
    for index, seal_dir in enumerate(sorted(p for p in seals.path.iterdir() if p.is_dir())):
        if index >= max_seals or pools_seen >= max_pools:
            break
        pools_path = seal_dir / "POOLS.json"
        if not pools_path.exists():
            continue
        try:
            payload = json.loads(pools_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pools = payload.get("pools") or {}
        for pool in (pools.values() if isinstance(pools, Mapping) else pools):
            if pools_seen >= max_pools:
                break
            if not isinstance(pool, Mapping) or str(pool.get("answer_format")) != "code":
                continue
            texts = _item_candidates(cells.path, str(pool.get("cell_id") or ""),
                                     str(pool.get("source_id") or ""))
            if not texts:
                continue
            pools_seen += 1
            stored_keys = pool.get("vote_keys") or {}
            keys = []
            for candidate_id in pool.get("candidate_ids") or []:
                candidates_seen += 1
                text = texts.get(str(candidate_id))
                if text is None:
                    unresolved += 1
                    continue
                # ``code_key`` returns ``(digest, grouping_mode)``; the atlas stored the digest.
                key = code_key(text)
                recomputed = str(key[0]) if isinstance(key, (tuple, list)) else str(key)
                modes.add(str(key[1]) if isinstance(key, (tuple, list)) and len(key) > 1 else "")
                keys.append(recomputed)
                stored = stored_keys.get(str(candidate_id))
                if stored is None:
                    no_stored_key += 1
                elif str(stored) == recomputed:
                    key_matches += 1
                else:
                    key_mismatches += 1
                    if len(mismatch_examples) < 3:
                        mismatch_examples.append({"candidate_id": str(candidate_id),
                                                  "stored": str(stored),
                                                  "recomputed": recomputed})
            group_sizes.update(Counter(keys).values())
    if not pools_seen:
        return {"status": BLOCKED,
                "reason": f"{seals.path}: no code pool resolved candidate text in {max_seals} seals"}
    return {"status": STATUS_OK, "pools": pools_seen, "candidates": candidates_seen,
            "unresolved_candidate_ids": unresolved,
            "vote_key_matches": key_matches, "vote_key_mismatches": key_mismatches,
            "candidates_without_stored_key": no_stored_key,
            "mismatch_examples": mismatch_examples, "grouping_modes": sorted(modes),
            "group_size_histogram": {str(k): int(v) for k, v in sorted(group_sizes.items())},
            "note": ("the reproduction is the stored vote_key vs the recomputed code_key; the "
                     "histogram describes the grouping those keys produce")}


def b3_agreement(sources: R.LegacySources) -> dict[str, Any]:
    """The published agreement column, recomputed -- or reported absent.

    The atlas published "30% null, 54% match" for ``geometry.outcome_agreement``.  The column is
    not in the geometry table this registry names, so the claim is ``unavailable``: there is
    nothing to recompute *against*, which is a different finding from a recomputation that
    disagreed.
    """
    geometry = R.resolve(sources, "study_v4.geometry", hash_files=False)
    if not geometry.exists:
        return {"status": BLOCKED, "reason": f"{geometry.path} is absent"}
    rows = _read(geometry.path)
    columns = sorted(rows[0]) if rows else []
    if "outcome_agreement" not in columns:
        return {"status": STATUS_UNAVAILABLE, "n_rows": len(rows), "columns": columns[:20],
                "reason": ("the geometry table carries no outcome_agreement column, so the "
                           "published 30%-null / 54%-match pair has no artifact to check against")}
    values = [r.get("outcome_agreement") for r in rows]
    n_null = sum(1 for v in values if v is None)
    # ``outcome_agreement`` is a *fraction* of agreeing members, not a flag: "match" is full
    # agreement (1.0).  Reading it as a boolean would score 0.0625 agreement as a match, which
    # is how the published 54% and a recomputed 100% can both be "the same column".
    present = [float(v) for v in values if v is not None]
    null_rate = n_null / len(values) if values else None
    match_rate = (sum(1.0 for v in present if v >= 1.0) / len(present)) if present else None
    published = {"null_rate": 0.30, "match_rate": 0.54}
    within = (null_rate is not None and match_rate is not None
              and abs(null_rate - published["null_rate"]) <= B3_TOLERANCE
              and abs(match_rate - published["match_rate"]) <= B3_TOLERANCE)
    return {"status": STATUS_OK if within else STATUS_FAIL, "n_rows": len(values),
            "null_rate": null_rate, "match_rate": match_rate, "published": published,
            "tolerance": B3_TOLERANCE, "n_present": len(present),
            "mean_agreement_where_present": (sum(present) / len(present)) if present else None,
            "note": ("a recomputed rate outside the tolerance is reported as a failed "
                     "reproduction, not silently as the new number")}


def b4_selection_gap(sources: R.LegacySources) -> dict[str, Any]:
    """Current-pool vs archive-pool selection gap, from the atlas's own selections table."""
    tables_dir = R.resolve(sources, "study_v4.tables", hash_files=False)
    path = tables_dir.path / "selections.parquet"
    if not path.exists():
        return {"status": BLOCKED, "reason": f"{path} is absent"}
    rows = _read(path, ("pool_kind", "oracle_coverage", "selection_gap", "selected_correct",
                        "selector_id"))
    if not rows:
        return {"status": BLOCKED, "reason": f"{path} is unreadable"}
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row.get("pool_kind"))].append(row)

    def mean(values: Sequence[Any]) -> float | None:
        clean = [float(v) for v in values if v is not None]
        return (sum(clean) / len(clean)) if clean else None

    return {"status": STATUS_OK, "n_rows": len(rows),
            "by_pool_kind": {kind: {
                "n": len(members),
                "oracle_coverage": mean([m.get("oracle_coverage") for m in members]),
                "selection_gap": mean([m.get("selection_gap") for m in members]),
                "selected_correct": mean([m.get("selected_correct") for m in members]),
            } for kind, members in sorted(by_kind.items())}}


def b5_change_rates(sources: R.LegacySources) -> dict[str, Any]:
    """The published 0.392 / 0.085 change-rate pair: ``unavailable``, both candidates carried.

    Two definitions of "changed" survive in the atlas code base -- a per-candidate change and a
    per-selection change -- and no generating script for the published pair does.  Reporting
    whichever one happens to land near 0.392 would be choosing the definition after seeing the
    answer, so both are reported and neither is declared the reproduction.
    """
    tables_dir = R.resolve(sources, "study_v4.tables", hash_files=False)
    path = tables_dir.path / "selections.parquet"
    rows = _read(path, ("selected_candidate_id", "selected_correct", "pool_kind", "source_id",
                        "cell_id", "winning_count", "valid_count", "all_singleton"))
    candidates: dict[str, Any] = {"status": STATUS_UNAVAILABLE,
                                  "published": {"rate_a": 0.392, "rate_b": 0.085},
                                  "reason": "no surviving code produced the published pair"}
    if rows:
        singleton = [bool(r.get("all_singleton")) for r in rows if r.get("all_singleton") is not None]
        disagreement = [1.0 - (float(r["winning_count"]) / float(r["valid_count"]))
                        for r in rows
                        if r.get("winning_count") is not None and r.get("valid_count")]
        candidates["candidate_recomputations"] = {
            "all_singleton_rate": (sum(singleton) / len(singleton)) if singleton else None,
            "mean_non_winning_share": (sum(disagreement) / len(disagreement)) if disagreement else None,
            "n_rows": len(rows),
        }
    return candidates


def b6_hle_judge_audit(sources: R.LegacySources) -> dict[str, Any]:
    """No HLE judge audit exists: ``BLOCKED_MISSING_SOURCE``, and the label is corrected."""
    eval_hle = R.resolve(sources, "study_v4.eval_hle", hash_files=False)
    return {
        "status": BLOCKED,
        "path": str(eval_hle.path),
        "path_exists": eval_hle.exists,
        "reason": ("no judge audit artifact exists anywhere under the named HLE evaluation "
                   "path, so the judge's agreement with a human grader was never measured"),
        "label_correction": ("every absolute HLE accuracy from this run is a judge-agreement "
                             "rate and is reported under that name"),
    }


def claims(sources: R.LegacySources) -> list[dict[str, Any]]:
    """The B1-B6 rows of ``legacy_claim_reproduction.csv``."""
    legs = (
        ("Atlas-B1", "nominal design cells collapse to distinct correctness/compute vectors",
         "59 nominal -> 40 correctness / 43 compute, largest group 7", b1_realized_cells(sources)),
        ("Atlas-B2", "AST vote grouping reproduced from the sealed candidate pools",
         "grouping as published", b2_ast_grouping(sources)),
        ("Atlas-B3", "outcome agreement recomputed from the item files",
         "30% null, 54% match", b3_agreement(sources)),
        ("Atlas-B4", "current-pool vs archive-pool selection gap",
         "archive pools close part of the gap", b4_selection_gap(sources)),
        ("Atlas-B5", "the 0.392 / 0.085 change-rate pair", "0.392 / 0.085", b5_change_rates(sources)),
        ("Atlas-B6", "HLE judge audit", "absolute HLE accuracy", b6_hle_judge_audit(sources)),
    )
    rows = []
    for claim_id, statement, published_text, result in legs:
        rows.append({
            "claim_id": claim_id, "study": "study_v4", "statement": statement,
            "published": published_text, "recomputed_original_code": "",
            "recomputed_independent": json.dumps(
                {k: v for k, v in result.items() if k not in ("status", "reason", "note")},
                default=str)[:2000] if result.get("status") in (STATUS_OK, STATUS_FAIL) else "",
            "population": str(result.get("n_rows") or result.get("nominal_cells") or ""),
            "denominator": "atlas cells / selections",
            "status": result.get("status", BLOCKED),
            "detail": json.dumps(result, default=str),
        })
    return rows


__all__ = [
    "B1_PUBLISHED", "B3_TOLERANCE", "BLOCKED", "NOMINAL_AXES", "STATUS_FAIL", "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "b1_realized_cells", "b2_ast_grouping", "b3_agreement", "b4_selection_gap",
    "b5_change_rates", "b6_hle_judge_audit", "claims",
]
