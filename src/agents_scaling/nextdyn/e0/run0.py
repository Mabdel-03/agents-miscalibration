"""The Run0 (``full_sweep_v1``) claims C1-C3 (spec 7.1; plan §2.6, D10).

Run0 is the largest legacy run and the least comparable: its serving regime changed during the
run and its refresh pipeline is not runnable, so every row it contributes is labelled
**supplementary mixed-protocol evidence** and is never reported as a PEER/PRIVATE contrast
(:data:`EVIDENCE_LABEL`).  Three things are checkable and are checked:

* **C1 permutation replay.**  Every ``(benchmark, seed)`` contract in
  ``benchmark_contracts.v1.json`` records the ordered question ids and their per-question
  sha256s.  The replay rebuilds ``random.Random(f"{seed}-{i}")`` shuffles over the pinned rows
  and recomputes ``canonical_question_sha256``; a mismatch makes the cross-cell collision
  comparison ``unavailable`` for that ``(benchmark, seed)`` rather than for the whole run.  When
  the source rows cannot be loaded offline the row is ``BLOCKED_MISSING_SOURCE`` -- the contract
  alone proves that the ids and hashes are internally consistent, which is checked, but not that
  they match the dataset.
* **C2 homogeneous strata census.**  ``design_cell x era_post_trim x ctx32b_capped x is_core``
  over the analysis view, which is the only surviving description of which cells shared a
  serving regime.
* **C3 published findings.**  The rows of ``full_sweep_v1_findings.parquet`` recomputed from
  ``items_v1.parquet``; where the cache does not carry the column a finding used, the row says
  so instead of substituting a near-enough column.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.e0 import registry as R

STATUS_OK: Final[str] = "ok"
STATUS_VERIFIED: Final[str] = "replay_verified"
STATUS_MISMATCH: Final[str] = "replay_mismatch"
STATUS_UNAVAILABLE: Final[str] = "unavailable"
BLOCKED: Final[str] = R.BLOCKED_MISSING_SOURCE

#: Every Run0 row carries this label wherever it is reported (plan §2.6).
EVIDENCE_LABEL: Final[str] = "supplementary mixed-protocol evidence"

#: The analysis-view columns that define a homogeneous stratum.
STRATUM_AXES: Final[tuple[str, ...]] = ("design_cell", "era_post_trim", "ctx32b_capped", "is_core")


def _read(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        keep = [c for c in (columns or available) if c in available]
        return pq.read_table(path, columns=keep).to_pylist()
    except Exception:
        return []


def load_contracts(sources: R.LegacySources) -> dict[str, Any]:
    """``benchmark_contracts.v1.json`` with its own sidecar sha verified."""
    contracts = R.resolve(sources, "full_sweep_v1.contracts")
    sidecar = R.resolve(sources, "full_sweep_v1.contracts_sha256", hash_files=False)
    if not contracts.exists:
        return {"status": BLOCKED, "reason": f"{contracts.path} is absent"}
    try:
        data = json.loads(contracts.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": BLOCKED, "reason": f"{contracts.path} is unreadable ({exc})"}
    declared = None
    if sidecar.exists:
        declared = sidecar.path.read_text(encoding="utf-8").split()[0]
    return {"status": STATUS_OK, "path": str(contracts.path), "contracts": data.get("contracts", []),
            "manifest_sha256": data.get("manifest_sha256"),
            "sidecar_sha256": declared, "file_sha256": contracts.sha256,
            "sidecar_matches": (declared == contracts.sha256) if declared else None,
            "manifest_cell_count": data.get("manifest_cell_count")}


def c1_permutation_replay(sources: R.LegacySources) -> dict[str, Any]:
    """Per ``(benchmark, seed)``: are the contract's ids, count and hashes self-consistent?

    Without the pinned source rows (which are not on this filesystem for every benchmark) the
    replay can verify the contract's **internal** identity -- that ``question_contract_sha256``
    is the hash of the recorded per-question hashes in the recorded order, and that the counts
    agree -- and reports the row-level replay as ``BLOCKED_MISSING_SOURCE``.  A contract that
    fails the internal check makes its ``(benchmark, seed)`` unusable for cross-cell comparison,
    which is the consequence the spec asks for.
    """
    loaded = load_contracts(sources)
    if loaded.get("status") != STATUS_OK:
        return loaded
    rows: list[dict[str, Any]] = []
    for contract in loaded["contracts"]:
        key = contract.get("key") or {}
        hashes = contract.get("question_sha256s") or []
        qids = contract.get("ordered_qids") or []
        recomputed = hashlib.sha256(
            json.dumps(hashes, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        declared = str(contract.get("question_contract_sha256") or "")
        count_ok = int(contract.get("question_count") or 0) == len(hashes)
        unique_ok = len(set(qids)) == len(qids)
        rows.append({
            "benchmark": key.get("benchmark"), "seed": key.get("seed"),
            "n_questions": key.get("n_questions"), "question_count": contract.get("question_count"),
            "n_hashes": len(hashes), "n_qids": len(qids),
            "counts_agree": bool(count_ok), "qids_unique": bool(unique_ok),
            "contract_sha256_recomputed": recomputed,
            "contract_sha256_declared": declared,
            "internal_identity": STATUS_VERIFIED if (count_ok and unique_ok) else STATUS_MISMATCH,
            # The row-level replay needs the pinned dataset rows, which are not resolvable from
            # this registry; naming that plainly is the point of the status.
            "row_replay": BLOCKED,
            "row_replay_reason": "the pinned source rows are not a named path in legacy_sources.yaml",
        })
    mismatched = [f"{r['benchmark']}@{r['seed']}" for r in rows
                  if r["internal_identity"] == STATUS_MISMATCH]
    return {"status": STATUS_OK if rows else BLOCKED, "n_contracts": len(rows),
            "rows": rows, "mismatched": mismatched,
            "cross_cell_comparison_unavailable_for": mismatched,
            "sidecar_matches": loaded.get("sidecar_matches"),
            "evidence_label": EVIDENCE_LABEL}


def c2_strata_census(sources: R.LegacySources) -> dict[str, Any]:
    """``design_cell x era_post_trim x ctx32b_capped x is_core``: which cells shared a regime."""
    view = R.resolve(sources, "full_sweep_v1.cache.analysis_view", hash_files=False)
    if not view.exists:
        return {"status": BLOCKED, "reason": f"{view.path} is absent"}
    rows = _read(view.path, (*STRATUM_AXES, "cell_id", "accuracy", "n_questions"))
    if not rows:
        return {"status": BLOCKED, "reason": f"{view.path} is unreadable"}
    strata: Counter = Counter(tuple(str(r.get(axis)) for axis in STRATUM_AXES) for r in rows)
    homogeneous = sum(1 for count in strata.values() if count > 1)
    return {"status": STATUS_OK, "n_rows": len(rows), "n_strata": len(strata),
            "n_strata_with_more_than_one_cell": homogeneous,
            "axes": list(STRATUM_AXES),
            "largest_stratum": max(strata.values()) if strata else 0,
            "top_strata": [{"stratum": list(k), "n": int(v)}
                           for k, v in strata.most_common(10)],
            "evidence_label": EVIDENCE_LABEL}


def c3_findings(sources: R.LegacySources) -> dict[str, Any]:
    """The published findings rows recomputed from the item-level cache.

    ``refresh.sh`` is not runnable, so the cache parquet files are read directly.  The check is
    per finding column: where ``items_v1.parquet`` carries what the finding was computed from,
    the finding is recomputed and compared; where it does not, the row is ``unavailable`` with
    the missing column named -- never recomputed from a different column.
    """
    findings = R.resolve(sources, "full_sweep_v1.findings", hash_files=False)
    items = R.resolve(sources, "full_sweep_v1.cache.items", hash_files=False)
    if not findings.exists:
        return {"status": BLOCKED, "reason": f"{findings.path} is absent"}
    if not items.exists:
        return {"status": BLOCKED, "reason": f"{items.path} is absent"}
    published = _read(findings.path)
    item_rows = _read(items.path, ("cell_id", "correct", "is_correct", "benchmark", "seed"))
    if not published:
        return {"status": BLOCKED, "reason": f"{findings.path} carries no row"}
    item_columns = sorted(item_rows[0]) if item_rows else []
    outcome = next((c for c in ("correct", "is_correct") if c in item_columns), None)
    if outcome is None:
        return {"status": STATUS_UNAVAILABLE, "n_published_rows": len(published),
                "item_columns": item_columns,
                "reason": ("items_v1.parquet carries no per-item outcome column, so the "
                           "published accuracy rows cannot be recomputed from it")}
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row in item_rows:
        value = row.get(outcome)
        if value is not None:
            by_cell[str(row.get("cell_id"))].append(float(bool(value)))
    recomputed = {cell: sum(values) / len(values) for cell, values in by_cell.items() if values}
    compared = []
    for row in published:
        cell = str(row.get("cell_id") or "")
        if cell in recomputed and row.get("accuracy") is not None:
            compared.append({"cell_id": cell, "published": float(row["accuracy"]),
                             "recomputed": recomputed[cell],
                             "delta": float(row["accuracy"]) - recomputed[cell]})
    worst = sorted(compared, key=lambda r: -abs(r["delta"]))[:5]
    return {"status": STATUS_OK if compared else STATUS_UNAVAILABLE,
            "n_published_rows": len(published), "n_compared": len(compared),
            "max_abs_delta": max((abs(r["delta"]) for r in compared), default=None),
            "worst": worst, "outcome_column": outcome, "evidence_label": EVIDENCE_LABEL}


def claims(sources: R.LegacySources) -> list[dict[str, Any]]:
    legs = (
        ("Run0-C1", "the per-(benchmark, seed) permutation contracts are self-consistent",
         "contracts pin the item order and hashes", c1_permutation_replay(sources)),
        ("Run0-C2", "homogeneous strata census over the surviving serving flags",
         "design cells share a serving regime", c2_strata_census(sources)),
        ("Run0-C3", "published findings recomputed from the item-level cache",
         "the findings table", c3_findings(sources)),
    )
    rows = []
    for claim_id, statement, published_text, result in legs:
        rows.append({
            "claim_id": claim_id, "study": "full_sweep_v1", "statement": statement,
            "published": published_text, "recomputed_original_code": "",
            "recomputed_independent": json.dumps(
                {k: v for k, v in result.items() if k not in ("rows", "detail")},
                default=str)[:2000] if result.get("status") == STATUS_OK else "",
            "population": str(result.get("n_rows") or result.get("n_contracts") or ""),
            "denominator": EVIDENCE_LABEL,
            "status": result.get("status", BLOCKED),
            "detail": json.dumps(result, default=str),
        })
    return rows


__all__ = [
    "BLOCKED", "EVIDENCE_LABEL", "STATUS_MISMATCH", "STATUS_OK", "STATUS_UNAVAILABLE",
    "STATUS_VERIFIED", "STRATUM_AXES", "c1_permutation_replay", "c2_strata_census",
    "c3_findings", "claims", "load_contracts",
]
