"""The ARD36 claims A1-A5, recomputed twice (spec 7.1; plan §2.6).

A1 is the load-bearing one -- ``PEER - PRIVATE`` root-to-round-3 coverage change on the 240 8B
test items, published as ``-0.045833`` with ``p = 0.0080`` and Holm ``0.0240`` -- and it is
recomputed through **two independent paths**:

1. **the original code path**: ``ard.analysis.coverage_change_arm`` on ``tables_v2`` with
   ``ard.stats.paired_contrast`` at the same 20,000 resamples and the same
   ``analysis_seed(study_seed, label)``.  This must match the published number to floating
   tolerance, because it *is* the published computation;
2. **an independent nextdyn recomputation**: the per-team state is rebuilt from
   ``decisions`` + ``labels`` with this study's own three-state logic and
   ``outcomes.team_round``, and never touches ARD36's ``coverage_current`` column.

A claim that reproduces through one path and not the other is the interesting case, and the
row carries both numbers so a reader can see which.  Path 1 imports ``ard.analysis``, which is
**not** on the ``nextdyn.ARD_IMPORTS`` allowlist; it is therefore resolved dynamically at call
time through :data:`ARD_ANALYSIS_MODULE` and a failure to import is reported as
``unavailable`` rather than crashing the audit.  (Open issue for WP0: the allowlist should name
``agents_scaling.ard.analysis.coverage_change_arm`` so this can be a static import.)

The v1 -> v2 delta is reported as a **row pair**, not as a correction: ARD36's v1 tables give
``+0.0125`` for the same estimand and the difference is the repair, which is a finding about
the earlier analysis rather than a number to quietly replace.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from agents_scaling.nextdyn import outcomes, stats
from agents_scaling.nextdyn.e0 import registry as R

#: The ARD36 analysis module, resolved at call time (see the module docstring).
ARD_ANALYSIS_MODULE: Final[str] = "agents_scaling.ard.analysis"

STATUS_PASS: Final[str] = "pass"
STATUS_FAIL: Final[str] = "fail"
STATUS_UNAVAILABLE: Final[str] = "unavailable"
STATUS_PENDING_AUDIT: Final[str] = "pending_audit"
BLOCKED: Final[str] = R.BLOCKED_MISSING_SOURCE

#: Tolerance for "the recomputation reproduces the published number".  The published values are
#: printed to six decimals, so anything tighter would be testing the printer.
TOLERANCE: Final[float] = 1e-6

#: The names ARD36 and nextdyn give the model's stated confidence signal.
CONFIDENCE_SIGNALS: Final[frozenset[str]] = frozenset({"confidence", "self_reported"})

A1_LABEL: Final[str] = "PEER_minus_PRIVATE_root_to_round3_coverage_change"
A1_ARMS: Final[tuple[str, str]] = ("PEER", "PRIVATE")
ARD36_STUDY_SEED: Final[str] = "ARD36_v1"
ARD36_ROUNDS: Final[int] = 3
ARD36_GRADED_STATUSES: Final[frozenset[str]] = frozenset({"correct", "incorrect"})
BANK_PROTOCOL: Final[str] = "NONE"
RECEIVER_PHASE: Final[str] = "receiver"


def _read(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        keep = [c for c in (columns or available) if c in available]
        return pq.read_table(path, columns=keep).to_pylist()
    except Exception:
        return []


def published(sources: R.LegacySources) -> dict[str, Any]:
    """The published ARD36 numbers, read from ``tables_v2/ard_stats.json``.

    They are read rather than transcribed: a transcript can drift from the artifact, and the
    whole point of this leg is to compare a recomputation with what the run actually reported.
    """
    stats_path = R.resolve(sources, "ard36.stats", hash_files=False)
    if not stats_path.exists:
        return {"status": BLOCKED, "reason": f"{stats_path.path} is absent"}
    try:
        data = json.loads(stats_path.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": BLOCKED, "reason": f"{stats_path.path} is unreadable ({exc})"}
    return {"status": "ok", "path": str(stats_path.path), "stats": data}


# --------------------------------------------------------------------------- A1, path 1


def coverage_change_original(sources: R.LegacySources, *, tables_key: str = "ard36.tables",
                             n_resamples: int = 20_000) -> dict[str, Any]:
    """A1 through ARD36's own code: ``coverage_change_arm`` + ``ard.stats.paired_contrast``."""
    resolved = R.resolve(sources, tables_key, hash_files=False)
    if not resolved.exists:
        return {"status": BLOCKED, "reason": f"{resolved.path} is absent", "path": str(resolved.path)}
    try:
        ard_analysis = importlib.import_module(ARD_ANALYSIS_MODULE)
    except ImportError as exc:  # pragma: no cover - the module is in the repo
        return {"status": STATUS_UNAVAILABLE,
                "reason": f"{ARD_ANALYSIS_MODULE} could not be imported ({exc})"}
    root = resolved.path.parent
    tables_dir = resolved.path.name
    legacy = ard_analysis.load_tables(root, tables_dir=tables_dir)
    index = ard_analysis.episode_index(legacy)
    arms = {arm: ard_analysis.coverage_change_arm(legacy, index, protocol=arm) for arm in A1_ARMS}
    items = sorted(set(arms[A1_ARMS[0]]) & set(arms[A1_ARMS[1]]))
    if not items:
        return {"status": STATUS_UNAVAILABLE, "reason": "no item carries both arms"}
    families = sorted({arms[A1_ARMS[0]][i][0] for i in items})
    diff = {f: np.asarray([arms[A1_ARMS[0]][i][1] - arms[A1_ARMS[1]][i][1]
                           for i in items if arms[A1_ARMS[0]][i][0] == f], dtype=float)
            for f in families}
    seed = stats.analysis_seed(ARD36_STUDY_SEED, A1_LABEL)
    resampler = stats.Resampler({f: int(diff[f].size) for f in families}, n_resamples, seed,
                                domains=families)
    result = stats.paired_contrast(diff, resampler, domains=families,
                                   weights={f: 1.0 / len(families) for f in families},
                                   n_primary=3)
    return {"status": "ok", "path": str(resolved.path), "estimate": float(result["estimate"]),
            "p": float(result["p"]), "n_items": len(items),
            "by_family": {f: float(v) for f, v in result["by_domain"].items()},
            "n": {f: int(diff[f].size) for f in families}, "n_resamples": n_resamples,
            "seed": seed, "label": A1_LABEL}


# --------------------------------------------------------------------------- A1, path 2


def _team_states(sources: R.LegacySources, tables_key: str = "ard36.tables",
                 ) -> tuple[dict[tuple[str, str], dict[int, dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Rebuild ARD36's per-team state from ``decisions`` + ``labels`` with nextdyn's logic."""
    resolved = R.resolve(sources, tables_key, hash_files=False)
    decisions = _read(resolved.path / "decisions.parquet",
                      ["decision_id", "cell_id", "episode_id", "round", "agent_id", "protocol",
                       "phase", "answer_class"])
    labels = {str(r["decision_id"]): r for r in
              _read(resolved.path / "labels.parquet", ["decision_id", "evaluator_status", "is_correct"])}
    episodes = {str(r["episode_id"]): r for r in
                _read(resolved.path / "episodes.parquet",
                      ["episode_id", "item_id", "family", "split", "checkpoint", "cohort",
                       "protocol", "fallback"])}
    grouped: dict[tuple[str, str], dict[int, list[outcomes.AgentState]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(decisions):
        if str(row.get("phase")) == RECEIVER_PHASE or str(row.get("protocol")) == BANK_PROTOCOL:
            continue
        label = labels.get(str(row["decision_id"]))
        status = str(label["evaluator_status"]) if label else ""
        if status in ARD36_GRADED_STATUSES:
            state = outcomes.STATE_CORRECT if label["is_correct"] else outcomes.STATE_WRONG
        else:
            state = outcomes.STATE_UNRESOLVED
        cls = row.get("answer_class")
        grouped[(str(row.get("cell_id") or ""), str(row["episode_id"]))][int(row["round"] or 0)].append(
            # ARD36 graded 503 rows correct while leaving answer_class empty, so validity here
            # is "the evaluator resolved it, or a class was extracted" -- not "a class exists".
            outcomes.AgentState(
                slot=index, valid=bool(cls) or status in ARD36_GRADED_STATUSES,
                cls=(str(cls) if cls else None), state=state,
                u_reason=None if state != outcomes.STATE_UNRESOLVED else (status or "not_graded"),
                confidence=None, decision_id=str(row["decision_id"])))
    per_team: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for key, by_round in grouped.items():
        rows: dict[int, dict[str, Any]] = {}
        h_lo = h_hi = 0.0
        for round_ in sorted(by_round):
            states = by_round[round_]
            row = outcomes.team_round(states, len(states), None, None, {}, root_states={},
                                      prev_h_lo=h_lo, prev_h_hi=h_hi)
            h_lo, h_hi = row["h_lo"], row["h_hi"]
            rows[round_] = row
        per_team[key] = rows
    return per_team, episodes


def coverage_change_independent(sources: R.LegacySources, *, tables_key: str = "ard36.tables",
                                n_resamples: int = 20_000) -> dict[str, Any]:
    """A1 recomputed from the raw rows with nextdyn's ``team_round``, never ``coverage_current``."""
    resolved = R.resolve(sources, tables_key, hash_files=False)
    if not resolved.exists:
        return {"status": BLOCKED, "reason": f"{resolved.path} is absent"}
    per_team, episodes = _team_states(sources, tables_key)
    if not per_team:
        return {"status": BLOCKED, "reason": f"{resolved.path}: decisions/labels are unreadable"}
    arms: dict[str, dict[str, list[float]]] = {arm: defaultdict(list) for arm in A1_ARMS}
    families: dict[str, str] = {}
    for (_cell_id, episode_id), by_round in per_team.items():
        meta = episodes.get(episode_id, {})
        if (str(meta.get("cohort")) != "natural" or str(meta.get("split")) != "test"
                or str(meta.get("checkpoint")) != "8B" or meta.get("fallback")):
            continue
        arm = str(meta.get("protocol"))
        if arm not in arms or 0 not in by_round or ARD36_ROUNDS not in by_round:
            continue
        item = str(meta.get("item_id"))
        arms[arm][item].append(by_round[ARD36_ROUNDS]["q_lo"] - by_round[0]["q_lo"])
        families[item] = str(meta.get("family"))
    items = sorted(set(arms[A1_ARMS[0]]) & set(arms[A1_ARMS[1]]))
    if not items:
        return {"status": STATUS_UNAVAILABLE, "reason": "no item carries both arms"}
    fams = sorted({families[i] for i in items})
    diff = {f: np.asarray([float(np.mean(arms[A1_ARMS[0]][i])) - float(np.mean(arms[A1_ARMS[1]][i]))
                           for i in items if families[i] == f], dtype=float) for f in fams}
    seed = stats.analysis_seed(ARD36_STUDY_SEED, A1_LABEL)
    test = stats.wild_cluster_test(diff, weights={f: 1.0 for f in fams},
                                   n_resamples=min(n_resamples, 10_000), seed=seed,
                                   label=f"{A1_LABEL}:independent")
    return {"status": "ok", "estimate": float(test["estimate"]), "p": float(test["p"]),
            "n_items": len(items), "by_family": {f: float(v) for f, v in test["by_family"].items()},
            "n": {f: int(diff[f].size) for f in fams},
            "note": ("recomputed from decisions + labels with outcomes.team_round; the test of "
                     "record here is nextdyn's wild cluster bootstrap, so its p is not expected "
                     "to equal ARD36's percentile-t p to the digit -- the estimate is")}


# --------------------------------------------------------------------------- A2-A5


def mcnemar_root_paired(sources: R.LegacySources, *, tables_key: str = "ard36.tables") -> dict[str, Any]:
    """A2: exact McNemar on the root-paired ``Q_lo`` at round 3, PEER vs PRIVATE."""
    per_team, episodes = _team_states(sources, tables_key)
    if not per_team:
        return {"status": BLOCKED, "reason": "decisions/labels are unreadable"}
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    for (_cell_id, episode_id), by_round in per_team.items():
        meta = episodes.get(episode_id, {})
        if (str(meta.get("cohort")) != "natural" or str(meta.get("split")) != "test"
                or str(meta.get("checkpoint")) != "8B" or meta.get("fallback")):
            continue
        arm = str(meta.get("protocol"))
        if arm in A1_ARMS and ARD36_ROUNDS in by_round:
            by_item[str(meta.get("item_id"))][arm] = by_round[ARD36_ROUNDS]["q_lo"]
    items = sorted(i for i, arms in by_item.items() if set(A1_ARMS) <= set(arms))
    if not items:
        return {"status": STATUS_UNAVAILABLE, "reason": "no item carries both arms at round 3"}
    result = stats.mcnemar_exact([by_item[i][A1_ARMS[0]] for i in items],
                                 [by_item[i][A1_ARMS[1]] for i in items])
    return {"status": "ok", **result, "n_items": len(items)}


def within_item_confidence_auc(sources: R.LegacySources) -> dict[str, Any]:
    """A3: the within-item confidence AUC, read from the repaired calibration panel."""
    resolved = R.resolve(sources, "ard36.calibration_panel", hash_files=False)
    if not resolved.exists:
        return {"status": BLOCKED, "reason": f"{resolved.path} is absent"}
    rows = _read(resolved.path)
    if not rows:
        return {"status": STATUS_UNAVAILABLE, "reason": f"{resolved.path} carries no row",
                "path": str(resolved.path)}
    signals = sorted({str(r.get("signal")) for r in rows})
    # ARD36 named the model's stated confidence ``self_reported``; nextdyn names it
    # ``confidence``.  Both are admitted so the claim does not turn on a rename.
    eligible = [r for r in rows
                if r.get("within_item_auc") is not None
                and str(r.get("signal")) in CONFIDENCE_SIGNALS]
    if not eligible:
        return {"status": STATUS_UNAVAILABLE, "n_rows": len(rows), "signals_present": signals,
                "reason": ("the panel carries no confidence row with a within-item AUC; every "
                           "such cell was excluded for holding no mixed-correctness item")}
    values = [float(r["within_item_auc"]) for r in eligible]
    weights = [int(r.get("eligible_items") or 0) for r in eligible]
    weighted = (float(np.average(values, weights=weights)) if sum(weights) else float(np.mean(values)))
    return {"status": "ok", "n_rows": len(values), "signals_present": signals,
            "mean_within_item_auc": float(np.mean(values)),
            "eligible_item_weighted_auc": weighted,
            "n_eligible_items": int(sum(weights)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "v1_note": ("v1's panel clustered on an item_id column that did not exist, so its "
                        "within-item numbers are pooled numbers: no comparable v1 value exists")}


def passk_fixed_set(sources: R.LegacySources, *, ks: Sequence[int] = (1, 2, 5, 16)) -> dict[str, Any]:
    """A4: pass@k on the fixed, fully graded bank set (the D-11 items excluded by construction)."""
    from agents_scaling.ard.reference_metrics import pass_at_k

    resolved = R.resolve(sources, "ard36.bank_summary", hash_files=False)
    if not resolved.exists:
        return {"status": BLOCKED, "reason": f"{resolved.path} is absent"}
    rows = _read(resolved.path)
    if not rows:
        return {"status": STATUS_UNAVAILABLE, "reason": f"{resolved.path} carries no row"}
    correct_column = next((c for c in ("c_correct", "n_correct") if c in rows[0]), None)
    if correct_column is None:
        return {"status": STATUS_UNAVAILABLE,
                "reason": "bank_summary carries no per-item correct count",
                "columns": sorted(rows[0])}
    # The fixed set: fully graded items only, and a single draw count, so pass@k at every k is
    # a reading of one panel.  Items at a different n_draws are the D-11 duplicated banks and
    # are excluded by construction rather than by an id list.
    usable = [r for r in rows
              if not r.get("n_ungraded") and r.get("n_draws")
              and r.get(correct_column) is not None
              and str(r.get("checkpoint") or "8B") == "8B"
              and str(r.get("split") or "test") == "test"]
    if not usable:
        return {"status": STATUS_UNAVAILABLE,
                "reason": "bank_summary carries no fully graded 8B test row",
                "columns": sorted(rows[0])}
    sizes = Counter(int(r["n_draws"]) for r in usable)
    modal = sizes.most_common(1)[0][0]
    fixed = [r for r in usable if int(r["n_draws"]) == modal]
    out: dict[str, Any] = {
        "status": "ok", "n_items": len(fixed), "n_draws": modal,
        "n_draw_sizes_seen": dict(sorted(sizes.items())),
        "excluded_other_draw_counts": len(usable) - len(fixed),
        "correct_column": correct_column,
        "published_vote_at_16": 0.8480,
        "recomputed_plurality": (float(np.mean([float(r["plurality_correct"]) for r in fixed
                                                if r.get("plurality_correct") is not None]))
                                 if any(r.get("plurality_correct") is not None for r in fixed)
                                 else None),
    }
    for k in ks:
        if k <= modal:
            out[f"pass_at_{k}"] = float(np.mean(
                [pass_at_k(int(r["n_draws"]), int(r[correct_column]), int(k)) for r in fixed]))
    return out


def receiver_rate(sources: R.LegacySources, *, tables_key: str = "ard36.tables") -> dict[str, Any]:
    """A5 (supplementary): the ARD36 receiver probe rate."""
    resolved = R.resolve(sources, tables_key, hash_files=False)
    path = resolved.path / "receiver_outcomes.parquet"
    if not path.exists():
        return {"status": BLOCKED, "reason": f"{path} is absent"}
    rows = _read(path)
    if not rows:
        return {"status": STATUS_UNAVAILABLE, "reason": f"{path} carries no row"}
    column = next((c for c in ("selected_correct", "is_correct", "correct") if c in rows[0]), None)
    if column is None:
        return {"status": STATUS_UNAVAILABLE, "reason": "no outcome column in receiver_outcomes",
                "columns": sorted(rows[0])}
    values = [float(bool(r[column])) for r in rows if r.get(column) is not None]
    return {"status": "ok", "n": len(values), "n_correct": int(sum(values)),
            "rate": float(np.mean(values)) if values else None, "column": column}


def discordant_audit(sources: R.LegacySources, *, tables_key: str = "ard36.tables",
                     limit: int = 200) -> list[dict[str, Any]]:
    """Rows whose grade and canonical class disagree, exported for a blinded re-grade.

    MATH rows stay ``pending_manual``: nextdyn's MATH grader is the same sympy path ARD36 used,
    so a second run of it is not an independent check.  MC rows get an independent re-grade
    because nextdyn's MC grader is a second implementation.
    """
    resolved = R.resolve(sources, tables_key, hash_files=False)
    labels = _read(resolved.path / "labels.parquet",
                   ["decision_id", "evaluator_status", "is_correct", "answer_class", "item_id"])
    decisions = {str(r["decision_id"]): r for r in
                 _read(resolved.path / "decisions.parquet",
                       ["decision_id", "answer_class", "item_id", "episode_id"])}
    episodes = {str(r["episode_id"]): r for r in
                _read(resolved.path / "episodes.parquet", ["episode_id", "family"])}
    out: list[dict[str, Any]] = []
    for row in labels:
        decision = decisions.get(str(row["decision_id"]))
        if decision is None:
            continue
        label_class = row.get("answer_class")
        decision_class = decision.get("answer_class")
        if label_class and decision_class and str(label_class) != str(decision_class):
            family = str(episodes.get(str(decision.get("episode_id")), {}).get("family") or "")
            out.append({
                "decision_id": str(row["decision_id"]), "item_id": str(row.get("item_id") or ""),
                "family": family, "label_class": str(label_class),
                "decision_class": str(decision_class),
                "evaluator_status": str(row.get("evaluator_status") or ""),
                "audit_status": STATUS_PENDING_AUDIT,
                "audit_route": ("pending_manual" if family.startswith("math")
                                else "independent_mc_regrade"),
            })
        if len(out) >= limit:
            break
    return out


def write_discordant(out_dir: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    import csv

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ard36_discordant_audit.csv"
    header = ["decision_id", "item_id", "family", "label_class", "decision_class",
              "evaluator_status", "audit_status", "audit_route"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return path


# --------------------------------------------------------------------------- assembly


def _compare(recomputed: Mapping[str, Any], published_value: float | None) -> str:
    if recomputed.get("status") != "ok":
        return str(recomputed.get("status"))
    if published_value is None:
        return STATUS_UNAVAILABLE
    return STATUS_PASS if abs(float(recomputed["estimate"]) - float(published_value)) <= TOLERANCE \
        else STATUS_FAIL


def claims(sources: R.LegacySources, *, n_resamples: int = 20_000) -> list[dict[str, Any]]:
    """The A1-A5 rows of ``legacy_claim_reproduction.csv``, plus the v1 delta pair for A1."""
    pub = published(sources)
    contrasts = ((pub.get("stats") or {}).get("primary") or {}).get("contrasts") or {}
    holm = ((pub.get("stats") or {}).get("primary") or {}).get("holm") or {}
    a1_published = contrasts.get(A1_LABEL, {}).get("estimate")
    a1_p = contrasts.get(A1_LABEL, {}).get("p")
    a1_holm = holm.get(A1_LABEL, {}).get("p_holm")

    original = coverage_change_original(sources, n_resamples=n_resamples)
    independent = coverage_change_independent(sources, n_resamples=n_resamples)
    v1 = coverage_change_original(sources, tables_key="ard36.tables_v1", n_resamples=n_resamples)

    rows: list[dict[str, Any]] = [{
        "claim_id": "ARD36-A1",
        "study": "ard36",
        "statement": ("PEER minus PRIVATE root-to-round-3 active coverage change on the 240 8B "
                      "natural test items"),
        "published": _fmt(a1_published, a1_p, a1_holm),
        "recomputed_original_code": _fmt(original.get("estimate"), original.get("p")),
        "recomputed_independent": _fmt(independent.get("estimate"), independent.get("p")),
        "population": f"{original.get('n_items', 0)} items, 8B natural test, both arms",
        "denominator": "source items, equal task weights",
        "status": _compare(original, a1_published),
        "detail": json.dumps({"original": original, "independent": independent}, default=str),
    }, {
        "claim_id": "ARD36-A1-v1",
        "study": "ard36",
        "statement": "the same estimand computed on the pre-repair v1 tables (delta row, not a correction)",
        "published": "v1 reported +0.0125 for this estimand",
        "recomputed_original_code": _fmt(v1.get("estimate"), v1.get("p")),
        "recomputed_independent": "",
        "population": f"{v1.get('n_items', 0)} items on tables_v1",
        "denominator": "source items, equal task weights",
        "status": "ok" if v1.get("status") == "ok" else str(v1.get("status")),
        "detail": json.dumps(v1, default=str),
    }]

    a2 = mcnemar_root_paired(sources)
    rows.append({
        "claim_id": "ARD36-A2", "study": "ard36",
        "statement": "root-paired exact McNemar on round-3 coverage, PEER vs PRIVATE",
        "published": "12 vs 1 discordant, p = 0.0034",
        "recomputed_original_code": "",
        "recomputed_independent": (f"n10={a2.get('n10')} n01={a2.get('n01')} p={a2.get('p')}"
                                   if a2.get("status") == "ok" else ""),
        "population": f"{a2.get('n_items', 0)} root-paired items",
        "denominator": "discordant pairs only", "status": a2.get("status", BLOCKED),
        "detail": json.dumps(a2, default=str),
    })

    a3 = within_item_confidence_auc(sources)
    rows.append({
        "claim_id": "ARD36-A3", "study": "ard36",
        "statement": "within-item confidence AUC (published 0.5723)",
        "published": "0.5723", "recomputed_original_code": "",
        "recomputed_independent": (f"mean {a3.get('mean_within_item_auc'):.4f} over "
                                   f"{a3.get('n_rows')} panel rows" if a3.get("status") == "ok" else ""),
        "population": "mixed-correctness items only", "denominator": "eligible items",
        "status": a3.get("status", BLOCKED), "detail": json.dumps(a3, default=str),
    })

    a4 = passk_fixed_set(sources)
    rows.append({
        "claim_id": "ARD36-A4", "study": "ard36",
        "statement": "pass@k on the fixed fully graded bank set",
        "published": "pass@1 0.8304, pass@2 0.8567, pass@5 0.8780, pass@16 0.9123, vote@16 0.8480",
        "recomputed_original_code": "",
        "recomputed_independent": json.dumps(
            {k: v for k, v in a4.items() if k.startswith("pass_at_")}) if a4.get("status") == "ok" else "",
        "population": f"{a4.get('n_items', 0)} items at {a4.get('n_draws')} draws",
        "denominator": "fully graded bank items", "status": a4.get("status", BLOCKED),
        "detail": json.dumps(a4, default=str),
    })

    a5 = receiver_rate(sources)
    rows.append({
        "claim_id": "ARD36-A5", "study": "ard36",
        "statement": "receiver probe success rate (supplementary; published 947/333 = 0.7398)",
        "published": "0.7398", "recomputed_original_code": "",
        "recomputed_independent": (f"{a5.get('n_correct')}/{a5.get('n')} = {a5.get('rate')}"
                                   if a5.get("status") == "ok" else ""),
        "population": f"{a5.get('n', 0)} receiver probes", "denominator": "probes",
        "status": a5.get("status", BLOCKED), "detail": json.dumps(a5, default=str),
    })
    return rows


def _fmt(estimate: Any, p: Any = None, holm: Any = None) -> str:
    if estimate is None:
        return ""
    parts = [f"{float(estimate):.6f}"]
    if p is not None:
        parts.append(f"p={float(p):.6f}")
    if holm is not None:
        parts.append(f"holm={float(holm):.6f}")
    return " ".join(parts)


__all__ = [
    "A1_ARMS", "A1_LABEL", "ARD36_STUDY_SEED", "CONFIDENCE_SIGNALS", "ARD_ANALYSIS_MODULE", "BLOCKED", "STATUS_FAIL",
    "STATUS_PASS", "STATUS_PENDING_AUDIT", "STATUS_UNAVAILABLE", "TOLERANCE", "claims",
    "coverage_change_independent", "coverage_change_original", "discordant_audit",
    "mcnemar_root_paired", "passk_fixed_set", "published", "receiver_rate",
    "within_item_confidence_auc", "write_discordant",
]
