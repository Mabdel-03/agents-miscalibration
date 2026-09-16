"""The miscalibration panel (plan §5.8) — declared in ``compact.py`` and never built.

``compact.py`` declares ``calibration_panel`` in its schema and emits ``[]`` for it, so the one
axis this repository originally existed to measure came out of the pipeline empty. This module
fills it from the compacted ``decisions`` and ``labels`` tables.

**This panel is exploratory and beyond the specification.** The string "ECE" does not occur in
the 880-line spec; it is carried at the user's request and recorded as amendment M1. Nothing
here enters a controller feature, and no row of it is a primary endpoint.

Two rules travel with the estimates, both from plan §5.8:

* **Missing confidence is missing.** A decision whose report carried no ``<confidence>`` is
  excluded from the confidence signal and counted in ``conf_missing_rate`` — never imputed to
  0.5. That missingness is *not* random: it lands on cap-hit draws (amendment I7), so the rate
  is reported beside every estimate rather than buried.
* **Plug-in ECE is never compared across cells with different n.** Valid-confidence counts vary
  by round because missingness rises with cap hits, so the debiased-L2 estimator is reported
  alongside, and a reader comparing rows is handed both n and valid_n to do it honestly.

The headline statistic is ``system_minus_agent_ece``: the ECE of a team-level confidence signal
minus the mean per-agent ECE, per protocol × round. It answers whether multi-round exchange
amplifies or damps *system* overconfidence relative to its members — which is the question a
round-dynamics study can ask about calibration that a single-turn study cannot.

**Three defects made the frozen v1 panel unreadable, and a fourth is only reported here.**

* **The item clustering was inert.**  ``within_item_auc`` clusters on ``item_id`` read off the
  decision rows, but v1's ``decisions.parquet`` carried no ``item_id`` column at all, so every
  row's item was ``None``, every row landed in one pseudo-cluster, and every "within-item"
  number in the v1 file is a *pooled* number wearing a within-item label.  298 of its 300 rows
  report ``eligible_items = 1``, which is the fingerprint.  ``item_id`` exists in the repaired
  tables, so the clustering works from them without a code change — :data:`N_ITEMS` is written
  beside the AUC so a future reader can see at a glance whether it clustered or collapsed.
* **A decision could be counted twice in one estimator.**  A round-0 decision is generated once
  and aliased into every arm that starts from it (``panels.py``).  That replication is almost
  entirely *across* cells, and this panel estimates *within* a cell, so it is nearly harmless
  here — but the D-11 aliased-root duplication does put the same ``decision_id`` twice inside a
  single bank cell.  :func:`_dedupe_cell` enforces the invariant directly: one row per
  ``decision_id`` per panel cell, first occurrence wins, count reported in ``dupes_dropped``.
* **The discrete estimator never fired.**  ``cell_calibration`` dispatches on the signal name,
  and neither of this module's two discrete signals was in the registry, so a 10-bin equal-mass
  estimator was run on finite support — where the quantile edges collapse and the bin count is
  not what it says.  :data:`DISCRETE_PANEL_SIGNALS` declares them, and their debiased-L2 column
  is NaN afterwards *by design* (the variance correction is degenerate on atoms).

* **Not repaired: the bank lane forms pseudo-teams.**  The team signals group by
  ``(protocol, checkpoint, cohort, round, episode_id)``.  In the ``NONE``/ROOTS lane that group
  is a *bank* of 14-16 independent draws of one item, not a five-agent team, so a ``NONE`` row's
  ``system_vote_fraction`` is a draw-agreement rate and its "team" is not a team.  Changing the
  population is outside this repair, so ``median_team_size`` is written on every system row and
  the ``NONE`` rows should be read with it in hand.

**Confidence on this run is degenerate**, which governs how every number here may be read: the
self-reported signal lives on ~20 atoms, 95% of its mass sits in four of them, and the modal
atom alone carries over 40%.  An ECE computed on that support is dominated by where the atoms
fall, so :func:`tie_rate` — the probability that two draws from the cell share an atom — is
written beside every estimate.  A small ECE difference between two rows with tie rates near 1
is not a finding; it is the grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

#: Per-decision confidence signals.  ``self_reported`` is the report's own forecast; the
#: logprob signal is the model's answer-token confidence, which is available even when the
#: verbal forecast is missing and so has a different denominator.
SIGNAL_SELF = "self_reported"
SIGNAL_LOGPROB = "answer_logprob"

#: Team-level signals over the five current reports of a round (plan §5.8 "system confidence").
SIGNAL_VOTE_FRACTION = "system_vote_fraction"
SIGNAL_MEAN_AGREEING = "system_mean_agreeing"
SIGNAL_MEAN_ALL = "system_mean_all"

AGENT_SIGNALS = (SIGNAL_SELF, SIGNAL_LOGPROB)
SYSTEM_SIGNALS = (SIGNAL_VOTE_FRACTION, SIGNAL_MEAN_AGREEING, SIGNAL_MEAN_ALL)

#: Signals this panel scores with the exact-atom estimator.  ``system_vote_fraction`` is
#: discrete by construction (k/n over the round's reports) and is registered as such in
#: ``calib.DISCRETE_SIGNALS``.  ``self_reported`` is *not* discrete by construction — a verbal
#: forecast could take any value in [0, 1] — it is discrete in this run because the models
#: answer on a fixed ordinal grid.  That is a property of the data, so it is declared here at
#: the call site rather than added to the shared by-construction registry, where it would
#: silently change the estimator for every other caller of the name ``self_reported``.
DISCRETE_PANEL_SIGNALS: Final[frozenset[str]] = frozenset({SIGNAL_SELF, SIGNAL_VOTE_FRACTION})

#: Which unit a row's ``n`` counts.  Agent rows count decisions, system rows count
#: episode-rounds; the two denominators are not comparable and are never mixed.
GRAIN_AGENT: Final[str] = "decision"
GRAIN_SYSTEM: Final[str] = "episode_round"

#: The evaluator verdicts that make a decision usable.  Anything else (``ambiguous``,
#: ``timeout``, no label at all) is an unobserved outcome, never a zero.
GRADED_STATUSES: Final[frozenset[str]] = frozenset({"correct", "incorrect"})


def _calib():
    """``analysis/nb_lib/calib.py`` — the estimators this repo already validated."""
    root = Path(__file__).resolve().parents[3] / "analysis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from nb_lib import calib  # type: ignore

    return calib


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def within_item_auc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean within-item ranking AUC, its denominator, and the cluster count behind both.

    Plan §5.8: an item whose candidates are all correct or all wrong carries no ranking
    information and returns ``None`` rather than 0.5, so a panel row's AUC denominator can
    never equal its row count.  Pooled AUC is a different quantity and is not computed here.

    ``n_items`` is the number of distinct clusters this actually ran over.  It exists because
    the v1 panel silently reported the pooled AUC under this name: the ``item_id`` it clustered
    on did not exist in the table it read, so every row fell into one cluster.  A caller that
    sees ``n_items == 1`` on a cell of a thousand rows is looking at a pooled number.
    """
    calib = _calib()
    by_item: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        conf, correct = row.get("conf"), row.get("correct")
        if conf is None or correct is None:
            continue
        by_item[str(row.get("item_id") or "")].append((float(conf), int(correct)))
    scores: list[float] = []
    for pairs in by_item.values():
        labels = {c for _p, c in pairs}
        if len(labels) < 2:
            continue                      # no mixed pair: undefined, not 0.5
        conf = [p for p, _c in pairs]
        correct = [c for _p, c in pairs]
        value = calib.auroc(conf, correct)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            scores.append(float(value))
    return {
        "within_item_auc": (sum(scores) / len(scores)) if scores else None,
        "eligible_items": len(scores),
        "n_items": len(by_item),
    }


def tie_rate(conf: Sequence[float]) -> dict[str, Any]:
    """How much of the signal is ties: the atom census that governs how an ECE may be read.

    ``tie_rate`` is the probability that two rows drawn without replacement from the cell carry
    the *same* confidence value, ``sum_a n_a (n_a - 1) / n (n - 1)``.  It is the honest
    companion to an ECE on this run, where confidence is reported on a coarse ordinal grid: a
    tie rate near 1 means the equal-mass bins have collapsed onto a handful of atoms and the
    estimate is a property of the grid, not of a bin count anyone chose.  It is also exactly
    the quantity that flattens :func:`within_item_auc` toward 0.5.
    """
    n = len(conf)
    if n == 0:
        return {"tie_rate": None, "n_atoms": 0, "modal_atom": None, "modal_atom_share": None}
    atoms = Counter(round(float(c), 6) for c in conf)
    modal, modal_n = max(atoms.items(), key=lambda kv: (kv[1], kv[0]))
    ties = sum(k * (k - 1) for k in atoms.values())
    return {
        "tie_rate": (ties / (n * (n - 1))) if n > 1 else None,
        "n_atoms": len(atoms),
        "modal_atom": float(modal),
        "modal_atom_share": modal_n / n,
    }


def _panel_row(rows: Sequence[Mapping[str, Any]], *, key: Mapping[str, Any], signal: str,
               n_total: int, n_missing: int, extra: Mapping[str, Any] | None = None,
               ) -> dict[str, Any]:
    calib = _calib()
    signal_type = calib.DISCRETE if signal in DISCRETE_PANEL_SIGNALS else calib.CONTINUOUS
    conf = [float(r["conf"]) for r in rows]
    correct = [int(r["correct"]) for r in rows]
    summary = (calib.cell_calibration(conf, correct, signal, signal_type=signal_type)
               if conf else {})
    mean_conf = sum(conf) / len(conf) if conf else None
    accuracy = sum(correct) / len(correct) if correct else None
    nll = None
    if conf:
        eps = 1e-12
        nll = -sum(
            math.log(max(p, eps)) if y else math.log(max(1.0 - p, eps))
            for p, y in zip(conf, correct)
        ) / len(conf)
    mce = None
    if conf:
        edges = calib.equal_mass_edges(conf, 10)
        idx = calib.bin_index(conf, edges)
        gaps = []
        for b in set(idx):
            sel = [(p, y) for p, y, i in zip(conf, correct, idx) if i == b]
            if sel:
                gaps.append(abs(sum(p for p, _ in sel) / len(sel)
                                - sum(y for _, y in sel) / len(sel)))
        mce = max(gaps) if gaps else None
    return {
        **key,
        "signal": signal,
        "signal_type": signal_type,
        "n": int(n_total),
        "valid_n": int(len(rows)),
        "conf_missing_rate": (n_missing / n_total) if n_total else None,
        "mean_confidence": mean_conf,
        "accuracy": accuracy,
        "signed_gap": summary.get("signed_gap"),
        "brier": summary.get("brier"),
        "brier_reliability": summary.get("brier_rel"),
        "brier_resolution": summary.get("resolution"),
        "nll": nll,
        "ece_primary": summary.get("ece"),
        "ece_debiased_l2": summary.get("ece_db"),
        "ece_15bin_equal_width": summary.get("ece_plugin15"),
        "mce": mce,
        "cox_intercept": summary.get("cox_intercept"),
        "cox_slope": summary.get("cox_slope"),
        **within_item_auc(rows),
        **tie_rate(conf),
        **(extra or {}),
    }


def build_panel(tables_dir: str | os.PathLike) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(calibration_panel, system_minus_agent)`` from the compacted tables."""
    tables = Path(tables_dir)
    decisions = _read_parquet(tables / "decisions.parquet")
    labels = {str(r["decision_id"]): r for r in _read_parquet(tables / "labels.parquet")}
    if not decisions:
        raise SystemExit(f"no decisions.parquet under {tables}; compact the run first")

    # Per-decision rows, keyed by the panel's grain.
    agent_rows: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {SIGNAL_SELF: [], SIGNAL_LOGPROB: [], "n": 0, "missing": 0, "dupes": 0})
    # Team rows need the whole round assembled first.
    per_round: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    # One decision may enter one cell's estimators exactly once.  Round-0 decisions are aliased
    # into every arm that starts from them, but the arms are *different* panel cells, so that
    # replication never reaches inside an estimator; what does is the D-11 aliased-root
    # duplication, which lands one decision_id twice in one bank cell — and some of those pairs
    # disagree on their payload, so the tie-break has to be stated rather than assumed.  First
    # in table order wins: the table's order is fixed, and dropping every copy of a
    # contradictory id would discard graded outcomes over a defect in a different lane.
    seen: dict[tuple, set[str]] = defaultdict(set)

    for row in decisions:
        decision_id = str(row.get("decision_id"))
        label = labels.get(decision_id)
        if label is None or str(label.get("evaluator_status")) not in GRADED_STATUSES:
            continue                      # ungraded / ambiguous: unobserved, never 0
        # Read the verdict, not ``labels.is_correct``: that column is the old coerced bool and
        # cannot distinguish "graded wrong" from "the grader could not resolve".
        correct = int(str(label.get("evaluator_status")) == "correct")
        key = (str(row.get("protocol")), str(row.get("checkpoint")),
               str(row.get("cohort") or "natural"), int(row.get("round") or 0))
        bucket = agent_rows[key]
        if decision_id in seen[key]:
            bucket["dupes"] += 1
            continue
        seen[key].add(decision_id)
        bucket["n"] += 1
        conf = row.get("confidence")
        if conf is None:
            bucket["missing"] += 1
        else:
            bucket[SIGNAL_SELF].append(
                {"conf": float(conf), "correct": correct, "item_id": row.get("item_id")})
        logprob = row.get("answer_logprob")
        if logprob is not None:
            bucket[SIGNAL_LOGPROB].append(
                {"conf": float(math.exp(min(0.0, float(logprob)))), "correct": correct,
                 "item_id": row.get("item_id")})
        per_round[(*key, str(row.get("episode_id")))].append(
            {"conf": conf, "correct": correct, "class": row.get("answer_class"),
             "item_id": row.get("item_id")})

    panel: list[dict[str, Any]] = []
    for key, bucket in sorted(agent_rows.items(), key=lambda kv: str(kv[0])):
        base = {"protocol": key[0], "checkpoint": key[1], "cohort": key[2], "round": key[3]}
        extra = {"dupes_dropped": bucket["dupes"], "grain": GRAIN_AGENT, "median_team_size": None}
        for signal in AGENT_SIGNALS:
            rows = bucket[signal]
            missing = bucket["missing"] if signal == SIGNAL_SELF else bucket["n"] - len(rows)
            panel.append(_panel_row(rows, key=base, signal=signal, n_total=bucket["n"],
                                    n_missing=missing, extra=extra))

    # Team-level signals: one row per episode-round, scored against the plurality's correctness.
    system: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {s: [] for s in SYSTEM_SIGNALS} | {"n": 0, "missing": 0, "sizes": []})
    for (protocol, checkpoint, cohort, round_, _episode), members in per_round.items():
        classes = [m["class"] for m in members if m["class"]]
        if not classes:
            continue
        winner = max(set(classes), key=lambda c: (classes.count(c), c))
        agreeing = [m for m in members if m["class"] == winner]
        # The team is "correct" when its plurality answer is correct.
        team_correct = int(bool(agreeing and agreeing[0]["correct"]))
        key = (protocol, checkpoint, cohort, round_)
        bucket = system[key]
        bucket["n"] += 1
        bucket["sizes"].append(len(members))
        item = members[0]["item_id"]
        bucket[SIGNAL_VOTE_FRACTION].append(
            {"conf": len(agreeing) / len(members), "correct": team_correct, "item_id": item})
        verbal_agreeing = [m["conf"] for m in agreeing if m["conf"] is not None]
        verbal_all = [m["conf"] for m in members if m["conf"] is not None]
        if verbal_agreeing:
            bucket[SIGNAL_MEAN_AGREEING].append(
                {"conf": sum(verbal_agreeing) / len(verbal_agreeing),
                 "correct": team_correct, "item_id": item})
        else:
            bucket["missing"] += 1
        if verbal_all:
            bucket[SIGNAL_MEAN_ALL].append(
                {"conf": sum(verbal_all) / len(verbal_all),
                 "correct": team_correct, "item_id": item})

    for key, bucket in sorted(system.items(), key=lambda kv: str(kv[0])):
        base = {"protocol": key[0], "checkpoint": key[1], "cohort": key[2], "round": key[3]}
        sizes = sorted(bucket["sizes"])
        # A "team" in the ROOTS lane is a bank of independent draws, not five agents; this is
        # the column that says so.  See the fourth defect in the module docstring.
        extra = {"dupes_dropped": agent_rows[key]["dupes"], "grain": GRAIN_SYSTEM,
                 "median_team_size": float(sizes[len(sizes) // 2]) if sizes else None}
        for signal in SYSTEM_SIGNALS:
            rows = bucket[signal]
            panel.append(_panel_row(rows, key=base, signal=signal, n_total=bucket["n"],
                                    n_missing=bucket["n"] - len(rows), extra=extra))

    # The headline: system ECE minus mean agent ECE, per protocol x round.
    by_key = {(r["protocol"], r["checkpoint"], r["cohort"], r["round"], r["signal"]): r
              for r in panel}
    delta: list[dict[str, Any]] = []
    for (protocol, checkpoint, cohort, round_), _bucket in sorted(system.items(),
                                                                  key=lambda kv: str(kv[0])):
        agent = by_key.get((protocol, checkpoint, cohort, round_, SIGNAL_SELF))
        for signal in SYSTEM_SIGNALS:
            sysrow = by_key.get((protocol, checkpoint, cohort, round_, signal))
            if not agent or not sysrow:
                continue
            a, s = agent.get("ece_primary"), sysrow.get("ece_primary")
            if a is None or s is None:
                continue
            delta.append({
                "protocol": protocol, "checkpoint": checkpoint, "cohort": cohort,
                "round": round_, "system_signal": signal,
                "agent_ece": a, "system_ece": s, "system_minus_agent_ece": s - a,
                "agent_valid_n": agent.get("valid_n"), "system_valid_n": sysrow.get("valid_n"),
                "agent_conf_missing_rate": agent.get("conf_missing_rate"),
                # Both sides sit on coarse atom grids, so a delta is only as meaningful as the
                # grids under it: a difference smaller than the spacing between atoms is not a
                # finding.  Carried here so no reader has to join back to the panel to see it.
                "agent_tie_rate": agent.get("tie_rate"),
                "system_tie_rate": sysrow.get("tie_rate"),
                "agent_n_atoms": agent.get("n_atoms"),
                "system_n_atoms": sysrow.get("n_atoms"),
                "median_team_size": sysrow.get("median_team_size"),
            })
    return panel, delta


def check_discrete_invariant(panel: Iterable[Mapping[str, Any]]) -> None:
    """Every discrete-signal row must carry a NaN debiased-L2 ECE, and no other row may.

    The debiased estimator's bias correction is degenerate on finite support, so a *number* in
    ``ece_debiased_l2`` on a discrete row is the signature of the dispatch silently falling
    through to the binned path — the exact defect this module was repaired for.  Checking it
    where the panel is written means a future regression fails loudly instead of publishing a
    plausible-looking column.
    """
    for row in panel:
        discrete = row["signal"] in DISCRETE_PANEL_SIGNALS
        value = row.get("ece_debiased_l2")
        is_nan = value is None or (isinstance(value, float) and math.isnan(value))
        if discrete and not is_nan:
            raise SystemExit(f"discrete signal {row['signal']} at {row['protocol']}/"
                             f"{row['checkpoint']}/{row['cohort']}/r{row['round']} carries "
                             f"ece_debiased_l2={value}; the exact-atom dispatch did not fire")
        if not discrete and row["valid_n"] and is_nan:
            raise SystemExit(f"continuous signal {row['signal']} at {row['protocol']}/"
                             f"{row['checkpoint']}/{row['cohort']}/r{row['round']} has no "
                             f"ece_debiased_l2 despite {row['valid_n']} valid rows")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.calibration_panel",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True, help="the compacted tables directory")
    ap.add_argument("--out", default=None, help="defaults to --tables")
    args = ap.parse_args(argv)

    tables = Path(args.tables)
    out = Path(args.out) if args.out else tables
    panel, delta = build_panel(tables)
    out.mkdir(parents=True, exist_ok=True)

    from agents_scaling.study.aggregate import _write_parquet

    check_discrete_invariant(panel)
    _write_parquet(out / "calibration_panel.parquet", panel)
    _write_parquet(out / "calibration_system_minus_agent.parquet", delta)
    ties = [r["tie_rate"] for r in panel if r["tie_rate"] is not None]
    summary = {
        "panel_rows": len(panel),
        "delta_rows": len(delta),
        "signals": sorted({r["signal"] for r in panel}),
        "discrete_signals": sorted(DISCRETE_PANEL_SIGNALS),
        "protocols": sorted({r["protocol"] for r in panel}),
        "rounds": sorted({r["round"] for r in panel}),
        "duplicate_rows_dropped": sum(r["dupes_dropped"] for r in panel
                                      if r["signal"] == SIGNAL_SELF),
        "collapsed_item_clustering_rows": sum(1 for r in panel if r["n_items"] <= 1),
        "median_tie_rate": sorted(ties)[len(ties) // 2] if ties else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
