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
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


def within_item_auc(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Mean within-item ranking AUC, and the count of items eligible to contribute.

    Plan §5.8: an item whose candidates are all correct or all wrong carries no ranking
    information and returns ``None`` rather than 0.5, so a panel row's AUC denominator can
    never equal its row count.  Pooled AUC is a different quantity and is not computed here.
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
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def _panel_row(rows: Sequence[Mapping[str, Any]], *, key: Mapping[str, Any], signal: str,
               n_total: int, n_missing: int) -> dict[str, Any]:
    calib = _calib()
    conf = [float(r["conf"]) for r in rows]
    correct = [int(r["correct"]) for r in rows]
    summary = calib.cell_calibration(conf, correct, signal) if conf else {}
    auc, eligible = within_item_auc(rows)
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
        "within_item_auc": auc,
        "eligible_items": eligible,
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
        lambda: {SIGNAL_SELF: [], SIGNAL_LOGPROB: [], "n": 0, "missing": 0})
    # Team rows need the whole round assembled first.
    per_round: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for row in decisions:
        label = labels.get(str(row.get("decision_id")))
        if label is None or label.get("is_correct") is None:
            continue                      # ungraded / ambiguous: unobserved, never 0
        if str(label.get("evaluator_status")) not in ("correct", "incorrect"):
            continue
        correct = int(bool(label.get("is_correct")))
        key = (str(row.get("protocol")), str(row.get("checkpoint")),
               str(row.get("cohort") or "natural"), int(row.get("round") or 0))
        bucket = agent_rows[key]
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
        for signal in AGENT_SIGNALS:
            rows = bucket[signal]
            missing = bucket["missing"] if signal == SIGNAL_SELF else bucket["n"] - len(rows)
            panel.append(_panel_row(rows, key=base, signal=signal,
                                    n_total=bucket["n"], n_missing=missing))

    # Team-level signals: one row per episode-round, scored against the plurality's correctness.
    system: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {s: [] for s in SYSTEM_SIGNALS} | {"n": 0, "missing": 0})
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
        for signal in SYSTEM_SIGNALS:
            rows = bucket[signal]
            panel.append(_panel_row(rows, key=base, signal=signal, n_total=bucket["n"],
                                    n_missing=bucket["n"] - len(rows)))

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
            })
    return panel, delta


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

    _write_parquet(out / "calibration_panel.parquet", panel)
    _write_parquet(out / "calibration_system_minus_agent.parquet", delta)
    summary = {
        "panel_rows": len(panel),
        "delta_rows": len(delta),
        "signals": sorted({r["signal"] for r in panel}),
        "protocols": sorted({r["protocol"] for r in panel}),
        "rounds": sorted({r["round"] for r in panel}),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
