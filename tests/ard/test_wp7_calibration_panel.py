"""WP7 miscalibration panel: the three defects that made the frozen v1 file unreadable.

Two layers, following ``test_panels.py``.  The synthetic layer builds the smallest tables that
exhibit each defect -- a cell whose confidence sits on a handful of atoms, one ``decision_id``
aliased twice into a single cell, and an item whose pooled ranking disagrees with its
within-item ranking -- and asserts the repaired panel notices.  The frozen layer runs against
the repaired ``ard36/tables_v2`` and pins the numbers this repair measured; it skips when that
run root is not mounted, so the file is still useful off the cluster.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

from agents_scaling.ard import calibration_panel as CP
from agents_scaling.study.aggregate import _write_parquet

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from nb_lib import calib  # noqa: E402

V2_TABLES = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36/tables_v2")

FROZEN = pytest.mark.skipif(not (V2_TABLES / "calibration_panel.parquet").exists(),
                            reason="repaired ard36 tables_v2 not mounted")


def _decision(decision_id: str, *, item_id: str, confidence: float | None,
              episode_id: str = "ep0", agent_id: str = "a0", round_: int = 0,
              answer_class: str = "expr:1", answer_logprob: float | None = -0.1,
              protocol: str = "PEER") -> dict[str, Any]:
    return {"decision_id": decision_id, "item_id": item_id, "confidence": confidence,
            "episode_id": episode_id, "agent_id": agent_id, "round": round_,
            "answer_class": answer_class, "answer_logprob": answer_logprob,
            "protocol": protocol, "checkpoint": "8B", "cohort": "natural"}


def _label(decision_id: str, correct: bool) -> dict[str, Any]:
    return {"decision_id": decision_id, "evaluator_status": "correct" if correct else "incorrect",
            "is_correct": correct}


def _build(tmp_path: Path, decisions: list[dict], labels: list[dict]):
    _write_parquet(tmp_path / "decisions.parquet", decisions)
    _write_parquet(tmp_path / "labels.parquet", labels)
    return CP.build_panel(tmp_path)


def _row(panel, signal: str):
    return next(r for r in panel if r["signal"] == signal)


# --------------------------------------------------------------- (c) discrete dispatch


def test_self_reported_is_scored_with_the_exact_atom_estimator(tmp_path):
    """The panel's two discrete signals must not reach the equal-mass binned path.

    The fixture is twelve atoms that alternate all-correct / all-wrong, so equal-mass bins
    straddle atoms with opposite-signed gaps and cancel part of the miscalibration away --
    the failure mode the exact-atom estimator exists to avoid.
    """
    conf = [0.40 + 0.05 * (i // 3) for i in range(36)]
    correct = [1 if (i // 3) % 2 == 0 else 0 for i in range(36)]
    decisions, labels = [], []
    for i, (c, y) in enumerate(zip(conf, correct)):
        decisions.append(_decision(f"d{i}", item_id=f"q{i // 3}", confidence=c,
                                   episode_id=f"ep{i // 3}", answer_logprob=None))
        labels.append(_label(f"d{i}", correct=bool(y)))
    panel, _ = _build(tmp_path, decisions, labels)

    self_row = _row(panel, CP.SIGNAL_SELF)
    assert self_row["signal_type"] == calib.DISCRETE
    assert self_row["ece_primary"] == pytest.approx(calib.ece_discrete(conf, correct))
    assert self_row["ece_primary"] == pytest.approx(0.525)
    # ...and it is genuinely a different number from the estimator that used to run, which
    # understates the gap by binning atoms together.
    assert calib.ece_equal_mass(conf, correct) == pytest.approx(0.441667, abs=1e-5)
    assert self_row["ece_primary"] > calib.ece_equal_mass(conf, correct)


def test_debiased_l2_is_nan_on_every_discrete_row_and_a_number_on_every_other(tmp_path):
    decisions, labels = [], []
    for i in range(40):
        decisions.append(_decision(f"d{i}", item_id=f"q{i // 4}", confidence=0.9 + 0.01 * (i % 5),
                                   episode_id=f"ep{i // 4}", answer_logprob=-0.01 * (i + 1),
                                   answer_class="expr:1" if i % 4 else "expr:2"))
        labels.append(_label(f"d{i}", correct=i % 3 != 0))
    panel, _ = _build(tmp_path, decisions, labels)

    for row in panel:
        value = row["ece_debiased_l2"]
        is_nan = value is None or (isinstance(value, float) and math.isnan(value))
        if row["signal"] in CP.DISCRETE_PANEL_SIGNALS:
            assert is_nan, f"{row['signal']} kept a debiased-L2 value"
        elif row["valid_n"]:
            assert not is_nan, f"{row['signal']} lost its debiased-L2 value"
    CP.check_discrete_invariant(panel)          # the shipped guard agrees


def test_the_invariant_guard_fails_loudly_when_the_dispatch_falls_through():
    """A regression that reverts the dispatch must not publish a plausible-looking column."""
    bad = [{"signal": CP.SIGNAL_SELF, "ece_debiased_l2": 0.12, "valid_n": 10,
            "protocol": "PEER", "checkpoint": "8B", "cohort": "natural", "round": 0}]
    with pytest.raises(SystemExit, match="did not fire"):
        CP.check_discrete_invariant(bad)


def test_the_shared_registry_keeps_meaning_discrete_by_construction():
    """``self_reported`` is discrete in this run, not by definition; the override says so."""
    assert "system_vote_fraction" in calib.DISCRETE_SIGNALS   # k/n: finite by construction
    assert "self_reported" not in calib.DISCRETE_SIGNALS      # a verbal forecast is not
    assert calib.is_discrete("self_reported") is False
    assert calib.is_discrete("self_reported", calib.DISCRETE) is True
    assert calib.is_discrete("vote_fraction", calib.CONTINUOUS) is False
    with pytest.raises(ValueError):
        calib.is_discrete("self_reported", "ordinal")


# --------------------------------------------------------------- (b) round-0 deduplication


def test_a_decision_aliased_twice_into_one_cell_is_counted_once(tmp_path):
    decisions, labels = [], []
    for i in range(12):
        decisions.append(_decision(f"d{i}", item_id=f"q{i}", confidence=0.9,
                                   episode_id=f"ep{i}"))
        labels.append(_label(f"d{i}", correct=i % 2 == 0))
    # The D-11 shape: the same decision_id a second time in the same cell, payload disagreeing.
    decisions.append(_decision("d0", item_id="q0", confidence=0.5, episode_id="ep0"))
    panel, _ = _build(tmp_path, decisions, labels)

    self_row = _row(panel, CP.SIGNAL_SELF)
    assert self_row["valid_n"] == 12                 # not 13
    assert self_row["dupes_dropped"] == 1
    assert self_row["mean_confidence"] == pytest.approx(0.9)   # first occurrence won


def test_the_same_decision_in_two_arms_is_not_a_duplicate(tmp_path):
    """Round-0 aliasing across arms is real, but the arms are different cells."""
    decisions, labels = [], []
    for i in range(8):
        labels.append(_label(f"d{i}", correct=True))
        for protocol in ("PEER", "PRIVATE"):
            decisions.append(_decision(f"d{i}", item_id=f"q{i}", confidence=0.9,
                                       episode_id=f"ep{i}", protocol=protocol))
    panel, _ = _build(tmp_path, decisions, labels)

    for protocol in ("PEER", "PRIVATE"):
        row = next(r for r in panel
                   if r["signal"] == CP.SIGNAL_SELF and r["protocol"] == protocol)
        assert row["valid_n"] == 8 and row["dupes_dropped"] == 0


# --------------------------------------------------------------- (a) item clustering


def test_within_item_auc_clusters_on_item_and_is_not_the_pooled_value(tmp_path):
    """Two items whose easy/hard offset reverses the pooled ranking."""
    rows = [("q0", 0.9, 1), ("q0", 0.8, 0), ("q1", 0.7, 1), ("q1", 0.6, 0)]
    decisions, labels = [], []
    for i, (item, conf, correct) in enumerate(rows * 3):
        decisions.append(_decision(f"d{i}", item_id=item, confidence=conf,
                                   episode_id=f"ep{i}", answer_logprob=None))
        labels.append(_label(f"d{i}", correct=bool(correct)))
    panel, _ = _build(tmp_path, decisions, labels)

    row = _row(panel, CP.SIGNAL_SELF)
    assert row["n_items"] == 2                       # the clustering ran
    assert row["eligible_items"] == 2                # both items are mixed
    assert row["within_item_auc"] == pytest.approx(1.0)
    # Pooled, q1's correct answers rank below q0's wrong ones, so the pooled AUC is lower.
    assert calib.auroc([r[1] for r in rows * 3], [r[2] for r in rows * 3]) < 1.0


def test_a_missing_item_id_collapses_to_one_cluster_and_says_so(tmp_path):
    """The v1 defect: no item column means one pseudo-cluster, and n_items reports it."""
    decisions, labels = [], []
    for i in range(10):
        decisions.append(_decision(f"d{i}", item_id=None, confidence=0.5 + 0.04 * i,
                                   episode_id=f"ep{i}", answer_logprob=None))
        labels.append(_label(f"d{i}", correct=i % 2 == 0))
    panel, _ = _build(tmp_path, decisions, labels)

    row = _row(panel, CP.SIGNAL_SELF)
    assert row["n_items"] == 1
    assert row["within_item_auc"] is not None        # it is the pooled number...
    assert row["eligible_items"] == 1                # ...and this is the tell.


def test_a_homogeneous_item_is_excluded_from_the_auc_denominator(tmp_path):
    decisions, labels = [], []
    for i in range(6):
        decisions.append(_decision(f"d{i}", item_id=f"q{i // 3}", confidence=0.5 + 0.05 * i,
                                   episode_id=f"ep{i}", answer_logprob=None))
        labels.append(_label(f"d{i}", correct=True))      # every candidate correct
    panel, _ = _build(tmp_path, decisions, labels)

    row = _row(panel, CP.SIGNAL_SELF)
    assert row["n_items"] == 2 and row["eligible_items"] == 0
    assert row["within_item_auc"] is None                 # never silently 0.5


# --------------------------------------------------------------- tie rate


def test_tie_rate_is_the_probability_two_draws_share_an_atom():
    # Four rows on two atoms: 2 tied pairs of 6 unordered pairs.
    out = CP.tie_rate([0.9, 0.9, 0.8, 0.8])
    assert out["tie_rate"] == pytest.approx(2 / 6)
    assert out["n_atoms"] == 2 and out["modal_atom_share"] == pytest.approx(0.5)

    assert CP.tie_rate([0.1, 0.2, 0.3])["tie_rate"] == pytest.approx(0.0)
    assert CP.tie_rate([0.5, 0.5])["tie_rate"] == pytest.approx(1.0)
    assert CP.tie_rate([0.5])["tie_rate"] is None          # undefined on one row, not 0
    assert CP.tie_rate([])["n_atoms"] == 0


# --------------------------------------------------------------- frozen layer


@FROZEN
def test_repaired_panel_pins_the_measured_repair():
    import pyarrow.parquet as pq

    panel = pq.read_table(V2_TABLES / "calibration_panel.parquet").to_pylist()
    assert len(panel) == 300

    # (c) every discrete row lost its debiased-L2 value; every continuous one kept it.
    CP.check_discrete_invariant(panel)

    # (b) the only within-cell duplication on this run is the D-11 bank lane: 212 rows,
    # all in NONE/8B/natural round 0.  The protocol arms have none.
    dupes = {(r["protocol"], r["checkpoint"], r["cohort"], r["round"]): r["dupes_dropped"]
             for r in panel if r["signal"] == CP.SIGNAL_SELF}
    assert dupes[("NONE", "8B", "natural", 0)] == 212
    assert sum(dupes.values()) == 212

    # (a) the clustering is live: the median cell holds >100 items, not one.
    items = sorted(r["n_items"] for r in panel)
    assert items[len(items) // 2] > 100
    # Only the two genuinely empty cells collapse.
    assert [r["valid_n"] for r in panel if r["n_items"] <= 1] == [0, 0]

    # A team row carries one candidate per item, so within-item ranking is undefined there.
    assert all(r["eligible_items"] == 0
               for r in panel if r["signal"] == CP.SIGNAL_VOTE_FRACTION)


@FROZEN
def test_repaired_panel_reports_a_tie_rate_beside_every_ece():
    import pyarrow.parquet as pq

    panel = pq.read_table(V2_TABLES / "calibration_panel.parquet").to_pylist()
    for row in panel:
        if row["valid_n"] > 1:
            assert row["tie_rate"] is not None and row["n_atoms"] >= 1

    # Confidence is degenerate and uniformly overconfident, so the L1 ECE carries essentially
    # nothing beyond mean_confidence - accuracy.  Pinning this keeps a later reader from
    # reading a third-decimal ECE gap as a calibration finding.
    selfr = [r for r in panel if r["signal"] == CP.SIGNAL_SELF]
    assert all(r["signed_gap"] > 0 for r in selfr)
    assert max(abs(r["ece_primary"] - abs(r["signed_gap"])) for r in selfr) < 0.01
