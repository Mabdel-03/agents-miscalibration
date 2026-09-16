"""Calibration measurement rules that the spec names as tests (spec 18.3, 13.1).

Two of ARD36's repairs live here as executable rules rather than as prose in a defect log:

* **Within-item AUC has a denominator.**  ARD36's v1 panel clustered on an ``item_id`` that
  did not exist in the table it read, so every row fell into one cluster and the pooled AUC
  was published under the within-item name.  The repair (``ard.calibration_panel``) returns
  ``None`` for an item whose candidates are all correct or all wrong -- such an item carries
  no ranking information -- and reports the cluster count behind the number, so a collapsed
  clustering is visible instead of silent.  Confidence AUC fell 0.7555 -> 0.5723 when that
  was fixed, which is the difference between "confidence ranks answers" and "it does not".

* **A missing confidence is missing.**  Never 0.5, never the base rate, never a model
  statement.  ARD36's cap-hit draws lost their confidence tail, and the missingness is not
  random (it lands on the longer, harder items), so every panel row reports its own
  missing rate beside the estimate.
"""

from __future__ import annotations

import math

import pytest

from agents_scaling.nextdyn import calib


def _rows(pairs, *, item):
    return [{"item_id": item, "conf": conf, "correct": correct} for conf, correct in pairs]


def test_within_item_auc_requires_mixed_candidate_groups():
    """Spec 18.3: an item with no correct/wrong pair contributes no ranking information."""
    homogeneous = _rows([(0.9, 1.0), (0.8, 1.0), (0.7, 1.0)], item="all-correct")
    homogeneous += _rows([(0.6, 0.0), (0.5, 0.0)], item="all-wrong")
    blank = calib.within_item_auc(homogeneous)
    assert blank["within_item_auc"] is None, "a homogeneous item must not score 0.5"
    assert blank["eligible_items"] == 0 and blank["n_items"] == 2

    # One mixed item: confidence ranks the correct candidate above the wrong one -> AUC 1.
    mixed = _rows([(0.9, 1.0), (0.1, 0.0)], item="mixed")
    scored = calib.within_item_auc(homogeneous + mixed)
    assert scored["within_item_auc"] == pytest.approx(1.0)
    # The denominator is the MIXED items, never the row count: 1 of 3 items, 7 rows.
    assert scored["eligible_items"] == 1 and scored["n_items"] == 3
    assert scored["eligible_items"] < len(homogeneous + mixed)

    # A perfectly inverted ranking is 0.0, and a tie is half credit -- so the statistic can
    # distinguish "ranks badly" from "carries no information", which 0.5-for-everything cannot.
    assert calib.within_item_auc(_rows([(0.1, 1.0), (0.9, 0.0)], item="i"))["within_item_auc"] == pytest.approx(0.0)
    assert calib.within_item_auc(_rows([(0.5, 1.0), (0.5, 0.0)], item="i"))["within_item_auc"] == pytest.approx(0.5)

    # Clustering must be real, and the two quantities genuinely differ.  This is the shape
    # study_v4 measured: confidence tracks how hard the ITEM is, so it ranks candidates well
    # inside an item and badly across items.  Within-item is 1.0 on both items here; pooled is
    # 0.75, because the hard item's correct answer (0.2) sits below the easy item's wrong one
    # (0.5).  Publishing the pooled number under the within-item name is ARD36's v1 defect.
    by_difficulty = (_rows([(0.6, 1.0), (0.5, 0.0)], item="easy")
                     + _rows([(0.2, 1.0), (0.1, 0.0)], item="hard"))
    per_item = calib.within_item_auc(by_difficulty)
    pooled = calib.within_item_auc([dict(r, item_id="one") for r in by_difficulty])
    assert per_item["n_items"] == 2 and per_item["within_item_auc"] == pytest.approx(1.0)
    assert pooled["n_items"] == 1 and pooled["within_item_auc"] == pytest.approx(0.75)


def test_confidence_not_imputed_as_model_statement():
    """Spec 18.3 / 13.1: a missing confidence stays missing and is reported, never filled in."""
    rows = [
        {"confidence": 0.9, "correct": 1.0},
        {"confidence": None, "correct": 0.0},   # cap-hit draw: the tail was cut
        {"confidence": 0.2, "correct": 0.0},
        {"confidence": None, "correct": 1.0},
    ]
    signal = calib.confidence_signal(rows, signal="confidence")
    assert signal.n == 2, "only the rows that actually carry a confidence are scored"
    assert signal.missing_rate == pytest.approx(0.5)
    assert all(value is not None and not math.isnan(value) for value in signal.conf)
    assert 0.5 not in set(signal.conf), "0.5 is not a value the model stated"

    panel = calib.cell_calibration(rows, signal="confidence")
    assert panel["n"] == 2 and panel["n_missing_conf"] == 2 and panel["n_total"] == 4
    assert panel["conf_missing_rate"] == pytest.approx(0.5)
    # Imputing the two missing values as 0.5 would change the estimate, which is exactly why
    # the panel must not do it: the two panels below are different numbers.
    imputed = calib.cell_calibration(
        [dict(r, confidence=0.5 if r["confidence"] is None else r["confidence"]) for r in rows],
        signal="confidence")
    assert imputed["n"] == 4 and imputed["brier"] != panel["brier"]

    # A signal that is missing everywhere is undefined, not perfectly calibrated.
    empty = calib.cell_calibration([{"confidence": None, "correct": 1.0}], signal="confidence")
    assert empty["n"] == 0 and empty["brier"] is None and empty["conf_missing_rate"] == pytest.approx(1.0)
