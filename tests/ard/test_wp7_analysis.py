"""WP7 analysis: the three primaries under Holm, the registered secondary families, the
within-item AUC panel, and the predictions-before-labels seal.

Spec §15.1 (unit = the original task; within-item ranking averaged over mixed items with the
eligible count reported), §15.2 (three confirmatory contrasts, Holm across them); plan §8
(registered secondary families; held-out predictions written and hashed before the label join).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agents_scaling.ard import analysis as A
from agents_scaling.ard import contract as K
from agents_scaling.ard import stats as S
from agents_scaling.study.aggregate import _write_parquet

N_ITEMS_PER_FAMILY = 12
RESAMPLES = 300
SEED = "0" * 64


def _items() -> list[tuple[str, str]]:
    out = []
    for family in (K.Family.MATH500.value, K.Family.MMLUPRO.value):
        for i in range(N_ITEMS_PER_FAMILY):
            out.append((f"{family}-{i:04d}", family))
    return out


def _episode_id(protocol: str, item: str, checkpoint: str = K.CHECKPOINT_SMALL) -> str:
    return f"{protocol}.{checkpoint}.{item}"


def _build_tables(
    *,
    gate_correct,
    peer_correct,
    random_correct,
    peer_coverage_change=0.5,
    private_coverage_change=0.1,
    fresh_correct=None,
    checkpoint: str = K.CHECKPOINT_SMALL,
    cohort: str = K.Cohort.NATURAL.value,
) -> dict[str, list[dict]]:
    """The three tables the primaries read: episodes, round_selection, round_summary."""
    episodes, selection, summary = [], [], []
    arms = {
        K.Protocol.GATE.value: gate_correct,
        K.Protocol.PEER.value: peer_correct,
        K.Protocol.RANDOM_PROTECT.value: random_correct,
    }
    if fresh_correct is not None:
        arms[K.Protocol.FRESH.value] = fresh_correct
    for index, (item, family) in enumerate(_items()):
        for protocol, correct_of in arms.items():
            eid = _episode_id(protocol, item, checkpoint)
            episodes.append({
                "episode_id": eid, "item_id": item, "family": family, "split": "test",
                "cohort": cohort, "protocol": protocol, "checkpoint": checkpoint,
                "team_id": None, "reasoning_allowance": K.THINKING_TOKENS_LOW,
                "parent_ref": None, "fallback": False,
            })
            pool = "archive" if protocol == K.Protocol.FRESH.value else "current"
            for selector in ("plurality", "confidence_weighted", "CURRENT"):
                selection.append({
                    "episode_id": eid, "round": K.UPDATE_ROUNDS, "item_id": item,
                    "protocol": protocol, "checkpoint": checkpoint, "cohort": cohort,
                    "selector": selector, "pool": pool,
                    "selected_correct": bool(correct_of(index)),
                })
        for protocol, change in ((K.Protocol.PEER.value, peer_coverage_change),
                                 (K.Protocol.PRIVATE.value, private_coverage_change)):
            eid = _episode_id(protocol, item, checkpoint)
            episodes.append({
                "episode_id": eid, "item_id": item, "family": family, "split": "test",
                "cohort": cohort, "protocol": protocol, "checkpoint": checkpoint,
                "team_id": None, "reasoning_allowance": K.THINKING_TOKENS_LOW,
                "parent_ref": None, "fallback": False,
            })
            for round_, coverage in ((0, 0.4), (K.UPDATE_ROUNDS, 0.4 + change)):
                summary.append({
                    "episode_id": eid, "round": round_, "item_id": item, "family": family,
                    "split": "test", "cohort": cohort, "protocol": protocol,
                    "checkpoint": checkpoint, "coverage_current": coverage,
                    "constructed_C0": None,
                })
    return {"episodes": episodes, A.SELECTION_TABLE: selection, "round_summary": summary,
            "decisions": [], "labels": [], "calibration_panel": [], "suffix_summary": [],
            "neural_geometry": []}


def _separating_tables():
    """GATE beats both controls on half the items; PEER gains more coverage than PRIVATE."""
    return _build_tables(
        gate_correct=lambda i: True,
        peer_correct=lambda i: i % 2 != 0,
        random_correct=lambda i: i % 2 != 0,
        peer_coverage_change=0.5,
        private_coverage_change=0.1,
        fresh_correct=lambda i: i % 3 != 0,
    )


# --------------------------------------------------------------------------- primaries


def test_primary_analysis_runs_exactly_three_contrasts_under_holm():
    tables = _separating_tables()
    out = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    assert set(out["contrasts"]) == set(K.PRIMARY_CONTRASTS)
    assert out["n_primary"] == K.N_PRIMARY == 3
    assert set(out["holm"]) == set(K.PRIMARY_CONTRASTS)
    assert out["complete"] is True
    for name, res in out["contrasts"].items():
        assert res["estimate"] is not None, name
        assert res["n_items"] == 2 * N_ITEMS_PER_FAMILY
        assert set(res["n"]) == {f.value for f in K.FAMILIES}


def test_primary_estimates_have_the_expected_sign_and_the_family_axis():
    tables = _separating_tables()
    out = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    gate_peer = out["contrasts"][K.PRIMARY_CONTRASTS[0]]
    coverage = out["contrasts"][K.PRIMARY_CONTRASTS[2]]
    assert gate_peer["estimate"] == pytest.approx(0.5)
    assert coverage["estimate"] == pytest.approx(0.4)
    assert gate_peer["domains"] == list(S.DOMAINS) == ["math500", "mmlu_pro"]
    assert gate_peer["weights"] == {"math500": 0.5, "mmlu_pro": 0.5}
    assert out["holm"][K.PRIMARY_CONTRASTS[0]]["reject"] is True


def test_a_null_effect_is_not_rejected():
    tables = _build_tables(gate_correct=lambda i: i % 2 == 0,
                           peer_correct=lambda i: i % 2 == 0,
                           random_correct=lambda i: i % 2 == 0,
                           peer_coverage_change=0.2, private_coverage_change=0.2)
    out = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    for name in (K.PRIMARY_CONTRASTS[0], K.PRIMARY_CONTRASTS[2]):
        assert out["contrasts"][name]["estimate"] == pytest.approx(0.0)
        assert out["holm"][name]["reject"] is False


def test_an_absent_arm_is_a_loud_failure_by_default():
    """The vendoring exists so a missing arm cannot be read as 'no rejection' (contract §3)."""
    tables = _separating_tables()
    tables[A.SELECTION_TABLE] = [row for row in tables[A.SELECTION_TABLE]
                                 if row["protocol"] != K.Protocol.GATE.value]
    with pytest.raises(S.StatsError):
        A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    relaxed = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES,
                                 allow_unavailable=True)
    assert relaxed["complete"] is False
    assert relaxed["contrasts"][K.PRIMARY_CONTRASTS[0]]["status"] == "unavailable"
    assert relaxed["holm"][K.PRIMARY_CONTRASTS[0]]["reject"] is False


def test_an_item_outside_the_family_axis_raises():
    tables = _separating_tables()
    stray = tables["episodes"][0]["item_id"]
    for row in tables["episodes"]:
        if row["item_id"] == stray:
            row["family"] = "gsm8k"
    with pytest.raises(S.StatsAxisError):
        A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)


def test_one_item_carrying_two_families_raises():
    tables = _separating_tables()
    tables["episodes"][0]["family"] = "gsm8k"
    with pytest.raises(A.AnalysisError):
        A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)


def test_rows_are_averaged_to_the_item_before_resampling():
    """Two episodes of one item (challenge teams, budget arms) count as ONE cluster."""
    tables = _separating_tables()
    extra = []
    for row in list(tables[A.SELECTION_TABLE]):
        if row["protocol"] == K.Protocol.GATE.value and row["selector"] == "plurality":
            twin = dict(row, episode_id=row["episode_id"] + ".rep1", selected_correct=False)
            extra.append(twin)
            tables["episodes"].append(
                dict(next(e for e in tables["episodes"]
                          if e["episode_id"] == row["episode_id"]),
                     episode_id=twin["episode_id"]))
    tables[A.SELECTION_TABLE].extend(extra)
    out = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    res = out["contrasts"][K.PRIMARY_CONTRASTS[0]]
    assert res["n_items"] == 2 * N_ITEMS_PER_FAMILY     # not doubled by the second episode
    # GATE's per-item mean is now 0.5 (one correct episode, one wrong), so the contrast
    # against PEER's per-item 0.5 collapses to zero.  Summing episodes instead of averaging
    # to the item would report 24 extra clusters and a different estimate.
    assert res["estimate"] == pytest.approx(0.0)


def test_coverage_change_is_formed_inside_the_episode():
    tables = _build_tables(gate_correct=lambda i: True, peer_correct=lambda i: True,
                           random_correct=lambda i: True,
                           peer_coverage_change=0.3, private_coverage_change=0.3)
    # Drop PEER's root row for one episode: that episode contributes no change at all.
    dropped = tables["round_summary"][0]["episode_id"]
    tables["round_summary"] = [r for r in tables["round_summary"]
                               if not (r["episode_id"] == dropped and r["round"] == 0)]
    out = A.primary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES,
                             allow_unavailable=False)
    res = out["contrasts"][K.PRIMARY_CONTRASTS[2]]
    assert res["n_items"] == 2 * N_ITEMS_PER_FAMILY - 1


# --------------------------------------------------------------------------- secondary


def test_every_registered_secondary_family_is_present_and_labelled():
    tables = _separating_tables()
    out = A.secondary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    assert set(out) == {
        "challenge_c0", "receiver", "checkpoint", "suffix_reversal", "calibration",
        "neural_increment", "decision_matched_independent_sampling",
    }
    for name, family in out.items():
        assert family["label"] in ("secondary", "exploratory"), name


def test_decision_matched_family_contrasts_gate_and_peer_against_fresh20():
    tables = _separating_tables()
    out = A.secondary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    family = out["decision_matched_independent_sampling"]
    assert set(family["contrasts"]) == {
        f"{p}_minus_FRESH20_{s}"
        for p in ("GATE", "PEER")
        for s in ("plurality", "confidence_weighted", "CURRENT")
    }
    gate = family["contrasts"]["GATE_minus_FRESH20_plurality"]
    assert gate["estimate"] is not None
    assert "NOT FLOP-matched" in family["note"]
    # Holm is applied only inside this family.
    assert set(family["holm"]) == set(family["contrasts"])


def test_unexecuted_families_are_registered_not_dropped():
    tables = _separating_tables()
    out = A.secondary_analysis(tables, study_seed=SEED, n_resamples=RESAMPLES)
    for name in ("receiver", "suffix_reversal", "neural_increment"):
        assert out[name]["status"] == "unexecuted"
        assert "amendment register" in out[name]["note"]


def test_challenge_family_flags_estimation_only_below_thirty_clusters():
    tables = _build_tables(gate_correct=lambda i: i % 2 == 0,
                           peer_correct=lambda i: False,
                           random_correct=lambda i: False,
                           cohort=K.Cohort.CHALLENGE.value)
    for row in tables["round_summary"]:
        row["constructed_C0"] = 4
    for row in tables[A.SELECTION_TABLE]:
        row["cohort"] = K.Cohort.CHALLENGE.value
    # give the challenge episodes their C0 through round_summary rows
    tables["round_summary"].extend([
        {"episode_id": row["episode_id"], "round": K.UPDATE_ROUNDS, "item_id": row["item_id"],
         "family": None, "split": "test", "cohort": K.Cohort.CHALLENGE.value,
         "protocol": row["protocol"], "checkpoint": row["checkpoint"],
         "coverage_current": 0.5, "constructed_C0": 4}
        for row in tables[A.SELECTION_TABLE] if row["selector"] == "plurality"
    ])
    out = A.challenge_family(tables, A.episode_index(tables), study_seed=SEED,
                             n_resamples=RESAMPLES)
    res = out["contrasts"]["challenge_C04_GATE_minus_PEER"]
    assert res["estimate"] == pytest.approx(0.5)
    assert res["estimation_only"] is True
    assert "estimation only" in res["note"]


# --------------------------------------------------------------------------- within-item AUC


def _auc_tables(labels_by_agent, confidences):
    episodes = [{"episode_id": "E1", "item_id": "math500-0000", "family": "math500",
                 "split": "test", "cohort": K.Cohort.NATURAL.value,
                 "protocol": K.Protocol.PEER.value, "checkpoint": K.CHECKPOINT_SMALL,
                 "fallback": False}]
    decisions, labels = [], []
    for agent, (correct, conf) in enumerate(zip(labels_by_agent, confidences)):
        did = f"D{agent}"
        decisions.append({"decision_id": did, "episode_id": "E1", "agent_id": agent,
                          "round": K.UPDATE_ROUNDS, "protocol": K.Protocol.PEER.value,
                          "checkpoint": K.CHECKPOINT_SMALL,
                          "cohort": K.Cohort.NATURAL.value, "confidence": conf})
        labels.append({"decision_id": did, "is_correct": correct})
    return {"episodes": episodes, "decisions": decisions, "labels": labels,
            "round_summary": [], A.SELECTION_TABLE: []}


def test_within_item_auc_on_a_mixed_item():
    tables = _auc_tables([True, False, True, False, False], [0.9, 0.2, 0.8, 0.3, 0.1])
    rows = A.within_item_rows(tables)
    assert len(rows) == 1
    assert rows[0]["within_item_auc"] == pytest.approx(1.0)
    assert rows[0]["eligible_items"] == 1
    assert rows[0]["excluded_single_class"] == 0


def test_single_class_item_returns_none_and_increments_the_excluded_counter():
    """An all-correct item has no AUC; it must never be folded into a pooled number."""
    tables = _auc_tables([True] * 5, [0.9, 0.8, 0.7, 0.6, 0.5])
    rows = A.within_item_rows(tables)
    assert rows[0]["within_item_auc"] is None
    assert rows[0]["eligible_items"] == 0
    assert rows[0]["excluded_single_class"] == 1
    assert rows[0]["n_items"] == 1
    # the panel's AUC denominator can never equal its row count
    assert rows[0]["eligible_items"] != rows[0]["n_items"]


def test_missing_confidence_is_excluded_never_imputed_as_half():
    """Plan §5.7: the atlas probe script's ``conf.fillna(0.5)`` is not lifted."""
    tables = _auc_tables([True, False, True, False, False],
                         [0.9, 0.2, None, 0.3, None])
    rows = A.within_item_rows(tables)
    assert rows[0]["n_missing_score"] == 2
    assert rows[0]["n_candidates"] == 3
    assert rows[0]["within_item_auc"] == pytest.approx(1.0)
    imputed = A.within_item_rows(_auc_tables([True, False, True, False, False],
                                             [0.9, 0.2, 0.5, 0.3, 0.5]))
    assert imputed[0]["within_item_auc"] != rows[0]["within_item_auc"]


def test_unlabelled_decisions_do_not_enter_the_panel():
    tables = _auc_tables([True, False, None, False, False], [0.9, 0.2, 0.8, 0.3, 0.1])
    tables["labels"] = [row for row in tables["labels"] if row["is_correct"] is not None]
    rows = A.within_item_rows(tables)
    assert rows[0]["n_candidates"] == 4


# --------------------------------------------------------------------------- seal


def test_predictions_are_written_and_hashed_before_labels_join(tmp_path):
    rows = [{"decision_id": f"D{i}", "p_peer": 0.1 * i, "p_private": 0.2} for i in range(4)]
    entry = A.write_held_out_predictions(tmp_path, "controller_heldout", rows)
    assert entry["n_rows"] == 4
    assert len(entry["sha256"]) == 64
    manifest = A.predictions_manifest(tmp_path)
    assert manifest["controller_heldout"]["sha256"] == entry["sha256"]
    labels = {f"D{i}": {"is_correct": i % 2 == 0} for i in range(4)}
    joined = A.join_labels(tmp_path, "controller_heldout", labels)
    assert [row["is_correct"] for row in joined] == [True, False, True, False]
    assert "is_correct" not in entry["columns"]


def test_joining_labels_without_a_seal_is_refused(tmp_path):
    with pytest.raises(A.LabelJoinRefused):
        A.join_labels(tmp_path, "never_sealed", {})


def test_predictions_carrying_a_label_column_are_refused(tmp_path):
    with pytest.raises(A.LabelJoinRefused) as excinfo:
        A.write_held_out_predictions(tmp_path, "leaky",
                                     [{"decision_id": "D0", "is_correct": True}])
    assert "is_correct" in str(excinfo.value)


def test_predictions_rewritten_after_sealing_break_the_join(tmp_path):
    rows = [{"decision_id": "D0", "p": 0.5}]
    entry = A.write_held_out_predictions(tmp_path, "heldout", rows)
    path = A.predictions_dir(tmp_path) / entry["file"]
    _write_parquet(path, [{"decision_id": "D0", "p": 0.99}])
    with pytest.raises(A.LabelJoinRefused) as excinfo:
        A.join_labels(tmp_path, "heldout", {"D0": {"is_correct": True}})
    assert "changed after they were sealed" in str(excinfo.value)


# --------------------------------------------------------------------------- driver


def test_run_analysis_writes_the_stats_artifacts(tmp_path):
    tables = _separating_tables()
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True)
    for name, rows in tables.items():
        _write_parquet(tables_dir / f"{name}.parquet", rows)
    result = A.run_analysis(tmp_path, study_seed=SEED, n_resamples=RESAMPLES)
    assert (tables_dir / A.STATS_FILE).exists()
    assert (tables_dir / A.STATS_MARKDOWN).exists()
    on_disk = json.loads((tables_dir / A.STATS_FILE).read_text())
    assert set(on_disk["primary"]["contrasts"]) == set(K.PRIMARY_CONTRASTS)
    assert on_disk["nonclaims"] == list(A.NONCLAIMS)
    assert "no claim of superiority over optimised single-agent inference" in \
        " ".join(on_disk["nonclaims"]).lower()
    assert "Holm over exactly three contrasts" in (tables_dir / A.STATS_MARKDOWN).read_text()


def test_calibration_metrics_never_impute_missing_confidence():
    out = A.calibration_metrics([0.9, None, 0.2, 0.8], [1, 1, 0, 1])
    assert out["valid_n"] == 3
    assert out["n"] == 4
    assert out["conf_missing_rate"] == pytest.approx(0.25)
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["brier"] == pytest.approx(np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.8 - 1) ** 2]))
    empty = A.calibration_metrics([None, None], [1, 0])
    assert empty["valid_n"] == 0 and empty["ece_primary"] is None
