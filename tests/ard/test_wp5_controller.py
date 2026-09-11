"""WP5 controller: the fork frame's refusals, the freeze hash, and the argmax-u policy.

Spec §12.1 ("No branch, round, candidate, alternative option permutation, or challenge team
from a test item can enter training or calibration"; two L2-logistic heads whose difference
estimates ``u``; a validation-only ridge comparator), §12.2 (CURRENT and its baselines),
§12.3 (``protected = argmax_j u[j]``, "tie broken by frozen random schedule", no test-tuned
threshold); plan §5.4 and §10 (``controller:<sha>`` is a hard gate, so the artifact must be
tamper-evident).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import controller as CT
from agents_scaling.ard import features as FE
from agents_scaling.ard import protocols as P
from agents_scaling.ard import runner as ard_runner
from agents_scaling.study import identity
from tests.ard import wp5_support as S

SEED = bytes.fromhex("5c" * 32)


@pytest.fixture(scope="module")
def frames():
    return (S.paired_frame(12, split="train"), S.paired_frame(6, split="validation"),
            S.current_frame(12, split="train"), S.current_frame(6, split="validation"))


@pytest.fixture(scope="module")
def controller(frames):
    train, val, current_train, current_val = frames
    return CT.fit_controller(train, val, study_seed=SEED, current_train=current_train,
                             current_val=current_val)


# --------------------------------------------------------------------------- the frame


def test_the_fork_frame_refuses_a_test_item(frames):
    train = frames[0]
    leaked = dataclasses.replace(train.rows[0], split="test")
    with pytest.raises(CT.ControllerError, match="spec §12.1"):
        CT.PairedFrame((leaked,) + train.rows[1:]).validate()


def test_the_fork_frame_refuses_the_outcome_columns_as_features(frames):
    train = frames[0]
    for label in ("y_private", "y_peer"):
        features = {**dict(train.rows[0].features), label: 1.0}
        row = dataclasses.replace(train.rows[0], features=features)
        with pytest.raises(CT.ControllerError, match="the label is not a feature"):
            CT.PairedFrame((row,)).validate()


def test_the_current_frame_refuses_its_own_outcome_column(frames):
    current = frames[2]
    features = {**dict(current.rows[0].features), "y_correct": 1.0}
    row = dataclasses.replace(current.rows[0], features=features)
    with pytest.raises(CT.ControllerError, match="the label is not a feature"):
        CT.CurrentFrame((row,)).validate()


def test_the_frame_refuses_any_forbidden_feature_source(frames):
    features = {**dict(frames[0].rows[0].features), "root_pass_rate": 0.5}
    row = dataclasses.replace(frames[0].rows[0], features=features)
    with pytest.raises(FE.FeatureFirewallError):
        CT.PairedFrame((row,)).validate()


def test_a_row_missing_a_frozen_column_is_refused(frames):
    features = dict(frames[0].rows[0].features)
    features.pop("count_margin")
    row = dataclasses.replace(frames[0].rows[0], features=features)
    with pytest.raises(CT.ControllerError, match="lacks the columns"):
        CT.PairedFrame((row,)).validate()


def test_the_paired_difference_is_the_minus_one_zero_one_contrast(frames):
    for row in frames[0].rows:
        assert row.paired
        assert row.difference == int(row.y_private) - int(row.y_peer)
        assert row.difference in (-1, 0, 1)


# --------------------------------------------------------------------------- the fit


def test_the_fit_produces_both_heads_current_and_the_five_baselines(controller):
    assert CT.HEAD_PRIVATE in controller.models and CT.HEAD_PEER in controller.models
    assert CT.HEAD_CURRENT in controller.models and CT.HEAD_DYNAMICS in controller.models
    for baseline in (CT.BASELINE_FREQUENCY, CT.BASELINE_ANSWER_LOGPROB,
                     CT.BASELINE_FREQUENCY_CONFIDENCE):
        assert baseline in controller.models
    assert controller.calibration is not None      # the recalibrated-confidence baseline


def test_the_penalty_comes_only_from_the_frozen_grid(controller):
    """Spec §12.1: "do not search a large controller family"."""
    for name, selection in controller.selections.items():
        assert float(selection["penalty"]) in set(K.PENALTY_GRID), name


def test_the_ridge_comparator_is_fitted_and_never_used_by_the_policy(controller):
    assert controller.comparator["status"] == "fitted"
    assert controller.comparator["used_by_policy"] is False
    assert controller.meta["model_choice"] == "difference_of_two_logistic_heads"


def test_u_is_exactly_private_minus_peer(controller, frames):
    rows = [dict(r.features) for r in frames[1].rows]
    expected = controller.predict(CT.HEAD_PRIVATE, rows) - controller.predict(CT.HEAD_PEER, rows)
    assert np.allclose(controller.u(rows), expected)


def test_the_dynamics_head_reads_only_the_dynamics_columns(controller):
    assert controller.names_for(CT.HEAD_DYNAMICS) == tuple(FE.DYNAMICS_ONLY_FEATURES)
    assert controller.names_for(CT.HEAD_CURRENT) == tuple(controller.feature_names)


def test_the_raw_confidence_baseline_uses_the_prior_when_confidence_is_missing(controller):
    present = {**S.feature_row(), "own_confidence": 0.8, "own_confidence_missing": 0.0}
    missing = {**S.feature_row(), "own_confidence": 0.0, "own_confidence_missing": 1.0}
    values = controller.predict(CT.BASELINE_CONFIDENCE_RAW, [present, missing])
    assert values[0] == pytest.approx(0.8)
    assert values[1] == pytest.approx(controller.prior)     # never a hard-coded 0.5
    assert controller.prior != 0.8


# --------------------------------------------------------------------------- the freeze


def test_the_freeze_path_agrees_with_the_dispatcher(tmp_path, controller):
    """WP4 resolves ``controller:<sha>`` at this path; a disagreement would be a silent gate."""
    assert CT.CONTROLLER_DIR == ard_runner.CONTROLLER_DIR
    assert CT.CONTROLLER_FILE == ard_runner.CONTROLLER_FILE
    sha = CT.freeze_controller(tmp_path, controller)
    assert (tmp_path / CT.CONTROLLER_DIR / sha / CT.CONTROLLER_FILE).is_file()


def test_freeze_and_load_round_trip_preserves_the_hash_and_the_predictions(tmp_path, controller, frames):
    sha = CT.freeze_controller(tmp_path, controller)
    loaded = CT.load_controller(tmp_path / CT.CONTROLLER_DIR / sha)
    assert loaded.sha256 == controller.sha256 == sha
    rows = [dict(r.features) for r in frames[1].rows]
    assert np.allclose(loaded.u(rows), controller.u(rows))


def test_the_sha_is_a_pure_function_of_the_payload(controller):
    assert CT.controller_sha256(controller.payload()) == controller.sha256
    assert CT.controller_sha256({**controller.payload(), CT.SHA_FIELD: "deadbeef"}) == controller.sha256


@pytest.mark.parametrize("mutate", [
    lambda p: p["models"][CT.HEAD_PRIVATE]["coef"].__setitem__(0, 99.0),
    lambda p: p.__setitem__("prior", 0.123),
    lambda p: p["meta"].__setitem__("model_choice", "something else"),
    lambda p: p["selections"][CT.HEAD_PEER].__setitem__("penalty", 1234.0),
])
def test_loading_refuses_a_tampered_artifact(tmp_path, controller, mutate):
    sha = CT.freeze_controller(tmp_path, controller)
    path = tmp_path / CT.CONTROLLER_DIR / sha / CT.CONTROLLER_FILE
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))          # the stamped sha is left untouched
    with pytest.raises(CT.ControllerError, match="mismatch"):
        CT.load_controller(path)


def test_loading_refuses_an_artifact_with_no_stamped_hash(tmp_path, controller):
    sha = CT.freeze_controller(tmp_path, controller)
    path = tmp_path / CT.CONTROLLER_DIR / sha / CT.CONTROLLER_FILE
    payload = json.loads(path.read_text())
    payload.pop(CT.SHA_FIELD)
    path.write_text(json.dumps(payload))
    with pytest.raises(CT.ControllerError, match=CT.SHA_FIELD):
        CT.load_controller(path)


def test_loading_refuses_a_foreign_or_stale_schema(tmp_path, controller):
    for field, value, message in (("kind", "SOMETHING_ELSE", "artifact"),
                                  ("feature_schema_version", 99, "feature schema")):
        payload = {**controller.payload(), field: value}
        payload[CT.SHA_FIELD] = CT.controller_sha256(payload)
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(CT.ControllerError, match=message):
            CT.load_controller(path)


def test_a_non_finite_payload_is_refused_before_it_is_written(tmp_path, controller):
    broken = dataclasses.replace(controller, meta={**controller.meta, "nan": float("nan")})
    with pytest.raises(CT.ControllerError, match="NaN"):
        broken.save(tmp_path / "broken.json")


# --------------------------------------------------------------------------- the policy


def _tied_controller(controller):
    """A controller whose two heads are identical, so ``u`` is exactly 0 for every agent."""
    return dataclasses.replace(
        controller, models={**controller.models, CT.HEAD_PEER: controller.models[CT.HEAD_PRIVATE]})


def test_the_protection_schedule_is_the_same_hmac_random_protect_uses():
    state = S.state_at(("7", "7", "9", "9", "11"))
    expected = min(range(K.N_AGENTS), key=lambda slot: identity.blind_order_key(
        state.study_seed, K.ORDER_PROTECT, state.pairing_id, 1, slot))
    assert P.random_protected_slot(state, round_=1) == expected
    assert CT.random_protected_slot(state, round_=1) == expected
    assert CT.protect_schedule_key(state.study_seed, state.pairing_id, 1, 0) == \
        identity.blind_order_key(state.study_seed, K.ORDER_PROTECT, state.pairing_id, 1, 0)


def test_an_argmax_tie_is_broken_by_the_frozen_schedule_and_is_deterministic(controller):
    tied = _tied_controller(controller)
    state = S.state_at(("7", "7", "9", "9", "11"), round_=1)
    choice = CT.protection_choice(tied, state, round_=1, episode_id="e1")
    assert choice.tied_slots == tuple(range(K.N_AGENTS))      # every u is identical
    expected = min(range(K.N_AGENTS), key=lambda slot: CT.protect_schedule_key(
        state.study_seed, state.pairing_id, 1, slot))
    assert choice.protected_slot == expected
    assert {CT.choose_protected(tied, state, round_=1) for _ in range(8)} == {expected}


def test_the_tie_break_agrees_with_random_protect_on_the_same_state(controller):
    """Spec §12.3: GATE and RANDOM_PROTECT share one recorded schedule and differ in signal."""
    tied = _tied_controller(controller)
    state = S.state_at(("7", "7", "9", "9", "11"), round_=1)
    assert CT.choose_protected(tied, state, round_=1) == P.random_protected_slot(state, round_=1)


def test_exactly_one_agent_is_protected_and_it_is_the_argmax(controller):
    state = S.state_at(("7", "7", "9", "9", "11"), round_=1,
                       confidences=[0.95, 0.1, 0.1, 0.1, 0.1])
    choice = CT.protection_choice(controller, state, round_=1, episode_id="e1")
    assert 0 <= choice.protected_slot < K.N_AGENTS
    assert len(choice.u) == K.N_AGENTS
    assert choice.u[choice.protected_slot] == max(choice.u)
    assert choice.controller_sha256 == controller.sha256


def test_all_negative_u_still_protects_one_agent_and_says_so(controller):
    """Spec §12.3: "If all predicted values are negative ... that is a meaningful failure mode"."""
    state = S.state_at(("7", "7", "9", "9", "11"), round_=1)
    negative = dataclasses.replace(controller)
    row = CT.protection_choice(negative, state, round_=1, episode_id="e1").to_row()
    assert row["protected_slot"] in range(K.N_AGENTS)
    assert isinstance(row["all_negative"], bool)
    assert row["u"] == [pytest.approx(v) for v in
                        CT.protection_choice(negative, state, round_=1).u]


def test_the_choice_never_reads_a_gold_or_an_item_id(controller):
    """The only inputs are the frozen feature rows, the seed and the pairing id."""
    state = S.state_at(("7", "7", "9", "9", "11"), round_=1, item_id="math500:test-0001")
    other = S.state_at(("7", "7", "9", "9", "11"), round_=1, item_id="math500:test-0001")
    assert CT.choose_protected(controller, state, round_=1) == \
        CT.choose_protected(controller, other, round_=1)


# --------------------------------------------------------------------------- evaluation


def test_within_item_auc_excludes_homogeneous_items_from_its_denominator():
    out = CT.within_item_auc([0.9, 0.1, 0.4, 0.6], [1, 0, 1, 1], ["a", "a", "b", "b"])
    assert out["eligible_items"] == 1 and out["excluded_items"] == 1
    assert out["within_item_auc"] == pytest.approx(1.0)
    nothing = CT.within_item_auc([0.9, 0.1], [1, 1], ["a", "a"])
    assert nothing["within_item_auc"] is None and nothing["eligible_items"] == 0


def test_the_brier_decomposition_is_exact_at_singleton_bins_and_reports_its_binning_gap():
    """Murphy holds exactly when p is constant inside a bin; coarser bins report a residual."""
    p = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.35, 0.55]
    y = [0, 0, 1, 0, 1, 1, 0, 1]
    exact = CT.brier_decomposition(p, y, n_bins=len(p))
    assert exact["n_bins"] == len(p)
    assert exact["residual"] == pytest.approx(0.0, abs=1e-9)
    assert exact["brier"] == pytest.approx(
        exact["reliability"] - exact["resolution"] + exact["uncertainty"], abs=1e-9)
    coarse = CT.brier_decomposition(p, y, n_bins=4)
    assert coarse["brier"] == pytest.approx(exact["brier"])
    assert coarse["residual"] != 0.0            # the gap is reported, never hidden
    assert abs(coarse["residual"]) < 0.05
    assert CT.brier_decomposition([], [])["n_bins"] == 0


def test_evaluate_controller_reports_policy_relevance_beside_oracle_and_random(controller, frames):
    out = CT.evaluate_controller(controller, frames[1], study_seed=SEED)
    assert out["p_private"]["n"] and out["p_peer"]["n"]
    policy = out["policy"]
    assert policy["n_teams"] >= 1
    for key in ("policy_no_harm_rate", "mean_chosen_gain", "mean_oracle_gain",
                "mean_random_gain", "regret_vs_oracle"):
        assert key in policy
    assert policy["mean_oracle_gain"] >= policy["mean_chosen_gain"]


def test_evaluate_current_scores_every_baseline_on_identical_rows(controller, frames):
    out = CT.evaluate_current(controller, frames[3])
    for head in (CT.HEAD_CURRENT, CT.HEAD_DYNAMICS, CT.BASELINE_FREQUENCY,
                 CT.BASELINE_CONFIDENCE_RAW, CT.BASELINE_CONFIDENCE_RECALIBRATED,
                 CT.BASELINE_ANSWER_LOGPROB, CT.BASELINE_FREQUENCY_CONFIDENCE):
        assert head in out, head
        assert "status" not in out[head], (head, out[head])
    assert out["n_rows"] == len(frames[3])


def test_fitting_refuses_a_frame_with_no_outcome():
    rows = tuple(dataclasses.replace(r, y_private=None, y_peer=None)
                 for r in S.paired_frame(4).rows)
    frame = CT.PairedFrame(rows)
    with pytest.raises(CT.ControllerError, match="no y_private outcome"):
        CT.fit_controller(frame, frame, study_seed=SEED)
