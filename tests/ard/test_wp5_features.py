"""WP5 features: the firewall, the frozen schema, determinism and the missing-is-missing rule.

Spec §12.2 ("Predefine a maximum of approximately twenty behavioral features"; "No item ID,
reference answer, judged correctness, root pass rate, future stability, or final consensus
feature is allowed"), §6.3 (a missing confidence is missing, never 0.5); plan §5.4.
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from tests.ard import wp5_support as S


# --------------------------------------------------------------------------- the schema


def test_the_schema_is_capped_and_flat_and_unique():
    assert len(FE.FEATURE_GROUPS) <= FE.BEHAVIOUR_FEATURE_CAP
    assert len(set(FE.FEATURES_V1)) == len(FE.FEATURES_V1)
    flat = tuple(c for group in FE.FEATURE_GROUPS for c in group.columns)
    assert FE.FEATURES_V1 == flat


def test_no_family_and_dynamics_subsets_are_subsets_of_the_frozen_schema():
    assert set(FE.NO_FAMILY_FEATURES) < set(FE.FEATURES_V1)
    assert set(FE.DYNAMICS_ONLY_FEATURES) < set(FE.FEATURES_V1)
    assert not any(f"family_is_{f.value}" in FE.NO_FAMILY_FEATURES for f in K.FAMILIES)


def test_the_dynamics_only_selector_reads_no_confidence_and_no_logprob():
    """Spec §12.4's "dynamics-only static selector" is the comparator without self-report."""
    for column in FE.DYNAMICS_ONLY_FEATURES:
        assert "confidence" not in column and "logprob" not in column


# --------------------------------------------------------------------------- the firewall


@pytest.mark.parametrize("name", [
    "item_id", "gold", "gold_answer", "is_correct", "root_pass_rate", "constructed_C0",
    "y_peer", "y_private", "final_consensus", "future_state", "answer_key", "pass_at_k",
    "mean_is_correct", "peer_label", "team_is_correct",
])
def test_the_firewall_refuses_every_forbidden_source(name):
    with pytest.raises(FE.FeatureFirewallError):
        FE.assert_allowed_source(name)


@pytest.mark.parametrize("name", [
    "changed_since_root", "own_class_count", "peer_class_entropy", "count_margin",
    "prev_confidence_delta", "history_unavailable", "thinking_length", "valid",
])
def test_the_firewall_admits_the_legitimate_behavioural_names(name):
    assert FE.assert_allowed_source(name) == name


def test_every_frozen_column_survives_its_own_firewall():
    for column in FE.FEATURES_V1:
        assert FE.assert_allowed_source(column) == column


def test_a_forbidden_key_anywhere_in_a_payload_is_refused():
    with pytest.raises(FE.FeatureFirewallError):
        FE.assert_no_forbidden_source({"own_confidence": 1.0, "is_correct": 1.0})


def test_an_extra_payload_cannot_smuggle_a_label_into_a_feature_row():
    state = S.state_at(("7", "7", "9", "9", None))
    with pytest.raises(FE.FeatureFirewallError):
        FE.build_feature_row(state, 0, round_=0, split="train", extra={"constructed_C0": 4.0})
    with pytest.raises(FE.FeatureFirewallError):
        FE.build_feature_row(state, 0, round_=0, split="train", extra={"y_private": 1.0})


def test_an_extra_payload_may_not_overwrite_the_frozen_schema():
    state = S.state_at(("7", "7", "9", "9", None))
    with pytest.raises(FE.FeatureFirewallError):
        FE.build_feature_row(state, 0, round_=0, split="train", extra={"own_confidence": 0.9})


# --------------------------------------------------------------------------- the builder


def test_the_builder_emits_exactly_the_frozen_columns_as_floats():
    row = S.feature_row()
    assert list(row) == list(FE.FEATURES_V1)
    assert all(isinstance(v, float) for v in row.values())


def test_features_are_deterministic():
    """The same state must give byte-identical rows: the controller's input is frozen."""
    answers = ("7", "7", "9", "9", None)
    first = S.feature_row(answers)
    second = S.feature_row(answers)
    assert first == second
    assert FE.feature_vector(first) == FE.feature_vector(second)


def test_features_do_not_depend_on_the_agent_slot_ordering_of_the_peer_set():
    """Peer statistics are multiset statistics: permuting the four peers changes nothing."""
    base = S.feature_row(("7", "9", "11", "9", None), slot=0)
    permuted = S.feature_row(("7", "9", "9", None, "11"), slot=0)
    peer_columns = [c for c in FE.FEATURES_V1 if c.startswith(("peer_", "largest_", "count_"))]
    assert [base[c] for c in peer_columns] == [permuted[c] for c in peer_columns]


def test_a_missing_confidence_is_missing_and_never_imputed_to_a_half():
    row = S.feature_row(("7", "7", "9", "9", None), confidences=[None, 0.6, 0.6, 0.6, 0.6])
    assert row["own_confidence_missing"] == 1.0
    assert row["own_confidence"] == FE.HISTORY_IMPUTATION == 0.0
    assert row["own_confidence"] != 0.5


def test_history_is_unavailable_at_round_zero_and_available_later():
    at_zero = S.feature_row(round_=0)
    assert at_zero["history_unavailable"] == 1.0
    assert at_zero["prev_confidence"] == FE.HISTORY_IMPUTATION
    assert at_zero["prev_confidence_delta"] == FE.HISTORY_IMPUTATION
    assert at_zero["own_class_count_prior"] == FE.HISTORY_IMPUTATION
    at_one = S.feature_row(round_=1)
    assert at_one["history_unavailable"] == 0.0
    assert at_one["own_class_count_prior"] > 0.0


def test_runtime_access_flags_follow_the_injected_records_mapping():
    state = S.state_at(("7", "7", "9", "9", None))
    without = FE.features_from_state(state, 0, round_=0)
    assert without["own_answer_logprob_missing"] == 1.0
    assert without["peer_answer_logprob_missing"] == 1.0
    assert without["own_answer_logprob"] == FE.HISTORY_IMPUTATION
    records = {state.latest(slot).decision_id: S.StubRecord(answer_logprob=-0.75,
                                                            thinking_tokens=321, cap_hit=True)
               for slot in range(K.N_AGENTS)}
    with_access = FE.features_from_state(state, 0, round_=0, records=records)
    assert with_access["own_answer_logprob_missing"] == 0.0
    assert with_access["own_answer_logprob"] == -0.75
    assert with_access["peer_answer_logprob_mean"] == -0.75
    assert with_access["thinking_length"] == 321.0
    assert with_access["cap_hit"] == 1.0


def test_counts_and_entropy_describe_the_realised_class_multiset():
    row = S.feature_row(("7", "7", "9", "9", "11"), slot=0)
    assert row["own_class_count"] == 2.0          # two agents hold "7"
    assert row["peer_unique_classes"] == 3.0      # 7, 9, 11 among the four peers
    assert row["largest_competing_count"] == 2.0  # the two "9"s
    assert row["count_margin"] == 0.0
    assert row["peer_class_entropy"] == pytest.approx(FE.peer_class_entropy([1, 2, 1]))


def test_an_invalid_own_report_is_flagged_and_forms_no_class():
    row = S.feature_row((None, "7", "7", "9", "9"), slot=0)
    assert row["valid"] == 0.0
    assert row["own_class_count"] == 0.0
    assert row["own_answer_length"] == 0.0


def test_changed_since_root_uses_the_class_where_there_is_one():
    same = S.feature_row(("7", "7", "9", "9", "11"), roots=("7", "7", "9", "9", "11"))
    changed = S.feature_row(("7", "7", "9", "9", "11"), roots=("9", "7", "9", "9", "11"))
    assert same["changed_since_root"] == 0.0
    assert changed["changed_since_root"] == 1.0


# --------------------------------------------------------------------------- the row


def test_training_eligibility_excludes_test_items():
    assert FE.training_eligibility("train") == "eligible"
    assert FE.training_eligibility("validation") == "eligible"
    assert FE.training_eligibility("test") == "excluded_test"


def test_the_parquet_row_carries_the_spec_field_list_and_a_json_feature_object():
    import json

    state = S.state_at(("7", "7", "9", "9", None))
    row = FE.build_feature_row(state, 0, round_=0, split="test").to_row()
    assert set(row) == {
        "decision_or_state_id", "feature_time", "last_allowed_round", "feature_schema_version",
        "behavior_features", "neural_feature_file", "model_revision", "training_eligibility",
    }
    assert row["training_eligibility"] == "excluded_test"
    assert row["feature_schema_version"] == K.FEATURE_SCHEMA_VERSION
    assert list(json.loads(row["behavior_features"])) == sorted(FE.FEATURES_V1)


def test_a_feature_row_refuses_an_unknown_feature_time_or_round():
    state = S.state_at(("7", "7", "9", "9", None))
    with pytest.raises(ValueError):
        FE.build_feature_row(state, 0, round_=0, split="train", feature_time="whenever")
    with pytest.raises(ValueError):
        FE.features_from_state(state, 0, round_=9)


def test_team_features_are_indexed_by_true_slot():
    state = S.state_at(("7", "9", "9", "9", "9"))
    rows = FE.team_features(state, round_=0)
    assert len(rows) == K.N_AGENTS
    assert rows[0]["own_class_count"] == 1.0
    assert rows[1]["own_class_count"] == 4.0


def test_feature_vector_refuses_a_row_missing_a_frozen_column():
    row = dict(S.feature_row())
    row.pop("own_confidence")
    with pytest.raises(FE.FeatureFirewallError):
        FE.feature_vector(row)
