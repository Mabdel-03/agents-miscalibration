"""The literal list of the 74 spec-18 test names (spec 18.1-18.5, lines 1413-1512).

Every name is used verbatim by the module that owns it.  ``test_spec18_names_not_yet_collected``
prints the names no test file under ``tests/nextdyn`` collects yet; it never fails while work
packages are pending (the release gate ``release-check`` requires the missing list to be
empty), it only asserts that the index has exactly 74 entries.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

COLLECTED_ATTR = "nextdyn_collected_test_names"
TESTS_DIR = Path(__file__).resolve().parent

SPEC18_IDENTITY = (
    "test_source_item_id_never_null",
    "test_bank_role_cannot_alias_private_episode",
    "test_composite_decision_key_unique",
    "test_independent_draws_have_distinct_generation_uids",
    "test_same_content_does_not_imply_same_sample",
    "test_request_cache_invalidates_on_model_prompt_seed_or_parent_change",
    "test_all_variants_of_source_item_share_split_and_cluster",
    "test_prior_use_inventory_marks_unknown_not_untouched",
    "test_solver_export_contains_no_gold_or_construction_correctness",
    "test_candidate_ids_and_slot_labels_do_not_encode_correctness",
    "test_selector_service_cannot_read_private_grades",
    "test_hidden_code_tests_absent_from_generator_and_selection_sandbox",
)
SPEC18_PROTOCOL = (
    "test_paired_protocols_have_byte_identical_roots",
    "test_c1_c4_exact_occupancy_and_shared_wrong_class",
    "test_no_donor_record_duplicated_within_team",
    "test_peer_messages_use_only_previous_round",
    "test_private_has_no_peer_content",
    "test_fresh_has_no_own_or_peer_history",
    "test_four_revisions_produce_five_observed_states",
    "test_all_agents_update_after_unanimity",
    "test_c11_c01_c10_c00_have_20_16_16_12_edges",
    "test_isolated_focal_does_not_isolate_other_agents",
    "test_focal_slot_is_fixed_after_root",
    "test_ring_indegree_and_outdegree_are_two",
    "test_n3_ring_equals_all_to_all_is_not_duplicate_treatment",
    "test_r4_prefix_of_r8_is_identical_when_prompts_are_horizon_free",
    "test_missing_packet_padding_has_no_task_content",
)
SPEC18_MEASUREMENT = (
    "test_c_w_u_counts_sum_to_n",
    "test_three_state_transition_identity",
    "test_validity_change_is_not_answer_switch",
    "test_wrong_unanimity_requires_one_shared_wrong_class",
    "test_historical_coverage_is_nondecreasing",
    "test_active_coverage_is_bounded_by_historical_coverage",
    "test_current_selector_accuracy_not_above_current_oracle_on_same_pool",
    "test_failure_decomposition_on_fully_graded_fixture",
    "test_lower_upper_coverage_bounds_include_all_feasible_labels",
    "test_passk_uses_fixed_items_and_correct_bank_role",
    "test_passk_prefix_oracle_and_vote_use_same_actual_candidates",
    "test_tie_break_is_invariant_to_protocol_and_agent_order",
    "test_selector_returns_existing_permitted_candidate",
    "test_missing_or_unfitted_selector_is_unexecuted_not_zero_accuracy",
    "test_within_item_auc_requires_mixed_candidate_groups",
    "test_confidence_not_imputed_as_model_statement",
    "test_option_shuffle_preserves_canonical_answer_identity",
)
SPEC18_RECEIVER = (
    "test_mod97_evaluator_independent_of_generator",
    "test_all_depths_show_24_operations",
    "test_answer_dependency_depth_exact",
    "test_wrong_trace_has_exactly_one_violated_local_step",
    "test_wrong_and_correct_trace_final_answers_differ",
    "test_support_conditions_have_identical_packet_bodies",
    "test_support_metadata_is_not_counted_as_independent_generations",
    "test_checker_only_reads_named_public_packet_and_step",
    "test_checker_does_not_reveal_gold_or_recompute_final_answer",
    "test_disabled_checker_never_returns_validity",
    "test_at_most_four_check_requests_and_fixed_duplicate_policy",
    "test_two_phase_thinking_caps_sum_to_declared_episode_budget",
    "test_14400_receiver_episodes_are_28800_generation_calls_per_model",
)
SPEC18_SERVING = (
    "test_reasoning_cap_is_enforced_at_declared_boundary",
    "test_final_phase_has_its_reserved_cap",
    "test_native_and_public_reasoning_fields_not_confused",
    "test_pinned_backend_response_fields_supported",
    "test_worst_case_context_fits_every_compared_model",
    "test_infrastructure_retry_keeps_exact_request_and_seed",
    "test_malformed_model_output_is_not_stochastically_retried",
    "test_activation_layer_and_anchor_offsets_valid",
    "test_teacher_forced_replay_uses_exact_stored_token_ids",
    "test_all_root_agent_captures_exist_or_are_explicitly_missing",
    "test_zero_variance_spectral_metrics_are_undefined",
    "test_geometry_rank_not_above_n_minus_one",
    "test_read_only_capture_cannot_change_generation",
    "test_attributed_gpu_time_conserves_device_totals",
    "test_campaign_and_per_system_cost_are_separate",
    "test_budget_admission_reserves_selection_cost",
    "test_stop_is_available_in_every_action_policy",
)
SPEC18_TESTS: tuple[str, ...] = (
    *SPEC18_IDENTITY, *SPEC18_PROTOCOL, *SPEC18_MEASUREMENT, *SPEC18_RECEIVER, *SPEC18_SERVING,
)

#: The subset WP0 makes pass now (plan §5 WP0; the task brief).
WP0_OWNED = (
    "test_source_item_id_never_null",
    "test_bank_role_cannot_alias_private_episode",
    "test_composite_decision_key_unique",
    "test_request_cache_invalidates_on_model_prompt_seed_or_parent_change",
    "test_same_content_does_not_imply_same_sample",
)


def test_spec18_index_has_exactly_74_distinct_names():
    assert len(SPEC18_TESTS) == 74
    assert len(set(SPEC18_TESTS)) == 74
    assert (len(SPEC18_IDENTITY), len(SPEC18_PROTOCOL), len(SPEC18_MEASUREMENT), len(SPEC18_RECEIVER),
            len(SPEC18_SERVING)) == (12, 15, 17, 13, 17)
    assert all(name.startswith("test_") for name in SPEC18_TESTS)


def test_spec18_names_not_yet_collected(request, capsys):
    """Reports (never fails on) the spec-18 names no test under tests/nextdyn collects yet."""
    collected = getattr(request.config, COLLECTED_ATTR, set())
    missing = [name for name in SPEC18_TESTS if name not in collected]
    present = [name for name in SPEC18_TESTS if name in collected]
    print(f"spec-18 collected: {len(present)}/74; missing ({len(missing)}):")
    for name in missing:
        print(f"  - {name}")
    for name in WP0_OWNED:
        assert name in collected, f"WP0 owns {name} and it must be collected now"
    assert len(present) + len(missing) == 74


def _defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


@pytest.mark.parametrize("name", WP0_OWNED)
def test_wp0_owned_spec18_names_are_defined_in_wp0_test_modules(name):
    defined = _defined_tests(TESTS_DIR / "test_identity.py") | _defined_tests(TESTS_DIR / "test_tables.py")
    assert name in defined
