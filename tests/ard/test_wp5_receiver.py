"""WP5 receiver assay: the frozen 64 programs, the content-matched packets, the outcomes.

Spec §5.4 ("64 fresh, deterministic small program-trace tasks ... Disallow division,
imports, loops, randomness, file access, and undefined behavior.  A second independent
evaluator validates the gold output"), §11 (the five presentations; "Keep packet content and
length identical between the two count assignments apart from the swapped counts"; two
preregistered internal checks with ``NOT CHECKED`` placeholders; "grade its produced answer
rather than forcing agreement with the packet"); plan §5.5 (the frozen per-program
order/provisional schedule, balanced within depth x operator-mix strata; the answer-logit
margin is missing, not zero, when the alternative is outside the top-20).
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from agents_scaling.ard import client as ard_client
from agents_scaling.ard import contract as K
from agents_scaling.ard import receiver as RX
from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard.types import EXIT_DONE
from tests.ard import wp5_support as S


@pytest.fixture(scope="module")
def programs():
    return RX.programs()


# --------------------------------------------------------------------------- the set


def test_there_are_sixty_four_programs_balanced_over_depth_and_operator_mix(programs):
    assert len(programs) == K.RECEIVER_PROGRAMS == 64
    assert len({p.program_id for p in programs}) == len(programs)
    counts = Counter(p.stratum for p in programs)
    assert set(counts) == set(RX.STRATA)
    assert set(counts.values()) == {RX.PROGRAMS_PER_STRATUM}
    assert {p.depth for p in programs} == set(K.RECEIVER_DEPTHS)
    assert {p.mix for p in programs} == set(RX.MIXES)


def test_the_programs_are_a_pure_function_of_the_contract(programs):
    rebuilt = RX.build_programs()
    assert [p.content_sha256 for p in rebuilt] == [p.content_sha256 for p in programs]
    assert [p.source for p in rebuilt] == [p.source for p in programs]


def test_coefficients_seeds_and_depths_come_from_the_frozen_contract(programs):
    for program in programs:
        assert program.seed_x in K.RECEIVER_SEEDS and program.seed_y in K.RECEIVER_SEEDS
        assert len(program.updates) == program.depth
        for update in program.updates:
            for value in (update.a, update.b):
                assert value is None or value in K.RECEIVER_COEFFICIENTS


# --------------------------------------------------------------------------- the grammar


def test_no_rendered_program_carries_a_forbidden_construct(programs):
    """Spec §5.4: no division, exponentiation, imports, loops, randomness or file access."""
    for program in programs:
        RX.assert_permitted_program(program.source)          # the allowlist
        text = program.source
        for token in ("/", "**", "^", "import", "for", "while", "lambda", "open", "eval",
                      "exec", "def", "class", "return", "print", "[", "]", ".", "&", "|",
                      "<", ">", "!", '"', "'", ";", "random"):
            assert token not in text, (program.program_id, token)
        assert set(text) <= set("xy0123456789+*%()= \n")


def test_the_second_evaluator_accepts_every_rendered_program(programs):
    """The WP3 evaluator's own denylist and grammar must accept what WP5 renders."""
    from agents_scaling.ard.graders import trace as TR

    for program in programs:
        statements = TR.parse_program(program.source)
        assert len(statements) == program.n_statements


def test_the_allowlist_refuses_anything_outside_the_three_permitted_forms():
    for bad in ("x = (3*x + 5) / 97", "import os", "for i in range(3): x = 1",
                "z = (x + y) % 97", "x = (3**x + 5) % 97", "x = (3*x + 5) % 13"):
        with pytest.raises(RX.ReceiverError):
            RX.assert_permitted_program(bad)


def test_every_program_depends_on_prior_state_and_computes_nothing_unused(programs):
    for program in programs:
        RX.assert_state_dependent(program)
        assert program.updates[-1].target == "x"
        for update in program.updates:
            assert update.reads                      # every update reads prior state


# --------------------------------------------------------------------------- two evaluators


def test_the_two_independent_evaluators_agree_on_all_sixty_four_golds(programs):
    """Spec §5.4: "A second independent evaluator validates the gold output"."""
    report = RX.verify_programs(programs)
    assert report["n_programs"] == 64
    assert report["disagreements"] == []
    assert report["agrees"] is True
    assert len(report["golds"]) == 64
    assert all(0 <= value < K.RECEIVER_MODULUS for value in report["golds"].values())


def test_the_interpreter_matches_the_trace_evaluator_step_by_step(programs):
    from agents_scaling.ard.graders import trace as TR

    program = programs[0]
    mine = RX.interpret(program)
    theirs = TR.evaluate_program(program.source, modulus=K.RECEIVER_MODULUS)
    assert [step.value for step in mine] == [step.value for step in theirs.steps]
    assert [step.target for step in mine] == [step.target for step in theirs.steps]


# --------------------------------------------------------------------------- the traces


def test_the_two_candidate_traces_always_end_on_different_answers(programs):
    for program in programs:
        first, second = RX.candidate_traces(program)
        assert {first.kind, second.kind} == {"correct", "corrupted"}
        assert first.final_answer != second.final_answer
        assert 0 <= first.final_answer < K.RECEIVER_MODULUS


def test_the_single_error_is_internal_and_propagates_consistently(programs):
    from agents_scaling.ard.graders import trace as TR

    for program in programs:
        assert 1 <= program.error_update <= program.depth - 2      # never first, never last
        first, second = RX.candidate_traces(program)
        corrupted = first if first.kind == "corrupted" else second
        checks = TR.check_trace(program.source, corrupted.render(), modulus=K.RECEIVER_MODULUS)
        divergence = TR.first_divergence(checks)
        assert divergence is not None
        assert divergence.index == program.statement_index(program.error_update)
        # consistent propagation: everything before the error agrees, and the error is a
        # single arithmetic shift at one step, not a second independent edit.
        assert all(check.agrees for check in checks[:divergence.index])
        assert 1 <= program.error_delta < K.RECEIVER_MODULUS


def test_the_correct_trace_verifies_everywhere(programs):
    from agents_scaling.ard.graders import trace as TR

    for program in programs[:8]:
        first, second = RX.candidate_traces(program)
        correct = first if first.kind == "correct" else second
        checks = TR.check_trace(program.source, correct.render(), modulus=K.RECEIVER_MODULUS)
        assert TR.first_divergence(checks) is None
        assert correct.final_answer == program.gold


# --------------------------------------------------------------------------- the schedule


def test_the_order_and_provisional_schedule_is_balanced_within_every_stratum(programs):
    """Plan §5.5: 16 programs per order x provisional cell, balanced within depth x mix."""
    overall = Counter((p.schedule.order, p.schedule.provisional) for p in programs)
    assert set(overall) == set(RX.SCHEDULE_CELLS)
    assert set(overall.values()) == {16}
    per_stratum = Counter((p.stratum, p.schedule.order, p.schedule.provisional) for p in programs)
    assert len(per_stratum) == len(RX.STRATA) * len(RX.SCHEDULE_CELLS)
    assert set(per_stratum.values()) == {RX.PROGRAMS_PER_CELL_PER_STRATUM}


def test_the_schedule_is_a_frozen_per_program_attribute(programs):
    once, twice = RX.schedule(), RX.schedule()
    assert once == twice
    assert {p.program_id: p.schedule for p in programs} == once


def test_the_schedule_is_identical_across_every_presentation(programs):
    """This is what makes the endorsement contrast within-program and content-matched."""
    for program in programs[:6]:
        packets = [RX.build_packet(program, name) for name in K.RECEIVER_PRESENTATIONS]
        assert {p.order for p in packets} == {program.schedule.order}
        assert {p.provisional for p in packets} == {program.schedule.provisional}
        assert {p.provisional_answer for p in packets} == {packets[0].provisional_answer}
        assert {p.candidate_answers for p in packets} == {packets[0].candidate_answers}
        assert {p.correct_label for p in packets} == {program.schedule.correct_label}


def test_the_schedule_does_not_read_the_budget_the_seed_or_the_checkpoint(programs):
    """The packet is a function of (program, presentation) alone — nothing else varies it."""
    program = programs[3]
    first = RX.build_packet(program, "MINORITY_CHECKED")
    for _budget in (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH):
        for _seed in range(K.RECEIVER_SEEDS_PER_CONDITION):
            for _checkpoint in (K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE):
                again = RX.build_packet(program, "MINORITY_CHECKED")
                assert again.text == first.text
                assert again.content_sha256 == first.content_sha256


def test_the_provisional_answer_follows_the_frozen_attribute(programs):
    for program in programs:
        packet = RX.build_packet(program, "PRIVATE_TRACE")
        if program.schedule.provisional_is_correct:
            assert packet.provisional_answer == program.gold
        else:
            assert packet.provisional_answer != program.gold


# --------------------------------------------------------------------------- the packets


def test_the_five_frozen_presentations_are_exactly_the_contract_set(programs):
    assert K.RECEIVER_PRESENTATIONS == ("PRIVATE_TRACE", "MINORITY_UNCHECKED",
                                        "MAJORITY_UNCHECKED", "MINORITY_CHECKED",
                                        "MAJORITY_CHECKED")
    with pytest.raises(RX.ReceiverError, match="unknown presentation"):
        RX.build_packet(programs[0], "SOMETHING_ELSE")


def test_the_private_trace_presentation_shows_one_trace_and_no_social_metadata(programs):
    packet = RX.build_packet(programs[0], "PRIVATE_TRACE")
    assert packet.counts is None and packet.checks == ()
    assert "Candidate A:" not in packet.text and "Endorsements" not in packet.text
    assert "Local checks" not in packet.text
    assert "Provisional trace:" in packet.text


@pytest.mark.parametrize("suffix", ["UNCHECKED", "CHECKED"])
def test_the_two_count_assignments_are_byte_identical_apart_from_the_swapped_counts(
        programs, suffix):
    """Spec §11: "Keep packet content and length identical ... apart from the swapped counts"."""
    for program in programs[:12]:
        minority = RX.build_packet(program, f"MINORITY_{suffix}")
        majority = RX.build_packet(program, f"MAJORITY_{suffix}")
        assert len(minority.text) == len(majority.text)
        differences = [(i, a, b) for i, (a, b) in enumerate(zip(minority.text, majority.text))
                       if a != b]
        assert len(differences) == 2                      # exactly the two endorsement digits
        assert {a for _, a, _ in differences} == {"1", "4"}
        assert {b for _, _, b in differences} == {"1", "4"}
        # and the difference really is the endorsement line, not a coincidence elsewhere
        swapped = minority.text.replace(RX.endorsement_line(*minority.counts),
                                        RX.endorsement_line(*majority.counts))
        assert swapped == majority.text
        assert tuple(reversed(minority.counts)) == majority.counts


def test_the_minority_assignment_gives_the_correct_candidate_one_of_five(programs):
    for program in programs:
        packet = RX.build_packet(program, "MINORITY_UNCHECKED")
        counts = dict(zip(RX.CANDIDATE_LABELS, packet.counts))
        assert counts[program.schedule.correct_label] == 1
        assert counts[program.schedule.wrong_label] == 4
        majority = RX.build_packet(program, "MAJORITY_UNCHECKED")
        counts = dict(zip(RX.CANDIDATE_LABELS, majority.counts))
        assert counts[program.schedule.correct_label] == 4


def test_the_counts_are_labelled_as_synthetic_randomized_metadata(programs):
    text = RX.build_packet(programs[0], "MAJORITY_CHECKED").text
    assert "synthetic randomized metadata, not independent generation" in text


# --------------------------------------------------------------------------- the checks


def test_a_checked_packet_reveals_exactly_two_preregistered_local_checks(programs):
    for program in programs:
        packet = RX.build_packet(program, "MINORITY_CHECKED")
        assert len(packet.checks) == K.RECEIVER_CHECKS_PER_PACKET == 2
        corrupted = [c for c in packet.checks if c.is_corrupted_location]
        assert len(corrupted) == 1
        statuses = [s for check in packet.checks for _, s in check.statuses]
        assert statuses.count(RX.CHECK_FAILED) == 1        # only the corrupted claim fails
        assert RX.CHECK_NOT_CHECKED not in statuses


def test_the_checks_never_touch_the_final_transition_or_the_final_output(programs):
    for program in programs:
        packet = RX.build_packet(program, "MAJORITY_CHECKED")
        last = program.n_statements - 1
        for check in packet.checks:
            assert check.statement_index < last
            assert check.statement_index >= 2          # never an initialisation either
        assert packet.checks[0].statement_index < packet.checks[1].statement_index


def test_the_matched_valid_step_shares_the_corrupted_step_s_update_form(programs):
    for program in programs:
        assert program.updates[program.matched_update].form == \
            program.updates[program.error_update].form
        assert program.matched_update < program.error_update
        packet = RX.build_packet(program, "MINORITY_CHECKED")
        matched = packet.checks[0]
        assert not matched.is_corrupted_location
        # both candidates claim the same value at a matched valid step
        assert len({value for _, value in matched.claims}) == 1


def test_the_unchecked_counterpart_shows_the_same_locations_with_no_extra_explanation(programs):
    for program in programs[:12]:
        checked = RX.build_packet(program, "MINORITY_CHECKED")
        unchecked = RX.build_packet(program, "MINORITY_UNCHECKED")
        assert [c.statement_index for c in unchecked.checks] == \
            [c.statement_index for c in checked.checks]
        assert [c.claims for c in unchecked.checks] == [c.claims for c in checked.checks]
        statuses = {s for check in unchecked.checks for _, s in check.statuses}
        assert statuses == {RX.CHECK_NOT_CHECKED}
        # "not an additional explanation": blanking the checked packet's status column
        # reproduces the unchecked packet exactly — same locations, same claimed values,
        # not one extra word.
        blanked = re.sub(rf": ({RX.CHECK_VERIFIED}|{RX.CHECK_FAILED})$",
                         f": {RX.CHECK_NOT_CHECKED}", checked.text, flags=re.M)
        assert blanked == unchecked.text
        extra = set(unchecked.text.split()) - set(checked.text.split())
        assert extra <= set(RX.CHECK_NOT_CHECKED.split())


# --------------------------------------------------------------------------- the margin


def test_the_receiver_purpose_asks_for_the_full_top_twenty(programs):
    """Plan §5.5: the margin is only computable because WP1 widens this one purpose."""
    assert K.RECEIVER_TOP_LOGPROBS == 20
    assert ard_client.top_logprobs_for(K.PURPOSE_RECEIVER) == K.RECEIVER_TOP_LOGPROBS
    assert ard_client.top_logprobs_for(K.PURPOSE_REVISE) == 1


def test_the_first_differing_answer_token_is_found_at_the_right_index():
    assert RX.first_differing_token("41", "58") == (0, "4", "5")
    assert RX.first_differing_token("41", "48") == (1, "1", "8")
    assert RX.first_differing_token("41", "41") is None
    assert RX.first_differing_token("4", "41") == (1, None, "1")


def test_the_margin_is_the_logit_difference_at_that_token():
    block = S.logprobs_block(["<think>", "</think>", "<final>", "4", "1", "</final>"],
                             alternatives={3: (("5", -3.0),)}, logprob=-0.25)
    margin = RX.answer_logit_margin(block, chosen_answer="41", alternative_answer="58")
    assert margin.status == RX.MARGIN_OK
    assert margin.value == pytest.approx(2.75)
    assert margin.chosen_token == "4" and margin.alternative_token == "5"
    assert margin.missing is False


def test_the_margin_is_missing_not_zero_when_the_alternative_is_outside_the_top_k():
    """Plan §5.5: "missing, not zero, when outside the top-20"."""
    block = S.logprobs_block(["<final>", "4", "1", "</final>"], logprob=-0.25)
    margin = RX.answer_logit_margin(block, chosen_answer="41", alternative_answer="58")
    assert margin.status == RX.MARGIN_OUTSIDE_TOP_K
    assert margin.value is None and margin.missing is True
    assert margin.to_row()["answer_logit_margin"] is None
    assert margin.to_row()["answer_logit_margin_missing"] is True


@pytest.mark.parametrize("block,chosen,alternative,status", [
    (None, "41", "58", RX.MARGIN_NO_LOGPROBS),
    ([], "41", "58", RX.MARGIN_NO_LOGPROBS),
    ([{"token": "<final>", "logprob": -0.1, "top_logprobs": []}], "41", "41",
     RX.MARGIN_NO_DIFFERENCE),
    ([{"token": "<final>", "logprob": -0.1, "top_logprobs": []}], "41", "58",
     RX.MARGIN_TOKEN_NOT_FOUND),
])
def test_every_missing_margin_names_its_own_missingness(block, chosen, alternative, status):
    margin = RX.answer_logit_margin(block, chosen_answer=chosen, alternative_answer=alternative)
    assert margin.status == status and margin.value is None


def test_a_record_without_the_top_k_block_records_the_gap_rather_than_a_zero(programs):
    packet = RX.build_packet(programs[0], "MINORITY_CHECKED")
    margin = RX.margin_from_record(object(), packet, produced=str(packet.candidate_answers[0]))
    assert margin.status == RX.MARGIN_UNAVAILABLE and margin.value is None


# --------------------------------------------------------------------------- grading


def test_the_vendored_residue_extractor_agrees_with_the_second_evaluator():
    from agents_scaling.ard.graders import trace as TR

    for text in ("41", " 41 ", "the answer is 41", "x = 41", "97", "-3", "", "no digits",
                 "41 then 58", "0", "96", "1000", "12.5"):
        assert RX.extract_residue(text) == TR.extract_residue(text), text
    assert RX.extract_residue(None) is None and TR.extract_residue(None) is None


def _outcome(program, presentation, answer):
    packet = RX.build_packet(program, presentation)
    return RX.grade_receiver(program, packet, S.report(answer))


def test_the_outcome_classes_are_the_frozen_five(programs):
    assert RX.OUTCOMES == ("picked-correct", "picked-wrong", "produced-other-correct",
                           "produced-other-wrong", "invalid")


def test_picking_the_correct_candidate_is_picked_correct(programs):
    program = programs[0]
    outcome = _outcome(program, "MINORITY_CHECKED", str(program.gold))
    assert outcome.outcome == RX.OUTCOME_PICKED_CORRECT
    assert outcome.correct and outcome.picked_label == program.schedule.correct_label


def test_picking_the_endorsed_wrong_candidate_is_graded_wrong_not_agreed_with(programs):
    """Spec §11: grade the produced answer, never agreement with the packet's metadata."""
    program = programs[0]
    packet = RX.build_packet(program, "MAJORITY_CHECKED")
    wrong = next(v for v in packet.candidate_answers if v != packet.gold)
    outcome = RX.grade_receiver(program, packet, S.report(str(wrong)))
    assert outcome.outcome == RX.OUTCOME_PICKED_WRONG
    assert outcome.picked_label == program.schedule.wrong_label
    assert outcome.correct is False
    assert max(packet.counts) == 4                      # the packet endorsed it four to one


def test_rejecting_both_candidates_is_graded_against_the_interpreter_gold(programs):
    program = programs[0]
    packet = RX.build_packet(program, "MINORITY_UNCHECKED")
    other = next(v for v in range(K.RECEIVER_MODULUS) if v not in packet.candidate_answers)
    outcome = RX.grade_receiver(program, packet, S.report(str(other)))
    assert outcome.outcome == RX.OUTCOME_PRODUCED_OTHER_WRONG
    assert outcome.picked_label is None
    assert outcome.gold == program.gold


def test_an_unparseable_or_unkeyable_answer_is_invalid_and_keeps_its_confidence(programs):
    program = programs[0]
    packet = RX.build_packet(program, "MINORITY_CHECKED")
    for report in (S.report(None, confidence=0.42), S.report("no residue here", confidence=0.42)):
        outcome = RX.grade_receiver(program, packet, report)
        assert outcome.outcome == RX.OUTCOME_INVALID and outcome.residue is None
    kept = RX.grade_receiver(program, packet, S.report("no residue here", confidence=0.42))
    assert kept.confidence == 0.42


def test_final_confidence_missing_is_missing_and_never_a_half(programs):
    program = programs[0]
    packet = RX.build_packet(program, "MINORITY_CHECKED")
    outcome = RX.grade_receiver(program, packet, S.report(str(packet.gold), confidence=None))
    assert outcome.confidence is None
    assert outcome.to_row()["final_confidence_missing"] is True
    assert outcome.to_row()["final_confidence"] is None


def test_retention_and_correction_are_defined_against_the_provisional_attribute(programs):
    correct_provisional = next(p for p in programs if p.schedule.provisional_is_correct)
    wrong_provisional = next(p for p in programs if not p.schedule.provisional_is_correct)

    kept = _outcome(correct_provisional, "MINORITY_CHECKED", str(correct_provisional.gold))
    assert kept.retained is True and kept.corrected is None

    packet = RX.build_packet(wrong_provisional, "MINORITY_CHECKED")
    fixed = RX.grade_receiver(wrong_provisional, packet, S.report(str(packet.gold)))
    assert fixed.corrected is True and fixed.retained is None
    assert fixed.switched_to_correct is True

    stuck = RX.grade_receiver(wrong_provisional, packet,
                              S.report(str(packet.provisional_answer)))
    assert stuck.corrected is False and stuck.switched_to_correct is False


def test_the_outcome_row_carries_the_design_axes_of_the_registered_estimand(programs):
    row = _outcome(programs[0], "MINORITY_CHECKED", str(programs[0].gold)).to_row()
    for key in ("presentation", "endorsement_counts", "checked", "order", "provisional",
                "depth", "operator_mix", "final_confidence", "answer_logit_margin",
                "answer_logit_margin_status"):
        assert key in row, key


# --------------------------------------------------------------------------- the cell


def test_the_receiver_worker_is_the_one_the_kind_table_names():
    from agents_scaling.ard.kinds import KIND_TABLE
    from agents_scaling.ard.types import ArdCellKind

    spec = KIND_TABLE[ArdCellKind.RECEIVER]
    assert spec.worker == "agents_scaling.ard.receiver:run_receiver_cell"
    assert ard_runner.resolve_worker(ArdCellKind.RECEIVER) is RX.run_receiver_cell


def test_the_synthetic_program_tasks_are_test_split_and_carry_no_gold_pointer():
    tasks = RX.receiver_tasks()
    assert len(tasks) == K.RECEIVER_PROGRAMS
    for task in tasks.values():
        assert task.split == "test"           # never eligible for controller training
        assert task.gold_location == ""       # the E3 reference is computed, not stored
        assert task.family == RX.RECEIVER_TASK_FAMILY in {f.value for f in K.FAMILIES}
        assert RX.RECEIVER_FAMILY in task.stratum      # the real E3 tag travels here
        assert task.options_json == "[]"


def test_the_receiver_family_tag_matches_the_grader_s():
    from agents_scaling.ard import graders

    assert RX.RECEIVER_FAMILY == graders.TRACE_FAMILY


@pytest.mark.parametrize("presentation", ["PRIVATE_TRACE", "MAJORITY_CHECKED"])
def test_the_receiver_cell_runs_one_decision_per_program_and_records_the_outcome(
        tmp_path, presentation):
    run_root = S.make_run_root(tmp_path)
    programs = RX.programs()[:2]
    tasks = [RX.receiver_task(p) for p in programs]
    cell = S.receiver_cell(tasks, presentation=presentation)
    gold = str(programs[0].gold)

    tokenizer = S.shared_tokenizer()
    with S.fake_server(run_root, tokenizer=tokenizer, answer_pool=(gold,)) as server:
        code = S.cell_runner(cell, run_root, tokenizer=tokenizer,
                             task_loader=RX.receiver_task_loader)
        assert code == EXIT_DONE
        assert server.chat_count == len(cell.items)

    payload = json.loads((run_root / "cells" / cell.cell_id / "items" /
                          f"{programs[0].program_id}.json").read_text())
    episode = payload["episodes"][0]
    assert episode["packet_policy"] == f"receiver:{presentation}"
    assert episode["condition_manifest_hash"] == programs[0].content_sha256
    decision = episode["decisions"][0]
    assert decision["phase"] == K.PURPOSE_RECEIVER
    assert decision["answer_class"] == f"mod{K.RECEIVER_MODULUS}:{programs[0].gold}"
    outcome = next(e for e in episode["events"] if e["event"] == "receiver_outcome")
    assert outcome["outcome"] == RX.OUTCOME_PICKED_CORRECT
    assert outcome["gold"] == programs[0].gold
    assert outcome["presentation"] == presentation
    assert outcome["answer_logit_margin_missing"] is True     # missing, never zero
    packet_row = next(e for e in episode["events"] if e["event"] == "receiver_packet")
    assert packet_row["checked"] is (presentation == "MAJORITY_CHECKED")


def test_the_receiver_prompt_is_the_frozen_packet_text(tmp_path):
    run_root = S.make_run_root(tmp_path)
    program = RX.programs()[0]
    cell = S.receiver_cell([RX.receiver_task(program)], presentation="MINORITY_CHECKED")
    expected = RX.build_packet(program, "MINORITY_CHECKED").text

    tokenizer = S.shared_tokenizer()
    with S.fake_server(run_root, tokenizer=tokenizer, answer_pool=("11",)) as server:
        assert S.cell_runner(cell, run_root, tokenizer=tokenizer,
                             task_loader=RX.receiver_task_loader) == EXIT_DONE
        sent = [e["body"]["messages"][0]["content"] for e in server.requests
                if e["path"] == "/v1/chat/completions"]
    assert sent == [expected]


def test_the_module_level_import_closure_never_reaches_the_gold_reader():
    """Contract §0.2: the grader package is reached only from inside a function."""
    forbidden = {"agents_scaling.ard.graders", "agents_scaling.ard.graders.trace",
                 "agents_scaling.ard.items_build", "datasets"}
    tree = ast.parse(Path(RX.__file__).read_text())
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module, f"{node.module}.{node.names[0].name}"]
        assert not set(targets) & forbidden, targets
