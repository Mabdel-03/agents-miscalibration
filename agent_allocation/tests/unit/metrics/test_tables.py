"""metrics/tables.py: the sealed grade join, bounds, invariants, transitions and fixed populations."""
from __future__ import annotations

import warnings
from dataclasses import replace
from fractions import Fraction

import pytest

from agent_allocation.contracts import CheckpointView, GradeStatus, SelectionStatus
from agent_allocation.metrics import tables as T
from agent_allocation.metrics.outcomes import CandidateState
from agent_allocation.metrics.passk import pass_at_k
from agent_allocation.metrics.transitions import check_identity, kind_counts
from agent_allocation.paths import RunPaths
from agent_allocation.selection.seal import SealError, SealPendingWarning, load_sealed_selections, seal_selections
from tests.fixtures.metrics.synthetic_panel import (
    MISSING_FINAL, MiniGrid, build_panel, fixture_math_equivalent, select_panel, standard_items,
)


@pytest.fixture
def paths(tmp_path):
    return RunPaths.for_run(tmp_path / "run", locks_dir=tmp_path / "locks", local_state_dir=tmp_path / "local")


def _tables(paths, panel, *, strict=True, math_equivalent=fixture_math_equivalent, grades=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SealPendingWarning)       # the code item's paused sample/5, sample/6
        seal_selections(paths, panel.partition, select_panel(panel, math_equivalent=math_equivalent))
    sealed = load_sealed_selections(paths, panel.partition)
    return T.build_partition_tables(sealed, panel.records, panel.grades if grades is None else grades,
                                    strict_populations=strict)


@pytest.fixture
def standard(paths):
    panel = build_panel(standard_items())
    return panel, _tables(paths, panel)


def _rows(tables, item, name, view="current"):
    return next(r for r in tables.checkpoints if r.item_id == item and r.name == name and r.view == view)


# ------------------------------------------------------------------------------------------ pass@k and the bank


def test_bank_identities_pass1_is_mean_and_pass_bank_is_coverage(standard):
    _, tables = standard
    for item in {r.item_id for r in tables.bank}:
        rows = sorted((r for r in tables.bank if r.item_id == item), key=lambda r: r.k)
        bank = rows[0].bank_size
        assert [r.k for r in rows] == list(range(1, bank + 1))
        full = _rows(tables, item, f"sample/{bank}")
        assert rows[-1].pass_at_k_lo == full.coverage_lo and rows[-1].pass_at_k_hi == full.coverage_hi
        assert Fraction(rows[0].pass_at_k_lo_exact) == Fraction(rows[0].c_lo, bank)
        assert Fraction(rows[0].pass_at_k_hi_exact) == Fraction(rows[0].c_hi, bank)
        for r in rows:
            assert Fraction(r.pass_at_k_lo_exact) == pass_at_k(bank, r.c_lo, r.k)


def test_bank_is_the_sample_prefix_and_same_population_at_every_k(standard):
    _, tables = standard
    items = {r.item_id for r in tables.checkpoints}
    for k in range(1, 7):
        assert {r.item_id for r in tables.checkpoints if r.name == f"sample/{k}"} == items
        assert {r.item_id for r in tables.bank if r.k == k} == items
    summary = T.summarize_checkpoints(tables.checkpoints)
    for benchmark in {s["benchmark"] for s in summary}:
        sizes = {s["n_items"] for s in summary if s["benchmark"] == benchmark and s["name"].startswith("sample/")}
        assert len(sizes) == 1


def test_missing_prefix_checkpoint_is_a_population_error(paths):
    panel = build_panel(standard_items()[:2])
    item = panel.records.items[0]
    dropped = replace(item, checkpoints=tuple(c for c in item.checkpoints if c.name != "sample/3"))
    panel = replace(panel, records=replace(panel.records, items=(dropped, panel.records.items[1])))
    with pytest.raises(T.PopulationError):
        _tables(paths, panel)


def test_non_prefix_sample_pool_raises(paths):
    panel = build_panel(standard_items()[:1])
    item = panel.records.items[0]
    swapped = []
    for cp in item.checkpoints:
        if cp.name == "sample/2":
            cp = replace(cp, candidate_ids=cp.candidate_ids[::-1])
        swapped.append(cp)
    panel = replace(panel, records=replace(panel.records, items=(replace(item, checkpoints=tuple(swapped)),)))
    with pytest.raises(T.MetricsInvariantError):
        _tables(paths, panel)


# ------------------------------------------------------------------------------------------ coverage and delivery


def test_candidate_only_delivery_never_exceeds_coverage(standard):
    _, tables = standard
    for row in tables.checkpoints:
        if row.candidate_only:
            assert row.delivered_lo <= row.coverage_lo and row.delivered_hi <= row.coverage_hi
            assert 0 <= row.selection_gap_lo <= row.selection_gap_hi <= row.coverage_hi


def test_pooled_synthesis_is_exempt_and_may_beat_its_input_bank(standard):
    _, tables = standard
    pooled = _rows(tables, "hmmt/fixture-3", "pooled/4")
    assert pooled.delivered_state == "correct" and pooled.input_bank_coverage_lo == 0
    assert pooled.input_bank_coverage_hi == 0 and not pooled.candidate_only
    assert pooled.selection_gap_lo is None and pooled.selection_gap_hi is None
    archive = _rows(tables, "hmmt/fixture-3", "pooled/4", "archive")
    assert archive.candidate_only and archive.archive_coverage_lo == 1


def test_archive_coverage_is_monotone_and_nested(standard):
    _, tables = standard
    for (item, family, n) in {(r.item_id, r.family, r.n) for r in tables.checkpoints}:
        if family not in ("sample", "private_single", "private_team", "peer", "cp"):
            continue
        rows = sorted((r for r in tables.checkpoints if (r.item_id, r.family, r.n) == (item, family, n)
                       and r.view == "current"), key=lambda r: r.step)
        for a, b in zip(rows, rows[1:]):
            assert b.archive_coverage_lo >= a.archive_coverage_lo and b.archive_coverage_hi >= a.archive_coverage_hi
            assert b.archive_size >= a.archive_size


def test_current_accuracy_and_coverage_may_fall(standard):
    _, tables = standard
    delivered = [_rows(tables, "gpqa_diamond/fixture-2", f"private_single/{t}").delivered_state for t in range(1, 7)]
    assert delivered == ["wrong", "correct", "wrong", "correct", "invalid_output", "correct"]
    peer = [_rows(tables, "gpqa_diamond/fixture-2", f"peer/4/{r}").coverage_lo for r in range(3)]
    assert peer == [1, 0, 1]                                      # the last correct candidate was lost, then found
    step = next(r for r in tables.team_transitions if r.item_id == "gpqa_diamond/fixture-2" and r.family == "peer"
                and r.n == 4 and r.from_step == 0)
    assert step.coverage_lost is True and step.coverage_recovered is False


def test_unknown_grades_stay_unknown(standard):
    _, tables = standard
    row = _rows(tables, "aime/fixture-1", "private_single/4")      # EVALUATOR_ERROR on the delivered answer
    assert row.selected_state == "U" and row.delivered_state == "unknown" and row.delivered_correct is None
    assert (row.delivered_lo, row.delivered_hi) == (0, 1) and (row.coverage_lo, row.coverage_hi) == (0, 1)
    ungraded = _rows(tables, "omni_math_hard/fixture-4", "peer/4/2")
    assert ungraded.pool_n_unknown == 1
    summary = next(s for s in T.summarize_checkpoints(tables.checkpoints)
                   if s["name"] == "private_single/4" and s["benchmark"] == "aime" and s["view"] == "current")
    assert summary["delivered"] is None and summary["delivered_lo"] == 0.0 and summary["delivered_hi"] == 1.0


def test_abstention_and_invalid_output_are_distinct_not_correct_states(standard):
    _, tables = standard
    abstained = _rows(tables, "omni_math_hard/fixture-4", "sample/3")
    assert abstained.selection_status == SelectionStatus.ABSTAINED_ALL_INVALID.value
    assert abstained.delivered_state == "abstained" and abstained.selected_candidate_id is None
    assert abstained.delivered_correct is False and (abstained.delivered_lo, abstained.delivered_hi) == (0, 0)
    invalid = _rows(tables, "omni_math_hard/fixture-4", "sample/1")
    assert invalid.delivered_state == "invalid_output" and invalid.selected_parse_status == "empty_content"
    assert invalid.delivered_correct is False


def test_paused_and_missing_selections_are_blockers_not_zeros(standard):
    panel, tables = standard
    code = next(i for i in panel.records.items if i.task.item_id == "omni_math_hard/fixture-5")
    cp = next(c for c in code.checkpoints if c.name == "sample/5" and c.view.value == "current")
    checks = {cid: c for cid, c in panel.public_checks.items() if c.item_id == code.task.item_id}
    from agent_allocation.selection.readout import select_checkpoint
    assert select_checkpoint(cp, code.task, code.candidates, checks, seed=20260917).status is \
        SelectionStatus.PAUSED_CHECK_ERROR
    paused = _rows(tables, "omni_math_hard/fixture-5", "sample/5")      # a pause is never sealed: the row is a blocker
    assert paused.delivered_state == "missing_selection" and paused.delivered_correct is None
    assert (paused.delivered_lo, paused.delivered_hi) == (0, 1) and paused.selection_gap_lo == 0
    missing = _rows(tables, "omni_math_hard/fixture-4", "cp/3")
    assert missing.delivered_state == "missing_selection" and missing.pool_n_missing == 1
    summary = [s for s in T.summarize_checkpoints(tables.checkpoints) if s["name"] == "cp/3"]
    assert any(s["blocked"] for s in summary)


def test_code_rows_carry_no_text_agreement(standard):
    _, tables = standard
    row = _rows(tables, "omni_math_hard/fixture-5", "sample/4")
    assert row.pool_n_votes is None and row.pool_unanimous is None and row.public_evidence_hash
    assert row.selected_candidate_id.endswith("root/0") or row.selected_candidate_id.endswith("root/2")
    assert all(r.answer_changed is None for r in tables.slot_transitions if r.item_id == "omni_math_hard/fixture-5")


# ------------------------------------------------------------------------------------------ transitions


def test_transition_identity_holds_on_every_step(standard):
    _, tables = standard
    assert tables.team_transitions
    for row in tables.team_transitions:
        inflow = row.n_WC + row.n_IC + row.n_UC
        outflow = row.n_CW + row.n_CI + row.n_CU
        assert row.c_next - row.c_prev == row.delta_c == inflow - outflow
        if row.fully_known:
            assert row.c_next - row.c_prev == row.n_w_to_c - row.n_c_to_w
        slots = [s for s in tables.slot_transitions if (s.item_id, s.family, s.n, s.from_step) ==
                 (row.item_id, row.family, row.n, row.from_step)]
        assert len(slots) == row.n_slots


def test_exact_transition_identity_on_a_known_panel(paths):
    grid = MiniGrid(bank=4, private_cycles=2, team_sizes=(2,), rounds=1, deep=(), extension=False)
    item = standard_items()[1]
    item = replace(item, answers={"root/0": "C", "root/1": "B", "peer/2/1/0": "B", "peer/2/1/1": "C",
                                  "private/1/1": MISSING_FINAL})
    panel = build_panel([item], grid)
    tables = _tables(paths, panel)
    peer = next(r for r in tables.team_transitions if r.family == "peer")
    assert (peer.c_prev, peer.c_next, peer.n_CW, peer.n_WC, peer.fully_known) == (1, 1, 1, 1, True)
    team = next(r for r in tables.team_transitions if r.family == "private_team")
    assert (team.n_CC, team.n_WI, team.c_prev, team.c_next) == (1, 1, 1, 1)
    slot = next(s for s in tables.slot_transitions if s.family == "private_team" and s.slot == 1)
    assert slot.kind == "WI" and slot.validity_change is True and slot.answer_changed is None


def test_identity_check_rejects_inconsistent_counts():
    before = [CandidateState.CORRECT, CandidateState.WRONG]
    after = [CandidateState.WRONG, CandidateState.CORRECT]
    counts = kind_counts(zip(before, after))
    assert check_identity(before, after, counts) == 0
    with pytest.raises(ValueError):
        check_identity(before, after, dict(counts, WC=0))


def test_parse_failure_is_not_an_answer_switch(standard):
    _, tables = standard
    rows = [s for s in tables.slot_transitions if s.item_id == "gpqa_diamond/fixture-2" and s.family == "private_single"]
    to_truncated = next(s for s in rows if s.to_step == 5)
    assert to_truncated.kind == "CI" and to_truncated.answer_changed is None and to_truncated.validity_change


# ------------------------------------------------------------------------------------------ strata and failures


def test_root_strata(standard):
    _, tables = standard
    boolean = {r.n: r for r in tables.root_strata if r.item_id == "hmmt/fixture-3"}
    assert boolean[2].stratum == "none_correct" and boolean[4].stratum == "none_correct"
    option = {r.n: r for r in tables.root_strata if r.item_id == "gpqa_diamond/fixture-2"}
    assert (option[2].c0_lo, option[2].stratum) == (1, "correct_half")
    assert (option[4].c0_lo, option[4].stratum) == (2, "correct_half")
    assert option[4].reference_bank_size == 2 and option[4].reference_c_lo == 1       # roots 4..5: "K" (illegal), "(C)"


def test_failure_rows_count_truncation_and_parse_failures(standard):
    _, tables = standard
    rows = [r for r in tables.failures if r.item_id == "omni_math_hard/fixture-4"]
    roots = next(r for r in rows if r.generation_kind == "root")
    assert (roots.n_nodes, roots.n_completion_length, roots.n_parse_empty_content, roots.n_parse_missing_final) == \
        (6, 2, 2, 1)
    cp = next(r for r in rows if r.generation_kind == "communicate_then_private")
    assert cp.n_nodes == 12 and cp.n_candidates == 11
    option_roots = next(r for r in tables.failures if r.item_id == "gpqa_diamond/fixture-2" and r.generation_kind == "root")
    assert (option_roots.n_parse_valid, option_roots.n_invalid_after_valid_parse) == (6, 1)     # "K" is illegal
    assert option_roots.n_invalid_output == 1 and option_roots.n_graded_correct + option_roots.n_graded_wrong == 5


# ------------------------------------------------------------------------------------------ the seal and gold


def test_join_requires_a_verified_seal(paths):
    panel = build_panel(standard_items()[:1])
    with pytest.raises(SealError):
        T.build_partition_tables(object(), panel.records, panel.grades)  # type: ignore[arg-type]


def test_candidates_changed_after_sealing_are_refused(paths):
    panel = build_panel(standard_items()[:1])
    seal_selections(paths, "main", select_panel(panel, math_equivalent=fixture_math_equivalent))
    item = panel.records.items[0]
    root0 = panel.node_ids[(item.task.item_id, "root/0")]
    tampered = dict(item.candidates)
    tampered[root0] = replace(tampered[root0], final_answer="3")
    records = replace(panel.records, items=(replace(item, candidates=tampered),))
    with pytest.raises(T.MetricsInvariantError):
        T.build_partition_tables(load_sealed_selections(paths, "main"), records, panel.grades)


def test_selections_do_not_depend_on_gold(tmp_path):
    items = standard_items()
    flipped = [replace(i, gold=frozenset({"B"}), code_correct=frozenset()) for i in items]
    a = select_panel(build_panel(items), math_equivalent=fixture_math_equivalent)
    b = select_panel(build_panel(flipped), math_equivalent=fixture_math_equivalent)
    assert a == b


def test_contradictory_grades_raise(paths):
    panel = build_panel(standard_items()[3:4])
    item = panel.records.items[0]
    truncated = panel.node_ids[(item.task.item_id, "root/0")]
    bad = dict(panel.grades)
    bad[truncated] = replace(bad[truncated], status=GradeStatus.GRADED, semantic_correct=True)
    with pytest.raises(ValueError):
        _tables(paths, panel, grades=bad)


def test_nested_designs_can_report_population_gaps(paths):
    grid = MiniGrid(bank=3, private_cycles=3, team_sizes=(3,), rounds=2, deep=(), extension=False)
    item_a = standard_items()[1]
    item_b = replace(item_a, task=replace(item_a.task, item_id="gpqa_diamond/fixture-2b"))
    panel = build_panel([item_a, item_b], grid)
    b = panel.records.items[1]
    trimmed = replace(b, checkpoints=tuple(c for c in b.checkpoints if c.name not in ("peer/3/2",)))
    panel = replace(panel, records=replace(panel.records, items=(panel.records.items[0], trimmed)))
    with pytest.raises(T.PopulationError):
        _tables(paths, panel)
    tables = T.build_partition_tables(load_sealed_selections(paths, "main"), panel.records, panel.grades,
                                      strict_populations=False)
    gaps = T.population_gaps(tables.checkpoints)
    assert {(g["name"], g["view"]) for g in gaps} == {("peer/3/2", "current"), ("peer/3/2", "archive")}


def test_schema_lists_every_column():
    for table, row_type in T.TABLE_ROW_TYPES.items():
        assert [name for name, _ in T.TABLE_SCHEMAS[table]] == [f.name for f in row_type.__dataclass_fields__.values()]
    assert CheckpointView.ARCHIVE.value == "archive"
