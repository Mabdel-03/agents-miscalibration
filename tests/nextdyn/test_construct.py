"""The composition-driven builder and its label-free export (spec 8.2, 8.3, 10.1; plan §2.4).

The occupancy test is parametrized over the ``(n, c0)`` shapes the plan names, and each shape
is built as a *pair* -- ``(n, c0)`` and its mirror ``(n, n - c0)`` -- because the C1/C4 design
is only matched when both members sit on the same item, the same ``w*`` and the same shared
donors.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import construct as CO
from agents_scaling.nextdyn import bundles as BU
from agents_scaling.nextdyn import banks as BK
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import ids as ID
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.errors import IdentityError, ContractViolation, LeakageError
from tests.nextdyn.test_banks import bank_row, public_output, stub_item, ITEM, VARIANT

SEED = bytes.fromhex("33" * 32)
RULE = "nextdyn_construct_v1"
CORRECT_CLASS = "opt:3"
WRONG_A, WRONG_B, WRONG_C = "opt:1", "opt:5", "opt:7"


def coherent(n: int, c0: int) -> CO.Composition:
    return CO.Composition(name=f"C{c0}_OF_{n}", n=n, c0=c0,
                          wrong_pattern=CO.WrongPattern(kind=CO.PATTERN_COHERENT))


def fragmented(n: int, c0: int, counts) -> CO.Composition:
    return CO.Composition(name=f"C{c0}_FRAG_{n}", n=n, c0=c0,
                          wrong_pattern=CO.WrongPattern(kind=CO.PATTERN_FRAGMENTED,
                                                        fragmented=tuple(counts)))


def donor_bank(n_correct: int = 10, wrong: dict[str, int] | None = None):
    """``(bank rows, grades, classes)`` for one item's donor bank."""
    wrong = wrong if wrong is not None else {WRONG_A: 10, WRONG_B: 3, WRONG_C: 2}
    rows, grades, classes = [], {}, {}
    draw = 0
    for _ in range(n_correct):
        row = bank_row(BK.BankRole.DONOR, draw)
        rows.append(row)
        grades[row.generation_uid] = BK.STATE_CORRECT
        classes[row.generation_uid] = CORRECT_CLASS
        draw += 1
    for name, count in wrong.items():
        for _ in range(count):
            row = bank_row(BK.BankRole.DONOR, draw)
            rows.append(row)
            grades[row.generation_uid] = BK.STATE_WRONG
            classes[row.generation_uid] = name
            draw += 1
    for _ in range(3):                                   # unresolved draws are never donors
        row = bank_row(BK.BankRole.DONOR, draw, validity=False)
        rows.append(row)
        grades[row.generation_uid] = BK.STATE_UNKNOWN
        classes[row.generation_uid] = None
        draw += 1
    return rows, grades, classes


def classes_of(n_correct=10, wrong=None) -> CO.DonorClasses:
    rows, grades, classes = donor_bank(n_correct, wrong)
    return CO.classify_donors(rows, grades, classes, study_seed=SEED)


def drafts_for(team: CO.RootTeam) -> dict:
    return {uid: public_output(f"answer for {uid[-4:]}") for uid in team.generation_uids}


# --------------------------------------------------------------------------- compositions.yaml


def test_the_committed_compositions_compile_and_name_their_pairs():
    comps = CO.load_compositions()
    assert comps.construction_rule_version == RULE
    primary = comps.pair("primary")
    assert [c.name for c in primary] == ["C1_COHERENT", "C4_COHERENT"]
    assert [(c.n, c.c0) for c in primary] == [(5, 1), (5, 4)]
    frag = comps.composition("C1_FRAGMENTED")
    assert frag.wrong_pattern.counts(frag.n_wrong) == (2, 1, 1)
    assert comps.extension_rule.min_correct_resolved == 1
    with pytest.raises(ContractViolation):
        comps.composition("NOPE")


def test_a_composition_knows_whether_it_has_a_unique_minority_slot():
    assert coherent(5, 1).has_focal_slot and coherent(5, 1).minority_state == BK.STATE_CORRECT
    assert coherent(5, 4).has_focal_slot and coherent(5, 4).minority_state == BK.STATE_WRONG
    assert not coherent(5, 2).has_focal_slot
    assert coherent(5, 2).minority_state is None
    with pytest.raises(ContractViolation):
        CO.WrongPattern(kind=CO.PATTERN_FRAGMENTED)
    with pytest.raises(ContractViolation):
        fragmented(5, 1, (2, 1)).wrong_pattern.counts(4)


# --------------------------------------------------------------------------- donors


def test_classify_donors_keeps_only_resolved_classed_draws_and_orders_them_blind():
    dc = classes_of()
    assert dc.n_draws == 28 and dc.n_resolved == 25 and dc.n_unknown == 3
    assert dc.n_correct == 10 and dc.n_wrong == 15
    assert set(dc.correct) == {CORRECT_CLASS}
    assert {name: len(members) for name, members in dc.wrong.items()} == {WRONG_A: 10, WRONG_B: 3,
                                                                          WRONG_C: 2}
    assert dc.largest_correct == CORRECT_CLASS
    order = [d.generation_uid for d in dc.correct[CORRECT_CLASS]]
    assert order != sorted(order), "the class order is a blind key, not the draw order"
    rows, grades, classes = donor_bank()
    shuffled = CO.classify_donors(list(reversed(rows)), grades, classes, study_seed=SEED)
    assert [d.generation_uid for d in shuffled.correct[CORRECT_CLASS]] == order


def test_a_difficulty_draw_is_never_a_donor():
    rows, grades, classes = donor_bank()
    rows.append(bank_row(BK.BankRole.DIFFICULTY, 99))
    with pytest.raises(ContractViolation, match="donor bank"):
        CO.classify_donors(rows, grades, classes, study_seed=SEED)


def test_eligibility_needs_the_largest_correct_class_and_a_wrong_class():
    comps = [coherent(5, 1), coherent(5, 4)]
    rich = CO.eligibility(classes_of(), comps, {"primary": [c.name for c in comps]})
    assert rich.per_composition == {"C1_OF_5": True, "C4_OF_5": True}
    assert rich.per_pair["primary"] and rich.any_pair
    thin = CO.eligibility(classes_of(n_correct=2), comps, {"primary": [c.name for c in comps]})
    assert thin.per_composition["C1_OF_5"] and not thin.per_composition["C4_OF_5"]
    assert not thin.per_pair["primary"], "a pair needs both members"
    assert "c0=4" in thin.detail["C4_OF_5"]


def test_fragmented_eligibility_counts_the_pattern_classes():
    comp = fragmented(5, 1, (2, 1, 1))
    assert CO.composition_eligible(classes_of(), comp)[0]
    assert not CO.composition_eligible(classes_of(wrong={WRONG_A: 10, WRONG_B: 1}), comp)[0]


def test_choose_w_star_is_a_hash_and_ignores_class_size_and_order():
    dc = classes_of()
    chosen = CO.choose_w_star(dc, SEED, ITEM, min_size=2)
    assert chosen in (WRONG_A, WRONG_B, WRONG_C)
    assert CO.choose_w_star(dc, SEED, ITEM, min_size=2) == chosen
    assert CO.choose_w_star(dc, SEED, "other-item", min_size=2) is not None
    assert CO.choose_w_star(dc, SEED, ITEM, min_size=4) == WRONG_A      # only one qualifies
    assert CO.choose_w_star(dc, SEED, ITEM, min_size=99) is None


# --------------------------------------------------------------------------- the teams


@pytest.mark.parametrize("n,c0", [(3, 1), (5, 1), (5, 4), (5, 2), (9, 1), (9, 3)])
def test_c1_c4_exact_occupancy_and_shared_wrong_class(n, c0):
    """Spec 18.2: both members of a pair sit on one item, one ``w*`` and shared donors."""
    members = [coherent(n, c0), coherent(n, n - c0)]
    dc = classes_of(n_correct=n, wrong={WRONG_A: n, WRONG_B: 3, WRONG_C: 2})
    need = max(comp.wrong_pattern.counts(comp.n_wrong)[0] for comp in members)
    w_star = CO.choose_w_star(dc, SEED, ITEM, min_size=need)
    shared = CO.shared_donors(dc, members, w_star)
    teams = [CO.build_team(stub_item(), dc, comp, w_star, SEED, shared=shared, rule_version=RULE)
             for comp in members]
    for team, comp in zip(teams, members):
        states = team.slot_states
        assert len(states) == n
        assert states.count(BK.STATE_CORRECT) == comp.c0
        assert states.count(BK.STATE_WRONG) == comp.n_wrong
        wrong_classes = {m.canonical_class for m in team.members if m.state == BK.STATE_WRONG}
        assert wrong_classes == {w_star}, "the wrong members share exactly w*"
        correct_classes = {m.canonical_class for m in team.members if m.state == BK.STATE_CORRECT}
        assert correct_classes <= {CORRECT_CLASS}
        assert team.w_star_class == w_star and team.c0 == comp.c0
        CO.validate_team(team, comp)
    assert teams[0].root_team_id != teams[1].root_team_id
    assert teams[0].w_star_class == teams[1].w_star_class
    both = set(teams[0].generation_uids) & set(teams[1].generation_uids)
    assert shared.uids() <= both and both
    assert set(teams[0].shared_donor_uids) == shared.uids() & set(teams[0].generation_uids)


def test_no_donor_record_duplicated_within_team():
    """Spec 18.2: four copies of one record are not four draws."""
    dc = classes_of()
    comp = coherent(5, 1)
    team = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED,
                         shared=CO.SharedDonors(), rule_version=RULE)
    uids = team.generation_uids
    assert len(set(uids)) == len(uids) == 5
    assert len({m.decision_id for m in team.members}) == 5
    import dataclasses

    duplicated = dataclasses.replace(team, members=(team.members[0],) * 5)
    # Two layers refuse it, and the test keeps them apart: construct refuses to *validate* a team
    # holding one record twice (ContractViolation), and identity refuses to *name* one, because a
    # root_team_id is content-addressed over its ordered member generation uids (IdentityError).
    with pytest.raises(ContractViolation, match="twice"):
        CO.validate_team(duplicated, comp)
    with pytest.raises(IdentityError):
        ID.root_team_id(ID.CellKind.DONOR_BANK, ITEM, VARIANT, ["g:a", "g:a"], None, RULE)


def test_same_content_does_not_imply_same_sample():
    """Spec 18.1: two draws with identical text are still two independent members."""
    rows, grades, classes = donor_bank()
    same = [tables.row_from_dict(tables.BankRow,
                                 dict(tables.row_to_dict(row), content_hash="c:" + "9" * 64))
            for row in rows]
    dc = CO.classify_donors(same, grades, classes, study_seed=SEED)
    team = CO.build_team(stub_item(), dc, coherent(5, 1), WRONG_A, SEED,
                         shared=CO.SharedDonors(), rule_version=RULE)
    assert len({m.content_hash for m in team.members}) == 1
    assert len(set(team.generation_uids)) == 5
    CO.validate_team(team, coherent(5, 1))


def test_the_focal_slot_is_the_unique_minority_slot_or_none():
    dc = classes_of()
    for comp, state in ((coherent(5, 1), BK.STATE_CORRECT), (coherent(5, 4), BK.STATE_WRONG)):
        team = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=CO.SharedDonors(),
                             rule_version=RULE)
        assert team.focal_slot is not None
        assert team.slot_states[team.focal_slot] == state
        assert sum(1 for s in team.slot_states if s == state) == 1
    balanced = CO.build_team(stub_item(), dc, coherent(5, 2), WRONG_A, SEED,
                            shared=CO.SharedDonors(), rule_version=RULE)
    assert balanced.focal_slot is None


def test_the_slot_permutation_is_blind_and_the_team_id_is_content_addressed():
    dc = classes_of()
    comp = coherent(5, 1)
    first = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=CO.SharedDonors(),
                          rule_version=RULE)
    again = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=CO.SharedDonors(),
                          rule_version=RULE)
    assert first.root_team_id == again.root_team_id == first.root_team_id
    assert first.construction_hash == again.construction_hash
    other_seed = CO.build_team(stub_item(), dc, comp, WRONG_A, bytes.fromhex("44" * 32),
                               shared=CO.SharedDonors(), rule_version=RULE)
    assert other_seed.root_team_id != first.root_team_id
    assert first.root_team_id.startswith("team:") and len(first.team_tag) == 16


def test_fragmented_teams_spread_the_wrong_members_over_the_pattern_classes():
    dc = classes_of()
    comp = fragmented(5, 1, (2, 1, 1))
    team = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=CO.SharedDonors(),
                         rule_version=RULE)
    wrong = [m.canonical_class for m in team.members if m.state == BK.STATE_WRONG]
    assert sorted(wrong.count(name) for name in set(wrong)) == [1, 1, 2]
    assert wrong.count(WRONG_A) == 2
    CO.validate_team(team, comp)


def test_a_natural_team_takes_the_draws_in_order_with_a_blind_protection_slot():
    rows = [bank_row(BK.BankRole.NATURAL_ROOT, draw) for draw in range(5)]
    team = CO.natural_team(stub_item(), list(reversed(rows)), SEED, rule_version=RULE)
    assert [m.draw_index for m in team.members] == [0, 1, 2, 3, 4]
    assert team.cohort == CO.COHORT_NATURAL and team.composition == CO.NATURAL_COMPOSITION
    assert 0 <= team.focal_slot < 5
    assert CO.natural_team(stub_item(), rows, SEED, rule_version=RULE).root_team_id == team.root_team_id
    with pytest.raises(ContractViolation):
        CO.natural_team(stub_item(), [bank_row(BK.BankRole.DONOR, 0)], SEED, rule_version=RULE)


# --------------------------------------------------------------------------- export and seal


def test_solver_export_contains_no_gold_or_construction_correctness():
    """Spec 18.1: the bundle a solver reads carries no composition, no c0, no grade."""
    dc = classes_of()
    comp = coherent(5, 1)
    team = CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=CO.SharedDonors(),
                         rule_version=RULE)
    payload = CO.export_root_bundle(team, drafts_for(team))
    text = json.dumps(payload).casefold()
    for forbidden in ("composition", "c1_of_5", "w_star", "opt:", "semantic_correct",
                      "canonical_class", "is_correct", "gold"):
        assert forbidden not in text, forbidden
    assert set(payload) == set(BU.BUNDLE_TEAM_FIELDS)
    assert payload["team_tag"] == team.team_tag and team.composition not in text
    BU.assert_label_free(payload)
    with pytest.raises(LeakageError):
        BU.assert_label_free(dict(payload, composition="C1_COHERENT"))
    with pytest.raises(LeakageError):
        BU.assert_label_free(dict(payload, note="built as C1_COHERENT"))


def test_candidate_ids_and_slot_labels_do_not_encode_correctness():
    """Spec 18.1: slot numbers and ids carry no correctness signal."""
    focal_slots = set()
    for index in range(12):
        dc = classes_of(n_correct=10, wrong={WRONG_A: 10, WRONG_B: 3, WRONG_C: 2})
        item = type("Item", (), {"family": F.MATH500})()
        team = CO.build_team(item, dc, coherent(5, 1), WRONG_A,
                             bytes([index]) * 32, shared=CO.SharedDonors(), rule_version=RULE)
        payload = CO.export_root_bundle(team, drafts_for(team))
        assert [slot["slot"] for slot in payload["slots"]] == [0, 1, 2, 3, 4]
        # Slots are emitted in index order, never in correctness order: the exported sequence
        # is identical in shape whatever the composition, so position reveals nothing.  (The
        # team_tag is a hex digest and necessarily contains decimal digits, so "the slot index
        # does not appear in the tag" is not the property to assert -- the property is that the
        # export is ordered by slot and that the minority slot moves between teams.)
        assert payload["slots"] == sorted(payload["slots"], key=lambda slot: slot["slot"])
        focal_slots.add(team.focal_slot)
        # the correct member is not systematically first
        assert team.slot_states.index(BK.STATE_CORRECT) == team.focal_slot
    assert len(focal_slots) > 1, "the minority slot never moved"
    assert focal_slots - {0}, "the minority slot is always index 0"


def test_seal_root_teams_writes_a_bucket_once_and_never_mutates_it(tmp_run_root):
    dc = classes_of()
    members = [coherent(5, 1), coherent(5, 4)]
    shared = CO.shared_donors(dc, members, WRONG_A)
    teams = [CO.build_team(stub_item(), dc, comp, WRONG_A, SEED, shared=shared,
                           rule_version=RULE, bucket="0007") for comp in members]
    drafts = {}
    for team in teams:
        drafts.update(drafts_for(team))
    export_hash, public, private = CO.seal_root_teams(tmp_run_root, F.MATH500, "0007", teams,
                                                      drafts, rule_version=RULE)
    assert len(public) == len(private) == 2
    assert {row.export_hash for row in public} == {export_hash}
    assert {row.bucket for row in public} == {"0007"}
    assert json.loads(public[0].ordered_generation_uids_json) == list(teams[0].generation_uids)
    assert public[0].n_independent_generations == 5
    assert private[0].composition == "C1_OF_5" and private[0].c0 == 1
    assert json.loads(private[0].slot_states_json) == list(teams[0].slot_states)
    # the same content re-seals; different content is refused
    assert CO.seal_root_teams(tmp_run_root, F.MATH500, "0007", teams, drafts,
                              rule_version=RULE)[0] == export_hash
    from agents_scaling.nextdyn.errors import ManifestError

    with pytest.raises(ManifestError):
        CO.seal_root_teams(tmp_run_root, F.MATH500, "0007", teams[:1], drafts, rule_version=RULE)


def test_recruitment_flow_counts_the_screen_by_family():
    records = [CO.RecruitmentRecord(source_item_id=f"i{i}", family=F.MATH500, screened=True,
                                    resolved=25, eligible_initial=i % 2 == 0, extended=i % 2 == 1,
                                    eligible_extended=i % 3 == 0, paired=i % 2 == 0,
                                    fragmented=i % 4 == 0, single=False)
               for i in range(8)]
    flow = CO.recruitment_flow(records, stop_reason="quota_reached")
    assert flow["stop_reason"] == "quota_reached"
    counts = flow["families"][F.MATH500]
    assert counts["screened"] == 8 and counts["eligible_32"] == 4 and counts["extended"] == 4
    assert counts["paired"] == 4 and counts["fragmented"] == 2


# --------------------------------------------------------------------------- the worker


class StubConstructRunner:
    def __init__(self, run_root, study_cfg, data):
        self.run_root = run_root
        self.contract = study_cfg.contract
        self.compositions = CO.load_compositions()
        self.cell = type("Cell", (), {"cell_id": "construct/test/0000", "kind": "construct"})()
        self._data = data

    def donor_input(self, source_item_id):
        assert source_item_id == ITEM
        return self._data


def test_the_construct_worker_seals_a_pair_and_keeps_the_labels_protected(tmp_run_root, study_cfg):
    rows, grades, classes = donor_bank()
    dc = CO.classify_donors(rows, grades, classes, study_seed=study_cfg.contract.study_seed)
    drafts = {row.generation_uid: public_output(f"a{index}") for index, row in enumerate(rows)}
    data = CO.DonorInput(item=stub_item(), bank_rows=tuple(rows), grades=grades, classes=classes,
                         drafts=drafts, bucket="0000", pair="primary")
    runner = StubConstructRunner(tmp_run_root, study_cfg, data)
    item_file = CO.construct_worker(runner, ITEM)
    public = item_file.rows["root_teams"]
    private = item_file.private_rows["root_teams_private"]
    assert len(public) == len(private) == 2
    assert item_file.bundle_export_hash
    assert {row.cohort for row in public} == {CO.COHORT_CONSTRUCTED}
    del dc
    # the runtime reader sees both members and no label
    loaded = BU.load_bucket(tmp_run_root, stub_item().family, "0000")
    assert {team.root_team_id for team in loaded} == {row.root_team_id for row in public}
    assert all(team.n == 5 for team in loaded)


def test_an_ineligible_item_seals_nothing_and_is_not_a_failure(tmp_run_root, study_cfg):
    rows, grades, classes = donor_bank(n_correct=1, wrong={WRONG_A: 1})
    drafts = {row.generation_uid: public_output("a") for row in rows}
    data = CO.DonorInput(item=stub_item(), bank_rows=tuple(rows), grades=grades, classes=classes,
                         drafts=drafts, bucket="0001", pair="primary")
    item_file = CO.construct_worker(StubConstructRunner(tmp_run_root, study_cfg, data), ITEM)
    assert item_file.rows["root_teams"] == ()
    assert item_file.private_rows["root_teams_private"] == ()
    assert item_file.bundle_export_hash is None
    assert item_file.status is tables.ItemStatus.COMPLETE
