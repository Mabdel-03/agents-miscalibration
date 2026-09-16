"""WP5: sealed pools and the five selectors (spec 6.5, 15.1; spec 18.3 names).

Selection is where a study most easily fools itself.  Three failure modes are tested here
because each one silently *improves* a reported number: a selector that could not run being
scored as 0% (it should be ``unexecuted``), a tie broken by something correlated with the
treatment (the arm order, the agent order), and a deployable selector that quietly saw a label
(it should be structurally unable to).
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from agents_scaling.nextdyn import audit as AU
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import selectors as S
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.ids import CellKind
from tests.nextdyn.test_mod97 import CONFIG_DIR

SEED = b"selection-seed-0123456789abcdef"
ITEM = "nditem0000000001"


def candidate(i: int, cls: str | None, *, confidence: float | None = None,
              packet: str | None = None) -> S.Candidate:
    """A candidate with class ``cls``; ``cls=None`` is the ungradeable (invalid) candidate."""
    return S.Candidate(candidate_id=f"d:cand{i}", generation_uid=f"g:{i}", content_hash=f"c:{i}",
                       canonical_class=cls, valid=cls is not None, confidence=confidence,
                       packet_hash=packet or f"pkt{i}", slot=i)


def pool(*classes, scope: str = "terminal_only", protocol_hash: str = "cond-a",
         episode_id: str = "episode|t|c|m|b|rep0", **kwargs) -> S.SealedPool:
    candidates = tuple(candidate(i, cls, **kwargs) for i, cls in enumerate(classes))
    return S.SealedPool(source_item_id=ITEM, episode_id=episode_id, cell_kind=CellKind.EPISODE,
                        round_=4, scope=scope, candidates=candidates, protocol_hash=protocol_hash)


def ctx(**kwargs) -> S.SelectionContext:
    kwargs.setdefault("study_seed", b"study")
    kwargs.setdefault("selection_seed", SEED)
    return S.SelectionContext(**kwargs)


# --------------------------------------------------------------------------- spec 18.3


def test_current_selector_accuracy_not_above_current_oracle_on_same_pool():
    """Spec 18.3.  On every pool, the deployable selector can never beat the analysis-only
    oracle -- the oracle ships a correct candidate whenever one exists, so any pool where
    plurality is right is a pool where the oracle is right.

    A run of random pools is worth more than a hand-picked one here: the invariant has to hold
    for ties, all-invalid pools and singleton classes alike.
    """
    rng = random.Random(20260916)
    plurality = S.selector_for(S.CURRENT_ROUND_PLURALITY)
    oracle = S.selector_for(S.ORACLE)
    wins = {"plurality": 0, "oracle": 0}
    for _ in range(300):
        n = rng.randint(2, 6)
        classes = [rng.choice(["A", "B", "C", None]) for _ in range(n)]
        sealed = pool(*classes)
        truth = {c.candidate_id: ("C" if c.canonical_class == "A" else "W")
                 for c in sealed.valid_candidates}
        labels = S.AnalysisOnlyContext(labels=truth)
        deployed = plurality.select(sealed, ctx())
        ceiling = oracle.select(sealed, ctx(analysis_only=labels))
        hit_d = deployed.selected_id is not None and truth.get(deployed.selected_id) == "C"
        hit_o = ceiling.selected_id is not None and truth.get(ceiling.selected_id) == "C"
        wins["plurality"] += hit_d
        wins["oracle"] += hit_o
        assert not (hit_d and not hit_o), (classes, deployed, ceiling)
    assert wins["oracle"] >= wins["plurality"] > 0


def test_tie_break_is_invariant_to_protocol_and_agent_order():
    """Spec 18.3.  The tie key reads ``(source_item_id, canonical_class, selection_seed)`` and
    nothing else, so shuffling the candidates or relabelling the arm cannot move the winner.

    ARD36 broke ties on a representative candidate id, which meant two arms that produced the
    same classes in a different slot order could disagree -- a treatment effect manufactured by
    the tie-breaker.
    """
    classes = ["A", "A", "B", "B"]
    base = pool(*classes)
    first = S.selector_for(S.CURRENT_ROUND_PLURALITY).select(base, ctx())
    assert first.tie_broken and set(first.tie_class_ids) == {"A", "B"}

    rng = random.Random(7)
    for arm in ("cond-a", "cond-b", "cond-c"):
        shuffled = list(classes)
        rng.shuffle(shuffled)
        other = pool(*shuffled, protocol_hash=arm, episode_id=f"episode|t|{arm}|m|b|rep0")
        again = S.selector_for(S.CURRENT_ROUND_PLURALITY).select(other, ctx())
        assert again.selected_class == first.selected_class
    assert S.tie_break(ITEM, ["A", "B"], SEED) == S.tie_break(ITEM, ["B", "A"], SEED)
    assert S.tie_break(ITEM, ["A", "B"], SEED) != S.tie_break("other-item", ["A", "B"], SEED)


def test_selector_returns_existing_permitted_candidate():
    """Spec 18.3.  Whatever a selector returns is a member of the pool it was given, and is a
    *valid* member: a selector may not ship an answer nobody generated."""
    sealed = pool("A", "B", None, "A")
    for name, extra in ((S.CURRENT_ROUND_PLURALITY, {}), (S.RANDOM_VALID, {}),
                        (S.ORACLE, {"analysis_only": S.AnalysisOnlyContext(labels={})})):
        selection = S.selector_for(name).select(sealed, ctx(**extra))
        assert selection.selected_id in {c.candidate_id for c in sealed.valid_candidates}
        assert sealed.candidate_of(selection.selected_id).valid


def test_missing_or_unfitted_selector_is_unexecuted_not_zero_accuracy():
    """Spec 18.3.  Three different "could not run" cases, three ``unexecuted`` statuses and no
    selected candidate -- never a wrong answer that would drag the reported accuracy down."""
    sealed = pool("A", "B", confidence=None)
    unfitted = S.ConfidenceWeights(mapping="task_dev_root_success_rate",
                                   fallback="global_dev_base_rate", min_task_support=10,
                                   missing_confidence_weight="global_dev_base_rate",
                                   complete_confidence_sensitivity=True)
    weighted = S.selector_for(S.CONFIDENCE_WEIGHTED).select(sealed, ctx(confidence_weights=unfitted))
    audit = S.selector_for(S.POINTWISE_AUDIT).select(sealed, ctx())
    oracle = S.selector_for(S.ORACLE).select(sealed, ctx())
    for selection in (weighted, audit, oracle):
        assert selection.fallback_status == S.FALLBACK_UNEXECUTED
        assert selection.selected_id is None and selection.failed is True
    assert S.selector_for(S.CONFIDENCE_WEIGHTED).select(
        sealed, ctx(confidence_weights=None)).fallback_status == S.FALLBACK_UNEXECUTED


def test_selector_service_cannot_read_private_grades():
    """Spec 18.1/18.3.  The import closure of ``selectors`` and ``audit`` never reaches
    ``grading``, the datasets or the item builder, so a deployable selector has no code path to
    a label at all; the oracle gets one only through an explicit analysis-only envelope that
    nothing in the runtime constructs.
    """
    forbidden = {"grading", "datasets", "items_build", "inventory", "construct", "e0", "analysis"}
    package = Path(S.__file__).resolve().parent
    seen, frontier = set(), ["selectors", "audit"]
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        path = package / f"{module}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                if not name.startswith("agents_scaling.nextdyn"):
                    continue
                leaf = name.split(".")[2] if len(name.split(".")) > 2 else ""
                assert leaf not in forbidden, f"{module} reaches {name} (R7)"
                if leaf:
                    frontier.append(leaf)
    assert {"selectors", "audit"} <= seen
    # and the oracle is not deployable, so a deployed list naming it is refused
    assert S.selector_for(S.ORACLE).deployable is False


def test_a_second_generative_selector_dispatches_without_editing_audit_py():
    """Spec 18.3 / plan §1.10.  A new generative selector is one registry entry: the audit cell
    finds it through ``needs_cells`` and gives it its own cell id and selection id."""

    class SecondVerifier(S._BaseSelector):
        name = "pairwise_audit"
        needs_cells = True

        def select(self, sealed, context):
            return S.Selection(selector=self.name, version=self.version, pool_id=sealed.pool_id,
                               selected_id=sealed.candidates[0].candidate_id,
                               selected_class=sealed.candidates[0].canonical_class,
                               tie_class_ids=(), fallback_status=S.FALLBACK_NONE, cost={},
                               failed=False)

    registry = {**S.SELECTORS, SecondVerifier.name: SecondVerifier()}
    assert AU.generative_selectors(registry) == (S.POINTWISE_AUDIT, SecondVerifier.name)
    sealed = pool("A", "B")
    made = AU.selections_for([sealed], {}, {}, ctx=ctx(), registry=registry)
    by_selector = {s.selector for s in made}
    assert by_selector == {S.POINTWISE_AUDIT, SecondVerifier.name}
    ids = {AU.audit_cell_id(s.selector, s.version, sealed.pool_id) for s in made}
    assert len(ids) == 2 and all(f"{s.selector}.{s.version}" in s.selection_id for s in made)

    # scores are per selector: two verifiers that scored the same packet made two different
    # measurements, and handing one the other's numbers would report the wrong verifier
    scores = {S.POINTWISE_AUDIT: {c.packet_hash: 0.9 for c in sealed.valid_candidates},
              SecondVerifier.name: {}}
    routed = {s.selector: s for s in
              AU.selections_for([sealed], scores, {}, ctx=ctx(), registry=registry)}
    assert routed[S.POINTWISE_AUDIT].fallback_status == S.FALLBACK_NONE
    assert routed[SecondVerifier.name].fallback_status == S.FALLBACK_NONE

    # binding is per selector too: only the pointwise entry takes this cell's concrete pair,
    # and a selector that brought its own is returned untouched (the request and parse path is
    # driven end to end by
    # ``test_a_second_generative_selector_is_asked_for_its_own_requests_and_parser``)
    bound = AU.bound_selector(S.POINTWISE_AUDIT, request_builder=lambda *a: {},
                              parser=lambda r: None, registry=registry)
    assert isinstance(bound, S.PointwiseAudit) and bound is not registry[S.POINTWISE_AUDIT]
    assert AU.bound_selector(SecondVerifier.name, request_builder=lambda *a: {},
                             parser=lambda r: None, registry=registry) is registry[SecondVerifier.name]


# --------------------------------------------------------------------------- pools


def test_an_all_invalid_pool_is_no_valid_candidate_not_a_dropped_item():
    """D9.  The opportunity stays in the denominator and is scored incorrect downstream."""
    sealed = pool(None, None)
    for name in (S.CURRENT_ROUND_PLURALITY, S.RANDOM_VALID):
        selection = S.selector_for(name).select(sealed, ctx())
        assert selection.fallback_status == S.FALLBACK_NO_CANDIDATE
        assert selection.selected_id is None and selection.selected_class is None
    row = S.selection_row(sealed, S.selector_for(S.CURRENT_ROUND_PLURALITY).select(sealed, ctx()))
    assert row.valid_count == 0 and row.selected_decision_id is None


def test_pool_id_is_content_addressed_and_a_reordered_pool_is_a_different_pool():
    first = pool("A", "B", "A")
    assert first.pool_id == pool("A", "B", "A").pool_id
    assert first.pool_id != pool("B", "A", "A").pool_id
    assert first.pool_id.startswith(S.POOL_ID_PREFIX)
    with pytest.raises(ContractViolation):
        S.SealedPool(source_item_id=ITEM, episode_id="e", cell_kind=CellKind.EPISODE, round_=4,
                     scope="terminal_only", candidates=(), protocol_hash="c")


def test_a_candidate_cannot_carry_a_correctness_field():
    """R7 in the type system: there is nowhere on a candidate to put a label, and an invalid
    candidate may not carry a voting class either."""
    assert "correct" not in {f for f in S.Candidate.__dataclass_fields__}
    with pytest.raises(ContractViolation):
        S.Candidate(candidate_id="d:1", generation_uid="g", content_hash=None,
                    canonical_class="A", valid=False, confidence=None, packet_hash="p", slot=0)


# --------------------------------------------------------------------------- confidence


def test_a_missing_confidence_is_never_imputed_as_a_model_statement(analysis_cfg):
    """Spec 6.3 / plan §1.9.  The policy comes from ``analysis.yaml``; a candidate with no
    reported confidence keeps ``confidence=None`` and contributes the *fallback rate* as a
    ranking weight -- or is dropped under the declared complete-case sensitivity."""
    profile = analysis_cfg.profile
    weights = S.ConfidenceWeights.from_profile(profile, task_rate=0.42, task_support=40,
                                               base_rate=0.30)
    assert (weights.mapping, weights.fallback) == ("task_dev_root_success_rate",
                                                   "global_dev_base_rate")
    assert weights.complete_confidence_sensitivity is True
    assert weights.prior() == pytest.approx(0.42)
    assert weights.weight_of(None) == pytest.approx(0.42)

    complete = S.ConfidenceWeights.from_profile(profile, task_rate=0.42, task_support=40,
                                                base_rate=0.30, complete_cases_only=True)
    assert complete.weight_of(None) is None

    sealed = pool("A", "A", "B", confidence=None)
    selection = S.selector_for(S.CONFIDENCE_WEIGHTED).select(
        sealed, ctx(confidence_weights=complete))
    assert selection.fallback_status == S.FALLBACK_UNEXECUTED
    assert selection.selected_id is None
    assert all(c.confidence is None for c in sealed.candidates)

    # "nobody reported a confidence" is a statement about this selector, not about the pool:
    # ``no_valid_candidate`` is scored incorrect downstream and would charge the sensitivity
    # arm with a wrong answer on every such item, while ``unexecuted`` leaves the opportunity
    # out of its numerator.  A row can never say both at once.
    row = S.selection_row(sealed, selection)
    assert row.valid_count == 3
    assert not (row.valid_count and row.fallback_status == S.FALLBACK_NO_CANDIDATE)
    for classes in (("A", "B"), ("A", None), (None, None), ("A", "A", "B")):
        other = pool(*classes, confidence=None)
        status = S.selector_for(S.CONFIDENCE_WEIGHTED).select(
            other, ctx(confidence_weights=complete)).fallback_status
        assert (status == S.FALLBACK_NO_CANDIDATE) == (not other.valid_candidates)
    # an all-invalid pool is still no_valid_candidate, complete-case arm or not
    assert S.selector_for(S.CONFIDENCE_WEIGHTED).select(
        pool(None, None), ctx(confidence_weights=complete)).fallback_status == \
        S.FALLBACK_NO_CANDIDATE


def test_below_min_task_support_the_global_dev_base_rate_is_used(analysis_cfg):
    profile = analysis_cfg.profile
    thin = S.ConfidenceWeights.from_profile(profile, task_rate=0.9,
                                            task_support=profile.confidence_weighted.min_task_support - 1,
                                            base_rate=0.25)
    assert thin.prior() == pytest.approx(0.25)
    thick = S.ConfidenceWeights.from_profile(profile, task_rate=0.9,
                                             task_support=profile.confidence_weighted.min_task_support,
                                             base_rate=0.25)
    assert thick.prior() == pytest.approx(0.9)


def test_confidence_weighting_can_outvote_a_bare_plurality(analysis_cfg):
    """The selector is a real competitor, not a relabelled plurality: two diffident votes lose
    to one confident vote once the reported values are mapped, and two confident ones win."""
    weights = S.ConfidenceWeights.from_profile(analysis_cfg.profile, task_rate=0.5,
                                               task_support=50, base_rate=0.5)
    sealed = S.SealedPool(
        source_item_id=ITEM, episode_id="episode|t|c|m|b|rep0", cell_kind=CellKind.EPISODE,
        round_=4, scope="terminal_only", protocol_hash="cond-a",
        candidates=(candidate(0, "A", confidence=0.05), candidate(1, "A", confidence=0.05),
                    candidate(2, "B", confidence=1.0)))
    assert S.selector_for(S.CURRENT_ROUND_PLURALITY).select(sealed, ctx()).selected_class == "A"
    assert S.selector_for(S.CONFIDENCE_WEIGHTED).select(
        sealed, ctx(confidence_weights=weights)).selected_class == "B"
    # ... and a class that is both larger and confident keeps the vote
    louder = S.SealedPool(
        source_item_id=ITEM, episode_id="episode|t|c|m|b|rep0", cell_kind=CellKind.EPISODE,
        round_=4, scope="terminal_only", protocol_hash="cond-a",
        candidates=(candidate(0, "A", confidence=0.8), candidate(1, "A", confidence=0.8),
                    candidate(2, "B", confidence=1.0)))
    assert S.selector_for(S.CONFIDENCE_WEIGHTED).select(
        louder, ctx(confidence_weights=weights)).selected_class == "A"


# --------------------------------------------------------------------------- the registry


def test_the_deployed_list_comes_from_study_yaml_and_refuses_the_oracle():
    contract = C.contract_of(C.load_study(CONFIG_DIR / "study.yaml"))
    assert S.deployed_selectors(contract.selection) == (
        "current_round_plurality", "random_valid", "confidence_weighted", "pointwise_audit")
    leaky = type(contract.selection)(
        **{**{f: getattr(contract.selection, f) for f in contract.selection.__dataclass_fields__},
           "secondary_deployed": (S.ORACLE,)})
    with pytest.raises(LeakageError):
        S.deployed_selectors(leaky)


def test_a_failed_audit_score_falls_back_to_the_preregistered_vote():
    """D9.  "Verifier did not answer" and "candidate is wrong" are different facts: the
    fallback is a typed status carrying the plurality answer, not a zero score."""
    sealed = pool("A", "A", "B")
    scores = {sealed.candidates[0].packet_hash: 0.9}
    selection = S.selector_for(S.POINTWISE_AUDIT).select(
        sealed, ctx(scores=scores, score_failures={sealed.candidates[2].packet_hash: "no_json"}))
    assert selection.fallback_status == S.FALLBACK_AUDIT
    assert selection.selected_class == "A"


def test_a_complete_audit_score_set_picks_the_best_scored_class():
    sealed = pool("A", "A", "B")
    scores = {c.packet_hash: value for c, value in zip(sealed.candidates, (0.1, 0.1, 0.95))}
    selection = S.selector_for(S.POINTWISE_AUDIT).select(sealed, ctx(scores=scores))
    assert selection.selected_class == "B" and selection.fallback_status == S.FALLBACK_NONE


def test_the_runtime_iterates_selector_objects_from_the_contract():
    """R2: ``StudyContract.selection`` is what the runtime deploys, and it deploys objects --
    so the "no analysis-only selector in a deployed list" rule is one function, not two."""
    contract = C.contract_of(C.load_study(CONFIG_DIR / "study.yaml"))
    deployed = S.deployed_for(contract)
    assert [s.name for s in deployed] == list(S.deployed_selectors(contract.selection))
    assert all(s.deployable for s in deployed)
    assert S.selector_for(contract.selection.primary) is deployed[0]
    assert sum(1 for s in deployed if s.needs_cells) == 1


# --------------------------------------------------------------------------- spec 18.3 pass@k
#
# The pass@k panel is a *selection* claim, not a sampling one: oracle@k is the ceiling a
# perfect selector would reach on a fixed set of draws and vote@k is what the deployable one
# reaches on the **same** draws, so the gap between them is a property of selection.  Both
# names below are spec-18.3 names assigned to this module by the plan (§5), and both read the
# committed WP7 implementation (``panels.passk_population`` / ``analysis.passk_panel``) rather
# than restating its arithmetic here.


def passk_context(bank_rows, fresh_rows, grades, study_cfg, analysis_cfg):
    """An ``AnalysisContext`` over hand-built rows: the frame is the only thing under test."""
    from pathlib import Path

    from agents_scaling.nextdyn import analysis as AN
    from agents_scaling.nextdyn import panels as PN
    from agents_scaling.nextdyn import stats as ST

    return AN.AnalysisContext(
        run_root=Path("/nonexistent"), cfg=study_cfg, analysis=analysis_cfg,
        settings=ST.settings(analysis_cfg.profile, study_cfg.analysis),
        frame=PN.PanelFrame(agent_rounds=tuple(fresh_rows), banks=tuple(bank_rows)),
        grades=grades)


def passk_draws(item, *, states, bank_role=None, label=None, n_roots=5, model="m1", budget="b1"):
    """``n_roots`` bank draws plus the rest as FRESH decisions, all of one item.

    ``states`` is the tri-state of each of the 25 draws **in blind order** -- the order
    ``passk_panel`` puts them in, which is ``sorted(generation_uid)``.
    """
    from agents_scaling.nextdyn import panels as PN

    bank, fresh, grades = [], [], {}
    for index, state in enumerate(states):
        uid = f"g:{item}:{index:03d}"
        row = {"source_item_id": item, "generation_uid": uid, "model_hash": model,
               "budget_hash": budget}
        grades[uid] = {"generation_uid": uid, "semantic_correct": state,
                       "canonical_class": None if state == "U" else f"cls-{state}"}
        if index < n_roots:
            bank.append({**row, "bank_role": bank_role or PN.BANK_ROLE_NATURAL_ROOT})
        else:
            fresh.append({**row, "display_label": label or PN.PASSK_FRESH_LABEL})
    return bank, fresh, grades


def test_passk_uses_fixed_items_and_correct_bank_role(study_cfg, analysis_cfg):
    """Spec 18.3.  An item joins the pass@k panel only with a *complete, comparable* bank:
    exactly the declared natural-root draws in the ``natural_root`` role, exactly the FRESH
    decisions that complete the 25, and one model and one budget across all of them.

    The role matters because difficulty and donor draws exist for the same items at the same
    model and budget; counting them would silently change what pass@k is measured over.  And
    the panel is one fixed item set for **every** k -- pass@1 and pass@25 are two readings of
    one set of draws, not two populations.
    """
    from agents_scaling.nextdyn import analysis as AN
    from agents_scaling.nextdyn import panels as PN

    n_roots = int(study_cfg.cohorts.natural_root_draws)
    good = ["C"] + ["W"] * (PN.PASSK_DRAWS - 1)
    banks, fresh, grades = passk_draws("item-ok", states=good, n_roots=n_roots)

    # the wrong bank role: same item shape, draws that are not natural roots
    b2, f2, g2 = passk_draws("item-role", states=good, n_roots=n_roots,
                             bank_role=PN.BANK_ROLE_NATURAL_ROOT + "_not")
    # an incomplete FRESH set
    b3, f3, g3 = passk_draws("item-short", states=good, n_roots=n_roots)
    f3 = f3[:-1]
    # draws that span two models: not one comparable bank
    b4, f4, g4 = passk_draws("item-mixed", states=good, n_roots=n_roots)
    f4 = [{**row, "model_hash": "m2"} for row in f4]

    ctx = passk_context(banks + b2 + b3 + b4, fresh + f2 + f3 + f4,
                        {**grades, **g2, **g3, **g4}, study_cfg, analysis_cfg)
    panel = PN.passk_population(ctx.frame, ctx.cfg)
    assert panel.keys == frozenset({"item-ok"})
    assert panel.unit == PN.UNIT_ITEM
    assert panel.detail["n_roots_required"] == n_roots
    assert panel.detail["n_fresh_required"] == PN.PASSK_DRAWS - n_roots
    assert sorted(panel.detail["rejected_examples"]) == ["item-mixed", "item-role", "item-short"]

    rows = AN.passk_panel(ctx)
    ks = PN.passk_ks(analysis_cfg.profile)
    assert {r["k"] for r in rows} == set(ks)
    for k in ks:                                   # one fixed item set at every k
        assert {r["source_item_id"] for r in rows if r["k"] == k} == {"item-ok"}
    assert {r["n"] for r in rows} == {PN.PASSK_DRAWS}


def test_passk_prefix_oracle_and_vote_use_same_actual_candidates(study_cfg, analysis_cfg):
    """Spec 18.3.  oracle@k and vote@k read **one** blind-ordered prefix of the same draws.

    If they read different prefixes -- or if oracle@k were computed over the whole bank while
    vote@k saw only k -- the reported selection gap would be an artefact of two samplings
    rather than the thing it is meant to measure.  The bank below is built so the two readings
    have to disagree in a known way: the first draws are a wrong majority and the only correct
    draw is late, so no prefix that excludes it can be right for either selector, and every
    prefix that includes it is right for the oracle and still wrong for the vote.
    """
    from agents_scaling.nextdyn import analysis as AN
    from agents_scaling.nextdyn import panels as PN

    n_roots = int(study_cfg.cohorts.natural_root_draws)
    correct_at = 20
    states = ["W"] * PN.PASSK_DRAWS
    states[correct_at] = "C"
    banks, fresh, grades = passk_draws("item-gap", states=states, n_roots=n_roots)
    ctx = passk_context(banks, fresh, grades, study_cfg, analysis_cfg)

    rows = {row["k"]: row for row in AN.passk_panel(ctx)}
    assert rows and set(rows) == set(PN.passk_ks(analysis_cfg.profile))
    # the declared k values must straddle the correct draw, or the contrast below is vacuous
    assert min(rows) <= correct_at < max(rows)
    for k, row in rows.items():
        assert row["prefix_size"] == min(k, PN.PASSK_DRAWS)
        reachable = k > correct_at
        assert row["oracle_at_k"] == float(reachable)
        assert row["vote_at_k"] == 0.0            # the wrong class is the plurality at every k
        assert row["selection_gap_lo"] == row["oracle_at_k"] - row["vote_at_k"]
        assert row["vote_at_k"] <= row["oracle_at_k"]
        assert row["complete_case"] is True and row["c"] == 1
    assert {row["oracle_at_k"] for row in rows.values()} == {0.0, 1.0}
