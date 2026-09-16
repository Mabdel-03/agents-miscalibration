"""The SELECTION_AUDIT cell: every generative selector, one draw per unique packet (spec 6.5).

ARD36 built both ends of the selector audit and never joined the middle: the verifier ran, the
compactor never read its output, and "how much better than plurality is a verifier?" could not
be answered from the tables at all.  This module is the middle, and three things make it
extensible instead of hard-coded to the one verifier the spec names:

* it dispatches **every** registry selector whose ``needs_cells`` is true, not just
  ``pointwise_audit`` (``test_a_second_generative_selector_dispatches_without_editing_audit_py``);
* the audit cell id and the ``selection_id`` both carry ``<selector>.<version>``, so a second
  generative selector gets its own cell directory and its own ``selections`` rows without
  colliding with the first;
* each selector owns its own request builder and parser (plan §1.9): :func:`audit_worker`
  binds the concrete pointwise pair onto the registry entry through :func:`bound_selector` and
  then calls ``selector.audit_requests`` / ``selector.parse_score`` for **every** generative
  selector, so a pairwise verifier is handed its own prompts and its own parser instead of
  pointwise ones (``test_a_second_generative_selector_is_asked_for_its_own_requests_and_parser``).
  Scores, failures and costs are kept per selector name, because two verifiers scoring the
  same packet are two different measurements.

What the verifier may see is the tight part.  ``templates/audit.txt`` renders the question and
**one candidate packet**, so the draw is pointwise by construction: candidate confidence,
endorsement counts, the source slot and every correctness signal are absent from the prompt,
and one draw per *unique packet hash* means two identical answers cost one call and cannot be
told apart by the verifier even in principle.  A score that fails to parse is a typed
``fallback_status`` (``audit_failed_fallback_plurality``) and never a semantic label: "the
verifier did not answer" and "the candidate is wrong" are different facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn import prompts
from agents_scaling.nextdyn import selectors as SEL
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.schema import ParseStatus, PublicOutput
from agents_scaling.nextdyn.selectors import (
    Failure,
    Score,
    SealedPool,
    Selection,
    SelectionContext,
    Selector,
)
from agents_scaling.nextdyn.tables import ItemFile, ItemStatus, SCHEMA_VERSION

#: The single field the pointwise verifier may emit (``templates/audit.txt``).
P_CORRECT: Final[str] = "p_correct"

#: Typed parse failures.  Each names what the verifier did, never what the candidate is.
FAILURE_NO_OBJECT: Final[str] = "no_json_object"
FAILURE_MISSING_FIELD: Final[str] = "p_correct_absent"
FAILURE_NOT_A_NUMBER: Final[str] = "p_correct_not_a_probability"

AUDIT_CELL_PREFIX: Final[str] = CellKind.SELECTION_AUDIT.value


@dataclass(frozen=True)
class AuditPlan:
    """One unique packet to score: which pool it came from, and at which candidate index."""

    packet_hash: str
    pool_id: str
    candidate_index: int
    candidate_id: str
    packet_body: str


def generative_selectors(registry: Mapping[str, Selector] | None = None) -> tuple[str, ...]:
    """Every registered selector that needs generation cells, in registry order.

    This is the whole extension point: a second generative selector is one registry entry and
    this function finds it (plan §1.10).
    """
    entries = SEL.SELECTORS if registry is None else registry
    return tuple(name for name, selector in entries.items() if selector.needs_cells)


def audit_cell_id(selector: str, version: str, pool_id: str) -> str:
    """``selection_audit/<selector>.<version>/<pool12>`` -- one directory per selector."""
    short = pool_id[len(SEL.POOL_ID_PREFIX):] if pool_id.startswith(SEL.POOL_ID_PREFIX) else pool_id
    return f"{AUDIT_CELL_PREFIX}/{selector}.{version}/{short[:12]}"


# --------------------------------------------------------------------------- requests


def audit_plans(pools: Sequence[SealedPool]) -> tuple[AuditPlan, ...]:
    """One entry per **unique packet hash** across the pools, in first-seen order.

    Deduplicating on the packet hash is what makes the audit cost proportional to distinct
    evidence rather than to team size, and it is also a leakage property: two agents that
    produced byte-identical packets are one object to the verifier.
    """
    seen: dict[str, AuditPlan] = {}
    for pool in pools:
        for index, candidate in enumerate(pool.candidates):
            if not candidate.valid or candidate.packet_hash in seen:
                continue
            seen[candidate.packet_hash] = AuditPlan(
                packet_hash=candidate.packet_hash, pool_id=pool.pool_id, candidate_index=index,
                candidate_id=candidate.candidate_id, packet_body="")
    return tuple(seen.values())


def audit_requests(pools: Sequence[SealedPool], *, item: Any, bodies: Mapping[str, str],
                   contract: K.StudyContract, split_role: str, top_logprobs: int,
                   pool_of: Mapping[str, SealedPool] | None = None) -> Mapping[str, Any]:
    """``packet_hash -> DecisionRequest``: one pointwise draw per unique displayed packet.

    ``bodies`` maps a packet hash to the serialized packet bytes that will be shown.  The
    audit model, its token contract and the ``PHASE_AUDIT`` key layout all come from the
    contract (``selection.audit_model``, ``audit_tokens``), so an audit row is keyed exactly
    like every other generation row and is never mistaken for the episode it audits.
    """
    from agents_scaling.nextdyn.episode import decision_request_type

    model = contract.model(contract.selection.audit_model)
    tokens = contract.audit_tokens
    model_hash = ids.model_hash(model)
    budget_hash = tokens.budget_hash()
    by_id = {pool.pool_id: pool for pool in pools} if pool_of is None else dict(pool_of)
    out: dict[str, Any] = {}
    for plan in audit_plans(pools):
        body = bodies.get(plan.packet_hash)
        if body is None:
            raise ContractViolation(
                f"no packet body supplied for {plan.packet_hash[:12]}; the audit shows the exact "
                "bytes the team transmitted, never a re-render")
        pool = by_id[plan.pool_id]
        key = pool.audit_key(plan.candidate_index, study_id=contract.study_id,
                             item_variant_id=item.item_variant_id, audit_model_hash=model_hash,
                             audit_budget_hash=budget_hash)
        out[plan.packet_hash] = decision_request_type()(
            messages=prompts.render_audit(item, body),
            model=model,
            adapter=model.adapter,
            tokens=tokens,
            seed_key=ids.seed_key_for(key, split_role=split_role),
            purpose=K.PHASE_AUDIT,
            top_logprobs=int(top_logprobs),
            key=key,
        )
    return out


# --------------------------------------------------------------------------- parsing


def _parse(text: str, *, packet_hash: str) -> Score | Failure:
    try:
        obj = json.loads(text)
    except ValueError:
        return Failure(packet_hash=packet_hash, reason=FAILURE_NO_OBJECT)
    if not isinstance(obj, dict):
        return Failure(packet_hash=packet_hash, reason=FAILURE_NO_OBJECT)
    if P_CORRECT not in obj:
        return Failure(packet_hash=packet_hash, reason=FAILURE_MISSING_FIELD)
    value = obj[P_CORRECT]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return Failure(packet_hash=packet_hash, reason=FAILURE_NOT_A_NUMBER)
    if not 0.0 <= float(value) <= 1.0:
        return Failure(packet_hash=packet_hash, reason=FAILURE_NOT_A_NUMBER)
    return Score(packet_hash=packet_hash, value=float(value))


def parse_score(record: Any, *, packet_hash: str = "") -> Score | Failure:
    """The ``pointwise_audit`` selector's parser, bound onto the registry entry by this cell.

    Strict ``{"p_correct": <probability>}`` (spec 6.5): a verifier that wrote prose instead of
    the object has not scored the candidate, and guessing a number out of its prose would
    manufacture evidence.  The failure is typed and becomes a ``fallback_status`` upstream,
    never a zero.  It accepts a record or a bare string so the rule can be exercised directly.
    """
    text = getattr(record, "final_text", record)
    return _parse(str(text or ""), packet_hash=packet_hash)


#: How a typed parse failure is reported on the audit ``decisions`` row.  Three reasons, three
#: statuses: only one of them is actually malformed JSON, and a row that said ``invalid_json``
#: for a well-formed object that simply stated no score would misreport what the verifier did.
PARSE_STATUS_OF: Final[Mapping[str, ParseStatus]] = {
    FAILURE_NO_OBJECT: ParseStatus.NO_JSON,
    FAILURE_MISSING_FIELD: ParseStatus.MISSING_FINAL,
    FAILURE_NOT_A_NUMBER: ParseStatus.INVALID_JSON,
}


def parse_status_of(reason: str) -> ParseStatus:
    """The ``decisions.parse_status`` of one typed failure reason.

    A reason this module did not declare comes from another generative selector's parser; it
    cannot be diagnosed more precisely than "the answer did not satisfy the declared schema",
    which is :attr:`ParseStatus.INVALID_JSON`.  That is a stated rule, not a silent default.
    """
    return PARSE_STATUS_OF.get(str(reason), ParseStatus.INVALID_JSON)


def request_builder(*, item: Any, bodies: Mapping[str, str], contract: K.StudyContract,
                    split_role: str, top_logprobs: int) -> Any:
    """The pointwise builder as the :class:`selectors.Selector` protocol wants it.

    The protocol method is ``audit_requests(pool, ctx)`` -- one pool -- while the module-level
    :func:`audit_requests` deduplicates across pools.  Both are needed and neither is
    redundant: the cell merges the per-pool dictionaries with first-seen wins, which is the
    same set :func:`audit_plans` would have produced over all the pools at once, while the
    *binding* stays per selector so a second verifier's own builder is what gets called.
    """
    def build(pool: SealedPool, ctx: Any) -> Mapping[str, Any]:
        del ctx                                   # a pointwise draw reads no selection context
        return audit_requests([pool], item=item, bodies=bodies, contract=contract,
                              split_role=split_role, top_logprobs=top_logprobs)

    return build


def bound_selector(name: str, *, request_builder: Any, parser: Any,
                   registry: Mapping[str, Selector] | None = None) -> Selector:
    """A registry selector with this cell's request builder and parser attached.

    ``selectors.py`` deliberately holds neither, so it never imports ``prompts`` or the
    inference layer and its import closure stays free of anything that could reach gold
    (test 18.1).  Only ``pointwise_audit`` is bound: a selector that already carries its own
    builder and parser is returned unchanged, which is what makes this module free of any
    knowledge of a second verifier's prompt or output format.
    """
    if registry is None:
        selector = SEL.selector_for(name)
    else:
        try:
            selector = registry[name]
        except KeyError as exc:                             # pragma: no cover - registry typo
            raise ContractViolation(f"unknown selector {name!r}") from exc
    if not isinstance(selector, SEL.PointwiseAudit):
        return selector
    return SEL.PointwiseAudit(request_builder=request_builder, parser=parser)


# --------------------------------------------------------------------------- the cell


def selections_for(pools: Sequence[SealedPool], scores: Mapping[str, Mapping[str, float]],
                   failures: Mapping[str, Mapping[str, str]], *, ctx: SelectionContext,
                   registry: Mapping[str, Selector] | None = None) -> tuple[Selection, ...]:
    """Run every generative selector over every pool with **its own** scores.

    ``scores`` and ``failures`` are keyed by selector name first: two verifiers that scored the
    same packet made two different measurements, and handing one selector the other's numbers
    would silently report the wrong verifier's answer.  A selector with no entry is given an
    empty score set, which is how it reports ``unexecuted`` for itself.
    """
    entries = SEL.SELECTORS if registry is None else registry
    out: list[Selection] = []
    for name in generative_selectors(entries):
        selector = entries[name]
        mine = dict(scores.get(name, {}))
        failed = dict(failures.get(name, {}))
        for pool in pools:
            out.append(selector.select(pool, replace(ctx, scores=mine, score_failures=failed)))
    return tuple(out)


def audit_worker(runner: Any, source_item_id: str) -> ItemFile:
    """The SELECTION_AUDIT cell worker (``kinds.KIND_TABLE``).

    Runner contract (WP4 supplies it, WP5 states it):

    ``runner.cell_id``                the cell this item belongs to
    ``runner.generate``               ``Generate`` bound to the store, pool and cost function
    ``runner.audit_pools(item)``      the sealed pools of this item (from ``register:pool/<sha>``)
    ``runner.audit_bodies(item)``     ``packet_hash -> serialized packet bytes``
    ``runner.audit_item(item)``       the public task item the verifier is shown
    ``runner.contract`` / ``runner.split_role`` / ``runner.top_logprobs`` / ``runner.selection_ctx``
    ``runner.selector_registry``      optional; the registry to dispatch (default ``SELECTORS``)

    Every generative selector in the registry is dispatched, and each is asked for **its own**
    requests and hands back **its own** scores through its own parser -- the cell names no
    selector and knows nothing about ``p_correct`` beyond binding the pointwise pair.
    """
    try:
        from agents_scaling.nextdyn import telemetry
    except ImportError as exc:                              # pragma: no cover - WP1 pending
        raise ManifestError(f"the SELECTION_AUDIT worker needs nextdyn.telemetry (WP1): {exc}") from exc

    pools = getattr(runner, "audit_pools", None)
    if pools is None:                                       # pragma: no cover - WP4 pending
        raise ManifestError("the cell runner does not expose audit_pools(source_item_id); the "
                            "SELECTION_AUDIT worker is driven by nextdyn.runner (WP4)")
    sealed = tuple(pools(source_item_id))
    for pool in sealed:
        if pool.scope != runner.contract.selection.pool_scope:
            raise ManifestError(
                f"{pool.pool_id} was sealed at scope {pool.scope!r} but study.yaml declares "
                f"selection.pool_scope={runner.contract.selection.pool_scope!r}; the audited "
                "population is a frozen design choice, not a per-cell option")
    if not sealed:
        return ItemFile(schema_version=SCHEMA_VERSION, cell_id=runner.cell_id,
                        cell_kind=CellKind.SELECTION_AUDIT, source_item_id=source_item_id,
                        status=ItemStatus.COMPLETE, bundle_export_hash=None,
                        rows={"decisions": (), "selections": ()})
    item = runner.audit_item(source_item_id)
    bodies = dict(runner.audit_bodies(source_item_id))
    contract: K.StudyContract = runner.contract
    registry = getattr(runner, "selector_registry", None) or SEL.SELECTORS
    ctx = runner.selection_ctx
    builder = request_builder(item=item, bodies=bodies, contract=contract,
                              split_role=runner.split_role, top_logprobs=runner.top_logprobs)

    decision_rows: list[Any] = []
    seen_decisions: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    failures: dict[str, dict[str, str]] = {}
    costs: dict[str, dict[str, int]] = {}
    audit_model_hash = ids.model_hash(contract.model(contract.selection.audit_model))
    budget_hash = contract.audit_tokens.budget_hash()
    for name in generative_selectors(registry):
        selector = bound_selector(name, request_builder=builder, parser=parse_score,
                                  registry=registry)
        mine = scores.setdefault(name, {})
        failed = failures.setdefault(name, {})
        spent = costs.setdefault(name, {})
        # first-seen wins across the item's pools: the same dedupe ``audit_plans`` applies,
        # kept here so the *selector* is the one that built every request (plan §1.9)
        requests: dict[str, Any] = {}
        for pool in sealed:
            for packet_hash, request in selector.audit_requests(pool, ctx).items():
                requests.setdefault(packet_hash, request)
        for packet_hash, request in requests.items():
            record = runner.generate(request)
            spent[packet_hash] = (int(getattr(record, "thinking_tokens", 0))
                                  + int(getattr(record, "final_tokens", 0)))
            # the selector says *what* the score is; the cell says which packet it belongs to,
            # because the packet hash is the cell's fact (the request it dispatched)
            parsed = replace(selector.parse_score(record), packet_hash=packet_hash)
            if isinstance(parsed, Score):
                mine[packet_hash] = parsed.value
            else:
                failed[packet_hash] = parsed.reason
            episode_id = ids.audit_episode_id(request.key.root_team_id, audit_model_hash,
                                              budget_hash)
            decision_id = request.key.decision_id()
            clash = seen_decisions.get(decision_id)
            if clash is not None and clash != name:
                raise ContractViolation(
                    f"{clash} and {name} issued the same audit decision id {decision_id}: a "
                    "second generative selector must key its own requests (contract §2.4), "
                    "because two prompts cannot share one decisions row")
            seen_decisions[decision_id] = name
            cell = telemetry.CellContext(cell_id=runner.cell_id, episode_id=episode_id,
                                         display_label=None, bundle_export_hash=None)
            output = _audit_output(record, parsed)
            decision_rows.append(telemetry.decision_row(
                record, request.key, output, cell=cell, parents=(), seen_transmissions=(),
                aliased=bool(getattr(record, "aliased", False)), reuse_of=None,
                is_reference=False))

    rows = []
    for selection in selections_for(sealed, scores, failures, ctx=ctx, registry=registry):
        pool = next(p for p in sealed if p.pool_id == selection.pool_id)
        spent = costs.get(selection.selector, {})
        rows.append(SEL.selection_row(
            pool, selection,
            audit_cell_id=audit_cell_id(selection.selector, selection.version, pool.pool_id),
            cost_tokens=sum(spent.get(h, 0) for h in pool.unique_packet_hashes)))
    return ItemFile(schema_version=SCHEMA_VERSION, cell_id=runner.cell_id,
                    cell_kind=CellKind.SELECTION_AUDIT, source_item_id=source_item_id,
                    status=ItemStatus.COMPLETE, bundle_export_hash=None,
                    rows={"decisions": tuple(decision_rows), "selections": tuple(rows)})


def _audit_output(record: Any, parsed: Score | Failure) -> PublicOutput:
    """The verifier's public output: its score is its answer, and it produces nothing else.

    A verifier never writes a new answer (spec 6.5 forbids synthesis), so the ``final_answer``
    of an audit row is the *score it stated* -- or ``None`` when it stated none, which is what
    makes ``validity`` on an audit row mean "the verifier answered".

    ``confidence`` stays ``None`` on purpose.  That column is the spec-13.1 signal "how sure
    was the model that produced this answer", tracked per solver decision and calibrated
    against the solver's own correctness; a verifier's ``p_correct`` is a judgement about
    *someone else's* answer, and writing it there would put two different quantities in one
    column and quietly merge them in any panel that reads it.  The score lives in the answer
    column of the ``audit`` phase row, where ``cell_kind`` and ``phase`` say what it is.

    The failure statuses are three, not one: only :data:`FAILURE_NO_OBJECT` is malformed JSON
    (:func:`parse_status_of`).
    """
    text = str(getattr(record, "final_text", "") or "")
    ok = isinstance(parsed, Score)
    return PublicOutput(
        reasoning=None, shareable_claims=(),
        final_answer=None if not ok else repr(float(parsed.value)),
        confidence=None,
        field_order=(P_CORRECT,) if ok else (),
        order_violation=False,
        parse_status=ParseStatus.OK if ok else parse_status_of(parsed.reason),
        raw_text=text, content_hash=ids.content_hash(text.encode("utf-8")),
        over_limits={}, final_answer_span=None, confidence_span=None,
        confidence_emitted_before_answer=False)


__all__ = [
    "AUDIT_CELL_PREFIX", "AuditPlan", "FAILURE_MISSING_FIELD", "FAILURE_NOT_A_NUMBER",
    "FAILURE_NO_OBJECT", "PARSE_STATUS_OF", "P_CORRECT", "audit_cell_id", "audit_plans",
    "audit_requests", "audit_worker", "bound_selector", "generative_selectors",
    "parse_score", "parse_status_of", "request_builder", "selections_for",
]
