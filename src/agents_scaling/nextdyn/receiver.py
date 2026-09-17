"""The receiver assays: E2a one-call and E2b two-phase (spec 9.1, 9.4; plan §1.8, D17).

A receiver episode is a *replay* assay: the working draft and both candidate packets are
fixed in advance, and the only thing the arm changes is what the receiver is told about them.
Two properties carry the whole design, and both are asserted here rather than described:

**The evidence bytes never move.**  "The two displayed packet bodies are byte-identical under
all support conditions for an item" (spec 9.1).  So the endorsement cue is rendered into its
own ``SUPPORT`` block of the prompt, the packet display order is drawn from a blind key that
has no condition in it, and :func:`assert_support_invariant_bodies` compares the rendered
bodies across the three support levels of one item.  If the cue could nudge the bytes, the
1:4 / 4:1 contrast would not be a cue contrast at all.

**The reasoning budget is per episode, not per call.**  "Total reasoning cap per logical
episode is B, not B per call" (spec 9.4).  E2a is therefore **one** call at the full cap B
with 512 public tokens; E2b is **two** calls at ``B/2`` each -- plan (256 public tokens),
check service, decision (512 public tokens) -- and the two per-phase caps sum to B
(:func:`assert_budget_split`).  The *identity* budget of both phases is the episode-level
``receiver_tokens["<B>/one_call"]`` contract (contract §2.0), so one E2b episode has one
``episode_id`` and one ``budget_hash``; the per-phase ``B/2`` contracts enter only the request
identity through the decoding.

**Support counts are metadata, not generations.**  The counts describe "a constructed panel of
five advisers ... randomized experimental metadata, not five independently generated votes"
(spec 9.1), and the prompt says exactly that.  :attr:`ReceiverEpisode.independent_generations`
counts the model calls this episode actually made -- one or two -- and never the panel
(``test_support_metadata_is_not_counted_as_independent_generations``).  How large that panel
is comes from ``study.yaml`` through :func:`support_panel_size`, never from this module.

**E2b evidence is built here too.**  :func:`mod97_pair` is the E2b analogue of
``natural_packets.fix_natural_pair``: it renders the generator's correct and wrong traces
through ``packets.render_packet`` -- the same cap, the same priority clipping, the same
metadata denylist and the same tokenizer stamp an E1 packet gets -- and returns the trace map
keyed by the **displayed** ``packet_id``, which is exactly what :class:`checker.CheckStepTool`
serves.  Without it the two artefacts would drift apart: the tool would answer about rows the
receiver never saw, and spec 9.2's error position would reach no table.

A note on the plan-phase row.  ``templates/receiver_plan.txt`` asks for
``{"check_requests": [...]}``, not for the spec-6.1 answer object, so a ``receiver_plan``
decision row carries the answer schema's parse status by construction (there is no
``final_answer`` in a plan) and its ``validity`` is ``False``.  Whether the *plan* parsed is
read from the ``tool_events`` the phase produced and from
:attr:`ReceiverEpisode.plan_parsed`; spec 9.4 is explicit that invalid plan output continues
to the decision phase without a repair call, so it is never an error here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.ard.packets import pseudonym_map
from agents_scaling.nextdyn import checker as CH
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn import mod97
from agents_scaling.nextdyn import packets as PK
from agents_scaling.nextdyn import prompts
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind, CompositeKey
from agents_scaling.nextdyn.protocols import DraftRole, PacketContent, ReceiverSpec, Support
from agents_scaling.nextdyn.protocols import TransmissionIntent
from agents_scaling.nextdyn.states import pairing_id_for
from agents_scaling.nextdyn.tables import ItemFile, ItemStatus, SCHEMA_VERSION, TransmissionRow
from agents_scaling.study import identity
from agents_scaling.study.packets import make_counter

#: ``run_receiver_episode``'s generator, bound by WP1 (``episode.Generate``).
Generate = Callable[[Any], Any]

#: Spec 9.1 assays.  The two differ in *how the evidence was produced* (a real donor bank vs
#: the mod-97 generator), never in the episode machinery below.
ASSAY_E2A: Final[str] = "E2a"
ASSAY_E2B: Final[str] = "E2b"
ASSAYS: Final[tuple[str, ...]] = (ASSAY_E2A, ASSAY_E2B)

#: ``transmissions.declared_edge`` for a receiver slot: the packet was *assigned* by the
#: design, not delivered by a resolved communication graph (plan §1.8).
DECLARED_EDGE_ASSIGNED: Final[str] = "assigned"

#: The per-budget receiver token contracts ``config.contract_of`` registers, by name.
BUDGET_ONE_CALL: Final[str] = "one_call"
BUDGET_PLAN: Final[str] = "plan"
BUDGET_DECISION: Final[str] = "decision"

#: What the prompt says about the check tool, so the two checker arms differ in one sentence
#: and not in prompt shape (spec 9.4 step 2).
CHECKER_AVAILABLE: Final[str] = (
    "check_step(packet_id, step_id) is available. You may request at most {n} checks."
)
CHECKER_UNAVAILABLE: Final[str] = "check_step is not available in this episode."

#: Spec 9.1: the support block names a *constructed panel*, and says so in the prompt.  ``{n}``
#: is the panel size, which is an experimental quantity and therefore has no home in this
#: module: see :func:`support_panel_size`.
SUPPORT_HEAD: Final[str] = (
    "The counts below come from a constructed panel of {n} advisers. They are randomized "
    "experimental metadata, not {n} independently generated answers."
)


def support_panel_size(e2a_block: Any) -> int:
    """``study.yaml``'s ``experiments.E2a.support_panel_size``: how many advisers endorse.

    It is *not* a module constant, because it is not a free choice of the code: the spec-9.1
    endorsement levels 1:4 and 4:1 are **derived** from it by :func:`support_counts`, so a
    different panel would be a different declared design and must be a config edit the
    compiler and the consumer check can see (contract preamble: "No experimental number is a
    module constant anywhere in the package").

    One leaf serves both assays.  Spec 9.1 describes one constructed panel and E2b displays
    the same block, so a second copy under ``experiments.E2b`` would be a second declaration
    of one fact (R1).  It is deliberately not ``protocol.n_agents`` either: the panel is "not
    literally the other four agents of E1".
    """
    size = int(e2a_block.support_panel_size)
    if size < 2:
        raise ContractViolation(
            f"experiments.E2a.support_panel_size is {size}: an endorsement split needs at least "
            "two advisers, one of whom is the minority")
    return size


def support_counts(endorsement: Support, panel_size: int) -> tuple[int, int]:
    """``(count for the correct candidate, count for the wrong one)`` (spec 9.1).

    Swapping the endorsement swaps the two numbers and changes nothing else, which is what
    makes the 1:4 / 4:1 pair a pure cue contrast.  ``hidden`` has no counts at all.
    """
    endorsement = Support(endorsement)
    if endorsement is Support.HIDDEN:
        raise ContractViolation("the hidden arm carries no endorsement counts (spec 9.1)")
    size = int(panel_size)
    if size < 2:
        raise ContractViolation(f"an endorsement panel of {size} cannot be split 1:{size - 1}")
    minority = endorsement is Support.CORRECT_MINORITY
    return (1, size - 1) if minority else (size - 1, 1)


def support_fields(endorsement: Support, labels: tuple[str, str], seed: bytes, *,
                   panel_size: int) -> tuple[str, ...]:
    """The ``SUPPORT`` block lines for one episode, in a blind order.

    ``labels`` is ``(label of the correct packet, label of the wrong packet)`` -- the opaque
    pseudonyms the packets already carry, so the support lines and the packets refer to each
    other without any correctness cue.  Which line is printed first is drawn from ``seed``
    independently of correctness (spec 9.1: "support-field order ... balanced independently of
    correctness"); the ``hidden`` arm prints no lines, and the caller renders the literal
    ``NONE``.
    """
    endorsement = Support(endorsement)
    if endorsement is Support.HIDDEN:
        return ()
    correct_label, wrong_label = labels
    size = int(panel_size)
    correct_count, wrong_count = support_counts(endorsement, size)
    lines = [f"{correct_label}: supported by {correct_count} of {size} advisers.",
             f"{wrong_label}: supported by {wrong_count} of {size} advisers."]
    if _blind_bit(seed):
        lines.reverse()
    return (SUPPORT_HEAD.format(n=size), *lines)


def _blind_bit(seed: bytes) -> bool:
    return bool(bytes(seed)[0] & 1) if seed else False


# --------------------------------------------------------------------------- fixed packets


@dataclass(frozen=True)
class FixedPacketPair:
    """The two frozen candidates of one receiver item, plus the two possible working drafts.

    "For every item, freeze two packets and two possible working drafts, one correct and one
    wrong.  The working draft is explicitly an assigned draft, not an assertion that this
    receiver generated it previously" (spec 9.1).  The draft body is the corresponding
    packet's bytes, so the assigned-draft factor cannot accidentally introduce a third text.
    """

    source_item_id: str
    panel_id: str
    correct: PK.TransmittedPacket
    wrong: PK.TransmittedPacket
    correct_generation_uid: str | None = None
    wrong_generation_uid: str | None = None
    #: Where the injected error sits along the answer chain, relative (E2b only; spec 9.2
    #: "Log token length and error position").  Recorded on the wrong packet's transmission row.
    error_position_rel: float | None = None

    def __post_init__(self) -> None:
        if self.correct.packet_hash == self.wrong.packet_hash:
            raise ContractViolation(
                f"{self.source_item_id}: the correct and wrong packets are byte-identical")
        if self.correct.pseudonym == self.wrong.pseudonym:
            raise ContractViolation(
                f"{self.source_item_id}: the two packets must carry distinct display labels")

    @property
    def labels(self) -> tuple[str, str]:
        """``(correct label, wrong label)`` -- the opaque pseudonyms the support block uses."""
        return (self.correct.pseudonym, self.wrong.pseudonym)

    def draft_for(self, role: DraftRole) -> PK.TransmittedPacket:
        role = DraftRole(role)
        if role is DraftRole.CORRECT:
            return self.correct
        if role is DraftRole.WRONG:
            return self.wrong
        raise ContractViolation("a receiver episode is always assigned a working draft "
                                "(spec 9.1: correct or wrong)")

    def display(self, seed: bytes) -> tuple[PK.TransmittedPacket, ...]:
        """The two packets in their blind display order: balanced, correctness-independent.

        The order is a function of the seed alone, so it is the same under every support and
        checker condition of one item -- which is what keeps the ``OTHER DRAFTS`` block
        byte-identical across the arms (spec 9.1).
        """
        pair = (self.correct, self.wrong)
        return tuple(reversed(pair)) if _blind_bit(seed) else pair


def mod97_pair(problem: mod97.Problem, *, study_seed: bytes, counter: Callable[[str], int],
               counter_tokenizer_hash: str,
               tokens: K.TokenContract) -> tuple[FixedPacketPair, dict[str, mod97.Trace]]:
    """Freeze one E2b problem's two displayed packets, and say which trace each one displays.

    The E2b analogue of ``natural_packets.fix_natural_pair``, and it exists for the same
    reason: a packet the receiver reads must be the same *kind of object* whatever produced
    it, so the two mod-97 traces go through :func:`packets.render_packet` -- same cap, same
    priority clipping, same metadata denylist, same tokenizer stamp -- rather than being
    pasted into the prompt raw.  Three things are wired here and nowhere else:

    * the **displayed** ``packet_id`` is the key of the returned trace map, which is the
      ``packets`` argument :class:`checker.CheckStepTool` serves, so ``check_step`` can only
      ever answer about a row the receiver can see (a tool keyed on anything else would be
      answering about a different artefact);
    * ``error_position_rel`` is carried from ``Trace`` to the pair, and from there to the
      wrong packet's ``transmissions`` row -- spec 9.2's "log ... error position";
    * the two pseudonyms come from the same ``pseudonym_map`` every packet slot in the study
      uses, drawn on a pairing id that names the problem and nothing about the arm (R5).

    A clipped trace is refused rather than transmitted: the checker would then serve rows the
    receiver never read, which is the one correspondence this assay cannot lose.
    """
    correct = mod97.correct_trace(problem)
    wrong = mod97.wrong_trace(problem, study_seed)
    labels = pseudonym_map(study_seed, e2b_pairing_id(problem.problem_id), 2)
    rendered: list[PK.TransmittedPacket] = []
    for trace, pseudonym in ((correct, labels[0]), (wrong, labels[1])):
        packet = PK.render_packet(_trace_output(trace), pseudonym=pseudonym, counter=counter,
                                  counter_tokenizer_hash=counter_tokenizer_hash,
                                  cap=tokens.packet_cap, content=PacketContent.FULL)
        if packet.any_truncated:
            raise ContractViolation(
                f"{problem.problem_id}: the displayed trace does not fit the "
                f"{tokens.packet_cap}-token packet cap ({sorted(packet.truncated)} clipped); a "
                "checker that can answer about a clipped row would be serving evidence the "
                "receiver never saw (budgets.public_packet_token_cap vs "
                "experiments.E2b.total_displayed_operations)")
        rendered.append(packet)
    pair = FixedPacketPair(
        source_item_id=problem.source_item_id, panel_id=problem.problem_id,
        correct=rendered[0], wrong=rendered[1],
        error_position_rel=wrong.error_position_rel,
    )
    return pair, {rendered[0].packet_id: correct, rendered[1].packet_id: wrong}


def e2b_pairing_id(problem_id: str) -> str:
    """The pseudonym map's pairing id for an E2b problem: the problem, nothing about the arm."""
    if not problem_id:
        raise ContractViolation("an E2b packet pair names its generated problem")
    return f"e2b|{problem_id}"


def _trace_output(trace: mod97.Trace) -> schema.PublicOutput:
    """One displayed trace as the public object a packet is rendered from (spec 6.1 shape).

    The trace body is the rationale and the claimed final is the answer; there is no
    confidence, because a generated packet states none and ``render_packet`` never transmits
    one anyway (spec 9.1: confidence is hidden from receivers).
    """
    body = mod97.render_trace(trace)
    answer = str(int(trace.final))
    raw = json.dumps({"reasoning": body, "final_answer": answer}, sort_keys=True)
    return schema.PublicOutput(
        reasoning=body, shareable_claims=(), final_answer=answer, confidence=None,
        field_order=("reasoning", "final_answer"), order_violation=False,
        parse_status=schema.ParseStatus.OK, raw_text=raw,
        content_hash=ids.content_hash(raw.encode("utf-8")), over_limits={},
        final_answer_span=None, confidence_span=None, confidence_emitted_before_answer=False)


# --------------------------------------------------------------------------- the condition


@dataclass(frozen=True)
class ReceiverCondition:
    """One receiver cell: the arm, the item, the model, the budget and the replicate.

    The arm itself is the :class:`~agents_scaling.nextdyn.protocols.ReceiverSpec` loaded from
    ``protocols.yaml`` (conditions are data, D7); everything else here is the instance the arm
    is applied to.  ``protocol_hash`` is the spec's condition hash, so two cells that differ
    only in their label share a hash and alias in the request store.
    """

    assay: str
    spec: ReceiverSpec
    item: Any                          # ``prompts.TaskItem``: a SourceItem or a mod97.Problem
    source_item_id: str
    item_variant_id: str
    study_id: str
    panel_id: str                      # E2a "packets:<register sha>", E2b the mod97 problem_id
    model: K.ModelCell
    budget: int
    replicate: int
    split_role: str
    top_logprobs: int
    tokens: Mapping[str, K.TokenContract]
    #: ``experiments.E2a.support_panel_size`` as :func:`support_panel_size` read it; the cell
    #: builder supplies it because no experimental number may live in this module.
    support_panel_size: int
    display_label: str | None = None

    def __post_init__(self) -> None:
        if self.assay not in ASSAYS:
            raise ContractViolation(f"unknown receiver assay {self.assay!r}; expected {ASSAYS}")
        if not self.panel_id:
            raise ContractViolation("a receiver episode names the fixed-packet panel it replays")
        if int(self.support_panel_size) < 2:
            raise ContractViolation(
                f"{self.panel_id}: an endorsement panel of {self.support_panel_size} cannot carry "
                "a minority and a majority (experiments.E2a.support_panel_size)")
        object.__setattr__(self, "tokens", dict(self.tokens))
        for name in (BUDGET_ONE_CALL, BUDGET_PLAN, BUDGET_DECISION):
            key = f"{int(self.budget)}/{name}"
            if key not in self.tokens:
                raise ContractViolation(
                    f"contract.receiver_tokens has no {key!r}; budget {self.budget} is not one of "
                    f"the declared receiver budgets")
        assert_budget_split(self.episode_tokens, self.plan_tokens, self.decision_tokens)
        assert_templates(self.spec)

    @classmethod
    def from_contract(cls, contract: K.StudyContract, **kwargs: Any) -> "ReceiverCondition":
        """Bind the per-budget token contracts from ``StudyContract.receiver_tokens``."""
        return cls(study_id=contract.study_id, tokens=contract.receiver_tokens, **kwargs)

    # -- the arm ------------------------------------------------------------------
    @property
    def endorsement(self) -> Support:
        return self.spec.support

    @property
    def initial_draft(self) -> DraftRole:
        return self.spec.draft_role

    @property
    def checker(self) -> bool:
        return bool(self.spec.checker)

    @property
    def phases(self) -> int:
        return int(self.spec.phases)

    @property
    def shows_packets(self) -> bool:
        """``packet_content: none`` is the working-draft-only control (spec 9.1, 9.5)."""
        return self.spec.packet_content is not PacketContent.NONE

    # -- budgets ------------------------------------------------------------------
    def _tokens(self, name: str) -> K.TokenContract:
        return self.tokens[f"{int(self.budget)}/{name}"]

    @property
    def episode_tokens(self) -> K.TokenContract:
        """The **episode-level** contract: full B.  Its hash is the key's ``budget_hash``."""
        return self._tokens(BUDGET_ONE_CALL)

    @property
    def plan_tokens(self) -> K.TokenContract:
        return self._tokens(BUDGET_PLAN)

    @property
    def decision_tokens(self) -> K.TokenContract:
        return self._tokens(BUDGET_DECISION)

    def tokens_for(self, phase: str) -> K.TokenContract:
        """What the *request* is decoded at: B for a one-call episode, B/2 for each E2b phase."""
        if self.phases == 1:
            return self.episode_tokens
        return self.plan_tokens if phase == K.PHASE_RECEIVER_PLAN else self.decision_tokens

    # -- identity -----------------------------------------------------------------
    @property
    def model_hash(self) -> str:
        return ids.model_hash(self.model)

    @property
    def budget_hash(self) -> str:
        return self.episode_tokens.budget_hash()

    @property
    def protocol_hash(self) -> str:
        return self.spec.condition_hash

    @property
    def pairing_id(self) -> str:
        """Shared by every arm of one item at one model/budget/replicate: no condition in it."""
        return pairing_id_for(self.source_item_id, self.panel_id, self.model_hash,
                              self.budget_hash, self.replicate)

    @property
    def episode_id(self) -> str:
        return ids.episode_id(CellKind.RECEIVER, self.panel_id, self.protocol_hash,
                              self.model_hash, self.budget_hash, self.replicate)

    def key_for(self, phase: str) -> CompositeKey:
        """Both phases sit at round 0, slot 0: one receiver, two phases (contract §2.0)."""
        return CompositeKey(
            study_id=self.study_id, cell_kind=CellKind.RECEIVER,
            source_item_id=self.source_item_id, item_variant_id=self.item_variant_id,
            root_team_id=self.panel_id, protocol_hash=self.protocol_hash,
            receiver_model_hash=self.model_hash, budget_hash=self.budget_hash,
            replicate_id=int(self.replicate), round=0, agent_slot=0, phase=phase)

    def display_seed(self, study_seed: bytes) -> bytes:
        """The blind key for candidate display order: item, panel, replicate -- no condition."""
        return ids.blind(study_seed, K.ORDER_PACKET, self.pairing_id, "candidate_order")

    def support_seed(self, study_seed: bytes) -> bytes:
        """The blind key for support-field order, drawn independently of the display order."""
        return ids.blind(study_seed, K.ORDER_SLOT, self.pairing_id, "support_order")

    def pseudonyms(self, study_seed: bytes) -> tuple[str, ...]:
        """Two per-episode pseudonyms for the two fixed packets (``ard.packets.pseudonym_map``)."""
        return pseudonym_map(study_seed, self.pairing_id, 2)


def assert_budget_split(episode: K.TokenContract, plan: K.TokenContract,
                        decision: K.TokenContract) -> None:
    """Spec 9.4: the two phase caps sum to the declared episode budget, not to twice it."""
    if plan.thinking_cap + decision.thinking_cap != episode.thinking_cap:
        raise ContractViolation(
            f"two-phase thinking caps {plan.thinking_cap} + {decision.thinking_cap} do not sum to "
            f"the declared episode budget {episode.thinking_cap} (spec 9.4: total reasoning per "
            "logical episode is B, not B per call)")


#: Which template each phase of a receiver arm renders, in phase order (spec 9.4).  The arm
#: declares its own list in ``protocols.yaml``; this is the shape that list must have, so a
#: one-call arm cannot quietly acquire a plan phase by naming two templates.
PHASE_TEMPLATES: Final[Mapping[int, tuple[str, ...]]] = {
    1: (prompts.RECEIVER_DECISION,),
    2: (prompts.RECEIVER_PLAN, prompts.RECEIVER_DECISION),
}


def assert_templates(spec: ReceiverSpec) -> None:
    """``protocols.yaml``'s ``templates`` list is the arm's render order, and it is checked.

    ``ReceiverSpec`` already refuses a list whose *length* disagrees with ``phases``; what is
    checked here is which templates, in which order -- a two-phase arm that named the decision
    template twice would run two decision calls at B/2 and would silently be a different
    experiment.
    """
    expected = PHASE_TEMPLATES[int(spec.phases)]
    if tuple(spec.templates) != expected:
        raise ContractViolation(
            f"{spec.display_label}: a {spec.phases}-phase receiver renders {expected}, "
            f"not {tuple(spec.templates)}")


def assert_generation_phases(e2b_block: Any, spec: ReceiverSpec) -> None:
    """``experiments.E2b.generation_phases`` is the declared call count of an E2b arm."""
    declared = int(e2b_block.generation_phases)
    if int(spec.phases) != declared:
        raise ContractViolation(
            f"{spec.display_label}: protocols.yaml declares {spec.phases} phase(s) but "
            f"study.yaml experiments.E2b.generation_phases is {declared}")


# --------------------------------------------------------------------------- the episode


@dataclass(frozen=True)
class ReceiverDecision:
    """One generation call of one receiver episode."""

    key: CompositeKey
    decision_id: str
    phase: str
    output: schema.PublicOutput
    record: Any
    messages: tuple[Mapping[str, str], ...]

    @property
    def aliased(self) -> bool:
        return bool(getattr(self.record, "aliased", False))


@dataclass(frozen=True)
class ReceiverEpisode:
    """Everything one receiver episode produced."""

    cond: ReceiverCondition
    episode_id: str
    decisions: tuple[ReceiverDecision, ...]
    displayed: tuple[PK.TransmittedPacket, ...]
    support_lines: tuple[str, ...]
    check_requests: tuple[CH.CheckRequest, ...]
    check_responses: tuple[CH.CheckResponse, ...]
    tool_events: tuple[Any, ...]
    plan_parsed: bool

    @property
    def generation_calls(self) -> int:
        return len(self.decisions)

    @property
    def independent_generations(self) -> int:
        """How many independent model generations this episode contains.

        Exactly the generation calls.  The support block's five advisers are randomized
        metadata (spec 9.1) and the two displayed packets are *fixed replay material* drawn
        once for the whole panel -- neither is a generation of this episode, and this property
        is the single place that count is defined.
        """
        return self.generation_calls

    @property
    def decision(self) -> ReceiverDecision:
        for item in self.decisions:
            if item.phase == K.PHASE_RECEIVER_DECISION:
                return item
        raise ManifestError(f"{self.episode_id} produced no decision-phase row")


def _request(cond: ReceiverCondition, phase: str, messages: Sequence[Mapping[str, str]]) -> Any:
    from agents_scaling.nextdyn.episode import decision_request_type

    key = cond.key_for(phase)
    return decision_request_type()(
        messages=tuple(messages),
        model=cond.model,
        adapter=cond.model.adapter,
        tokens=cond.tokens_for(phase),
        seed_key=ids.seed_key_for(key, split_role=cond.split_role),
        purpose=phase,
        top_logprobs=int(cond.top_logprobs),
        key=key,
    )


def _decision(cond: ReceiverCondition, phase: str, messages: Sequence[Mapping[str, str]],
              generate: Generate, counter: Callable[[str], int]) -> ReceiverDecision:
    request = _request(cond, phase, messages)
    record = generate(request)
    output = schema.parse_public_output(record.final_text, record.finish_reason,
                                        record.completion_token_ids, counter,
                                        limits=cond.tokens_for(phase))
    return ReceiverDecision(key=request.key, decision_id=request.key.decision_id(), phase=phase,
                            output=output, record=record, messages=tuple(request.messages))


def _packet_bodies(cond: ReceiverCondition, packets: FixedPacketPair,
                   study_seed: bytes) -> tuple[PK.TransmittedPacket, ...]:
    if not cond.shows_packets:
        return ()
    return packets.display(cond.display_seed(study_seed))


def run_receiver_episode(cond: ReceiverCondition, packets: FixedPacketPair, generate: Generate,
                         tool: CH.CheckStepTool | None, *, study_seed: bytes,
                         tokenizer: Any) -> ReceiverEpisode:
    """One logical receiver episode: one call (E2a) or plan + check + decision (E2b).

    The one-call path has **no plan phase and no tool at all** (plan D17): E2a and its PRIVATE
    controls are a single decision-phase call at the full cap B with 512 public tokens, the
    assigned working draft in ``WORKING DRAFT`` and both fixed packets in ``OTHER DRAFTS``.
    The two-phase path serves the checker between the calls and hands the decision phase the
    plan transcript together with the typed responses -- ``disabled`` responses when the arm
    has no checker, so the two arms differ in the response *content* and not in its shape.

    **Both E2b arms are served by a tool object**, and ``tool=None`` is refused for a two-phase
    arm rather than read as "the checker is off".  Two things depend on it.  The no-check arm
    has to spend its request slots identically -- "no adaptive exception grants extra checks"
    (spec 9.3) -- so a ten-request plan must cost four served lines in *both* arms or the two
    prompts differ in length as well as in content, which is exactly the placebo symmetry spec
    9.4 step 2 demands.  And ``tool_events`` is where spec 9.6's checker-use and
    requested-step distributions come from: an arm that wrote none would be missing from that
    report entirely.  Reading ``None`` as "disabled" would also silently downgrade a
    ``checker: true`` arm to an unchecked episode whose rows still carry the checker-on
    ``protocol_hash``, which is a corrupted H4 contrast and not a missing one.
    """
    counter = make_counter(tokenizer)
    displayed = _packet_bodies(cond, packets, study_seed)
    bodies = [p.serialized for p in displayed]
    support = support_fields(cond.endorsement, packets.labels, cond.support_seed(study_seed),
                             panel_size=cond.support_panel_size)
    draft = packets.draft_for(cond.initial_draft).serialized

    if cond.phases == 1:
        if cond.checker:                                   # pragma: no cover - ReceiverSpec refuses
            raise ContractViolation("a one-call assay cannot serve the checker (spec 9.4)")
        messages = prompts.render_receiver_decision(cond.item, draft, bodies, support,
                                                    check_results=())
        decision = _decision(cond, K.PHASE_RECEIVER_DECISION, messages, generate, counter)
        return ReceiverEpisode(cond=cond, episode_id=cond.episode_id, decisions=(decision,),
                               displayed=displayed, support_lines=support, check_requests=(),
                               check_responses=(), tool_events=(), plan_parsed=False)

    if tool is None:
        raise ContractViolation(
            f"{cond.episode_id}: a two-phase arm is served by a CheckStepTool "
            f"(enabled={cond.checker}); None is not the disabled arm, because the disabled arm "
            "still spends its four request slots and still writes tool_events (spec 9.3, 9.6)")
    if bool(tool.enabled) != cond.checker:
        raise ContractViolation(
            f"{cond.episode_id}: the arm declares checker={cond.checker} but the tool is "
            f"{'enabled' if tool.enabled else 'disabled'}")
    plan_messages = prompts.render_receiver_plan(
        cond.item, draft, bodies, support,
        checker_available=_checker_status(cond, tool))
    plan = _decision(cond, K.PHASE_RECEIVER_PLAN, plan_messages, generate, counter)
    requests = CH.parse_plan_output(plan.record.final_text)
    plan_parsed = bool(requests) or _plan_is_wellformed(plan.record.final_text)
    responses, events = tool.serve(requests, decision_id=plan.decision_id,
                                   episode_id=cond.episode_id)
    decision_messages = prompts.render_receiver_decision(
        cond.item, draft, bodies, support,
        check_results=_transcript(plan.record.final_text, responses))
    decision = _decision(cond, K.PHASE_RECEIVER_DECISION, decision_messages, generate, counter)
    return ReceiverEpisode(cond=cond, episode_id=cond.episode_id, decisions=(plan, decision),
                           displayed=displayed, support_lines=support, check_requests=requests,
                           check_responses=responses, tool_events=events, plan_parsed=plan_parsed)


def _checker_status(cond: ReceiverCondition, tool: CH.CheckStepTool) -> str:
    """One sentence, and the arm is the only thing that chooses which (spec 9.4 step 2)."""
    if not cond.checker:
        return CHECKER_UNAVAILABLE
    return CHECKER_AVAILABLE.format(n=tool.max_checks)


def _plan_is_wellformed(text: str | None) -> bool:
    """Did the plan phase emit the declared object at all (even with an empty request list)?"""
    if not isinstance(text, str):
        return False
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    return isinstance(obj, dict) and isinstance(obj.get(CH.PLAN_FIELD), list)


def _transcript(plan_text: str, responses: Sequence[CH.CheckResponse]) -> tuple[str, ...]:
    """The decision phase's ``CHECK RESULTS`` block: the plan output and the typed responses."""
    lines = [line for line in [str(plan_text or "").strip()] if line]
    lines.extend(response.as_line() for response in responses)
    return tuple(lines)


# --------------------------------------------------------------------------- rows


def transmission_rows(episode: ReceiverEpisode, packets: FixedPacketPair, *,
                      receiver_counter: Callable[[str], int]) -> tuple[TransmissionRow, ...]:
    """One ``transmissions`` row per displayed packet of the decision-phase call.

    ``declared_edge`` is :data:`DECLARED_EDGE_ASSIGNED`: a receiver's packets are assigned by
    the design, not delivered by a resolved graph.  ``transmitted_tokens`` is the packet's
    token length and ``error_position_rel`` is where the injected error sits along the answer
    chain of the wrong trace -- the two quantities spec 9.2 asks to be logged.

    ``receiver_counter`` is the **reading** model's counter and fills
    ``transmitted_tokens_receiver``.  That column exists for exactly this design: an E2a packet
    is clipped and counted once with the donor model's tokenizer (``counter_tokenizer_hash``)
    and then replayed verbatim to both receiver models (spec 9.1's receiver-model factor), so
    "how long was this evidence for the model that read it?" is a different number from "how
    long was it for the model that wrote it", and a cross-model prompt-length comparison that
    used the sender's count would be comparing the wrong quantity.  The bytes are never
    re-clipped (plan §1.6): only the count is recomputed.
    """
    decision = episode.decision
    cond = episode.cond
    intents = tuple(
        TransmissionIntent(sender_slot=position, source_round=0, source_decision_id=None,
                           content=packet.content, filler=False, display_position=position)
        for position, packet in enumerate(episode.displayed))
    row_ids = PK.transmission_ids(decision.decision_id, intents)
    rows: list[TransmissionRow] = []
    for intent, row_id, packet in zip(intents, row_ids, episode.displayed):
        is_wrong = packet.packet_hash == packets.wrong.packet_hash
        rows.append(TransmissionRow(
            transmission_id=row_id,
            episode_id=episode.episode_id,
            cell_kind=CellKind.RECEIVER.value,
            protocol_hash=cond.protocol_hash,
            source_round=0,
            destination_round=0,
            sender_slot=int(intent.sender_slot),
            receiver_slot=0,
            receiver_decision_id=decision.decision_id,
            source_decision_id=None,
            packet_hash=packet.packet_hash,
            original_packet_hash=packet.original_content_hash,
            transmitted_tokens=int(packet.token_count),
            counter_tokenizer_hash=packet.counter_tokenizer_hash,
            transmitted_tokens_receiver=int(receiver_counter(packet.serialized)),
            original_tokens=None,
            truncated_fields_json=json.dumps(dict(packet.truncated), sort_keys=True),
            filler_flag=False,
            filler_tokens=None,
            packet_content_mode=PacketContent(packet.content).value,
            declared_edge=DECLARED_EDGE_ASSIGNED,
            display_position=int(intent.display_position),
            error_position_rel=packets.error_position_rel if is_wrong else None,
        ))
    return tuple(rows)


def assert_support_invariant_bodies(cond_by_support: Mapping[Support, ReceiverEpisode]) -> None:
    """Spec 9.1: the displayed packet bodies are byte-identical under every support level.

    Compares the rendered ``OTHER DRAFTS`` bytes of the given episodes of one item; the
    support block is *not* compared, because that is the treatment.
    """
    bodies = {support: tuple(p.serialized for p in episode.displayed)
              for support, episode in cond_by_support.items()}
    distinct = {value for value in bodies.values()}
    if len(distinct) > 1:
        raise ContractViolation(
            "the displayed packet bodies differ between support conditions; the endorsement cue "
            f"must live only in the support block (got {sorted(bodies)})")


# --------------------------------------------------------------------------- planner counts


@dataclass(frozen=True)
class GridCount:
    """What one model's receiver grid costs, counted the way spec 9.5 counts it."""

    episodes: int
    calls: int
    control_episodes: int
    control_calls: int

    @property
    def total_calls(self) -> int:
        return self.calls + self.control_calls


def e2b_grid_counts(e2b_block: Any, *, budgets: Sequence[int]) -> GridCount:
    """Spec 9.5: 300 x 3 x 2 x 2 x 2 x 2 = 14,400 logical episodes, 28,800 calls per model.

    Depth is **not** a separate factor: the three depths are already represented inside the
    300 source problems ("Do not multiply by the three depth levels again").  The controls are
    working-draft-only, no-peer, no-check episodes at both budgets and both initial drafts, and
    they run the same two-phase adapter, so they cost two calls each as well.
    """
    phases = int(e2b_block.generation_phases)
    cells = (len(tuple(e2b_block.endorsement)) * len(tuple(e2b_block.checker_access))
             * len(tuple(e2b_block.initial_draft)))
    per_problem = cells * len(tuple(budgets)) * int(e2b_block.replicates)
    episodes = int(e2b_block.source_problems) * per_problem
    controls = (int(e2b_block.source_problems) * len(tuple(e2b_block.initial_draft))
                * len(tuple(budgets)) * int(e2b_block.replicates))
    return GridCount(episodes=episodes, calls=episodes * phases, control_episodes=controls,
                     control_calls=controls * phases)


def e2a_grid_counts(e2a_block: Any, *, supports: Sequence[Any], drafts: Sequence[Any],
                    budgets: Sequence[int]) -> GridCount:
    """Spec 9.1: 100 x 3 x 2 x 2 x 2 = 2,400 **one-call** episodes per task/model, + 800 controls."""
    per_item = len(tuple(supports)) * len(tuple(drafts)) * len(tuple(budgets)) * int(e2a_block.replicates)
    episodes = int(e2a_block.items_per_task) * per_item
    controls = (int(e2a_block.items_per_task) * len(tuple(drafts)) * len(tuple(budgets))
                * int(e2a_block.replicates))
    return GridCount(episodes=episodes, calls=episodes, control_episodes=controls,
                     control_calls=controls)


# --------------------------------------------------------------------------- the worker


def receiver_worker(runner: Any, source_item_id: str) -> ItemFile:
    """The RECEIVER cell worker (``kinds.KIND_TABLE``); the runner writes what it returns.

    Runner contract (WP4 supplies it, WP5 states it):

    ``runner.cell_id``                   the cell this item belongs to
    ``runner.study_seed``                the study seed bytes
    ``runner.generate``                  ``Generate`` bound to the store, pool and cost fn
    ``runner.receiver_episodes(item)``   ``(ReceiverCondition, FixedPacketPair, tool)`` triples
    ``runner.tokenizer_for(model)``      the receiver model's tokenizer

    A one-call (E2a) arm is handed ``tool=None``; **every** two-phase arm is handed a
    :class:`checker.CheckStepTool` built over :func:`mod97_pair`'s trace map, with
    ``enabled=cond.checker`` and ``max_checks=experiments.E2b.max_checks`` -- the no-check arm
    included, so both arms spend their slots and write their events identically (spec 9.3/9.6).

    Rows are built at their single sites: ``telemetry.decision_row`` (R6) for decisions,
    :func:`transmission_rows` for the assigned packet slots and ``CheckStepTool.serve`` for
    the tool events.
    """
    try:
        from agents_scaling.nextdyn import telemetry
    except ImportError as exc:                              # pragma: no cover - WP1 pending
        raise ManifestError(f"the RECEIVER worker needs nextdyn.telemetry (WP1): {exc}") from exc

    build = getattr(runner, "receiver_episodes", None)
    if build is None:                                       # pragma: no cover - WP4 pending
        raise ManifestError("the cell runner does not expose receiver_episodes(source_item_id); "
                            "the RECEIVER worker is driven by nextdyn.runner (WP4)")
    decision_rows: list[Any] = []
    transmissions: list[Any] = []
    tool_events: list[Any] = []
    for cond, pair, tool in build(source_item_id):
        tokenizer = runner.tokenizer_for(cond.model)
        episode = run_receiver_episode(cond, pair, runner.generate, tool,
                                       study_seed=runner.study_seed, tokenizer=tokenizer)
        cell = telemetry.CellContext(cell_id=runner.cell_id, episode_id=episode.episode_id,
                                     display_label=cond.display_label or cond.spec.display_label,
                                     bundle_export_hash=None)
        # the receiving model's own count of the replayed bytes (never a re-clip)
        rows = transmission_rows(episode, pair, receiver_counter=make_counter(tokenizer))
        transmissions.extend(rows)
        tool_events.extend(episode.tool_events)
        seen = tuple(row.transmission_id for row in rows)
        for decision in episode.decisions:
            is_decision_phase = decision.phase == K.PHASE_RECEIVER_DECISION
            decision_rows.append(telemetry.decision_row(
                decision.record, decision.key, decision.output, cell=cell,
                parents=(), seen_transmissions=seen if is_decision_phase else (),
                aliased=decision.aliased, reuse_of=None, is_reference=False,
                # Only the plan row carries it: a plan is the one phase whose output is not a
                # public answer, so it is the one phase whose validity is a different question.
                plan_parsed=None if is_decision_phase else episode.plan_parsed))
    return ItemFile(schema_version=SCHEMA_VERSION, cell_id=runner.cell_id,
                    cell_kind=CellKind.RECEIVER, source_item_id=source_item_id,
                    status=ItemStatus.COMPLETE, bundle_export_hash=None,
                    rows={"decisions": tuple(decision_rows),
                          "transmissions": tuple(transmissions),
                          "tool_events": tuple(tool_events)})


def panel_id_for(register_sha: str) -> str:
    """E2a's panel id: ``packets:<register sha>`` (contract §2.0 per-kind key layout)."""
    if not register_sha:
        raise ContractViolation("an E2a panel id names the sealed packets register")
    return f"packets:{register_sha}"


def episode_seed_label(cond: ReceiverCondition) -> str:      # pragma: no cover - diagnostics
    return identity.sha256_hex(identity.jcs([cond.episode_id, cond.protocol_hash]))[:16]


__all__ = [
    "ASSAYS", "ASSAY_E2A", "ASSAY_E2B", "BUDGET_DECISION", "BUDGET_ONE_CALL", "BUDGET_PLAN",
    "CHECKER_AVAILABLE", "CHECKER_UNAVAILABLE", "DECLARED_EDGE_ASSIGNED", "FixedPacketPair",
    "Generate", "GridCount", "ReceiverCondition", "ReceiverDecision", "ReceiverEpisode",
    "PHASE_TEMPLATES", "SUPPORT_HEAD", "assert_budget_split",
    "assert_generation_phases", "assert_templates",
    "assert_support_invariant_bodies", "e2a_grid_counts", "e2b_grid_counts", "panel_id_for",
    "mod97_pair", "receiver_worker", "run_receiver_episode", "support_counts",
    "support_fields", "support_panel_size", "transmission_rows",
]
