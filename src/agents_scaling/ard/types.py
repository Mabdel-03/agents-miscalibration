"""ARD36 dispatch vocabulary: cell specs, episode and state records (WP4).

Spec §10 (E2 modules → cells), §12.3 (GATE needs a frozen controller), §13 (E5 parent
states), §18 (``episodes``/``decisions``/``forks`` field lists), §19 (``build_episode`` /
``run_round`` / ``fork_private`` contracts).  Plan §5.3 (cell kinds and sizes), §5.8
(measurement grains), §10 (``state:`` / ``controller:`` gating).

Why a new ``ArdCellSpec`` instead of :class:`agents_scaling.study.types.CellSpec`: the
study cell is the v4 factorial (``method``/``framing``/``N``/``B``/``degree``) and its
vocabulary is frozen.  ARD36's axes are protocol × cohort × checkpoint × reasoning
allowance × parent state, so it gets its own record and a *projection*
(:meth:`ArdCellSpec.as_study_cell`) onto the frozen ``CellSpec`` for the parts of
``study/runner.py`` that are reused verbatim (resume, ``incomplete/``, ``SUSPENDED.json``,
``meta.json``-last, the per-cell flock).  Nothing here re-declares a value frozen in
:mod:`agents_scaling.ard.contract`.

Lanes: :data:`agents_scaling.ard.contract.LANES` is the study lane tuple with
``('grade', 'analysis', 'capture')` appended additively; the three duplicated study lane
tuples (``study/cells.py``, ``slurm/study_loops.py``, ``slurm/study_launch_chunked.py``)
are **not** edited — the ARD lane tuple is re-exported here and the ARD forks of the two
slurm drivers import it, so the frozen study path keeps its own five lanes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.study.types import (
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    CellKind,
    CellSpec,
    Framing,
    Method,
    ProtocolError,
    _Record,
)

ARD_SCHEMA_VERSION: int = 1

#: The single synthetic item id of a whole-run analysis cell (COMPACT / E0): those kinds run
#: once over the run root, not per public item, so this id is deliberately absent from
#: ``items.parquet`` and the runner does not look it up there.
ANALYSIS_ITEM: str = "ALL"

#: The ARD36 lane tuple (contract §"Cohorts, lanes, serving profiles"), re-exported so no
#: ARD module re-declares it and no study module is edited.
LANES: tuple[str, ...] = K.LANES

#: New exit code (contract §3: hard dependency gating is new plumbing).  ``run_one`` returns
#: it when a ``controller:`` / ``state:`` dependency is absent or tampered; the cell writes
#: no ``meta.json`` and generates no request, so the lane driver re-queues it later instead
#: of counting it failed.  0/2/3/4 stay exactly as ``study.types`` defines them.
EXIT_DEPS_UNMET: int = 5

EXIT_CODES: tuple[int, ...] = (
    EXIT_DONE, EXIT_NO_SERVER, EXIT_INCOMPLETE, EXIT_SUSPENDED, EXIT_DEPS_UNMET,
)


class ArdCellKind(str, Enum):
    """What one ARD36 cell does (plan §5.3).

    ``EPISODE`` covers every protocol of the natural and high-budget cohorts (the protocol
    is a field, not a kind), so GATE / RANDOM_PROTECT / CONF_PROTECT differ from PEER only
    by ``protocol`` and by the ``controller:``/``schedule`` dependency they carry.
    """

    ROOTS = "ROOTS"                    # the stateless bank (16 draws), no protocol
    EPISODE = "EPISODE"                # 3 update rounds of one protocol
    CHALLENGE = "CHALLENGE"            # constructed teams (root candidate ids only)
    FORK = "FORK"                      # one private counterfactual branch off a PEER state
    SUFFIX = "SUFFIX"                  # one common private continuation off a round-1 state
    RECEIVER = "RECEIVER"              # frozen-content receiver assay (WP5)
    SELECTOR_AUDIT = "SELECTOR_AUDIT"  # set-wise LLM audit selector (WP5)
    GRADE = "GRADE"                    # CPU evaluator lane
    E0 = "E0"                          # CPU legacy reanalysis (WP7)
    COMPACT = "COMPACT"                # CPU compaction to the §18 + §5.8 tables
    CAPTURE = "CAPTURE"                # GPU replay capture (WP6)


#: Kinds whose work is one or more episodes of the frozen round loop.
EPISODE_KINDS: frozenset[ArdCellKind] = frozenset({
    ArdCellKind.EPISODE, ArdCellKind.CHALLENGE, ArdCellKind.FORK, ArdCellKind.SUFFIX,
})

# --------------------------------------------------------------------------- dependencies

CELL_DEP_PREFIX = "cell:"
CONTROLLER_DEP_PREFIX = "controller:"
STATE_DEP_PREFIX = "state:"
#: Prefixes resolved against a register **before** any work is admitted (plan §10).
HARD_DEP_PREFIXES: tuple[str, ...] = (CONTROLLER_DEP_PREFIX, STATE_DEP_PREFIX)

_SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9]{1,16}$")
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,180}$")


class ArdManifestError(ValueError):
    """A manifest request the frozen ARD36 design does not admit."""


def controller_dependency(sha: str) -> str:
    """``controller:<sha>`` — the frozen controller a GATE cell refuses to run without."""
    return CONTROLLER_DEP_PREFIX + _check_sha(sha, "controller")


def state_dependency(sha: str) -> str:
    """``state:<sha>`` — the sealed round-t state register a FORK/SUFFIX cell reads."""
    return STATE_DEP_PREFIX + _check_sha(sha, "state")


def cell_dependency(cell_id: str) -> str:
    """``cell:<cell_id>`` — ordering advice (the study ``meta.json`` rule), never a gate."""
    if not _CELL_ID_RE.match(str(cell_id)):
        raise ArdManifestError(f"bad cell id in a dependency: {cell_id!r}")
    return CELL_DEP_PREFIX + str(cell_id)


def _check_sha(sha: str, what: str) -> str:
    text = str(sha).lower()
    if not _SHA_RE.match(text):
        raise ArdManifestError(f"{what} dependency needs 8..64 lowercase hex, got {sha!r}")
    return text


def parse_dependency(dep: str) -> tuple[str, str]:
    """``('controller'|'state'|'cell', value)`` for one ``depends_on`` entry."""
    for prefix, kind in ((CONTROLLER_DEP_PREFIX, "controller"), (STATE_DEP_PREFIX, "state"),
                         (CELL_DEP_PREFIX, "cell")):
        if dep.startswith(prefix):
            return kind, dep[len(prefix):]
    # A bare cell id keeps the study convention (``unmet_dependencies`` treats it as advice).
    return "cell", dep


def hard_dependencies(depends_on: Sequence[str]) -> list[str]:
    """The entries that gate admission (``controller:`` / ``state:``), in manifest order."""
    return [d for d in depends_on if d.startswith(HARD_DEP_PREFIXES)]


# --------------------------------------------------------------------------- cell spec

#: Keys a cell option block may never carry: the manifest is read by generation workers,
#: and every one of these is label-conditioned (spec §5.3, §12.2 line 420).  ``constructed_C0``
#: is deliberately absent from :class:`ArdCellSpec` itself for the same reason — the
#: challenge manifest lists root candidate ids only, and C0 lives in ``challenge_teams``.
FORBIDDEN_OPTION_KEYS: frozenset[str] = K.FORBIDDEN_FEATURE_SOURCES | {"answer", "solution"}


@dataclass(frozen=True)
class ArdCellSpec(_Record):
    """One unit of ARD36 Slurm work.

    ``reasoning_allowance`` is the *update* thinking budget (roots always alias the frozen
    1,024-token bank, contract ``ROOT_THINKING_TOKENS``); ``final_allowance`` is the
    independent report allowance.  ``team_id`` names a constructed challenge team (never
    its composition), ``parent_ref`` the sealed state a FORK/SUFFIX cell continues, and
    ``leader_seed`` the recorded HUB5 leader draw.  ``instance`` is the wave tag that keeps
    two dispatches of one dependent stage in different directories.
    """

    cell_id: str
    kind: ArdCellKind
    protocol: K.Protocol | None
    round: int | None
    cohort: K.Cohort
    checkpoint: str
    reasoning_allowance: int
    final_allowance: int
    items: tuple[str, ...]
    team_id: str | None
    parent_ref: str | None
    leader_seed: str | None
    depends_on: tuple[str, ...]
    max_inflight: int
    parallel_items: int
    lane: str
    split: str = "test"
    instance: str | None = None
    shard: int = 0
    episode_rep: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    _enum_fields: ClassVar[Mapping[str, type[Enum]]] = {
        "kind": ArdCellKind, "protocol": K.Protocol, "cohort": K.Cohort,
    }

    def __post_init__(self) -> None:
        if not _CELL_ID_RE.match(self.cell_id):
            raise ArdManifestError(f"bad cell id {self.cell_id!r}")
        if self.lane not in LANES:
            raise ArdManifestError(f"cell {self.cell_id}: lane {self.lane!r} not in {LANES}")
        if self.checkpoint not in (K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE):
            raise ArdManifestError(f"cell {self.cell_id}: unknown checkpoint {self.checkpoint!r}")
        if self.split not in K.SPLITS:
            raise ArdManifestError(f"cell {self.cell_id}: unknown split {self.split!r}")
        if self.reasoning_allowance not in (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH):
            raise ArdManifestError(
                f"cell {self.cell_id}: reasoning_allowance {self.reasoning_allowance} is not a "
                f"frozen budget ({K.THINKING_TOKENS_LOW} / {K.THINKING_TOKENS_HIGH})")
        if self.final_allowance != K.FINAL_TOKENS:
            raise ArdManifestError(f"cell {self.cell_id}: final_allowance is frozen at {K.FINAL_TOKENS}")
        if not self.items:
            raise ArdManifestError(f"cell {self.cell_id}: a cell with no items is never dispatched")
        if len(set(self.items)) != len(self.items):
            raise ArdManifestError(f"cell {self.cell_id}: duplicate item ids")
        if self.max_inflight < 1 or self.parallel_items < 1:
            raise ArdManifestError(f"cell {self.cell_id}: concurrency must be >= 1")
        if self.round is not None and not 0 <= int(self.round) <= K.UPDATE_ROUNDS:
            raise ArdManifestError(f"cell {self.cell_id}: round {self.round} outside {K.ROUNDS}")
        if self.instance is not None and not _TAG_RE.match(self.instance):
            raise ArdManifestError(f"cell {self.cell_id}: instance tag {self.instance!r} must match {_TAG_RE.pattern}")
        leaked = set(self.options) & FORBIDDEN_OPTION_KEYS
        if leaked:
            raise ArdManifestError(
                f"cell {self.cell_id}: options carry label-conditioned keys {sorted(leaked)}; "
                "eligibility, constructed_C0 and correctness live in the analysis-side tables")
        for dep in self.depends_on:
            parse_dependency(dep)

    # ---- derived ---------------------------------------------------------------------
    @property
    def hard_dependencies(self) -> list[str]:
        return hard_dependencies(self.depends_on)

    @property
    def needs_controller(self) -> bool:
        return any(d.startswith(CONTROLLER_DEP_PREFIX) for d in self.depends_on)

    @property
    def namespace(self) -> str:
        """Seed namespace of this cell's update decisions (contract ``PROTOCOL_NAMESPACE``)."""
        if self.protocol is None:
            return K.NS_STATELESS_BANK
        return K.PROTOCOL_NAMESPACE[self.protocol]

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if value is None and name in cls._enum_fields:
            return None
        return super()._decode_field(name, value)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["options"] = dict(self.options)
        return data

    def as_study_cell(self) -> CellSpec:
        """Projection onto the frozen :class:`CellSpec` for the reused runner machinery.

        Only the fields ``study/runner.py`` actually reads on the reused path carry
        meaning: ``cell_id`` (paths and the flock), ``items`` (resume), ``depends_on``
        (advice), ``max_inflight`` / ``parallel_items`` (pools), ``lane`` and ``split``.
        The v4 factorial fields carry the documented placeholders (``Method.BANK`` /
        ``Framing.NATIVE`` / ``B=0``), exactly as ``study/cells.py`` does for its
        evaluation-kind cells; ``budget_flops()`` is never called on an ARD cell.
        """
        return CellSpec(
            cell_id=self.cell_id,
            kind=CellKind.GENERATE,
            module="ard",
            method=Method.BANK,
            checkpoint=self.checkpoint,
            N=K.N_AGENTS,
            B=0,
            framing=Framing.NATIVE,
            episode_rep=int(self.episode_rep),
            split=self.split,
            items=tuple(self.items),
            depends_on=tuple(self.depends_on),
            max_inflight=int(self.max_inflight),
            parallel_items=int(self.parallel_items),
            lane=self.lane,
            degree=None,
        )


# --------------------------------------------------------------------------- run records


@dataclass(frozen=True)
class DecisionRef(_Record):
    """One logical decision inside an episode (spec §18 ``decisions.parquet`` grain).

    The heavy payload (token ids, prompt bytes) stays in the content-addressed request
    store; this is the joinable row plan §5.8 compacts.  ``fallback`` marks a decision
    produced by the WP4 baseline executor rather than by ``ard/episode.py`` — such rows are
    never counted as protocol evidence (see :mod:`agents_scaling.ard.compact`).
    """

    decision_id: str
    episode_id: str
    request_id: str
    agent_id: int
    round: int
    phase: str
    protocol: str
    checkpoint: str
    parent_state_id: str | None = None
    parent_decision_id: str | None = None
    root_report_id: str | None = None
    prior_report_id: str | None = None
    seen_message_ids: tuple[str, ...] = ()
    message_order: tuple[int, ...] = ()
    n_packets: int = 0
    packet_tokens: int = 0
    protected: bool = False
    answer_raw: str | None = None
    answer_class: str | None = None
    content_sha256: str | None = None      # report content identity (the second change rule)
    evidence_text: str | None = None
    confidence: float | None = None
    parse_status: str = K.ParseStatus.OK.value
    thinking_tokens: int = 0
    final_tokens: int = 0
    prompt_tokens: int = 0
    cached_prefill_tokens: int = 0
    budget_tokens: int = K.THINKING_TOKENS_LOW
    cap_hit: bool = False
    boundary_injected: bool = False
    final_overrun: bool = False
    answer_logprob: float | None = None
    #: Every physical request the logical decision cost.  A cap-hit draw is generated in
    #: two phases (spec §6.2: "Count both physical requests and any repeated prefill in
    #: runtime cost"), so counting one per decision under-reports the run's real cost.
    physical_request_ids: tuple[str, ...] = ()
    finish_reason: str = ""
    latency_s: float = 0.0
    arrival_ts: float = 0.0
    endpoint_id: str = ""
    flops: dict[str, Any] = field(default_factory=dict)
    aliased: bool = False
    fallback: bool = False

    @property
    def valid(self) -> bool:
        """A delivered answer: an ``ok`` parse with a keyable canonical class."""
        return self.parse_status == K.ParseStatus.OK.value and self.answer_class is not None


@dataclass(frozen=True)
class RoundState(_Record):
    """The team state after round ``round`` (``state:<sha>`` register grain, plan §4.3)."""

    state_id: str
    episode_id: str
    round: int
    decision_ids: tuple[str, ...]
    answer_classes: tuple[str | None, ...]
    valid: tuple[bool, ...]
    protected_slot: int | None = None
    leader_id: int | None = None

    @property
    def n_valid(self) -> int:
        return sum(1 for v in self.valid if v)


@dataclass(frozen=True)
class EpisodeRecord(_Record):
    """One episode (spec §18 ``episodes.parquet``); ``constructed_C0`` is deliberately absent.

    Eligibility counts and ``constructed_C0`` are computed offline from labels and live in
    ``challenge_teams`` (spec §5.3), never in a manifest, a prompt or a feature.
    """

    episode_id: str
    item_id: str
    cohort: str
    protocol: str
    checkpoint: str
    checkpoint_revision: str
    N: int = K.N_AGENTS
    horizon: int = K.UPDATE_ROUNDS
    reasoning_allowance: int = K.THINKING_TOKENS_LOW
    final_allowance: int = K.FINAL_TOKENS
    packet_policy: str = ""
    memory_policy: str = "task+own_root+own_latest+packet"
    leader_id: int | None = None
    team_id: str | None = None
    parent_ref: str | None = None
    root_candidate_ids: tuple[str, ...] = ()
    root_seed_ids: tuple[str, ...] = ()
    condition_manifest_hash: str = ""
    decisions: tuple[DecisionRef, ...] = ()
    states: tuple[RoundState, ...] = ()
    events: tuple[dict[str, Any], ...] = ()

    _nested_seq: ClassVar[Mapping[str, type]] = {"decisions": DecisionRef, "states": RoundState}

    def round_decisions(self, round_: int) -> tuple[DecisionRef, ...]:
        return tuple(d for d in self.decisions if int(d.round) == int(round_))

    def state(self, round_: int) -> RoundState | None:
        for state in self.states:
            if int(state.round) == int(round_):
                return state
        return None


@dataclass(frozen=True)
class ArdItemResult(_Record):
    """One item file ``cells/<cell_id>/items/<item_id>.json``.

    ``source_id`` (not ``item_id``) is the key, because the reused
    ``study/runner.item_file_valid`` reads exactly that field plus ``cell.cell_id`` and
    ``status`` when it decides whether an item is already done.
    """

    schema_version: int
    cell: ArdCellSpec
    source_id: str
    family: str
    split: str
    status: str                       # "complete" | "context_failure"
    episodes: tuple[EpisodeRecord, ...]
    timing: dict[str, Any]
    worker: dict[str, Any]
    grades: tuple[dict[str, Any], ...] = ()
    code_version: str | None = None

    _nested: ClassVar[Mapping[str, type]] = {"cell": ArdCellSpec}
    _nested_seq: ClassVar[Mapping[str, type]] = {"episodes": EpisodeRecord}

    @property
    def item_id(self) -> str:
        return self.source_id


def episode_id_for(cell: ArdCellSpec, item_id: str, *, episode_rep: int = 0) -> str:
    """Stable episode id: the cell already carries protocol, cohort, budget and instance."""
    rep = f".e{int(episode_rep)}" if episode_rep else ""
    return f"{cell.cell_id}|{item_id}{rep}"


def decision_id_for(episode_id: str, *, round_: int, agent_id: int, phase: str) -> str:
    return f"{episode_id}|r{int(round_)}|a{int(agent_id)}|{phase}"


__all__ = [
    "ANALYSIS_ITEM", "ARD_SCHEMA_VERSION", "ArdCellKind", "ArdCellSpec", "ArdItemResult", "ArdManifestError",
    "CELL_DEP_PREFIX", "CONTROLLER_DEP_PREFIX", "STATE_DEP_PREFIX", "HARD_DEP_PREFIXES",
    "DecisionRef", "EPISODE_KINDS", "EXIT_CODES", "EXIT_DEPS_UNMET", "EXIT_DONE",
    "EXIT_INCOMPLETE", "EXIT_NO_SERVER", "EXIT_SUSPENDED", "EpisodeRecord",
    "FORBIDDEN_OPTION_KEYS", "LANES", "ProtocolError", "RoundState", "cell_dependency",
    "controller_dependency", "decision_id_for", "episode_id_for", "hard_dependencies",
    "parse_dependency", "state_dependency",
]
