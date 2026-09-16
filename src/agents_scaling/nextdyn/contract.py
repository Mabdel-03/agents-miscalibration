"""Parametric study contract: the *shapes* are frozen here, every *value* comes from config.

ARD36 froze N=5, three rounds, two thinking budgets and one final cap as module constants
(``ard/contract.py``), and every module that needed a different value branched on it.  That is
where D-11 (one bank doubling as roots) and the unread ``max_inflight`` came from.  This module
declares the same facts as frozen dataclasses with **no defaults and no numbers**:
:func:`agents_scaling.nextdyn.config.load_study` fills them from ``configs/nextdyn/study.yaml``,
and a test pins that no numeric experimental value lives in code.

The only literals here are *names*: seed namespaces, phase names, blind-order namespaces and
adapter names.  They are identity vocabulary (spec 17.2), not knobs.

Spec: 5.3 (banks), 5.4 (model cells), 5.5 (sampling and token contract), 6.1 (output limits),
6.5 (selectors), 15.3 (failure policy), 17.4 (compile rules).  Plan: D5 (token contract and
``max_tokens = 1 + T + 1 + F``), D6 (output limits), D9 (selectors), §1.2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Final, Mapping

from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study import identity

# --------------------------------------------------------------------------- identity vocabulary
# These are names, not knobs (spec 17.2, plan D4).  Every seed namespace starts with the prefix so
# no nextdyn request can alias an ARD36 (``ARD36_v1``/``ARD_*``) or study_v4 request.

SEED_NAMESPACE_PREFIX: Final[str] = "ND1"

PHASE_BANK: Final[str] = "bank"                       # difficulty / donor / natural-root draws (D14)
PHASE_ROOT_REF: Final[str] = "root_ref"               # episode round 0: a reference to a bank draw
PHASE_REVISION: Final[str] = "revision"               # rounds 1..R of an episode (FRESH included)
PHASE_RECEIVER_PLAN: Final[str] = "receiver_plan"     # E2b phase 1
PHASE_RECEIVER_DECISION: Final[str] = "receiver_decision"   # E2a one-call and E2b phase 2
PHASE_AUDIT: Final[str] = "audit"                     # pointwise selection audit
PHASES: Final[tuple[str, ...]] = (
    PHASE_BANK, PHASE_ROOT_REF, PHASE_REVISION, PHASE_RECEIVER_PLAN, PHASE_RECEIVER_DECISION,
    PHASE_AUDIT,
)

#: HMAC namespaces for correctness-blind ordering decisions (``study.identity.blind_order_key``).
ORDER_PACKET: Final[str] = "ND_PACKET_ORDER"          # peer packet display order per recipient
ORDER_PSEUDONYM: Final[str] = "ND_PSEUDONYM"          # slot -> pseudonym map per pairing
ORDER_PROTECT: Final[str] = "ND_PROTECT"              # natural random-protection slot (spec 10.4)
ORDER_OPTION_PERM: Final[str] = "ND_OPTION_PERM"      # MC option permutation per item variant
ORDER_WSTAR: Final[str] = "ND_WSTAR"                  # w* among qualifying wrong classes (spec 8.2)
ORDER_DONOR_ORDER: Final[str] = "ND_DONOR_ORDER"      # donor order within a class at construction
ORDER_SLOT: Final[str] = "ND_SLOT"                    # slot permutation of a constructed team
ORDER_RING: Final[str] = "ND_RING"                    # ring permutation (E3b, per round or once)
ORDER_DISPATCH: Final[str] = "ND_DISPATCH"            # manifest order (spec 17.2 randomized dispatch)
ORDER_PASSK: Final[str] = "ND_PASSK"                  # blind prefix for pass@k / vote@k panels
VOTE_TIE: Final[str] = "VOTE_TIE"                     # plurality tie key (spec 6.5), study convention
ORDER_NAMESPACES: Final[tuple[str, ...]] = (
    ORDER_PACKET, ORDER_PSEUDONYM, ORDER_PROTECT, ORDER_OPTION_PERM, ORDER_WSTAR,
    ORDER_DONOR_ORDER, ORDER_SLOT, ORDER_RING, ORDER_DISPATCH, ORDER_PASSK, VOTE_TIE,
)

ADAPTER_NATIVE_THINKING: Final[str] = "native_thinking"
ADAPTER_TOTAL_OUTPUT_ENVELOPE: Final[str] = "total_output_envelope"
ADAPTERS: Final[tuple[str, ...]] = (ADAPTER_NATIVE_THINKING, ADAPTER_TOTAL_OUTPUT_ENVELOPE)

#: Selector names that are evaluation-only ceilings and may never appear in a deployed list.
NON_DEPLOYABLE_SELECTORS: Final[frozenset[str]] = frozenset({"oracle", "oracle_analysis"})

#: Study modes; ``confirmation`` is the only one the compiler holds to the null-lock rule.
MODE_TEST: Final[str] = "test"
MODE_PILOT_ONLY: Final[str] = "pilot_only"
MODE_CONFIRMATION: Final[str] = "confirmation"
MODES: Final[tuple[str, ...]] = (MODE_TEST, MODE_PILOT_ONLY, MODE_CONFIRMATION)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation(f"{name} must be a non-negative int, got {value!r}")
    return value


# --------------------------------------------------------------------------- token contract


@dataclass(frozen=True)
class TokenContract:
    """One generation-phase allowance (spec 5.5; plan D5).

    ``thinking_cap`` is vLLM's ``thinking_token_budget``; ``final_cap`` is the independent
    post-``</think>`` allowance that stays available even when thinking hits its cap.  The
    receiver phases use ``B/2`` contracts; the audit uses its own.  ``packet_cap``,
    ``rationale_cap``, ``claims_max`` and ``claim_cap`` are the public-output limits of spec
    6.1 (rationale/claim overruns are recorded as flags, never rewritten).
    """

    thinking_cap: int
    final_cap: int
    served_window: int
    reserve: int
    packet_cap: int
    rationale_cap: int
    claims_max: int
    claim_cap: int

    def __post_init__(self) -> None:
        for name in ("thinking_cap", "final_cap", "served_window", "reserve", "packet_cap",
                     "rationale_cap", "claims_max", "claim_cap"):
            _positive(f"TokenContract.{name}", getattr(self, name))
        if self.final_cap < 1:
            raise ContractViolation("TokenContract.final_cap must be at least 1 token")
        if self.max_new_tokens + self.reserve > self.served_window:
            raise ContractViolation(
                f"TokenContract: 1+{self.thinking_cap}+1+{self.final_cap} + reserve {self.reserve} "
                f"does not fit the served window {self.served_window}")

    @property
    def max_new_tokens(self) -> int:
        """``<think>`` + thinking + ``</think>`` + final: the two delimiters are sampled tokens,
        so they sit inside ``max_tokens`` or a cap hit eats the final allowance (amendment A1:
        3074 for the 2048/1024 main contract, not the spec's 3072)."""
        return 1 + self.thinking_cap + 1 + self.final_cap

    def budget_hash(self) -> str:
        """``CompositeKey.budget_hash``: the full contract, canonical JSON, sha256 hex."""
        return identity.sha256_hex(identity.jcs(dataclasses.asdict(self)))


# --------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class Sampling:
    """Spec 5.5 sampling recipe with floats as fixed strings (JCS forbids floats)."""

    temperature: str
    top_p: str
    top_k: int
    min_p: str
    repetition_penalty: str
    presence_penalty: str
    frequency_penalty: str

    @classmethod
    def from_floats(cls, *, temperature: float, top_p: float, top_k: int, min_p: float,
                    repetition_penalty: float, presence_penalty: float,
                    frequency_penalty: float) -> "Sampling":
        return cls(
            temperature=identity.float_str(temperature), top_p=identity.float_str(top_p),
            top_k=_positive("Sampling.top_k", top_k), min_p=identity.float_str(min_p),
            repetition_penalty=identity.float_str(repetition_penalty),
            presence_penalty=identity.float_str(presence_penalty),
            frequency_penalty=identity.float_str(frequency_penalty),
        )

    def as_floats(self) -> dict[str, float | int]:
        """The values a decoding object needs (``study.types.Decoding`` takes floats)."""
        return {
            "temperature": float(self.temperature), "top_p": float(self.top_p),
            "top_k": int(self.top_k), "min_p": float(self.min_p),
            "repetition_penalty": float(self.repetition_penalty),
            "presence_penalty": float(self.presence_penalty),
            "frequency_penalty": float(self.frequency_penalty),
        }


# --------------------------------------------------------------------------- banks, team, models


@dataclass(frozen=True)
class BankSpec:
    """Spec 5.3: three disjoint banks per (item, model, budget)."""

    difficulty_draws: int
    donor_initial: int
    donor_max: int
    natural_root_draws: int

    def __post_init__(self) -> None:
        for name in ("difficulty_draws", "donor_initial", "donor_max", "natural_root_draws"):
            if _positive(f"BankSpec.{name}", getattr(self, name)) < 1:
                raise ContractViolation(f"BankSpec.{name} must be >= 1")
        if self.donor_max < self.donor_initial:
            raise ContractViolation("BankSpec.donor_max must be >= donor_initial")


@dataclass(frozen=True)
class TeamShape:
    """N agents, R synchronous revisions, and the rounds at which outcomes are scored."""

    n_agents: int
    revision_rounds: int
    score_rounds: tuple[int, ...]

    def __post_init__(self) -> None:
        if _positive("TeamShape.n_agents", self.n_agents) < 2:
            raise ContractViolation("TeamShape.n_agents must be >= 2")
        if _positive("TeamShape.revision_rounds", self.revision_rounds) < 1:
            raise ContractViolation("TeamShape.revision_rounds must be >= 1")
        rounds = tuple(int(r) for r in self.score_rounds)
        object.__setattr__(self, "score_rounds", rounds)
        if not rounds or rounds[0] != 0 or any(r > self.revision_rounds for r in rounds):
            raise ContractViolation(
                f"TeamShape.score_rounds {rounds} must start at 0 and stay <= revision_rounds")
        if list(rounds) != sorted(set(rounds)):
            raise ContractViolation("TeamShape.score_rounds must be strictly increasing")


@dataclass(frozen=True)
class ModelCell:
    """One pinned model + adapter + serving profile (spec 5.4, 5.6).

    ``revision`` / ``tokenizer_revision`` may be ``None`` only for a disabled cell; the lock
    and :func:`agents_scaling.nextdyn.ids.model_hash` refuse a null revision.

    ``lane`` is the generation lane the cell's requests run on (``8B`` / ``32B``; a scale
    class, not the served name, so the family-transfer entry that serves as ``llama8B``
    still compiles onto the ``8B`` lane and blocks only at request time, plan D11).

    ``total_output_tokens`` is the ``total_output_envelope`` adapter's single cap (spec 5.6:
    3,072 for both companions) and must be null on a native-thinking cell: the native cap
    is ``TokenContract.thinking_cap`` and lives in one place only.  It is part of the
    identity object, so ``StudyContract.sha256`` changes with it.
    """

    name: str
    hf_id: str
    revision: str | None
    tokenizer_revision: str | None
    adapter: str
    profile: str
    tp: int
    served_model_name: str
    lane: str
    total_output_tokens: int | None

    def __post_init__(self) -> None:
        if self.adapter not in ADAPTERS:
            raise ContractViolation(
                f"ModelCell {self.name}: adapter {self.adapter!r} not in {ADAPTERS}")
        if _positive(f"ModelCell.{self.name}.tp", self.tp) < 1:
            raise ContractViolation(f"ModelCell {self.name}: tp must be >= 1")
        if not self.name or not self.hf_id or not self.profile or not self.served_model_name \
                or not self.lane:
            raise ContractViolation(f"ModelCell {self.name!r}: name, hf_id, profile, "
                                    "served_model_name and lane must be non-empty")
        if self.adapter == ADAPTER_TOTAL_OUTPUT_ENVELOPE:
            if self.total_output_tokens is None:
                raise ContractViolation(f"ModelCell {self.name}: the {ADAPTER_TOTAL_OUTPUT_ENVELOPE} "
                                        "adapter needs total_output_tokens (spec 5.6)")
            if _positive(f"ModelCell.{self.name}.total_output_tokens", self.total_output_tokens) < 1:
                raise ContractViolation(f"ModelCell {self.name}: total_output_tokens must be >= 1")
        elif self.total_output_tokens is not None:
            raise ContractViolation(f"ModelCell {self.name}: total_output_tokens belongs to the "
                                    f"{ADAPTER_TOTAL_OUTPUT_ENVELOPE} adapter; a native-thinking cell "
                                    "takes its caps from TokenContract only (R1)")

    @property
    def pinned(self) -> bool:
        return bool(self.revision) and bool(self.tokenizer_revision)


# --------------------------------------------------------------------------- selection, failures


@dataclass(frozen=True)
class SelectionSpec:
    """Spec 6.5 selectors.  ``secondary_deployed`` lists deployable secondaries; the
    analysis-only ceilings live in ``analysis_only`` so the compiler can refuse an oracle in
    a deployed list (spec 17.4)."""

    primary: str
    pool_scope: str
    tie_break: str
    secondary_deployed: tuple[str, ...]
    analysis_only: tuple[str, ...]
    audit_model: str
    audit_thinking_tokens: int
    audit_final_tokens: int
    audit_failure_fallback: str
    synthesis_allowed: bool
    gold_access: bool
    archive_audit_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "secondary_deployed", tuple(self.secondary_deployed))
        object.__setattr__(self, "analysis_only", tuple(self.analysis_only))
        if self.synthesis_allowed or self.gold_access:
            raise ContractViolation("selection.synthesis_allowed and selection.gold_access are "
                                    "frozen at false (spec 6.5: select an existing candidate)")

    @property
    def deployed(self) -> tuple[str, ...]:
        return (self.primary, *self.secondary_deployed)


@dataclass(frozen=True)
class FailurePolicy:
    """Spec 15.3 operational failure policy, every field read by the client or the runner."""

    transient_retries: int
    retry_same_request_and_seed: bool
    model_output_repair_calls: int
    wrong_answer_retry: bool
    context_overflow: str
    unresolved_semantics: str
    null_item_ids: str
    duplicate_logical_keys: str
    hidden_label_leakage: str

    def __post_init__(self) -> None:
        _positive("FailurePolicy.transient_retries", self.transient_retries)
        if self.model_output_repair_calls != 0 or self.wrong_answer_retry:
            raise ContractViolation("model outputs are never repaired or re-rolled "
                                    "(spec 15.3: model_output_repair_calls 0, wrong_answer_retry false)")
        if not self.retry_same_request_and_seed:
            raise ContractViolation("infrastructure retries must keep the exact request and seed")
        if self.context_overflow != "typed_failure":
            raise ContractViolation("context_overflow must be 'typed_failure' (never a retry)")
        if self.unresolved_semantics != "unknown":
            raise ContractViolation("unresolved semantics are 'unknown' (U), never W")
        for name in ("null_item_ids", "duplicate_logical_keys", "hidden_label_leakage"):
            if getattr(self, name) != "fatal":
                raise ContractViolation(f"failure_policy.{name} must be 'fatal'")


# --------------------------------------------------------------------------- the contract


@dataclass(frozen=True)
class StudyContract:
    """Everything the identity scheme and the generation path need, in one frozen object.

    ``receiver_tokens`` is keyed by the receiver budget label (``"1024"``, ``"4096"``) and
    holds the per-phase ``B/2`` contract; ``audit_tokens`` is the pointwise-audit contract.
    """

    study_id: str
    study_seed_hex: str
    split_salt_hex: str
    contract_version: int
    families: tuple[str, ...]
    models: Mapping[str, ModelCell]
    sampling: Sampling
    tokens: TokenContract
    receiver_tokens: Mapping[str, TokenContract]
    audit_tokens: TokenContract
    banks: BankSpec
    team: TeamShape
    selection: SelectionSpec
    failure_policy: FailurePolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", tuple(self.families))
        object.__setattr__(self, "models", dict(self.models))
        object.__setattr__(self, "receiver_tokens", dict(self.receiver_tokens))
        for name, hexstr in (("study_seed_hex", self.study_seed_hex),
                             ("split_salt_hex", self.split_salt_hex)):
            try:
                raw = bytes.fromhex(hexstr)
            except (TypeError, ValueError) as exc:
                raise ContractViolation(f"{name} is not hex: {hexstr!r}") from exc
            if len(raw) < 16:
                raise ContractViolation(f"{name} must be at least 16 bytes of hex")
        if not self.families:
            raise ContractViolation("StudyContract.families is empty")
        if not self.models:
            raise ContractViolation("StudyContract.models is empty")
        for name, cell in self.models.items():
            if cell.name != name:
                raise ContractViolation(f"model cell {name!r} carries name {cell.name!r}")

    @property
    def study_seed(self) -> bytes:
        return bytes.fromhex(self.study_seed_hex)

    @property
    def split_salt(self) -> bytes:
        return bytes.fromhex(self.split_salt_hex)

    def model(self, name: str) -> ModelCell:
        try:
            return self.models[name]
        except KeyError as exc:
            raise ContractViolation(f"unknown model cell {name!r}; have {sorted(self.models)}") from exc

    def identity_object(self) -> dict[str, Any]:
        """JCS-safe form: every dataclass expanded, mappings sorted by JCS."""
        return {
            "study_id": self.study_id,
            "study_seed_hex": self.study_seed_hex,
            "split_salt_hex": self.split_salt_hex,
            "contract_version": int(self.contract_version),
            "families": list(self.families),
            "models": {k: dataclasses.asdict(v) for k, v in self.models.items()},
            "sampling": dataclasses.asdict(self.sampling),
            "tokens": dataclasses.asdict(self.tokens),
            "receiver_tokens": {k: dataclasses.asdict(v) for k, v in self.receiver_tokens.items()},
            "audit_tokens": dataclasses.asdict(self.audit_tokens),
            "banks": dataclasses.asdict(self.banks),
            "team": dataclasses.asdict(self.team),
            "selection": dataclasses.asdict(self.selection),
            "failure_policy": dataclasses.asdict(self.failure_policy),
        }

    def sha256(self) -> str:
        return identity.sha256_hex(identity.jcs(self.identity_object()))


# --------------------------------------------------------------------------- field consumers (R2)

_M = "agents_scaling.nextdyn."

#: ``(dataclass, field) -> consuming module``: the module that reads that attribute of the
#: compiled contract.  Rule R2 applied to the contract itself: a YAML leaf that flows into a
#: contract field satisfies the config-leaf guard, so without this table a contract field could
#: be filled and then read by nobody (the D-03 shape one level up).  Checked by
#: ``tests/nextdyn/test_guards.py::test_every_contract_dataclass_field_is_read_outside_contract_py``:
#: a consumer that has not landed is *pending*, a landed one must read the attribute by name.
CONTRACT_FIELD_CONSUMERS: Final[Mapping[tuple[str, str], str]] = {
    ("TokenContract", "thinking_cap"): _M + "adapters",
    ("TokenContract", "final_cap"): _M + "adapters",
    ("TokenContract", "served_window"): _M + "inference",
    ("TokenContract", "reserve"): _M + "inference",
    ("TokenContract", "packet_cap"): _M + "packets",
    ("TokenContract", "rationale_cap"): _M + "schema",
    ("TokenContract", "claims_max"): _M + "schema",
    ("TokenContract", "claim_cap"): _M + "schema",
    ("Sampling", "temperature"): _M + "decoding",
    ("Sampling", "top_p"): _M + "decoding",
    ("Sampling", "top_k"): _M + "decoding",
    ("Sampling", "min_p"): _M + "decoding",
    ("Sampling", "repetition_penalty"): _M + "decoding",
    ("Sampling", "presence_penalty"): _M + "decoding",
    ("Sampling", "frequency_penalty"): _M + "decoding",
    ("BankSpec", "difficulty_draws"): _M + "banks",
    ("BankSpec", "donor_initial"): _M + "banks",
    ("BankSpec", "donor_max"): _M + "banks",
    ("BankSpec", "natural_root_draws"): _M + "banks",
    ("TeamShape", "n_agents"): _M + "episode",
    ("TeamShape", "revision_rounds"): _M + "episode",
    # analysis reads the contract and hands the set to outcomes.derive, which takes it as an
    # argument: spec 11.2 generates to r=8 once and scores prefixes, so the scored set is a
    # study-level fact, not something the estimator should re-derive from the data it sees.
    ("TeamShape", "score_rounds"): _M + "analysis",
    ("ModelCell", "name"): _M + "kinds",
    ("ModelCell", "hf_id"): _M + "locks",
    ("ModelCell", "revision"): _M + "locks",
    ("ModelCell", "tokenizer_revision"): _M + "locks",
    ("ModelCell", "adapter"): _M + "locks",
    ("ModelCell", "profile"): _M + "fleet",
    ("ModelCell", "tp"): _M + "fleet",
    ("ModelCell", "served_model_name"): _M + "inference",
    ("ModelCell", "lane"): _M + "kinds",
    ("ModelCell", "total_output_tokens"): _M + "adapters",
    ("SelectionSpec", "primary"): _M + "selectors",
    ("SelectionSpec", "pool_scope"): _M + "audit",
    ("SelectionSpec", "tie_break"): _M + "selectors",
    ("SelectionSpec", "secondary_deployed"): _M + "selectors",
    ("SelectionSpec", "analysis_only"): _M + "analysis",
    ("SelectionSpec", "audit_model"): _M + "audit",
    ("SelectionSpec", "audit_thinking_tokens"): _M + "config",
    ("SelectionSpec", "audit_final_tokens"): _M + "config",
    ("SelectionSpec", "audit_failure_fallback"): _M + "selectors",
    ("SelectionSpec", "synthesis_allowed"): _M + "config",
    ("SelectionSpec", "gold_access"): _M + "config",
    ("SelectionSpec", "archive_audit_enabled"): _M + "plan",
    ("FailurePolicy", "transient_retries"): _M + "inference",
    ("FailurePolicy", "retry_same_request_and_seed"): _M + "inference",
    ("FailurePolicy", "model_output_repair_calls"): _M + "inference",
    ("FailurePolicy", "wrong_answer_retry"): _M + "inference",
    ("FailurePolicy", "context_overflow"): _M + "inference",
    ("FailurePolicy", "unresolved_semantics"): _M + "grading",
    ("FailurePolicy", "null_item_ids"): _M + "inventory",
    ("FailurePolicy", "duplicate_logical_keys"): _M + "materialize",
    ("FailurePolicy", "hidden_label_leakage"): _M + "bundles",
    ("StudyContract", "study_id"): _M + "cells",
    ("StudyContract", "study_seed_hex"): _M + "inference",
    ("StudyContract", "split_salt_hex"): _M + "inventory",
    ("StudyContract", "contract_version"): _M + "freeze",
    ("StudyContract", "families"): _M + "inventory",
    ("StudyContract", "models"): _M + "cells",
    ("StudyContract", "sampling"): _M + "decoding",
    ("StudyContract", "tokens"): _M + "cells",
    ("StudyContract", "receiver_tokens"): _M + "receiver",
    ("StudyContract", "audit_tokens"): _M + "audit",
    ("StudyContract", "banks"): _M + "banks",
    ("StudyContract", "team"): _M + "episode",
    ("StudyContract", "selection"): _M + "selectors",
    ("StudyContract", "failure_policy"): _M + "inference",
}

#: The dataclasses the field-consumer guard covers (every field of each must appear above).
CONTRACT_DATACLASSES: Final[tuple[type, ...]] = (
    TokenContract, Sampling, BankSpec, TeamShape, ModelCell, SelectionSpec, FailurePolicy, StudyContract,
)


__all__ = [
    "ADAPTERS", "ADAPTER_NATIVE_THINKING", "CONTRACT_DATACLASSES", "CONTRACT_FIELD_CONSUMERS", "ADAPTER_TOTAL_OUTPUT_ENVELOPE", "BankSpec",
    "FailurePolicy", "MODES", "MODE_CONFIRMATION", "MODE_PILOT_ONLY", "MODE_TEST", "ModelCell",
    "NON_DEPLOYABLE_SELECTORS", "ORDER_DISPATCH", "ORDER_DONOR_ORDER", "ORDER_NAMESPACES",
    "ORDER_OPTION_PERM", "ORDER_PACKET", "ORDER_PASSK", "ORDER_PROTECT", "ORDER_PSEUDONYM",
    "ORDER_RING", "ORDER_SLOT", "ORDER_WSTAR", "PHASES", "PHASE_AUDIT", "PHASE_BANK",
    "PHASE_RECEIVER_DECISION", "PHASE_RECEIVER_PLAN", "PHASE_REVISION", "PHASE_ROOT_REF",
    "SEED_NAMESPACE_PREFIX", "Sampling", "SelectionSpec", "StudyContract", "TeamShape",
    "TokenContract", "VOTE_TIE",
]
