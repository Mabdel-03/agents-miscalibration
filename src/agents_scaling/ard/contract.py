"""ARD36 frozen contract (WP0).

Every constant in this module is part of the frozen experimental contract for the
Agents Round Dynamics 36h study.  Parallel work packages import from here; nothing
here may change after ``FROZEN.yaml`` is written for a run root.

Spec: expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md
Plan: ~/.claude/plans/expermt-agent-specs-agents-round-dynamic-woolly-naur.md

The study package (``agents_scaling.study``) is frozen and is imported, never edited.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# --------------------------------------------------------------------------- #
# Study identity
# --------------------------------------------------------------------------- #

STUDY_ID: Final[str] = "ard36"
SEED_NAMESPACE: Final[str] = "ARD36_v1"
CONTRACT_VERSION: Final[int] = 1

# --------------------------------------------------------------------------- #
# Team and horizon (spec 7; frozen for every cohort, protocol and checkpoint)
# --------------------------------------------------------------------------- #

N_AGENTS: Final[int] = 5
UPDATE_ROUNDS: Final[int] = 3          # rounds 1, 2, 3
ROUNDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
EARLY_STOP_ON_UNANIMITY: Final[bool] = False   # spec 7 line 266: never


class Protocol(str, Enum):
    """Update rules.  Round 0 is the common neutral root team for all of them."""

    FRESH = "FRESH"                     # 5 new task-only draws each round
    PRIVATE = "PRIVATE"                 # task + own root + own latest, empty packet
    PEER = "PEER"                       # + the other four previous-round reports
    BALANCED = "BALANCED"               # one report per canonical class, multiplicity hidden
    HUB5 = "HUB5"                       # 1 leader then 4 workers
    GATE = "GATE"                       # controller protects one agent per round
    RANDOM_PROTECT = "RANDOM_PROTECT"   # random one-agent protection control
    CONF_PROTECT = "CONF_PROTECT"       # confidence-chosen protection (extension 3)


#: Protocols whose round-t packet is built from round t-1 only (spec 19 assertion).
SYNCHRONOUS_PROTOCOLS: Final[frozenset[Protocol]] = frozenset(
    {Protocol.FRESH, Protocol.PRIVATE, Protocol.PEER, Protocol.BALANCED,
     Protocol.GATE, Protocol.RANDOM_PROTECT, Protocol.CONF_PROTECT}
)
#: Protocols where exactly one agent receives an empty packet per round (spec 12.3).
PROTECTION_PROTOCOLS: Final[frozenset[Protocol]] = frozenset(
    {Protocol.GATE, Protocol.RANDOM_PROTECT, Protocol.CONF_PROTECT}
)

NATURAL_PROTOCOLS_SMALL: Final[tuple[Protocol, ...]] = (
    Protocol.FRESH, Protocol.PRIVATE, Protocol.PEER, Protocol.BALANCED, Protocol.HUB5,
)
NATURAL_PROTOCOLS_LARGE: Final[tuple[Protocol, ...]] = (
    Protocol.PRIVATE, Protocol.PEER, Protocol.BALANCED,
)
CHALLENGE_PROTOCOLS: Final[tuple[Protocol, ...]] = (
    Protocol.PRIVATE, Protocol.PEER, Protocol.BALANCED,
    Protocol.GATE, Protocol.RANDOM_PROTECT,
)
HIGH_BUDGET_PROTOCOLS: Final[tuple[Protocol, ...]] = (Protocol.PRIVATE, Protocol.PEER)
CAPTURE_PROTOCOLS: Final[tuple[Protocol, ...]] = (
    Protocol.PRIVATE, Protocol.PEER, Protocol.BALANCED,
)

# --------------------------------------------------------------------------- #
# Task families and splits (spec 5.1)
# --------------------------------------------------------------------------- #


class Family(str, Enum):
    MATH500 = "math500"
    MMLUPRO = "mmlu_pro"


FAMILIES: Final[tuple[Family, ...]] = (Family.MATH500, Family.MMLUPRO)
FAMILY_WEIGHTS: Final[dict[str, float]] = {Family.MATH500.value: 0.5,
                                           Family.MMLUPRO.value: 0.5}

SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")
#: Per family.  Totals: 96 train / 48 validation / 240 test.
SPLIT_COUNTS_PER_FAMILY: Final[dict[str, int]] = {"train": 48, "validation": 24, "test": 120}
NATURAL_TOTAL: Final[int] = 384

#: 32B frozen subset, chosen by hash WITHIN each natural split (never on measured gains).
LARGE_SUBSET_PER_FAMILY: Final[int] = 48
LARGE_SPLIT_COUNTS_PER_FAMILY: Final[dict[str, int]] = {"train": 12, "validation": 6, "test": 30}

#: MMLU-Pro is restricted to the ten-option rows so one A-J alphabet covers every item.
MMLU_REQUIRED_OPTIONS: Final[int] = 10
MC_LETTERS: Final[str] = "ABCDEFGHIJ"

# --------------------------------------------------------------------------- #
# Stateless root bank layout (spec 5.2; plan 5.1)
# --------------------------------------------------------------------------- #

ROOT_DRAWS: Final[int] = 16
TEAM_DRAWS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)      # the natural team
AUXILIARY_DRAWS: Final[tuple[int, ...]] = tuple(range(5, 16))   # E1 / difficulty only
FRESH_DRAW_BASE: Final[int] = 16                           # FRESH round t -> 16 + 5*(t-1) + j


def fresh_draw_index(round_: int, agent_slot: int) -> int:
    """Draw ordinal for FRESH's round-``round_`` agent ``agent_slot`` (1-based rounds)."""
    if not 1 <= round_ <= UPDATE_ROUNDS:
        raise ValueError(f"FRESH draws exist for rounds 1..{UPDATE_ROUNDS}, got {round_}")
    if not 0 <= agent_slot < N_AGENTS:
        raise ValueError(f"agent_slot must be in 0..{N_AGENTS - 1}, got {agent_slot}")
    return FRESH_DRAW_BASE + N_AGENTS * (round_ - 1) + agent_slot


#: Cumulative FRESH vote pool sizes after rounds 0,1,2,3 (spec 7 line 286).
FRESH_PREFIX_K: Final[tuple[int, ...]] = (5, 10, 15, 20)

# --------------------------------------------------------------------------- #
# Generation contract (spec 6.2, 6.3; plan 5.2)
# --------------------------------------------------------------------------- #

TEMPERATURE: Final[float] = 0.6
TOP_P: Final[float] = 0.95
TOP_K: Final[int] = 20
MIN_P: Final[float] = 0.0
PRESENCE_PENALTY: Final[float] = 0.0
REPETITION_PENALTY: Final[float] = 1.0
ENABLE_THINKING: Final[bool] = True
DTYPE: Final[str] = "bfloat16"          # spec 6.1: one precision throughout

THINKING_TOKENS_LOW: Final[int] = 1024
THINKING_TOKENS_HIGH: Final[int] = 4096
FINAL_TOKENS: Final[int] = 256

#: Qwen3 control ids; identical in both pinned snapshots (verified: same tokenizer blob).
THINK_START_ID: Final[int] = 151667
THINK_END_ID: Final[int] = 151668
IM_END_ID: Final[int] = 151645


def max_tokens_for(thinking_tokens: int) -> int:
    """`<think>` + thinking budget + `</think>` + the independent final allowance.

    The two delimiters are sampled output tokens, so they must be inside max_tokens or
    a cap hit would eat into the 256-token report (spec 6.2: not a shortened shared cap).
    """
    return 1 + int(thinking_tokens) + 1 + FINAL_TOKENS


#: Root decoding is ONE frozen setting used by every arm, so roots alias into the
#: high-budget arms instead of being regenerated (spec 10 E2d line 341).
ROOT_THINKING_TOKENS: Final[int] = THINKING_TOKENS_LOW

# --------------------------------------------------------------------------- #
# Report contract (spec 6.3)
# --------------------------------------------------------------------------- #

FINAL_OPEN, FINAL_CLOSE = "<final>", "</final>"
EVIDENCE_OPEN, EVIDENCE_CLOSE = "<evidence>", "</evidence>"
CONFIDENCE_OPEN, CONFIDENCE_CLOSE = "<confidence>", "</confidence>"
EVIDENCE_MAX_TOKENS: Final[int] = 120


class ParseStatus(str, Enum):
    """Outcome of parsing the post-`</think>` span.  Only OK is a delivered answer."""

    OK = "ok"
    MISSING_FINAL = "missing_final"           # no <final> tag -> incorrect answer
    MALFORMED_FINAL = "malformed_final"       # unclosed / nested / duplicate <final>
    EMPTY_FINAL = "empty_final"               # <final></final>
    NO_THINK_OPEN = "no_think_open"           # completion did not start with <think>
    PREMATURE_REASONING_EOS = "premature_reasoning_eos"   # never closed </think>
    TRUNCATED = "truncated"                   # report span cut before <final> closed
    #: Span cut by ``max_tokens``, but a well-formed ``<final>...</final>`` closed before
    #: the cut.  The answer channel is intact and the report votes; the tail (evidence,
    #: confidence) is lost, so ``confidence`` is *missing* under the frozen
    #: missing-is-missing policy (spec §6.3) and never imputed.  Amendment I7.
    TRUNCATED_AFTER_FINAL = "truncated_after_final"
    CONTEXT_FAILURE = "context_failure"       # prompt refused at build time


#: Statuses that still yield a gradeable answer string.  Validity is a property of the
#: ``<final>`` channel alone (spec §6.3: "missing/malformed ``<final>`` = invalid"), so a
#: cut tail after a closed ``<final>`` does not void the answer.
VALID_PARSE_STATUSES: Final[frozenset[ParseStatus]] = frozenset(
    {ParseStatus.OK, ParseStatus.TRUNCATED_AFTER_FINAL}
)

# --------------------------------------------------------------------------- #
# Prompt budget (plan 5.2; asserted at build time by generate_decision)
# --------------------------------------------------------------------------- #

TASK_TOKENS_CAP: Final[int] = 1024
REPORT_TOKENS_CAP: Final[int] = 256      # own root / own latest / one peer packet
TEMPLATE_OVERHEAD_CAP: Final[int] = 512
CONTEXT_RESERVE_TOKENS: Final[int] = 128
#: task + own root + own latest + four peer packets + template
PROMPT_TOKENS_CAP: Final[int] = (
    TASK_TOKENS_CAP + 2 * REPORT_TOKENS_CAP
    + (N_AGENTS - 1) * REPORT_TOKENS_CAP + TEMPLATE_OVERHEAD_CAP
)

# --------------------------------------------------------------------------- #
# Seed keys (spec 18 cache identity; reuses study.identity.semantic_seed verbatim)
# --------------------------------------------------------------------------- #

NS_STATELESS_BANK: Final[str] = "stateless_bank"     # roots AND FRESH draws
NS_PRIVATE: Final[str] = "ARD_PRIVATE"
NS_PEER: Final[str] = "ARD_PEER"
NS_BALANCED: Final[str] = "ARD_BALANCED"
NS_HUB5: Final[str] = "ARD_HUB5"
NS_GATE: Final[str] = "ARD_GATE"
NS_RANDOM: Final[str] = "ARD_RANDOM"
NS_CONF: Final[str] = "ARD_CONF"
NS_FORK: Final[str] = "ARD_FORK"
NS_SUFFIX: Final[str] = "ARD_SUFFIX"
NS_RECEIVER: Final[str] = "ARD_RECEIVER"
NS_SELECTOR: Final[str] = "ARD_SELECTOR"

PROTOCOL_NAMESPACE: Final[dict[Protocol, str]] = {
    Protocol.FRESH: NS_STATELESS_BANK,
    Protocol.PRIVATE: NS_PRIVATE,
    Protocol.PEER: NS_PEER,
    Protocol.BALANCED: NS_BALANCED,
    Protocol.HUB5: NS_HUB5,
    Protocol.GATE: NS_GATE,
    Protocol.RANDOM_PROTECT: NS_RANDOM,
    Protocol.CONF_PROTECT: NS_CONF,
}

PURPOSE_ROOT: Final[str] = "root"
PURPOSE_REVISE: Final[str] = "revise"
PURPOSE_HUB: Final[str] = "hub"
PURPOSE_RECEIVER: Final[str] = "receiver"
PURPOSE_SELECTOR: Final[str] = "selector"

#: HMAC namespaces for correctness-blind ordering decisions (study.identity.blind_order_key).
ORDER_PACKET: Final[str] = "ARD_PACKET_ORDER"
ORDER_PSEUDONYM: Final[str] = "ARD_PSEUDONYM"
ORDER_BALANCED_REP: Final[str] = "ARD_BALANCED_REP"
ORDER_HUB_LEADER: Final[str] = "ARD_HUB_LEADER"
ORDER_PROTECT: Final[str] = "ARD_PROTECT"      # RANDOM_PROTECT choice and GATE argmax ties
ORDER_VOTE_TIE: Final[str] = "VOTE_TIE"        # reuses the study tie-seed derivation

# --------------------------------------------------------------------------- #
# Cohorts, lanes, serving profiles
# --------------------------------------------------------------------------- #


class Cohort(str, Enum):
    NATURAL = "natural"
    CHALLENGE = "challenge"          # label-conditioned, test items only
    HIGH_BUDGET = "high_budget"      # E2d
    RECEIVER = "receiver"            # E3 synthetic program traces
    FORK = "fork"                    # E4 private counterfactual branches
    SUFFIX = "suffix"                # E5 common continuations
    CAPTURE = "capture"              # E6 replay cohort


LANES: Final[tuple[str, ...]] = ("8B", "32B", "grade", "analysis", "capture")

PROFILE_SMALL: Final[str] = "ard36-8B"
PROFILE_LARGE: Final[str] = "ard36-32B"
#: prompt cap + 4354 output + reserve, rounded up; NOT 40960 (KV waste) and NOT 8192
#: (the 4,096-thinking arms would not fit).
MAX_MODEL_LEN: Final[int] = 16384
MAX_NUM_SEQS_SMALL: Final[int] = 128
MAX_NUM_SEQS_LARGE: Final[int] = 64

CHECKPOINT_SMALL: Final[str] = "8B"
CHECKPOINT_LARGE: Final[str] = "32B"
#: Pinned revisions, verified present in the offline HF cache.
REVISIONS: Final[dict[str, str]] = {
    CHECKPOINT_SMALL: "b968826d9c46dd6066d109eabc6255188de91218",
    CHECKPOINT_LARGE: "9216db5781bf21249d130ec9da846c4624c16137",
}

# --------------------------------------------------------------------------- #
# Cohort sizes (spec 5.2-5.4, 10, 12, 13, 14.2)
# --------------------------------------------------------------------------- #

CHALLENGE_MAX_ITEMS: Final[int] = 96
CHALLENGE_MIN_CORRECT_ROOTS: Final[int] = 4
CHALLENGE_MIN_INCORRECT_ROOTS: Final[int] = 4
CHALLENGE_C0_LEVELS: Final[tuple[int, ...]] = (1, 4)

HIGH_BUDGET_TEST_ITEMS: Final[int] = 32          # 16 per family
RECEIVER_PROGRAMS: Final[int] = 64
RECEIVER_DEPTHS: Final[tuple[int, ...]] = (8, 16)
RECEIVER_MODULUS: Final[int] = 97
RECEIVER_COEFFICIENTS: Final[tuple[int, ...]] = tuple(range(2, 12))
RECEIVER_SEEDS: Final[tuple[int, ...]] = tuple(range(0, 97))
RECEIVER_SEEDS_PER_CONDITION: Final[int] = 2
RECEIVER_PRESENTATIONS: Final[tuple[str, ...]] = (
    "PRIVATE_TRACE", "MINORITY_UNCHECKED", "MAJORITY_UNCHECKED",
    "MINORITY_CHECKED", "MAJORITY_CHECKED",
)
RECEIVER_CHECKS_PER_PACKET: Final[int] = 2
RECEIVER_TOP_LOGPROBS: Final[int] = 20     # answer-logit margin needs more than 1

FORK_PARENT_ROUNDS: Final[tuple[int, ...]] = (1, 2)   # plus the free t=0 PRIVATE/PEER pair
SUFFIX_SMALL_TEST_ITEMS: Final[int] = 64
SUFFIX_LARGE_TEST_ITEMS: Final[int] = 60
SUFFIX_PARENT_PROTOCOLS: Final[tuple[Protocol, ...]] = (Protocol.PRIVATE, Protocol.PEER)
SUFFIX_SEEDS: Final[int] = 2

CAPTURE_TOTAL_ITEMS: Final[int] = 144
CAPTURE_SPLIT_ITEMS: Final[dict[str, int]] = {"train": 64, "validation": 16, "test": 64}
CAPTURE_RELATIVE_LAYERS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
CAPTURE_PRIMARY_LAYER: Final[float] = 0.5
CAPTURE_ANCHORS: Final[tuple[str, ...]] = (
    "PROMPT_END",        # last prompt token, before generated reasoning
    "PRE_FINAL",         # last token before <final> content
    "ANSWER_END",        # last answer-content token, before <confidence>
    "REPORT_END",        # end of the final report
)
#: Anchors whose features may be used to predict the answer that follows (spec 14.4).
PROSPECTIVE_ANCHORS: Final[frozenset[str]] = frozenset({"PROMPT_END", "PRE_FINAL"})
#: Full-vocabulary entropy is computed at these anchors only (spec 14.5).
ENTROPY_ANCHORS: Final[frozenset[str]] = frozenset({"PROMPT_END", "ANSWER_END"})

SELECTOR_AUDIT_POOLS: Final[tuple[str, ...]] = ("root", "peer_final", "gate_final")

# --------------------------------------------------------------------------- #
# Controller (spec 12)
# --------------------------------------------------------------------------- #

FEATURE_SCHEMA_VERSION: Final[int] = 1
PENALTY_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0)
AGENTS_PROTECTED_PER_ROUND: Final[int] = 1
#: Features may never be derived from these, at any grain (spec 12.2 line 420).
FORBIDDEN_FEATURE_SOURCES: Final[frozenset[str]] = frozenset({
    "item_id", "source_id", "gold", "answer_key", "reference", "is_correct",
    "correct", "label", "root_pass_rate", "pass_at_k", "constructed_C0", "C0",
    "future_state", "final_consensus", "y_peer", "y_private",
})

# --------------------------------------------------------------------------- #
# Selection (spec 7)
# --------------------------------------------------------------------------- #

SELECTORS: Final[tuple[str, ...]] = (
    "plurality", "confidence_weighted", "CURRENT", "dynamics_only", "llm_audit",
)
POOL_KINDS: Final[tuple[str, ...]] = ("root", "current", "archive")
PASS_KS: Final[tuple[int, ...]] = (1, 2, 3, 5, 9, 10, 16)

# --------------------------------------------------------------------------- #
# Output tables (spec 18 + plan 5.8)
# --------------------------------------------------------------------------- #

SPEC_TABLES: Final[tuple[str, ...]] = (
    "items", "episodes", "decisions", "labels", "features", "forks", "resources",
)
DESCRIPTIVE_TABLES: Final[tuple[str, ...]] = (
    "round_summary", "transitions", "c_dist", "drift", "bank_summary",
    "calibration_panel", "cost_summary", "cost_accuracy",
    "neural_geometry", "neural_agent", "challenge_teams", "suffix_summary",
)
ALL_TABLES: Final[tuple[str, ...]] = SPEC_TABLES + DESCRIPTIVE_TABLES

# --------------------------------------------------------------------------- #
# Statistics (spec 15)
# --------------------------------------------------------------------------- #

PRIMARY_CONTRASTS: Final[tuple[str, ...]] = (
    "GATE_minus_PEER_round3_selected_accuracy",
    "GATE_minus_RANDOM_PROTECT_round3_selected_accuracy",
    "PEER_minus_PRIVATE_root_to_round3_coverage_change",
)
N_PRIMARY: Final[int] = len(PRIMARY_CONTRASTS)
N_RESAMPLES: Final[int] = 20_000
ALPHA: Final[float] = 0.05
#: Pre-registered noninferiority margin on C0=4 challenge destruction (plan 8).
NONINFERIORITY_MARGIN: Final[float] = 0.08

__all__ = [name for name in dir() if not name.startswith("_")]
