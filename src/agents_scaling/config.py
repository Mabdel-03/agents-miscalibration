"""Single config schema for one experiment cell.

An ``ExperimentCell`` is the atomic unit of the sweep: one combination of the three
scaling axes plus a topology, benchmark, and seed. It is fully serializable so that a
SLURM array task can reconstruct it from ``cells.json`` by index, and so that every
result record can carry the exact config that produced it (config hash in ``meta.json``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class ContextShareLevel(str, Enum):
    """Axis 2: what one agent sees of another's work.

    Ordered smallest -> largest. ``PLUS_INTERMEDIATE`` includes everything in
    ``ARTIFACT_ONLY``; ``PLUS_COT`` includes everything in ``PLUS_INTERMEDIATE``.
    """

    ARTIFACT_ONLY = "artifact_only"      # final answer string only
    PLUS_INTERMEDIATE = "plus_intermediate"  # + scratchpad / sub-answers / tool outputs
    PLUS_COT = "plus_cot"                # + raw chain-of-thought tokens

    @property
    def rank(self) -> int:
        return {"artifact_only": 0, "plus_intermediate": 1, "plus_cot": 2}[self.value]


class ReasoningLevel(str, Enum):
    """Axis 4: per-agent reasoning effort, from none to maximum.

    Ordered smallest -> largest. Implemented for Qwen3 unified models via the
    ``enable_thinking`` chat-template kwarg plus a ``thinking_token_budget`` cap:
      OFF       -> enable_thinking=False (plain instruct, no <think> block)
      B512/2048/8192 -> enable_thinking=True with that thinking-token budget
      UNLIMITED -> enable_thinking=True, no budget (bounded only by max_tokens)
    """

    OFF = "off"
    B512 = "b512"
    B2048 = "b2048"
    B8192 = "b8192"
    UNLIMITED = "unlimited"

    @property
    def rank(self) -> int:
        return {"off": 0, "b512": 1, "b2048": 2, "b8192": 3, "unlimited": 4}[self.value]

    @property
    def enable_thinking(self) -> bool:
        return self is not ReasoningLevel.OFF

    @property
    def thinking_budget(self) -> int | None:
        """Thinking-token cap, or None for OFF (no thinking) and UNLIMITED (no cap)."""
        return {"b512": 512, "b2048": 2048, "b8192": 8192}.get(self.value)


class Topology(str, Enum):
    SINGLE_AGENT = "single_agent"    # baseline: defines P_SA, T_SAS, E_SAS
    INDEPENDENT = "independent"      # n agents, no peer context, majority vote
    DECENTRALIZED = "decentralized"  # all-to-all debate over `rounds`
    CENTRALIZED = "centralized"      # orchestrator + sub-agents (star)

    @property
    def is_multi_agent(self) -> bool:
        return self is not Topology.SINGLE_AGENT

    @property
    def shares_context(self) -> bool:
        """Whether the topology routes inter-agent messages through the message builder.

        Independent agents never see peers, so the context-share axis is a no-op for
        them (collapsed to a canonical value during sweep generation).
        """
        return self in (Topology.DECENTRALIZED, Topology.CENTRALIZED)


@dataclass(frozen=True)
class ExperimentCell:
    """One cell of the sweep. Frozen so it is hashable and cannot drift mid-run."""

    # --- the four scaling axes ---
    model_size: str                      # key into the model registry, e.g. "8B"
    context_share_level: ContextShareLevel
    prompt_complexity_level: int         # 0..3, index into the prompt ladder
    reasoning_level: ReasoningLevel      # none -> max thinking effort

    # --- experimental setup ---
    topology: Topology
    benchmark: str                       # "gpqa" | "mmlu_pro" | "math" | "truthfulqa"

    # --- knobs held fixed across the headline sweep (overridable per config) ---
    n_agents: int = 3
    rounds: int = 2                      # debate rounds / orchestrator iterations
    n_samples: int = 5                   # samples per question for self-consistency
    temperature: float = 0.7
    n_questions: int | None = None       # None -> full benchmark split
    seed: int = 0

    def __post_init__(self) -> None:
        # Allow YAML to pass plain strings; coerce to enums.
        if not isinstance(self.context_share_level, ContextShareLevel):
            object.__setattr__(
                self, "context_share_level", ContextShareLevel(self.context_share_level)
            )
        if not isinstance(self.reasoning_level, ReasoningLevel):
            object.__setattr__(self, "reasoning_level", ReasoningLevel(self.reasoning_level))
        if not isinstance(self.topology, Topology):
            object.__setattr__(self, "topology", Topology(self.topology))

    @property
    def cell_id(self) -> str:
        """Stable short id; same config -> same id (good for resume / dedup)."""
        return (
            f"{self.model_size}_{self.topology.value}_{self.context_share_level.value}"
            f"_p{self.prompt_complexity_level}_r{self.reasoning_level.value}"
            f"_{self.benchmark}_s{self.seed}"
        )

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["context_share_level"] = self.context_share_level.value
        d["reasoning_level"] = self.reasoning_level.value
        d["topology"] = self.topology.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentCell":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# Weights stay on scratch (large, regenerable cache). Results moved to the tpoggio DATA
# filesystem (8T+ free) after the per-user SCRATCH quota repeatedly hit EDQUOT and halted
# the run — data has a group quota with ample headroom, so writes never block there.
DEFAULT_HF_HOME = "/orcd/scratch/orcd/012/mabdel03/.cache/huggingface"
DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"
