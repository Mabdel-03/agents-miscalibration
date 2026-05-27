"""The per-question result record (one JSONL row) and the per-cell meta record.

One row captures everything needed to (a) score performance, (b) compute efficiency, and
(c) compute calibration under multiple confidence definitions — both per-agent and at the
system level — without re-running anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuestionResult:
    cell_id: str
    qid: str
    benchmark: str
    model_size: str
    topology: str
    context_share_level: str
    prompt_complexity_level: int
    reasoning_level: str

    final_answer: str | None
    answer_key: str
    correct: bool

    per_agent: list[dict[str, Any]] = field(default_factory=list)   # AgentOutput.to_dict()
    system_conf: dict[str, float] = field(default_factory=dict)     # multiple definitions
    self_consistency: dict[str, Any] = field(default_factory=dict)  # samples + agreement
    efficiency_raw: dict[str, Any] = field(default_factory=dict)    # turns/messages/tokens/wall
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellMeta:
    cell_id: str
    config: dict[str, Any]            # ExperimentCell.to_dict()
    config_hash: str
    model_hf_id: str
    served_model_name: str
    prompt_token_count: int
    prompt_quality: dict[str, Any]    # heuristic + llm_judge + features
    git_commit: str | None = None
    n_questions: int = 0
    mean_reasoning_tokens: float = 0.0  # Axis 4 measured attribute (mean thinking tokens/q)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
