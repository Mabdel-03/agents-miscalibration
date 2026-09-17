"""Small builders shared by the acceptance tests. Only frozen signatures (docs/00_contract.md section 2) are used."""
from __future__ import annotations

import dataclasses
import json
import re
from decimal import Decimal
from typing import Any, Mapping, Sequence

from agent_allocation.contracts import (
    STUDY_ID,
    Attempt,
    AttemptState,
    Candidate,
    CheckpointView,
    CompletionStatus,
    GenerationOutcome,
    GenerationResult,
    GenerationSpec,
    GridSpec,
    ItemGraph,
    OutputKind,
    ParseStatus,
    PolicyCheckpoint,
    PublicTask,
    TransportStatus,
    Usage,
)

from tests.integration import adversarial
from tests.integration.openrouter_stub import completion_payload
from tests.integration.reference_impl import RefGraph, reference_graph

MODEL = adversarial.MODEL
LOCK_HASH = adversarial.LOCK_HASH
DECODING_HASH = adversarial.DECODING_HASH
PROMPT_VERSION = adversarial.PROMPT_VERSION
T0 = "2026-09-17T07:00:00+00:00"


def headroom_for(task: PublicTask) -> int:
    return 2048 if task.benchmark == "mbpp_plus" else 1024


def real_prompt_version() -> str:
    """The renderer refuses specs frozen under another template set, so nodes that will be rendered must carry the
    on-disk prompt version; before 03 is merged the adversarial constant is used."""
    try:
        from agent_allocation.prompts.render import prompt_version
    except ModuleNotFoundError:
        return PROMPT_VERSION
    return prompt_version()


def graph_kwargs(task: PublicTask, *, partition: str = "main", budget: int = 1024,
                 prompt_version: str | None = None) -> dict[str, Any]:
    return dict(run_id=STUDY_ID, partition_id=partition, task=task, model=MODEL, base_budget=budget,
                headroom=headroom_for(task), endpoint_lock_hash=LOCK_HASH, decoding_config_hash=DECODING_HASH,
                prompt_version=prompt_version or real_prompt_version())


def production_graph(grid_module, grid: GridSpec, task: PublicTask, **kw) -> ItemGraph:
    return grid_module.build_item_graph(grid, **graph_kwargs(task, **kw))


def ref_graph(grid: GridSpec, task: PublicTask, **kw) -> RefGraph:
    return reference_graph(grid, **graph_kwargs(task, **kw))


def checkpoint(graph: ItemGraph, name: str, view: CheckpointView = CheckpointView.CURRENT) -> PolicyCheckpoint:
    matches = [c for c in graph.checkpoints if c.name == name and c.view == view]
    assert len(matches) == 1, f"expected exactly one checkpoint {name}/{view.value}, found {len(matches)}"
    return matches[0]


def marker(node_id: str) -> str:
    return f"MARK{node_id[:20].upper()}"


MARK_RE = re.compile(r"MARK[0-9A-F]{20}")


def marked_content(spec: GenerationSpec) -> str:
    m = marker(spec.node_id)
    return f"<rationale>rationale {m}R</rationale>\n<final>{m}</final>"


def valid_candidate(spec: GenerationSpec, task: PublicTask, content: str | None = None,
                    final: str | None = None) -> Candidate:
    content = content if content is not None else marked_content(spec)
    if final is None:
        found = re.search(r"<final>(.*?)</final>", content, re.S)
        final = found.group(1).strip() if found else None
    return Candidate(candidate_id=spec.node_id, node_id=spec.node_id, item_id=spec.item_id, visible_content=content,
                     final_answer=final, rationale=None, parse_status=ParseStatus.VALID if final else ParseStatus.MISSING,
                     finish_reason="stop", output_kind=task.output_kind)


def usage(prompt: int = 101, completion: int = 307, reasoning: int | None = 211, cost: str = "0.000131") -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=reasoning,
                 cached_prompt_tokens=0, actual_cost_usd=Decimal(cost), raw_usage={"cost": cost})


def ok_result(spec: GenerationSpec, content: str, *, attempt_no: int = 1, use: Usage | None = None) -> GenerationResult:
    return GenerationResult(node_id=spec.node_id, attempt_id=f"{spec.node_id}:{attempt_no}",
                            transport_status=TransportStatus.OK, completion_status=CompletionStatus.COMPLETED,
                            generation_id=f"gen-{spec.node_id[:8]}-{attempt_no}", observed_model=MODEL,
                            observed_provider="Google AI Studio", finish_reason="stop", native_finish_reason="STOP",
                            visible_content=content, reasoning_returned=False, usage=use or usage(),
                            cache_status="MISS", raw_response_path=None, raw_response_sha256=None,
                            latency_seconds=1.0, http_status=200)


def outcome_for(result: GenerationResult, messages: Sequence[Mapping[str, str]] = ()) -> GenerationOutcome:
    body = completion_payload(result.visible_content or "", generation_id=result.generation_id or "gen")
    return GenerationOutcome(result=result, request_body={"model": MODEL, "messages": list(messages)},
                             response_body=json.dumps(body).encode(), response_headers={"x-openrouter-cache-status": "MISS"})


def persist_success(store, raw_module, paths, spec: GenerationSpec, task: PublicTask, content: str) -> GenerationResult:
    """The documented order: attempt recorded -> raw written atomically -> terminal result committed."""
    attempt_id = f"{spec.node_id}:1"
    store.record_attempt(Attempt(attempt_id=attempt_id, node_id=spec.node_id, attempt_no=1, state=AttemptState.SENT,
                                 reservation_usd=Decimal("0.002"), created_at=T0, sent_at=T0))
    result = ok_result(spec, content)
    rel, sha = raw_module.write_raw_response(paths, attempt_id, outcome_for(result))
    final = dataclasses.replace(result, raw_response_path=rel, raw_response_sha256=sha)
    store.record_result(final, valid_candidate(spec, task, content))
    return final


def all_text(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def is_candidate_output(kind: OutputKind) -> bool:
    return kind in (OutputKind.MATH, OutputKind.OPTION, OutputKind.BOOLEAN, OutputKind.PYTHON)
