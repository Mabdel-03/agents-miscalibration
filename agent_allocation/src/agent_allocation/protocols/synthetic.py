"""Deterministic two-item synthetic run for downstream agents (02 scheduler/ledger, 06 selection) -- offline, no gold.

* Item A: an AIME-style math task on the ``main_extension`` grid (138 nodes; every family including CP/POOLED).
* Item B: a GPQA-Diamond-style four-option task on the ``main`` grid (125 nodes).

Every node gets a synthetic :class:`Candidate` (mostly VALID with a few competing answers; some raw MISSING text;
some EMPTY, which renders as absent downstream; root slot 2 carries an over-envelope rationale so its receivers show
trimming) and a synthetic :class:`GenerationResult` with exact-decimal usage. Every node's request is rendered by the
production renderer from its parents' synthetic candidates, in topological order, exactly as the dispatcher would.

``write_synthetic_run(out_dir)`` writes gzip JSONL (gzip mtime 0) plus ``MANIFEST.json`` (sha256 of each file's
*decompressed* bytes). Records use the one codec: one ``dump_record`` per line (``prompts`` lines are
``{"node_id", "item_id", "messages", "report"}``). Regenerate with
``uv run --frozen --no-sync python -m agent_allocation.protocols.synthetic tests/fixtures/protocols/two_item_run``.

Config keys consumed (R9): ``seed``, ``primary_model``, ``reasoning.main_a``, the headroom keys via
``config.headroom_for``, the decoding keys via ``config.decoding_config_hash`` and the grids via
``config.grids_from_config`` -- all from ``configs/balanced150.json``.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from decimal import Decimal
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Any

from agent_allocation.config import decoding_config_hash, grids_from_config, headroom_for, load_config
from agent_allocation.contracts import (
    STUDY_ID, Candidate, CompletionStatus, EndpointLock, GenerationKind, GenerationResult, GenerationSpec, ItemGraph,
    OutputKind, ParseStatus, PublicTask, TransportStatus, Usage, dump_record,
)
from agent_allocation.paths import PROJECT_ROOT
from agent_allocation.prompts.render import PACKET_ENVELOPE_CHARS, prompt_version, render_messages_with_report
from agent_allocation.protocols.grid import build_item_graph

CONFIG_PATH = PROJECT_ROOT / "configs" / "balanced150.json"
PARTITION = "main"
INPUT_PRICE = Decimal("0.0000001")
OUTPUT_PRICE = Decimal("0.0000004")
FILES = ("tasks.jsonl.gz", "graphs.jsonl.gz", "candidates.jsonl.gz", "results.jsonl.gz", "prompts.jsonl.gz")

TEXT_TASK = PublicTask(
    item_id="aime/synthetic/0001", benchmark="aime", source_revision="synthetic",
    prompt="Compute $\\frac{1}{2} + \\frac{1}{2}$. Simplify your answer.", output_kind=OutputKind.MATH)
OPTION_TASK = PublicTask(
    item_id="gpqa_diamond/synthetic0002", benchmark="gpqa_diamond", source_revision="synthetic",
    prompt=("Which particle mediates the electromagnetic force?\n\nOptions:\n"
            "A. gluon\nB. photon\nC. W boson\nD. graviton"),
    output_kind=OutputKind.OPTION, allowed_choices=("A", "B", "C", "D"))

SYNTHETIC_LOCK = EndpointLock(
    model="google/gemini-2.5-flash-lite", provider_tag="google-ai-studio", provider_name="Google AI Studio",
    service_tier="standard", input_price_per_token=str(INPUT_PRICE), output_price_per_token=str(OUTPUT_PRICE),
    internal_reasoning_price_per_token=str(OUTPUT_PRICE), context_length=1048576, max_completion_tokens=65535,
    supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"))

_TEXT_ANSWERS = ("1", "1", "1", "\\frac{2}{2}", "2")
_OPTION_ANSWERS = ("B", "B", "B", "A", "C")


def _h(node: GenerationSpec) -> int:
    return int(node.node_id[:8], 16)


def synthetic_candidate(node: GenerationSpec, task: PublicTask) -> Candidate:
    """Deterministic, gold-blind synthetic candidate for one node (a function of its node_id and coordinates)."""
    h = _h(node)
    status = ParseStatus.VALID
    finish = "stop"
    option = task.output_kind is OutputKind.OPTION
    answers = _OPTION_ANSWERS if option else _TEXT_ANSWERS
    answer = answers[h % len(answers)]
    rationale = ("The photon carries the electromagnetic interaction." if option
                 else f"Half plus half; the result is {answer}.")
    if node.kind is GenerationKind.ROOT and node.root_slot == 2:
        rationale = " ".join(f"Check {i}: the decisive step holds." for i in range(PACKET_ENVELOPE_CHARS // 30 + 50))
    visible = f"<rationale>\n{rationale}\n</rationale>\n<final>\n{answer}\n</final>"
    final: str | None = answer
    if h % 23 == 0:
        status, finish, visible, final, rationale = ParseStatus.EMPTY, "length", "", None, None
    elif h % 11 == 0:
        status, visible, final, rationale = ParseStatus.MISSING, f"I believe the result is {answer}.", None, None
    return Candidate(candidate_id=node.node_id, node_id=node.node_id, item_id=node.item_id, visible_content=visible,
                     final_answer=final, rationale=rationale, parse_status=status, finish_reason=finish,
                     output_kind=task.output_kind)


def synthetic_result(node: GenerationSpec, candidate: Candidate, rendered_chars: int) -> GenerationResult:
    h = _h(node)
    reasoning = min(node.reasoning_budget, 256 + h % 700)
    prompt_tokens = ceil(rendered_chars / 4)
    completion_tokens = reasoning + ceil(len(candidate.visible_content) / 4)
    cost = prompt_tokens * INPUT_PRICE + completion_tokens * OUTPUT_PRICE
    empty = candidate.parse_status is ParseStatus.EMPTY
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, reasoning_tokens=reasoning,
                  cached_prompt_tokens=0, actual_cost_usd=cost,
                  raw_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost": str(cost)})
    return GenerationResult(
        node_id=node.node_id, attempt_id=f"{node.node_id}:1", transport_status=TransportStatus.OK,
        completion_status=CompletionStatus.LENGTH if empty else CompletionStatus.COMPLETED,
        generation_id=f"gen-synthetic-{node.node_id[:16]}", observed_model=node.model,
        observed_provider=SYNTHETIC_LOCK.provider_name, finish_reason=candidate.finish_reason,
        native_finish_reason="MAX_TOKENS" if empty else "STOP", visible_content=candidate.visible_content,
        reasoning_returned=False, usage=usage, cache_status="MISS", raw_response_path=None, raw_response_sha256=None,
        latency_seconds=1.0 + (h % 900) / 100, http_status=200)


def build_synthetic_run() -> dict[str, Any]:
    """In-memory run: ``tasks``, ``graphs``, ``candidates``/``results`` (by node_id), ``prompts`` (topological)."""
    cfg = load_config(CONFIG_PATH)
    grids = grids_from_config(cfg)
    seed = int(cfg["seed"])
    version = prompt_version()
    tasks = (TEXT_TASK, OPTION_TASK)
    graphs: list[ItemGraph] = []
    candidates: dict[str, Candidate] = {}
    results: dict[str, GenerationResult] = {}
    prompts: list[dict[str, Any]] = []
    for task, grid_name in ((TEXT_TASK, "main_extension"), (OPTION_TASK, "main")):
        graph = build_item_graph(grids[grid_name], run_id=STUDY_ID, partition_id=PARTITION, task=task,
                                 model=cfg["primary_model"], base_budget=int(cfg["reasoning"]["main_a"]),
                                 headroom=headroom_for(cfg, task.benchmark), endpoint_lock_hash=SYNTHETIC_LOCK.lock_hash,
                                 decoding_config_hash=decoding_config_hash(cfg), prompt_version=version)
        graphs.append(graph)
        index = {node.node_id: node for node in graph.nodes}
        for node in graph.nodes:
            parents = [(index[p], candidates[p]) for p in node.parent_ids]
            messages, report = render_messages_with_report(node, task, parents, seed=seed)
            candidate = synthetic_candidate(node, task)
            candidates[node.node_id] = candidate
            results[node.node_id] = synthetic_result(node, candidate, report["rendered_chars"])
            prompts.append({"node_id": node.node_id, "item_id": task.item_id, "messages": messages, "report": report})
    return {"seed": seed, "prompt_version": version, "tasks": tasks, "graphs": graphs, "candidates": candidates,
            "results": results, "prompts": prompts}


def _jsonl(rows: list[Any]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def synthetic_run_payloads(run: dict[str, Any] | None = None) -> dict[str, bytes]:
    """Decompressed file contents keyed by file name, plus ``MANIFEST.json``."""
    run = run or build_synthetic_run()
    payloads = {
        "tasks.jsonl.gz": _jsonl([dump_record(t) for t in run["tasks"]]),
        "graphs.jsonl.gz": _jsonl([dump_record(g) for g in run["graphs"]]),
        "candidates.jsonl.gz": _jsonl([dump_record(c) for c in run["candidates"].values()]),
        "results.jsonl.gz": _jsonl([dump_record(r) for r in run["results"].values()]),
        "prompts.jsonl.gz": _jsonl(run["prompts"]),
    }
    statuses: dict[str, int] = {}
    for candidate in run["candidates"].values():
        statuses[candidate.parse_status.value] = statuses.get(candidate.parse_status.value, 0) + 1
    manifest = {
        "generator": "agent_allocation.protocols.synthetic",
        "study_id": STUDY_ID,
        "partition_id": PARTITION,
        "seed": run["seed"],
        "prompt_version": run["prompt_version"],
        "packet_envelope_chars": PACKET_ENVELOPE_CHARS,
        "endpoint_lock_hash": SYNTHETIC_LOCK.lock_hash,
        "items": [{"item_id": g.item_id, "benchmark": g.nodes[0].benchmark, "grid": g.grid.name,
                   "nodes": len(g.nodes), "checkpoints": len(g.checkpoints)} for g in run["graphs"]],
        "candidate_parse_status_counts": dict(sorted(statuses.items())),
        "trimmed_prompts": sum(1 for p in run["prompts"] if p["report"]["trimmed_packets"]),
        "files_sha256_decompressed": {name: sha256(data).hexdigest() for name, data in payloads.items()},
        "notes": "Synthetic, gold-free. Load records with contracts.load_record; prompts lines are plain JSON.",
    }
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return payloads


def write_synthetic_run(out_dir: str | Path) -> dict[str, bytes]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = synthetic_run_payloads()
    for name, data in payloads.items():
        if name.endswith(".gz"):
            buffer = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
                handle.write(data)
            (out / name).write_bytes(buffer.getvalue())
        else:
            (out / name).write_bytes(data)
    return payloads


def read_payload(path: str | Path) -> bytes:
    path = Path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "tests" / "fixtures" / "protocols" / "two_item_run")
    written = write_synthetic_run(target)
    print(json.dumps({name: len(data) for name, data in written.items()}, indent=2))
