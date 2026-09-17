"""Typed offline fakes for the scheduler/ledger tests; also a mock fixture downstream agents can copy.

``FakeClient.generate`` has the frozen ``OpenRouterClient.generate`` signature and returns a
``contracts.GenerationOutcome``. It holds an ``httpx.MockTransport`` (never used for I/O) so the dispatcher's
``allow_paid=False`` guard recognizes it as offline. No network, no API key.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from agent_allocation.config import load_config
from agent_allocation.contracts import (
    STUDY_ID, Candidate, CompletionStatus, EndpointLock, GenerationKind, GenerationOutcome, GenerationResult,
    GenerationSpec, GridSpec, ItemGraph, OutputKind, ParseStatus, TransportStatus, Usage,
)
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.ledger import CollectionLedger
from agent_allocation.runtime.scheduler import AdmissionPlan, Dispatcher
from agent_allocation.storage.journal import LedgerJournal
from agent_allocation.storage.state import StateStore

ROOT = Path(__file__).resolve().parents[3]
ABSENT = "[No prior candidate response is available.]"

LOCK = EndpointLock(
    model="google/gemini-2.5-flash-lite", provider_tag="google-ai-studio", provider_name="Google AI Studio",
    service_tier="standard", input_price_per_token="0.0000001", output_price_per_token="0.0000004",
    internal_reasoning_price_per_token="0.0000004", context_length=1048576, max_completion_tokens=65535,
    supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"))

_HTTP = {TransportStatus.OK: 200, TransportStatus.ERROR_ENVELOPE: 200, TransportStatus.RATE_LIMITED: 429,
         TransportStatus.AUTH_FAILED: 401, TransportStatus.PAYMENT_REQUIRED: 402, TransportStatus.BAD_REQUEST: 400,
         TransportStatus.UPSTREAM_ERROR: 502, TransportStatus.CONNECT_FAILED: None,
         TransportStatus.AMBIGUOUS_TIMEOUT: None, TransportStatus.CACHE_HIT: 200, TransportStatus.PROVIDER_MISMATCH: 200}


def spec(item_id: str, kind: GenerationKind, *, benchmark: str = "math500", partition: str = "main",
         root_slot: int = 0, team_size: int = 0, update_round: int = 0, parents: Sequence[str] = (),
         budget: int = 1024, headroom: int = 1024, draw_id: int = 0) -> GenerationSpec:
    return GenerationSpec(
        run_id=STUDY_ID, partition_id=partition, item_id=item_id, benchmark=benchmark, kind=kind, model=LOCK.model,
        endpoint_lock_hash=LOCK.lock_hash, input_manifest_hash=f"task-{item_id}", decoding_config_hash="decoding",
        prompt_version="prompts-v1-test", reasoning_budget=budget, completion_headroom=headroom, root_slot=root_slot,
        team_size=team_size, update_round=update_round, draw_id=draw_id, parent_ids=tuple(parents))


def make_graph(item_id: str, *, benchmark: str = "math500", partition: str = "main", bank: int = 5, chain: int = 3,
               team_slots: int = 3, team_rounds: int = 1, peer_sizes: Sequence[int] = (2,), peer_rounds: int = 1,
               deep: Sequence[int] = (2,)) -> ItemGraph:
    """A reduced reference-DAG shape: roots, root[0]'s private chain, private team chains, PEER rounds, DEEP."""
    nodes: list[GenerationSpec] = []
    roots = [spec(item_id, GenerationKind.ROOT, benchmark=benchmark, partition=partition, root_slot=j)
             for j in range(bank)]
    nodes.extend(roots)
    prev = roots[0]
    for r in range(1, chain):
        prev = spec(item_id, GenerationKind.PRIVATE, benchmark=benchmark, partition=partition, root_slot=0,
                    update_round=r, parents=(prev.node_id,))
        nodes.append(prev)
    for j in range(1, team_slots):
        prev = roots[j]
        for r in range(1, team_rounds + 1):
            prev = spec(item_id, GenerationKind.PRIVATE, benchmark=benchmark, partition=partition, root_slot=j,
                        update_round=r, parents=(prev.node_id,))
            nodes.append(prev)
    for n in peer_sizes:
        prev_round = [r.node_id for r in roots[:n]]
        for r in range(1, peer_rounds + 1):
            current = [spec(item_id, GenerationKind.PEER, benchmark=benchmark, partition=partition, root_slot=j,
                            team_size=n, update_round=r, parents=prev_round) for j in range(n)]
            nodes.extend(current)
            prev_round = [c.node_id for c in current]
    for m in deep:
        nodes.append(spec(item_id, GenerationKind.DEEP, benchmark=benchmark, partition=partition, budget=1024 * m))
    grid = GridSpec(name="test", bank_size=bank, private_single_cycles=chain, private_team_rounds=team_rounds,
                    team_sizes=tuple(peer_sizes), peer_rounds=peer_rounds, deep_multipliers=tuple(deep))
    return ItemGraph(item_id=item_id, partition_id=partition, grid=grid, nodes=tuple(nodes), checkpoints=())


def plan_of(*waves: Sequence[ItemGraph]) -> AdmissionPlan:
    return AdmissionPlan(waves=tuple(tuple(w) for w in waves))


@dataclass
class Behavior:
    status: TransportStatus = TransportStatus.OK
    delay: float = 0.0
    cost: Decimal | None = Decimal("0.0001")
    retry_after: float | None = None
    generation_id: str | None = None
    content: str = "<rationale>r</rationale>\n<final>42</final>"
    raise_exc: BaseException | None = None


def make_outcome(s: GenerationSpec, attempt_id: str, messages: Sequence[Mapping[str, str]], b: Behavior) -> GenerationOutcome:
    ok = b.status is TransportStatus.OK
    usage = None if b.cost is None else Usage(100, 50, 10, 0, b.cost, {"cost": str(b.cost)})
    gen_id = b.generation_id if b.generation_id is not None else (f"gen-{attempt_id[:8]}-{attempt_id[-1]}" if ok else None)
    result = GenerationResult(
        node_id=s.node_id, attempt_id=attempt_id, transport_status=b.status,
        completion_status=CompletionStatus.COMPLETED if ok else None, generation_id=gen_id,
        observed_model=s.model if ok else None, observed_provider=LOCK.provider_name if ok else None,
        finish_reason="stop" if ok else None, native_finish_reason="STOP" if ok else None,
        visible_content=b.content if ok else None, reasoning_returned=False, usage=usage,
        cache_status="MISS" if ok else None, raw_response_path=None, raw_response_sha256=None,
        latency_seconds=b.delay, http_status=_HTTP[b.status], retry_after_seconds=b.retry_after,
        error_detail=None if ok else b.status.value)
    body = {"id": gen_id, "choices": [{"message": {"content": b.content}}]} if ok else {"error": b.status.value}
    return GenerationOutcome(result=result,
                             request_body={"model": s.model, "messages": [dict(m) for m in messages],
                                           "max_tokens": s.max_completion_tokens},
                             response_body=json.dumps(body).encode(),
                             response_headers={"content-type": "application/json"})


def _refuse(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never used for I/O
    raise AssertionError("FakeClient never performs HTTP")


class FakeClient:
    def __init__(self, script: Callable[[GenerationSpec, str, int], Behavior] | None = None, *,
                 delay: float = 0.0) -> None:
        self.transport = httpx.MockTransport(_refuse)
        self.script = script
        self.delay = delay
        self.calls: list[tuple[str, str]] = []
        self.completed: list[str] = []
        self.active = 0
        self.peak = 0
        self.metadata: dict[str, list[Mapping[str, Any] | None]] = {}
        self.metadata_calls: list[str] = []

    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], *, attempt_id: str,
                       lock: EndpointLock, sampling: Mapping[str, Any]) -> GenerationOutcome:
        index = len(self.calls)
        self.calls.append((spec.node_id, attempt_id))
        behavior = self.script(spec, attempt_id, index) if self.script else Behavior(delay=self.delay)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(behavior.delay)
        finally:
            self.active -= 1
        if behavior.raise_exc is not None:
            raise behavior.raise_exc
        self.completed.append(spec.node_id)
        return make_outcome(spec, attempt_id, messages, behavior)

    async def generation_metadata(self, generation_id: str) -> Mapping[str, Any] | None:
        self.metadata_calls.append(generation_id)
        queue = self.metadata.get(generation_id)
        return queue.pop(0) if queue else None

    def sent_nodes(self) -> list[str]:
        return [n for n, _ in self.calls]


class RenderLog:
    """``RenderFn`` fake: records what each node saw from its parents."""

    def __init__(self) -> None:
        self.seen: dict[str, list[str]] = {}

    def __call__(self, s: GenerationSpec, parents: Sequence[tuple[GenerationSpec, Candidate | None]]) -> list[dict[str, str]]:
        shown = [(c.final_answer or "") if c is not None else ABSENT for _, c in parents]
        self.seen[s.node_id] = shown
        text = f"item={s.item_id} kind={s.kind.value} slot={s.root_slot} round={s.update_round}\n" + "\n".join(shown)
        return [{"role": "system", "content": "Answer inside <final>."}, {"role": "user", "content": text}]


def fake_extract(result: GenerationResult, s: GenerationSpec) -> Candidate:
    content = result.visible_content or ""
    final = None
    if "<final>" in content and "</final>" in content:
        final = content.split("<final>", 1)[1].split("</final>", 1)[0].strip()
    status = ParseStatus.VALID if final else (ParseStatus.EMPTY if not content else ParseStatus.MISSING)
    return Candidate(candidate_id=s.node_id, node_id=s.node_id, item_id=s.item_id, visible_content=content,
                     final_answer=final, rationale=None, parse_status=status, finish_reason=result.finish_reason,
                     output_kind=OutputKind.MATH)


def make_config(**scheduler: Any) -> dict[str, Any]:
    cfg = deepcopy(load_config(ROOT / "configs" / "balanced150.json"))
    cfg["scheduler"].update({
        "concurrency_start": 8, "concurrency_max": 32, "requests_per_second_start": 1000,
        "requests_per_second_max": 2000, "aimd_window_seconds": 3600, "lease_heartbeat_seconds": 0.2,
        "lease_stale_seconds": 5, "connect_timeout_seconds": 1, "read_timeout_seconds": 5,
    })
    cfg["scheduler"].update(scheduler)
    return cfg


@dataclass
class Env:
    tmp: Path
    paths: RunPaths
    config: dict[str, Any]
    group_commit_ms: int = 2
    ceiling: Decimal = Decimal("135")
    store: StateStore | None = None
    journal: LedgerJournal | None = None
    ledger: CollectionLedger | None = None
    render: RenderLog = field(default_factory=RenderLog)

    @classmethod
    def create(cls, tmp: Path, **scheduler: Any) -> "Env":
        paths = RunPaths.for_run(run_dir=tmp / "run", local_state_dir=tmp / "local")
        paths.ensure()
        env = cls(tmp, paths, make_config(**scheduler))
        env.open()
        return env

    def open(self, *, recover: bool = False) -> None:
        self.store = StateStore.recover(self.paths) if recover else StateStore(self.paths)
        self.journal = LedgerJournal(self.paths.ledger_journal, group_commit_ms=self.group_commit_ms)
        self.ledger = CollectionLedger(self.journal, ceiling_usd=self.ceiling,
                                       max_unresolved_usd=Decimal(str(self.config["budget"]["max_unresolved_spend_usd"])))

    def reopen(self, *, recover: bool = True) -> None:
        self.store.close()
        self.open(recover=recover)

    def dispatcher(self, client: Any, *, extract=fake_extract) -> Dispatcher:
        return Dispatcher(config=self.config, paths=self.paths, store=self.store, ledger=self.ledger, client=client,
                          lock=LOCK, render=self.render, extract=extract)
