"""Harness for dispatcher-level failure injection (agents 01 + 02 only; independent of 03's graph builder).

Graphs come from ``reference_impl.reference_graph`` wrapped in contract ``ItemGraph``s; the render function puts
``NODE <node_id>`` in the prompt so ``DispatchStub`` can script faults per node; the extract function is a
minimal gold-blind ``<final>`` reader. Every observation used by the assertions comes from outside the code under
test: the stub's wire log, the fsync'd ledger journal, the raw/parsed files on disk and ``StateStore``.
"""
from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

from agent_allocation.contracts import (
    Candidate,
    GenerationResult,
    GenerationSpec,
    GridSpec,
    ItemGraph,
    LedgerEventKind,
    OutputKind,
    ParseStatus,
    PublicTask,
)

from tests.integration import helpers as h
from tests.integration.conftest import contract_lock, optional_module
from tests.integration.openrouter_stub import DispatchStub
from tests.integration.reference_impl import LedgerChecker, RefGraph

RUN_TIMEOUT_SECONDS = 60.0


def fast_config(cfg: dict[str, Any], **scheduler: Any) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["scheduler"].update({"concurrency_start": 32, "concurrency_max": 64, "requests_per_second_start": 1000,
                             "requests_per_second_max": 1000, "aimd_window_seconds": 0.05, "lease_heartbeat_seconds": 0.05,
                             "lease_stale_seconds": 5, "max_calls_per_item_in_flight": 64})
    out["scheduler"].update(scheduler)
    return out


def item_graph(ref: RefGraph, task: PublicTask, grid: GridSpec, partition: str) -> ItemGraph:
    return ItemGraph(item_id=task.item_id, partition_id=partition, grid=grid, nodes=tuple(ref.nodes.values()),
                     checkpoints=())


@dataclass
class RenderLog:
    calls: list[tuple[str, tuple[tuple[str, bool], ...], float]] = field(default_factory=list)


def make_render(log: RenderLog, loop_time):
    def render(spec: GenerationSpec, parents: Sequence[tuple[GenerationSpec, Candidate | None]]) -> list[dict[str, str]]:
        log.calls.append((spec.node_id, tuple((p.node_id, c is None) for p, c in parents), loop_time()))
        return [{"role": "system", "content": "QA harness"},
                {"role": "user", "content": f"NODE {spec.node_id}\nPARENTS {' '.join(p.node_id for p, _ in parents)}"}]
    return render


def make_extract(tasks: dict[str, PublicTask]):
    def extract(result: GenerationResult, spec: GenerationSpec) -> Candidate:
        content = result.visible_content or ""
        finals = re.findall(r"<final>(.*?)</final>", content, re.S)
        status = ParseStatus.VALID if len(finals) == 1 and finals[0].strip() else (
            ParseStatus.EMPTY if not content.strip() else (ParseStatus.MISSING if not finals else ParseStatus.MALFORMED))
        return Candidate(candidate_id=spec.node_id, node_id=spec.node_id, item_id=spec.item_id, visible_content=content,
                         final_answer=finals[0].strip() if status is ParseStatus.VALID else None, rationale=None,
                         parse_status=status, finish_reason=result.finish_reason,
                         output_kind=tasks[spec.item_id].output_kind if spec.item_id in tasks else OutputKind.MATH)
    return extract


@dataclass
class Harness:
    paths: Any
    cfg: dict[str, Any]
    graphs: list[ItemGraph]
    tasks: dict[str, PublicTask]
    stub: DispatchStub
    ceiling: Decimal = Decimal("5")
    max_unresolved: Decimal = Decimal("2")
    render_log: RenderLog = field(default_factory=RenderLog)
    summaries: list[Any] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)

    @property
    def specs(self) -> dict[str, GenerationSpec]:
        return {n.node_id: n for g in self.graphs for n in g.nodes}

    async def run(self, *, store=None, resume: bool = False, max_usd: Decimal | None = None,
                  waves: Sequence[Sequence[ItemGraph]] | None = None, expect_error: bool = False):
        state_mod = optional_module("agent_allocation.storage.state")
        journal_mod = optional_module("agent_allocation.storage.journal")
        ledger_mod = optional_module("agent_allocation.runtime.ledger")
        sched = optional_module("agent_allocation.runtime.scheduler")
        client_mod = optional_module("agent_allocation.api.client")
        if store is None:
            store = state_mod.StateStore.recover(self.paths) if resume else state_mod.StateStore(self.paths)
        store.register_nodes([n for g in (waves[0] if waves else self.graphs) for n in g.nodes], wave_index=0)
        journal = journal_mod.LedgerJournal(self.paths.ledger_journal, group_commit_ms=1)
        ledger = ledger_mod.CollectionLedger(journal, ceiling_usd=self.ceiling, max_unresolved_usd=self.max_unresolved)
        client = client_mod.OpenRouterClient(api_key=None, transport=self.stub.transport())
        loop = asyncio.get_running_loop()
        dispatcher = sched.Dispatcher(config=self.cfg, paths=self.paths, store=store, ledger=ledger, client=client,
                                      lock=contract_lock(), render=make_render(self.render_log, loop.time),
                                      extract=make_extract(self.tasks))
        plan = sched.AdmissionPlan(waves=tuple(tuple(w) for w in (waves or [self.graphs])))
        try:
            summary = await asyncio.wait_for(dispatcher.run(plan, allow_paid=True, max_usd=max_usd or self.ceiling),
                                             RUN_TIMEOUT_SECONDS)
            self.summaries.append(summary)
        except asyncio.TimeoutError:
            raise AssertionError(f"dispatcher did not terminate within {RUN_TIMEOUT_SECONDS}s (hang)")
        except Exception as exc:                                   # injected crashes surface here
            if not expect_error:
                raise
            self.errors.append(exc)
        finally:
            await client.aclose()
        return store, ledger

    # --- observations ---------------------------------------------------------------------------------------
    def journal_events(self):
        journal_mod = optional_module("agent_allocation.storage.journal")
        return list(journal_mod.LedgerJournal(self.paths.ledger_journal, group_commit_ms=1).replay())

    def ledger_violations(self, ceiling: Decimal | None = None) -> list[str]:
        return LedgerChecker(ceiling or self.ceiling).replay(self.journal_events()).violations

    def events_for(self, attempt_prefix: str):
        return [e for e in self.journal_events() if (e.attempt_id or "").startswith(attempt_prefix)]

    def raw_files(self) -> list:
        return sorted(p for p in (self.paths.run_dir / "raw").rglob("*.json.gz"))

    def parent_order_violations(self) -> list[str]:
        """Every rendered parent must be terminal: a present candidate only after its successful response came
        back; an absent candidate only for a parent that never succeeded."""
        problems = []
        for node, parents, when in self.render_log.calls:
            for parent, missing in parents:
                done = self.stub.first_ok_completed(parent)
                if not missing and (done is None):
                    problems.append(f"{node[:10]} rendered parent {parent[:10]} with a candidate it never received")
                if missing and done is not None:
                    problems.append(f"{node[:10]} rendered parent {parent[:10]} as absent although it succeeded")
        return problems

    def request_before_parent_violations(self) -> list[str]:
        problems = []
        specs = self.specs
        for req in self.stub.served:
            if req.node_id is None:
                problems.append("request without a NODE marker")
                continue
            for parent in specs[req.node_id].parent_ids:
                done = self.stub.first_ok_completed(parent)
                if done is not None and done > req.received_at:
                    problems.append(f"{req.node_id[:10]} requested before parent {parent[:10]} returned")
        return problems


def reserve_count(events) -> int:
    return sum(1 for e in events if e.kind is LedgerEventKind.RESERVE)
