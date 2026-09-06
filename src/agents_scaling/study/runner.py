"""Resumable per-cell worker: ``run_cell(cell, run_root, server_run_root, shard, stop_event)`` (WP5).

Spec §10.4 (resumable idempotent workers; atomic publish; INCOMPLETE infrastructure
records, never model-invalid candidates; on preemption stop admitting work, commit
completed artifacts), §6.5 (overshoot → suspend the cell, never trim), §9.1/§11.2
(generate cells refuse to run without the frozen manifest; stale hashes rejected), §3.6
(seals gate evaluation), §4.4 (JUDGE_BEST companion outside B, ``companion_cost``),
§3.3 (judge / official tests).  Architecture §3 steps 1–10; corrections P0-3 (evaluation
kinds consume a seal), P0-4 (``parallel_items`` outer pool, per-item inner pool), §4
items 11–12 (judge outputs under ``eval/``; ``incomplete/`` blocks ``meta.json``).

Exit codes (``types``): ``EXIT_DONE`` 0 (every item published, ``meta.json`` written — or a
SIGUSR1 stop with in-flight items finished and NO ``meta.json``), ``EXIT_NO_SERVER`` 2
(no live endpoint within the wait; the driver re-queues), ``EXIT_INCOMPLETE`` 3 (some
items ended in ``incomplete/`` after the bounded retries), ``EXIT_SUSPENDED`` 4
(``ProtocolError`` → ``SUSPENDED.json``; needs a human).

Injection points (tests): ``tokenizer`` (must be the server's), ``oracle_factory``,
``client_factory``, ``bcb_evaluator``, ``clock``.  Nothing here reads protected data
except through :mod:`agents_scaling.study.evaluation` (§10.6 firewall).
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.cells import SEAL_DEP_PREFIX, cell_seal
from agents_scaling.study.config import FROZEN_FILENAME, FrozenStudy, StudyConfig, load_config
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.evaluation import bcb as bcb_eval
from agents_scaling.study.evaluation import hle_judge
from agents_scaling.study.inference.client import EndpointPool, VllmChatClient
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.inference.tokens import load_tokenizer
from agents_scaling.study.policies import policy_for
from agents_scaling.study.policies.base import EPISODE_SCHEMA_VERSION, EpisodeContext
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import EpisodeLedger, StopReason
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.selection.judge_best import parse_judge_best
from agents_scaling.study.selection.vote import keyed_pool
from agents_scaling.study.types import (
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    JUDGE_DECODING,
    NS_JUDGE,
    NS_STATELESS_BANK,
    PURPOSE_JUDGE_BEST,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    CellKind,
    CellSpec,
    Checkpoint,
    Domain,
    EpisodeResult,
    InfraFailure,
    Method,
    ProtocolError,
    PublicTask,
    RequestSpec,
    SeedKey,
)

R_BANK = 10
ROLE_JUDGE_BEST = "judge_best"
ITEM_KIND_KEY = "kind"
DEFAULT_WAIT_S = 300.0


@dataclass
class CellPaths:
    root: Path
    cell_dir: Path
    items: Path
    incomplete: Path
    events: Path
    meta: Path
    suspended: Path

    @classmethod
    def of(cls, run_root: str | os.PathLike, cell_id: str) -> "CellPaths":
        root = Path(run_root)
        cell_dir = root / "cells" / cell_id
        return cls(root, cell_dir, cell_dir / "items", cell_dir / "incomplete", cell_dir / "events.jsonl", cell_dir / "meta.json", cell_dir / "SUSPENDED.json")

    def item(self, source_id: str) -> Path:
        return self.items / f"{source_id}.json"

    def incomplete_item(self, source_id: str) -> Path:
        return self.incomplete / f"{source_id}.json"

    def incomplete_items(self) -> list[str]:
        if not self.incomplete.exists():
            return []
        return sorted(p.stem for p in self.incomplete.glob("*.json"))


def item_file_valid(path: Path, cell: CellSpec, source_id: str) -> bool:
    """A completed item file: strict JSON describing exactly ``(cell, source_id)``."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"{path} is not valid JSON ({exc}); refusing to guess (remove it to regenerate)") from exc
    if not isinstance(data, dict) or data.get("source_id") != source_id:
        raise ProtocolError(f"{path} does not describe item {source_id}")
    cell_block = data.get("cell")
    if not isinstance(cell_block, dict) or cell_block.get("cell_id") != cell.cell_id:
        raise ProtocolError(f"{path} belongs to cell {cell_block.get('cell_id') if isinstance(cell_block, dict) else None!r}")
    if cell.kind is CellKind.GENERATE and data.get("status") not in ("complete", "context_failure"):
        raise ProtocolError(f"{path} has status {data.get('status')!r}")
    return True


def unmet_dependencies(cell: CellSpec, run_root: str | os.PathLike) -> list[str]:
    """Cell dependencies without ``meta.json`` (ordering advice for generate cells: aliases are
    content-addressed, so running early only costs GPU time — architecture §2.3).  ``seal:``
    entries are not cells and are resolved by the seal registers instead."""
    return [dep for dep in cell.depends_on if not dep.startswith(SEAL_DEP_PREFIX) and not (Path(run_root) / "cells" / dep / "meta.json").exists()]


@dataclass
class _Shared:
    """Mutable state shared by the item workers of one cell."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    suspended: BaseException | None = None
    n_aliased: int = 0
    n_generated: int = 0
    n_alias_races: int = 0
    flops_total: int = 0
    stopped: bool = False


class CellRunner:
    """One ``run_cell`` invocation (kept as an object so the item workers share state)."""

    def __init__(
        self,
        cell: CellSpec,
        run_root: str | os.PathLike,
        server_run_root: str | os.PathLike,
        shard: int,
        stop_event: threading.Event | None,
        *,
        cfg: StudyConfig | None = None,
        max_inflight: int | None = None,
        tokenizer: Any | None = None,
        oracle_factory: Callable[[Checkpoint], FlopOracle] | None = None,
        client_factory: Callable[[EndpointPool, Any, Any], Any] | None = None,
        bcb_evaluator: bcb_eval.BcbEvaluator | None = None,
        wait_timeout_s: float = DEFAULT_WAIT_S,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
        guided_judges: bool = False,
    ) -> None:
        if not isinstance(cell, CellSpec):
            raise TypeError("cell must be a CellSpec")
        self.cell = cell
        self.run_root = Path(run_root)
        self.server_run_root = Path(server_run_root)
        self.shard = int(shard)
        self.stop_event = stop_event or threading.Event()
        self.cfg = cfg or load_config()
        self.max_inflight = int(max_inflight) if max_inflight else int(cell.max_inflight)
        if self.max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        self._tokenizer = tokenizer
        self._oracle_factory = oracle_factory or FlopOracle.from_checkpoint
        self._client_factory = client_factory
        self._bcb_evaluator = bcb_evaluator
        self.wait_timeout_s = float(wait_timeout_s)
        self.run_id = run_id or self.run_root.name
        self.clock = clock
        self.guided_judges = bool(guided_judges)
        self.paths = CellPaths.of(self.run_root, cell.cell_id)
        self.shared = _Shared()
        self.code_version = io.git_commit()
        self.started_at = clock()
        self.frozen: FrozenStudy | None = None
        self.store = RequestStore(self.run_root / "requests")
        self.tasks: dict[str, PublicTask] = {}
        self.pool: EndpointPool | None = None
        self.client: Any = None
        self.oracle: FlopOracle | None = None
        self.tokenizer: Any = None

    # ---- events ----------------------------------------------------------------------------
    def event(self, kind: str, **payload: Any) -> None:
        record = {"t": self.clock(), "event": kind, "cell_id": self.cell.cell_id, **payload}
        if kind == "alias_race":
            with self.shared.lock:
                self.shared.n_alias_races += 1
        io.append_jsonl(self.paths.events, record)

    def _on_store_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.event(kind, **{k: v for k, v in payload.items() if k != "cell_id"})

    # ---- setup -----------------------------------------------------------------------------
    def _checkpoint(self) -> Checkpoint:
        if self.cell.kind is CellKind.GENERATE:
            return self.cfg.checkpoint(self.cell.checkpoint)
        return self.cfg.judge

    def _freeze_gate(self) -> None:
        """Generate cells (except F banks) need ``FROZEN.yaml`` with B0 (architecture §3 step 1)."""
        if self.cell.kind is not CellKind.GENERATE or Method(self.cell.method) is Method.BANK:
            if (self.run_root / FROZEN_FILENAME).exists():
                self.frozen = self.cfg.frozen(self.run_root)
            return
        self.frozen = self.cfg.frozen(self.run_root)
        if self.frozen.b0_flops is None:
            raise ProtocolError(f"{self.frozen.path} carries no B0_flops; profile the study before generate cells (§6.5)")

    def budget_flops(self) -> int:
        if self.frozen is None or self.frozen.b0_flops is None:
            raise ProtocolError("budget requested without a frozen B0")
        if int(self.cell.B) not in self.cfg.budget.budget_multipliers:
            raise ProtocolError(f"cell {self.cell.cell_id}: B={self.cell.B} is not in the frozen grid")
        return int(round(int(self.cell.B) * self.frozen.b0_flops))

    def _verify_oracle(self, oracle: FlopOracle, size: str) -> None:
        if self.frozen is None:
            return
        tables = (self.frozen.manifest.get("extra") or {}).get("profile", {}).get("oracle_tables", {})
        frozen_hash = tables.get(size, {}).get("oracle_hash")
        if frozen_hash is None:
            self.event("oracle_unverified", size=size, oracle_hash=oracle.oracle_hash)
            return
        if frozen_hash != oracle.oracle_hash:
            raise ProtocolError(f"oracle for {size} ({oracle.oracle_hash[:12]}) differs from the frozen one ({frozen_hash[:12]})")

    def _connect(self, ckpt: Checkpoint) -> None:
        self.pool = EndpointPool(self.server_run_root, ckpt.profile, shard=self.shard)
        self.pool.wait(self.wait_timeout_s)  # TimeoutError → EXIT_NO_SERVER (caller)
        self.tokenizer = self._tokenizer if self._tokenizer is not None else load_tokenizer(ckpt)
        oracle = self._oracle_factory(ckpt)
        self._verify_oracle(oracle, ckpt.size)
        self.oracle = oracle
        cost_fn = make_cost_fn({ckpt.size: oracle})
        if self._client_factory is not None:
            self.client = self._client_factory(self.pool, self.tokenizer, cost_fn)
        else:
            self.client = VllmChatClient(self.pool, self.tokenizer, cost_fn=cost_fn, run_id=self.run_id)

    def _todo(self) -> list[str]:
        todo: list[str] = []
        for sid in self.cell.items:
            if sid not in self.tasks:
                raise ProtocolError(f"cell {self.cell.cell_id} names {sid}, absent from the public export")
            if not item_file_valid(self.paths.item(sid), self.cell, sid):
                todo.append(sid)
        return todo

    # ---- main ------------------------------------------------------------------------------
    def run(self) -> int:
        cell = self.cell
        if self.paths.suspended.exists():
            # SUSPENDED.json is terminal (§10.4: a harness defect needs a fix and a numbered
            # rerun, never a silent re-queue).  The operator removes the file after the fix.
            self.event("already_suspended", path=str(self.paths.suspended))
            return EXIT_SUSPENDED
        self.paths.items.mkdir(parents=True, exist_ok=True)
        try:
            self._freeze_gate()
            self.tasks = {t.source_id: t for t in load_public_tasks(self.run_root)}
            todo = self._todo()
            deps = unmet_dependencies(cell, self.run_root)
            if deps:
                self.event("deps_pending", missing=deps)
            if not todo and not self.paths.incomplete_items():
                self._write_meta(n_done=len(cell.items))
                return EXIT_DONE
            if cell.kind in (CellKind.GENERATE, CellKind.JUDGE_BEST, CellKind.JUDGE_HLE):
                try:
                    self._connect(self._checkpoint())
                except TimeoutError as exc:
                    self.event("no_server", error=str(exc))
                    return EXIT_NO_SERVER
            worker = self._item_worker()
            self.event("started", n_todo=len(todo), parallel_items=cell.parallel_items, max_inflight=self.max_inflight)
            self._run_items(todo, worker)
            if self.shared.suspended is not None:
                raise self.shared.suspended
            remaining = self._todo()
        except ProtocolError as exc:
            self._suspend(exc)
            return EXIT_SUSPENDED
        incomplete = self.paths.incomplete_items()
        if self.shared.stopped and remaining:
            self.event("stopped", remaining=len(remaining))
            return EXIT_DONE
        if incomplete or remaining:
            self.event("incomplete", incomplete=incomplete, remaining=remaining)
            return EXIT_INCOMPLETE
        self._write_meta(n_done=len(cell.items))
        return EXIT_DONE

    def _run_items(self, todo: Sequence[str], worker: Callable[[str], None]) -> None:
        queue = list(todo)
        lock = threading.Lock()

        def pull() -> str | None:
            with lock:
                if self.stop_event.is_set():
                    self.shared.stopped = True
                    return None
                if self.shared.suspended is not None or not queue:
                    return None
                return queue.pop(0)

        def loop() -> None:
            while True:
                sid = pull()
                if sid is None:
                    return
                self._run_one_item(sid, worker)

        n_workers = max(1, min(int(self.cell.parallel_items), len(queue)))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="item") as outer:
            futures = [outer.submit(loop) for _ in range(n_workers)]
            for fut in futures:
                fut.result()

    def _run_one_item(self, sid: str, worker: Callable[[str], None]) -> None:
        started = self.clock()
        self.event("item_started", source_id=sid)
        try:
            worker(sid)
        except InfraFailure as exc:
            record = {
                "source_id": sid,
                "cell_id": self.cell.cell_id,
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": getattr(exc, "attempts", None),
                "at": self.clock(),
                "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            io.write_json(self.paths.incomplete_item(sid), record)
            self.event("item_incomplete", source_id=sid, error=record["error"])
            return
        except ProtocolError as exc:
            with self.shared.lock:
                if self.shared.suspended is None:
                    self.shared.suspended = exc
            self.event("item_protocol_error", source_id=sid, error=f"{type(exc).__name__}: {exc}")
            return
        except Exception as exc:  # an unexpected harness defect is a protocol error too
            wrapped = ProtocolError(f"unexpected {type(exc).__name__} in item {sid}: {exc}\n{traceback.format_exc()}")
            with self.shared.lock:
                if self.shared.suspended is None:
                    self.shared.suspended = wrapped
            self.event("item_protocol_error", source_id=sid, error=str(wrapped)[:2000])
            return
        io.remove_file(self.paths.incomplete_item(sid))
        self.event("item_finished", source_id=sid, wall_s=self.clock() - started)

    def _suspend(self, exc: BaseException) -> None:
        io.write_json(
            self.paths.suspended,
            {"cell_id": self.cell.cell_id, "error": f"{type(exc).__name__}: {exc}", "at": self.clock(), "host": socket.gethostname(),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "code_version": self.code_version},
        )
        self.event("suspended", error=f"{type(exc).__name__}: {exc}"[:2000])

    def _write_meta(self, *, n_done: int) -> None:
        if self.paths.incomplete_items():
            raise ProtocolError("meta.json requested while incomplete/ is non-empty")
        io.write_json(
            self.paths.meta,
            {
                "cell_id": self.cell.cell_id,
                "kind": self.cell.kind.value,
                "lane": self.cell.lane,
                "seal": cell_seal(self.cell),
                "n_items": n_done,
                "n_aliased_requests": self.shared.n_aliased,
                "n_generated_requests": self.shared.n_generated,
                "n_alias_races": self.shared.n_alias_races,
                "flops_total": self.shared.flops_total,
                "wall_s": self.clock() - self.started_at,
                "code_version": self.code_version,
                "config_sha256": self.cfg.config_sha256,
                "frozen_sha256": None if self.frozen is None else self.frozen.frozen_sha256,
                "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "run_id": self.run_id,
                "finished_at": self.clock(),
            },
        )

    # ---- per-kind workers -------------------------------------------------------------------
    def _item_worker(self) -> Callable[[str], None]:
        kind = self.cell.kind
        if kind is CellKind.GENERATE:
            return self._generate_item
        if kind is CellKind.JUDGE_BEST:
            self._pools = seals.load_pools(self.run_root, self._seal())
            return self._judge_best_item
        if kind is CellKind.JUDGE_HLE:
            self._pools = seals.load_pools(self.run_root, self._seal())
            self._selections = seals.load_selections(self.run_root, self._seal())
            self._labels = hle_judge.load_labels(self.run_root)
            return self._judge_hle_item
        if kind is CellKind.EVAL_BCB:
            self._pools = seals.load_pools(self.run_root, self._seal())
            self._selections = seals.load_selections(self.run_root, self._seal())
            self._tests = bcb_eval.load_tests(self.run_root)
            if self._bcb_evaluator is None:
                # study-v4: compute nodes have no /usr/bin/apptainer; the site module binary is
                # /orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer (exported as
                # ASYS_APPTAINER_BIN by slurm/common.sh; verified on mit_preemptable node2403).
                self._bcb_evaluator = bcb_eval.BcbEvaluator(
                    work_root=self.run_root / "eval" / "bcb" / "work",
                    apptainer_bin=os.environ.get("ASYS_APPTAINER_BIN", "apptainer"),
                )
            self._prior_verdicts = bcb_eval.load_bcb_eval(self.run_root)
            return self._eval_bcb_item
        raise ProtocolError(f"cell kind {kind.value} has no runner (FORECAST is cut: 04_critic_corrections §5)")

    def _seal(self) -> str:
        seal = cell_seal(self.cell)
        if not seal:
            raise ProtocolError(f"{self.cell.kind.value} cell {self.cell.cell_id} names no seal (depends_on 'seal:<sha>')")
        return seal

    def _worker_block(self) -> dict[str, Any]:
        return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname()}

    # -- GENERATE ------------------------------------------------------------------------------
    def _context(self, task: PublicTask, ledger: EpisodeLedger, executor: ThreadPoolExecutor) -> EpisodeContext:
        assert self.oracle is not None
        return EpisodeContext(
            task=task,
            cell=self.cell,
            cfg=self.cfg,
            checkpoint=self.cfg.checkpoint(self.cell.checkpoint),
            store=self.store,
            client=self.client,
            tokenizer=self.tokenizer,
            oracle=self.oracle,
            ledger=ledger,
            executor=executor,
            on_event=self._on_store_event,
            clock=self.clock,
        )

    def _generate_item(self, sid: str) -> None:
        task = self.tasks[sid]
        method = Method(self.cell.method)
        with ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix=f"req-{sid[-6:]}") as inner:
            if method is Method.BANK:
                result = self._run_bank(task, inner)
            else:
                ledger = EpisodeLedger(
                    self.budget_flops(), self.oracle, 0,
                    solver_call_cap=int(self.cfg.caps.solver_calls), selector_call_cap=int(self.cfg.caps.selector_calls), clock=self.clock,
                )
                ctx = self._context(task, ledger, inner)
                result = policy_for(self.cell).run(ctx)
                self._count_calls(ctx)
        result.code_version = self.code_version
        io.write_json(self.paths.item(sid), result.to_dict())

    def _count_calls(self, ctx: EpisodeContext) -> None:
        with self.shared.lock:
            self.shared.n_generated += ctx.n_generated
            self.shared.n_aliased += ctx.n_aliased
            self.shared.flops_total += sum(c.actual_flops for c in ctx.calls)

    def _run_bank(self, task: PublicTask, inner: ThreadPoolExecutor) -> EpisodeResult:
        """F bank: ten stateless draws of ``render_root(task, framing)`` (§5.5; keys
        ``(0, root, k, stateless_bank)``).  B is not applicable: the ledger is sized to
        exactly the ten full-cap reservations so the broker invariants still hold."""
        assert self.oracle is not None
        ckpt = self.cfg.checkpoint(self.cell.checkpoint)
        rendered = R.render_root(task, self.cell.framing)
        specs: list[RequestSpec] = [
            RequestSpec(
                messages=tuple(dict(m) for m in rendered.messages), decoding=SOLVER_DECODING, checkpoint=ckpt,
                seed_key=SeedKey(task.source_id, task.split, ckpt.model_cell, int(self.cell.episode_rep), 0, PURPOSE_ROOT, k, NS_STATELESS_BANK),
                role="root", study_id=self.cfg.study_id, study_seed_hex=self.cfg.study_seed_hex,
            )
            for k in range(R_BANK)
        ]
        probe = EpisodeLedger(1, self.oracle, 0, solver_call_cap=R_BANK, clock=self.clock)
        ctx = self._context(task, probe, inner)
        amount = sum(self.oracle.reservation(ctx.prompt_tokens(s), int(SOLVER_DECODING.max_tokens)) for s in specs)
        ledger = EpisodeLedger(max(1, amount), self.oracle, 0, solver_call_cap=R_BANK, clock=self.clock)
        ctx = self._context(task, ledger, inner)
        results = ctx.generate_group(specs, "root", owner="bank", actor_slots=[0] * R_BANK, steps=list(range(R_BANK)))
        records = [ctx.candidate_record(r, slot=0, stage="root") for r in results]
        keyed = keyed_pool(records, task.answer_format)
        ledger.stop(StopReason.COMPLETED)
        summary = ledger.close()
        self._count_calls(ctx)
        finished = self.clock()
        return EpisodeResult(
            schema_version=EPISODE_SCHEMA_VERSION,
            cell=self.cell,
            source_id=task.source_id,
            domain=Domain(task.domain),
            split=task.split,
            status="complete",
            episode=None,
            bank=tuple(keyed),
            timing={"started_at": ctx.started_at, "finished_at": finished, "wall_s": finished - ctx.started_at,
                    "calls": [c.call_entry() for c in ctx.calls], "ledger": summary},
            worker=self._worker_block(),
            code_version=self.code_version,
        )

    # -- JUDGE_BEST ----------------------------------------------------------------------------
    def _judge_best_item(self, sid: str) -> None:
        task = self.tasks[sid]
        ckpt = self.cfg.judge
        started = self.clock()
        pools = seals.pools_of_item(self._pools, sid)
        if not pools:
            raise ProtocolError(f"{sid}: no sealed pool under seal {self._seal()[:12]}; JUDGE_BEST scores only sealed pools")
        candidates = seals.load_pool_candidates(self.run_root, pools)
        by_sha: dict[str, list[str]] = {}
        for cid, rec in candidates.items():
            if rec.valid and rec.candidate is not None:
                by_sha.setdefault(rec.candidate_sha256, []).append(cid)
        decoding = dataclasses.replace(JUDGE_DECODING, guided_json=self.guided_judges)

        def score_one(sha: str) -> dict[str, Any]:
            rec = candidates[by_sha[sha][0]]
            rendered = R.render_judge_best(task, rec.candidate)  # type: ignore[arg-type]
            spec = RequestSpec(
                messages=tuple(dict(m) for m in rendered.messages), decoding=decoding, checkpoint=ckpt,
                seed_key=SeedKey(sid, task.split, ckpt.model_cell, 0, 0, PURPOSE_JUDGE_BEST, 0, NS_JUDGE),
                role=ROLE_JUDGE_BEST, study_id=self.cfg.study_id, study_seed_hex=self.cfg.study_seed_hex,
            )
            record, aliased = self.store.get_or_generate(spec, self.client, self.cell.cell_id, on_event=self._on_store_event)
            score, fields = parse_judge_best(record.response.get("content"), record.response.get("finish_reason", "stop"))
            flops = record.flops.get("total") if isinstance(record.flops, dict) else None
            return {"request_id": record.request_id, "aliased": bool(aliased), "score": score, "fields": fields,
                    "prompt_tokens": record.prompt_tokens, "completion_tokens": record.response.get("completion_tokens"), "flops": flops}

        with ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix=f"jb-{sid[-6:]}") as inner:
            shas = sorted(by_sha)
            outcomes = list(inner.map(score_one, shas))
        records = dict(zip(shas, outcomes))
        scores: dict[str, float | None] = {}
        for sha, cids in by_sha.items():
            for cid in cids:
                scores[cid] = records[sha]["score"]
        companion = {"calls": len(records), "flops": sum(int(r["flops"] or 0) for r in records.values()),
                     "aliased": sum(1 for r in records.values() if r["aliased"])}
        with self.shared.lock:
            self.shared.flops_total += companion["flops"]
            self.shared.n_generated += companion["calls"] - companion["aliased"]
            self.shared.n_aliased += companion["aliased"]
        io.write_json(
            self.paths.item(sid),
            {
                "schema_version": 1, ITEM_KIND_KEY: CellKind.JUDGE_BEST.value, "cell": self.cell.to_dict(), "source_id": sid,
                "seal": self._seal(), "pool_ids": [p["pool_id"] for p in pools], "started_at": started, "completed_at": self.clock(),
                "scores": scores, "records": records, "invalid_candidate_ids": sorted(c for c, r in candidates.items() if not r.valid),
                "companion_cost": companion, "worker": self._worker_block(), "code_version": self.code_version,
            },
        )

    # -- JUDGE_HLE -----------------------------------------------------------------------------
    def _sealed_records(self, sid: str) -> list[Any]:
        seals.assert_item_sealed(self._selections, sid)  # T18
        wanted = seals.sealed_candidate_ids(self._selections, sid)
        pools = seals.pools_of_item(self._pools, sid)
        candidates = seals.load_pool_candidates(self.run_root, pools)
        missing = wanted - set(candidates)
        if missing:
            raise ProtocolError(f"{sid}: sealed selection references candidates outside the sealed pools: {sorted(missing)[:3]}")
        return [candidates[cid] for cid in sorted(wanted)]

    def _judge_hle_item(self, sid: str) -> None:
        task = self.tasks[sid]
        if Domain(task.domain) is not Domain.HLE:
            raise ProtocolError(f"JUDGE_HLE cell {self.cell.cell_id} lists non-HLE item {sid}")
        started = self.clock()
        records = self._sealed_records(sid)
        gold = self._labels.get(sid)
        if gold is None:
            raise ProtocolError(f"{sid}: no protected label")
        existing = hle_judge.load_hle_eval(self.run_root, sid)
        with ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix=f"hj-{sid[-6:]}") as inner:
            record = hle_judge.judge_item(
                task, gold, records, checkpoint=self.cfg.judge, study_id=self.cfg.study_id, study_seed_hex=self.cfg.study_seed_hex,
                store=self.store, client=self.client, cell_id=self.cell.cell_id, seal=self._seal(), executor=inner,
                existing=existing, guided=self.guided_judges, clock=self.clock, on_event=self._on_store_event,
            )
        io.write_json(hle_judge.hle_eval_path(self.run_root, sid), record)
        new_entries = [e for sha, e in record["judgements"].items() if not (existing and sha in existing.get("judgements", {}))]
        with self.shared.lock:
            self.shared.n_aliased += sum(1 for e in new_entries if e.get("aliased"))
            self.shared.n_generated += sum(1 for e in new_entries if e.get("request_id") and not e.get("aliased"))
        io.write_json(
            self.paths.item(sid),
            {"schema_version": 1, ITEM_KIND_KEY: CellKind.JUDGE_HLE.value, "cell": self.cell.to_dict(), "source_id": sid, "seal": self._seal(),
             "started_at": started, "completed_at": self.clock(), "n_candidates": len(records), "n_distinct_answers": record["n_distinct_answers"],
             "n_ambiguous": record["n_ambiguous"], "eval_path": str(hle_judge.hle_eval_path(self.run_root, sid)),
             "worker": self._worker_block(), "code_version": self.code_version},
        )

    # -- EVAL_BCB ------------------------------------------------------------------------------
    def _eval_bcb_item(self, sid: str) -> None:
        task = self.tasks[sid]
        if Domain(task.domain) is not Domain.BCB:
            raise ProtocolError(f"EVAL_BCB cell {self.cell.cell_id} lists non-BCB item {sid}")
        started = self.clock()
        records = self._sealed_records(sid)
        test_src = self._tests.get(sid)
        if test_src is None:
            raise ProtocolError(f"{sid}: no protected test")
        jobs, invalid = bcb_eval.jobs_for_item(sid, test_src, records)
        reused: dict[str, dict[str, Any]] = {}
        fresh: list[dict[str, Any]] = []
        for job in jobs:
            prior = self._prior_verdicts.get(job["cid"])
            if prior is not None and prior.get("program_sha256") == identity.sha256_hex(job["code"]):
                reused[job["cid"]] = prior
            else:
                fresh.append(job)
        assert self._bcb_evaluator is not None
        verdicts = self._bcb_evaluator.evaluate_many(fresh) if fresh else {}
        rows: list[dict[str, Any]] = []
        for job in jobs:
            cid = job["cid"]
            base = verdicts.get(cid) if cid in verdicts else reused[cid]
            rows.append({
                "source_id": sid, "candidate_id": cid, "cell_id": self.cell.cell_id, "seal": self._seal(), "started_at": started,
                "program_sha256": identity.sha256_hex(job["code"]), "status": base["status"], "returncode": base.get("returncode"),
                "stdout_tail": base.get("stdout_tail", ""), "stderr_tail": base.get("stderr_tail", ""), "elapsed": base.get("elapsed"),
                "reused_from": base.get("cell_id") if cid in reused else base.get("deduplicated_from"),
            })
        for row in rows:
            io.append_jsonl(bcb_eval.bcb_eval_path(self.run_root, self.cell.cell_id), row)
        io.write_json(
            self.paths.item(sid),
            {"schema_version": 1, ITEM_KIND_KEY: CellKind.EVAL_BCB.value, "cell": self.cell.to_dict(), "source_id": sid, "seal": self._seal(),
             "started_at": started, "completed_at": self.clock(), "n_candidates": len(records), "n_evaluated": len(rows),
             "n_reused": len(reused), "invalid_candidate_ids": invalid,
             "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in bcb_eval.STATUSES},
             "eval_path": str(bcb_eval.bcb_eval_path(self.run_root, self.cell.cell_id)), "worker": self._worker_block(), "code_version": self.code_version},
        )


def run_cell(
    cell: CellSpec,
    run_root: str | os.PathLike,
    server_run_root: str | os.PathLike,
    shard: int,
    stop_event: threading.Event | None,
    **kwargs: Any,
) -> int:
    """Run one cell to completion (or to a stop) and return its exit code (module docstring)."""
    return CellRunner(cell, run_root, server_run_root, shard, stop_event, **kwargs).run()


__all__ = [
    "CellPaths",
    "CellRunner",
    "DEFAULT_WAIT_S",
    "ITEM_KIND_KEY",
    "R_BANK",
    "ROLE_JUDGE_BEST",
    "item_file_valid",
    "run_cell",
    "unmet_dependencies",
]
