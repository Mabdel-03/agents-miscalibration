"""One cell, run to completion or to a stop: the resumable worker every lane driver invokes.

Plan §3.3/§3.4 (kinds, hard dependencies, ``EXIT_DEPS_UNMET``), spec 15.3 (retries and
idempotency), 17.2, contract §2.5.  The study's ``CellRunner`` supplies the mechanisms that
are genuinely study-agnostic -- ``CellPaths``, the outer/inner worker pools, the
``incomplete/`` infrastructure records, ``SUSPENDED.json``, ``meta.json`` written last -- and
this module supplies everything that carries a vocabulary:

* **the task loader** (``CellRunner.run`` calls ``study``'s public-task loader, which admits
  only ``dev``/``main``/``reserve`` splits);
* **the hard dependency gate** (``study``'s is *advice*: it logs ``deps_pending`` and
  proceeds).  Here :func:`NextdynCellRunner.run` resolves every ``cell:`` / ``bundle:`` /
  ``register:`` dependency **before** ``_todo()``, before any endpoint is opened and before
  any request is issued; an unmet one writes ``DEPS_UNMET.json`` and exits
  :data:`EXIT_DEPS_UNMET` with no ``meta.json`` (plan §3.4, D12);
* **the cell-directory ownership refusal** (``ard/runner.py:415``): a directory holding an
  item this cell does not declare means two manifests were given one cell id, and the cell
  refuses to run rather than merge two cohorts (defect D-04);
* **the one site that reads the kind knobs**: :func:`worker_concurrency` is where
  ``KindSpec`` and ``compute.yaml`` meet the thread pools, so a declared knob that nothing
  reads is impossible (defect D-03).

Private rows never land under ``cells/``: the runner writes what a worker returns through
``ItemFile.write(path, protected_path=...)``, so a GRADE cell's grades go to
``protected/cells/<cell_id>/items/`` (0600 in 0700) and the public item file that
``deps.dependency_status`` opens carries none of them (R7).
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.experiment import io
from agents_scaling.nextdyn import bundles as BU
from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import config as C, contract as K
from agents_scaling.nextdyn import deps, freeze as FREEZE, ids, kinds, plan as PLAN, tables
from agents_scaling.nextdyn.cells import CellSpec
from agents_scaling.nextdyn.errors import DependencyUnmet, ManifestError, NextdynError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.study import identity
from agents_scaling.study.packets import make_counter
from agents_scaling.study.runner import CellRunner
from agents_scaling.study.types import (
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    CellSpec as StudyCellSpec,
    CellKind as StudyCellKind,
    Framing,
    InfraFailure,
    Method,
    ProtocolError,
)

#: A dependency that is not here yet is not a failure: no ``meta.json``, no request, and the
#: lane driver re-dispatches the cell on a later pass (plan §3.4).
EXIT_DEPS_UNMET: Final[int] = 5
DEPS_UNMET_FILE: Final[str] = PLAN.DEPS_UNMET_FILE
MODULE_PREFIX: Final[str] = "agents_scaling.nextdyn."
#: Server pools are sub-run-roots (plan §3.2).
POOLS_DIR: Final[str] = "pools"


@dataclass(frozen=True)
class Concurrency:
    """How wide one cell runs: items in parallel, and requests in flight per item."""

    parallel_items: int
    max_inflight: int


def worker_concurrency(kind: CellKind | str, *, compute: Any, contract: K.StudyContract,
                       model: K.ModelCell | None = None) -> Concurrency:
    """The ONE site that turns the kind table and ``compute.yaml`` into thread-pool widths.

    ``test_every_kindspec_field_is_read_outside_kinds_py`` holds ``KindSpec`` to having a
    reader; this is that reader for the shape knobs, and ``episode``'s ``max_inflight:
    n_agents`` resolves here from the team shape rather than from a constant (defect D-03:
    ARD36 declared ``max_inflight`` and never read it, so every round went out serially).
    """
    shape = kinds.cell_shape(kind, compute=compute, model=model,
                             thinking_cap=contract.tokens.thinking_cap, team=contract.team)
    return Concurrency(parallel_items=int(shape.parallel_items),
                       max_inflight=int(shape.max_inflight))


def resolve_worker(kind: CellKind | str) -> Callable[[Any, str], tables.ItemFile]:
    """``KindSpec.worker`` (``"module:callable"``) resolved lazily.

    Lazily on purpose: the worker modules of the writer side (``grading``, ``construct``)
    must not enter the generation lane's import closure (R7), and a work package that has
    not landed must fail on its own cell rather than at import time.
    """
    spec = kinds.spec_for(kind)
    module_name, _, attr = spec.worker.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ManifestError(f"{spec.kind.value}: worker module {module_name} is not importable "
                            f"({exc}); its work package has not landed") from exc
    worker = getattr(module, attr, None)
    if worker is None or not callable(worker):
        raise ManifestError(f"{spec.kind.value}: {module_name} has no callable {attr!r}")
    return worker


def as_study_cell(cell: CellSpec, *, concurrency: Concurrency) -> StudyCellSpec:
    """Project onto the frozen ``study`` shape for the reused runner machinery.

    Only what ``study/runner.py`` reads on the reused path carries meaning: ``cell_id``
    (paths and the flock), ``items`` (resume), ``parallel_items`` / ``max_inflight`` (pools),
    ``lane`` and ``split``.  The v4 factorial fields take the documented placeholders exactly
    as ``ard/types.as_study_cell`` does; no nextdyn fact is expressed through them.
    """
    return StudyCellSpec(
        cell_id=cell.cell_id, kind=StudyCellKind.GENERATE, module="nextdyn", method=Method.BANK,
        checkpoint=cell.model or "", N=0, B=0, framing=Framing.NATIVE,
        episode_rep=int(cell.replicate), split=cell.split_role or "", items=tuple(cell.items),
        depends_on=tuple(cell.depends_on), max_inflight=int(concurrency.max_inflight),
        parallel_items=int(concurrency.parallel_items), lane=cell.lane.value, degree=None)


@dataclass(frozen=True)
class _RootSlotWithRecord(BU.RootSlot):
    """A bundle slot that also carries the bank draw's record (see ``_with_records``).

    ``bundles.RootSlot`` deliberately does not declare it -- the exported bundle is bytes on
    disk and a ``DecisionRecord`` is a runtime object -- and ``episode`` reads it with
    ``getattr(slot, "record", None)``, which is the seam this fills.
    """

    record: Any = None


class CountingTokenizer:
    """A tokenizer that is also a ``count(text) -> int`` callable.

    The workers read ``runner.tokenizer_for(model)`` for two different things and both
    readings are right: ``episode`` and ``receiver`` want the *tokenizer* (they build their own
    counter with ``study.packets.make_counter`` and the client needs
    ``apply_chat_template`` / ``decode``), while ``banks`` passes the same object straight to
    ``schema.parse_public_output`` as the token *counter* for the spec-6.1 over-limit flags.
    Rather than give the runner two accessors that differ by one call -- the shape that makes a
    caller pick the wrong one -- this wrapper satisfies both: attribute access forwards to the
    tokenizer, and calling it counts.  ``make_counter`` prefers ``.encode``, so a consumer that
    wraps this object gets the tokenizer's own counting, not a second indirection.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._count = make_counter(tokenizer)

    @classmethod
    def of(cls, tokenizer: Any) -> "CountingTokenizer":
        return tokenizer if isinstance(tokenizer, cls) else cls(tokenizer)

    def __call__(self, text: str) -> int:
        return int(self._count(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __repr__(self) -> str:                           # pragma: no cover - diagnostics
        return f"CountingTokenizer({self._tokenizer!r})"


@dataclass(frozen=True)
class CellOutcome:
    """The five spec-17.5 counters of one cell (``cli.Report`` carries the same shape)."""

    completed: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    reused: int = 0
    exit_code: int = EXIT_DONE
    detail: str = ""


class NextdynCellRunner(CellRunner):
    """One nextdyn cell (module docstring)."""

    def __init__(self, cell: CellSpec, run_root: str | os.PathLike,
                 server_run_root: str | os.PathLike, shard: int,
                 stop_event: threading.Event | None, *, cfg: Any,
                 compute: Any | None = None, dispatch_context: Any | None = None,
                 items_loader: Callable[[Path], Mapping[str, Any]] | None = None,
                 capture_engine: Callable[[], Any] | None = None,
                 **kwargs: Any) -> None:
        if not isinstance(cell, CellSpec):
            raise TypeError("cell must be a nextdyn cells.CellSpec")
        self.nd_cell = cell
        self.nd_cfg = cfg
        self.contract: K.StudyContract = cfg.contract
        self.compute = compute if compute is not None else C.load_compute(C.DEFAULT_COMPUTE_CONFIG)
        self.kind_spec = kinds.spec_for(cell.kind)
        self.model: K.ModelCell | None = None if cell.model is None else self.contract.model(cell.model)
        self.concurrency = worker_concurrency(cell.kind, compute=self.compute,
                                              contract=self.contract, model=self.model)
        self._ctx = dispatch_context
        self._items_loader = items_loader
        self._capture_engine = capture_engine
        self._items: Mapping[str, Any] | None = None
        self._request_rows: dict[str, list[Any]] = {}
        self._rows_lock = threading.Lock()
        self._current_item = threading.local()
        self.n_aliased = 0
        self.n_generated = 0
        self.study_cell = as_study_cell(cell, concurrency=self.concurrency)
        super().__init__(self.study_cell, run_root, server_run_root, shard, stop_event, cfg=cfg,
                         max_inflight=self.concurrency.max_inflight, **kwargs)
        # ``worker(runner, source_item_id)`` reads ``runner.cell`` and means the *nextdyn*
        # cell (contract §2.5): its kind, its options and its condition.  The study projection
        # stays reachable as ``study_cell`` for the frozen machinery that reads it.
        self.cell = cell

    # ---- what the workers read (contract §2.5 "worker(runner, source_item_id)") ----------
    @property
    def cell_id(self) -> str:
        return self.nd_cell.cell_id

    @property
    def study_seed(self) -> bytes:
        return self.contract.study_seed

    @property
    def top_logprobs(self) -> int:
        """One logprob per final token: enough for ``answer_logprob``, no extra wire cost."""
        return 1

    @property
    def tokens(self) -> K.TokenContract:
        return self.contract.tokens

    @property
    def split_role(self) -> str:
        return self.nd_cell.split_role

    def tokenizer_for(self, model: K.ModelCell | str) -> Any:
        """The model's tokenizer, which is also its token counter (see :class:`CountingTokenizer`)."""
        name = model if isinstance(model, str) else model.name
        return CountingTokenizer.of(self.ctx.tokenizer_for(name))

    @property
    def ctx(self) -> Any:
        """The dispatch context; built on first use so a cpu cell never touches serving."""
        if self._ctx is None:
            self._ctx = self._build_context()
        return self._ctx

    def items(self) -> Mapping[str, Any]:
        if self._items is None:
            loader = self._items_loader or self._default_items_loader
            self._items = dict(loader(self.run_root))
        return self._items

    def item(self, source_item_id: str) -> Any:
        try:
            return self.items()[source_item_id]
        except KeyError as exc:
            raise ProtocolError(f"cell {self.cell_id} names {source_item_id}, absent from the "
                                "public item store") from exc

    def _default_items_loader(self, run_root: Path) -> Mapping[str, Any]:
        from agents_scaling.nextdyn import items as I

        return I.load_items(run_root)

    # ---- generation ----------------------------------------------------------------------
    def _build_context(self) -> Any:
        from agents_scaling.nextdyn import inference as INF
        from agents_scaling.study.inference.store import RequestStore

        pools: dict[str, Any] = {}
        allowed = self._allowed_partitions()

        def pool_for(profile: str) -> Any:
            pool = pools.get(profile)
            if pool is None:
                pool = INF.NextdynEndpointPool(self.server_run_root, profile, shard=self.shard,
                                               allowed_partitions=allowed,
                                               job_info=self._job_info)
                pools[profile] = pool
            return pool

        return INF.DispatchContext(
            store=RequestStore(self.run_root / "requests"), pool_for=pool_for,
            cost_fn=INF.flop_cost_fn(self.contract.models),
            tokenizer_for=self._tokenizer_factory(), contract=self.contract,
            cell_id=self.cell_id, run_id=self.run_id)

    def _allowed_partitions(self) -> frozenset[str] | None:
        """Filter preemptable endpoints away when the *pool* cannot already guarantee it.

        Two mechanisms, and only one of them is needed at a time.  The pool is the primary
        one: ``kinds.pool_for`` refuses a deadline-bearing kind on a pool whose
        ``preemptable_allowed`` is true, and each pool is a separate sub-run-root the keepalive
        fills from that pool's own spec -- so a deadline cell reading
        ``$RUN/pools/deadline/servers`` cannot see a preemptable replica at all.  The partition
        filter is the second belt, and it matters exactly where the pool cannot speak: a pool
        that *does* admit preemptable replicas serving a deadline-bearing kind.

        It is not applied unconditionally because it is fail-closed by design (an endpoint
        whose Slurm job cannot be resolved is excluded), and a deadline lane that excluded
        every endpoint because ``squeue`` was slow would be a worse failure than the one it
        prevents -- while the pool separation still holds.
        """
        if not self.kind_spec.deadline_bearing or self.nd_cell.pool is None:
            return None
        if not getattr(self.compute.pools, self.nd_cell.pool.value).preemptable_allowed:
            return None
        return frozenset(name for name, block in self.compute.partitions.items()
                         if not block.preemptable)

    def _job_info(self, slurm_job_id: str) -> Any:
        """Partition and GPU type of one serve job (``squeue -j``), cached by the pool."""
        from agents_scaling.nextdyn import inference as INF

        out = subprocess.run(["squeue", "-h", "-j", str(slurm_job_id), "-o", "%P %b"],
                             capture_output=True, text=True).stdout.split()
        partition = out[0] if out else None
        gpu = None
        for token in out:
            if token.startswith("gpu:"):
                parts = token.split(":")
                gpu = parts[1] if len(parts) > 1 and not parts[1].isdigit() else None
        return INF.JobInfo(partition=partition, gpu_type=gpu)

    def _tokenizer_factory(self) -> Callable[[str], Any]:
        from agents_scaling.nextdyn import inference as INF
        from agents_scaling.study.inference.tokens import load_tokenizer

        cache: dict[str, Any] = {}

        def _for(name: str) -> Any:
            tokenizer = cache.get(name)
            if tokenizer is None:
                tokenizer = load_tokenizer(INF.checkpoint_for(self.contract.model(name)))
                cache[name] = tokenizer
            return tokenizer

        return _for

    def generate(self, request: Any) -> Any:
        """``Generate``: one decision, plus the ``requests`` rows it owes the tables.

        The request rows are built here rather than in the worker because
        ``telemetry.request_rows`` is the single site that turns physical ids into rows (R6)
        and only the runner holds the producer stamp and the specs.

        **An aliased record writes its rows too.**  It is tempting to skip them -- the table
        keys on ``request_hash``, so a cross-cell alias would only produce a duplicate -- but
        the alias that matters on this fleet is not cross-cell: it is the *same* cell
        re-running its own item after a preemption or a walltime SIGUSR1, and the item file it
        is about to rewrite is the only home its ``requests`` rows ever had.  Returning early
        there deletes the per-request archive of everything the cell had already done
        (measured on the keystone dry run: 769 store records, 69 published rows), and with it
        every ``ReplicaStamp``, latency, token count and attempt row that ``resources.parquet``
        and ``conservation_report.json`` are built from (plan §3.7) -- the ARD36 "resources
        telemetry NULL" defect, reintroduced by a resume.  So the rows are rebuilt from the
        committed record (``inference.request_specs_for`` and ``inference.replica_of`` are
        written to do exactly that) and the genuine cross-cell duplicate is collapsed at
        materialize by ``collapse_by_key(requests, "request_hash")``.  The *reuse* is still
        recorded on the decision -- ``aliased`` here and ``first_decision_id`` at materialize
        -- which is what plan §1.3 asks for.
        """
        from agents_scaling.nextdyn import inference as INF
        from agents_scaling.nextdyn import telemetry as TM

        record = INF.generate_decision(request, ctx=self.ctx, on_event=self._on_store_event)
        with self.shared.lock:
            if record.aliased:
                self.shared.n_aliased += 1
                self.n_aliased += 1
            else:
                self.shared.n_generated += 1
                self.n_generated += 1
        specs = INF.request_specs_for(request, self.ctx, record)
        stamp = INF.ProducerStamp(
            cell_id=self.cell_id, run_id=self.run_id,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
            endpoint_id=record.endpoint_id, gpu_type=record.gpu_type, partition=record.partition)
        rows = TM.request_rows(record, request.key, cell_id=self.cell_id, request_specs=specs,
                               producer=stamp)
        item_id = getattr(self._current_item, "source_item_id", request.key.source_item_id)
        with self._rows_lock:
            self._request_rows.setdefault(item_id, []).extend(rows)
        return record

    # ---- study overrides -------------------------------------------------------------------
    def _checkpoint(self) -> Any:                        # pragma: no cover - never reached
        from agents_scaling.nextdyn import inference as INF

        if self.model is None:
            raise ProtocolError(f"{self.cell_id} is not a generation cell")
        return INF.checkpoint_for(self.model)

    def _freeze_gate(self) -> None:
        """Every cell calls ``freeze.gate`` before its first request (contract §2.5)."""
        self.frozen = FREEZE.gate(self.run_root, self.nd_cfg)

    def _connect(self, ckpt: Any) -> None:
        """Wait for an endpoint of this cell's model; the client itself is the context's."""
        del ckpt
        if self.model is None:
            return
        pool = self.ctx.pool(self.model)
        pool.wait(self.wait_timeout_s)                   # TimeoutError -> EXIT_NO_SERVER
        self.pool = pool

    def _todo(self) -> list[str]:
        """Items without a valid, self-claiming item file (the nextdyn item-file shape)."""
        todo: list[str] = []
        for item_id in self.nd_cell.items:
            path = tables.item_path(self.run_root, self.cell_id, item_id)
            if not self._item_file_valid(path, item_id):
                todo.append(item_id)
        return todo

    def _item_file_valid(self, path: Path, item_id: str) -> bool:
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProtocolError(f"{path} is not valid JSON ({exc}); remove it to regenerate") from exc
        if data.get("cell_id") != self.cell_id or data.get("source_item_id") != item_id:
            raise ProtocolError(f"{path} does not describe {self.cell_id}/{item_id}")
        # Every *committed* status is terminal for the resume: ``complete`` and
        # ``context_failure`` are outcomes, and ``failed`` is the typed outcome a worker writes
        # when the item cannot be produced at all (an E0 source that is not on this filesystem,
        # spec 15.3's BLOCKED_MISSING_SOURCE).  Re-running it would loop forever; an
        # *infrastructure* failure is not an item file at all, it is a record under
        # ``incomplete/``, and that is what the bounded retries re-drive.
        return data.get("status") in {status.value for status in tables.ItemStatus}

    def _assert_directory_is_this_cells(self) -> None:
        """A cell directory holds exactly this cell's items, or the cell refuses to run (D-04)."""
        declared = set(self.nd_cell.items)
        present = {p.stem for p in self.paths.items.glob("*.json")} if self.paths.items.exists() else set()
        foreign = sorted(present - declared)
        if foreign:
            raise ProtocolError(
                f"cell directory {self.cell_id!r} holds {len(foreign)} item(s) this cell does not "
                f"declare (e.g. {foreign[:3]}): two manifests have been given the same cell id. "
                "Rebuild the manifests so the ids are distinct and re-dispatch; the request store "
                "makes the re-run free.")

    def _deps_unmet_path(self) -> Path:
        return self.paths.cell_dir / DEPS_UNMET_FILE

    def _record_deps_unmet(self, unmet: Sequence[Any]) -> None:
        io.write_json(self._deps_unmet_path(), {
            "cell_id": self.cell_id, "kind": self.nd_cell.kind.value, "at": self.clock(),
            "exit_code": EXIT_DEPS_UNMET,
            "unmet": [{"dep": s.dep, "kind": s.kind, "path": s.path, "status": s.status,
                       "detail": s.detail} for s in unmet],
        })
        self.event("deps_unmet", unmet=[s.dep for s in unmet], statuses=[s.status for s in unmet])

    def unmet_dependencies(self) -> list[Any]:
        return deps.unmet(self.run_root, self.nd_cell.depends_on)

    def _write_meta(self, *, n_done: int) -> None:
        """``meta.json``, written last, in the nextdyn vocabulary (no study seal, no B0)."""
        if self.paths.incomplete_items():
            raise ProtocolError("meta.json requested while incomplete/ is non-empty")
        io.write_json(self.paths.meta, {
            "cell_id": self.cell_id, "kind": self.nd_cell.kind.value,
            "lane": self.nd_cell.lane.value,
            "pool": None if self.nd_cell.pool is None else self.nd_cell.pool.value,
            "block": self.nd_cell.block, "n_items": n_done,
            "n_aliased_requests": self.shared.n_aliased,
            "n_generated_requests": self.shared.n_generated,
            "wall_s": self.clock() - self.started_at, "code_version": self.code_version,
            "config_sha256": getattr(self.nd_cfg, "raw_sha256", ""),
            "frozen_sha256": None if self.frozen is None else self.frozen.sha256,
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "run_id": self.run_id, "finished_at": self.clock(),
        })

    def _item_worker(self) -> Callable[[str], None]:
        worker = resolve_worker(self.nd_cell.kind)

        def _run(source_item_id: str) -> None:
            self._current_item.source_item_id = source_item_id
            with self._rows_lock:
                self._request_rows.pop(source_item_id, None)
            try:
                item_file = worker(self, source_item_id)
            except (DependencyUnmet, InfraFailure):
                raise
            except NextdynError as exc:
                raise ProtocolError(f"{self.cell_id}/{source_item_id}: {type(exc).__name__}: {exc}") from exc
            if not isinstance(item_file, tables.ItemFile):
                raise ProtocolError(f"worker for {self.nd_cell.kind.value} returned "
                                    f"{type(item_file).__name__}, not an ItemFile")
            self._write_item(item_file, source_item_id)

        return _run

    def _write_item(self, item_file: tables.ItemFile, source_item_id: str) -> None:
        with self._rows_lock:
            rows = tuple(self._request_rows.pop(source_item_id, ()))
        if rows:
            item_file = replace(item_file, rows={**item_file.rows, "requests": rows})
        item_file.write(tables.item_path(self.run_root, self.cell_id, source_item_id),
                        protected_path=tables.protected_item_path(self.run_root, self.cell_id,
                                                                  source_item_id))

    def _run_items(self, todo: Sequence[str], worker: Callable[[str], None]) -> None:
        """The frozen study loop at the nextdyn width.

        Copied rather than called because ``study/runner.py`` reads ``self.cell.parallel_items``
        and ``self.cell`` is the *nextdyn* cell here (the workers' contract); the width comes
        from :func:`worker_concurrency`, which is the one site that reads the kind knobs.
        """
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
                item_id = pull()
                if item_id is None:
                    return
                self._run_one_item(item_id, worker)

        n_workers = max(1, min(int(self.concurrency.parallel_items), len(queue)))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="item") as outer:
            for future in [outer.submit(loop) for _ in range(n_workers)]:
                future.result()

    # ---- main --------------------------------------------------------------------------
    def run(self) -> int:
        """The study ``run()`` sequence with the hard dependency gate in front of it."""
        if self.paths.suspended.exists():
            self.event("already_suspended", path=str(self.paths.suspended))
            return EXIT_SUSPENDED
        self.paths.items.mkdir(parents=True, exist_ok=True)
        try:
            self._assert_directory_is_this_cells()
        except ProtocolError as exc:
            self._suspend(exc)
            return EXIT_SUSPENDED
        unmet = self.unmet_dependencies()
        if unmet:
            self._record_deps_unmet(unmet)
            return EXIT_DEPS_UNMET
        io.remove_file(self._deps_unmet_path())
        try:
            self._freeze_gate()
            todo = self._todo()
            if not todo and not self.paths.incomplete_items():
                self._write_meta(n_done=len(self.nd_cell.items))
                return EXIT_DONE
            if self.kind_spec.needs_endpoint:
                try:
                    self._connect(None)
                except TimeoutError as exc:
                    self.event("no_server", error=str(exc))
                    return EXIT_NO_SERVER
            worker = self._item_worker()
            self.event("started", n_todo=len(todo), cell_kind=self.nd_cell.kind.value,
                       parallel_items=self.concurrency.parallel_items,
                       max_inflight=self.concurrency.max_inflight)
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
        self._write_meta(n_done=len(self.nd_cell.items))
        return EXIT_DONE

    # ================================================================== per-kind inputs
    # Each block below is what one kind's worker asks the runner for (contract §2.5).

    # -- EPISODE ---------------------------------------------------------------------
    def episode_specs(self, source_item_id: str) -> list[Any]:
        from agents_scaling.nextdyn import episode as EP

        cell = self.nd_cell
        item = self.item(source_item_id)
        registry = self._registry()
        label = cell.display_label or ""
        spec = next((s for s in registry.episodes.values() if s.display_label == label), None)
        if spec is None:
            raise ManifestError(f"{self.cell_id}: protocols.yaml declares no condition labelled "
                                f"{label!r}")
        bundle = self.root_bundle(source_item_id)
        revisions = int(cell.option("revisions", self.contract.team.revision_rounds))
        return [EP.EpisodeSpec(
            cell_kind=CellKind.EPISODE, source_item=item, roots=bundle, visibility=spec,
            model=self.model, tokens=self.tokens, replicate=int(cell.replicate),
            revisions=revisions, n=int(bundle.n), split_role=self.split_role,
            study_id=self.contract.study_id, root_round=int(self.nd_cfg.protocol.root_round),
            top_logprobs=self.top_logprobs, family=self._family_spec(item))]

    def classifier_for(self, spec: Any) -> Callable[[str | None], str | None]:
        """The in-loop answer grouping: the raw final answer is its own class.

        Deliberately **not** the family's canonicalizer.  ``FamilySpec.canonicalizer`` is a
        ``"module:callable"`` in ``nextdyn.graders``, and resolving it here would put a grader
        module into the generation lane's import closure -- which is exactly what R7 forbids
        -- while buying only a finer grouping for the loop's own coverage bookkeeping.  WP2
        designed for this: ``run_episode``'s ``classify`` is optional and "when omitted the raw
        final answer is its own class, which keeps the loop free of family knowledge".

        The consequence is stated rather than hidden: inside the loop, ``A`` and ``(A)`` are
        two classes.  Every *reported* grouping -- unanimity, coverage, the three-state
        transitions -- is derived at analysis time from ``grades_private.canonical_class``,
        which the writer-side grader produced, so no result depends on this.
        """
        del spec
        return lambda text: text

    def root_bundle(self, source_item_id: str) -> Any:
        """The label-free roots of this cell's episode: a bundle, or the natural draws."""
        cell = self.nd_cell
        bucket = cell.option("bucket")
        if bucket:
            # the *panel* the item came from, which is where ``construct`` wrote the bundle;
            # the cell's ``family`` is the cohort it is planned for (families.TASK_FAMILY)
            panel = self.item(source_item_id).family
            teams = BU.load_bucket(self.run_root, panel, bucket)
            index = self._composition_index()
            if index >= len(teams):
                raise DependencyUnmet(
                    f"bundle:{panel}/{bucket} holds {len(teams)} team(s); this cell needs "
                    f"member {index} of the declared pair")
            return self._with_records(teams[index], source_item_id)
        return self._with_records(self._natural_bundle(source_item_id), source_item_id)

    # -- root references ---------------------------------------------------------------
    def _with_records(self, bundle: Any, source_item_id: str) -> Any:
        """Attach each slot's **bank draw record**, so round 0 is a reference, not a draw.

        ``run_episode`` writes round 0 as a ``root_ref`` row and ``telemetry.decision_row``
        needs the physical record behind it (spec 8.2: an episode never generates its roots).
        The bundle cannot carry it: it is label-free and a draw index is a construction fact
        (``bundles.LABEL_DENYLIST``).  So the record is re-derived here through the **request
        store**: the draw's ``banks`` row names its request, the request is rebuilt exactly as
        ``banks.bank_worker`` built it, and ``generate`` returns the committed record with
        ``aliased=True`` and no call to any endpoint.  ``test_..._resume`` is the proof: the
        fake server's call count does not move when an episode cell runs.
        """
        index = self._bank_index()
        slots = []
        for slot in bundle.slots:
            row = index.get(slot.generation_uid)
            record = None if row is None else self._bank_record(row, source_item_id)
            slots.append(_RootSlotWithRecord(
                slot=slot.slot, generation_uid=slot.generation_uid,
                decision_id=slot.decision_id, content_hash=slot.content_hash,
                output=slot.output, record=record))
        return replace(bundle, slots=tuple(slots))

    def _bank_index(self) -> Mapping[str, Any]:
        """``{generation_uid: banks row}`` over every complete bank cell of this run root."""
        cached = getattr(self, "_bank_index_cache", None)
        if cached is not None:
            return cached
        out: dict[str, Any] = {}
        base = self.run_root / deps.CELLS_DIR
        for directory in sorted(base.glob("*")) if base.exists() else []:
            if not directory.is_dir() or not any(
                    directory.name.startswith(kind.value + CE.ID_SEP) for kind in ids.BANK_KINDS):
                continue
            for path in sorted((directory / tables.ITEMS_DIR).glob("*.json")):
                for row in tables.ItemFile.read(path).rows.get("banks", ()):
                    out.setdefault(row.generation_uid, row)
        self._bank_index_cache = out
        return out

    def _bank_record(self, row: Any, source_item_id: str) -> Any:
        """The draw's record, from the store: the same request the bank cell committed."""
        from agents_scaling.nextdyn import banks as BK
        from agents_scaling.nextdyn import inference as INF
        from agents_scaling.nextdyn import prompts as PR

        item = self.item(row.source_item_id or source_item_id)
        variant = item.variant(row.item_variant_id)
        model = self._model_by_hash(row.model_hash)
        tokens = self.contract.tokens
        role = BK.role_of(ids.coerce_kind(row.cell_kind))
        key = BK.bank_key(role, row.source_item_id, row.item_variant_id, model, tokens,
                          int(row.draw_index), study_id=self.contract.study_id)
        request = INF.DecisionRequest(
            messages=PR.render_root(variant, item), model=model, adapter=model.adapter,
            tokens=tokens, seed_key=ids.seed_key_for(key, split_role=item.split_role),
            purpose=key.phase, top_logprobs=self.top_logprobs, key=key)
        return self.generate(request)

    def _model_by_hash(self, model_hash: str) -> K.ModelCell:
        for model in self.contract.models.values():
            if ids.model_hash(model) == model_hash:
                return model
        raise ManifestError(f"no model cell hashes to {model_hash!r}; the bank was drawn by a "
                            "model this configuration does not declare")

    def _composition_index(self) -> int:
        """Which member of the bucket's declared pair this cell runs.

        The bundle is label-free, so a cell cannot ask for "the C1 team": it asks for member
        *k* of the pair, and the CONSTRUCT worker writes the pair's members in the frozen
        ``compositions.yaml`` order.  The composition name lives on the manifest's
        ``ConditionRef`` (a planning fact) and never reaches the bundle or a prompt.
        """
        condition = self.nd_cell.condition
        declared = tuple(self.nd_cfg.cohorts.paired_compositions)
        if condition is None or condition.composition not in declared:
            return 0
        return declared.index(condition.composition)

    def _natural_bundle(self, source_item_id: str) -> Any:
        """Build a natural team's bundle from the NATURAL_ROOT cells this cell depends on.

        Built here rather than read from ``plan/root_teams`` because a natural team has no
        construction: it is the item's natural-root draws in draw order, and the protection
        slot is drawn blind from the seed before any of them is graded (spec 10.4).
        """
        from agents_scaling.nextdyn import states as ST

        rows: list[Any] = []
        for dep in self.nd_cell.depends_on:
            parsed = deps.parse_dependency(dep)
            if parsed.kind != "cell":
                continue
            path = tables.item_path(self.run_root, parsed.value, source_item_id)
            if not path.exists():
                continue
            item_file = tables.ItemFile.read(path)
            rows.extend(r for r in item_file.rows.get("decisions", ())
                        if r.cell_kind == CellKind.NATURAL_ROOT.value
                        and r.source_item_id == source_item_id)
        if not rows:
            raise DependencyUnmet(
                f"{self.cell_id}: no natural-root draws for {source_item_id} in "
                f"{[d for d in self.nd_cell.depends_on]}")
        rows.sort(key=lambda r: int(r.agent_slot))
        slots = tuple(BU.RootSlot(slot=index, generation_uid=row.generation_uid,
                                  decision_id=row.decision_id, content_hash=row.content_hash,
                                  output=ST._output_of(row))
                      for index, row in enumerate(rows))
        variant = rows[0].item_variant_id
        focal = int.from_bytes(ids.blind(self.study_seed, K.ORDER_PROTECT, source_item_id,
                                         variant)[:8], "big") % len(slots)
        root_id = ids.root_team_id(CellKind.NATURAL_ROOT, source_item_id, variant,
                                   [s.generation_uid for s in slots], focal,
                                   NATURAL_ROOT_RULE)
        return BU.RootBundle(
            root_team_id=root_id, team_tag=root_id[-12:], family=self.nd_cell.family,
            bucket="", source_item_id=source_item_id, item_variant_id=variant, n=len(slots),
            slots=slots, export_hash="", focal_slot=focal)

    def _family_spec(self, item: Any) -> Any:
        try:
            from agents_scaling.nextdyn import families as F
        except ImportError:                              # pragma: no cover - WP3 pending
            return None
        return F.family(item.family)

    def _registry(self) -> Any:
        from agents_scaling.nextdyn import protocols as P

        if getattr(self, "_registry_cache", None) is None:
            path = C.sibling_config(self.nd_cfg.path, C.DEFAULT_PROTOCOLS_CONFIG.name) \
                or C.DEFAULT_PROTOCOLS_CONFIG
            self._registry_cache = P.load_registry(path)
        return self._registry_cache

    # -- RECEIVER --------------------------------------------------------------------
    def receiver_episodes(self, source_item_id: str) -> list[tuple[Any, Any, Any]]:
        from agents_scaling.nextdyn import checker as CH
        from agents_scaling.nextdyn import receiver as RX

        cell = self.nd_cell
        registry = self._registry()
        label = cell.display_label or ""
        spec = next((s for s in registry.receivers.values() if s.display_label == label), None)
        if spec is None:
            raise ManifestError(f"{self.cell_id}: protocols.yaml declares no receiver arm "
                                f"{label!r}")
        budget = int(cell.option("budget", self.nd_cfg.budgets.receiver_thinking_tokens[0]))
        counter = self.tokenizer_for(self.model)
        counter_hash = self._counter_tokenizer_hash()
        tokens = self.contract.receiver_tokens[f"{budget}/one_call"]
        if cell.option("assay") == CE.ASSAY_E2B:
            problem = self._mod97_problem(source_item_id)
            pair, traces = RX.mod97_pair(problem, study_seed=self.study_seed, counter=counter,
                                         counter_tokenizer_hash=counter_hash, tokens=tokens)
            tool = CH.CheckStepTool(traces, problem, enabled=bool(spec.checker),
                                    max_checks=int(self.nd_cfg.experiments.E2b.max_checks))
            item: Any = problem
            variant = problem.problem_id
            panel_id = problem.problem_id
        else:
            pair = self._packet_pair(source_item_id)
            tool = None
            item = self.item(source_item_id)
            variant = item.item_variant_id
            panel_id = RX.panel_id_for(str(cell.option("packets")))
        condition = RX.ReceiverCondition.from_contract(
            self.contract, assay=str(cell.option("assay")), spec=spec, item=item,
            source_item_id=source_item_id, item_variant_id=variant, panel_id=panel_id,
            model=self.model, budget=budget, replicate=int(cell.replicate),
            split_role=self.split_role, top_logprobs=self.top_logprobs,
            support_panel_size=RX.support_panel_size(self.nd_cfg.experiments.E2a),
            display_label=spec.display_label)
        return [(condition, pair, tool)]

    def _mod97_problem(self, problem_id: str) -> Any:
        from agents_scaling.nextdyn import mod97

        if getattr(self, "_mod97_cache", None) is None:
            spec = mod97.spec_from_config(self.nd_cfg.experiments.E2b)
            panel = mod97.generate_panel(self.study_seed, spec=spec,
                                         per_depth=self._problems_per_depth())
            self._mod97_cache = {p.problem_id: p for p in panel}
        try:
            return self._mod97_cache[problem_id]
        except KeyError as exc:
            raise ManifestError(f"{self.cell_id}: {problem_id} is not in the generated mod97 "
                                "panel; the panel is a pure function of the study seed") from exc

    def _problems_per_depth(self) -> int:
        e2b = self.nd_cfg.experiments.E2b
        return max(int(e2b.source_problems) // max(len(e2b.chain_depths), 1), 1)

    def _packet_pair(self, source_item_id: str) -> Any:
        """Rebuild one E2a fixed pair from the sealed ``register:packets/<sha>`` entry."""
        from agents_scaling.nextdyn import natural_packets as NP
        from agents_scaling.nextdyn import packets as PK
        from agents_scaling.nextdyn import receiver as RX

        sha = str(self.nd_cell.option("packets") or "")
        body = deps.read_register(self.run_root, deps.REGISTER_PACKETS, sha)
        for entry in body[deps.REGISTER_ENTRIES_FIELD]:
            if entry["source_item_id"] != source_item_id:
                continue
            by_role = {p[NP.ROLE_KEY]: p for p in entry["packets"]}
            return RX.FixedPacketPair(
                source_item_id=source_item_id, panel_id=RX.panel_id_for(sha),
                correct=_packet_from_entry(PK, by_role[NP.ROLE_CORRECT]),
                wrong=_packet_from_entry(PK, by_role[NP.ROLE_WRONG]))
        raise DependencyUnmet(f"register:{deps.REGISTER_PACKETS}/{sha} has no entry for "
                              f"{source_item_id}")

    def _counter_tokenizer_hash(self) -> str:
        """Which tokenizer counted a packet's tokens (spec 6.1, cross-model replay).

        ``ids.model_hash`` is over ``[hf_id, revision, tokenizer_revision, adapter]``, so two
        cells that share a tokenizer share this stamp and two that do not, do not -- which is
        exactly the claim ``transmissions.counter_tokenizer_hash`` has to support.
        """
        return ids.model_hash(self.model)

    # -- SELECTION_AUDIT --------------------------------------------------------------
    def audit_pools(self, source_item_id: str) -> list[Any]:
        sha = str(self.nd_cell.option("pool") or "")
        body = deps.read_register(self.run_root, deps.REGISTER_POOL, sha)
        return [SealedPoolCodec.from_entry(entry) for entry in body[deps.REGISTER_ENTRIES_FIELD]
                if entry["source_item_id"] == source_item_id]

    def audit_bodies(self, source_item_id: str) -> Mapping[str, str]:
        sha = str(self.nd_cell.option("pool") or "")
        body = deps.read_register(self.run_root, deps.REGISTER_POOL, sha)
        out: dict[str, str] = {}
        for entry in body[deps.REGISTER_ENTRIES_FIELD]:
            if entry["source_item_id"] != source_item_id:
                continue
            out.update({str(k): str(v) for k, v in entry.get("bodies", {}).items()})
        return out

    def audit_item(self, source_item_id: str) -> Any:
        return self.item(source_item_id)

    @property
    def selection_ctx(self) -> Any:
        from agents_scaling.nextdyn import selectors as SEL

        return SEL.SelectionContext(
            study_seed=self.study_seed,
            selection_seed=ids.blind(self.study_seed, K.VOTE_TIE, self.cell_id))

    # -- GRADE -----------------------------------------------------------------------
    def decisions_for(self, source_item_id: str) -> list[tuple[Any, Any]]:
        """``(decision row, PublicOutput)`` per **generation** of the cells this GRADE grades.

        Two filters, both from D14/D15's "grades key on ``generation_uid``":

        * a **reference** row is not a new sample -- round 0 of every episode is the bank
          draw, and its grade belongs to the draw the bank cell already graded;
        * an **aliased** row shares its generation with one already collected (IN1_OUT1's
          requests are PEER's, byte for byte), so the generation is graded once.

        Without them the same ``(generation_uid, grader_hash)`` is written twice and
        ``materialize`` aborts -- which is the right failure, but the cause is here.
        """
        from agents_scaling.nextdyn import states as ST

        out: list[tuple[Any, Any]] = []
        seen: set[str] = set()
        for cell_id in self._dependency_cells():
            path = tables.item_path(self.run_root, str(cell_id), source_item_id)
            if not path.exists():
                continue
            item_file = tables.ItemFile.read(path)
            for row in item_file.rows.get("decisions", ()):
                if bool(row.is_reference) or row.generation_uid in seen:
                    continue
                output = ST._output_of(row)
                if output is None:
                    continue
                seen.add(row.generation_uid)
                out.append((row, output))
        return out

    # -- CONSTRUCT -------------------------------------------------------------------
    @property
    def compositions(self) -> Any:
        from agents_scaling.nextdyn import construct as CO

        if getattr(self, "_compositions_cache", None) is None:
            path = C.sibling_config(self.nd_cfg.path, "compositions.yaml")
            self._compositions_cache = CO.load_compositions(path)
        return self._compositions_cache

    def donor_input(self, source_item_id: str) -> Any:
        """One CONSTRUCT cell's donor evidence: the draws from the banks, the grades from GRADE.

        Two sources on purpose.  The draws are public rows in the bank cells' item files; the
        grades are ``grades_private`` rows under ``protected/cells/<grade cell>/`` (R7), and
        the CONSTRUCT cell's declared dependency is the GRADE cell precisely because a team
        built from ungraded donors would silently be no team at all.
        """
        from agents_scaling.nextdyn import banks as BK
        from agents_scaling.nextdyn import construct as CO
        from agents_scaling.nextdyn import grading as GR
        from agents_scaling.nextdyn import states as ST

        bank_rows: list[Any] = []
        drafts: dict[str, Any] = {}
        donor_role = BK.BankRole.DONOR.value
        for _cell_id, item_file in self._bank_item_files(source_item_id):
            for row in item_file.rows.get("banks", ()):
                if row.bank_role == donor_role:
                    bank_rows.append(row)
            for row in item_file.rows.get("decisions", ()):
                output = ST._output_of(row)
                if output is not None and row.generation_uid:
                    drafts.setdefault(row.generation_uid, output)

        grade_rows: list[Any] = []
        for cell_id in self._dependency_cells():
            public = tables.item_path(self.run_root, cell_id, source_item_id)
            protected = tables.protected_item_path(self.run_root, cell_id, source_item_id)
            if not public.exists():
                continue
            item_file = tables.ItemFile.read(public, protected_path=protected)
            grade_rows.extend(item_file.private_rows.get("grades_private", ()))

        seen: dict[str, Any] = {}
        for row in bank_rows:
            seen.setdefault(row.bank_draw_id, row)
        return CO.DonorInput(
            item=self.item(source_item_id), bank_rows=tuple(seen.values()),
            grades=GR.tri_state_index(grade_rows), classes=GR.canonical_class_index(grade_rows),
            drafts=drafts, bucket=str(self.nd_cell.option("bucket") or ""),
            pair=str(self.nd_cell.option("pair") or ""))

    def _dependency_cells(self) -> tuple[str, ...]:
        """The cell ids this cell names, through its options and its ``cell:`` dependencies."""
        out = [str(cell_id) for cell_id in self.nd_cell.option("cells", ())]
        for dep in self.nd_cell.depends_on:
            parsed = deps.parse_dependency(dep)
            if parsed.kind == "cell" and parsed.value not in out:
                out.append(parsed.value)
        return tuple(out)

    def _bank_item_files(self, source_item_id: str) -> list[tuple[str, Any]]:
        """This item's committed bank item files, whichever bank cells hold them."""
        out: list[tuple[str, Any]] = []
        base = self.run_root / deps.CELLS_DIR
        for directory in sorted(base.glob("*")) if base.exists() else []:
            if not directory.is_dir() or not any(
                    directory.name.startswith(kind.value + CE.ID_SEP) for kind in ids.BANK_KINDS):
                continue
            path = directory / tables.ITEMS_DIR / f"{source_item_id}.json"
            if path.exists():
                out.append((directory.name, tables.ItemFile.read(path)))
        return out

    # -- CAPTURE ---------------------------------------------------------------------
    def capture_engine(self) -> Any:
        if self._capture_engine is None:
            raise ManifestError(f"{self.cell_id}: no capture engine was injected; the capture "
                                "lane builds one from the pinned snapshot")
        return self._capture_engine()


#: The construction rule a natural team records: none.  The draws in draw order are the
#: team, which is what makes the id reproducible from the bank alone (spec 8.1).
NATURAL_ROOT_RULE: Final[str] = "natural_roots_in_draw_order"


def _counter(tokenizer: Any) -> Callable[[str], int]:
    def _count(text: str) -> int:
        return len(tokenizer.encode(text))

    return _count


def _packet_from_entry(PK: Any, entry: Mapping[str, Any]) -> Any:
    """A :class:`packets.TransmittedPacket` rebuilt from a sealed register entry.

    The register keeps the *bytes* and the counts, not the parsed fields: a cross-model replay
    transmits the stored bytes verbatim and never re-clips (spec 9.1), so the fields a clipper
    would need are deliberately absent and are rebuilt empty.
    """
    return PK.TransmittedPacket(
        packet_id=str(entry["packet_id"]), pseudonym=str(entry["pseudonym"]), fields={},
        truncated={}, token_count=int(entry["token_count"]),
        counter_tokenizer_hash=str(entry["counter_tokenizer_hash"]),
        serialized=str(entry["serialized"]), packet_hash=str(entry["packet_hash"]),
        original_content_hash=entry.get("original_content_hash"),
        content=PK.PacketContent.FULL, filler=False)


class SealedPoolCodec:
    """Sealed-pool (de)serialization, so the planner's writer and this reader are one fact."""

    @staticmethod
    def from_entry(entry: Mapping[str, Any]) -> Any:
        from agents_scaling.nextdyn import selectors as SEL

        return SEL.SealedPool(
            source_item_id=str(entry["source_item_id"]), episode_id=str(entry["episode_id"]),
            cell_kind=entry["cell_kind"], round_=int(entry["round"]), scope=str(entry["scope"]),
            protocol_hash=str(entry["protocol_hash"]),
            candidates=tuple(SEL.Candidate(
                candidate_id=str(c["candidate_id"]), generation_uid=str(c["generation_uid"]),
                content_hash=c.get("content_hash"), canonical_class=c.get("canonical_class"),
                valid=bool(c["valid"]),
                confidence=None if c.get("confidence") in (None, "") else float(c["confidence"]),
                packet_hash=str(c.get("packet_hash", "")), slot=int(c["slot"]))
                for c in entry["candidates"]))

    @staticmethod
    def to_entry(pool: Any, *, bodies: Mapping[str, str] | None = None) -> dict[str, Any]:
        return {
            "source_item_id": pool.source_item_id, "episode_id": pool.episode_id,
            "cell_kind": pool.cell_kind.value, "round": int(pool.round_), "scope": pool.scope,
            "protocol_hash": pool.protocol_hash, "pool_id": pool.pool_id,
            "candidates": [
                {"candidate_id": c.candidate_id, "generation_uid": c.generation_uid,
                 "content_hash": c.content_hash, "canonical_class": c.canonical_class,
                 "valid": bool(c.valid),
                 # a register holds ids, hashes and ints: a confidence is a float, so it is
                 # sealed as its fixed string form (deps.write_register refuses the float)
                 "confidence": None if c.confidence is None else identity.float_str(c.confidence),
                 "packet_hash": c.packet_hash, "slot": int(c.slot)} for c in pool.candidates],
            "bodies": dict(bodies or {}),
        }


def run_cell(cell: CellSpec, run_root: str | os.PathLike, *,
             server_run_root: str | os.PathLike | None = None, pool: Any = None,
             stop_event: threading.Event | None = None, cfg: Any,
             max_inflight: int | None = None, dry_run: bool = False, shard: int = 0,
             **kwargs: Any) -> CellOutcome:
    """Run one cell and report the five counters (spec 17.5).

    ``pool`` names the server pool the cell's endpoints come from; it selects the sub-run-root
    ``$RUN/pools/<pool>/servers`` (plan §3.2) unless the caller passed ``server_run_root``.
    """
    if pool is not None and server_run_root is None:
        server_run_root = pool_run_root(run_root, pool)
    if dry_run:
        return CellOutcome(skipped=1, detail=f"dry run: {cell.cell_id} not dispatched")
    if max_inflight is not None:
        kwargs["max_inflight"] = int(max_inflight)
    runner = NextdynCellRunner(cell, run_root, server_run_root or run_root, shard, stop_event,
                               cfg=cfg, **kwargs)
    code = runner.run()
    reused = runner.n_aliased
    if code == EXIT_DONE:
        return CellOutcome(completed=1, reused=reused, exit_code=code)
    if code == EXIT_DEPS_UNMET:
        return CellOutcome(blocked=1, reused=reused, exit_code=code,
                           detail="dependencies unmet; no request issued")
    if code == EXIT_NO_SERVER:
        return CellOutcome(skipped=1, reused=reused, exit_code=code, detail="no live endpoint")
    return CellOutcome(failed=1, reused=reused, exit_code=code,
                       detail="suspended" if code == EXIT_SUSPENDED else "incomplete")


def pool_run_root(run_root: str | os.PathLike, pool: Any) -> Path:
    """``$RUN/pools/<pool>/servers`` is a sub-run-root: ``registry.servers_dir`` is hard-coded
    to ``<root>/servers``, so a pool is a *root*, never a directory name (plan §3.2)."""
    name = getattr(pool, "value", pool)
    return Path(run_root) / POOLS_DIR / str(name)


__all__ = [
    "CellOutcome", "Concurrency", "DEPS_UNMET_FILE", "EXIT_DEPS_UNMET",
    "CountingTokenizer", "NextdynCellRunner", "POOLS_DIR", "SealedPoolCodec", "as_study_cell", "pool_run_root",
    "resolve_worker", "run_cell",
    "worker_concurrency",
]
