"""Resumable per-cell ARD36 worker: ``study.runner.CellRunner`` + ARD tasks + hard deps (WP4).

Spec §10.4-style resumability (atomic publish, ``incomplete/`` infrastructure records,
``SUSPENDED.json`` for harness defects, ``meta.json`` written last), §12.3 (a GATE cell
refuses to run without the frozen controller), §13 (E5 reads a sealed state register),
§19 (``build_episode`` / ``run_round`` / ``fork_private`` contracts).  Plan §5.3, §10
("``state:`` / ``controller:`` gating ... a GATE cell run with an absent or tampered
``controller/<sha>/MODEL.json`` exits ``EXIT_DEPS_UNMET`` with no ``meta.json`` and zero
requests generated").

Two things the frozen ``study/runner.py`` cannot do for ARD36, and how this module composes
around them without editing it:

1. **Task loading.**  ``CellRunner.run`` calls ``load_public_tasks`` unconditionally, and
   that loader rejects any split outside ``dev``/``main``/``reserve`` (contract §3).
   :class:`ArdCellRunner` overrides ``run()`` — and only ``run()`` — so the task loader is
   injected (:func:`agents_scaling.ard.items.load_ard_tasks` by default).  Every other
   mechanism (``CellPaths``, ``item_file_valid``, ``_run_items``, ``_run_one_item``,
   ``_suspend``, ``_write_meta``, ``_connect``, the FLOP oracle verification) is the frozen
   study one, called on ``super()``.

2. **Hard dependency gating.**  ``study/runner.py:285-287`` logs ``deps_pending`` and
   proceeds, which is right for content-addressed aliasing and wrong for a controller: a
   GATE cell that ran without its frozen controller would burn GPU time on an arm that is
   not the registered arm.  :func:`resolve_hard_dependencies` resolves ``controller:<sha>``
   and ``state:<sha>`` against their registers **before** ``_todo()``, and an unmet or
   tampered dependency returns :data:`agents_scaling.ard.types.EXIT_DEPS_UNMET` with no
   ``meta.json``, no endpoint connection and no request.

The per-kind work itself is resolved through :data:`agents_scaling.ard.kinds.KIND_TABLE`,
so a kind owned by a work package that has not landed yet fails loudly on its own cell.
The episode-shaped kinds run WP2's loop through :func:`wp2_episode_executor`, which adapts
the frozen contract §2 interface (``run_episode`` / ``run_roots``) to the WP4 record types
over WP2's own spec §18 row shapes; :func:`baseline_episode_executor` stands in only while
``ard/episode.py`` does not exist, and stamps every row it writes so no analysis can mistake
it for protocol evidence.  GATE's protection choice comes from WP5 through
:func:`resolve_controller_chooser`; without it a GATE cell refuses to run (spec §12.3).
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents_scaling.ard import contract as K
from agents_scaling.ard import kinds as KINDS
from agents_scaling.ard.client import (
    ContextTooLong,
    DecisionContext,
    DecisionRecord,
    DecisionRequest,
    generate_decision,
)
from agents_scaling.ard.items import ArdTask, load_ard_tasks
from agents_scaling.ard.types import (
    ANALYSIS_ITEM,
    ARD_SCHEMA_VERSION,
    EXIT_DEPS_UNMET,
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    ArdCellKind,
    ArdCellSpec,
    ArdItemResult,
    DecisionRef,
    EpisodeRecord,
    RoundState,
    decision_id_for,
    episode_id_for,
    parse_dependency,
    physical_request_ids_of,
)
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.config import FROZEN_FILENAME, StudyConfig, load_config
from agents_scaling.study.resources.oracle import make_cost_fn
from agents_scaling.study.runner import CellRunner, item_file_valid, unmet_dependencies
from agents_scaling.study.types import Checkpoint, ProtocolError, SeedKey

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The committed ARD36 study manifest (checkpoints, seeds, profiles).
ARD_CONFIG_PATH = REPO_ROOT / "configs" / "ard36.yaml"

CONTROLLER_DIR = "controller"
CONTROLLER_FILE = "MODEL.json"
STATE_DIR = "states"
STATE_FILE = "REGISTER.json"
#: Written next to a cell instead of ``meta.json`` when a hard dependency is unmet, so the
#: lane driver and the top-up tool can tell "waiting for its dependency" from "failed".
DEPS_UNMET_FILE = "DEPS_UNMET.json"


def load_ard_config(path: str | os.PathLike | None = None) -> StudyConfig:
    return load_config(path or ARD_CONFIG_PATH)


# --------------------------------------------------------------------------- dependencies


@dataclass(frozen=True)
class DependencyStatus:
    """One resolved hard dependency."""

    dep: str
    kind: str            # "controller" | "state"
    path: str
    status: str          # "met" | "missing" | "tampered" | "malformed"
    detail: str = ""

    @property
    def met(self) -> bool:
        return self.status == "met"


def register_path(run_root: str | os.PathLike, kind: str, sha: str) -> Path:
    root = Path(run_root)
    if kind == "controller":
        return root / CONTROLLER_DIR / sha / CONTROLLER_FILE
    if kind == "state":
        return root / STATE_DIR / sha / STATE_FILE
    raise ProtocolError(f"no register for dependency kind {kind!r}")


def _content_digest(path: Path) -> tuple[str, str]:
    """``(digest, rule)`` for one register file.

    Rule, in order (documented so WP5/WP6 can freeze against it):

    1. a ``<file>.sha256`` sidecar, when present, must match the raw bytes — a mismatch is
       a tamper, not a miss;
    2. a self-describing JSON object whose ``controller_sha256`` / ``model_sha256`` /
       ``state_sha256`` field must equal the canonical-JSON hash of the object without that
       field (the FrozenG pattern);
    3. otherwise the sha256 of the raw file bytes.
    """
    raw = path.read_bytes()
    raw_digest = identity.sha256_hex(raw)
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        if expected != raw_digest:
            raise ValueError(f"sidecar sha256 {expected} != file sha256 {raw_digest}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError:
        return raw_digest, "raw_bytes"
    if isinstance(payload, dict):
        for field_name in ("controller_sha256", "model_sha256", "state_sha256"):
            declared = payload.get(field_name)
            if isinstance(declared, str):
                body = {k: v for k, v in payload.items() if k != field_name}
                computed = identity.sha256_hex(
                    json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
                if computed != declared.lower():
                    raise ValueError(
                        f"{field_name} {declared[:12]} != canonical-JSON hash {computed[:12]}")
                return declared.lower(), field_name
    return raw_digest, "raw_bytes"


def freeze_register(
    run_root: str | os.PathLike, kind: str, payload: Mapping[str, Any]
) -> str:
    """Write a ``controller:``/``state:`` register and return the sha the dependency names.

    The writer that matches :func:`_content_digest` rule 2: the payload is hashed as
    canonical JSON, the digest is stamped into the object as ``<kind>_sha256``, and the file
    lands at ``<run_root>/controller/<sha>/MODEL.json`` (or ``states/<sha>/REGISTER.json``).
    WP5 freezes the controller through this function so a GATE cell's gate and the freeze
    can never disagree about what "the frozen controller" hashes to.
    """
    field_name = "controller_sha256" if kind == "controller" else "state_sha256"
    body = {key: value for key, value in dict(payload).items() if key != field_name}
    digest = identity.sha256_hex(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    path = register_path(run_root, kind, digest)
    io.write_json(path, {**body, field_name: digest})
    return digest


def resolve_hard_dependencies(
    cell: ArdCellSpec, run_root: str | os.PathLike
) -> list[DependencyStatus]:
    """Resolve every ``controller:`` / ``state:`` dependency against its register."""
    out: list[DependencyStatus] = []
    for dep in cell.hard_dependencies:
        kind, sha = parse_dependency(dep)
        path = register_path(run_root, kind, sha)
        if not path.exists():
            out.append(DependencyStatus(dep, kind, str(path), "missing",
                                        f"{path} does not exist"))
            continue
        try:
            digest, rule = _content_digest(path)
        except ValueError as exc:
            out.append(DependencyStatus(dep, kind, str(path), "tampered", str(exc)))
            continue
        except OSError as exc:  # pragma: no cover - unreadable register
            out.append(DependencyStatus(dep, kind, str(path), "malformed", str(exc)))
            continue
        if not digest.startswith(sha):
            out.append(DependencyStatus(
                dep, kind, str(path), "tampered",
                f"content digest {digest[:16]} ({rule}) does not match the frozen {sha}"))
            continue
        out.append(DependencyStatus(dep, kind, str(path), "met", rule))
    return out


def unmet_hard_dependencies(
    cell: ArdCellSpec, run_root: str | os.PathLike
) -> list[DependencyStatus]:
    return [status for status in resolve_hard_dependencies(cell, run_root) if not status.met]


# --------------------------------------------------------------------------- worker seam


def resolve_worker(kind: ArdCellKind) -> Callable[..., Any]:
    """Import ``KIND_TABLE[kind].worker`` (``module:callable``).

    A kind whose owning work package has not landed raises :class:`ProtocolError`, which the
    reused runner turns into ``SUSPENDED.json`` + ``EXIT_SUSPENDED``: a missing worker is a
    harness defect that needs a fix and a numbered rerun, never a silent re-queue.
    """
    spec = KINDS.spec_for(kind)
    module_name, _, attr = spec.worker.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProtocolError(
            f"cell kind {spec.kind.value} needs {spec.worker} ({spec.owner}), which is not "
            f"importable yet: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ProtocolError(f"{module_name} defines no {attr!r} for kind {spec.kind.value}") from exc


@dataclass
class EpisodeJob:
    """Everything an episode executor needs for one item of one cell.

    ``generate`` and ``decide`` are the only ways to reach vLLM.  ``generate`` takes a fully
    built :class:`DecisionRequest` (WP2's ``episode.Generate``); ``decide`` builds the seed
    key from named arguments for callers that do not construct their own.  Either way the
    seed key is part of the request identity, so nothing can be sent to the server that is
    not inside the cache key (contract §0.3).

    ``records`` accumulates every :class:`DecisionRecord` this item generated, keyed by
    request id, so a caller that only gets report references back (WP2's ``run_roots``) can
    still recover the token counts, latency and FLOPs of each root.
    """

    cell: ArdCellSpec
    task: ArdTask
    checkpoint: Checkpoint
    decide: Callable[..., DecisionRecord]
    generate: Callable[[DecisionRequest], DecisionRecord]
    classify: Callable[[str | None], str | None]
    study_seed: bytes
    model_cell: str
    run_root: Path
    event: Callable[..., None]
    records: dict[str, DecisionRecord] = field(default_factory=dict)
    #: Wall-clock arrival per request id, for the GPU-second attribution of
    #: ``ard/resources.py`` (a record carries only its latency, never its arrival).
    arrivals: dict[str, float] = field(default_factory=dict)
    choose_protected: Callable[[Any, int], int] | None = None
    clock: Callable[[], float] = time.time
    options: Mapping[str, Any] = field(default_factory=dict)


#: An episode executor turns one :class:`EpisodeJob` into the episodes of that item.
EpisodeExecutor = Callable[[EpisodeJob], Sequence[EpisodeRecord]]


def resolve_episode_executor() -> tuple[EpisodeExecutor, bool]:
    """``(executor, is_wp2)`` — WP2's episode loop when it is importable, else the baseline.

    Resolution order:

    1. ``ard.episode.episode_executor(job)``, if WP2 ever publishes this exact seam;
    2. :func:`wp2_episode_executor`, the adapter over WP2's frozen contract interface
       ``run_episode(item, protocol, checkpoint, *, generate, …) -> Episode`` (contract §2);
    3. :func:`baseline_episode_executor`, the WP4 dispatch shape, used only while
       ``ard/episode.py`` does not exist.  Its rows are stamped ``fallback=True`` with a
       ``packet_policy`` of ``fallback:<protocol>``, and ``ard/compact.py`` keeps them out of
       the scientific tables, so a dry run can never be mistaken for protocol evidence.
    """
    try:
        module = importlib.import_module("agents_scaling.ard.episode")
    except ImportError:
        return baseline_episode_executor, False
    executor = getattr(module, "episode_executor", None)
    if callable(executor):
        return executor, True
    if callable(getattr(module, "run_episode", None)):
        return wp2_episode_executor, True
    return baseline_episode_executor, False


def resolve_controller_chooser(
    run_root: Path, cell: ArdCellSpec
) -> Callable[[Any, int], int] | None:
    """GATE's ``choose_protected(state, round_) -> slot`` from WP5's frozen controller.

    WP5 (``ard/controller.py``) is expected to publish ``load_controller(path)`` and the
    contract §2 ``choose_protected(controller, state, *, round_, episode_id)``.  Until it
    does this returns ``None``, and a GATE or CONF_PROTECT cell refuses to run rather than
    inventing a protection choice (spec §12.3).
    """
    if cell.protocol not in (K.Protocol.GATE, K.Protocol.CONF_PROTECT):
        return None
    try:
        module = importlib.import_module("agents_scaling.ard.controller")
        load = getattr(module, "load_controller")
        choose = getattr(module, "choose_protected")
    except (ImportError, AttributeError):
        return None
    shas = [sha for kind, sha in (parse_dependency(dep) for dep in cell.depends_on)
            if kind == "controller"]
    if not shas:
        return None
    controller = load(register_path(run_root, "controller", shas[0]))

    def _choose(state: Any, round_: int) -> int:
        return int(choose(controller, state, round_=round_,
                          episode_id=getattr(state, "episode_id", "")))

    return _choose


# --------------------------------------------------------------------------- runner


class ArdCellRunner(CellRunner):
    """One ARD36 cell, run to completion or to a stop (module docstring)."""

    def __init__(
        self,
        cell: ArdCellSpec,
        run_root: str | os.PathLike,
        server_run_root: str | os.PathLike,
        shard: int,
        stop_event: threading.Event | None,
        *,
        cfg: StudyConfig | None = None,
        task_loader: Callable[[Path], Mapping[str, ArdTask]] | None = None,
        episode_executor: EpisodeExecutor | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(cell, ArdCellSpec):
            raise TypeError("cell must be an ArdCellSpec")
        self.ard_cell = cell
        self.kind_spec = KINDS.spec_for(cell.kind)
        self._task_loader = task_loader or _default_task_loader(cell)
        self._episode_executor = episode_executor
        self._executor_is_wp2 = episode_executor is not None
        self.cost_fn: Callable[..., Mapping[str, Any]] | None = None
        self.decision_context: DecisionContext | None = None
        self.n_decisions = 0
        super().__init__(cell.as_study_cell(), run_root, server_run_root, shard, stop_event,
                         cfg=cfg or load_ard_config(), **kwargs)

    # ---- overrides of the two study behaviours ARD cannot use -------------------------
    def _checkpoint(self) -> Checkpoint:
        """ARD has no judge checkpoint; every kind resolves its own (contract §2)."""
        return self.cfg.checkpoint(self.ard_cell.checkpoint)

    def _freeze_gate(self) -> None:
        """``FROZEN.yaml`` when it exists (for the oracle hash); never a B0 budget gate.

        ARD36 has no ``B`` budget multiplier, so ``study``'s "profile the study first" rule
        does not apply; the oracle verification it enables does, and runs unchanged.
        """
        if (self.run_root / FROZEN_FILENAME).exists():
            self.frozen = self.cfg.frozen(self.run_root)

    def _todo(self) -> list[str]:
        """The study resume rule, minus the public-export lookup for analysis cells.

        ``study/runner._todo`` refuses an item that is not in the public export; a COMPACT or
        E0 cell runs once over the whole run root under the synthetic
        :data:`agents_scaling.ard.types.ANALYSIS_ITEM` id, which is deliberately not a task.
        """
        if self.kind_spec.needs_endpoint or self.ard_cell.kind is ArdCellKind.GRADE:
            return super()._todo()
        assert ANALYSIS_ITEM  # documented above: analysis cells carry the synthetic item id
        return [sid for sid in self.cell.items
                if not item_file_valid(self.paths.item(sid), self.cell, sid)]

    def _deps_unmet_path(self) -> Path:
        return self.paths.cell_dir / DEPS_UNMET_FILE

    def _record_deps_unmet(self, unmet: Sequence[DependencyStatus]) -> None:
        io.write_json(self._deps_unmet_path(), {
            "cell_id": self.cell.cell_id,
            "kind": self.ard_cell.kind.value,
            "protocol": None if self.ard_cell.protocol is None else self.ard_cell.protocol.value,
            "at": self.clock(),
            "exit_code": EXIT_DEPS_UNMET,
            "unmet": [{"dep": s.dep, "kind": s.kind, "path": s.path, "status": s.status,
                       "detail": s.detail} for s in unmet],
        })
        self.event("deps_unmet", unmet=[s.dep for s in unmet],
                   statuses=[s.status for s in unmet])

    def _assert_directory_is_this_cells(self, cell: ArdCellSpec) -> None:
        """A cell directory holds exactly the cell's items, or the cell refuses to run.

        A cell id *is* a directory, and shard numbers restart at ``s000`` in every builder
        run, so two manifests of one module can name the same cell with different item lists
        (defect D-04).  When that happens the second run appends its items beside the first's
        and overwrites ``meta.json``, leaving one directory that claims one manifest's cohort
        while holding two -- and nothing downstream can tell, because every item file is
        individually well formed.

        Refusing here is cheap and reversible: the requests are content-addressed, so a cell
        re-dispatched under a correct id re-derives its items from the store without new
        generation.  Silently merging two cohorts is neither.
        """
        declared = set(cell.items)
        present = {path.stem for path in self.paths.items.glob("*.json")}
        foreign = sorted(present - declared)
        if foreign:
            raise ProtocolError(
                f"cell directory {self.paths.items.parent.name!r} holds "
                f"{len(foreign)} item(s) this cell does not declare "
                f"(e.g. {foreign[:3]}): two manifests have been given the same cell id. "
                "Rebuild the manifests so the ids are distinct and re-dispatch; the store "
                "makes the re-run free."
            )

    def run(self) -> int:
        """The study ``run()`` sequence with an injected task loader and a hard dep gate."""
        cell = self.cell
        if self.paths.suspended.exists():
            self.event("already_suspended", path=str(self.paths.suspended))
            return EXIT_SUSPENDED
        self.paths.items.mkdir(parents=True, exist_ok=True)
        self._assert_directory_is_this_cells(cell)
        # ---- hard gate, BEFORE _todo(), before any endpoint, before any request --------
        unmet = unmet_hard_dependencies(self.ard_cell, self.run_root)
        if unmet:
            self._record_deps_unmet(unmet)
            return EXIT_DEPS_UNMET
        io.remove_file(self._deps_unmet_path())
        try:
            self._freeze_gate()
            self.tasks = dict(self._task_loader(self.run_root))
            todo = self._todo()
            advisory = unmet_dependencies(cell, self.run_root)
            if advisory:
                self.event("deps_pending", missing=advisory)
            if not todo and not self.paths.incomplete_items():
                self._write_meta(n_done=len(cell.items))
                return EXIT_DONE
            if self.kind_spec.needs_endpoint:
                try:
                    self._connect(self._checkpoint())
                except TimeoutError as exc:
                    self.event("no_server", error=str(exc))
                    return EXIT_NO_SERVER
            worker = self._item_worker()
            self.event("started", n_todo=len(todo), cell_kind=self.ard_cell.kind.value,
                       parallel_items=cell.parallel_items, max_inflight=self.max_inflight)
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

    def _connect(self, ckpt: Checkpoint) -> None:
        """The frozen study connect (pool, tokenizer, verified oracle) + the ARD client.

        ``make_cost_fn({size: oracle})`` is attached exactly as ``study/runner.py:254-261``
        attaches it — without it every ``DecisionRecord.flops`` is the ``None`` placeholder
        and the run produces no FLOP number at all (contract §2).
        """
        if self._client_factory is None:
            from agents_scaling.ard.client import ArdChatClient

            self._client_factory = lambda pool, tokenizer, cost_fn: ArdChatClient(
                pool, tokenizer, cost_fn=cost_fn, run_id=self.run_id)
        super()._connect(ckpt)
        assert self.oracle is not None
        self.cost_fn = make_cost_fn({ckpt.size: self.oracle})
        self.decision_context = DecisionContext(
            checkpoints={ckpt.size: ckpt},
            tokenizers={ckpt.size: self.tokenizer},
            study_seed_hex=self.cfg.study_seed_hex,
            study_id=self.cfg.study_id,
            run_id=self.run_id,
        )

    # ---- decisions -------------------------------------------------------------------
    def decide(
        self,
        *,
        task: ArdTask,
        messages: Sequence[Mapping[str, Any]],
        purpose: str,
        agent_id: int,
        step_slot: int,
        namespace: str,
        thinking_tokens: int | None = None,
        episode_rep: int = 0,
    ) -> DecisionRecord:
        """One logical decision → one committed request (contract §2)."""
        if self.pool is None or self.cost_fn is None or self.decision_context is None:
            raise ProtocolError("decide() before _connect(): this kind declared needs_endpoint=False")
        ckpt = self._checkpoint()
        seed_key = SeedKey(
            source_id=task.item_id,
            split=task.split,
            model_cell=ckpt.model_cell,
            episode_rep=int(episode_rep),
            actor_slot=int(agent_id),
            purpose=purpose,
            step_slot=int(step_slot),
            namespace=namespace,
        )
        request = DecisionRequest(
            messages=tuple(dict(m) for m in messages),
            checkpoint=self.ard_cell.checkpoint,
            thinking_tokens=int(thinking_tokens if thinking_tokens is not None
                                else self.ard_cell.reasoning_allowance),
            seed_key=seed_key,
            purpose=purpose,
        )
        return self.generate(request)

    # ---- per-kind workers ------------------------------------------------------------
    def _item_worker(self) -> Callable[[str], None]:
        worker = resolve_worker(self.ard_cell.kind)
        return lambda item_id: worker(self, item_id)

    def generate(self, request: DecisionRequest) -> DecisionRecord:
        """WP2's ``episode.Generate``: one built request -> one committed decision record."""
        if self.pool is None or self.cost_fn is None or self.decision_context is None:
            raise ProtocolError("generate() before _connect(): this kind declared needs_endpoint=False")
        record = generate_decision(
            request, store=self.store, pool=self.pool, cost_fn=self.cost_fn,
            context=self.decision_context, client=self.client, cell_id=self.cell.cell_id,
            on_event=self._on_store_event,
        )
        with self.shared.lock:
            self.n_decisions += 1
            self.shared.flops_total += int((record.flops or {}).get("total") or 0)
        return record

    def episode_job(self, task: ArdTask) -> EpisodeJob:
        records: dict[str, DecisionRecord] = {}
        arrivals: dict[str, float] = {}

        def _generate(request: DecisionRequest) -> DecisionRecord:
            arrived = self.clock()
            record = self.generate(request)
            records.setdefault(record.request_id, record)
            arrivals.setdefault(record.request_id, arrived)
            return record

        return EpisodeJob(
            cell=self.ard_cell, task=task, checkpoint=self._checkpoint(),
            decide=lambda **kwargs: self.decide(task=task, **kwargs),
            generate=_generate,
            classify=lambda answer: _canonical_class(task.family, answer),
            study_seed=self.cfg.study_seed,
            model_cell=self._checkpoint().model_cell,
            run_root=self.run_root, event=self.event, records=records, arrivals=arrivals,
            choose_protected=resolve_controller_chooser(self.run_root, self.ard_cell),
            clock=self.clock, options=dict(self.ard_cell.options),
        )

    def executor(self) -> EpisodeExecutor:
        if self._episode_executor is None:
            self._episode_executor, self._executor_is_wp2 = resolve_episode_executor()
            if not self._executor_is_wp2:
                self.event("fallback_executor", reason="agents_scaling.ard.episode is absent",
                           cell_kind=self.ard_cell.kind.value)
        return self._episode_executor

    def write_item(
        self,
        task: ArdTask,
        episodes: Sequence[EpisodeRecord],
        *,
        status: str = "complete",
        started: float | None = None,
        grades: Sequence[Mapping[str, Any]] = (),
    ) -> Path:
        result = ArdItemResult(
            schema_version=ARD_SCHEMA_VERSION,
            cell=self.ard_cell,
            source_id=task.item_id,
            family=task.family,
            split=task.split,
            status=status,
            episodes=tuple(episodes),
            timing={"started_at": started if started is not None else self.started_at,
                    "finished_at": self.clock()},
            worker=self._worker_block(),
            grades=tuple(dict(row) for row in grades),
            code_version=self.code_version,
        )
        path = self.paths.item(task.item_id)
        io.write_json(path, result.to_dict())
        return path


# --------------------------------------------------------------------------- workers


def episode_worker(runner: ArdCellRunner, item_id: str) -> None:
    """ROOTS / EPISODE / CHALLENGE / FORK / SUFFIX: run the item's episodes and publish it."""
    started = runner.clock()
    task = runner.tasks[item_id]
    executor = runner.executor()
    job = runner.episode_job(task)
    if (runner.ard_cell.protocol in (K.Protocol.GATE, K.Protocol.CONF_PROTECT)
            and job.choose_protected is None
            and executor is not baseline_episode_executor):
        raise ProtocolError(
            f"{runner.ard_cell.protocol.value} needs the frozen controller's protection "
            "choice (spec §12.3): ard/controller.py (WP5) must publish load_controller(path) "
            "and choose_protected(controller, state, *, round_, episode_id)")
    try:
        episodes = list(executor(job))
    except ContextTooLong as exc:
        runner.event("context_failure", source_id=item_id, error=str(exc))
        runner.write_item(task, (), status="context_failure", started=started)
        return
    runner.write_item(task, episodes, started=started)


def grade_worker(runner: ArdCellRunner, item_id: str) -> None:
    """Evaluator lane: grade the committed answers of the cell's generation cells.

    ``ard.graders`` is imported **inside** this function on purpose: it is the module that
    can read the protected gold namespace, and the firewall is transitive over the
    module-level import closure of every runtime module (contract §0.2).
    """
    from agents_scaling.ard import graders  # local: evaluator namespace only

    started = runner.clock()
    task = runner.tasks[item_id]
    store = graders.bind_gold(runner.run_root)
    rows: list[dict[str, Any]] = []
    # WP2 keys a decision by its content-addressed request id, so one root decision is shared
    # by every protocol that drew it: grade it once, not once per aliasing cell.
    seen: set[str] = set()
    for cell_id in runner.ard_cell.options.get("cells", []):
        path = Path(runner.run_root) / "cells" / str(cell_id) / "items" / f"{item_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload.get("episodes") or []:
            for decision in episode.get("decisions") or []:
                decision_id = str(decision.get("decision_id") or "")
                if decision_id in seen:
                    continue
                seen.add(decision_id)
                answer = decision.get("answer_raw")
                result = graders.grade(task.family, answer, item_id, store=store)
                row = result.to_row()
                row.update({"decision_id": decision.get("decision_id"),
                            "episode_id": decision.get("episode_id"),
                            "source_cell_id": str(cell_id)})
                rows.append(row)
    io.write_json(Path(runner.run_root) / "eval" / "grades" / runner.cell.cell_id /
                  f"{item_id}.json", {"item_id": item_id, "rows": rows})
    runner.write_item(task, (), started=started, grades=rows)


# --------------------------------------------------------------------------- WP2 adapter


def _row_request_id(row: Mapping[str, Any]) -> tuple[str, ...]:
    """A decision replayed without its record still cost the request it is keyed by."""
    rid = str(row.get("request_hash") or "")
    return (rid,) if rid else ()


def _ref_from_wp2_row(
    row: Mapping[str, Any],
    record: DecisionRecord | None,
    *,
    cell: ArdCellSpec,
    episode_id: str,
    protected_slot: int | None,
    classify: Callable[[str | None], str | None],
    arrivals: Mapping[str, float],
) -> DecisionRef:
    """One WP2 ``Decision.to_row()`` (spec §18 shape) → the joinable :class:`DecisionRef`.

    Built from WP2's public row rather than from its internal fields, so the two packages
    meet on the §18 field list and nothing else.
    """
    flops = dict((record.flops or {}) if record is not None else {})
    agent_id = int(row.get("agent_id") or 0)
    answer = row.get("answer_raw")
    return DecisionRef(
        decision_id=str(row.get("decision_id") or ""),
        episode_id=str(row.get("episode_id") or episode_id),
        request_id=str(row.get("request_hash") or (record.request_id if record else "")),
        agent_id=agent_id,
        round=int(row.get("round") or 0),
        phase=str(row.get("phase") or ""),
        protocol=cell.protocol.value if cell.protocol else "NONE",
        checkpoint=cell.checkpoint,
        parent_state_id=row.get("parent_state_id"),
        parent_decision_id=row.get("parent_decision_id"),
        root_report_id=row.get("root_report_id"),
        prior_report_id=row.get("prior_report_id"),
        seen_message_ids=tuple(row.get("seen_message_ids") or ()),
        message_order=tuple(range(len(row.get("seen_message_ids") or ()))),
        n_packets=len(row.get("seen_message_ids") or ()),
        packet_tokens=int(row.get("packet_tokens") or 0),
        protected=protected_slot is not None and agent_id == int(protected_slot),
        answer_raw=answer,
        answer_class=classify(answer) if answer is not None else None,
        content_sha256=row.get("content_sha256"),
        evidence_text=row.get("evidence_text"),
        confidence=row.get("confidence"),
        parse_status=str(row.get("parse_status") or K.ParseStatus.OK.value),
        thinking_tokens=int(row.get("thinking_tokens") or 0),
        final_tokens=int(row.get("final_tokens") or 0),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        cached_prefill_tokens=int(row.get("cached_prefill_tokens") or 0),
        budget_tokens=int(record.budget_tokens) if record is not None else cell.reasoning_allowance,
        cap_hit=bool(row.get("cap_hit")),
        boundary_injected=bool(row.get("boundary_injected")),
        final_overrun=bool(row.get("final_overrun")),
        answer_logprob=row.get("answer_logprob"),
        # A cap-hit decision is two physical requests (client.continue_decision), and the
        # count lives on the record, not on WP2's §18 row.  Without this the cost tables
        # charge every continued decision as one request (defect D-02).
        physical_request_ids=(physical_request_ids_of(record) if record is not None
                              else _row_request_id(row)),
        finish_reason=str(row.get("finish_reason") or ""),
        latency_s=float(row.get("latency") or 0.0),
        arrival_ts=float(arrivals.get(str(row.get("request_hash") or ""), 0.0)),
        endpoint_id=str(row.get("endpoint_id") or ""),
        flops=flops,
        aliased=bool(row.get("aliased")),
        fallback=False,
    )


def _ref_from_report_ref(
    ref: Any, record: DecisionRecord | None, *, cell: ArdCellSpec, episode_id: str,
    classify: Callable[[str | None], str | None], arrivals: Mapping[str, float],
) -> DecisionRef:
    """A WP2 ``ReportRef`` (roots carry no ``Decision`` object) → a round-0 decision row."""
    row = {"decision_id": ref.decision_id, "episode_id": episode_id,
           "agent_id": ref.agent_slot, "round": ref.round_, "phase": "root",
           "request_hash": ref.decision_id, **ref.report.to_dict()}
    if record is not None:
        row.update({"thinking_tokens": record.thinking_tokens,
                    "final_tokens": record.final_tokens,
                    "prompt_tokens": record.prompt_tokens,
                    "cached_prefill_tokens": record.cached_prefill_tokens,
                    "cap_hit": record.cap_hit,
                    "boundary_injected": record.boundary_injected,
                    "answer_logprob": record.answer_logprob,
                    "latency": record.latency_s, "endpoint_id": record.endpoint_id,
                    "finish_reason": record.finish_reason, "aliased": record.aliased})
    return _ref_from_wp2_row(row, record, cell=cell, episode_id=episode_id,
                             protected_slot=None, classify=classify, arrivals=arrivals)


def _state_from_wp2(state: Any, refs: Sequence[DecisionRef], *, protected_slot: int | None) -> RoundState:
    classes = tuple(state.canonical_class(agent.latest.report) for agent in state.agents)
    return RoundState(
        state_id=identity.sha256_hex("|".join(state.state_ids())),
        episode_id=state.episode_id,
        round=int(state.round_),
        decision_ids=tuple(agent.latest.decision_id for agent in state.agents),
        answer_classes=classes,
        valid=tuple(agent.latest.report.valid for agent in state.agents),
        protected_slot=protected_slot,
        leader_id=state.leader,
    )


def _sealed_parent_index(job: EpisodeJob) -> dict[tuple[str, int], tuple[str, ...]]:
    """``{(episode_id, round): decision_ids}`` from the ``state:<sha>`` this cell depends on.

    The register is what makes the parent cohort frozen; without reading it back the seal is
    decorative, and a PEER cell regenerated after sealing would silently give the fork a
    different parent than the one the manifest pinned.
    """
    for dep in job.cell.hard_dependencies:
        kind, sha = parse_dependency(dep)
        if kind != "state":
            continue
        path = register_path(job.run_root, kind, sha)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            (str(row.get("episode_id") or ""), int(row.get("round", -1))):
                tuple(str(x) for x in (row.get("decision_ids") or []))
            for row in payload.get("states") or []
        }
    return {}


def _parent_peer_states(job: EpisodeJob, round_: int) -> Any:
    """The sealed PEER round-``round_`` state this item's forks branch from.

    The parent is another cell's episode, so it is read back from the run rather than
    regenerated: a fork whose parent is re-rolled is not a counterfactual of anything.  The
    PEER cells are named in the FORK cell's ``options['parent_cells']``, and the state that
    comes back is checked against the sealed register before a single request is issued.
    """
    from agents_scaling.ard.states import rebuild_round_states

    item_id = str(job.task.item_id)
    sealed = _sealed_parent_index(job)
    for cell_id in job.cell.options.get("parent_cells", []):
        path = Path(job.run_root) / "cells" / str(cell_id) / "items" / f"{item_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        states = rebuild_round_states(payload, job.classify,
                                      protocols=(K.Protocol.PEER.value,))
        for (episode_id, r), state in states.items():
            if int(r) != int(round_):
                continue
            if sealed:
                want = sealed.get((episode_id, int(r)))
                got = tuple(agent.latest.decision_id for agent in state.agents)
                if want is None:
                    raise ProtocolError(
                        f"fork parent {episode_id} round {r} is not in the sealed register "
                        f"{job.cell.parent_ref}: the fork would branch from a state the "
                        "manifest never pinned"
                    )
                if tuple(want) != got:
                    raise ProtocolError(
                        f"fork parent {episode_id} round {r} has changed since it was sealed "
                        f"(sealed {want[:1]}..., found {got[:1]}...); a fork off a re-rolled "
                        "parent is not a counterfactual of anything"
                    )
            return state
    return None


def _run_fork_cell(job: EpisodeJob, spec: Any) -> list[EpisodeRecord]:
    """E4 (spec §12.1, §19): one private counterfactual branch per focal agent.

    A FORK cell is **not** an episode.  It takes the sealed PEER state at its parent round
    and regenerates exactly one agent privately; every other agent's next step already exists
    in the main trajectory, and that is the pairing that makes ``y_private`` against
    ``y_peer`` a comparison of one prefix under two exposures.  Running a fresh PRIVATE
    episode here instead -- which is what happened before defect D-10 -- produces a different
    trajectory from different roots and pairs with nothing.
    """
    from agents_scaling.ard import forks as FK

    cell = job.cell
    round_ = int(cell.round or 0)
    parent = _parent_peer_states(job, round_)
    if parent is None:
        raise ProtocolError(
            f"fork cell {cell.cell_id} found no PEER round-{round_} state for "
            f"{job.task.item_id!r} in parent_cells={list(cell.options.get('parent_cells', []))}; "
            "a fork cannot branch from a parent that has not been generated"
        )
    focal = [int(a) for a in cell.options.get("focal_agents", range(K.N_AGENTS))]
    made = [
        FK.fork_private(parent, slot, spec=spec, generate=job.generate,
                        study_seed=job.study_seed,
                        thinking_tokens=int(cell.reasoning_allowance))
        for slot in focal
    ]
    decisions = [
        _ref_from_wp2_row(
            {"decision_id": fork.child_decision_ids[0] if fork.child_decision_ids else "",
             "episode_id": spec.episode_id, "agent_id": fork.focal_agent,
             "round": fork.parent_round + 1, "phase": FK.ACTION_PRIVATE,
             "request_hash": fork.child_decision_ids[0] if fork.child_decision_ids else "",
             "parent_state_id": fork.shared_prefix_hash,
             **fork.report.to_dict()},
            fork.record, cell=cell, episode_id=spec.episode_id, protected_slot=None,
            classify=job.classify, arrivals=job.arrivals)
        for fork in made
    ]
    return [EpisodeRecord(
        episode_id=spec.episode_id, item_id=str(job.task.item_id), cohort=cell.cohort.value,
        protocol=(cell.protocol or K.Protocol.PRIVATE).value, checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS[cell.checkpoint],
        reasoning_allowance=int(cell.reasoning_allowance),
        final_allowance=cell.final_allowance, packet_policy="fork_private",
        team_id=cell.team_id, parent_ref=cell.parent_ref,
        root_candidate_ids=tuple(a.root.decision_id for a in parent.agents),
        root_seed_ids=(), condition_manifest_hash=identity.sha256_hex(cell.cell_id),
        decisions=tuple(decisions), states=(),
        # No fork field on EpisodeRecord: compact.fork_rows derives forks.parquet from the
        # decisions of a FORK cell, and each decision already carries its parent_state_id.
        events=tuple(fork.to_row() for fork in made),
    )]


def wp2_episode_executor(job: EpisodeJob) -> list[EpisodeRecord]:
    """Adapter over WP2's ``ard/episode.py`` (contract §2: ``run_episode`` / ``run_roots``).

    A ROOTS cell runs the ``ROOT_DRAWS``-draw stateless bank (spec §5.2) through
    ``run_roots``; every other episode-shaped kind runs ``run_episode`` with this cell's
    protocol, cohort, budget and, for the protection protocols, WP5's controller choice.
    The result is translated into the WP4 record types through WP2's own §18 row shapes.
    """
    from agents_scaling.ard import episode as E

    cell, task = job.cell, job.task
    protocol = cell.protocol or K.Protocol.PRIVATE
    spec = E.EpisodeSpec(
        item_id=str(task.item_id), source_id=str(task.source_id), split=str(task.split),
        family=str(task.family), task_text=str(task.task_text), protocol=protocol,
        checkpoint=cell.checkpoint, model_cell=job.model_cell, cohort=cell.cohort.value,
        episode_rep=int(cell.episode_rep), thinking_tokens=int(cell.reasoning_allowance),
    )
    if cell.kind is ArdCellKind.FORK:
        return _run_fork_cell(job, spec)

    if cell.kind is ArdCellKind.ROOTS:
        refs = E.run_roots(spec, generate=job.generate, draws=range(K.ROOT_DRAWS))
        decisions = tuple(
            _ref_from_report_ref(ref, job.records.get(ref.decision_id), cell=cell,
                                 episode_id=spec.episode_id, classify=job.classify,
                                 arrivals=job.arrivals)
            for ref in refs)
        state = RoundState(
            state_id=identity.sha256_hex(spec.episode_id + "|roots"),
            episode_id=spec.episode_id, round=0,
            decision_ids=tuple(ref.decision_id for ref in decisions[:K.N_AGENTS]),
            answer_classes=tuple(ref.answer_class for ref in decisions[:K.N_AGENTS]),
            valid=tuple(ref.valid for ref in decisions[:K.N_AGENTS]))
        return [EpisodeRecord(
            episode_id=spec.episode_id, item_id=task.item_id, cohort=cell.cohort.value,
            protocol=protocol.value, checkpoint=cell.checkpoint,
            checkpoint_revision=K.REVISIONS[cell.checkpoint],
            reasoning_allowance=K.ROOT_THINKING_TOKENS, final_allowance=cell.final_allowance,
            packet_policy="roots", team_id=cell.team_id, parent_ref=cell.parent_ref,
            root_candidate_ids=tuple(ref.decision_id for ref in decisions[:K.N_AGENTS]),
            root_seed_ids=tuple(str(draw) for draw in range(K.ROOT_DRAWS)),
            condition_manifest_hash=identity.sha256_hex(cell.cell_id),
            decisions=decisions, states=(state,))]

    episode = E.run_episode(
        task, protocol, cell.checkpoint, generate=job.generate, study_seed=job.study_seed,
        model_cell=job.model_cell, classify=job.classify, cohort=cell.cohort.value,
        episode_rep=int(cell.episode_rep), thinking_tokens=int(cell.reasoning_allowance),
        choose_protected=job.choose_protected,
    )
    episode_row = episode.to_row()
    episode_id = str(episode_row["episode_id"])
    root_refs = tuple(
        _ref_from_report_ref(ref, job.records.get(ref.decision_id), cell=cell,
                             episode_id=episode_id, classify=job.classify,
                             arrivals=job.arrivals)
        for ref in episode.roots)
    decisions: list[DecisionRef] = list(root_refs)
    states: list[RoundState] = [_state_from_wp2(episode.root_state, root_refs, protected_slot=None)]
    for result in episode.rounds:
        refs = [
            _ref_from_wp2_row(decision.to_row(), decision.record, cell=cell,
                              episode_id=episode_id, protected_slot=result.protected_slot,
                              classify=job.classify, arrivals=job.arrivals)
            for decision in result.decisions]
        refs.sort(key=lambda ref: ref.agent_id)
        decisions.extend(refs)
        states.append(_state_from_wp2(result.state, refs, protected_slot=result.protected_slot))
    return [EpisodeRecord(
        episode_id=episode_id, item_id=task.item_id,
        cohort=str(episode_row.get("cohort") or cell.cohort.value),
        protocol=str(episode_row.get("protocol") or protocol.value),
        checkpoint=str(episode_row.get("checkpoint") or cell.checkpoint),
        checkpoint_revision=str(episode_row.get("checkpoint_revision") or ""),
        N=int(episode_row.get("N") or K.N_AGENTS),
        horizon=int(episode_row.get("horizon") or K.UPDATE_ROUNDS),
        reasoning_allowance=int(episode_row.get("reasoning_allowance") or cell.reasoning_allowance),
        final_allowance=int(episode_row.get("final_allowance") or cell.final_allowance),
        packet_policy=str(episode_row.get("packet_policy") or protocol.value),
        memory_policy=str(episode_row.get("memory_policy") or ""),
        leader_id=episode_row.get("leader_id"),
        team_id=cell.team_id, parent_ref=cell.parent_ref,
        root_candidate_ids=tuple(episode_row.get("root_candidate_ids") or ()),
        root_seed_ids=tuple(str(draw) for draw in (episode_row.get("root_draws") or ())),
        condition_manifest_hash=identity.sha256_hex(cell.cell_id),
        decisions=tuple(decisions), states=tuple(states))]


# --------------------------------------------------------------------------- baseline loop

_REPORT_CONTRACT = (
    f"Report exactly in this format: {K.FINAL_OPEN}answer{K.FINAL_CLOSE}"
    f"{K.EVIDENCE_OPEN}at most {K.EVIDENCE_MAX_TOKENS} tokens{K.EVIDENCE_CLOSE}"
    f"{K.CONFIDENCE_OPEN}p in [0,1]{K.CONFIDENCE_CLOSE}"
)


def _parse_report(final_text: str, finish_reason: str, completion_ids: Sequence[int]):
    """WP2's parser when it is importable; a tag-only fallback for the offline dry run."""
    try:
        from agents_scaling.ard.report import parse_report
    except ImportError:
        pass
    else:
        return parse_report(final_text, finish_reason, tuple(completion_ids))

    text = final_text or ""
    answer = evidence = None
    confidence = None
    status = K.ParseStatus.MISSING_FINAL
    if text.count(K.FINAL_OPEN) == 1 and K.FINAL_CLOSE in text:
        span = text.split(K.FINAL_OPEN, 1)[1].split(K.FINAL_CLOSE, 1)[0].strip()
        answer, status = (span, K.ParseStatus.OK) if span else (None, K.ParseStatus.EMPTY_FINAL)
    elif K.FINAL_OPEN in text:
        status = K.ParseStatus.MALFORMED_FINAL
    if K.EVIDENCE_OPEN in text and K.EVIDENCE_CLOSE in text:
        evidence = text.split(K.EVIDENCE_OPEN, 1)[1].split(K.EVIDENCE_CLOSE, 1)[0].strip()
    if K.CONFIDENCE_OPEN in text and K.CONFIDENCE_CLOSE in text:
        raw = text.split(K.CONFIDENCE_OPEN, 1)[1].split(K.CONFIDENCE_CLOSE, 1)[0].strip()
        try:
            value = float(raw)
        except ValueError:
            value = None
        confidence = value if value is not None and 0.0 <= value <= 1.0 else None
    return _FallbackReport(answer, evidence, confidence, status)


@dataclass(frozen=True)
class _FallbackReport:
    final_answer: str | None
    evidence: str | None
    confidence: float | None
    parse_status: K.ParseStatus


def _canonical_class(family: str, answer: str | None) -> str | None:
    if answer is None:
        return None
    from agents_scaling.ard import graders  # gold-independent entry point (WP3 interface)

    try:
        return graders.canonical_class(family, answer)
    except Exception:  # noqa: BLE001 - an unkeyable answer casts no vote
        return None


def _root_messages(task: ArdTask) -> tuple[dict[str, str], ...]:
    options = "".join(f"\n({K.MC_LETTERS[i]}) {opt}" for i, opt in enumerate(task.options))
    return ({"role": "user", "content":
             f"Solve the task independently and report in the required format.\n\n"
             f"Task: {task.task_text}{options}\n{_REPORT_CONTRACT}"},)


def _update_messages(
    task: ArdTask, *, own_root: str, own_latest: str, packet: Sequence[str]
) -> tuple[dict[str, str], ...]:
    options = "".join(f"\n({K.MC_LETTERS[i]}) {opt}" for i, opt in enumerate(task.options))
    blocks = [f"Task: {task.task_text}{options}",
              f"Your first report: {own_root}",
              f"Your latest report: {own_latest}"]
    if packet:
        blocks.append("Reports from other agents:\n" + "\n".join(packet))
    else:
        blocks.append("No reports from other agents are available this round.")
    blocks.append("Revise or keep your answer and report again.\n" + _REPORT_CONTRACT)
    return ({"role": "user", "content": "\n\n".join(blocks)},)


def _decision_ref(
    record: DecisionRecord,
    report: Any,
    *,
    episode_id: str,
    cell: ArdCellSpec,
    family: str,
    agent_id: int,
    round_: int,
    phase: str,
    protected: bool,
    seen: Sequence[str],
    parent_decision_id: str | None,
    root_report_id: str | None,
    prior_report_id: str | None,
    arrival_ts: float,
) -> DecisionRef:
    status = getattr(report, "parse_status", K.ParseStatus.OK)
    answer = getattr(report, "final_answer", None)
    return DecisionRef(
        decision_id=decision_id_for(episode_id, round_=round_, agent_id=agent_id, phase=phase),
        episode_id=episode_id,
        request_id=record.request_id,
        agent_id=int(agent_id),
        round=int(round_),
        phase=phase,
        protocol=cell.protocol.value if cell.protocol else "NONE",
        checkpoint=cell.checkpoint,
        parent_decision_id=parent_decision_id,
        root_report_id=root_report_id,
        prior_report_id=prior_report_id,
        seen_message_ids=tuple(seen),
        message_order=tuple(range(len(seen))),
        n_packets=len(seen),
        packet_tokens=0,
        protected=bool(protected),
        answer_raw=answer,
        answer_class=_canonical_class(family, answer),
        evidence_text=getattr(report, "evidence", None),
        confidence=getattr(report, "confidence", None),
        parse_status=(status.value if isinstance(status, K.ParseStatus) else str(status)),
        thinking_tokens=record.thinking_tokens,
        final_tokens=record.final_tokens,
        prompt_tokens=record.prompt_tokens,
        cached_prefill_tokens=record.cached_prefill_tokens,
        budget_tokens=record.budget_tokens,
        cap_hit=record.cap_hit,
        boundary_injected=record.boundary_injected,
        final_overrun=record.final_overrun,
        answer_logprob=record.answer_logprob,
        finish_reason=record.finish_reason,
        latency_s=record.latency_s,
        arrival_ts=arrival_ts,
        endpoint_id=record.endpoint_id,
        flops=dict(record.flops or {}),
        physical_request_ids=tuple(getattr(record, "physical_request_ids", None)
                                   or (record.request_id,)),
        aliased=record.aliased,
        fallback=True,
    )


def baseline_episode_executor(job: EpisodeJob) -> list[EpisodeRecord]:
    """The WP4 dispatch-shape loop used until ``ard/episode.py`` (WP2) lands.

    It honours the invariants WP4 owns — one request per logical decision, five agents in
    every one of the three update rounds even when the team is unanimous (spec §7), a round
    reads only round t-1, the protected slot of a protection protocol receives an empty
    packet, and roots always alias the frozen 1,024-token bank — and nothing else.  It is
    **not** the protocol implementation: packet construction (BALANCED's one-per-class rule,
    HUB5's leader-then-workers order, the pseudonym schedule) is WP2's, so every record it
    writes is stamped ``fallback=True`` / ``packet_policy='fallback:<protocol>'`` and
    ``ard/compact.py`` keeps such rows out of the scientific tables.
    """
    cell, task = job.cell, job.task
    protocol = cell.protocol or K.Protocol.PRIVATE
    episode_id = episode_id_for(cell, task.item_id, episode_rep=cell.episode_rep)
    n_draws = K.ROOT_DRAWS if cell.kind is ArdCellKind.ROOTS else K.N_AGENTS
    decisions: list[DecisionRef] = []
    states: list[RoundState] = []
    roots: list[tuple[DecisionRecord, Any]] = []

    for draw in range(n_draws):
        arrival = job.clock()
        record = job.decide(messages=_root_messages(task), purpose=K.PURPOSE_ROOT,
                            agent_id=0, step_slot=draw, namespace=K.NS_STATELESS_BANK,
                            thinking_tokens=K.ROOT_THINKING_TOKENS)
        report = _parse_report(record.final_text, record.finish_reason, record.completion_token_ids)
        roots.append((record, report))
        decisions.append(_decision_ref(
            record, report, episode_id=episode_id, cell=cell, family=task.family,
            agent_id=draw, round_=0, phase="root", protected=False, seen=(),
            parent_decision_id=None, root_report_id=None, prior_report_id=None,
            arrival_ts=arrival))

    team = roots[:K.N_AGENTS]
    states.append(_state(episode_id, 0, decisions[:len(team)]))

    if cell.kind is not ArdCellKind.ROOTS:
        latest = [report for _, report in team]
        for round_ in range(1, K.UPDATE_ROUNDS + 1):
            previous = list(latest)          # round t reads only round t-1
            protected_slot = (round_ - 1) % K.N_AGENTS if protocol in K.PROTECTION_PROTOCOLS else None
            round_refs: list[DecisionRef] = []
            for agent in range(K.N_AGENTS):
                packet: list[str] = []
                seen: list[str] = []
                if protocol is not K.Protocol.PRIVATE and agent != protected_slot:
                    for other in range(K.N_AGENTS):
                        if other == agent:
                            continue
                        text = previous[other].final_answer or "(no valid report)"
                        packet.append(f"agent-{other}: {text}")
                        seen.append(decision_id_for(episode_id, round_=round_ - 1,
                                                    agent_id=other,
                                                    phase="root" if round_ == 1 else "revise"))
                if protocol is K.Protocol.FRESH:
                    arrival = job.clock()
                    record = job.decide(
                        messages=_root_messages(task), purpose=K.PURPOSE_ROOT, agent_id=0,
                        step_slot=K.fresh_draw_index(round_, agent),
                        namespace=K.NS_STATELESS_BANK, thinking_tokens=K.ROOT_THINKING_TOKENS)
                    packet, seen = [], []
                else:
                    arrival = job.clock()
                    record = job.decide(
                        messages=_update_messages(
                            task,
                            own_root=team[agent][1].final_answer or "(no valid report)",
                            own_latest=previous[agent].final_answer or "(no valid report)",
                            packet=packet),
                        purpose=K.PURPOSE_HUB if protocol is K.Protocol.HUB5 and agent == 0
                        else K.PURPOSE_REVISE,
                        agent_id=agent, step_slot=round_, namespace=K.PROTOCOL_NAMESPACE[protocol],
                        episode_rep=cell.episode_rep)
                report = _parse_report(record.final_text, record.finish_reason,
                                       record.completion_token_ids)
                latest[agent] = report
                round_refs.append(_decision_ref(
                    record, report, episode_id=episode_id, cell=cell, family=task.family,
                    agent_id=agent, round_=round_, phase="revise",
                    protected=agent == protected_slot, seen=seen,
                    parent_decision_id=decision_id_for(
                        episode_id, round_=round_ - 1, agent_id=agent,
                        phase="root" if round_ == 1 else "revise"),
                    root_report_id=decision_id_for(episode_id, round_=0, agent_id=agent,
                                                   phase="root"),
                    prior_report_id=decision_id_for(
                        episode_id, round_=round_ - 1, agent_id=agent,
                        phase="root" if round_ == 1 else "revise"),
                    arrival_ts=arrival))
            decisions.extend(round_refs)
            states.append(_state(episode_id, round_, round_refs, protected_slot=protected_slot))

    episode = EpisodeRecord(
        episode_id=episode_id,
        item_id=task.item_id,
        cohort=cell.cohort.value,
        protocol=protocol.value,
        checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS[cell.checkpoint],
        reasoning_allowance=cell.reasoning_allowance,
        final_allowance=cell.final_allowance,
        packet_policy=f"fallback:{protocol.value}",
        leader_id=0 if protocol is K.Protocol.HUB5 else None,
        team_id=cell.team_id,
        parent_ref=cell.parent_ref,
        root_candidate_ids=tuple(record.request_id for record, _ in team),
        root_seed_ids=tuple(str(draw) for draw in K.TEAM_DRAWS),
        condition_manifest_hash=identity.sha256_hex(cell.cell_id),
        decisions=tuple(decisions),
        states=tuple(states),
        events=({"event": "fallback_executor", "reason": "ard/episode.py (WP2) not importable"},),
    )
    return [episode]


def _state(episode_id: str, round_: int, refs: Sequence[DecisionRef], *,
           protected_slot: int | None = None) -> RoundState:
    classes = tuple(ref.answer_class for ref in refs)
    return RoundState(
        state_id=identity.sha256_hex(json.dumps(
            [episode_id, round_, [ref.decision_id for ref in refs]], sort_keys=True)),
        episode_id=episode_id,
        round=int(round_),
        decision_ids=tuple(ref.decision_id for ref in refs),
        answer_classes=classes,
        valid=tuple(ref.valid for ref in refs),
        protected_slot=protected_slot,
    )


def _default_task_loader(cell: ArdCellSpec):
    """The task loader a cell kind needs when the caller injects none.

    The E3 receiver programs are *generated*, not drawn from ``items.parquet``, so the
    frozen ``study/runner._todo`` would refuse every one of them as "absent from the public
    export" and suspend the cell.  A RECEIVER cell therefore loads its own synthetic tasks;
    everything else reads the frozen public table.  Imported lazily so ``ard.runner`` does
    not depend on the assay module at import time.
    """
    if getattr(cell, "kind", None) == ArdCellKind.RECEIVER:
        def _receiver(root: Path) -> Mapping[str, ArdTask]:
            from agents_scaling.ard import receiver as RX

            return RX.receiver_task_loader(root)

        return _receiver
    return lambda root: load_ard_tasks(root)


def run_ard_cell(
    cell: ArdCellSpec,
    run_root: str | os.PathLike,
    server_run_root: str | os.PathLike,
    shard: int,
    stop_event: threading.Event | None,
    **kwargs: Any,
) -> int:
    """Run one ARD cell and return its exit code (0/2/3/4/5)."""
    return ArdCellRunner(cell, run_root, server_run_root, shard, stop_event, **kwargs).run()


__all__ = [
    "ARD_CONFIG_PATH", "ArdCellRunner", "DEPS_UNMET_FILE", "DependencyStatus", "EpisodeJob",
    "baseline_episode_executor", "episode_worker", "freeze_register", "grade_worker",
    "load_ard_config",
    "register_path", "resolve_controller_chooser", "resolve_episode_executor",
    "resolve_hard_dependencies", "resolve_worker", "run_ard_cell", "unmet_hard_dependencies",
    "wp2_episode_executor",
]
