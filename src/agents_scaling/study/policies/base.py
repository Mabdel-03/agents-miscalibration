"""Shared episode machinery for the generation policies (WP4).

Spec §3.4 (counters: roster, unique actors, fresh context epochs, peak live contexts,
invocations, candidates), §3.5 (candidate contract, sentinel), §3.6 (semantic seeds; the
engine-seed collision check within an episode), §4.2 (64 solver calls per episode), §4.3
(prompt envelope → ``ContextFailure`` is a *used* opportunity), §6.5 (reserve the known
prompt plus the full decode cap before launch; debit actual; release headroom), §6.7
(logical uncached accounting; a bank hit is revealed only after its reservation is
admitted; stateless draws may be pre-generated in batches with admission computed
afterwards from earlier debits only).  Architecture §1.11, §2.2 (item file), §3 step 5.
Corrections §4 items 7–8.

Design
* :class:`EpisodeContext` is built by the runner per (cell, item).  Policies only ever
  obtain records through :meth:`EpisodeContext.generate_group` (all-or-nothing symmetric
  group) or :meth:`EpisodeContext.generate_prefix` (stateless draws, one indivisible
  reservation each); both reserve before consulting the store and debit actuals afterwards.
* WP3 contracts (candidate/coordinator/subtask parsers, packet compiler, VOTE) are injected
  through :class:`Contracts`; :func:`default_contracts` resolves the real modules lazily and
  fails closed (:class:`ContractsUnavailable`) when one is missing.  Tests inject stubs.
* No module here imports ``study.evaluation`` or ``study.data.protected`` (§10.6).
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agents_scaling.study import identity
from agents_scaling.study.config import StudyConfig
from agents_scaling.study.inference.tokens import count_tokens, render_chat_token_ids
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import ROLE_SOLVER, EpisodeLedger, StopReason
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import (
    NAMESPACES,
    PROMPT_TOKENS_CAP,
    PURPOSES,
    SENTINEL_JSON,
    Candidate,
    CandidateRecord,
    CellSpec,
    Checkpoint,
    ContextFailure,
    Decoding,
    DelegateAction,
    Domain,
    EpisodeResult,
    FinalAction,
    InfraFailure,
    Method,
    Packet,
    ProtocolError,
    PublicTask,
    RequestRecord,
    RequestSpec,
    SeedKey,
    SubtaskResult,
)

EPISODE_SCHEMA_VERSION = 1
CONTEXT_FAILURE_CODE = "CONTEXT_FAILURE"
#: ``candidate_sha256`` of an opportunity without a valid candidate: the hash of the exact
#: sentinel bytes a later revision receives for it (§3.5).
INVALID_CANDIDATE_SHA256 = identity.sha256_hex(SENTINEL_JSON)
DEFAULT_PREFIX_BATCH = 8


class AdmissionRefused(RuntimeError):
    """The ledger refused the next legal group (§6.5 step 4): stop optional work.

    ``reason`` is the ledger's refusal (``BUDGET`` or ``CALL_CAP``).
    """

    def __init__(self, reason: StopReason, owner: str = "") -> None:
        super().__init__(f"admission refused ({reason.value}) for {owner or 'group'}")
        self.reason = reason
        self.owner = owner


class ContractsUnavailable(ProtocolError):
    """A WP3 contract module/function could not be resolved (fail closed, never a fallback)."""


# --------------------------------------------------------------------------- contracts


@dataclass(frozen=True)
class Contracts:
    """The WP3 functions a policy needs, behind fixed keyword signatures.

    * ``parse_candidate(content, finish_reason, reasoning) -> ParsedCandidate`` (attributes
      ``valid, failure_code, candidate, raw_sha256, canonical_sha256``);
    * ``parse_coordinator_action(content, N_total, allowed_handles, finish_reason)`` →
      ``FinalAction | DelegateAction | <error with .code>``;
    * ``parse_subtask_result(content, finish_reason)`` → ``SubtaskResult | <error with .code>``;
    * ``compile_packet(cand, tokenizer, sender_slot, cap, final_cap) -> Packet`` (§4.3);
    * ``subtask_result_packet(result, tokenizer, sender_slot, cap) -> Packet`` (Table E);
    * ``vote(pool, task, study_seed, pool_kind) -> SelectionRecord`` (§4.4/§5.3).
    """

    parse_candidate: Callable[..., Any]
    parse_coordinator_action: Callable[..., Any]
    parse_subtask_result: Callable[..., Any]
    compile_packet: Callable[..., Packet]
    subtask_result_packet: Callable[..., Packet]
    vote: Callable[..., Any]
    candidate_sha256: Callable[[Candidate], str]


def call_by_name(fn: Callable[..., Any], **available: Any) -> Any:
    """Call ``fn`` with the subset of ``available`` its signature accepts (by parameter name).

    Positional-only parameters are filled in declaration order from ``available`` by name.
    A required parameter that is not available raises :class:`ContractsUnavailable` — the
    contract shape is then genuinely different and must be reconciled, never guessed.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ContractsUnavailable(f"cannot inspect {fn!r}: {exc}") from exc
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in available:
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(available[name])
            else:
                kwargs[name] = available[name]
        elif param.default is inspect.Parameter.empty:
            raise ContractsUnavailable(
                f"{getattr(fn, '__qualname__', fn)!r} requires parameter {name!r} which the policy cannot supply "
                f"(available: {sorted(available)})"
            )
    if accepts_var_kw:
        for name, value in available.items():
            kwargs.setdefault(name, value)
    return fn(*args, **kwargs)


def _resolve(module: str, *names: str) -> Callable[..., Any]:
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ContractsUnavailable(f"contract module {module} is not importable: {exc}") from exc
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise ContractsUnavailable(f"{module} exposes none of {names}")


def default_contracts() -> Contracts:
    """Resolve the WP3 modules (lazily; raises :class:`ContractsUnavailable` if absent)."""
    parse_candidate = _resolve("agents_scaling.study.parse.candidate", "parse_candidate")
    parse_action = _resolve("agents_scaling.study.parse.coordinator", "parse_coordinator_action")
    parse_subtask = _resolve("agents_scaling.study.parse.coordinator", "parse_subtask_result")
    compile_packet = _resolve("agents_scaling.study.packets", "compile_packet")
    subtask_packet = _resolve("agents_scaling.study.packets", "subtask_result_packet", "compile_subtask_packet", "render_subtask_result_packet")
    vote = _resolve("agents_scaling.study.selection.vote", "vote")
    candidate_sha256 = _resolve("agents_scaling.study.parse.candidate", "candidate_sha256")
    return Contracts(
        candidate_sha256=candidate_sha256,
        parse_candidate=lambda content, finish_reason, reasoning: call_by_name(
            parse_candidate, content=content, finish_reason=finish_reason, reasoning=reasoning
        ),
        parse_coordinator_action=lambda content, N_total, allowed_handles, finish_reason: call_by_name(
            parse_action, content=content, N_total=N_total, n_total=N_total, allowed_handles=allowed_handles, finish_reason=finish_reason
        ),
        parse_subtask_result=lambda content, finish_reason: call_by_name(parse_subtask, content=content, finish_reason=finish_reason),
        compile_packet=lambda cand, tokenizer, sender_slot, cap, final_cap: call_by_name(
            compile_packet, cand=cand, candidate=cand, record=cand, tokenizer=tokenizer, sender_slot=sender_slot, cap=cap, final_cap=final_cap
        ),
        subtask_result_packet=lambda result, tokenizer, sender_slot, cap: call_by_name(
            subtask_packet, result=result, subtask_result=result, tokenizer=tokenizer, sender_slot=sender_slot, cap=cap
        ),
        vote=lambda pool, task, study_seed, pool_kind: call_by_name(
            vote,
            pool=pool,
            candidates=pool,
            task=task,
            source_id=task.source_id,
            domain=task.domain,
            answer_format=task.answer_format,
            study_seed=study_seed,
            pool_kind=pool_kind,
        ),
    )


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def action_error(parsed: Any) -> tuple[str, str] | None:
    """``(code, detail)`` when ``parsed`` is a typed coordinator-action error, else ``None``."""
    if isinstance(parsed, (FinalAction, DelegateAction)):
        return None
    code = _attr(parsed, "code", "failure_code", "error")
    if not isinstance(code, str) or not code:
        raise ProtocolError(f"coordinator parser returned an unrecognised object {type(parsed).__name__}")
    detail = _attr(parsed, "detail", "message", default="")
    return code, str(detail or "")


def subtask_error(parsed: Any) -> tuple[str, str] | None:
    """``(code, detail)`` when ``parsed`` is a typed subtask-result failure, else ``None``."""
    if isinstance(parsed, SubtaskResult):
        return None
    code = _attr(parsed, "code", "failure_code", "error")
    if not isinstance(code, str) or not code:
        raise ProtocolError(f"subtask parser returned an unrecognised object {type(parsed).__name__}")
    return code, str(_attr(parsed, "detail", "message", default="") or "")


# --------------------------------------------------------------------------- call results


@dataclass
class CallResult:
    """One opportunity of an episode: a committed record or a typed context failure."""

    index: int
    spec: RequestSpec
    role: str
    actor_slot: int
    step: int
    owner: str
    prompt_tokens: int
    reserved_flops: int
    record: RequestRecord | None
    aliased: bool
    context_failure: str | None
    actual_flops: int

    @property
    def launched(self) -> bool:
        return self.record is not None

    @property
    def content(self) -> str | None:
        return None if self.record is None else self.record.response.get("content")

    @property
    def reasoning(self) -> str | None:
        return None if self.record is None else self.record.response.get("reasoning")

    @property
    def finish_reason(self) -> str | None:
        return None if self.record is None else self.record.response.get("finish_reason")

    @property
    def completion_tokens(self) -> int:
        return 0 if self.record is None else int(self.record.response["completion_tokens"])

    @property
    def reasoning_tokens(self) -> int:
        return 0 if self.record is None else int(self.record.response.get("reasoning_tokens") or 0)

    def call_entry(self) -> dict[str, Any]:
        """The ``episode.calls[]`` line (architecture §2.2)."""
        return {
            "request_id": self.spec.request_id,
            "role": self.role,
            "actor_slot": self.actor_slot,
            "step": self.step,
            "owner": self.owner,
            "aliased": self.aliased,
            "context_failure": self.context_failure,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "finish_reason": self.finish_reason,
            "reserved_flops": self.reserved_flops,
            "actual_flops": self.actual_flops,
            "engine_seed": self.spec.engine_seed,
            "seed_key": self.spec.seed_key.as_array(),
        }


# --------------------------------------------------------------------------- episode context


@dataclass
class EpisodeContext:
    """Everything one policy run needs (architecture §1.11); built by the runner per item."""

    task: PublicTask
    cell: CellSpec
    cfg: StudyConfig
    checkpoint: Checkpoint
    store: Any
    client: Any
    tokenizer: Any
    oracle: FlopOracle
    ledger: EpisodeLedger
    executor: Executor | None = None
    contracts: Contracts | None = None
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None
    clock: Callable[[], float] = time.time
    prefix_batch: int = DEFAULT_PREFIX_BATCH

    calls: list[CallResult] = field(init=False, default_factory=list)
    packets: list[dict[str, Any]] = field(init=False, default_factory=list)
    started_at: float = field(init=False, default=0.0)
    n_aliased: int = field(init=False, default=0)
    n_generated: int = field(init=False, default=0)
    n_pregenerated_unadmitted: int = field(init=False, default=0)
    peak_live_contexts: int = field(init=False, default=0)
    _engine_seeds: dict[int, str] = field(init=False, default_factory=dict)
    _prompt_cache: dict[tuple[str, bool], int] = field(init=False, default_factory=dict)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not isinstance(self.task, PublicTask):
            raise TypeError("task must be a PublicTask")
        if not isinstance(self.cell, CellSpec):
            raise TypeError("cell must be a CellSpec")
        if self.task.source_id not in self.cell.items:
            raise ProtocolError(f"{self.task.source_id} is not an item of cell {self.cell.cell_id}")
        if self.cell.checkpoint != self.checkpoint.size:
            raise ProtocolError(f"cell {self.cell.cell_id} names checkpoint {self.cell.checkpoint}, got {self.checkpoint.size}")
        if self.prefix_batch < 1:
            raise ValueError("prefix_batch must be >= 1")
        self.started_at = self.clock()

    # ---- contracts -------------------------------------------------------------------
    @property
    def c(self) -> Contracts:
        if self.contracts is None:
            self.contracts = default_contracts()
        return self.contracts

    # ---- identity helpers ------------------------------------------------------------
    @property
    def study_seed(self) -> bytes:
        return self.cfg.study_seed

    @property
    def N(self) -> int:
        return int(self.cell.N)

    def seed(self, actor_slot: int, purpose: str, step_slot: int, namespace: str) -> SeedKey:
        """§3.6 semantic-seed key for this item/cell (purpose and namespace are frozen sets)."""
        if purpose not in PURPOSES:
            raise ValueError(f"unknown seed purpose {purpose!r}")
        if namespace not in NAMESPACES:
            raise ValueError(f"unknown seed namespace {namespace!r}")
        for name, value in (("actor_slot", actor_slot), ("step_slot", step_slot)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        return SeedKey(
            source_id=self.task.source_id,
            split=self.task.split,
            model_cell=self.checkpoint.model_cell,
            episode_rep=int(self.cell.episode_rep),
            actor_slot=actor_slot,
            purpose=purpose,
            step_slot=step_slot,
            namespace=namespace,
        )

    def spec(
        self,
        messages: Sequence[Mapping[str, Any]] | R.Rendered,
        decoding: Decoding,
        seed_key: SeedKey,
        role: str,
        guided_json: bool | None = None,
    ) -> RequestSpec:
        """A :class:`RequestSpec` for this cell's checkpoint; ``guided_json`` stays the
        decoding's flag unless overridden (never for solver roles, §3.5)."""
        if isinstance(messages, R.Rendered):
            messages = messages.messages
        if guided_json is not None:
            decoding = dataclasses.replace(decoding, guided_json=bool(guided_json))
        return RequestSpec(
            messages=tuple(dict(m) for m in messages),
            decoding=decoding,
            checkpoint=self.checkpoint,
            seed_key=seed_key,
            role=role,
            study_id=self.cfg.study_id,
            study_seed_hex=self.cfg.study_seed_hex,
        )

    # ---- envelopes ---------------------------------------------------------------------
    def prompt_tokens(self, spec: RequestSpec) -> int:
        """Exact rendered prompt token count under the pinned tokenizer (§4.3); memoised per
        (input_hash, enable_thinking) so identical stateless prompts render once."""
        key = (spec.input_hash, bool(spec.decoding.enable_thinking))
        with self._lock:
            cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        n = len(render_chat_token_ids(self.tokenizer, spec.messages, spec.decoding.enable_thinking))
        with self._lock:
            self._prompt_cache[key] = n
        return n

    def own_state_tokens(self, record: CandidateRecord | None) -> int:
        """Recipient tokens of a saved own candidate (the sentinel for an invalid one)."""
        return count_tokens(R.saved_object_text(record if record is not None else SENTINEL_JSON), self.tokenizer)

    def check_own_state(self, record: CandidateRecord | None) -> int:
        """Table E: a valid own candidate above ``caps.own_candidate_tokens`` cannot be carried
        → :class:`ContextFailure` for that scheduled opportunity (§4.3)."""
        n = self.own_state_tokens(record)
        cap = int(self.cfg.caps.own_candidate_tokens)
        if n > cap:
            raise ContextFailure(f"own saved candidate has {n} tokens > cap {cap}")
        return n

    # ---- generation ------------------------------------------------------------------
    def _generate_one(self, spec: RequestSpec) -> tuple[RequestRecord, bool]:
        def on_event(kind: str, payload: Mapping[str, Any]) -> None:
            if self.on_event is not None:
                self.on_event(kind, dict(payload))

        return self.store.get_or_generate(spec, self.client, self.cell.cell_id, on_event=on_event)

    def _physical(self, specs: Sequence[RequestSpec]) -> list[tuple[RequestRecord, bool] | BaseException]:
        """Run the store/client path for ``specs`` in parallel; exceptions are returned, not raised."""
        if not specs:
            return []
        if self.executor is None or len(specs) == 1:
            out: list[tuple[RequestRecord, bool] | BaseException] = []
            for spec in specs:
                try:
                    out.append(self._generate_one(spec))
                except (InfraFailure, ProtocolError) as exc:
                    out.append(exc)
            return out
        futures: list[Future[Any]] = [self.executor.submit(self._generate_one, spec) for spec in specs]
        results: list[tuple[RequestRecord, bool] | BaseException] = []
        for fut in futures:
            try:
                results.append(fut.result())
            except (InfraFailure, ProtocolError) as exc:
                results.append(exc)
        return results

    def _note_record(self, spec: RequestSpec, record: RequestRecord, prompt_tokens: int) -> int:
        """Verify a record against its spec and return the actual debit (§6.7 logical accounting)."""
        if record.request_id != spec.request_id:
            raise ProtocolError(f"store returned {record.request_id[:12]} for spec {spec.request_id[:12]}")
        if int(record.prompt_tokens) != prompt_tokens:
            raise ProtocolError(
                f"request {spec.request_id[:12]}: record has {record.prompt_tokens} prompt tokens, "
                f"the local render has {prompt_tokens} (§4.3 envelope mismatch)"
            )
        finish = record.response.get("finish_reason")
        if finish not in ("stop", "length"):
            raise ProtocolError(f"request {spec.request_id[:12]}: finish_reason {finish!r} is not a completed outcome")
        completion = int(record.response["completion_tokens"])
        if completion > int(spec.decoding.max_tokens):
            raise ProtocolError(f"request {spec.request_id[:12]}: {completion} completion tokens exceed the cap")
        return self.oracle.debit(prompt_tokens, completion)

    def _register_seed(self, spec: RequestSpec) -> None:
        seed = spec.engine_seed
        with self._lock:
            other = self._engine_seeds.get(seed)
            if other is not None and other != spec.request_id:
                raise ProtocolError(f"engine seed collision inside the episode: {other[:12]} vs {spec.request_id[:12]} (§3.6)")
            self._engine_seeds[seed] = spec.request_id

    def _prepare(
        self, specs: Sequence[RequestSpec], forced_failures: Sequence[str | None] | None = None
    ) -> tuple[list[int], list[int], list[str | None]]:
        """Prompt counts, reservation amounts and context-failure notes per spec (no store access).

        ``forced_failures[i]`` (a reason string) marks spec ``i`` as a context failure decided
        by a Table E bound the envelope check cannot see (an own candidate above
        ``caps.own_candidate_tokens``): the opportunity keeps its request id and call slot
        at 0 FLOPs.
        """
        forced = list(forced_failures) if forced_failures is not None else [None] * len(specs)
        if len(forced) != len(specs):
            raise ValueError("forced_failures must align with specs")
        counts: list[int] = []
        amounts: list[int] = []
        failures: list[str | None] = []
        for spec, forced_reason in zip(specs, forced):
            if not isinstance(spec, RequestSpec):
                raise TypeError("specs must be RequestSpec objects")
            if spec.checkpoint != self.checkpoint:
                raise ProtocolError("a spec names a different checkpoint than the episode")
            self._register_seed(spec)
            n = self.prompt_tokens(spec)
            counts.append(n)
            if forced_reason is not None:
                failures.append(str(forced_reason))
                amounts.append(0)
            elif n > PROMPT_TOKENS_CAP:
                failures.append(f"rendered prompt has {n} tokens > cap {PROMPT_TOKENS_CAP}")
                amounts.append(0)
            else:
                failures.append(None)
                amounts.append(self.oracle.reservation(n, int(spec.decoding.max_tokens)))
        return counts, amounts, failures

    def generate_group(
        self,
        specs: Sequence[RequestSpec],
        role: str,
        *,
        owner: str = "",
        keep_final: bool = True,
        extra_reserve: Sequence[int] = (),
        actor_slots: Sequence[int] | None = None,
        steps: Sequence[int] | None = None,
        forced_failures: Sequence[str | None] | None = None,
    ) -> list[CallResult]:
        """Reserve ``specs`` as one indivisible group, run them in parallel, debit actuals.

        Order of operations (§6.5, §6.7): render → count prompt tokens (an over-envelope
        prompt is a 0-FLOP context failure that keeps its call slot) → ``try_reserve`` (with
        ``extra_reserve`` placeholders for dependent work, e.g. CEN_FLAT's not-yet-planned
        workers) → only now ``store.get_or_generate`` (a bank hit is never seen before
        admission) → ``debit`` actuals (aliased records at their recorded cost; placeholders
        released) → results in request order.  Raises :class:`AdmissionRefused` when the
        group does not fit, :class:`InfraFailure`/:class:`ProtocolError` from the client
        after settling the ledger.
        """
        specs = list(specs)
        if not specs:
            raise ValueError("generate_group needs at least one spec")
        actor_slots = list(actor_slots) if actor_slots is not None else [s.seed_key.actor_slot for s in specs]
        steps = list(steps) if steps is not None else [s.seed_key.step_slot for s in specs]
        if len(actor_slots) != len(specs) or len(steps) != len(specs):
            raise ValueError("actor_slots/steps must align with specs")
        counts, amounts, failures = self._prepare(specs, forced_failures)
        extra = [int(x) for x in extra_reserve]
        res = self.ledger.try_reserve(amounts + extra, keep_final=keep_final, owner=owner, role=ROLE_SOLVER)
        if res is None:
            raise AdmissionRefused(self.ledger.last_refusal or StopReason.BUDGET, owner)
        launch = [i for i, f in enumerate(failures) if f is None]
        outcomes = self._physical([specs[i] for i in launch])
        actual: list[int | None] = [0 if f is not None else None for f in failures]
        results: dict[int, CallResult] = {}
        first_error: BaseException | None = None
        with self._lock:
            self.peak_live_contexts = max(self.peak_live_contexts, len(launch))
        for i, outcome in zip(launch, outcomes):
            if isinstance(outcome, BaseException):
                first_error = first_error or outcome
                continue
            record, aliased = outcome
            try:
                cost = self._note_record(specs[i], record, counts[i])
            except ProtocolError as exc:
                first_error = first_error or exc
                continue
            actual[i] = cost
            results[i] = CallResult(i, specs[i], role, actor_slots[i], steps[i], owner, counts[i], amounts[i], record, aliased, None, cost)
        self.ledger.debit(res, actual + [None] * len(extra))
        if first_error is not None:
            raise first_error
        out: list[CallResult] = []
        for i, spec in enumerate(specs):
            if failures[i] is not None:
                out.append(CallResult(i, spec, role, actor_slots[i], steps[i], owner, counts[i], 0, None, False, failures[i], 0))
            else:
                out.append(results[i])
        self._record_calls(out)
        return out

    def generate_prefix(
        self,
        specs: Sequence[RequestSpec],
        role: str,
        *,
        owner: str = "",
        actor_slots: Sequence[int] | None = None,
        steps: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> tuple[list[CallResult], StopReason | None]:
        """Admit stateless draws one at a time, each an indivisible reservation (§4.2), with
        physical pre-generation in batches (§6.7; audit A-1).

        Draw ``k`` is admitted iff its own reservation fits the ledger *after the debits of
        draws ``< k``*; its record is looked at (debited) only after its admission.  Batches
        are bounded by the remaining call slots so no physical work beyond the cap is done.
        Returns ``(admitted results in order, refusal reason or None when every spec was
        admitted)``; pre-generated records past the first refusal are discarded (counted in
        ``n_pregenerated_unadmitted`` — physical telemetry, never episode content).
        """
        specs = list(specs)
        actor_slots = list(actor_slots) if actor_slots is not None else [s.seed_key.actor_slot for s in specs]
        steps = list(steps) if steps is not None else [s.seed_key.step_slot for s in specs]
        if len(actor_slots) != len(specs) or len(steps) != len(specs):
            raise ValueError("actor_slots/steps must align with specs")
        batch = int(batch_size or self.prefix_batch)
        admitted: list[CallResult] = []
        position = 0
        while position < len(specs):
            slots = self.ledger.call_slots_remaining(ROLE_SOLVER)
            if slots <= 0:
                return admitted, StopReason.CALL_CAP
            chunk = specs[position : position + min(batch, slots)]
            counts, amounts, failures = self._prepare(chunk)
            launch = [i for i, f in enumerate(failures) if f is None]
            outcomes = self._physical([chunk[i] for i in launch])
            by_index = dict(zip(launch, outcomes))
            with self._lock:
                self.peak_live_contexts = max(self.peak_live_contexts, len(launch))
            for i, spec in enumerate(chunk):
                k = position + i
                item_owner = f"{owner}[{k}]" if owner else f"draw{k}"
                res = self.ledger.try_reserve([amounts[i]], keep_final=True, owner=item_owner, role=ROLE_SOLVER)
                if res is None:
                    self.n_pregenerated_unadmitted += sum(1 for j in launch if j >= i and not isinstance(by_index[j], BaseException))
                    return admitted, self.ledger.last_refusal or StopReason.BUDGET
                if failures[i] is not None:
                    self.ledger.debit(res, [0])
                    result = CallResult(k, spec, role, actor_slots[k], steps[k], item_owner, counts[i], 0, None, False, failures[i], 0)
                else:
                    outcome = by_index[i]
                    if isinstance(outcome, BaseException):
                        self.ledger.debit(res, [None])
                        raise outcome
                    record, aliased = outcome
                    try:
                        cost = self._note_record(spec, record, counts[i])
                    except ProtocolError:
                        self.ledger.debit(res, [None])
                        raise
                    self.ledger.debit(res, [cost])
                    result = CallResult(k, spec, role, actor_slots[k], steps[k], item_owner, counts[i], amounts[i], record, aliased, None, cost)
                self._record_calls([result])
                admitted.append(result)
            position += len(chunk)
        return admitted, None

    def _record_calls(self, results: Sequence[CallResult]) -> None:
        with self._lock:
            for r in results:
                self.calls.append(r)
                if r.record is not None:
                    if r.aliased:
                        self.n_aliased += 1
                    else:
                        self.n_generated += 1

    # ---- parsing helpers -------------------------------------------------------------
    def candidate_record(self, result: CallResult, *, slot: int, stage: str) -> CandidateRecord:
        """Parse one solver opportunity into its :class:`CandidateRecord` (§3.5; context
        failure → ``valid=False`` with ``failure_code=CONTEXT_FAILURE``)."""
        request_id = result.spec.request_id
        candidate_id = identity.sha256_hex(request_id)
        if result.record is None:
            return CandidateRecord(
                candidate_id=candidate_id,
                request_id=request_id,
                slot=slot,
                stage=stage,
                valid=False,
                failure_code=CONTEXT_FAILURE_CODE,
                candidate=None,
                candidate_sha256=INVALID_CANDIDATE_SHA256,
                raw_content_sha256=identity.sha256_hex(""),
            )
        parsed = self.c.parse_candidate(result.content, result.finish_reason, result.reasoning)
        valid = bool(_attr(parsed, "valid"))
        failure_code = _attr(parsed, "failure_code")
        candidate = _attr(parsed, "candidate")
        raw_sha = _attr(parsed, "raw_sha256", "raw_content_sha256") or identity.sha256_hex(result.content or "")
        cand_sha = _attr(parsed, "canonical_sha256", "candidate_sha256")
        if valid:
            if not isinstance(candidate, Candidate) or failure_code is not None:
                raise ProtocolError("candidate parser returned an inconsistent valid result")
            if not isinstance(cand_sha, str) or len(cand_sha) != 64:
                raise ProtocolError("candidate parser returned no canonical sha256 for a valid candidate")
        else:
            if not isinstance(failure_code, str) or not failure_code:
                raise ProtocolError("candidate parser returned an invalid result without a failure code")
            candidate = None
            cand_sha = INVALID_CANDIDATE_SHA256
        return CandidateRecord(
            candidate_id=candidate_id,
            request_id=request_id,
            slot=slot,
            stage=stage,
            valid=valid,
            failure_code=failure_code,
            candidate=candidate,
            candidate_sha256=cand_sha,
            raw_content_sha256=raw_sha,
        )

    def parse_action(self, result: CallResult, N_total: int, allowed_handles: Sequence[str]) -> Any:
        """``FinalAction | DelegateAction | error`` for a hub call (a context failure or
        ``length`` truncation are typed errors too)."""
        if result.record is None:
            return _Error(CONTEXT_FAILURE_CODE, result.context_failure or "")
        return self.c.parse_coordinator_action(result.content, N_total, tuple(allowed_handles), result.finish_reason)

    def parse_subtask(self, result: CallResult) -> Any:
        if result.record is None:
            return _Error(CONTEXT_FAILURE_CODE, result.context_failure or "")
        return self.c.parse_subtask_result(result.content, result.finish_reason)

    # ---- packets -------------------------------------------------------------------------
    def packet_for(self, record: CandidateRecord | None, sender_slot: int, *, recipient_slot: int, round_: int) -> Packet:
        """§4.3 peer packet from a candidate record (invalid/missing → typed unavailable)."""
        packet = self.c.compile_packet(
            record if (record is not None and record.valid) else None,
            self.tokenizer,
            sender_slot,
            int(self.cfg.caps.packet_tokens),
            int(self.cfg.caps.packet_final_tokens),
        )
        if not isinstance(packet, Packet):
            raise ProtocolError(f"packet compiler returned {type(packet).__name__}, not a Packet")
        if packet.recipient_tokens > int(self.cfg.caps.packet_tokens):
            raise ProtocolError(f"packet {packet.packet_id} has {packet.recipient_tokens} tokens > cap")
        self._note_packet(packet, recipient_slot, round_, "peer")
        return packet

    def subtask_packet(
        self, result: SubtaskResult | None, sender_slot: int, *, recipient_slot: int, cycle: int, cap: int | None = None
    ) -> Packet:
        """Table E: a returned subtask result clipped to ``caps.subtask_result_tokens``, or to
        the smaller ``cap`` a policy derives from its block bound (amendment B2b)."""
        limit = int(self.cfg.caps.subtask_result_tokens)
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError(f"cap must be a positive int, got {cap!r}")
            limit = min(limit, cap)
        packet = self.c.subtask_result_packet(result, self.tokenizer, sender_slot, limit)
        if not isinstance(packet, Packet):
            raise ProtocolError(f"subtask packet compiler returned {type(packet).__name__}, not a Packet")
        if packet.recipient_tokens > limit:
            raise ProtocolError(f"subtask packet {packet.packet_id} has {packet.recipient_tokens} tokens > cap {limit}")
        self._note_packet(packet, recipient_slot, cycle, "subtask_result")
        return packet

    def _note_packet(self, packet: Packet, recipient_slot: int, step: int, kind: str) -> None:
        with self._lock:
            self.packets.append(
                {
                    "packet_id": packet.packet_id,
                    "kind": kind,
                    "sender_slot": packet.sender_slot,
                    "recipient_slot": recipient_slot,
                    "round": step,
                    "candidate_sha256": packet.candidate_sha256,
                    "recipient_tokens": packet.recipient_tokens,
                    "final_partial": packet.final_partial,
                    "truncated": dict(packet.truncated),
                    "unavailable": packet.unavailable,
                    "bytes_sha256": packet.bytes_sha256,
                }
            )

    # ---- selection -------------------------------------------------------------------------
    def vote(self, pool: Sequence[CandidateRecord], pool_kind: str) -> tuple[dict[str, Any], list[CandidateRecord]]:
        """Deterministic VOTE over ``pool`` (0 model cost, §4.4) → ``(selection dict,
        pool with ``vote_key``/``grouping_mode`` filled from the selection record)``."""
        pool = list(pool)
        ids = [c.candidate_id for c in pool]
        if len(set(ids)) != len(ids):
            raise ProtocolError("duplicate candidate ids in a vote pool (§4.4 reject_duplicate_candidate_ids)")
        selection = self.c.vote(pool, self.task, self.study_seed, pool_kind)
        sel = selection.to_dict() if hasattr(selection, "to_dict") else dict(selection)
        sel.setdefault("selector_id", "VOTE")
        sel.setdefault("pool_kind", pool_kind)
        sel.setdefault("pool_candidate_ids", ids)
        keys = sel.get("vote_keys") or {}
        # The selection record carries per-candidate keys but only pool-level grouping-mode
        # counts; a per-candidate mode is exact only when the pool used a single mode.
        modes = sel.get("grouping_mode_counts") or {}
        mode = next(iter(modes)) if len(modes) == 1 else None
        filled = [
            dataclasses.replace(
                c,
                vote_key=keys.get(c.candidate_id, c.vote_key),
                grouping_mode=(mode if (c.valid and mode is not None) else c.grouping_mode),
            )
            for c in pool
        ]
        return sel, filled

    # ---- counters / result ---------------------------------------------------------------
    def counters(self, *, assigned_roster: int, candidates: Sequence[CandidateRecord], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """§3.4 counters from the recorded calls."""
        launched = [c for c in self.calls if c.record is not None]
        failed = [c for c in self.calls if c.record is None]
        calls_by_role: dict[str, int] = {}
        failures_by_role: dict[str, int] = {}
        tokens: dict[str, dict[str, int]] = {}
        for c in launched:
            calls_by_role[c.role] = calls_by_role.get(c.role, 0) + 1
            channel = tokens.setdefault(c.role, {"prompt": 0, "reasoning": 0, "content": 0, "completion": 0})
            channel["prompt"] += c.prompt_tokens
            channel["reasoning"] += c.reasoning_tokens
            channel["completion"] += c.completion_tokens
            channel["content"] += c.completion_tokens - c.reasoning_tokens
        for c in failed:
            failures_by_role[c.role] = failures_by_role.get(c.role, 0) + 1
        # §3.4 "unique logical actors actually used": the anonymous slot is the actor identity
        # (a DEC member's root and revisions are one actor; the hub is slot 0, workers 1..N-1).
        actors = {c.actor_slot for c in launched}
        out = {
            "assigned_roster": int(assigned_roster),
            "unique_actors_used": len(actors),
            "context_epochs": len(launched),
            "resets": max(0, len(launched) - len(actors)),
            "peak_live_contexts": self.peak_live_contexts,
            "model_invocations": len(launched),
            "opportunities": len(self.calls),
            "calls_by_role": calls_by_role,
            "context_failures_by_role": failures_by_role,
            "tokens_by_role_channel": tokens,
            "complete_candidates": sum(1 for c in candidates if c.valid),
            "candidate_opportunities": len(candidates),
            "aliased_calls": self.n_aliased,
            "generated_calls": self.n_generated,
            "pregenerated_unadmitted": self.n_pregenerated_unadmitted,
        }
        if extra:
            out.update(dict(extra))
        return out

    def finish(
        self,
        *,
        candidates: Sequence[CandidateRecord],
        selection: Mapping[str, Any] | None,
        native_final: CandidateRecord | None,
        assigned_roster: int,
        stop_reason: StopReason,
        counters_extra: Mapping[str, Any] | None = None,
        episode_extra: Mapping[str, Any] | None = None,
        code_version: str | None = None,
    ) -> EpisodeResult:
        """Close the ledger (asserting §6.5 invariants) and assemble the item file object."""
        if self.ledger.stop_reason is None:
            self.ledger.stop(stop_reason)
        elif self.ledger.stop_reason is not stop_reason:
            raise ProtocolError(f"ledger stop reason {self.ledger.stop_reason.value} != policy stop reason {stop_reason.value}")
        ledger_summary = self.ledger.close()
        finished_at = self.clock()
        episode_id = identity.sha256_hex(identity.jcs([self.cell.cell_id, self.task.source_id]))
        episode: dict[str, Any] = {
            "episode_id": episode_id,
            "method": self.cell.method.value,
            "N": self.N,
            "B": int(self.cell.B),
            "framing": self.cell.framing.value,
            "calls": [c.call_entry() for c in self.calls],
            "candidates": [c.to_dict() for c in candidates],
            "packets": list(self.packets),
            "ledger": ledger_summary,
            "counters": self.counters(assigned_roster=assigned_roster, candidates=candidates, extra=counters_extra),
            "selection": None if selection is None else dict(selection),
            "native_final": None
            if native_final is None
            else {
                "candidate_id": native_final.candidate_id,
                "valid": native_final.valid,
                "final_answer": None if native_final.candidate is None else native_final.candidate.final_answer,
                "confidence": None if native_final.candidate is None else native_final.candidate.confidence,
            },
            "stop_reason": stop_reason.value,
        }
        if episode_extra:
            episode.update(dict(episode_extra))
        status = "context_failure" if stop_reason is StopReason.CONTEXT_FAILURE else "complete"
        return EpisodeResult(
            schema_version=EPISODE_SCHEMA_VERSION,
            cell=self.cell,
            source_id=self.task.source_id,
            domain=Domain(self.task.domain),
            split=self.task.split,
            status=status,
            episode=episode,
            bank=None,
            timing={"started_at": self.started_at, "finished_at": finished_at, "wall_s": finished_at - self.started_at},
            worker={"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname()},
            code_version=code_version,
        )


@dataclass(frozen=True)
class _Error:
    code: str
    detail: str = ""


# --------------------------------------------------------------------------- policy protocol


@runtime_checkable
class Policy(Protocol):
    """One frozen generation policy (§4.2): ``run(ctx) -> EpisodeResult``."""

    id: Method

    def run(self, ctx: EpisodeContext) -> EpisodeResult: ...


def latest_or_sentinel(record: CandidateRecord | None) -> Any:
    """What a revision sees as its own saved state: the record (rendered as its candidate
    when valid, else the exact sentinel) — ``None`` means no parent at all → sentinel."""
    return record if record is not None else SENTINEL_JSON


__all__ = [
    "AdmissionRefused",
    "CONTEXT_FAILURE_CODE",
    "CallResult",
    "Contracts",
    "ContractsUnavailable",
    "DEFAULT_PREFIX_BATCH",
    "EPISODE_SCHEMA_VERSION",
    "EpisodeContext",
    "INVALID_CANDIDATE_SHA256",
    "Policy",
    "action_error",
    "call_by_name",
    "default_contracts",
    "latest_or_sentinel",
    "subtask_error",
]
