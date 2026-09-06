"""Lightweight stand-ins for the WP3 contracts and an ``EpisodeContext`` factory (WP4 tests).

The stubs implement the *documented* signatures of ``parse_candidate``,
``parse_coordinator_action``, ``parse_subtask_result``, ``compile_packet``,
``subtask_result_packet`` and ``vote`` (architecture §1.8–1.9) strictly enough for the
policy tests (one-fence strip, strict JSON, exact keys, typed error codes, packets bounded
by the stub tokenizer, plurality with an HMAC tie).  They are deliberately small: the real
WP3 modules own the edge cases.  ``make_context`` wires WP2's fake vLLM server, the request
store and client, the 32B oracle and a ledger into one :class:`EpisodeContext`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study import types as T
from agents_scaling.study.config import StudyConfig
from agents_scaling.study.inference import client as C
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.policies.base import Contracts, EpisodeContext
from agents_scaling.study.resources.broker import EpisodeLedger
from agents_scaling.study.resources.oracle import FlopOracle
from tests.study.wp2_support import fake_server  # noqa: F401  (re-exported for the tests)

CANDIDATE_KEYS = ("approach", "evidence", "alternatives_considered", "failure_checks", "final_answer", "confidence")


# --------------------------------------------------------------------------- strict json


def _reject_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("DUPLICATE_KEY")
        out[key] = value
    return out


def _reject_const(name: str) -> Any:
    raise ValueError("NONFINITE")


def strip_one_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1 and text.endswith("```"):
            text = text[first_nl + 1 : -3].strip()
    return text


def load_strict(text: str) -> tuple[Any | None, str | None]:
    text = strip_one_fence(text)
    if not text:
        return None, "EMPTY"
    try:
        value, end = json.JSONDecoder(object_pairs_hook=_reject_dupes, parse_constant=_reject_const).raw_decode(text)
    except ValueError as exc:
        code = str(exc) if str(exc) in ("DUPLICATE_KEY", "NONFINITE") else "NOT_JSON"
        return None, code
    if text[end:].strip():
        return None, "TRAILING_TEXT"
    return value, None


def _candidate_from_obj(obj: Any) -> T.Candidate | None:
    if not isinstance(obj, dict) or tuple(sorted(obj)) != tuple(sorted(CANDIDATE_KEYS)):
        return None
    try:
        ev = tuple(T.Evidence(str(e["claim"]), str(e["support"]), str(e["uncertainty"])) for e in obj["evidence"])
    except (KeyError, TypeError):
        return None
    if any(e.uncertainty not in T.UNCERTAINTY_LEVELS for e in ev):
        return None
    conf = obj["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
        return None
    if not all(isinstance(obj[k], str) for k in ("approach", "final_answer")):
        return None
    if not all(isinstance(obj[k], list) and all(isinstance(s, str) for s in obj[k]) for k in ("alternatives_considered", "failure_checks")):
        return None
    return T.Candidate(obj["approach"], ev, tuple(obj["alternatives_considered"]), tuple(obj["failure_checks"]), obj["final_answer"], float(conf))


def candidate_sha256(candidate: T.Candidate) -> str:
    return identity.sha256_hex(json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False))


@dataclass(frozen=True)
class StubParsed:
    valid: bool
    failure_code: str | None
    candidate: T.Candidate | None
    raw_sha256: str
    canonical_sha256: str | None


def parse_candidate(content: str | None, finish_reason: str, reasoning: str | None = None) -> StubParsed:
    content = content or ""
    raw = identity.sha256_hex(content)
    if finish_reason == "length":
        return StubParsed(False, "TRUNCATED", None, raw, None)
    obj, code = load_strict(content)
    if code is not None:
        return StubParsed(False, code, None, raw, None)
    cand = _candidate_from_obj(obj)
    if cand is None:
        return StubParsed(False, "SCHEMA", None, raw, None)
    return StubParsed(True, None, cand, raw, candidate_sha256(cand))


@dataclass(frozen=True)
class StubError:
    code: str
    detail: str = ""


def parse_coordinator_action(content: str | None, N_total: int, allowed_handles: Sequence[str], finish_reason: str) -> Any:
    if finish_reason == "length":
        return StubError("TRUNCATED")
    obj, code = load_strict(content or "")
    if code is not None:
        return StubError(code)
    if not isinstance(obj, dict) or "action" not in obj:
        return StubError("INVALID", "no action")
    if obj["action"] == "final":
        if set(obj) != {"action", "candidate"}:
            return StubError("INVALID", "final keys")
        cand = _candidate_from_obj(obj["candidate"])
        if cand is None:
            return StubError("INVALID", "final candidate schema")
        return T.FinalAction(cand)
    if obj["action"] == "delegate":
        if set(obj) != {"action", "assignments"} or not isinstance(obj["assignments"], list) or not obj["assignments"]:
            return StubError("INVALID", "delegate shape")
        if len(obj["assignments"]) > N_total - 1:
            return StubError("TOO_MANY", f"{len(obj['assignments'])} > {N_total - 1}")
        seen: set[int] = set()
        out = []
        for a in obj["assignments"]:
            keys = {"worker_slot", "subtask_id", "question", "source_handles", "required_output_type", "return_contract"}
            if not isinstance(a, dict) or set(a) != keys:
                return StubError("INVALID", "assignment keys")
            slot = a["worker_slot"]
            if not isinstance(slot, int) or isinstance(slot, bool) or not 1 <= slot <= N_total - 1:
                return StubError("BAD_SLOT", f"slot {slot!r}")
            if slot in seen:
                return StubError("DUP_SLOT", f"slot {slot}")
            seen.add(slot)
            handles = a["source_handles"]
            if not isinstance(handles, list) or any(h not in allowed_handles for h in handles):
                return StubError("HIDDEN_HANDLE", str(handles))
            out.append(T.Assignment(slot, str(a["subtask_id"]), str(a["question"]), tuple(handles), str(a["required_output_type"]), str(a["return_contract"])))
        return T.DelegateAction(tuple(out))
    return StubError("INVALID", f"action {obj['action']!r}")


def parse_subtask_result(content: str | None, finish_reason: str) -> Any:
    if finish_reason == "length":
        return StubError("TRUNCATED")
    obj, code = load_strict(content or "")
    if code is not None:
        return StubError(code)
    keys = {"subtask_id", "contract", "status", "result", "assumptions", "evidence_handles", "confidence"}
    if not isinstance(obj, dict) or set(obj) != keys or obj["status"] not in T.SUBTASK_STATUSES:
        return StubError("SCHEMA")
    conf = obj["confidence"]
    if conf is not None and (isinstance(conf, bool) or not isinstance(conf, (int, float))):
        return StubError("SCHEMA")
    return T.SubtaskResult(str(obj["subtask_id"]), str(obj["contract"]), obj["status"], str(obj["result"]), tuple(obj["assumptions"]), tuple(obj["evidence_handles"]), None if conf is None else float(conf))


# --------------------------------------------------------------------------- packets


def _ntok(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _clip(tokenizer: Any, text: str, cap: int) -> tuple[str, bool]:
    if _ntok(tokenizer, text) <= cap:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _ntok(tokenizer, text[:mid]) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo], True


def _packet(sender_slot: int, cand_sha: str, fields: dict[str, Any], truncated: dict[str, bool], final_partial: bool, tokenizer: Any) -> T.Packet:
    body = {"sender_slot": sender_slot, "candidate_sha256": cand_sha, "fields": fields, "truncated": truncated, "final_partial": final_partial}
    serialized = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return T.Packet(
        packet_id=identity.sha256_hex(serialized)[:16],
        sender_slot=sender_slot,
        candidate_sha256=cand_sha,
        fields=fields,
        spans=(),
        truncated=truncated,
        final_partial=final_partial,
        recipient_tokens=_ntok(tokenizer, serialized),
        serialized=serialized,
        unavailable=False,
    )


def unavailable_packet(sender_slot: int, tokenizer: Any) -> T.Packet:
    return T.Packet(
        packet_id=f"unavail-{sender_slot}",
        sender_slot=sender_slot,
        candidate_sha256="",
        fields={},
        spans=(),
        truncated={},
        final_partial=False,
        recipient_tokens=_ntok(tokenizer, T.PACKET_UNAVAILABLE_JSON),
        serialized=T.PACKET_UNAVAILABLE_JSON,
        unavailable=True,
    )


def compile_packet(cand: T.CandidateRecord | None, tokenizer: Any, sender_slot: int, cap: int = 2048, final_cap: int = 1024) -> T.Packet:
    if cand is None or not cand.valid or cand.candidate is None:
        return unavailable_packet(sender_slot, tokenizer)
    c = cand.candidate
    final, partial = _clip(tokenizer, c.final_answer, final_cap)
    ordered: list[tuple[str, Any]] = [
        ("final_answer", final),
        ("evidence", [e.to_dict() for e in c.evidence]),
        ("failure_checks", list(c.failure_checks)),
        ("alternatives_considered", list(c.alternatives_considered)),
        ("approach", c.approach),
        ("confidence", c.confidence),
    ]
    truncated = {k: False for k, _ in ordered}
    truncated["final_answer"] = partial
    for keep in range(len(ordered), 0, -1):
        fields = dict(ordered[:keep])
        trunc = {k: (v if k in fields else True) for k, v in truncated.items()}
        packet = _packet(sender_slot, cand.candidate_sha256, fields, trunc, partial, tokenizer)
        if packet.recipient_tokens <= cap:
            return packet
    final2, _ = _clip(tokenizer, final, max(1, final_cap // 4))
    return _packet(sender_slot, cand.candidate_sha256, {"final_answer": final2}, {k: True for k in truncated}, True, tokenizer)


def subtask_result_packet(result: T.SubtaskResult | None, tokenizer: Any, sender_slot: int, cap: int = 4096) -> T.Packet:
    if result is None:
        return unavailable_packet(sender_slot, tokenizer)
    fields = result.to_dict()
    truncated = {"result": False}
    serialized = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    if _ntok(tokenizer, serialized) > cap:
        fields["result"], truncated["result"] = _clip(tokenizer, fields["result"], max(1, cap // 2))
        serialized = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return T.Packet(
        packet_id=identity.sha256_hex(serialized)[:16],
        sender_slot=sender_slot,
        candidate_sha256="",
        fields=fields,
        spans=(),
        truncated=truncated,
        final_partial=False,
        recipient_tokens=_ntok(tokenizer, serialized),
        serialized=serialized,
        unavailable=False,
    )


# --------------------------------------------------------------------------- vote


def vote_key(final_answer: str, task: T.PublicTask) -> tuple[str, str]:
    text = final_answer.strip()
    if task.answer_format == "multipleChoice" and len(text) == 1 and text.isalpha():
        return text.upper(), "mc_letter"
    return " ".join(text.lower().split()), "exact_norm"


def vote(pool: Sequence[T.CandidateRecord], task: T.PublicTask, study_seed: bytes, pool_kind: str) -> T.SelectionRecord:
    keys: dict[str, str | None] = {}
    modes: dict[str, int] = {}
    classes: dict[str, list[str]] = {}
    for c in pool:
        if c.valid and c.candidate is not None:
            key, mode = vote_key(c.candidate.final_answer, task)
            keys[c.candidate_id] = key
            modes[mode] = modes.get(mode, 0) + 1
            classes.setdefault(key, []).append(c.candidate_id)
        else:
            keys[c.candidate_id] = None
    valid_count = sum(1 for k in keys.values() if k is not None)
    if not classes:
        return T.SelectionRecord("VOTE", pool_kind, tuple(c.candidate_id for c in pool), None, len(pool), 0, True, task.answer_format, None, {}, None, None, False, keys)
    top = max(len(v) for v in classes.values())
    tied = [k for k, v in classes.items() if len(v) == top]

    def tie(cid: str) -> bytes:
        return hmac.new(study_seed, T.VOTE_TIE_NAMESPACE + cid.encode(), hashlib.sha256).digest()

    winner_key = min(tied, key=lambda k: tie(min(classes[k], key=tie)))
    representative = min(classes[winner_key], key=tie)
    dominant = max(modes, key=modes.get)
    return T.SelectionRecord(
        "VOTE",
        pool_kind,
        tuple(c.candidate_id for c in pool),
        representative,
        len(pool),
        valid_count,
        False,
        task.answer_format,
        dominant,
        modes,
        top,
        len(tied),
        top == 1,
        keys,
    )


def stub_contracts() -> Contracts:
    return Contracts(
        parse_candidate=parse_candidate,
        parse_coordinator_action=parse_coordinator_action,
        parse_subtask_result=parse_subtask_result,
        compile_packet=compile_packet,
        subtask_result_packet=subtask_result_packet,
        vote=vote,
        candidate_sha256=candidate_sha256,
    )


# --------------------------------------------------------------------------- context factory


def make_task(source_id: str = "hle:wp4-item-1", text: str = "What is 2 + 2? Answer with the single letter of the correct option: A) 3 B) 4 C) 5", **kw: Any) -> T.PublicTask:
    defaults = dict(source_id=source_id, domain=T.Domain.HLE, split="dev", task_text=text, answer_format="multipleChoice", stratum="Gold", rank=0, task_tokens=len(text.split()))
    defaults.update(kw)
    return T.PublicTask(**defaults)


def make_cell(
    method: T.Method,
    *,
    N: int = 5,
    module: str = "A",
    framing: T.Framing = T.Framing.F00,
    B: int = 4,
    items: Sequence[str] = ("hle:wp4-item-1",),
    degree: int | None = None,
    checkpoint: str = "32B",
    episode_rep: int = 0,
    max_inflight: int = 8,
) -> T.CellSpec:
    return T.CellSpec(
        cell_id=f"{module}.{method.value}.{checkpoint}.N{N}.B{B}.F{framing.value}.e{episode_rep}.s000",
        kind=T.CellKind.GENERATE,
        module=module,
        method=method,
        checkpoint=checkpoint,
        N=N,
        B=B,
        framing=framing,
        episode_rep=episode_rep,
        split="dev",
        items=tuple(items),
        depends_on=(),
        max_inflight=max_inflight,
        parallel_items=1,
        lane=checkpoint,
        degree=degree,
    )


class ContextFactory:
    """Builds :class:`EpisodeContext` objects against one fake endpoint and one store."""

    def __init__(self, cfg: StudyConfig, run_root: Path, server: Any, *, size: str = "32B", max_workers: int = 8) -> None:
        self.cfg = cfg
        self.run_root = Path(run_root)
        self.server = server
        self.size = size
        self.store = RequestStore(self.run_root / "requests")
        pool = C.EndpointPool(self.run_root, cfg.checkpoints[size].profile, shard=0, refresh_min_interval_s=0.0, probe_timeout=2.0)
        self.oracle = FlopOracle.from_table(size)
        self.client = C.VllmChatClient(pool, server.tokenizer, run_id="wp4-test")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def ledger(self, B: int, **kw: Any) -> EpisodeLedger:
        kw.setdefault("solver_call_cap", self.cfg.caps.solver_calls)
        kw.setdefault("selector_call_cap", self.cfg.caps.selector_calls)
        return EpisodeLedger(B, self.oracle, 0, **kw)

    def context(
        self,
        task: T.PublicTask,
        cell: T.CellSpec,
        B: int,
        *,
        executor: bool = True,
        call_cap: int | None = None,
        contracts: str = "stub",
        **kw: Any,
    ) -> EpisodeContext:
        """``contracts="stub"`` injects this module's stand-ins; ``"real"`` leaves resolution
        to :func:`default_contracts` (the WP3 modules, lazily)."""
        if contracts not in ("stub", "real"):
            raise ValueError("contracts must be 'stub' or 'real'")
        ledger_kw = {} if call_cap is None else {"solver_call_cap": call_cap}
        return EpisodeContext(
            task=task,
            cell=cell,
            cfg=self.cfg,
            checkpoint=self.cfg.checkpoints[self.size],
            store=self.store,
            client=self.client,
            tokenizer=self.server.tokenizer,
            oracle=self.oracle,
            ledger=self.ledger(B, **ledger_kw),
            executor=self.executor if executor else None,
            contracts=stub_contracts() if contracts == "stub" else None,
            on_event=lambda kind, payload: self.events.append((kind, dict(payload))),
            **kw,
        )

    def root_reservation(self, task: T.PublicTask, framing: T.Framing = T.Framing.F00) -> int:
        """The reservation of one root draw on this tokenizer (for budget arithmetic in tests)."""
        from agents_scaling.study.inference.tokens import render_chat_token_ids
        from agents_scaling.study.prompts import render as R

        n = len(render_chat_token_ids(self.server.tokenizer, R.render_root(task, framing).messages, True))
        return self.oracle.reservation(n, T.SOLVER_OUT_CAP)

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def bank_request_id(cfg: StudyConfig, task: T.PublicTask, framing: T.Framing, k: int, size: str = "32B", episode_rep: int = 0) -> str:
    """The F-bank request id of draw ``k`` (framing cell) — the alias target (architecture §2.3)."""
    from agents_scaling.study.prompts import render as R

    ckpt = cfg.checkpoints[size]
    key = T.SeedKey(task.source_id, task.split, ckpt.model_cell, episode_rep, 0, T.PURPOSE_ROOT, k, T.NS_STATELESS_BANK)
    return T.RequestSpec(tuple(R.render_root(task, framing).messages), T.SOLVER_DECODING, ckpt, key, "root", cfg.study_id, cfg.study_seed_hex).request_id


__all__ = [name for name in globals() if not name.startswith("_")]
