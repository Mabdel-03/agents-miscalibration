"""§4.3 bounded message packets: the deterministic priority clipper (WP3).

Spec §4.3: "communication uses an intentionally bounded **message packet**, distinct from
its full archived candidate.  The deterministic compiler processes these fields in this
priority order: final-answer display; evidence claims and supports in their original
order; failure checks; alternatives; approach; explicitly personal confidence.  It wraps
the result with anonymous sender slot, original candidate hash, sent-span list and
per-field truncation flags.  Each field is represented as an exact source excerpt, never a
new model-generated semantic summary.  The final-answer display is capped at 1,024
recipient tokens and explicitly marked `partial` when shortened; remaining packet space is
allocated in priority order, with exact final serialized length checked against 2,048.
Empty or failed sources yield the typed unavailable packet.  Freeze the Unicode-prefix
removal rule and test escaping/tokenizer edge cases; tokenizing an excerpt alone is not
sufficient to bound the serialized packet."  Amendment B2 / Table E: "each rendered
subtask_result ≤ 4,096 recipient tokens (exact-excerpt clipping with partial flag, same
compiler as packets)".  Architecture §1.8; audit fixture T13.

Frozen rules
* **Unicode-prefix rule**: every excerpt is ``source[0:end]`` in *code points* (Python
  ``str`` indexing); a cut never lands between a UTF-16 surrogate pair (impossible for
  valid UTF-8 input, guarded anyway) and every span is ``[0, end)`` with the source length.
* **Bound**: the *serialized* wrapper (``json.dumps(..., ensure_ascii=False,
  separators=(",", ":"))``, exactly the bytes the recipient sees) is re-tokenized after
  every admission; a field that does not fit whole is shrunk by binary search over its
  units (list items; evidence ``claim``/``support`` in order) and then over the code-point
  prefix of the first unit that does not fit.  Later fields are absent (empty excerpt,
  ``truncated=True``).  Binary search is deterministic; BPE non-monotonicity can only make
  an excerpt shorter than the true maximum, never let the packet exceed the cap (the final
  count is asserted).
* **Token counting** goes through one ``count(text) -> int`` callable built from the
  tokenizer's ``encode(text, add_special_tokens=False)`` (stub or real), or a callable
  passed directly.
* **Unavailable**: a missing/invalid source (``None``, ``valid=False``, no candidate) or a
  failed subtask yields exactly ``PACKET_UNAVAILABLE_JSON`` (``{"status":"unavailable"}``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.parse.candidate import candidate_sha256
from agents_scaling.study.parse.coordinator import subtask_result_sha256
from agents_scaling.study.types import (
    PACKET_FIELD_PRIORITY,
    PACKET_FINAL_TOKENS_CAP,
    PACKET_TOKENS_CAP,
    PACKET_UNAVAILABLE_JSON,
    SUBTASK_RESULT_TOKENS_CAP,
    Candidate,
    CandidateRecord,
    Packet,
    ProtocolError,
    SubtaskResult,
)

Counter = Callable[[str], int]

#: Table E priority for a rendered subtask result (``subtask_id``/``status``/``confidence``
#: are always present; these are clipped in order).
SUBTASK_FIELD_PRIORITY: tuple[str, ...] = ("result", "evidence_handles", "assumptions", "contract")


# --------------------------------------------------------------------------- token counting


def make_counter(tokenizer: Any) -> Counter:
    """Return ``count(text) -> int`` for a tokenizer (``encode`` API) or pass a callable through."""
    if hasattr(tokenizer, "encode"):

        def count(text: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            return len(ids)

        return count
    if callable(tokenizer):
        return tokenizer
    raise TypeError("tokenizer must expose encode(text, add_special_tokens=False) or be a callable")


def serialize_wrapper(wrapper: Mapping[str, Any]) -> str:
    """The exact packet bytes: compact JSON, UTF-8 preserved, insertion order, no NaN."""
    return json.dumps(wrapper, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- prefix rule


def _is_high(ch: str) -> bool:
    return 0xD800 <= ord(ch) <= 0xDBFF


def _is_low(ch: str) -> bool:
    return 0xDC00 <= ord(ch) <= 0xDFFF


def safe_prefix_end(text: str, end: int) -> int:
    """Frozen Unicode-prefix rule: clamp ``end`` to ``[0, len]`` and never split a surrogate pair."""
    end = max(0, min(end, len(text)))
    if 0 < end < len(text) and _is_high(text[end - 1]) and _is_low(text[end]):
        end -= 1
    return end


def largest_prefix(text: str, budget: int, count: Counter) -> int:
    """Largest code-point prefix length whose *own* token count is ≤ ``budget`` (binary search)."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if count(text) <= budget:
        return len(text)
    lo, hi = 0, len(text)  # invariant: prefix(lo) fits; prefix(hi) does not
    while hi - lo > 1:
        mid = safe_prefix_end(text, (lo + hi) // 2)
        if mid <= lo or mid >= hi:
            break
        if count(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- generic clipper


@dataclass(frozen=True)
class _Unit:
    field: str
    index: int | None  # list item / evidence item
    part: str | None  # "claim" | "support" for evidence
    text: str


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    kind: str  # "str" | "list" | "evidence" | "number"
    value: Any

    def units(self) -> list[_Unit]:
        if self.kind == "str":
            return [_Unit(self.name, None, None, self.value)]
        if self.kind == "list":
            return [_Unit(self.name, i, None, s) for i, s in enumerate(self.value)]
        if self.kind == "evidence":
            out: list[_Unit] = []
            for i, ev in enumerate(self.value):
                out.append(_Unit(self.name, i, "claim", ev.claim))
                out.append(_Unit(self.name, i, "support", ev.support))
            return out
        if self.kind == "number":
            return []
        raise ValueError(f"unknown field kind {self.kind}")


@dataclass(frozen=True)
class _Admitted:
    """Admitted state of one field: ``whole`` full units, then ``partial`` code points of the next."""

    whole: int
    partial: int  # 0 = the next unit is absent
    full: bool


def _materialize(spec: _FieldSpec, adm: _Admitted | None) -> tuple[Any, list[dict[str, Any]]]:
    """Excerpt value + spans for a field under an admitted state (``None`` = absent)."""
    empty = {"str": "", "list": [], "evidence": [], "number": None}[spec.kind]
    if adm is None:
        return empty, []
    if spec.kind == "number":
        return (spec.value if adm.full else None), []
    units = spec.units()
    excerpts: list[tuple[_Unit, int]] = [(u, len(u.text)) for u in units[: adm.whole]]
    if adm.whole < len(units) and adm.partial > 0:
        unit = units[adm.whole]
        excerpts.append((unit, safe_prefix_end(unit.text, adm.partial)))
    spans = [
        {"field": u.field, "index": u.index, "part": u.part, "start": 0, "end": end, "source_len": len(u.text)}
        for u, end in excerpts
    ]
    if spec.kind == "str":
        return (excerpts[0][0].text[: excerpts[0][1]] if excerpts else ""), spans
    if spec.kind == "list":
        return [u.text[:end] for u, end in excerpts], spans
    # evidence: group by item
    items: list[dict[str, Any]] = []
    for u, end in excerpts:
        if u.part == "claim":
            items.append({"claim": u.text[:end], "support": "", "uncertainty": spec.value[u.index].uncertainty})
        else:
            items[-1]["support"] = u.text[:end]
    return items, spans


def clip_fields(
    specs: Sequence[_FieldSpec],
    wrap: Callable[[dict[str, Any], dict[str, bool]], dict[str, Any]],
    count: Counter,
    cap: int,
) -> tuple[dict[str, Any], dict[str, bool], list[dict[str, Any]], str]:
    """Admit ``specs`` in priority order under ``cap`` tokens of the *serialized* wrapper.

    Returns ``(fields, truncated, spans, serialized)``.  Raises ``ProtocolError`` when even
    the empty skeleton does not fit (a harness configuration defect, never a model outcome).
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    state: dict[str, _Admitted | None] = {s.name: None for s in specs}

    def render(current: Mapping[str, _Admitted | None]) -> tuple[dict[str, Any], dict[str, bool], list[dict[str, Any]], str]:
        fields: dict[str, Any] = {}
        truncated: dict[str, bool] = {}
        spans: list[dict[str, Any]] = []
        for spec in specs:
            value, field_spans = _materialize(spec, current[spec.name])
            fields[spec.name] = value
            truncated[spec.name] = not (current[spec.name] is not None and current[spec.name].full)
            spans.extend(field_spans)
        return fields, truncated, spans, serialize_wrapper(wrap(fields, truncated))

    def fits(current: Mapping[str, _Admitted | None]) -> bool:
        return count(render(current)[3]) <= cap

    if not fits(state):
        raise ProtocolError(f"empty packet skeleton exceeds the cap of {cap} tokens")
    for spec in specs:
        units = spec.units()
        trial = dict(state)
        trial[spec.name] = _Admitted(len(units), 0, True)
        if fits(trial):
            state = trial
            continue
        # binary search the number of whole units (0..len(units)-1) that fits
        lo, hi = 0, len(units)  # prefix(lo) fits (0 fits: skeleton), prefix(hi) does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            trial[spec.name] = _Admitted(mid, 0, False)
            if fits(trial):
                lo = mid
            else:
                hi = mid
        whole = lo
        partial = 0
        if whole < len(units):
            text = units[whole].text
            plo, phi = 0, len(text)  # prefix(plo) fits, prefix(phi) does not (whole+1 did not fit)
            while phi - plo > 1:
                mid = safe_prefix_end(text, (plo + phi) // 2)
                if mid <= plo or mid >= phi:
                    break
                trial[spec.name] = _Admitted(whole, mid, False)
                if fits(trial):
                    plo = mid
                else:
                    phi = mid
            partial = plo
        state[spec.name] = _Admitted(whole, partial, False)
        break  # later fields stay absent
    fields, truncated, spans, serialized = render(state)
    n = count(serialized)
    if n > cap:
        raise ProtocolError(f"packet clipper produced {n} tokens > cap {cap}")
    return fields, truncated, spans, serialized


# --------------------------------------------------------------------------- packet ids


def _packet_id(sender_slot: int, content_sha256: str, serialized: str) -> str:
    return identity.sha256_hex(identity.jcs(["packet", sender_slot, content_sha256, serialized]))


def _check_slot(sender_slot: int) -> None:
    if isinstance(sender_slot, bool) or not isinstance(sender_slot, int) or sender_slot < 0:
        raise ValueError(f"sender_slot must be a non-negative int, got {sender_slot!r}")


def unavailable_packet(sender_slot: int, count: Counter, candidate_sha256: str = "") -> Packet:
    """The typed unavailable packet (§4.3) — exact bytes ``PACKET_UNAVAILABLE_JSON``."""
    _check_slot(sender_slot)
    serialized = PACKET_UNAVAILABLE_JSON
    return Packet(
        packet_id=_packet_id(sender_slot, candidate_sha256, serialized),
        sender_slot=sender_slot,
        candidate_sha256=candidate_sha256,
        fields={},
        spans=(),
        truncated={},
        final_partial=False,
        recipient_tokens=count(serialized),
        serialized=serialized,
        unavailable=True,
    )


# --------------------------------------------------------------------------- candidate packets


def compile_candidate_packet(
    candidate: Candidate,
    tokenizer: Any,
    sender_slot: int,
    cap: int = PACKET_TOKENS_CAP,
    final_cap: int = PACKET_FINAL_TOKENS_CAP,
    candidate_hash: str | None = None,
) -> Packet:
    """Compile a *valid* candidate into a bounded packet (the core of :func:`compile_packet`).

    Wrapper: ``{"sender_slot","candidate_sha256","fields":{...},"truncated":{...},"final_partial"}``
    with ``fields``/``truncated`` keyed by the six names in ``PACKET_FIELD_PRIORITY`` order.
    ``final_answer`` is first cut to ≤ ``final_cap`` tokens on its own (``final_partial``
    when shortened), then all fields are admitted in priority order under ``cap`` on the
    serialized bytes; a packet-level cut of the final excerpt also sets ``final_partial``.
    """
    if not isinstance(candidate, Candidate):
        raise TypeError("compile_candidate_packet expects a Candidate")
    _check_slot(sender_slot)
    if final_cap <= 0 or cap <= 0:
        raise ValueError("caps must be positive")
    count = make_counter(tokenizer)
    sha = candidate_hash if candidate_hash is not None else candidate_sha256(candidate)
    final_end = safe_prefix_end(candidate.final_answer, largest_prefix(candidate.final_answer, final_cap, count))
    final_display = candidate.final_answer[:final_end]
    display_partial = final_end < len(candidate.final_answer)
    values = {
        "final_answer": _FieldSpec("final_answer", "str", final_display),
        "evidence": _FieldSpec("evidence", "evidence", candidate.evidence),
        "failure_checks": _FieldSpec("failure_checks", "list", candidate.failure_checks),
        "alternatives_considered": _FieldSpec("alternatives_considered", "list", candidate.alternatives_considered),
        "approach": _FieldSpec("approach", "str", candidate.approach),
        "confidence": _FieldSpec("confidence", "number", candidate.confidence),
    }
    specs = [values[name] for name in PACKET_FIELD_PRIORITY]

    def flags(truncated: Mapping[str, bool]) -> dict[str, bool]:
        # the display cut counts as truncation of final_answer (the same bytes at every probe)
        return dict(truncated, final_answer=bool(truncated["final_answer"] or display_partial))

    def wrap(fields: dict[str, Any], truncated: dict[str, bool]) -> dict[str, Any]:
        flagged = flags(truncated)
        return {
            "sender_slot": sender_slot,
            "candidate_sha256": sha,
            "fields": fields,
            "truncated": flagged,
            "final_partial": flagged["final_answer"],
        }

    fields, clipped, spans, serialized = clip_fields(specs, wrap, count, cap)
    truncated = flags(clipped)
    # the final-answer span must report the *source* length of the full answer, not the display
    for span in spans:
        if span["field"] == "final_answer":
            span["source_len"] = len(candidate.final_answer)
    final_partial = truncated["final_answer"]
    return Packet(
        packet_id=_packet_id(sender_slot, sha, serialized),
        sender_slot=sender_slot,
        candidate_sha256=sha,
        fields=fields,
        spans=tuple(spans),
        truncated=truncated,
        final_partial=final_partial,
        recipient_tokens=count(serialized),
        serialized=serialized,
        unavailable=False,
    )


def compile_packet(
    cand: CandidateRecord | None,
    tokenizer: Any,
    sender_slot: int,
    cap: int = PACKET_TOKENS_CAP,
    final_cap: int = PACKET_FINAL_TOKENS_CAP,
) -> Packet:
    """§4.3 packet of one sampling opportunity's outcome; unavailable for missing/invalid.

    ``cand`` is the sender's ``CandidateRecord`` (``None`` when the opportunity produced
    nothing).  A record with ``valid=False`` or no candidate yields the typed unavailable
    packet carrying the record's ``candidate_sha256`` (empty when there is no record).
    """
    count = make_counter(tokenizer)
    if cand is None:
        return unavailable_packet(sender_slot, count)
    if not isinstance(cand, CandidateRecord):
        raise TypeError("compile_packet expects a CandidateRecord or None")
    if not cand.valid or cand.candidate is None:
        return unavailable_packet(sender_slot, count, cand.candidate_sha256 or "")
    return compile_candidate_packet(cand.candidate, count, sender_slot, cap, final_cap, cand.candidate_sha256)


# --------------------------------------------------------------------------- subtask results (Table E)


def render_subtask_result_packet(
    result: SubtaskResult | None,
    tokenizer: Any,
    sender_slot: int,
    cap: int = SUBTASK_RESULT_TOKENS_CAP,
) -> Packet:
    """Clip a worker's ``SubtaskResult`` to ≤ ``cap`` recipient tokens (amendment B2, Table E).

    Wrapper: ``{"sender_slot","result_sha256","fields":{subtask_id,status,confidence,
    result,evidence_handles,assumptions,contract},"truncated":{...},"partial"}``; the
    three small fields are always whole, the rest are admitted in
    ``SUBTASK_FIELD_PRIORITY`` order with the same clipper as candidate packets.
    ``Packet.candidate_sha256`` carries the result hash (a subtask result is never a
    candidate, §3.5); ``final_partial`` mirrors ``partial`` (the ``result`` field was cut).
    ``None`` (a failed/absent worker return) → the typed unavailable packet.
    """
    count = make_counter(tokenizer)
    _check_slot(sender_slot)
    if result is None:
        return unavailable_packet(sender_slot, count)
    if not isinstance(result, SubtaskResult):
        raise TypeError("render_subtask_result_packet expects a SubtaskResult or None")
    sha = subtask_result_sha256(result)
    specs = [
        _FieldSpec("result", "str", result.result),
        _FieldSpec("evidence_handles", "list", result.evidence_handles),
        _FieldSpec("assumptions", "list", result.assumptions),
        _FieldSpec("contract", "str", result.contract),
    ]
    assert tuple(s.name for s in specs) == SUBTASK_FIELD_PRIORITY

    def wrap(fields: dict[str, Any], truncated: dict[str, bool]) -> dict[str, Any]:
        return {
            "sender_slot": sender_slot,
            "result_sha256": sha,
            "fields": {
                "subtask_id": result.subtask_id,
                "status": result.status,
                "confidence": result.confidence,
                **fields,
            },
            "truncated": truncated,
            "partial": truncated["result"],
        }

    fields, truncated, spans, serialized = clip_fields(specs, wrap, count, cap)
    full_fields = {"subtask_id": result.subtask_id, "status": result.status, "confidence": result.confidence, **fields}
    return Packet(
        packet_id=_packet_id(sender_slot, sha, serialized),
        sender_slot=sender_slot,
        candidate_sha256=sha,
        fields=full_fields,
        spans=tuple(spans),
        truncated=truncated,
        final_partial=truncated["result"],
        recipient_tokens=count(serialized),
        serialized=serialized,
        unavailable=False,
    )


__all__ = [
    "Counter",
    "SUBTASK_FIELD_PRIORITY",
    "clip_fields",
    "compile_candidate_packet",
    "compile_packet",
    "largest_prefix",
    "make_counter",
    "render_subtask_result_packet",
    "safe_prefix_end",
    "serialize_wrapper",
    "unavailable_packet",
]
