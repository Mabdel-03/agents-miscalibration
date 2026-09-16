"""Recover the per-decision metering that the 1,440 FORK rows of ``decisions.parquet`` lost.

The FORK lane wrote its rows without copying the measurement side of
:class:`~agents_scaling.ard.client.DecisionRecord` onto :class:`~agents_scaling.ard.types.DecisionRef`,
whose metering fields all default to ``0``/``False``.  Nothing was lost in the run itself: the
content-addressed request store (``<run_root>/requests/<id[:2]>/<id>.json``) holds every raw
record, so the fields can be re-derived by replaying the record through the *same* frozen
functions the run used — ``client.decision_from_record`` and, for a two-phase decision,
``client._merged_decision``.  Nothing here re-implements the delimiter accounting; a second,
drifting copy of it is exactly the defect class this repair exists to close.

Because a re-meterer that cannot reproduce known-good rows must not be used on broken ones,
:func:`verify_against` replays NON-fork decisions and asserts byte-equality with their frozen v1
values.  Everything below was measured, not assumed.

Re-verified facts (2026-09-12, ``tables/decisions.parquet``, 67,104 rows)
------------------------------------------------------------------------
* CONFIRMED — all 1,440 ``cohort == 'fork'`` rows carry
  ``thinking_tokens == final_tokens == prompt_tokens == cached_prefill_tokens == latency == 0``
  (``cap_hit`` and ``boundary_injected`` are ``False`` on every one), while
  ``flops_total``/``flops_prefill``/``flops_decode`` are non-zero on every one; means
  1.7246e13 / 8.0898e12 / 9.1565e12, matching the brief's 1.72e13 / 8.09e12 / 9.16e12.
* CONFIRMED — 0 of the 65,664 non-fork rows has ``latency == 0``.
* CONTRADICTED — the FORK rows are *not* uniformly zeroed.  ``final_overrun`` is ``True`` on 7
  of the 1,440, and ``parse_status``, ``answer_raw``, ``answer_class``, ``confidence`` and
  ``arrival_ts`` are populated throughout.  Two further losses the brief does not name:
  ``endpoint_id`` is ``""`` and ``answer_logprob`` is null on all 1,440.
* CONTRADICTED — ``decision_id`` is not a key.  14,336 of the 67,104 rows repeat one (alias
  reuse of a root decision across protocol cells).  The 1,440 FORK ids are unique, so the repair
  is unaffected, but any caller keying this module's output by decision id must know it.
* 196 of the 1,440 FORK decisions were executed as two physical requests, so the repair has to
  handle the continuation path, not just the single-call one.

Recovery result on the real run (all 1,440 FORK decisions)
----------------------------------------------------------
Every one resolves in the store; 0 missing records.  The decisive check is that the replayed
``flops`` block is byte-identical to the frozen ``flops_total`` column on **1,440 of 1,440** —
the frozen FLOPs were computed from this very record, so the record is provably the right one
and the token counts derived beside them are the run's own.  Recovered values: thinking mean
477.7 (124…1024), final mean 91.3 (max 312), prompt mean 335.9 (166…606), ``cached_prefill``
zero on all 1,440, latency mean 7.359 s (2.401…19.559), ``cap_hit`` on 6, ``boundary_injected``
on 196.

Verification result (``verify_against``, actually run; see :func:`main` ``--verify``)
------------------------------------------------------------------------------------
Drawing 200 non-fork decision ids (seed 7) from the 51,261 whose frozen rows agree with each
other expands to **263 compared rows**, and every one reproduces all nine of
:data:`VERIFIED_FIELDS` byte-for-byte: **0 mismatches**, 0 missing records (19 of the 263
needed the span rule for ``final_overrun``; see below).  Stratified, to prove the continuation
path is covered and not merely the easy one: 100 single-phase ids → 112 rows, 0 mismatches;
100 two-phase ids → 141 rows, 0 mismatches.  The replay is therefore fit to be pointed at the
FORK rows.

``--verify`` on the *unfiltered* id pool is expected to fail sometimes and that is the design:
at ``--sample 200 --seed 7`` it drew 231 rows, one of which is one of the 67 defective-provenance
rows below, so ``ok`` is false and the CLI exits 1.  A verification that quietly passed over
those would be worth nothing.

What is recoverable and what is not
-----------------------------------
Recoverable from the store record alone (this module returns them): ``thinking_tokens``,
``final_tokens``, ``prompt_tokens``, ``cached_prefill_tokens``, ``latency``, ``cap_hit``,
``boundary_injected``, ``final_overrun`` (+ ``final_overrun_span``), ``finish_reason``,
``endpoint_id``, ``answer_logprob``, ``think_status``, ``physical_request_ids`` and the whole
``flops`` block.

NOT recoverable, and deliberately absent from the output:

* ``aliased`` — a property of the *run* (did this cell generate the record or find it already
  committed), not of the record.  The store cannot distinguish the two after the fact.
* ``arrival_ts`` — the cell's arrival time, not the request's.  The record's
  ``timing.submitted_at`` is a different quantity and is not substituted for it.
* ``final_text``, ``answer_raw``, ``answer_class``, ``evidence_text``, ``confidence``,
  ``parse_status`` — these need the pinned tokenizer plus ``ard/report.parse_report``.  This
  module ships :class:`_NoDecodeTokenizer` on purpose so it stays hermetic and never claims a
  decoded string it did not decode; the FORK rows already carry these fields anyway.
* every episode-structural column (``episode_id``, ``parent_state_id``, ``round``,
  ``agent_id``, ``seen_message_ids``, ``protocol``, ``protected``, …) — none of it ever entered
  the request store, and no amount of replay can produce it.

``final_overrun`` has two definitions in frozen v1 (measured, previously unreported)
-----------------------------------------------------------------------------------
On a two-phase decision the two lanes that write the column disagree, because the harness
prefills ``"\\n\\n<final>"`` (4 token ids, measured constant ``(271, 27, 11822, 29)`` on every
two-phase record sampled) into phase 2's *prompt*:

* the WP2/episode lane takes it from ``report.parse_report``, which counts the whole
  post-``</think>`` span of the logical completion — prefilled opening included, so
  ``4 + final_tokens > 256``;
* the receiver and selector lanes take ``DecisionRecord.final_overrun``, which counts only what
  phase 2 *generated*, so ``final_tokens > 256``.

Measured agreement on non-fork two-phase rows: natural 20,718/20,718 and high_budget 490/490
follow the span rule; receiver 796/796 follows the generated rule.  Single-phase rows are
unaffected (the rules coincide).  The FORK lane follows the span rule: it reproduces the frozen
``final_overrun`` on 1,440/1,440 where the generated rule manages 1,434/1,440.  This module
therefore returns **both**, and :func:`verify_against` accepts either, counting how often only
the span rule worked.  Which one ``tables_v2`` should publish is a decision for the repair, not
for this module — but the column is not currently one quantity, and callers should stop treating
it as one.  Note the downstream consequence: 14 non-fork two-phase rows are marked overrun by
the span rule alone while ``cap_hit`` is ``False`` (``final_tokens`` 253–255), and all 14 carry
``parse_status == 'truncated_after_final'`` — reports that were never actually cut.

67 frozen rows cannot be reproduced, and the reason is in the row
-----------------------------------------------------------------
67 ``decision_id``s carry rows that disagree with each other on the metering fields.  Replaying
each of those 306 rows against *its own* ``physical_request_ids`` reproduces 239 exactly; the 67
that fail are all ``aliased == False`` with a **single** id in ``physical_request_ids`` while
carrying the *merged* two-phase metering.  The field-level pattern says so unambiguously: over
all 306 rows the mismatches land only on what the merge changes — ``latency`` 67,
``flops_total`` 67, ``final_tokens`` 54, ``final_overrun`` 16, ``cap_hit`` 10 — while
``thinking_tokens``, ``prompt_tokens`` and ``cached_prefill_tokens``, which the merge keeps from
phase 1, agree on all 306.  :func:`flop_oracle_consistent` flags exactly those same 67 rows and
no others, from the row alone, without touching the store.

So their provenance lost phase 2's id — a D-02-shaped defect, not a re-metering failure.  This
module reports them as mismatches rather than papering over them, and does not attempt to
re-derive the missing continuation id: that would need the pinned tokenizer and phase 1's spec,
and guessing it is precisely how a repair becomes the next defect.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from agents_scaling.ard import contract as K
from agents_scaling.ard.client import DecisionRecord, decision_from_record
# ``_merged_decision`` is the frozen two-phase merge (client.py §5.2).  Re-implementing its
# arithmetic here would create the second, drifting copy that this repair exists to remove, so
# it is imported as-is — the same trade ``ard/client.py`` itself makes for the study client's
# private helpers.  It reads exactly one attribute off ``spec`` (``spec.prompt_ids``), which is
# why :func:`remeter_one` can hand it a namespace instead of rebuilding a ``ContinuationSpec``
# whose nine identity components are not needed to measure anything.
from agents_scaling.ard.client import _merged_decision  # noqa: PLC2701
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import ProtocolError

__all__ = [
    "Mismatch",
    "Report",
    "VERIFIED_FIELDS",
    "decisions_table",
    "flop_oracle_consistent",
    "main",
    "remeter_decisions",
    "remeter_one",
    "store_for",
    "verify_against",
]

#: Run-root layout (``store.RequestStore`` owns the sharded path under ``requests/``).
REQUESTS_DIR: Final[str] = "requests"
TABLES_DIR: Final[str] = "tables"
DECISIONS_TABLE: Final[str] = "decisions.parquet"

#: The cohort whose metering this module exists to recover.
FORK_COHORT: Final[str] = "fork"

#: Fields :func:`verify_against` asserts byte-equality on.  The first eight are the brief's;
#: ``flops_total`` is added because it is a free exact-integer check and it is the one that
#: proves the replayed record *is* the record the frozen row was written from.
VERIFIED_FIELDS: Final[tuple[str, ...]] = (
    "thinking_tokens",
    "final_tokens",
    "prompt_tokens",
    "cached_prefill_tokens",
    "latency",
    "cap_hit",
    "boundary_injected",
    "final_overrun",
    "flops_total",
)

_BOOL_FIELDS: Final[frozenset[str]] = frozenset({"cap_hit", "boundary_injected", "final_overrun"})
_FLOAT_FIELDS: Final[frozenset[str]] = frozenset({"latency"})

#: Columns the frozen table must yield for an index + verification pass.  Named explicitly
#: because ``decisions.parquet`` is 31 MB and a full read is never warranted.
_COLUMNS: Final[tuple[str, ...]] = (
    "decision_id",
    "cohort",
    "checkpoint",
    "budget_tokens",
    "physical_request_ids",
    *VERIFIED_FIELDS,
)

#: How many mismatching rows a :class:`Report` keeps verbatim; enough to diagnose, bounded so a
#: whole-table run cannot produce an unreadable report.
MAX_EXAMPLES: Final[int] = 12

_ORACLES: dict[str, FlopOracle] = {}


class _NoDecodeTokenizer:
    """A tokenizer that refuses to invent text (see the module docstring).

    ``decision_from_record`` and ``_merged_decision`` both decode the report span, and a real
    tokenizer would make this module depend on the pinned HF snapshot for a field it does not
    publish.  Returning ``""`` keeps the replay hermetic; ``final_text`` is consequently absent
    from every override this module returns, and callers must not read it back off the
    :class:`~agents_scaling.ard.client.DecisionRecord` these functions build internally.
    """

    def decode(self, ids: Sequence[int]) -> str:
        return ""


# --------------------------------------------------------------------------- run-root layout


def store_for(run_root: str | Path) -> RequestStore:
    """The run's content-addressed request store."""
    return RequestStore(Path(run_root) / REQUESTS_DIR)


def decisions_table(run_root: str | Path) -> Path:
    """The frozen v1 ``decisions.parquet`` (read-only; repairs are written elsewhere)."""
    return Path(run_root) / TABLES_DIR / DECISIONS_TABLE


def _read_rows(run_root: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = decisions_table(run_root)
    if not path.exists():
        raise ProtocolError(f"no {DECISIONS_TABLE} under {path.parent}; compact the run first")
    return pq.read_table(path, columns=list(_COLUMNS)).to_pylist()


def _rows_by_id(
    run_root: str | Path, decision_ids: Iterable[str] | None
) -> dict[str, list[dict[str, Any]]]:
    """``{decision_id: [frozen row, ...]}`` — a list because ``decision_id`` is not a key."""
    wanted = None if decision_ids is None else {str(d) for d in decision_ids}
    out: dict[str, list[dict[str, Any]]] = {}
    for row in _read_rows(run_root):
        key = str(row.get("decision_id") or "")
        if wanted is not None and key not in wanted:
            continue
        out.setdefault(key, []).append(row)
    return out


def _physical_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(i) for i in (row.get("physical_request_ids") or ()))


# --------------------------------------------------------------------------- the replay


def remeter_one(
    store: RequestStore,
    physical_request_ids: Sequence[str],
    *,
    budget_tokens: int,
) -> dict[str, Any]:
    """Replay one logical decision's raw record(s) into a field-override mapping.

    ``physical_request_ids`` is the run's own provenance: one id on the single-call path, phase
    1 then phase 2 on a decision the forced-boundary continuation split in two (client §5.2).
    The phase-2 prompt is an exact token prefix, so the prefilled report opening is recovered
    structurally — everything before it is phase 1's prompt plus its completion through
    ``</think>`` — rather than re-tokenised, which would drag the HF snapshot in for a field
    that changes no measurement.
    """
    ids = tuple(str(i) for i in physical_request_ids)
    if not ids:
        raise ProtocolError("a decision needs at least one physical request id to replay")
    if len(ids) > 2:
        raise ProtocolError(
            f"{ids[0][:12]}: {len(ids)} physical requests; the frozen two-phase contract allows "
            "at most phase 1 and phase 2"
        )

    tokenizer = _NoDecodeTokenizer()
    first = store.get(ids[0])
    if first is None:
        raise FileNotFoundError(f"request record {ids[0]} is not in the store")
    decision = decision_from_record(first, budget_tokens=int(budget_tokens), tokenizer=tokenizer)
    # Single-call: the post-``</think>`` span and what the model generated are the same ids.
    span_tokens = decision.final_tokens

    if len(ids) == 2:
        second = store.get(ids[1])
        if second is None:
            raise FileNotFoundError(f"continuation record {ids[1]} is not in the store")
        if decision.think_status is not None:
            # Phase 2 is triggered by ``boundary_injected``, which requires a well-formed
            # thinking channel.  Without one there is no ``</think>`` to split the re-sent
            # prefix at, so the prefilled opening could not be located and every span-rule
            # field below would be quietly wrong.  Refuse rather than guess.
            raise ProtocolError(
                f"{ids[0][:12]}: phase 1 is {decision.think_status.value} yet a continuation "
                "was recorded; the two-phase prefix cannot be split"
            )
        prefix = tuple(int(i) for i in second.prompt_token_ids)
        prefill = prefix[len(decision.prompt_token_ids) + decision.thinking_tokens + 2:]
        decision = _merged_decision(
            decision,
            second,
            spec=SimpleNamespace(prompt_ids=prefix),
            prefill_ids=prefill,
            tokenizer=tokenizer,
            # Phase 2's endpoint bookkeeping is not in the record and is not inferred; it is
            # absent from the overrides below rather than guessed.
            endpoint_pinned=None,
            reprefilled=False,
            aliased=False,
        )
        span_tokens = len(prefill) + decision.final_tokens

    return _overrides(decision, span_tokens=span_tokens)


def _overrides(decision: DecisionRecord, *, span_tokens: int) -> dict[str, Any]:
    """The recoverable fields of one replayed decision, named as ``decisions.parquet`` names them."""
    flops = dict(decision.flops)
    return {
        "thinking_tokens": int(decision.thinking_tokens),
        "final_tokens": int(decision.final_tokens),
        "prompt_tokens": int(decision.prompt_tokens),
        "cached_prefill_tokens": int(decision.cached_prefill_tokens),
        "latency": float(decision.latency_s),
        "cap_hit": bool(decision.cap_hit),
        "boundary_injected": bool(decision.boundary_injected),
        # Both published rules; see the module docstring on why frozen v1 has two.
        "final_overrun": bool(decision.final_overrun),
        "final_overrun_span": bool(int(span_tokens) > K.FINAL_TOKENS),
        "finish_reason": str(decision.finish_reason),
        "endpoint_id": str(decision.endpoint_id),
        "answer_logprob": decision.answer_logprob,
        "budget_tokens": int(decision.budget_tokens),
        "think_status": None if decision.think_status is None else decision.think_status.value,
        "physical_request_ids": list(decision.physical_request_ids),
        "flops_prefill": flops.get("prefill"),
        "flops_decode": flops.get("decode"),
        "flops_total": flops.get("total"),
        "flops_oracle": flops.get("oracle"),
    }


def remeter_decisions(
    run_root: str | Path, decision_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """``{decision_id: field overrides}`` recovered from the request store.

    An id that names no frozen row, or whose record is not in the store, is **omitted** — a
    partial result the caller can count, never a row of invented zeros (which is the defect
    being repaired).  When one id carries rows with different ``physical_request_ids`` (67 ids
    do; module docstring) the longest tuple wins, because a decision executed as two physical
    requests is two, and the one-id spelling is the defective provenance.
    """
    store = store_for(run_root)
    out: dict[str, dict[str, Any]] = {}
    for decision_id, rows in _rows_by_id(run_root, decision_ids).items():
        row = max(rows, key=lambda r: len(_physical_ids(r)))
        ids = _physical_ids(row)
        if not ids:
            continue
        try:
            out[decision_id] = remeter_one(
                store, ids, budget_tokens=int(row.get("budget_tokens") or 0)
            )
        except (FileNotFoundError, ProtocolError):
            continue
    return out


# --------------------------------------------------------------------------- verification


@dataclass(frozen=True)
class Mismatch:
    """One frozen field the replay did not reproduce."""

    decision_id: str
    field: str
    frozen: Any
    remetered: Any
    physical_requests: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "field": self.field,
            "frozen": self.frozen,
            "remetered": self.remetered,
            "physical_requests": self.physical_requests,
        }


@dataclass(frozen=True)
class Report:
    """What :func:`verify_against` established, and on how many rows.

    ``ok`` is the gate the repair must pass before this module is pointed at the FORK rows:
    every compared row reproduced every field of :data:`VERIFIED_FIELDS`, nothing was missing
    from the store, and something was actually compared.  ``overrun_span_rule`` is not a
    failure — it counts rows whose ``final_overrun`` only the span rule reproduces, which is the
    measured two-definition finding, not a replay defect.
    """

    run_root: str
    requested: int
    unknown_ids: int
    rows_compared: int
    rows_skipped_fork: int
    missing_record: int
    unreplayable: int
    mismatches: Mapping[str, int]
    overrun_span_rule: int
    examples: tuple[Mismatch, ...]

    @property
    def ok(self) -> bool:
        return (
            self.rows_compared > 0
            and self.missing_record == 0
            and self.unreplayable == 0
            and not any(self.mismatches.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_root": self.run_root,
            "requested": self.requested,
            "unknown_ids": self.unknown_ids,
            "rows_compared": self.rows_compared,
            "rows_skipped_fork": self.rows_skipped_fork,
            "missing_record": self.missing_record,
            "unreplayable": self.unreplayable,
            "mismatches": dict(self.mismatches),
            "overrun_span_rule": self.overrun_span_rule,
            "examples": [m.to_dict() for m in self.examples],
            "ok": self.ok,
        }


def _agrees(field: str, frozen: Any, remetered: Any) -> bool:
    """Byte-equality in the column's own type; no tolerance, on any field."""
    if frozen is None or remetered is None:
        return frozen is None and remetered is None
    if field in _BOOL_FIELDS:
        return bool(frozen) == bool(remetered)
    if field in _FLOAT_FIELDS:
        return float(frozen) == float(remetered)
    return int(frozen) == int(remetered)


def verify_against(run_root: str | Path, decision_ids: Iterable[str]) -> Report:
    """Replay NON-fork decisions and assert byte-equality with their frozen v1 metering.

    Each frozen row is replayed against **its own** ``physical_request_ids``, not against a
    per-id consensus: a row that names one physical request while carrying two requests' worth
    of metering is a defect of that row, and collapsing it into a per-id view would hide it.

    FORK rows are skipped and counted: their frozen values are the zeros this module exists to
    replace, so comparing against them would measure the defect rather than the replay.
    """
    store = store_for(run_root)
    # De-duplicated, because ``decision_id`` is not a key and a caller that passes the column
    # straight through would otherwise have its repeats counted twice.
    wanted = list(dict.fromkeys(str(d) for d in decision_ids))
    by_id = _rows_by_id(run_root, wanted)

    counts = {field: 0 for field in VERIFIED_FIELDS}
    examples: list[Mismatch] = []
    compared = skipped = missing = unreplayable = span_rule = 0

    for decision_id in wanted:
        for row in by_id.get(decision_id, ()):
            if str(row.get("cohort")) == FORK_COHORT:
                skipped += 1
                continue
            ids = _physical_ids(row)
            try:
                got = remeter_one(store, ids, budget_tokens=int(row.get("budget_tokens") or 0))
            except FileNotFoundError:
                missing += 1
                continue
            except ProtocolError:
                unreplayable += 1
                continue
            compared += 1
            for field in VERIFIED_FIELDS:
                if _agrees(field, row.get(field), got.get(field)):
                    continue
                # The one field frozen v1 spells two ways; the span rule is a lane's convention,
                # not a disagreement about what the engine did.
                if field == "final_overrun" and bool(row.get(field)) == got["final_overrun_span"]:
                    span_rule += 1
                    continue
                counts[field] += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append(
                        Mismatch(decision_id, field, row.get(field), got.get(field), len(ids))
                    )

    return Report(
        run_root=str(run_root),
        requested=len(wanted),
        unknown_ids=sum(1 for d in wanted if d not in by_id),
        rows_compared=compared,
        rows_skipped_fork=skipped,
        missing_record=missing,
        unreplayable=unreplayable,
        mismatches=counts,
        overrun_span_rule=span_rule,
        examples=tuple(examples),
    )


# --------------------------------------------------------------------------- the FLOP oracle


def flop_oracle_consistent(row: Mapping[str, Any]) -> bool:
    """Does ``flops_total`` follow from this row's own token counts (spec §6.5)?

    The oracle is a pure function of token counts:
    ``call(L, T) = c_lin·S + c_attn·S(S+1)/2`` with ``S = L + T``
    (:class:`~agents_scaling.study.resources.oracle.FlopOracle`), charged as
    ``cost_fn(checkpoint, prompt_tokens, len(completion_token_ids))``.  The row does not carry
    the completion length, so it is reconstructed from the delimiter accounting:
    ``1 (<think>) + thinking_tokens + 1 (</think>) + final_tokens``, plus the terminating
    ``<|im_end|>`` — which is present exactly when the response did *not* stop on the whole-
    response cap, i.e. when ``cap_hit`` is false.  Measured: this reproduces ``flops_total``
    exactly on 43,593 of the 43,660 single-request non-fork rows, and the 67 that fail are
    precisely the 67 whose ``physical_request_ids`` dropped a phase-2 id (module docstring) —
    the predicate found them independently of the replay.

    Domain: rows naming **one** physical request.  A two-phase row's ``flops`` is the sum over
    both requests and phase 2's prompt length (a token prefix, ~1,400 ids) is nowhere in the
    row, so the closed form cannot be evaluated and this returns ``False``.  That is not
    evidence of a defect — 22,200 of 67,104 rows are two-phase — and callers must filter on
    ``len(row["physical_request_ids"]) == 1`` before reading a ``False`` as a finding (a row
    with no such key at all is taken to name one request).  On the FORK rows it returns
    ``False`` for all 1,440, which is the defect stated exactly: non-zero FLOPs cannot follow
    from zero token counts.  A row whose ``flops_total`` is null is inconsistent by definition.

    The oracle is built from the frozen reference dimensions rather than the pinned
    ``config.json``, so it needs no HF snapshot.  That reproduces the arithmetic but *not* the
    provenance hash: ``flops_oracle`` is never recomputed here, only carried through.
    """
    if len(tuple(row.get("physical_request_ids") or ())) > 1:
        return False
    if row.get("flops_total") is None:
        return False
    size = str(row.get("checkpoint"))
    oracle = _ORACLES.get(size)
    if oracle is None:
        oracle = _ORACLES[size] = FlopOracle.from_table(size)
    completion = (
        1 + int(row.get("thinking_tokens") or 0) + 1 + int(row.get("final_tokens") or 0)
        + (0 if bool(row.get("cap_hit")) else 1)
    )
    return oracle.call(int(row.get("prompt_tokens") or 0), completion) == int(row.get("flops_total"))


# --------------------------------------------------------------------------- CLI


def _sample_ids(rows: Sequence[Mapping[str, Any]], n: int, *, seed: int) -> list[str]:
    ids = sorted({str(r["decision_id"]) for r in rows})
    if n <= 0 or n >= len(ids):
        return ids
    return random.Random(seed).sample(ids, n)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agents_scaling.ard.remeter",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--run-root", required=True, help="ARD36 run root (holds requests/ and tables/)")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="replay non-fork decisions and assert byte-equality with frozen v1 (exit 1 on failure)",
    )
    ap.add_argument(
        "--recover-forks",
        action="store_true",
        help="replay every cohort=='fork' decision and summarise the recovered metering",
    )
    ap.add_argument("--sample", type=int, default=200, help="decisions to verify (0 = all)")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed, so a run is reproducible")
    ap.add_argument("--out", default=None, help="write the JSON summary here as well as to stdout")
    args = ap.parse_args(argv)

    rows = _read_rows(args.run_root)
    non_fork = [r for r in rows if str(r.get("cohort")) != FORK_COHORT]
    forks = [r for r in rows if str(r.get("cohort")) == FORK_COHORT]
    summary: dict[str, Any] = {
        "run_root": str(args.run_root),
        "rows": len(rows),
        "fork_rows": len(forks),
        "non_fork_rows": len(non_fork),
        "fork_rows_zeroed": sum(
            1 for r in forks
            if not any(int(r.get(f) or 0) for f in ("thinking_tokens", "final_tokens", "prompt_tokens"))
            and not float(r.get("latency") or 0.0)
        ),
        "non_fork_rows_zero_latency": sum(1 for r in non_fork if not float(r.get("latency") or 0.0)),
        "flop_oracle_inconsistent_single_request": sum(
            1 for r in rows
            if len(_physical_ids(r)) == 1 and not flop_oracle_consistent(r)
        ),
    }

    status = 0
    if args.verify:
        report = verify_against(args.run_root, _sample_ids(non_fork, args.sample, seed=args.seed))
        summary["verify"] = report.to_dict()
        # The whole point: a replay that cannot reproduce known-good rows must not be trusted on
        # the broken ones, so the exit status carries that and not just the printed JSON.
        status = 0 if report.ok else 1

    if args.recover_forks:
        recovered = remeter_decisions(args.run_root, [str(r["decision_id"]) for r in forks])
        summary["recover_forks"] = _fork_summary(forks, recovered)
        if len(recovered) != len(forks):
            status = 1

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return status


def _fork_summary(
    forks: Sequence[Mapping[str, Any]], recovered: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Headline numbers for the recovery, including the check that makes it credible."""
    got = [recovered[str(r["decision_id"])] for r in forks if str(r["decision_id"]) in recovered]
    flops_identical = sum(
        1 for r in forks
        if str(r["decision_id"]) in recovered
        and int(recovered[str(r["decision_id"])]["flops_total"]) == int(r["flops_total"])
    )
    overrun_span = sum(
        1 for r in forks
        if str(r["decision_id"]) in recovered
        and bool(recovered[str(r["decision_id"])]["final_overrun_span"]) == bool(r["final_overrun"])
    )
    return {
        "fork_decisions": len(forks),
        "recovered": len(got),
        "missing_record": len(forks) - len(got),
        # The frozen FLOPs were computed from this record, so byte-identity here is what makes
        # the token counts beside them the run's own and not a plausible reconstruction.
        "flops_total_identical_to_frozen": flops_identical,
        "frozen_final_overrun_reproduced_by_span_rule": overrun_span,
        "two_phase": sum(1 for o in got if len(o["physical_request_ids"]) == 2),
        "thinking_tokens_mean": (sum(o["thinking_tokens"] for o in got) / len(got)) if got else None,
        "final_tokens_mean": (sum(o["final_tokens"] for o in got) / len(got)) if got else None,
        "prompt_tokens_mean": (sum(o["prompt_tokens"] for o in got) / len(got)) if got else None,
        "latency_mean": (sum(o["latency"] for o in got) / len(got)) if got else None,
        "cap_hit": sum(1 for o in got if o["cap_hit"]),
        "boundary_injected": sum(1 for o in got if o["boundary_injected"]),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
