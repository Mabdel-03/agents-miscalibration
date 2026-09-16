"""The E3 receiver outcome table (Stage 0 step 10) — the 1,280 probes nothing ever graded.

WP5 generated every receiver decision the assay asks for and then left all of them
unlabelled: ``labels.parquet`` holds **no row at all** for any of the 1,280, so
``compact._is_correct`` returns ``None`` for every one and ``analysis.secondary_analysis``
registers ``/secondary/receiver`` as ``status="unexecuted", n_contrasts=0``.  This module
produces the missing grades as a *separate* table, ``receiver_outcomes.parquet``, so that no
main-panel denominator moves: nothing here is written into ``labels.parquet``, and the file
name is deliberately **not** ``receiver.parquet``, which is the key
``analysis.secondary_analysis`` would pick up and flip from "unexecuted" to "descriptive".

Why the ordinary GRADE lane cannot do this
------------------------------------------

Three independent blocks, each sufficient on its own:

1. ``graders.grade()`` dispatches ``math500 -> math_equiv`` and ``mmlu_pro -> mmlu`` and
   raises ``ValueError("unknown grader family")`` for anything else — including
   ``graders.TRACE_FAMILY``, which ``graders.grader_module`` and ``graders.canonical_class``
   both *do* know.  The facade's own dispatch is the narrower of the two.
2. ``grade()`` resolves ``store.reference(family, item_id)`` *before* it dispatches, and the
   protected ``GoldStore`` has no trace table: the E3 reference is computed, never stored
   (``receiver.receiver_task`` writes an empty ``gold_location`` for exactly this reason).
3. Receiver rows are stamped ``family = receiver.RECEIVER_TASK_FAMILY == "math500"``, because
   WP4's frozen ``cells.interleave_families`` admits only ``contract.FAMILIES``.  So
   ``runner.grade_worker`` would ask the gold store for ``math500``/``receiver:p00`` and get
   ``GoldUnavailable`` on all 1,280 — the family column is a lie the grading lane believes.

The instrument that *is* right is ``ard/graders/trace.py``: an independent lexer,
recursive-descent parser and stepper over the **rendered** program text, with
``grade(answer, reference) -> TraceVerdict`` in the same four-status vocabulary as every
other grader (``correct``/``incorrect``/``ambiguous``/``timeout``).

Re-grading, not transcribing
----------------------------

The cells already persist a ``receiver_outcome`` event per episode, so the grades *could* be
read off disk.  They are recomputed instead, because that event was produced by
``receiver.grade_receiver`` against ``receiver.interpret`` — the generator's own interpreter,
the very code that chose the gold.  Transcribing it would certify the generator with itself.
Here the reference comes from ``trace.evaluate_program`` over the rendered source and the
verdict from ``trace.grade``; :func:`reference_values` refuses to return at all unless the two
evaluators agree on all 64 programs.  The persisted events are still read (unless
``--no-event-check``) and every disagreement is counted into the run summary, so the
recomputation is checked against the run rather than trusted over it.

Recovering the answer-logit margin
----------------------------------

Spec §11 names the answer-logit margin a primary assay outcome, and every persisted
``receiver_outcome`` records it as missing under ``top_logprobs_not_retained``.  The data is
not missing: ``client.top_logprobs_for`` asked for twenty alternatives on the receiver purpose
and the request store kept them.  Two defects, not one, put "missing" in the event:

* ``receiver.margin_from_record`` looks for ``record.logprobs_content`` or
  ``record.top_logprobs``; the field WP1 actually added is ``DecisionRecord.final_top_logprobs``
  (``tests/ard/test_integration_margin.py`` pins the plumbing but never the lookup), so the
  attribute probe always fell through to the missing branch;
* even had it found the field, ``final_top_logprobs`` is a sequence of *alternative rows*,
  while ``answer_logit_margin`` consumes OpenAI ``logprobs.content`` **entries** — mappings
  carrying the chosen ``token``, its ``logprob`` and a nested ``top_logprobs``.  The two
  shapes are not interchangeable.

:func:`report_span_entries` rebuilds the entry shape from the store record.  The store keeps
``token_logprobs`` (the sampled token's logprob) and ``top_logprobs`` (the alternatives) but
only the *ids* of the sampled tokens, so the chosen token **text** is recovered by aligning
the response's ``content`` string backwards against the alternative rows: at each position the
chosen token is the alternative whose logprob equals the sampled logprob and whose text is a
suffix of the not-yet-consumed content.  The alignment is self-checking — it either consumes
``content`` exactly or the margin is reported missing under :data:`MARGIN_SPAN_UNALIGNED` —
and it needs no tokenizer, so this module stays free of the serving stack.  The span is read
off ``physical_request_ids[-1]``: for a two-phase decision that is the continuation, which is
the request that actually generated the report (``client.decision_from_continuation``).

One further margin defect is guarded rather than inherited: see :data:`MARGIN_PREFIX_PAIR`.

Caveats carried into every row (all three verified against ard36, see :func:`main`'s summary)
---------------------------------------------------------------------------------------------

* **31 of the 64 probes are degenerate** — one answer class across all 20 draws — which is
  ``panels.panel_p7``'s number reproduced here.  Measured additionally: all 31 are degenerate
  *correct*, none degenerate wrong, so the degeneracy is a ceiling effect, not a floor one.
  ``degenerate_probe`` is on every row; any sensitivity estimate must drop those probes.
* **The arm labels cannot be checked against packet content.** ``seen_message_ids`` is empty
  and ``packet_tokens == 0`` on all 1,280 decisions, so ``arm`` is read from
  ``episodes.packet_policy`` and nothing in the compacted tables can confirm the model was
  shown the endorsement counts that label claims.  ``arm_source`` records that provenance and
  ``packet_tokens``/``n_seen_message_ids`` are copied through as the evidence.
* **k = 20 is present exactly on receiver records.** All 2,076 physical requests behind the
  1,280 decisions carry 20 alternatives per token; a sample of 600 non-receiver requests
  carries 1 (occasionally realized as 2, the server adding the sampled token), which is
  ``top_logprobs_for``'s non-receiver width.

The control arm is ``PRIVATE_TRACE`` and ``deviation_from_control`` is this draw's tri-state
correctness minus that of the *matched* control — same program, same budget, same episode
rep — so it is a within-program, within-budget contrast and is ``None`` whenever either side
is unresolvable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import receiver as R
from agents_scaling.study.types import ProtocolError

__all__ = [
    "ARMS",
    "CONTROL_ARM",
    "DECISIONS_TABLE",
    "EPISODES_TABLE",
    "MARGIN_PREFIX_PAIR",
    "MARGIN_SPAN_UNALIGNED",
    "OUTPUT_TABLE",
    "PACKET_POLICY_PREFIX",
    "arm_of",
    "build_rows",
    "episode_rep_of",
    "main",
    "recover_margin",
    "reference_values",
    "report_span_entries",
]

OUTPUT_TABLE: Final[str] = "receiver_outcomes.parquet"
DECISIONS_TABLE: Final[str] = "decisions.parquet"
EPISODES_TABLE: Final[str] = "episodes.parquet"

#: ``episodes.packet_policy`` of a receiver episode is ``"receiver:<presentation>"``.
PACKET_POLICY_PREFIX: Final[str] = f"{K.PURPOSE_RECEIVER}:"

ARMS: Final[tuple[str, ...]] = K.RECEIVER_PRESENTATIONS
#: The no-endorsement arm every social arm is differenced against (spec §11).
CONTROL_ARM: Final[str] = "PRIVATE_TRACE"

#: Two margin statuses this module records beyond ``receiver.MARGIN_STATUSES``.
#: The first: the report span could not be realigned to the stored alternatives, so no position
#: is known.  Recorded as its own string rather than folded into ``top_logprobs_not_retained``,
#: which would claim the run discarded data it in fact kept.
MARGIN_SPAN_UNALIGNED: Final[str] = "report_span_not_realigned"
#: The second guards a defect in ``receiver.first_differing_token``.  It tokenises answers per
#: character, so when one candidate's decimal rendering is a *prefix* of the other's ("1" and
#: "15") there is no differing character pair: the shorter string simply ends.  With the longer
#: answer chosen, ``answer_logit_margin`` reports ``candidates_share_an_answer`` — untrue, the
#: answers differ; with the *shorter* one chosen, ``alternative_char`` is non-``None`` while
#: ``chosen_char`` is ``None``, the offset lands one past the answer and the margin is measured
#: at whatever token follows it — a wrong number rather than a missing one.  Only ard36's
#: ``receiver:p29`` (candidates 15 and 1) is such a pair and all 20 of its draws answered the
#: longer "15", so this run wrote no wrong margin; the guard is here so a rerun cannot.  The
#: string is chosen to match the patch offered to ``receiver.py``'s owner, so the two agree if
#: that lands and the guard becomes redundant rather than contradictory.
MARGIN_PREFIX_PAIR: Final[str] = "candidate_answers_are_a_decimal_prefix_pair"

#: Two sampled logprobs are the same token only if they agree to the last bit the store kept.
_LOGPROB_TOL: Final[float] = 1e-9
#: A completion may end in the turn terminator, which is not part of ``response.content``.
#: More than one unaccounted trailing token means the alignment has lost the span.
_MAX_TRAILING_SPECIAL: Final[int] = 1


# --------------------------------------------------------------------------- keys


def arm_of(packet_policy: str | None) -> str:
    """The presentation an episode ran, from ``episodes.packet_policy``.

    The label is the *policy's*, never the packet's: see the module docstring on why no row
    of the compacted tables can corroborate it.
    """
    text = str(packet_policy or "")
    if not text.startswith(PACKET_POLICY_PREFIX):
        raise ProtocolError(f"not a receiver packet policy: {packet_policy!r}")
    arm = text[len(PACKET_POLICY_PREFIX):]
    if arm not in ARMS:
        raise ProtocolError(f"unknown receiver presentation {arm!r}; frozen set is {list(ARMS)}")
    return arm


def episode_rep_of(episode_id: str, item_id: str) -> int:
    """The completion seed of one episode, from the ``.e<n>`` suffix ``types.episode_id_for`` adds.

    Rep 0 gets no suffix there, so the absence of one *is* the rep and must not be read as a
    malformed id — the two budgets x two reps are what make the control matchable.
    """
    tail = str(episode_id).split("|")[-1]
    if not tail.startswith(str(item_id)):
        raise ProtocolError(f"episode id {episode_id!r} does not name item {item_id!r}")
    suffix = tail[len(str(item_id)):]
    if not suffix:
        return 0
    if not suffix.startswith(".e") or not suffix[2:].isdigit():
        raise ProtocolError(f"episode id {episode_id!r} has an unreadable episode-rep suffix")
    return int(suffix[2:])


# --------------------------------------------------------------------------- the reference


def reference_values() -> dict[str, int]:
    """``{program_id: reference residue}`` from the *independent* evaluator.

    ``ard.graders`` is imported here rather than at module scope for the contract §0.2 reason
    ``receiver.verify_programs`` gives: the firewall is about the module-level import closure,
    and the grader package is the one allowed to read the protected gold namespace.

    The generator's interpreter is consulted only to be disagreed with: if the two evaluators
    part on any program the assay has no reference at all, and this raises rather than
    silently preferring one of them.
    """
    from agents_scaling.ard.graders import trace

    out: dict[str, int] = {}
    for program in R.programs():
        independent = int(trace.evaluate_program(program.source).final_value)
        generator = int(R.interpret(program)[-1].value)
        if independent != generator:
            raise ProtocolError(
                f"{program.program_id}: the trace evaluator says {independent} and the "
                f"generator's interpreter says {generator}; the E3 gold is not established"
            )
        out[program.program_id] = independent
    return out


# --------------------------------------------------------------------------- the margin


def _chosen_token(row: Sequence[Mapping[str, Any]], logprob: float | None,
                  remaining: str) -> str | None:
    """The sampled token's text at one position, or ``None`` when it is not determined.

    The store keeps sampled tokens as ids and alternatives as text, so the two are joined on
    the only field they share — the logprob — and disambiguated by the content string, which
    the sampled token must be a suffix of.  Ties (two alternatives at the same logprob that
    both fit) resolve to the longest, and a failure to consume ``content`` exactly is caught
    by the caller, so a wrong tie-break cannot pass silently.
    """
    if logprob is None or not remaining:
        return None
    best: str | None = None
    for alternative in row or ():
        token = str(alternative.get("token") or "")
        value = alternative.get("logprob")
        if not token or value is None or not remaining.endswith(token):
            continue
        if abs(float(value) - float(logprob)) > _LOGPROB_TOL:
            continue
        if best is None or len(token) > len(best):
            best = token
    return best


def report_span_entries(response: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """OpenAI ``logprobs.content`` entries for the report span of one store record.

    ``None`` when the record carries no alternatives, or when the content string cannot be
    realigned to them — never a partial span, because a partial span would put the margin at
    the wrong token rather than declare itself missing.
    """
    rows = response.get("top_logprobs")
    logprobs = response.get("token_logprobs")
    content = str(response.get("content") or "")
    if not rows or not logprobs or not content or len(rows) != len(logprobs):
        return None

    index = len(rows) - 1
    remaining = content
    picked: list[tuple[int, str]] = []
    trailing = 0
    while index >= 0 and remaining:
        token = _chosen_token(rows[index], logprobs[index], remaining)
        if token is None:
            # Only the turn terminator may sit past the end of ``content``; anything else
            # means the alignment is lost and the whole span is refused.
            if picked or trailing >= _MAX_TRAILING_SPECIAL:
                return None
            trailing += 1
            index -= 1
            continue
        remaining = remaining[: len(remaining) - len(token)]
        picked.append((index, token))
        index -= 1
    if remaining or not picked:
        return None
    picked.reverse()
    return [{"token": token, "logprob": logprobs[i], "top_logprobs": list(rows[i])}
            for i, token in picked]


def _is_prefix_pair(packet: R.ReceiverPacket, produced: str | None) -> bool:
    """Would ``receiver.first_differing_token`` be asked to compare a decimal prefix pair?

    Mirrors ``margin_from_record``'s choice of alternative so the guard tests the same two
    strings the margin would have been measured between.
    """
    first, second = (str(value) for value in packet.candidate_answers)
    chosen = str(produced if produced is not None else first)
    alternative = second if first == chosen else first
    return chosen != alternative and (chosen.startswith(alternative)
                                      or alternative.startswith(chosen))


def recover_margin(store: Any, physical_request_ids: Sequence[str],
                   packet: R.ReceiverPacket, *, produced: str | None) -> R.AnswerLogitMargin:
    """The answer-logit margin of one decision, recovered from the request store.

    The report span lives in the **last** physical request: a two-phase decision's report was
    generated by the continuation, and ``client.decision_from_continuation`` takes
    ``final_text`` from there for the same reason.  The rebuilt entries are handed to
    ``receiver.margin_from_record`` through the attribute it already probes, so the choice of
    which candidate is the alternative stays in the module that owns spec §11 rather than
    being restated here.
    """
    if _is_prefix_pair(packet, produced):
        return R.AnswerLogitMargin(None, MARGIN_PREFIX_PAIR, top_k=int(K.RECEIVER_TOP_LOGPROBS))
    if not physical_request_ids:
        return R.AnswerLogitMargin(None, R.MARGIN_NO_LOGPROBS, top_k=int(K.RECEIVER_TOP_LOGPROBS))
    record = store.get(str(physical_request_ids[-1]))
    if record is None:
        return R.AnswerLogitMargin(None, R.MARGIN_NO_LOGPROBS, top_k=int(K.RECEIVER_TOP_LOGPROBS))
    entries = report_span_entries(dict(record.response or {}))
    if entries is None:
        return R.AnswerLogitMargin(None, MARGIN_SPAN_UNALIGNED, top_k=int(K.RECEIVER_TOP_LOGPROBS))
    return R.margin_from_record(SimpleNamespace(logprobs_content=entries), packet,
                                produced=produced)


# --------------------------------------------------------------------------- the table


def _read_parquet(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        raise ProtocolError(f"{path} does not exist")
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


_DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "decision_id", "episode_id", "cell_id", "cell_kind", "item_id", "phase", "checkpoint",
    "answer_raw", "answer_class", "answer_logprob", "confidence", "parse_status",
    "budget_tokens", "thinking_tokens", "final_tokens", "latency", "cap_hit",
    "boundary_injected", "final_overrun", "n_packets", "packet_tokens", "seen_message_ids",
    "physical_request_ids",
)


def _persisted_outcomes(cells_root: Path) -> dict[str, Mapping[str, Any]]:
    """``{decision_id: receiver_outcome event}`` from the cell item files.

    Read only to be disagreed with (see the module docstring); the table is built whether or
    not these exist, so a run root without ``cells/`` is not an error here.
    """
    out: dict[str, Mapping[str, Any]] = {}
    if not cells_root.is_dir():
        return out
    for cell in sorted(cells_root.glob("RECEIVER.*")):
        for path in sorted((cell / "items").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            for episode in document.get("episodes") or ():
                events = [e for e in episode.get("events") or ()
                          if str(e.get("event")) == "receiver_outcome"]
                if not events:
                    continue
                for decision in episode.get("decisions") or ():
                    out[str(decision.get("decision_id"))] = events[0]
    return out


def build_rows(tables: Path, *, requests_root: Path | None = None,
               cells_root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(rows, summary)`` — one row per receiver decision, plus what the run measured.

    Every derived quantity is recomputed from the frozen construction: the packet from
    ``receiver.build_packet`` (which is a pure function of the program and the presentation),
    the reference from the independent evaluator, the verdict from ``trace.grade``.  Nothing
    is copied from a label row, because there are none.
    """
    from agents_scaling.ard import graders as G
    from agents_scaling.ard.graders import trace

    decisions = [row for row in _read_parquet(tables / DECISIONS_TABLE, _DECISION_COLUMNS)
                 if str(row.get("phase")) == K.PURPOSE_RECEIVER]
    episodes = _read_parquet(tables / EPISODES_TABLE, ("episode_id", "packet_policy", "cohort"))
    policy = {str(e["episode_id"]): e.get("packet_policy") for e in episodes}

    references = reference_values()
    programs = {program.program_id: program for program in R.programs()}
    grader_version = G.grader_hash(G.TRACE_FAMILY)
    persisted = _persisted_outcomes(cells_root) if cells_root is not None else {}

    # Degeneracy is a property of the probe over all of its draws, so it is measured before
    # any row is written -- and measured on the same ``answer_class`` column panels.P7 uses.
    classes: dict[str, set[str]] = defaultdict(set)
    for row in decisions:
        classes[str(row["item_id"])].add(str(row.get("answer_class") or ""))

    store = None
    if requests_root is not None and requests_root.is_dir():
        from agents_scaling.study.inference.store import RequestStore

        store = RequestStore(requests_root)

    rows: list[dict[str, Any]] = []
    for row in decisions:
        item_id = str(row["item_id"])
        program = programs.get(item_id)
        if program is None:
            raise ProtocolError(f"{row['decision_id']}: {item_id} is not one of the 64 E3 programs")
        arm = arm_of(policy.get(str(row["episode_id"])))
        packet = R.build_packet(program, arm)
        reference = references[item_id]
        if int(packet.gold) != reference:
            raise ProtocolError(
                f"{item_id}: the packet was built around gold {packet.gold} but the "
                f"independent evaluator computes {reference}")

        verdict = trace.grade(row.get("answer_raw"), reference)
        # Tri-state, on the same rule ``compact._is_correct`` uses: only the two resolving
        # statuses decide, and "the grader could not resolve" is never "the model was wrong".
        if verdict.status == G.STATUS_CORRECT:
            is_correct: bool | None = True
        elif verdict.status == G.STATUS_INCORRECT:
            is_correct = False
        else:
            is_correct = None

        residue = verdict.extracted
        picked_label: str | None = None
        first, second = packet.candidate_answers
        if residue is not None:
            if residue == first:
                picked_label = R.CANDIDATE_LABELS[0]
            elif residue == second:
                picked_label = R.CANDIDATE_LABELS[1]
        if residue is None:
            outcome = R.OUTCOME_INVALID
        elif picked_label is not None:
            outcome = (R.OUTCOME_PICKED_CORRECT if residue == reference
                       else R.OUTCOME_PICKED_WRONG)
        else:
            outcome = (R.OUTCOME_PRODUCED_OTHER_CORRECT if residue == reference
                       else R.OUTCOME_PRODUCED_OTHER_WRONG)

        margin = (recover_margin(store, row.get("physical_request_ids") or (), packet,
                                 produced=None if residue is None else str(residue))
                  if store is not None
                  else R.AnswerLogitMargin(None, R.MARGIN_UNAVAILABLE,
                                           top_k=int(K.RECEIVER_TOP_LOGPROBS)))

        event = persisted.get(str(row["decision_id"]))
        rows.append({
            "decision_id": str(row["decision_id"]),
            "episode_id": str(row["episode_id"]),
            "cell_id": str(row["cell_id"]),
            "item_id": item_id,
            "program_id": program.program_id,
            "checkpoint": str(row.get("checkpoint") or ""),
            "episode_rep": episode_rep_of(str(row["episode_id"]), item_id),

            "arm": arm,
            "arm_is_control": arm == CONTROL_ARM,
            "arm_count": None if arm == CONTROL_ARM else arm.split("_")[0],
            "arm_checked": None if arm == CONTROL_ARM else arm.endswith("_CHECKED"),
            # The label's provenance, since no packet content survives to corroborate it.
            "arm_source": "packet_policy",
            "endorsement_counts": None if packet.counts is None else list(packet.counts),
            "packet_tokens": int(row.get("packet_tokens") or 0),
            "n_seen_message_ids": len(row.get("seen_message_ids") or ()),
            "n_packets": int(row.get("n_packets") or 0),
            "packet_sha256": packet.content_sha256,

            "budget_tokens": int(row.get("budget_tokens") or 0),

            "evaluator_status": verdict.status,
            "evaluator_detail": verdict.detail,
            "is_correct": is_correct,
            "answer_class": trace.canonical_class(row.get("answer_raw")),
            "answer_class_compacted": row.get("answer_class"),
            "answer_raw": row.get("answer_raw"),
            "residue": residue,
            "reference": int(reference),
            "grader_family": G.TRACE_FAMILY,
            "grader_version": grader_version,
            "outcome": outcome,
            "picked_label": picked_label,

            "provisional": packet.provisional,
            "provisional_answer": int(packet.provisional_answer),
            "provisional_correct": int(packet.provisional_answer) == int(reference),
            "order": packet.order,
            "candidate_answers": list(packet.candidate_answers),
            "correct_label": packet.correct_label,
            "depth": int(program.depth),
            "operator_mix": str(program.mix),
            "degenerate_probe": len(classes[item_id]) == 1,

            "confidence": row.get("confidence"),
            "confidence_missing": row.get("confidence") is None,
            "thinking_tokens": int(row.get("thinking_tokens") or 0),
            "final_tokens": int(row.get("final_tokens") or 0),
            "latency_s": row.get("latency"),
            "parse_status": row.get("parse_status"),
            "cap_hit": bool(row.get("cap_hit")),
            "boundary_injected": bool(row.get("boundary_injected")),
            "final_overrun": bool(row.get("final_overrun")),
            "answer_logprob": row.get("answer_logprob"),
            "n_physical_requests": len(row.get("physical_request_ids") or ()),

            "persisted_outcome": None if event is None else str(event.get("outcome")),
            "agrees_with_persisted_event":
                None if event is None else str(event.get("outcome")) == outcome,
            **margin.to_row(),
        })

    _attach_control(rows)
    return rows, _summary(rows, persisted=persisted)


def _attach_control(rows: list[dict[str, Any]]) -> None:
    """Difference each social draw against its matched ``PRIVATE_TRACE`` draw.

    Matched means the *same* program at the *same* budget and the *same* completion seed, so
    the contrast is within everything the design froze and only the presentation differs.  A
    control draw differences against itself (0 by construction) rather than being left null,
    because a reader filtering on ``deviation_from_control is not None`` should get the
    control arm's own zeros in the denominator.
    """
    control = {(r["item_id"], r["budget_tokens"], r["episode_rep"]): r
               for r in rows if r["arm"] == CONTROL_ARM}
    for row in rows:
        peer = control.get((row["item_id"], row["budget_tokens"], row["episode_rep"]))
        row["control_decision_id"] = None if peer is None else peer["decision_id"]
        row["control_is_correct"] = None if peer is None else peer["is_correct"]
        mine, theirs = row["is_correct"], None if peer is None else peer["is_correct"]
        row["deviation_from_control"] = (None if mine is None or theirs is None
                                         else int(mine) - int(theirs))


def _summary(rows: Sequence[Mapping[str, Any]], *,
             persisted: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The numbers this run actually measured, including every verified caveat."""
    graded = [r for r in rows if r["is_correct"] is not None]
    correct = [r for r in graded if r["is_correct"]]
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arm_graded = [r for r in arm_rows if r["is_correct"] is not None]
        arm_hits = sum(1 for r in arm_graded if r["is_correct"])
        deviations = [r["deviation_from_control"] for r in arm_rows
                      if r["deviation_from_control"] is not None]
        by_arm[arm] = {
            "n": len(arm_rows),
            "graded": len(arm_graded),
            "correct": arm_hits,
            "accuracy": round(arm_hits / len(arm_graded), 6) if arm_graded else None,
            "mean_deviation_from_control": (round(sum(deviations) / len(deviations), 6)
                                            if deviations else None),
            "by_budget": {str(budget): _hits(group) for budget, group
                          in sorted(_group(arm_graded, "budget_tokens").items())},
        }
    probes = _group(rows, "item_id")
    degenerate = {p: g for p, g in probes.items() if g[0]["degenerate_probe"]}
    checked = [r for r in rows if r["agrees_with_persisted_event"] is not None]
    margins = [r for r in rows if r["answer_logit_margin"] is not None]
    return {
        "rows": len(rows),
        "graded": len(graded),
        "ungraded": len(rows) - len(graded),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(graded), 6) if graded else None,
        "evaluator_status": dict(sorted(Counter(r["evaluator_status"] for r in rows).items())),
        "outcome": dict(sorted(Counter(r["outcome"] for r in rows).items())),
        "parse_status": dict(sorted(Counter(str(r["parse_status"]) for r in rows).items())),
        "by_arm": by_arm,
        "probes": {
            "n": len(probes),
            "draws_per_probe": dict(sorted(Counter(len(g) for g in probes.values()).items())),
            "degenerate": len(degenerate),
            "degenerate_all_correct": sum(1 for g in degenerate.values() if g[0]["is_correct"]),
            "degenerate_all_wrong": sum(1 for g in degenerate.values()
                                        if g[0]["is_correct"] is False),
            "non_degenerate": len(probes) - len(degenerate),
        },
        "persisted_event_check": {
            "events_found": len(persisted),
            "rows_compared": len(checked),
            "agree": sum(1 for r in checked if r["agrees_with_persisted_event"]),
            "disagree": sum(1 for r in checked if r["agrees_with_persisted_event"] is False),
        },
        "answer_logit_margin": {
            "recovered": len(margins),
            "missing": len(rows) - len(margins),
            "status": dict(sorted(Counter(str(r["answer_logit_margin_status"])
                                          for r in rows).items())),
            "mean": round(sum(r["answer_logit_margin"] for r in margins) / len(margins), 6)
            if margins else None,
            "top_logprobs_realized": dict(sorted(Counter(
                int(r["top_logprobs_realized"] or 0) for r in rows).items())),
        },
        "packet_label_checkable": {
            "rows_with_packet_tokens": sum(1 for r in rows if r["packet_tokens"]),
            "rows_with_seen_message_ids": sum(1 for r in rows if r["n_seen_message_ids"]),
            "arm_source": "packet_policy",
        },
    }


def _hits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``n``/``correct``/``accuracy`` of an already-graded group (no tri-state left in it)."""
    correct = sum(1 for row in rows if row["is_correct"])
    return {"n": len(rows), "correct": correct,
            "accuracy": round(correct / len(rows), 6) if rows else None}


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    out: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return dict(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.receiver_table",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True, help="the compacted tables directory")
    ap.add_argument("--run-root", default=None,
                    help="run root holding requests/ and cells/ (default: the tables' parent)")
    ap.add_argument("--out", default=None, help="defaults to --tables")
    ap.add_argument("--no-margin", action="store_true",
                    help="skip the request store; every margin is recorded missing")
    ap.add_argument("--no-event-check", action="store_true",
                    help="skip the persisted receiver_outcome cross-check")
    args = ap.parse_args(argv)

    tables = Path(args.tables)
    run_root = Path(args.run_root) if args.run_root else tables.parent
    out = Path(args.out) if args.out else tables
    rows, summary = build_rows(
        tables,
        requests_root=None if args.no_margin else run_root / "requests",
        cells_root=None if args.no_event_check else run_root / "cells",
    )
    out.mkdir(parents=True, exist_ok=True)

    from agents_scaling.study.aggregate import _write_parquet

    _write_parquet(out / OUTPUT_TABLE, rows)
    print(json.dumps({"table": str(out / OUTPUT_TABLE), **summary}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
