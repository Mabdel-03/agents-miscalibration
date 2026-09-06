"""HLE correctness: exact MC letter scoring and the frozen cais/hle judge (WP5, evaluator only).

Spec §3.3: "Use exact multiple-choice scoring where sufficient.  For HLE short answers, use a
frozen isolated judge against the final authoritative answer ... Never auto-score an
ambiguous judge response as correct."  §3.6 (judge requests carry their own semantic seed:
``SeedKey(actor_slot=0, purpose="hle_judge", step_slot=0, namespace="judge")``), §4.4/§6.4
(selector/judge cap 1,024 tokens, thinking off, temperature 0 — ``JUDGE_DECODING``), §10.6
(protected labels only in the evaluator identity).  Architecture §1.12; corrections P1-10
(amendment E1: MC items get BOTH exact letter scoring and the judge so the judge's error
rates can be audited), P2-2 (non-letter MC answers fall back to the exact-answer key),
§4 item 11 (dedupe by ``sha256(final_answer)`` per item *within the sealed pool only*;
outputs under ``<run_root>/eval/hle/<source_id>.json``), 05_vllm_response_shape.md
(guided JSON is not enforced → strict parse: one fence strip, JSON ``{"correct": ...}``
or a ``correct:`` line; anything else is ``ambiguous`` → scored incorrect and flagged).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.data.protected import load_protected_hle  # evaluator identity only
from agents_scaling.study.parse.candidate import load_strict_json, strip_one_fence
from agents_scaling.study.prompts import render as R
from agents_scaling.study.selection import normalize
from agents_scaling.study.types import (
    JUDGE_DECODING,
    NS_JUDGE,
    PURPOSE_HLE_JUDGE,
    CandidateRecord,
    Checkpoint,
    Decoding,
    ProtocolError,
    PublicTask,
    RequestSpec,
    SeedKey,
)

EVAL_HLE_SUBDIR = ("eval", "hle")
HLE_EVAL_SCHEMA_VERSION = 1
JUDGE_YES = "yes"
JUDGE_NO = "no"
JUDGE_AMBIGUOUS = "ambiguous"
JUDGE_LABELS: tuple[str, ...] = (JUDGE_YES, JUDGE_NO, JUDGE_AMBIGUOUS)
ROLE_HLE_JUDGE = "hle_judge"
EMPTY_ANSWER_NOTE = "empty_final_answer"

#: ``correct: yes`` / ``correct: no`` line of the official judge format (case-insensitive,
#: optional markdown emphasis, one trailing period tolerated).
_CORRECT_LINE = re.compile(r"^[ \t*_]*correct[ \t*_]*:[ \t*_]*(yes|no)\b[ \t*_.]*$", re.IGNORECASE | re.MULTILINE)
_YES_NO = re.compile(r"^\s*(yes|no)\s*\.?\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- MC scoring


def gold_mc_letter(gold: str, hle_mc_key: Callable[[str], str | None] = normalize.hle_mc_key) -> str | None:
    """The authoritative choice letter of an MC gold answer (``None`` for the non-letter golds, P2-2)."""
    if not isinstance(gold, str):
        raise TypeError("gold must be str")
    return hle_mc_key(gold)


def score_mc(
    final_answer: str,
    gold_letter: str,
    hle_mc_key: Callable[[str], str | None] = normalize.hle_mc_key,
) -> bool:
    """Exact multiple-choice scoring (§3.3) under the frozen public MC key.

    ``gold_letter`` is the protected answer of a ``multipleChoice`` item.  When both sides
    reduce to a single letter the letters must match; when the gold is one of the few
    non-letter MC answers (P2-2) both sides are compared under :func:`normalize.hle_exact_key`.
    A candidate whose answer yields no letter while the gold does is incorrect.
    """
    if not isinstance(final_answer, str) or not isinstance(gold_letter, str):
        raise TypeError("final_answer and gold_letter must be str")
    gold = hle_mc_key(gold_letter)
    if gold is None:
        return normalize.hle_exact_key(final_answer) == normalize.hle_exact_key(gold_letter) and gold_letter.strip() != ""
    letter = hle_mc_key(final_answer)
    return letter is not None and letter == gold


# --------------------------------------------------------------------------- judge request


def hle_judge_seed_key(task: PublicTask, checkpoint: Checkpoint) -> SeedKey:
    """``SeedKey(source_id, split, judge model_cell, 0, 0, "hle_judge", 0, "judge")`` (§3.6)."""
    return SeedKey(
        source_id=task.source_id,
        split=task.split,
        model_cell=checkpoint.model_cell,
        episode_rep=0,
        actor_slot=0,
        purpose=PURPOSE_HLE_JUDGE,
        step_slot=0,
        namespace=NS_JUDGE,
    )


def hle_judge_request(
    task: PublicTask,
    final_answer: str,
    gold: str,
    checkpoint: Checkpoint,
    guided: bool = False,
    *,
    study_id: str,
    study_seed_hex: str,
) -> RequestSpec:
    """The frozen judge request for one distinct ``final_answer`` of ``task`` (§3.3, §3.6).

    ``JUDGE_DECODING`` (temperature 0, thinking off, 1,024-token cap) on the judge
    checkpoint; ``guided`` sets the identity flag only (never enforced by this stack).
    The prompt is the official cais/hle ``JUDGE_PROMPT`` rendered by WP1.  Because the
    prompt is a pure function of (question, final_answer, gold) and the seed key is fixed
    per item, identical answers of one item share one ``request_id`` (store-level dedupe).
    """
    if not isinstance(task, PublicTask):
        raise TypeError("task must be a PublicTask")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("final_answer must be a non-empty str (empty answers are scored without a judge)")
    if not isinstance(gold, str) or not gold.strip():
        raise ProtocolError(f"{task.source_id}: protected answer is empty")
    decoding: Decoding = dataclasses.replace(JUDGE_DECODING, guided_json=bool(guided))
    rendered = R.render_hle_judge(task.task_text, final_answer, gold)
    return RequestSpec(
        messages=tuple(dict(m) for m in rendered.messages),
        decoding=decoding,
        checkpoint=checkpoint,
        seed_key=hle_judge_seed_key(task, checkpoint),
        role=ROLE_HLE_JUDGE,
        study_id=study_id,
        study_seed_hex=study_seed_hex,
    )


# --------------------------------------------------------------------------- judge parse


def parse_hle_judge(content: str | None, finish_reason: str = "stop") -> str:
    """``"yes"`` | ``"no"`` | ``"ambiguous"`` from one judge completion (§3.3).

    Accepted forms, in order: (1) strict JSON (after one fence strip) whose ``correct`` key
    is exactly ``"yes"``/``"no"`` (case-insensitive); (2) exactly one ``correct: yes|no``
    line in the official plain format.  A truncated completion (``finish_reason=length``),
    no match, two conflicting matches, or anything else → ``ambiguous`` (scored incorrect
    and flagged for audit — never auto-scored correct).
    """
    if content is None:
        return JUDGE_AMBIGUOUS
    if not isinstance(content, str):
        raise TypeError("content must be str or None")
    if finish_reason == "length":
        return JUDGE_AMBIGUOUS
    text = strip_one_fence(content)
    if not text:
        return JUDGE_AMBIGUOUS
    try:
        value = load_strict_json(text)
    except Exception:  # not JSON at all → plain format below
        value = None
    if isinstance(value, dict):
        raw = value.get("correct")
        if isinstance(raw, str):
            match = _YES_NO.match(raw)
            return match.group(1).lower() if match else JUDGE_AMBIGUOUS
        return JUDGE_AMBIGUOUS
    labels = {m.group(1).lower() for m in _CORRECT_LINE.finditer(text)}
    if len(labels) == 1:
        return labels.pop()
    return JUDGE_AMBIGUOUS


# --------------------------------------------------------------------------- per-item judge


def hle_eval_path(run_root: str | os.PathLike, source_id: str) -> Path:
    """``<run_root>/eval/hle/<source_id>.json`` (``:`` in the id is kept; it is a valid POSIX name)."""
    return Path(run_root).joinpath(*EVAL_HLE_SUBDIR) / f"{source_id}.json"


def load_hle_eval(run_root: str | os.PathLike, source_id: str) -> dict[str, Any] | None:
    """The existing judge record of an item or ``None`` (a corrupt file is a ``ProtocolError``)."""
    path = hle_eval_path(run_root, source_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("source_id") != source_id or data.get("schema_version") != HLE_EVAL_SCHEMA_VERSION:
        raise ProtocolError(f"{path} is not an HLE eval record for {source_id}")
    return data


def answer_sha256(final_answer: str) -> str:
    return identity.sha256_hex(final_answer)


def judge_item(
    task: PublicTask,
    gold: str,
    candidates: Sequence[CandidateRecord],
    *,
    checkpoint: Checkpoint,
    study_id: str,
    study_seed_hex: str,
    store: Any,
    client: Any,
    cell_id: str,
    seal: str,
    executor: Executor | None = None,
    existing: Mapping[str, Any] | None = None,
    guided: bool = False,
    clock: Callable[[], float] = time.time,
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Judge every distinct valid ``final_answer`` among ``candidates`` (the item's sealed pool).

    Rules (§3.3, E1, §4 item 11): candidates are deduplicated by ``sha256(final_answer)``;
    invalid candidates are never judged (incorrect by the failure rule); MC items are scored
    by the exact letter rule AND judged (audit substitute); exact-answer items are correct
    iff the judge says ``yes``; ``ambiguous`` → incorrect + ``flagged``.  Hashes already
    present in ``existing`` (an earlier seal's judgements) are kept verbatim — the judge is
    deterministic and a completed judgement is immutable; only their candidate lists grow.
    Returns the full record to write to :func:`hle_eval_path`.
    """
    if task.domain.value != "hle":
        raise ProtocolError(f"{task.source_id} is not an HLE item")
    if not isinstance(gold, str) or not gold.strip():
        raise ProtocolError(f"{task.source_id}: protected answer is empty")
    is_mc = task.answer_format == "multipleChoice"
    started_at = clock()
    judgements: dict[str, dict[str, Any]] = {}
    if existing is not None:
        judgements = {k: dict(v) for k, v in existing.get("judgements", {}).items()}
    groups: dict[str, list[CandidateRecord]] = {}
    invalid_ids: list[str] = []
    for cand in candidates:
        if not cand.valid or cand.candidate is None:
            invalid_ids.append(cand.candidate_id)
            continue
        groups.setdefault(answer_sha256(cand.candidate.final_answer), []).append(cand)

    def judge_one(sha: str, cand: CandidateRecord) -> dict[str, Any]:
        answer = cand.candidate.final_answer  # type: ignore[union-attr]
        entry: dict[str, Any] = {
            "final_answer_sha256": sha,
            "started_at": clock(),
            "request_id": None,
            "judge": None,
            "note": None,
            "letter_correct": score_mc(answer, gold) if is_mc else None,
        }
        if not answer.strip():
            entry.update(judge=JUDGE_NO, note=EMPTY_ANSWER_NOTE)
        else:
            spec = hle_judge_request(task, answer, gold, checkpoint, guided, study_id=study_id, study_seed_hex=study_seed_hex)
            record, aliased = store.get_or_generate(spec, client, cell_id, on_event=on_event)
            entry["request_id"] = record.request_id
            entry["aliased"] = bool(aliased)
            entry["judge"] = parse_hle_judge(record.response.get("content"), record.response.get("finish_reason", "stop"))
            entry["finish_reason"] = record.response.get("finish_reason")
        entry["flagged"] = entry["judge"] == JUDGE_AMBIGUOUS
        entry["correct"] = bool(entry["letter_correct"]) if is_mc else entry["judge"] == JUDGE_YES
        entry["completed_at"] = clock()
        return entry

    todo = [(sha, cands[0]) for sha, cands in groups.items() if sha not in judgements]
    if executor is not None and len(todo) > 1:
        futures: list[tuple[str, Future[Any]]] = [(sha, executor.submit(judge_one, sha, cand)) for sha, cand in todo]
        for sha, fut in futures:
            judgements[sha] = fut.result()
    else:
        for sha, cand in todo:
            judgements[sha] = judge_one(sha, cand)
    for sha, cands in groups.items():
        entry = judgements[sha]
        ids = sorted(set(entry.get("candidate_ids", [])) | {c.candidate_id for c in cands})
        entry["candidate_ids"] = ids
        seals = sorted(set(entry.get("seals", [])) | {seal})
        entry["seals"] = seals
        # P0-3 evidence: when this seal's evaluation of the item started (a reused verdict
        # still records that the evaluator ran for this seal after its selections sealed).
        entry.setdefault("started_at_by_seal", {})[seal] = started_at
    prior_invalid = set(existing.get("invalid_candidate_ids", [])) if existing else set()
    return {
        "schema_version": HLE_EVAL_SCHEMA_VERSION,
        "source_id": task.source_id,
        "answer_format": task.answer_format,
        "judge_checkpoint": checkpoint.to_dict(),
        "judge_decoding": dataclasses.replace(JUDGE_DECODING, guided_json=bool(guided)).as_strings(),
        "seals": sorted(set((existing or {}).get("seals", [])) | {seal}),
        "started_at": min(started_at, float((existing or {}).get("started_at", started_at))),
        "last_started_at": started_at,
        "completed_at": clock(),
        "judgements": judgements,
        "invalid_candidate_ids": sorted(prior_invalid | set(invalid_ids)),
        "n_distinct_answers": len(judgements),
        "n_ambiguous": sum(1 for j in judgements.values() if j["judge"] == JUDGE_AMBIGUOUS),
    }


def candidate_correctness(record: Mapping[str, Any], candidates: Sequence[CandidateRecord]) -> dict[str, bool]:
    """``{candidate_id: correct}`` for every candidate of an item from its judge record.

    Invalid candidates are ``False`` (failure rule, §3.3); a valid candidate whose answer
    hash is absent from the record is a ``ProtocolError`` (an unsealed or unjudged pool).
    """
    judgements = record.get("judgements", {})
    out: dict[str, bool] = {}
    for cand in candidates:
        if not cand.valid or cand.candidate is None:
            out[cand.candidate_id] = False
            continue
        entry = judgements.get(answer_sha256(cand.candidate.final_answer))
        if entry is None:
            raise ProtocolError(f"{record.get('source_id')}: candidate {cand.candidate_id[:12]} has no judgement (pool judged?)")
        out[cand.candidate_id] = bool(entry["correct"])
    return out


def load_labels(run_root: str | os.PathLike) -> dict[str, str]:
    """``{source_id: authoritative answer}`` for HLE (protected; evaluator identity only)."""
    return {sid: label.answer for sid, label in load_protected_hle(run_root).items()}


__all__ = [
    "EMPTY_ANSWER_NOTE",
    "EVAL_HLE_SUBDIR",
    "HLE_EVAL_SCHEMA_VERSION",
    "JUDGE_AMBIGUOUS",
    "JUDGE_LABELS",
    "JUDGE_NO",
    "JUDGE_YES",
    "ROLE_HLE_JUDGE",
    "answer_sha256",
    "candidate_correctness",
    "gold_mc_letter",
    "hle_eval_path",
    "hle_judge_request",
    "hle_judge_seed_key",
    "judge_item",
    "load_hle_eval",
    "load_labels",
    "parse_hle_judge",
    "score_mc",
]
