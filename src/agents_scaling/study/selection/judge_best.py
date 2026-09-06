"""JUDGE_BEST: strict pointwise score parsing and max-score selection (WP3).

Spec §4.4: "score each candidate separately against the task/public interface using a
1,024-token cap; return a quality score in [0,1] and fixed rubric fields; choose the
maximum finite score with a blind tie rule ... Invalid scores use a frozen low score and a
failure flag.  An all-invalid bank still produces failure."  §5.3: the companion selector
on the same complete-candidate bank; "Its ranking result is never called majority voting."
05_vllm_response_shape.md: guided JSON is not enforced, so the record is parsed with the
same one-fence-strip + strict JSON rule as candidates (amendment E2 "not used").
Architecture §1.9 (the JUDGE_BEST *cell* that issues the calls is WP5).

``metrics_reference.judge_best`` performs the selection; this module builds its public
inputs, substitutes ``JUDGE_BEST_FROZEN_LOW_SCORE`` for failed records (flagging them) and
derives ``tie_seed = HMAC-SHA256(study_seed, b"JUDGE_BEST_TIE")``.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from typing import Any

from agents_scaling.study import metrics_reference
from agents_scaling.study.parse.candidate import (
    FINISH_LENGTH,
    FINISH_STOP,
    SchemaError,
    StrictJSONError,
    check_exact_keys,
    check_str,
    check_unit_number,
    load_strict_json,
    strip_one_fence,
)
from agents_scaling.study.selection.vote import keyed_pool
from agents_scaling.study.types import (
    JUDGE_BEST_FROZEN_LOW_SCORE,
    JUDGE_BEST_TIE_NAMESPACE,
    SELECTOR_JUDGE_BEST,
    CandidateRecord,
    ProtocolError,
    SelectionRecord,
)

#: ``prompts/render.py::JUDGE_BEST_SCHEMA_LINE`` field set.
JUDGE_KEYS: tuple[str, ...] = ("quality_score", "requirement_coverage", "reasoning_support", "unresolved_risks")
#: Rubric strings are "brief"; this bound only guards against runaway text (the 1,024-token
#: output cap already bounds them physically).
RUBRIC_MAX = 8192
JUDGE_FAILURE_CODES: tuple[str, ...] = (
    "TRUNCATED",
    "EMPTY",
    "NOT_JSON",
    "DUPLICATE_KEY",
    "NONFINITE",
    "TRAILING_TEXT",
    "SCHEMA",
)


def judge_best_tie_seed(study_seed: bytes) -> bytes:
    """``HMAC-SHA256(study_seed, b"JUDGE_BEST_TIE")`` — blind tie seed of JUDGE_BEST (§4.4)."""
    if not isinstance(study_seed, (bytes, bytearray)) or len(study_seed) == 0:
        raise ValueError("study_seed must be non-empty bytes")
    return hmac.new(bytes(study_seed), JUDGE_BEST_TIE_NAMESPACE, hashlib.sha256).digest()


def parse_judge_best(content: str | None, finish_reason: str = FINISH_STOP) -> tuple[float | None, dict[str, Any]]:
    """Parse one judge record → ``(score, fields)``.

    Valid: ``{"quality_score": finite number in [0,1], "requirement_coverage": str,
    "reasoning_support": str, "unresolved_risks": str}`` exactly (one fence stripped,
    strict JSON, no extra keys) → ``(score, {the four fields})``.  Invalid →
    ``(None, {"failure_code": <code>, "detail": <why>})`` with ``failure_code`` ∈
    ``JUDGE_FAILURE_CODES``; the caller substitutes the frozen low score and flags the
    failure (:func:`judge_best_select` does exactly that).
    """
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("content must be a str or None")
    if finish_reason not in (FINISH_STOP, FINISH_LENGTH):
        raise ValueError(f"finish_reason {finish_reason!r} is not a parser outcome (stop|length)")
    if finish_reason == FINISH_LENGTH:
        return None, {"failure_code": "TRUNCATED", "detail": "finish_reason=length"}
    try:
        value = load_strict_json(strip_one_fence(content))
        body = check_exact_keys(value, "judge record", JUDGE_KEYS)
        score = check_unit_number(body["quality_score"], "quality_score")
        fields = {
            "quality_score": score,
            "requirement_coverage": check_str(body["requirement_coverage"], "requirement_coverage", RUBRIC_MAX),
            "reasoning_support": check_str(body["reasoning_support"], "reasoning_support", RUBRIC_MAX),
            "unresolved_risks": check_str(body["unresolved_risks"], "unresolved_risks", RUBRIC_MAX),
        }
    except SchemaError as exc:
        return None, {"failure_code": "SCHEMA", "detail": exc.detail}
    except StrictJSONError as exc:
        return None, {"failure_code": exc.code, "detail": exc.detail}
    return score, fields


def judge_best_select(
    scores: Mapping[str, float | None],
    pool: Sequence[CandidateRecord],
    source_id: str,
    study_seed: bytes,
    answer_format: str,
    *,
    pool_kind: str = "latest_slots",
    prefix_k: int | None = None,
) -> SelectionRecord:
    """Pointwise JUDGE_BEST over a sealed pool given already-acquired public scores (§4.4).

    ``scores`` maps *every valid* candidate id to its parsed ``quality_score`` or ``None``
    (an invalid judge record).  ``None`` becomes ``JUDGE_BEST_FROZEN_LOW_SCORE`` and is
    flagged in ``score_failures``; a missing entry for a valid candidate or an entry for an
    unknown/invalid candidate is a harness defect (``ProtocolError``) — a score never
    silently appears or disappears.  Ties at the maximum are broken by the blind
    ``JUDGE_BEST_TIE`` order; an all-invalid pool yields ``no_valid_candidate``.  Vote keys
    are recorded for reporting only (they do not enter this selection).
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if len(pool) == 0:
        raise ValueError("a planned opportunity pool must not be empty")
    keyed = keyed_pool(pool, answer_format)
    valid_ids = [r.candidate_id for r in keyed if r.valid and r.candidate is not None]
    if len(set(r.candidate_id for r in keyed)) != len(keyed):
        raise ValueError("duplicate candidate IDs would double-count an opportunity")
    missing = [cid for cid in valid_ids if cid not in scores]
    extra = [cid for cid in scores if cid not in valid_ids]
    if missing or extra:
        raise ProtocolError(f"judge scores must cover exactly the valid candidates; missing={missing} extra={extra}")
    public_scores: dict[str, float] = {}
    failures: dict[str, bool] = {}
    for cid in valid_ids:
        raw = scores[cid]
        if raw is None:
            public_scores[cid] = JUDGE_BEST_FROZEN_LOW_SCORE
            failures[cid] = True
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ProtocolError(f"score of {cid} must be a number or None, got {type(raw).__name__}")
            public_scores[cid] = float(raw)
            failures[cid] = False
    # The reference helper requires a nonempty key for every valid candidate; keys do not
    # enter this selection, so an unkeyed valid candidate (empty answer) gets an opaque one.
    public = [
        metrics_reference.PublicCandidate(
            r.candidate_id,
            (r.vote_key or f"unkeyed:{r.candidate_id}") if r.candidate_id in valid_ids else None,
            r.candidate_id in valid_ids,
        )
        for r in keyed
    ]
    selection = metrics_reference.judge_best(
        public, public_scores, source_id=source_id, tie_seed=judge_best_tie_seed(study_seed)
    )
    return SelectionRecord(
        selector_id=SELECTOR_JUDGE_BEST,
        pool_kind=pool_kind,
        pool_candidate_ids=tuple(r.candidate_id for r in keyed),
        selected_candidate_id=selection.candidate_id,
        planned_count=selection.planned_count,
        valid_count=selection.valid_count,
        no_valid_candidate=selection.candidate_id is None,
        answer_format=answer_format,
        grouping_mode=None,
        grouping_mode_counts={},
        winning_count=None,
        tied_classes=None,
        all_singleton=False,
        vote_keys={r.candidate_id: r.vote_key for r in keyed},
        scores=public_scores,
        score_failures=failures,
        prefix_k=prefix_k,
    )


__all__ = [
    "JUDGE_FAILURE_CODES",
    "JUDGE_KEYS",
    "judge_best_select",
    "judge_best_tie_seed",
    "parse_judge_best",
]
