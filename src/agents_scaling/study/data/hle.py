"""HLE-Verified adapter: shard loading, the frozen eligibility predicate and the two views (WP1).

Spec §3.2 (text-only Gold and Revision, exclude Uncertain and image-dependent items,
deduplicate by original id, Gold/Revision are reporting strata), §3.3 (record every
exclusion with its reason), §4.3 (4,096-token source-task envelope under the flagship
tokenizer, applied before any outcome access), §10.6 / amendment P2 (the public view never
carries ``answer``, ``rationale`` or the nested ``json`` blob).  Architecture §1.4;
corrections P1-8 (plain SRS, strata are covariates only).

Row shape of the local shards (``data/raw/hle_shards/*.parquet``, 2,500 rows): columns
``id, Verified_Classes, category, raw_subject, question, answer, json`` where ``json`` is a
string-encoded object with ``answer_type, image, image_preview, rationale, ...``.  The
top-level ``question``/``answer`` are the revised authoritative fields (they differ from
``json.answer`` on 144 rows); the nested blob is used only for ``answer_type``, the image
predicate and the rationale.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study.types import TASK_TOKENS_CAP, Domain, PublicTask, _Record

SOURCE_PREFIX = "hle:"
GOLD = "Gold subset"
REVISION = "Revision subset"
STRATA: Mapping[str, str] = {GOLD: "Gold", REVISION: "Revision"}
ANSWER_TYPES: tuple[str, ...] = ("multipleChoice", "exactMatch")

REASON_NOT_GOLD_OR_REVISION = "not_gold_or_revision"
REASON_IMAGE_DEPENDENT = "image_dependent"
REASON_DUPLICATE = "duplicate"
REASON_ENVELOPE = "envelope"
REASON_ELIGIBLE = "eligible"
EXCLUSION_REASONS: tuple[str, ...] = (
    REASON_NOT_GOLD_OR_REVISION,
    REASON_IMAGE_DEPENDENT,
    REASON_DUPLICATE,
    REASON_ENVELOPE,
)
REQUIRED_COLUMNS: tuple[str, ...] = ("id", "Verified_Classes", "category", "raw_subject", "question", "answer", "json")
_SHARD_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ProtectedLabel(_Record):
    """Evaluator-only HLE label (§3.2, §10.6).  Never leaves ``data/protected/``."""

    source_id: str
    answer: str  # revised authoritative top-level answer
    answer_type: str  # "multipleChoice" | "exactMatch"
    rationale: str


class HleRowError(ValueError):
    """A shard row does not have the frozen shape (fail closed; never guess a field)."""


def source_id_for(hle_id: str) -> str:
    return f"{SOURCE_PREFIX}{hle_id}"


def _shard_sort_key(path: Path) -> tuple[int, str]:
    match = _SHARD_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def load_hle_rows(shards_dir: str | Path) -> list[dict[str, Any]]:
    """Read every ``*.parquet`` shard in numeric shard order and return plain row dicts.

    The order is deterministic (shard number, then row order) so that the duplicate rule
    ("first occurrence kept") is reproducible.  Raises if the directory has no shards or a
    shard lacks a required column.
    """
    directory = Path(shards_dir)
    files = sorted(directory.glob("*.parquet"), key=_shard_sort_key)
    if not files:
        raise FileNotFoundError(f"no parquet shards under {directory}")
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
        if missing:
            raise HleRowError(f"{path} lacks columns {missing}")
        rows.extend(table.to_pylist())
    return rows


def parse_json_blob(row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the string-encoded nested record (strict: must be a JSON object)."""
    blob = row.get("json")
    if isinstance(blob, Mapping):
        return dict(blob)
    if not isinstance(blob, str):
        raise HleRowError(f"row {row.get('id')!r}: json column must be a str, got {type(blob).__name__}")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise HleRowError(f"row {row.get('id')!r}: json column is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise HleRowError(f"row {row.get('id')!r}: json column is not an object")
    return dict(data)


def is_image_dependent(blob: Mapping[str, Any]) -> bool:
    """§3.2 image predicate: ``json.image`` non-empty OR ``json.image_preview`` truthy.

    The shards encode a missing preview as ``None``; the string ``"None"`` is treated as
    missing too because the nested fields are string-encoded in places.
    """
    image = blob.get("image")
    if image not in (None, "", "None"):
        return True
    preview = blob.get("image_preview")
    return preview not in (None, "", "None", False)


def hle_task_text(row: Mapping[str, Any]) -> str:
    text = row.get("question")
    if not isinstance(text, str) or not text.strip():
        raise HleRowError(f"row {row.get('id')!r}: empty question")
    return text


def hle_answer_format(blob: Mapping[str, Any], hle_id: str) -> str:
    answer_type = blob.get("answer_type")
    if answer_type not in ANSWER_TYPES:
        raise HleRowError(f"row {hle_id!r}: answer_type {answer_type!r} not in {ANSWER_TYPES}")
    return str(answer_type)


def hle_eligible(
    row: Mapping[str, Any],
    *,
    task_tokens: int,
    seen_ids: Collection[str] = (),
    cap: int = TASK_TOKENS_CAP,
) -> tuple[bool, str]:
    """Frozen source-only eligibility predicate (§3.2, §4.3), evaluated in this fixed order:

    1. ``not_gold_or_revision`` — ``Verified_Classes`` not in {Gold subset, Revision subset}
       (Uncertain excluded);
    2. ``image_dependent`` — :func:`is_image_dependent` on the nested blob;
    3. ``duplicate`` — ``id`` already in ``seen_ids`` (first occurrence kept);
    4. ``envelope`` — ``task_tokens`` (flagship tokenizer, raw question) above ``cap``.

    Returns ``(True, "eligible")`` or ``(False, <reason>)``.  ``task_tokens`` is required so
    the envelope rule can never be skipped silently.
    """
    if not isinstance(task_tokens, int) or isinstance(task_tokens, bool) or task_tokens < 0:
        raise TypeError("task_tokens must be a non-negative int")
    if row.get("Verified_Classes") not in STRATA:
        return False, REASON_NOT_GOLD_OR_REVISION
    if is_image_dependent(parse_json_blob(row)):
        return False, REASON_IMAGE_DEPENDENT
    if str(row["id"]) in seen_ids:
        return False, REASON_DUPLICATE
    if task_tokens > cap:
        return False, REASON_ENVELOPE
    return True, REASON_ELIGIBLE


def hle_public_task(row: Mapping[str, Any], task_tokens: int) -> PublicTask:
    """Task-public view of an eligible row (split/rank are assigned later by ``splits``)."""
    hle_id = str(row["id"])
    blob = parse_json_blob(row)
    return PublicTask(
        source_id=source_id_for(hle_id),
        domain=Domain.HLE,
        split="unassigned",
        task_text=hle_task_text(row),
        answer_format=hle_answer_format(blob, hle_id),
        stratum=STRATA[row["Verified_Classes"]],
        rank=-1,
        task_tokens=int(task_tokens),
        entry_point=None,
        category=None if row.get("category") is None else str(row["category"]),
    )


def hle_protected_label(row: Mapping[str, Any]) -> ProtectedLabel:
    """Protected view: revised top-level ``answer`` plus the nested ``rationale``."""
    hle_id = str(row["id"])
    blob = parse_json_blob(row)
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise HleRowError(f"row {hle_id!r}: empty answer")
    rationale = blob.get("rationale")
    return ProtectedLabel(
        source_id=source_id_for(hle_id),
        answer=answer,
        answer_type=hle_answer_format(blob, hle_id),
        rationale="" if rationale is None else str(rationale),
    )


@dataclass(frozen=True)
class HleBuild:
    """Result of :func:`build_hle`: eligible tasks, their labels and every exclusion."""

    tasks: list[PublicTask]
    labels: dict[str, ProtectedLabel]
    exclusions: list[dict[str, Any]]


def build_hle(rows: Sequence[Mapping[str, Any]], tokenizer, *, cap: int = TASK_TOKENS_CAP) -> HleBuild:
    """Apply the eligibility predicate to every row in order and build both views.

    ``tokenizer`` must be the flagship tokenizer (§4.3); token counts are computed for every
    row so the exclusion record of an over-envelope row carries its size.
    """
    from agents_scaling.study.inference.tokens import count_tokens

    tasks: list[PublicTask] = []
    labels: dict[str, ProtectedLabel] = {}
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        hle_id = str(row["id"])
        question = row.get("question")
        task_tokens = count_tokens(question, tokenizer) if isinstance(question, str) else 0
        ok, reason = hle_eligible(row, task_tokens=task_tokens, seen_ids=seen, cap=cap)
        if not ok:
            exclusions.append(
                {
                    "source_id": source_id_for(hle_id),
                    "domain": Domain.HLE.value,
                    "reason": reason,
                    "verified_class": row.get("Verified_Classes"),
                    "task_tokens": task_tokens,
                }
            )
            continue
        seen.add(hle_id)
        task = hle_public_task(row, task_tokens)
        tasks.append(task)
        labels[task.source_id] = hle_protected_label(row)
    return HleBuild(tasks=tasks, labels=labels, exclusions=exclusions)


__all__ = [name for name in globals() if not name.startswith("_")]
