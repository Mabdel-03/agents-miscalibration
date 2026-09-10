"""Build the frozen ARD36 item table from the pinned offline datasets (WP3).

**Evaluator lane only.**  This module is the *sole* importer of
:mod:`agents_scaling.benchmarks.loaders` (and therefore of ``datasets``) in the ``ard``
package — contract §0 rule 2.  Runtime code reads the public parquet through
:mod:`agents_scaling.ard.items`; nothing else may import this module.

Spec §5.1 (192 MATH-500 + 192 MMLU-Pro, 48/24/120 per family assigned **before** any
generation, stratified over the available subject/level metadata, one frozen option
permutation per MMLU item, no option key exposed, the natural panel never filtered on
measured correctness), §18 (``items.parquet`` field list; "gold/reference fields live in
the evaluator namespace and must not be loaded by serving/controller workers").
Plan §5.1: a **new** frozen split salt applied through
:func:`agents_scaling.study.data.splits.rank_key` *within strata*; proportional with at
least one item per stratum; MMLU-Pro restricted to the ten-option rows with duplicate
question strings removed; ``cot_content`` never exported; the unused rows form a named
reserve.

``benchmarks.loaders.load_math``/``load_mmlu_pro`` normalise into ``Question`` and **drop
``subject``/``level``/``category``**, which spec §5.1 requires for stratification, so this
module reads the raw pinned rows through ``loaders._load_hf`` (which enforces the
offline-cache and revision checks) instead.

Outputs under ``<out_dir>/data``::

    public/items.parquet        the public table (schema = ``items.PUBLIC_COLUMNS``)
    public/items_summary.json   counts, strata, salt/seed fingerprints, grader hashes
    public/items_reserve.json   the unused eligible source ids, in frozen rank order
    protected/math_gold.jsonl   0600: answer + solution           (dir 0700)
    protected/mmlu_gold.jsonl   0600: source answer_index, permuted key, permutation
    public/ITEMS_SHA256.json    sha256 of every written file
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents_scaling.ard import graders
from agents_scaling.ard.contract import (
    MC_LETTERS,
    MMLU_REQUIRED_OPTIONS,
    SPLIT_COUNTS_PER_FAMILY,
    SPLITS,
    STUDY_ID,
    Family,
)
from agents_scaling.ard.items import (
    ITEMS_SUMMARY_FILE,
    MATH_GOLD_FILE,
    MMLU_GOLD_FILE,
    PROTECTED_DIR,
    PROTECTED_DIR_MODE,
    PROTECTED_FILE_MODE,
    PUBLIC_COLUMNS,
    RESERVE_FILE,
    ArdTask,
    assert_public_row,
    items_path,
    protected_dir,
    public_dir,
)
from agents_scaling.study import identity
from agents_scaling.study.data.splits import rank_key

HASHES_FILE = "ITEMS_SHA256.json"

#: The NEW frozen ARD36 split salt (plan §5.1: "a freshly minted ARD36 ``study_seed`` and
#: ``split_salt`` (never study_v4's)").  Only its sha256 is written to the summary.
DEFAULT_SPLIT_SALT: bytes = b"ard36/v1/split-salt/2026-09-10"
#: The default study seed for identity derivations owned by this module (the option
#: permutations).  Callers pass the run's real seed; this default keeps the builder
#: reproducible on its own.
DEFAULT_STUDY_SEED: bytes = hashlib.sha256(b"ard36/v1/study-seed/2026-09-10").digest()

#: HMAC namespace of the frozen per-item MMLU option permutation.  Variant 0 is the one
#: every protocol and agent sees; variant 1 is the spec §14.4 answer-identity control.
NS_OPTION_PERMUTATION: str = "ARD_OPTION_PERM"
OPTION_PERMUTATION_VARIANTS: tuple[int, ...] = (0, 1)

ANSWER_FORMAT_MATH = "exactMatch"
ANSWER_FORMAT_MMLU = "multipleChoice"

#: Total natural items per family (spec §5.1: 48 + 24 + 120).
ITEMS_PER_FAMILY: int = sum(SPLIT_COUNTS_PER_FAMILY[name] for name in SPLITS)


class BuildError(ValueError):
    """The pinned rows cannot produce the frozen panel."""


# --------------------------------------------------------------------------- raw rows


@dataclass(frozen=True)
class RawSources:
    """Raw pinned rows plus the revisions they came from."""

    math_rows: tuple[Mapping[str, Any], ...]
    mmlu_rows: tuple[Mapping[str, Any], ...]
    math_revision: str
    mmlu_revision: str


def read_raw_sources() -> RawSources:
    """Read the pinned MATH-500 and MMLU-Pro rows from the shared offline HF cache.

    Goes through ``loaders._load_hf`` so the pinned-revision and shared-cache validation
    of ``benchmarks/loaders.py`` applies, then keeps the *raw* fields (``subject``,
    ``level``, ``category``) that the normalising loaders discard.
    """
    from agents_scaling.benchmarks.loaders import BENCHMARK_SOURCE_SPECS, _load_hf

    math_spec = BENCHMARK_SOURCE_SPECS["math"]
    mmlu_spec = BENCHMARK_SOURCE_SPECS["mmlu_pro"]
    math_rows = _load_hf(math_spec.repo_id, revision=math_spec.revision)[math_spec.split].to_list()
    mmlu_rows = _load_hf(mmlu_spec.repo_id, revision=mmlu_spec.revision)[mmlu_spec.split].to_list()
    if len(math_rows) != math_spec.question_count:
        raise BuildError(f"MATH-500 returned {len(math_rows)} rows, expected {math_spec.question_count}")
    if len(mmlu_rows) != mmlu_spec.question_count:
        raise BuildError(f"MMLU-Pro returned {len(mmlu_rows)} rows, expected {mmlu_spec.question_count}")
    return RawSources(tuple(math_rows), tuple(mmlu_rows), math_spec.revision, mmlu_spec.revision)


# --------------------------------------------------------------------------- eligibility


@dataclass(frozen=True)
class Candidate:
    """One eligible source row, before sampling."""

    source_id: str
    stratum: str
    subject: str
    difficulty: str
    task_text: str
    options: tuple[str, ...]
    gold: dict


@dataclass(frozen=True)
class Eligibility:
    candidates: tuple[Candidate, ...]
    dropped: dict[str, int]


def math_candidates(rows: Sequence[Mapping[str, Any]]) -> Eligibility:
    """MATH-500 rows keyed by ``unique_id``, stratified over ``subject`` x ``level``."""
    dropped = {"missing_field": 0, "duplicate_source_id": 0}
    seen: set[str] = set()
    out: list[Candidate] = []
    for row in rows:
        source_id = str(row.get("unique_id") or "")
        problem, answer = row.get("problem"), row.get("answer")
        if not source_id or not isinstance(problem, str) or not isinstance(answer, str) or not answer.strip():
            dropped["missing_field"] += 1
            continue
        if source_id in seen:
            dropped["duplicate_source_id"] += 1
            continue
        seen.add(source_id)
        level = row.get("level")
        out.append(Candidate(
            source_id=source_id,
            stratum=f"{row.get('subject')}|L{level}",
            subject=str(row.get("subject") or ""),
            difficulty=str(level if level is not None else ""),
            task_text=problem.strip(),
            options=(),
            gold={"answer": answer.strip(), "solution": str(row.get("solution") or "")},
        ))
    return Eligibility(tuple(out), dropped)


def mmlu_candidates(rows: Sequence[Mapping[str, Any]]) -> Eligibility:
    """Ten-option MMLU-Pro rows, de-duplicated by question string, stratified by category.

    Spec §5.1 / plan §5.1: only the ten-option rows are eligible, so one ``A``-``J``
    alphabet and one permutation semantics covers every item.  Duplicate question strings
    are removed *before* sampling; the survivor is the row with the smallest
    ``question_id`` so the choice never depends on iteration order.
    """
    dropped = {"wrong_option_count": 0, "bad_answer_index": 0, "duplicate_question": 0,
               "missing_field": 0}
    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        question = row.get("question")
        options = row.get("options")
        answer_index = row.get("answer_index")
        if not isinstance(question, str) or not question.strip() or not isinstance(options, list):
            dropped["missing_field"] += 1
            continue
        if len(options) != MMLU_REQUIRED_OPTIONS:
            dropped["wrong_option_count"] += 1
            continue
        if not isinstance(answer_index, int) or isinstance(answer_index, bool) or not 0 <= answer_index < len(options):
            dropped["bad_answer_index"] += 1
            continue
        by_question.setdefault(question.strip(), []).append(row)
    out: list[Candidate] = []
    for question, group in by_question.items():
        group = sorted(group, key=lambda row: int(row["question_id"]))
        dropped["duplicate_question"] += len(group) - 1
        row = group[0]
        options = tuple(str(option) for option in row["options"])
        out.append(Candidate(
            source_id=f"mmlu_pro-{int(row['question_id'])}",
            stratum=str(row.get("category") or ""),
            subject=str(row.get("category") or ""),
            difficulty="",
            task_text=question,
            options=options,
            gold={"answer_index": int(row["answer_index"]), "source_answer": str(row.get("answer") or "")},
        ))
    return Eligibility(tuple(sorted(out, key=lambda candidate: candidate.source_id)), dropped)


# --------------------------------------------------------------------------- allocation


def proportional_allocation(sizes: Mapping[str, int], total: int, salt: bytes) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` over strata, at least one each.

    Deterministic: remainders tie-break on ``rank_key(salt, stratum)``, so the allocation
    is a pure function of the salt and the stratum sizes (plan §5.1 "proportional, min 1
    per stratum").
    """
    if total < len(sizes):
        raise BuildError(f"cannot draw {total} items with at least one from {len(sizes)} strata")
    population = sum(sizes.values())
    if population < total:
        raise BuildError(f"only {population} eligible rows for {total} requested items")
    ideal = {name: total * size / population for name, size in sizes.items()}
    alloc = {name: min(sizes[name], max(1, int(ideal[name]))) for name in sizes}
    order_up = sorted(sizes, key=lambda name: (-(ideal[name] - int(ideal[name])), rank_key(salt, name)))
    order_down = sorted(sizes, key=lambda name: (ideal[name] - int(ideal[name]), rank_key(salt, name)))
    while sum(alloc.values()) < total:
        progressed = False
        for name in order_up:
            if sum(alloc.values()) == total:
                break
            if alloc[name] < sizes[name]:
                alloc[name] += 1
                progressed = True
        if not progressed:
            raise BuildError("allocation cannot reach the requested total")
    while sum(alloc.values()) > total:
        progressed = False
        for name in order_down:
            if sum(alloc.values()) == total:
                break
            if alloc[name] > 1:
                alloc[name] -= 1
                progressed = True
        if not progressed:
            raise BuildError("allocation cannot fall to the requested total")
    return alloc


def stratified_sample(
    candidates: Sequence[Candidate], total: int, salt: bytes
) -> tuple[list[Candidate], list[Candidate], dict[str, int]]:
    """``(sample, reserve, allocation)``; within a stratum, rank order is the frozen order."""
    strata: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        strata.setdefault(candidate.stratum, []).append(candidate)
    ordered = {
        name: sorted(group, key=lambda candidate: rank_key(salt, candidate.source_id))
        for name, group in strata.items()
    }
    allocation = proportional_allocation({name: len(group) for name, group in ordered.items()}, total, salt)
    sample: list[Candidate] = []
    reserve: list[Candidate] = []
    for name in sorted(ordered):
        group = ordered[name]
        take = allocation[name]
        sample.extend(group[:take])
        reserve.extend(group[take:])
    if len(sample) != total:
        raise BuildError(f"stratified sample produced {len(sample)} items, expected {total}")
    return sample, reserve, allocation


def assign_splits(sample: Sequence[Candidate], salt: bytes) -> dict[str, str]:
    """``{source_id: split}`` stratified over strata, exact per-split totals.

    Items are walked stratum by stratum (strata sorted by name, items in frozen rank
    order) and each is given to the split with the largest current deficit against its
    share.  That is streaming largest-remainder: every stratum is spread across the three
    splits in proportion, and the global totals land exactly on
    :data:`agents_scaling.ard.contract.SPLIT_COUNTS_PER_FAMILY`.
    """
    targets = {name: SPLIT_COUNTS_PER_FAMILY[name] for name in SPLITS}
    total = sum(targets.values())
    if len(sample) != total:
        raise BuildError(f"split assignment needs exactly {total} items, got {len(sample)}")
    by_stratum: dict[str, list[Candidate]] = {}
    for candidate in sample:
        by_stratum.setdefault(candidate.stratum, []).append(candidate)
    walk: list[Candidate] = []
    for name in sorted(by_stratum):
        walk.extend(sorted(by_stratum[name], key=lambda candidate: rank_key(salt, candidate.source_id)))
    assigned = {name: 0 for name in SPLITS}
    out: dict[str, str] = {}
    for position, candidate in enumerate(walk, start=1):
        best = max(SPLITS, key=lambda name: (targets[name] * position / total - assigned[name],
                                             -SPLITS.index(name)))
        assigned[best] += 1
        out[candidate.source_id] = best
    if assigned != targets:
        raise BuildError(f"split assignment landed on {assigned}, expected {targets}")
    return out


# --------------------------------------------------------------------------- identities


def item_id_for(family: Family, revision: str, source_id: str) -> str:
    """Stable public item id: family prefix + a hash over (study, family, revision, id)."""
    digest = identity.sha256_hex(
        identity.jcs([STUDY_ID, family.value, revision, source_id])
    )
    return f"{family.value}-{digest[:12]}"


def option_permutation(item_id: str, study_seed: bytes, *, variant: int = 0) -> tuple[int, ...]:
    """The frozen option permutation of one MMLU item: displayed position -> source index.

    Seeded by the item hash through
    :func:`agents_scaling.study.identity.blind_order_key` (an HMAC over the study seed), so
    it is reproducible, correctness-blind and independent of iteration order.  Variant 1
    is the second permutation of the spec §14.4 answer-identity control; it is derived
    here but is not what ``items.parquet`` stores.
    """
    if variant not in OPTION_PERMUTATION_VARIANTS:
        raise ValueError(f"unknown permutation variant {variant}")
    digest = identity.blind_order_key(study_seed, NS_OPTION_PERMUTATION, item_id, variant)
    remaining = list(range(MMLU_REQUIRED_OPTIONS))
    permutation: list[int] = []
    counter = 0
    while remaining:
        stream = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        counter += 1
        for byte in stream:
            if not remaining:
                break
            limit = len(remaining)
            bound = 256 - (256 % limit)
            if byte >= bound:            # rejection sampling keeps the draw uniform
                continue
            permutation.append(remaining.pop(byte % limit))
    return tuple(permutation)


def permuted_key_letter(permutation: Sequence[int], answer_index: int) -> str:
    """The displayed option letter of the source option ``answer_index``."""
    position = list(permutation).index(int(answer_index))
    return MC_LETTERS[position]


def render_mmlu_task(question: str, displayed_options: Sequence[str]) -> str:
    """Question plus the ``A.``-``J.`` lettered options, in displayed order."""
    lines = [question.strip(), ""]
    lines.extend(f"{MC_LETTERS[index]}. {option.strip()}" for index, option in enumerate(displayed_options))
    return "\n".join(lines)


def prompt_hash(family: Family, task_text: str, displayed_options: Sequence[str]) -> str:
    return identity.sha256_hex(identity.jcs({
        "family": family.value,
        "options": list(displayed_options),
        "task_text": task_text,
    }))


# --------------------------------------------------------------------------- writing


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {name: [row[name] for row in rows] for name in PUBLIC_COLUMNS}
    schema = pa.schema([
        (name, pa.int64() if name in ("n_options", "rank") else pa.string())
        for name in PUBLIC_COLUMNS
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)
    os.chmod(path, 0o644)


@dataclass(frozen=True)
class BuildSummary:
    run_root: str
    counts: dict[str, dict[str, int]]
    allocation: dict[str, dict[str, int]]
    eligible: dict[str, int]
    dropped: dict[str, dict[str, int]]
    reserve: dict[str, int]
    grader_hashes: dict[str, str]
    files: dict[str, str] = field(default_factory=dict)


def build_items(
    out_dir: str | os.PathLike,
    *,
    split_salt: bytes = DEFAULT_SPLIT_SALT,
    study_seed: bytes = DEFAULT_STUDY_SEED,
    sources: RawSources | None = None,
    force: bool = False,
) -> BuildSummary:
    """Build the public item table and the protected gold files under ``out_dir``.

    ``out_dir`` is the run root: files land under ``<out_dir>/data/``.  ``sources``
    overrides the offline HF read (tests inject synthetic pinned rows); when it is
    ``None`` the pinned revisions are read through ``loaders._load_hf``.

    Refuses to overwrite an existing table without ``force``: the panel is frozen before
    any generation and a silent rebuild would re-base every split.
    """
    if not isinstance(split_salt, (bytes, bytearray)) or not split_salt:
        raise ValueError("split_salt must be non-empty bytes")
    if not isinstance(study_seed, (bytes, bytearray)) or not study_seed:
        raise ValueError("study_seed must be non-empty bytes")
    root = Path(out_dir)
    targets = [items_path(root), public_dir(root) / ITEMS_SUMMARY_FILE,
               protected_dir(root) / MATH_GOLD_FILE, protected_dir(root) / MMLU_GOLD_FILE]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise BuildError(f"item export exists (pass force=True to overwrite): {[str(p) for p in existing]}")

    sources = sources or read_raw_sources()
    hashes = graders.grader_hashes()
    eligibility = {
        Family.MATH500: math_candidates(sources.math_rows),
        Family.MMLUPRO: mmlu_candidates(sources.mmlu_rows),
    }
    revisions = {Family.MATH500: sources.math_revision, Family.MMLUPRO: sources.mmlu_revision}

    rows: list[dict] = []
    math_gold: list[dict] = []
    mmlu_gold: list[dict] = []
    reserve: dict[str, list[str]] = {}
    allocation: dict[str, dict[str, int]] = {}
    counts: dict[str, dict[str, int]] = {}
    for family in (Family.MATH500, Family.MMLUPRO):
        candidates = eligibility[family].candidates
        sample, unused, alloc = stratified_sample(candidates, ITEMS_PER_FAMILY, split_salt)
        splits = assign_splits(sample, split_salt)
        allocation[family.value] = alloc
        reserve[family.value] = [candidate.source_id for candidate in unused]
        revision = revisions[family]
        family_rows: list[dict] = []
        for candidate in sample:
            item_id = item_id_for(family, revision, candidate.source_id)
            if family is Family.MMLUPRO:
                permutation = option_permutation(item_id, study_seed)
                displayed = [candidate.options[index] for index in permutation]
                task_text = render_mmlu_task(candidate.task_text, displayed)
                key_letter = permuted_key_letter(permutation, candidate.gold["answer_index"])
                mmlu_gold.append({
                    "item_id": item_id,
                    "source_id": candidate.source_id,
                    "dataset_revision": revision,
                    "answer_index": candidate.gold["answer_index"],
                    "source_answer": candidate.gold["source_answer"],
                    "option_permutation": list(permutation),
                    "permuted_key": key_letter,
                    "options": displayed,
                })
                gold_file, permutation_json, n_options = MMLU_GOLD_FILE, json.dumps(list(permutation)), MMLU_REQUIRED_OPTIONS
                answer_format = ANSWER_FORMAT_MMLU
            else:
                displayed = []
                task_text = candidate.task_text
                math_gold.append({
                    "item_id": item_id,
                    "source_id": candidate.source_id,
                    "dataset_revision": revision,
                    "answer": candidate.gold["answer"],
                    "solution": candidate.gold["solution"],
                })
                gold_file, permutation_json, n_options = MATH_GOLD_FILE, "[]", 0
                answer_format = ANSWER_FORMAT_MATH
            family_rows.append({
                "item_id": item_id,
                "family": family.value,
                "dataset_revision": revision,
                "source_id": candidate.source_id,
                "split": splits[candidate.source_id],
                "group_id": item_id,
                "prompt_hash": prompt_hash(family, task_text, displayed),
                "option_permutation": permutation_json,
                "gold_location": f"{PROTECTED_DIR}/{gold_file}#{item_id}",
                "grader_hash": hashes[family.value],
                "subject": candidate.subject,
                "declared_difficulty_metadata": candidate.difficulty,
                "n_options": n_options,
                "answer_format": answer_format,
                "stratum": candidate.stratum,
                "rank": -1,
                "task_text": task_text,
                "options_json": json.dumps(displayed),
            })
        for split_name in SPLITS:
            block = sorted(
                (row for row in family_rows if row["split"] == split_name),
                key=lambda row: rank_key(split_salt, row["source_id"]),
            )
            for position, row in enumerate(block):
                row["rank"] = position
        counts[family.value] = {name: sum(1 for row in family_rows if row["split"] == name) for name in SPLITS}
        rows.extend(family_rows)

    for row in rows:
        assert_public_row(row, where=f"items row {row['item_id']}")
        ArdTask.from_row(row)           # shape check before anything is written
    rows.sort(key=lambda row: (row["family"], SPLITS.index(row["split"]), row["rank"]))

    directory = protected_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, PROTECTED_DIR_MODE)
    _write_parquet(items_path(root), rows)
    _atomic_write(
        directory / MATH_GOLD_FILE,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in math_gold).encode("utf-8"),
        mode=PROTECTED_FILE_MODE,
    )
    _atomic_write(
        directory / MMLU_GOLD_FILE,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mmlu_gold).encode("utf-8"),
        mode=PROTECTED_FILE_MODE,
    )
    _atomic_write(
        public_dir(root) / RESERVE_FILE,
        json.dumps({"description": "eligible source ids not drawn into the natural panel, "
                                   "in frozen rank order", "reserve": reserve},
                   indent=2, sort_keys=True).encode("utf-8"),
    )
    summary = BuildSummary(
        run_root=str(root),
        counts=counts,
        allocation=allocation,
        eligible={family.value: len(eligibility[family].candidates) for family in eligibility},
        dropped={family.value: eligibility[family].dropped for family in eligibility},
        reserve={name: len(ids) for name, ids in reserve.items()},
        grader_hashes=hashes,
    )
    summary_payload = {
        "study_id": STUDY_ID,
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "split_salt_sha256": hashlib.sha256(bytes(split_salt)).hexdigest(),
        "study_seed_sha256": hashlib.sha256(bytes(study_seed)).hexdigest(),
        "dataset_revisions": {family.value: revisions[family] for family in revisions},
        "counts": counts,
        "totals": {family: sum(block.values()) for family, block in counts.items()},
        "stratum_allocation": allocation,
        "eligible": summary.eligible,
        "dropped": summary.dropped,
        "reserve": summary.reserve,
        "grader_hashes": hashes,
        "option_permutation_namespace": NS_OPTION_PERMUTATION,
        "rank_rule": "HMAC-SHA256(split_salt, b'rank:' + source_id) within stratum, then "
                     "within (family, split)",
        "split_rule": "streaming largest-remainder over strata; exact 48/24/120 per family",
        "public_columns": list(PUBLIC_COLUMNS),
        "gold_files": [f"{PROTECTED_DIR}/{MATH_GOLD_FILE}", f"{PROTECTED_DIR}/{MMLU_GOLD_FILE}"],
        "notes": [
            "cot_content is never read into any artefact",
            "the natural panel is never filtered on measured correctness (spec 5.1)",
        ],
    }
    _atomic_write(public_dir(root) / ITEMS_SUMMARY_FILE,
                  json.dumps(summary_payload, indent=2, sort_keys=True).encode("utf-8"))
    files = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in (items_path(root), public_dir(root) / ITEMS_SUMMARY_FILE,
                     public_dir(root) / RESERVE_FILE,
                     directory / MATH_GOLD_FILE, directory / MMLU_GOLD_FILE)
    }
    _atomic_write(public_dir(root) / HASHES_FILE,
                  json.dumps(files, indent=2, sort_keys=True).encode("utf-8"))
    return BuildSummary(**{**summary.__dict__, "files": files})


# --------------------------------------------------------------------------- validation


def validate_math_grader(
    rows: Sequence[Mapping[str, Any]] | None = None, *, timeout_s: float = 8.0
) -> dict:
    """Pre-freeze validation of the MATH grader over every pinned gold answer (plan §9.2).

    Reports the parse rate, the self-equivalence rate, the agreement between each row's
    ``answer`` field and the boxed string of its own ``solution``, and the list of answer
    shapes the lark backend could not parse — that list is the hand-rolled fallback set
    and an amendment-register entry.
    """
    from agents_scaling.benchmarks.grading import extract_boxed
    from agents_scaling.ard.graders import math_equiv

    if rows is None:
        from agents_scaling.benchmarks.loaders import BENCHMARK_SOURCE_SPECS, _load_hf

        spec = BENCHMARK_SOURCE_SPECS["math"]
        rows = _load_hf(spec.repo_id, revision=spec.revision)[spec.split].to_list()
    parse_failures: list[dict] = []
    self_failures: list[dict] = []
    agreement: dict[str, int] = {}
    for row in rows:
        answer = str(row["answer"])
        node = math_equiv.parse_answer(math_equiv.normalise(answer))
        if not node.parsed:
            parse_failures.append({"source_id": row.get("unique_id"), "answer": answer,
                                   "reason": node.reason})
        status, detail = math_equiv.self_equivalence(answer, timeout_s=timeout_s)
        if status != math_equiv.STATUS_CORRECT:
            self_failures.append({"source_id": row.get("unique_id"), "answer": answer,
                                  "status": status, "detail": detail})
        boxed = extract_boxed(str(row.get("solution") or ""))
        if boxed is None:
            agreement["no_boxed"] = agreement.get("no_boxed", 0) + 1
            continue
        verdict = math_equiv.compare(boxed, answer, timeout_s=timeout_s)
        agreement[verdict.status] = agreement.get(verdict.status, 0) + 1
    total = len(rows)
    return {
        "n": total,
        "parse_rate": (total - len(parse_failures)) / total if total else 0.0,
        "self_equivalence_rate": (total - len(self_failures)) / total if total else 0.0,
        "answer_vs_boxed_solution": agreement,
        "parse_failures": parse_failures,
        "self_equivalence_failures": self_failures,
        "lark_version": math_equiv.PINNED_LARK_VERSION,
        "grader_hash": graders.grader_hash(Family.MATH500),
    }


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen ARD36 item table.")
    parser.add_argument("--run-root", help="run root; files are written under <run-root>/data")
    parser.add_argument("--force", action="store_true", help="overwrite an existing export")
    parser.add_argument("--validate-math-grader", metavar="PATH", nargs="?", const="-",
                        help="run the 500-answer MATH grader validation and write JSON here")
    args = parser.parse_args(argv)
    if args.validate_math_grader:
        report = validate_math_grader()
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.validate_math_grader == "-":
            print(payload)
        else:
            Path(args.validate_math_grader).write_text(payload, encoding="utf-8")
        return 0
    if not args.run_root:
        parser.error("--run-root is required unless --validate-math-grader is given")
    summary = build_items(args.run_root, force=args.force)
    print(json.dumps({"counts": summary.counts, "eligible": summary.eligible,
                      "reserve": summary.reserve, "files": summary.files},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
