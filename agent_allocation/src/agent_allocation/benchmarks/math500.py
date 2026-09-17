"""MATH-500 adapter (master spec 2.1).

Source: ``HuggingFaceH4/MATH-500`` at the pinned revision, ``test.jsonl`` (500 rows), read offline from the
Hugging Face hub cache under ``$HF_HOME``. Licence: MIT (MATH by Hendrycks et al., hendrycks/math; MATH-500
subset curated by HuggingFaceH4).

Config keys consumed: ``benchmarks[id=math500].source`` (must be the pinned repo) and ``.split`` (``test``).

Preprocessing ``math500-prompt-v1``: ``PublicTask.prompt`` is the ``problem`` string byte-for-byte (no row has
outer whitespace). Public fields: problem only. Private: ``answer`` (gold), plus ``solution``, ``level``,
``subject`` and ``unique_id`` inside the hashed private record. ``item_id = "math500/<unique_id>"``.
Strata: ``level`` x ``subject``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from agent_allocation.benchmarks.text_manifest import (
    SourceIntegrityError,
    benchmark_config,
    hf_snapshot_file,
    read_verified,
)
from agent_allocation.contracts import (
    EvalKind,
    InventoryItem,
    OutputKind,
    PrivateEvalSpec,
    PublicTask,
    canonical_hash,
)
from agent_allocation.paths import RunPaths

BENCHMARK = "math500"
SOURCE_REPO = "HuggingFaceH4/MATH-500"
SOURCE_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SOURCE_SPLIT = "test"
SOURCE_FILE = "test.jsonl"
SOURCE_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
SOURCE_ROWS = 500
LICENSE = "MIT (MATH: hendrycks/math; MATH-500 subset: HuggingFaceH4/MATH-500)"
PREPROCESSING_VERSION = "math500-prompt-v1"
SOURCE_FIELDS = ("problem", "solution", "answer", "subject", "level", "unique_id")
UNIQUE_ID_PATTERN = re.compile(r"test/[a-z_]+/\d+\.json")

#: Frozen pre-generation engineering exclusions ``{item_id: reason}``. Empty: every pinned gold answer parses
#: and verifies against itself under the pinned Math-Verify rule (recomputed by the text unit tests).
ENGINEERING_EXCLUSIONS: Mapping[str, str] = {}


def source_record() -> dict[str, Any]:
    """Dataset-lock facts for this benchmark (the lead writes them into ``paths.dataset_lock``)."""
    return {"benchmark": BENCHMARK, "repo": SOURCE_REPO, "revision": SOURCE_REVISION, "split": SOURCE_SPLIT,
            "file": SOURCE_FILE, "sha256": SOURCE_SHA256, "rows": SOURCE_ROWS, "license": LICENSE,
            "preprocessing_version": PREPROCESSING_VERSION}


def item_id_for(row: Mapping[str, Any]) -> str:
    return f"{BENCHMARK}/{row['unique_id']}"


def build_item(row: Mapping[str, Any], *, source_revision: str = SOURCE_REVISION) -> InventoryItem:
    missing = [name for name in SOURCE_FIELDS if name not in row]
    if missing:
        raise SourceIntegrityError(f"MATH-500 row lacks fields {missing}")
    if not (isinstance(row["unique_id"], str) and UNIQUE_ID_PATTERN.fullmatch(row["unique_id"])):
        raise SourceIntegrityError(f"MATH-500 unique_id {row['unique_id']!r} is not test/<subject>/<n>.json")
    item_id = item_id_for(row)
    problem = row["problem"]
    answer = row["answer"]
    public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision, prompt=problem,
                        output_kind=OutputKind.MATH)
    private_record = {
        "benchmark": BENCHMARK, "item_id": item_id, "source_revision": source_revision,
        "preprocessing_version": PREPROCESSING_VERSION, "eval_kind": EvalKind.MATH_VERIFY.value,
        "gold_answer": answer, "source_record": {name: row[name] for name in SOURCE_FIELDS},
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                              eval_kind=EvalKind.MATH_VERIFY, gold_answer=answer, private_record_ref=None,
                              private_hash=canonical_hash(private_record))
    reason = ENGINEERING_EXCLUSIONS.get(item_id)
    if reason is None and not (isinstance(problem, str) and problem.strip()):
        reason = "empty_problem"
    if reason is None and not (isinstance(answer, str) and answer.strip()):
        reason = "empty_gold_answer"
    return InventoryItem(public=public, private=private,
                         strata=(("level", str(row["level"])), ("subject", str(row["subject"]))),
                         eligible=reason is None, exclusion_reason=reason)


def read_rows() -> list[dict[str, Any]]:
    path = hf_snapshot_file(SOURCE_REPO, SOURCE_REVISION, SOURCE_FILE)
    data = read_verified(path, SOURCE_SHA256)
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != SOURCE_ROWS:
        raise SourceIntegrityError(f"MATH-500: {len(rows)} rows, pinned {SOURCE_ROWS}")
    return rows


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    entry = benchmark_config(cfg, BENCHMARK, source=SOURCE_REPO)
    if entry.get("split") != SOURCE_SPLIT:
        raise ValueError(f"{BENCHMARK}: config split {entry.get('split')!r} != pinned {SOURCE_SPLIT!r}")
    items = [build_item(row) for row in read_rows()]
    ids = [item.public.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise SourceIntegrityError("MATH-500: duplicate unique_id")
    unknown = set(ENGINEERING_EXCLUSIONS) - set(ids)
    if unknown:
        raise SourceIntegrityError(f"MATH-500: exclusions name unknown items {sorted(unknown)}")
    return items
