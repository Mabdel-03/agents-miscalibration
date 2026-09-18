"""GPQA-Diamond adapter (198 graduate-level science questions, four options).

Source: ``Idavidrein/gpqa`` (gated) at the pinned revision, ``gpqa_diamond.csv`` (198 rows), read offline from the
Hugging Face hub cache under ``$HF_HOME`` with :mod:`csv`. Licence: CC BY 4.0 (the dataset card also asks that
examples are not revealed in plain text online; nothing here publishes them).

Config keys consumed: ``benchmarks[id=gpqa_diamond].source`` (must be the pinned repo) and ``.split`` (``train``).

Preprocessing ``gpqa_diamond-prompt-v1-shuffle20260917`` (frozen):

* Outer whitespace rule: the question and every option lose leading/trailing ASCII whitespace
  (``" \\t\\n\\r\\f\\v"``) and nothing else (47 source rows carry a trailing newline somewhere).
* Option order: the source lists ``Correct Answer``, ``Incorrect Answer 1..3``. They are permuted by
  ``random.Random(f"{SHUFFLE_SEED}-{record_id}").shuffle([0, 1, 2, 3])`` with ``SHUFFLE_SEED = 20260917`` (the
  study seed, frozen here so a config change can never silently move a gold letter); the recipe is the one in
  ``agents_scaling/benchmarks/loaders.py::_mcq`` keyed on the Record ID instead of the row position. Python's
  ``random.Random`` seeded with a ``str`` is stable across interpreter versions (sha512 of the seed).
* Prompt layout: ``<question>\\n\\nOptions:\\nA. <option>\\nB. ...\\nD. <option>`` (as MMLU-Pro).
* ``allowed_choices``: ``("A", "B", "C", "D")`` on both records. Gold: the letter the correct answer landed on.
* Public fields: question and shuffled options only. The source fields in ``SOURCE_FIELDS`` (the four answers by
  role, subdomain, domain) and the permutation live in the hashed private record; ``Explanation`` is never read.
  ``item_id = "gpqa_diamond/<Record ID>"``.
* Strata: ``High-level domain`` (Chemistry 93, Physics 86, Biology 19).
* Engineering exclusions (pre-generation, per row): an empty question/option after trimming, or two options with
  identical trimmed text (option identity would no longer be answer identity for voting). The pinned file has two
  such rows (``recZSGUkn56v9kEp1``, ``recnGEpF1srQpaqWq``: two identical distractors each), so 196 are eligible.
"""
from __future__ import annotations

import csv
import io
import random
import string
from typing import Any, Mapping, Sequence

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

BENCHMARK = "gpqa_diamond"
SOURCE_REPO = "Idavidrein/gpqa"
SOURCE_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
SOURCE_SPLIT = "train"
SOURCE_FILE = "gpqa_diamond.csv"
SOURCE_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
SOURCE_ROWS = 198
LICENSE = "CC BY 4.0 (Idavidrein/gpqa; gated: do not reveal examples in plain text online)"
SHUFFLE_SEED = 20260917
PREPROCESSING_VERSION = f"gpqa_diamond-prompt-v1-shuffle{SHUFFLE_SEED}"
OUTER_WHITESPACE = " \t\n\r\f\v"
LETTERS = string.ascii_uppercase[:4]
OPTION_FIELDS = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
SOURCE_FIELDS = ("Record ID", "Question", *OPTION_FIELDS, "Subdomain", "High-level domain")


def source_record() -> dict[str, Any]:
    return {"benchmark": BENCHMARK, "repo": SOURCE_REPO, "revision": SOURCE_REVISION, "split": SOURCE_SPLIT,
            "file": SOURCE_FILE, "sha256": SOURCE_SHA256, "rows": SOURCE_ROWS, "license": LICENSE,
            "preprocessing_version": PREPROCESSING_VERSION, "shuffle_seed": SHUFFLE_SEED,
            "shuffle_rule": "random.Random(f'{shuffle_seed}-{Record ID}').shuffle([0,1,2,3]) over "
                            "[Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3]",
            "outer_whitespace_rule": "strip leading/trailing ASCII whitespace ' \\t\\n\\r\\f\\v' from the question "
                                     "and each option; inner text unchanged"}


def trim_outer(text: str) -> str:
    return text.strip(OUTER_WHITESPACE)


def option_order(record_id: str, *, seed: int = SHUFFLE_SEED) -> list[int]:
    """Source option positions in presented order (position 0 is the correct answer)."""
    order = list(range(len(OPTION_FIELDS)))
    random.Random(f"{seed}-{record_id}").shuffle(order)
    return order


def render_prompt(question: str, options: Sequence[str]) -> str:
    lines = [trim_outer(question), "", "Options:"]
    lines.extend(f"{LETTERS[index]}. {trim_outer(option)}" for index, option in enumerate(options))
    return "\n".join(lines)


def _exclusion_reason(question: str, presented: Sequence[str]) -> str | None:
    if not trim_outer(question):
        return "empty_question"
    trimmed = [trim_outer(option) for option in presented]
    if any(not option for option in trimmed):
        return "empty_option"
    seen: dict[str, str] = {}
    for letter, option in zip(LETTERS, trimmed):
        if option in seen:
            return f"duplicate_option_text:{seen[option]},{letter}"
        seen[option] = letter
    return None


def build_item(row: Mapping[str, Any], *, source_revision: str = SOURCE_REVISION) -> InventoryItem:
    missing = [name for name in SOURCE_FIELDS if name not in row]
    if missing:
        raise SourceIntegrityError(f"GPQA row lacks fields {missing}")
    record = {name: str(row[name]) for name in SOURCE_FIELDS}
    record_id = record["Record ID"]
    if not (record_id and record_id.isalnum()):
        raise SourceIntegrityError(f"GPQA Record ID {record_id!r} is not alphanumeric")
    item_id = f"{BENCHMARK}/{record_id}"
    order = option_order(record_id)
    presented = [record[OPTION_FIELDS[position]] for position in order]
    gold = LETTERS[order.index(0)]
    allowed = tuple(LETTERS)
    public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                        prompt=render_prompt(record["Question"], presented), output_kind=OutputKind.OPTION,
                        allowed_choices=allowed)
    private_record = {
        "benchmark": BENCHMARK, "item_id": item_id, "source_revision": source_revision,
        "preprocessing_version": PREPROCESSING_VERSION, "eval_kind": EvalKind.OPTION_EXACT.value,
        "gold_answer": gold, "allowed_choices": list(allowed), "option_order": order, "source_record": record,
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                              eval_kind=EvalKind.OPTION_EXACT, gold_answer=gold, private_record_ref=None,
                              private_hash=canonical_hash(private_record), allowed_choices=allowed)
    reason = _exclusion_reason(record["Question"], presented)
    return InventoryItem(public=public, private=private, strata=(("domain", record["High-level domain"]),),
                         eligible=reason is None, exclusion_reason=reason)


def read_rows() -> list[dict[str, Any]]:
    path = hf_snapshot_file(SOURCE_REPO, SOURCE_REVISION, SOURCE_FILE)
    data = read_verified(path, SOURCE_SHA256)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    missing = [name for name in SOURCE_FIELDS if name not in (reader.fieldnames or ())]
    if missing:
        raise SourceIntegrityError(f"GPQA csv lacks columns {missing}")
    rows = list(reader)
    if len(rows) != SOURCE_ROWS:
        raise SourceIntegrityError(f"GPQA-Diamond: {len(rows)} rows, pinned {SOURCE_ROWS}")
    return rows


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    entry = benchmark_config(cfg, BENCHMARK, source=SOURCE_REPO)
    if entry.get("split") != SOURCE_SPLIT:
        raise ValueError(f"{BENCHMARK}: config split {entry.get('split')!r} != pinned {SOURCE_SPLIT!r}")
    items = [build_item(row) for row in read_rows()]
    ids = [item.public.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise SourceIntegrityError("GPQA-Diamond: duplicate Record ID")
    return items


#: Dataset-lock provenance read by ``manifest.dataset_lock_payload`` (plain dict so it serializes as JSON).
SOURCE_PROVENANCE: dict[str, Any] = source_record()
