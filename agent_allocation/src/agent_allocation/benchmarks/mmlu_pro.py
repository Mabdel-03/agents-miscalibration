"""MMLU-Pro adapter (master spec 2.2).

Source: ``TIGER-Lab/MMLU-Pro`` at the pinned revision, ``data/test-00000-of-00001.parquet`` (12,032 rows), read
offline from the Hugging Face hub cache under ``$HF_HOME`` with pyarrow. Licence: MIT.

Config keys consumed: ``benchmarks[id=mmlu_pro].source`` (must be the pinned repo) and ``.split`` (``test``).

Preprocessing ``mmlu_pro-prompt-v1`` (frozen):

* Outer whitespace rule: the question and every option lose leading/trailing ASCII whitespace
  (``" \\t\\n\\r\\f\\v"``) and nothing else. Inner text is untouched. Unicode whitespace is deliberately not
  stripped: one pinned question ends in ``\\x85`` (an ellipsis in the source encoding), which ``str.strip()``
  would delete.
* Prompt layout: ``<question>\\n\\nOptions:\\nA. <option 0>\\nB. <option 1>...`` with options in their original
  order, lettered from ``A``.
* ``allowed_choices``: the letters of this item's options only. Gold: the letter of ``answer_index``.
* Public fields: question and options. Never ``answer``, ``answer_index``, ``cot_content`` (they live in the
  hashed private record together with ``category`` and ``src``). ``item_id = "mmlu_pro/<question_id>"``.
* Strata: ``category``.
* Engineering exclusions (pre-generation, content-blind to models): ``answer`` letter disagreeing with
  ``answer_index``, fewer than two or more than ten options, an empty question/option after trimming, or two
  options with identical trimmed text (option identity would no longer be answer identity for voting).
"""
from __future__ import annotations

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

BENCHMARK = "mmlu_pro"
SOURCE_REPO = "TIGER-Lab/MMLU-Pro"
SOURCE_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SOURCE_SPLIT = "test"
SOURCE_FILE = "data/test-00000-of-00001.parquet"
SOURCE_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
SOURCE_ROWS = 12032
LICENSE = "MIT (TIGER-Lab/MMLU-Pro)"
PREPROCESSING_VERSION = "mmlu_pro-prompt-v1"
OUTER_WHITESPACE = " \t\n\r\f\v"
MAX_OPTIONS = 10
LETTERS = string.ascii_uppercase[:MAX_OPTIONS]
SOURCE_FIELDS = ("question_id", "question", "options", "answer", "answer_index", "cot_content", "category", "src")


def source_record() -> dict[str, Any]:
    return {"benchmark": BENCHMARK, "repo": SOURCE_REPO, "revision": SOURCE_REVISION, "split": SOURCE_SPLIT,
            "file": SOURCE_FILE, "sha256": SOURCE_SHA256, "rows": SOURCE_ROWS, "license": LICENSE,
            "preprocessing_version": PREPROCESSING_VERSION,
            "outer_whitespace_rule": "strip leading/trailing ASCII whitespace ' \\t\\n\\r\\f\\v' from the question "
                                     "and each option; inner text unchanged"}


def trim_outer(text: str) -> str:
    return text.strip(OUTER_WHITESPACE)


def render_prompt(question: str, options: Sequence[str]) -> str:
    lines = [trim_outer(question), "", "Options:"]
    lines.extend(f"{LETTERS[index]}. {trim_outer(option)}" for index, option in enumerate(options))
    return "\n".join(lines)


def _exclusion_reason(row: Mapping[str, Any]) -> str | None:
    options = row["options"]
    if not 2 <= len(options) <= MAX_OPTIONS:
        return f"option_count_{len(options)}"
    index = row["answer_index"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(options):
        return "answer_index_out_of_range"
    if row["answer"] != LETTERS[index]:
        return "answer_letter_disagrees_with_answer_index"
    if not trim_outer(row["question"]):
        return "empty_question"
    trimmed = [trim_outer(option) for option in options]
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
        raise SourceIntegrityError(f"MMLU-Pro row lacks fields {missing}")
    record = {name: row[name] for name in SOURCE_FIELDS}
    record["options"] = [str(option) for option in row["options"]]
    item_id = f"{BENCHMARK}/{int(record['question_id'])}"
    reason = _exclusion_reason(record)
    options = record["options"][:MAX_OPTIONS]
    allowed = tuple(LETTERS[:len(options)])
    gold = None
    index = record["answer_index"]
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options):
        gold = LETTERS[index]
    public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                        prompt=render_prompt(record["question"], options), output_kind=OutputKind.OPTION,
                        allowed_choices=allowed)
    private_record = {
        "benchmark": BENCHMARK, "item_id": item_id, "source_revision": source_revision,
        "preprocessing_version": PREPROCESSING_VERSION, "eval_kind": EvalKind.OPTION_EXACT.value,
        "gold_answer": gold, "allowed_choices": list(allowed), "source_record": record,
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                              eval_kind=EvalKind.OPTION_EXACT, gold_answer=gold, private_record_ref=None,
                              private_hash=canonical_hash(private_record))
    return InventoryItem(public=public, private=private, strata=(("category", str(record["category"])),),
                         eligible=reason is None, exclusion_reason=reason)


def read_rows() -> list[dict[str, Any]]:
    import io

    import pyarrow.parquet as pq

    path = hf_snapshot_file(SOURCE_REPO, SOURCE_REVISION, SOURCE_FILE)
    data = read_verified(path, SOURCE_SHA256)
    table = pq.read_table(io.BytesIO(data))
    missing = [name for name in SOURCE_FIELDS if name not in table.column_names]
    if missing:
        raise SourceIntegrityError(f"MMLU-Pro parquet lacks columns {missing}")
    rows = table.to_pylist()
    if len(rows) != SOURCE_ROWS:
        raise SourceIntegrityError(f"MMLU-Pro: {len(rows)} rows, pinned {SOURCE_ROWS}")
    return rows


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    entry = benchmark_config(cfg, BENCHMARK, source=SOURCE_REPO)
    if entry.get("split") != SOURCE_SPLIT:
        raise ValueError(f"{BENCHMARK}: config split {entry.get('split')!r} != pinned {SOURCE_SPLIT!r}")
    items = [build_item(row) for row in read_rows()]
    ids = [item.public.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise SourceIntegrityError("MMLU-Pro: duplicate question_id")
    return items
