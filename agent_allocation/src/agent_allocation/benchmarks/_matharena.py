"""Shared reader for the MathArena competition snapshots (one parquet per contest year) behind ``aime`` and ``hmmt``.

Each year is a separate Hugging Face dataset (``MathArena/<contest>_<year>``) pinned at its own revision, with one
file ``data/train-00000-of-00001.parquet`` whose sha256 and row count are pinned. A benchmark is the union of its
years. Rows gain one derived field, ``year`` (the pinned source's year), before :func:`build_item` sees them.

Preprocessing ``<benchmark>-prompt-v1`` (frozen): ``PublicTask.prompt`` is ``problem`` with leading/trailing ASCII
whitespace (``" \\t\\n\\r\\f\\v"``) removed and nothing else. Gold is ``answer`` as a string (AIME stores integers:
``str(int)``; HMMT stores LaTeX strings verbatim). Public fields: problem only. Private: gold plus the source row
(``problem_idx``, ``problem_type`` when the year has it) inside the hashed private record.
``item_id = "<benchmark>/<year>/<problem_idx>"``. ``source_revision`` is the revision of the item's own year file.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
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

OUTER_WHITESPACE = " \t\n\r\f\v"
PARQUET_FILE = "data/train-00000-of-00001.parquet"


@dataclass(frozen=True)
class YearSource:
    year: int
    repo: str
    revision: str
    sha256: str
    rows: int
    file: str = PARQUET_FILE

    def record(self) -> dict[str, Any]:
        return {"year": self.year, "repo": self.repo, "revision": self.revision, "file": self.file,
                "sha256": self.sha256, "rows": self.rows}


@dataclass(frozen=True)
class MathArenaSpec:
    benchmark: str
    config_source: str                     # the ``benchmarks[id].source`` string the config must carry
    split: str
    sources: tuple[YearSource, ...]
    license: str
    preprocessing_version: str
    stratify_problem_type: bool
    exclusions: Mapping[str, str]

    @property
    def rows(self) -> int:
        return sum(source.rows for source in self.sources)

    def revision_for(self, year: int) -> str:
        for source in self.sources:
            if source.year == year:
                return source.revision
        raise SourceIntegrityError(f"{self.benchmark}: no pinned source for year {year!r}")


def source_record(spec: MathArenaSpec) -> dict[str, Any]:
    return {"benchmark": spec.benchmark, "repo": spec.config_source, "split": spec.split,
            "files": [source.record() for source in spec.sources], "rows": spec.rows, "license": spec.license,
            "preprocessing_version": spec.preprocessing_version,
            "outer_whitespace_rule": "strip leading/trailing ASCII whitespace ' \\t\\n\\r\\f\\v' from the problem; "
                                     "inner text unchanged"}


def _gold(answer: Any) -> str | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return str(answer)
    if isinstance(answer, str):
        return answer
    return None


def build_item(spec: MathArenaSpec, row: Mapping[str, Any], *, source_revision: str | None = None) -> InventoryItem:
    missing = [name for name in ("year", "problem_idx", "problem", "answer") if name not in row]
    if missing:
        raise SourceIntegrityError(f"{spec.benchmark} row lacks fields {missing}")
    year, index = row["year"], row["problem_idx"]
    if isinstance(year, bool) or not isinstance(year, int) or isinstance(index, bool) or not isinstance(index, int):
        raise SourceIntegrityError(f"{spec.benchmark}: year/problem_idx must be ints, got {year!r}/{index!r}")
    revision = source_revision if source_revision is not None else spec.revision_for(year)
    item_id = f"{spec.benchmark}/{year}/{index}"
    problem = row["problem"] if isinstance(row["problem"], str) else ""
    prompt = problem.strip(OUTER_WHITESPACE)
    gold = _gold(row["answer"])
    types = [str(value) for value in (row.get("problem_type") or ())]
    record = {"year": year, "problem_idx": index, "problem": row["problem"], "answer": row["answer"]}
    if "problem_type" in row:
        record["problem_type"] = types
    public = PublicTask(item_id=item_id, benchmark=spec.benchmark, source_revision=revision, prompt=prompt,
                        output_kind=OutputKind.MATH)
    private_record = {
        "benchmark": spec.benchmark, "item_id": item_id, "source_revision": revision,
        "preprocessing_version": spec.preprocessing_version, "eval_kind": EvalKind.MATH_VERIFY.value,
        "gold_answer": gold, "source_record": record,
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=spec.benchmark, source_revision=revision,
                              eval_kind=EvalKind.MATH_VERIFY, gold_answer=gold, private_record_ref=None,
                              private_hash=canonical_hash(private_record))
    reason = spec.exclusions.get(item_id)
    if reason is None and not prompt:
        reason = "empty_problem"
    if reason is None and not (gold and gold.strip()):
        reason = "empty_gold_answer"
    strata: tuple[tuple[str, str], ...] = (("year", str(year)),)
    if spec.stratify_problem_type:
        strata += (("problem_type", types[0] if types else "none"),)
    return InventoryItem(public=public, private=private, strata=strata, eligible=reason is None,
                         exclusion_reason=reason)


def read_rows(spec: MathArenaSpec) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for source in spec.sources:
        path = hf_snapshot_file(source.repo, source.revision, source.file)
        table = pq.read_table(io.BytesIO(read_verified(path, source.sha256)))
        needed = ["problem_idx", "problem", "answer"] + (["problem_type"] if spec.stratify_problem_type else [])
        absent = [name for name in needed if name not in table.column_names]
        if absent:
            raise SourceIntegrityError(f"{source.repo}: parquet lacks columns {absent}")
        year_rows = table.to_pylist()
        if len(year_rows) != source.rows:
            raise SourceIntegrityError(f"{source.repo}: {len(year_rows)} rows, pinned {source.rows}")
        for row in year_rows:
            if "year" in row:
                raise SourceIntegrityError(f"{source.repo}: source already carries a 'year' column")
            rows.append({**row, "year": source.year})
    return rows


def load_inventory(spec: MathArenaSpec, cfg: Mapping[str, Any]) -> list[InventoryItem]:
    entry = benchmark_config(cfg, spec.benchmark, source=spec.config_source)
    if entry.get("split") != spec.split:
        raise ValueError(f"{spec.benchmark}: config split {entry.get('split')!r} != pinned {spec.split!r}")
    items = [build_item(spec, row) for row in read_rows(spec)]
    return check_unique(spec, items)


def check_unique(spec: MathArenaSpec, items: Sequence[InventoryItem]) -> list[InventoryItem]:
    ids = [item.public.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise SourceIntegrityError(f"{spec.benchmark}: duplicate (year, problem_idx)")
    unknown = set(spec.exclusions) - set(ids)
    if unknown:
        raise SourceIntegrityError(f"{spec.benchmark}: exclusions name unknown items {sorted(unknown)}")
    return list(items)
