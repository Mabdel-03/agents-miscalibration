"""Pinned dataset loaders and ``datasets.lock.json`` (spec 5.1, 5.2, 17.4; plan §2.1, D19).

**Evaluator side.**  This module is the sole importer of
:mod:`agents_scaling.benchmarks.loaders` (and therefore of the HF ``datasets`` library) in
``nextdyn``; runtime code reads items through :mod:`agents_scaling.nextdyn.items` (R7).

What it guarantees:

* every row comes from an exact commit revision -- ``_load_hf`` refuses a load that is not
  revision-pinned and offline, and ``_validate_loaded_cache_identity`` refuses the library's
  "latest cached version" fallback (``benchmarks/loaders.py:193``);
* the *broader MATH* panel (plan D19) is read straight from the cached
  ``EleutherAI/hendrycks_math`` snapshot parquet with pyarrow -- the script loader is dead --
  its gold is ``extract_boxed(solution)``, and **every row whose normalized text hash matches
  a MATH-500 row is dropped**, counted into ``n_overlap_removed``, so the two MATH panels are
  disjoint by construction rather than by assumption (MATH-500 is a subset of the hendrycks
  test split);
* :func:`normalized_text_hash` is the one near-duplicate / overlap key: NFKC, whitespace
  collapsed, casefolded, options **sorted** so the hash is permutation-invariant;
* :func:`lock_datasets` records, per family, ``repo_id``, ``config_name``, ``split``,
  ``revision``, the file it actually read, that file's sha256, ``n_rows``,
  ``row_content_sha256`` (the identity of the exact filtered rows), the filter counts,
  ``n_overlap_removed`` and the ``option_count_histogram`` -- the pins spec 5.4/17.4 require
  before confirmation, plus the legacy root manifest hashes E0 reads.

Nothing here decides a split, a panel size or a cohort: those are ``inventory``'s.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import yaml

from agents_scaling.ard.graders._boxed import extract_boxed
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn.config import CONFIG_DIR, StudyConfig, sibling_config
from agents_scaling.nextdyn.errors import LockSourceMissing, LockError
from agents_scaling.nextdyn.locks import (
    LOCK_SCHEMA_VERSION,
    DatasetLockEntry,
    DatasetsLock,
    LegacyRootEntry,
)
from agents_scaling.study import identity

#: ``$HF_HOME/hub/datasets--<org>--<name>/snapshots/<revision>/`` (the hub snapshot layout).
HUB_DIR: Final[str] = "hub"
SNAPSHOTS_DIR: Final[str] = "snapshots"

LEGACY_SOURCES_FILE: Final[str] = "legacy_sources.yaml"
#: The file whose sha256 stands for "this legacy root, as it was when we read it".
LEGACY_MANIFESTS: Final[Mapping[str, str]] = {
    "ard36": "data/public/items_summary.json",
    "study_v4": "data/DATA_SHA256.json",
    "full_sweep_v1": "benchmark_contracts.v1.json",
}

#: Filter reasons recorded per family in ``datasets.lock.json``'s ``filters``.
DROP_MISSING_FIELD: Final[str] = "missing_field"
DROP_DUPLICATE_ID: Final[str] = "duplicate_source_id"
DROP_DUPLICATE_QUESTION: Final[str] = "duplicate_question"
DROP_OPTION_COUNT: Final[str] = "option_count_outside_window"
DROP_BAD_ANSWER: Final[str] = "bad_answer_index"
DROP_NO_BOXED: Final[str] = "no_boxed_answer"
DROP_MATH500_OVERLAP: Final[str] = "math500_overlap"


# --------------------------------------------------------------------------- normalized text


def _normalize(text: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def normalized_text_hash(question: str, options: Sequence[str] | None = None) -> str:
    """The overlap / near-duplicate key of one item.

    NFKC, whitespace collapsed, casefolded; the options are normalized the same way and
    **sorted**, so the hash does not move when an item is re-permuted and two panels that
    present the same question in different option orders still collide.
    """
    payload = {"question": _normalize(question),
               "options": sorted(_normalize(o) for o in (options or ()))}
    return identity.sha256_hex(identity.jcs(payload))


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class SourceRow:
    """One eligible raw row, normalized across families and before any permutation.

    ``options`` are in the dataset's **canonical** order (option ``i`` is canonical option
    ``i``); ``gold`` is the evaluator-side reference and never leaves this module except
    through the protected gold store.
    """

    family: str
    original_id: str
    question: str
    options: tuple[str, ...]
    gold: Mapping[str, Any]
    subject: str | None
    level: str | None

    @property
    def n_options(self) -> int | None:
        return len(self.options) or None

    @property
    def normalized_text_hash(self) -> str:
        return normalized_text_hash(self.question, self.options)

    def content_hash(self) -> str:
        """Identity of the exact row content, gold included (archived in the lock)."""
        return identity.sha256_hex(identity.jcs({
            "question": self.question, "options": list(self.options),
            "gold": {k: self.gold[k] for k in sorted(self.gold)},
        }))


@dataclass(frozen=True)
class LoadedPanel:
    """The filtered rows of one family plus everything the lock records about them."""

    family: str
    rows: tuple[SourceRow, ...]
    file_path: str | None
    file_sha256: str | None
    n_rows_raw: int | None
    filters: Mapping[str, Any]
    n_overlap_removed: int
    option_count_histogram: Mapping[str, int] | None

    @property
    def row_content_sha256(self) -> str:
        return identity.sha256_hex(identity.jcs(
            [[row.original_id, row.content_hash()] for row in self.rows]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _histogram(rows: Iterable[SourceRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(len(row.options))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


# --------------------------------------------------------------------------- hub readers


def snapshot_dir(hf_home: str | os.PathLike, repo_id: str, revision: str) -> Path:
    org, _, name = str(repo_id).partition("/")
    if not org or not name:
        raise LockError(f"repo_id {repo_id!r} is not '<org>/<name>'")
    return Path(hf_home) / HUB_DIR / f"datasets--{org}--{name}" / SNAPSHOTS_DIR / revision


def _hf_rows(spec: F.LoaderSpec) -> tuple[list[dict[str, Any]], Path | None]:
    """Rows of a revision-pinned arrow split, with the cache file they came from."""
    from agents_scaling.benchmarks import loaders

    args = [spec.repo_id] + ([spec.config_name] if spec.config_name else [])
    try:
        split = loaders._load_hf(*args, revision=spec.revision)[spec.split]
        rows = split.to_list()
        files = [Path(record["filename"]) for record in split.cache_files]
        return rows, files[0] if files else None
    except (TypeError, ValueError) as exc:
        if not spec.benchmark_key:
            raise LockError(f"{spec.repo_id}@{spec.revision}: {exc}") from exc
        rows, path, _sha = loaders._read_cached_arrow(spec.benchmark_key)
        return list(rows), Path(path)


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


# --------------------------------------------------------------------------- per-family panels


def _math500(hf_home: str | os.PathLike) -> LoadedPanel:
    spec = F.family(F.MATH500).loader
    raw, path = _hf_rows(spec)
    filters = {DROP_MISSING_FIELD: 0, DROP_DUPLICATE_ID: 0}
    seen: set[str] = set()
    rows: list[SourceRow] = []
    for row in raw:
        original_id = str(row.get("unique_id") or "")
        problem, answer = row.get("problem"), row.get("answer")
        if not original_id or not isinstance(problem, str) or not isinstance(answer, str) \
                or not answer.strip():
            filters[DROP_MISSING_FIELD] += 1
            continue
        if original_id in seen:
            filters[DROP_DUPLICATE_ID] += 1
            continue
        seen.add(original_id)
        rows.append(SourceRow(family=F.MATH500, original_id=original_id, question=problem,
                              options=(), gold={"answer": answer.strip(),
                                                "solution": str(row.get("solution") or "")},
                              subject=_str_or_none(row.get("subject")),
                              level=_str_or_none(row.get("level"))))
    return LoadedPanel(family=F.MATH500, rows=tuple(rows), file_path=str(path) if path else None,
                       file_sha256=_sha256_file(path) if path else None, n_rows_raw=len(raw),
                       filters=filters, n_overlap_removed=0, option_count_histogram=None)


def _math_expanded(hf_home: str | os.PathLike, exclude: frozenset[str]) -> LoadedPanel:
    """The seven cached ``hendrycks_math`` subjects, MATH-500 overlap removed (plan D19)."""
    spec = F.family(F.MATH_EXPANDED).loader
    root = snapshot_dir(hf_home, spec.repo_id, spec.revision)
    filters = {DROP_MISSING_FIELD: 0, DROP_NO_BOXED: 0, DROP_MATH500_OVERLAP: 0,
               DROP_DUPLICATE_ID: 0, "subjects": list(spec.parts)}
    rows: list[SourceRow] = []
    shas: list[str] = []
    n_raw = 0
    seen_text: set[str] = set()
    for subject in spec.parts:
        path = root / subject / F.HENDRYCKS_TEST_FILE
        if not path.exists():
            raise LockSourceMissing(f"{path} is not in the HF cache; the {F.MATH_EXPANDED} panel needs "
                            f"every declared subject (have: {sorted(p.name for p in root.glob('*'))})")
        shas.append(_sha256_file(path))
        raw = _parquet_rows(path)
        n_raw += len(raw)
        for index, row in enumerate(raw):
            problem, solution = row.get("problem"), row.get("solution")
            if not isinstance(problem, str) or not problem.strip() or not isinstance(solution, str):
                filters[DROP_MISSING_FIELD] += 1
                continue
            answer = extract_boxed(solution)
            if not answer or not answer.strip():
                filters[DROP_NO_BOXED] += 1
                continue
            text_hash = normalized_text_hash(problem)
            if text_hash in exclude:
                filters[DROP_MATH500_OVERLAP] += 1
                continue
            if text_hash in seen_text:
                filters[DROP_DUPLICATE_ID] += 1
                continue
            seen_text.add(text_hash)
            rows.append(SourceRow(family=F.MATH_EXPANDED, original_id=f"{subject}/test/{index}",
                                  question=problem, options=(),
                                  gold={"answer": answer.strip(), "solution": solution},
                                  subject=subject, level=_str_or_none(row.get("level"))))
    # one file sha for a multi-file panel: the hash of the ordered per-subject shas
    file_sha = identity.sha256_hex(identity.jcs(shas))
    return LoadedPanel(family=F.MATH_EXPANDED, rows=tuple(rows), file_path=str(root),
                       file_sha256=file_sha, n_rows_raw=n_raw, filters=filters,
                       n_overlap_removed=filters[DROP_MATH500_OVERLAP], option_count_histogram=None)


def _mmlu_pro(hf_home: str | os.PathLike) -> LoadedPanel:
    """ARD36's eligibility, kept by D19: exactly ten options, duplicate questions removed,
    the survivor being the smallest ``question_id`` so the choice never depends on iteration
    order (``ard/items_build.py:183-227``; parity test on the 9,742 count)."""
    spec = F.family(F.MMLU_PRO).loader
    raw, path = _hf_rows(spec)
    filters = {DROP_MISSING_FIELD: 0, DROP_OPTION_COUNT: 0, DROP_BAD_ANSWER: 0,
               DROP_DUPLICATE_QUESTION: 0, "required_options": F.MMLU_REQUIRED_OPTIONS}
    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for row in raw:
        question, options, answer_index = row.get("question"), row.get("options"), row.get("answer_index")
        if not isinstance(question, str) or not question.strip() or not isinstance(options, list):
            filters[DROP_MISSING_FIELD] += 1
            continue
        if len(options) != F.MMLU_REQUIRED_OPTIONS:
            filters[DROP_OPTION_COUNT] += 1
            continue
        if not isinstance(answer_index, int) or isinstance(answer_index, bool) \
                or not 0 <= answer_index < len(options):
            filters[DROP_BAD_ANSWER] += 1
            continue
        by_question.setdefault(question.strip(), []).append(row)
    rows: list[SourceRow] = []
    for group in by_question.values():
        group = sorted(group, key=lambda r: int(r["question_id"]))
        filters[DROP_DUPLICATE_QUESTION] += len(group) - 1
        row = group[0]
        rows.append(SourceRow(
            family=F.MMLU_PRO, original_id=f"mmlu_pro-{int(row['question_id'])}",
            question=str(row["question"]).strip(),
            options=tuple(str(option) for option in row["options"]),
            gold={"canonical_index": int(row["answer_index"]),
                  "source_answer": str(row.get("answer") or "")},
            subject=_str_or_none(row.get("category")), level=None))
    rows.sort(key=lambda r: r.original_id)
    return LoadedPanel(family=F.MMLU_PRO, rows=tuple(rows), file_path=str(path) if path else None,
                       file_sha256=_sha256_file(path) if path else None, n_rows_raw=len(raw),
                       filters=filters, n_overlap_removed=0,
                       option_count_histogram=_histogram(rows))


def _truthfulqa(hf_home: str | os.PathLike) -> LoadedPanel:
    """Legacy MC1 with the existing validator: ``len(labels) == len(choices)`` and exactly one
    ``1`` (``benchmarks/loaders.py:439-461``), option count inside the 4-13 window (spec 5.1)."""
    spec = F.family(F.TRUTHFULQA).loader
    raw, path = _hf_rows(spec)
    filters = {DROP_MISSING_FIELD: 0, DROP_OPTION_COUNT: 0, DROP_BAD_ANSWER: 0,
               "option_window": [F.TRUTHFULQA_MIN_OPTIONS, F.TRUTHFULQA_MAX_OPTIONS]}
    rows: list[SourceRow] = []
    for index, row in enumerate(raw):
        question, targets = row.get("question"), row.get("mc1_targets")
        if not isinstance(question, str) or not question.strip() or not isinstance(targets, Mapping):
            filters[DROP_MISSING_FIELD] += 1
            continue
        choices, labels = targets.get("choices"), targets.get("labels")
        if not isinstance(choices, list) or not isinstance(labels, list) \
                or len(labels) != len(choices) or any(not isinstance(c, str) for c in choices):
            filters[DROP_MISSING_FIELD] += 1
            continue
        if not F.TRUTHFULQA_MIN_OPTIONS <= len(choices) <= F.TRUTHFULQA_MAX_OPTIONS:
            filters[DROP_OPTION_COUNT] += 1
            continue
        if any(label not in (0, 1) or isinstance(label, bool) for label in labels) \
                or labels.count(1) != 1:
            filters[DROP_BAD_ANSWER] += 1
            continue
        rows.append(SourceRow(
            family=F.TRUTHFULQA, original_id=f"tqa-{index}", question=question.strip(),
            options=tuple(str(choice) for choice in choices),
            gold={"canonical_index": int(labels.index(1)), "labels": [int(v) for v in labels]},
            subject=None, level=None))
    return LoadedPanel(family=F.TRUTHFULQA, rows=tuple(rows), file_path=str(path) if path else None,
                       file_sha256=_sha256_file(path) if path else None, n_rows_raw=len(raw),
                       filters=filters, n_overlap_removed=0,
                       option_count_histogram=_histogram(rows))


def _bigcodebench(registry: "LegacyRegistry") -> LoadedPanel:
    """E0 only: the ``study_v4`` raw parquet, read through ``study/data/bcb.load_bcb_rows``."""
    spec = F.family(F.BIGCODEBENCH).loader
    path = registry.resolve(spec.local_path)
    if path is None or not Path(path).exists():
        return LoadedPanel(family=F.BIGCODEBENCH, rows=(), file_path=str(path) if path else None,
                           file_sha256=None, n_rows_raw=None,
                           filters={"status": "missing_source"}, n_overlap_removed=0,
                           option_count_histogram=None)
    from agents_scaling.study.data.bcb import load_bcb_rows

    raw = load_bcb_rows(path, expected_rows=spec.expected_rows)
    rows = tuple(SourceRow(family=F.BIGCODEBENCH, original_id=str(row["task_id"]),
                           question=str(row.get("complete_prompt") or ""), options=(),
                           gold={"task_id": str(row["task_id"])}, subject=None, level=None)
                 for row in raw)
    return LoadedPanel(family=F.BIGCODEBENCH, rows=rows, file_path=str(path),
                       file_sha256=_sha256_file(Path(path)), n_rows_raw=len(raw),
                       filters={"status": "e0_only"}, n_overlap_removed=0,
                       option_count_histogram=None)


def _mod97() -> LoadedPanel:
    """MOD97 problems are generated, not loaded: the "source" is the study seed plus the
    generator (WP5 ``nextdyn.mod97``), so the lock entry records that and nothing else."""
    return LoadedPanel(family=F.MOD97, rows=(), file_path=None, file_sha256=None, n_rows_raw=None,
                       filters={"status": "generated_at_dispatch",
                                "generator": "agents_scaling.nextdyn.mod97"},
                       n_overlap_removed=0, option_count_histogram=None)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_panel(family: str, hf_home: str | os.PathLike, *,
               exclude_text_hashes: frozenset[str] | None = None,
               registry: "LegacyRegistry | None" = None) -> LoadedPanel:
    """The filtered rows of one family plus its lock facts."""
    name = F.family(family).name
    if name == F.MATH500:
        return _math500(hf_home)
    if name == F.MATH_EXPANDED:
        exclude = exclude_text_hashes
        if exclude is None:
            exclude = frozenset(row.normalized_text_hash for row in _math500(hf_home).rows)
        return _math_expanded(hf_home, exclude)
    if name == F.MMLU_PRO:
        return _mmlu_pro(hf_home)
    if name == F.TRUTHFULQA:
        return _truthfulqa(hf_home)
    if name == F.BIGCODEBENCH:
        return _bigcodebench(registry or load_legacy_sources())
    return _mod97()


def load_rows(family: str, hf_home: str | os.PathLike) -> tuple[SourceRow, ...]:
    """The eligible rows of one family, in frozen order."""
    return load_panel(family, hf_home).rows


# --------------------------------------------------------------------------- legacy registry


@dataclass(frozen=True)
class LegacyRegistry:
    """The bounded named paths of ``legacy_sources.yaml`` this package may read.

    ``e0.registry`` (WP7) owns the file's values; ``datasets`` needs the three roots for the
    lock's ``legacy_roots`` and ``inventory`` needs ``prior_roots`` -- the roots without
    contract sidecars, whose items are ``unknown`` and never "untouched" (spec 5.2).
    """

    path: Path
    roots: Mapping[str, str]
    prior_roots: tuple[str, ...]

    def resolve(self, reference: str | None) -> Path | None:
        """``"<root>:<relative path>"`` -> an absolute path, or ``None`` when unknown."""
        if not reference:
            return None
        root, sep, relative = str(reference).partition(":")
        if not sep:
            return Path(reference)
        base = self.roots.get(root)
        return None if base is None else Path(base) / relative


def load_legacy_sources(path: str | os.PathLike | None = None) -> LegacyRegistry:
    source = Path(path) if path is not None else CONFIG_DIR / LEGACY_SOURCES_FILE
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    roots = {name: str(block["root"]) for name, block in data.items()
             if isinstance(block, Mapping) and block.get("root")}
    prior = tuple(str(p) for p in (data.get("prior_roots") or ()))
    return LegacyRegistry(path=source, roots=roots, prior_roots=prior)


def _legacy_root_entries(registry: LegacyRegistry) -> dict[str, LegacyRootEntry]:
    out: dict[str, LegacyRootEntry] = {}
    for name, relative in LEGACY_MANIFESTS.items():
        root = registry.roots.get(name)
        if root is None:
            out[name] = LegacyRootEntry(path=None, manifest_sha256=None)
            continue
        manifest = Path(root) / relative
        out[name] = LegacyRootEntry(
            path=root,
            manifest_sha256=_sha256_file(manifest) if manifest.exists() else None)
    return out


# --------------------------------------------------------------------------- the lock


def lock_datasets(cfg: StudyConfig, hf_home: str | os.PathLike) -> DatasetsLock:
    """Resolve every family against the offline cache and build ``datasets.lock.json``.

    Called by ``locks.lock_inputs_main`` (the one lifecycle step that may touch the network,
    on the login node).  The two inventory hashes stay ``null`` here: they are the identity
    of the *built* item table and are filled by the ``inventory`` verb
    (:func:`record_inventory_hashes`).
    """
    registry = load_legacy_sources(sibling_config(cfg.path, LEGACY_SOURCES_FILE)
                                   or CONFIG_DIR / LEGACY_SOURCES_FILE)
    entries: dict[str, DatasetLockEntry] = {}
    math500 = load_panel(F.MATH500, hf_home)
    exclude = frozenset(row.normalized_text_hash for row in math500.rows)
    panels = {F.MATH500: math500}
    for name in F.FAMILIES:
        if name in panels:
            continue
        panels[name] = load_panel(name, hf_home, exclude_text_hashes=exclude, registry=registry)
    for name, panel in panels.items():
        spec = F.family(name).loader
        entries[name] = DatasetLockEntry(
            repo_id=spec.repo_id, config_name=spec.config_name, split=spec.split,
            revision=spec.revision, file_path=panel.file_path, file_sha256=panel.file_sha256,
            n_rows=len(panel.rows) or None, row_content_sha256=panel.row_content_sha256,
            filters=dict(panel.filters), n_overlap_removed=panel.n_overlap_removed,
            option_count_histogram=panel.option_count_histogram)
    return DatasetsLock(
        schema_version=LOCK_SCHEMA_VERSION,
        locked_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        hf_home=str(hf_home), datasets=entries, legacy_roots=_legacy_root_entries(registry),
        source_inventory_hash=None, prior_use_inventory_hash=None)


def verify_lock(lock: DatasetsLock, hf_home: str | os.PathLike) -> list[str]:
    """Every mismatch between the lock and the cache on disk (empty list = verified)."""
    out: list[str] = []
    if lock.locked_at is None:
        return [f"datasets.lock.json is a skeleton (locked_at null); run lock-inputs"]
    for name in F.FAMILIES:
        entry = lock.datasets.get(name)
        if entry is None:
            out.append(f"datasets.lock.json has no entry for family {name!r}")
            continue
        spec = F.family(name).loader
        for field_name, expected in (("repo_id", spec.repo_id), ("config_name", spec.config_name),
                                     ("split", spec.split), ("revision", spec.revision)):
            if getattr(entry, field_name) != expected:
                out.append(f"{name}.{field_name} is {getattr(entry, field_name)!r}, "
                           f"families.py declares {expected!r}")
        if entry.file_sha256 is None:
            continue
        try:
            panel = load_panel(name, hf_home)
        except (LockError, FileNotFoundError, OSError) as exc:
            out.append(f"{name}: cannot re-read the source ({exc})")
            continue
        if panel.file_sha256 != entry.file_sha256:
            out.append(f"{name}: file sha256 drifted ({entry.file_sha256[:12]} -> "
                       f"{(panel.file_sha256 or '')[:12]})")
        if entry.row_content_sha256 and panel.row_content_sha256 != entry.row_content_sha256:
            out.append(f"{name}: row content drifted")
        if entry.n_rows is not None and len(panel.rows) != entry.n_rows:
            out.append(f"{name}: {len(panel.rows)} eligible rows, lock says {entry.n_rows}")
    return out


def record_inventory_hashes(lock: DatasetsLock, *, source_inventory_hash: str,
                            prior_use_inventory_hash: str) -> DatasetsLock:
    """The ``inventory`` verb's half of the lock (spec 17.4 ``locks.*_inventory_hash``)."""
    import dataclasses

    return dataclasses.replace(lock, source_inventory_hash=source_inventory_hash,
                               prior_use_inventory_hash=prior_use_inventory_hash)


__all__ = [
    "DROP_BAD_ANSWER", "DROP_DUPLICATE_ID", "DROP_DUPLICATE_QUESTION", "DROP_MATH500_OVERLAP",
    "DROP_MISSING_FIELD", "DROP_NO_BOXED", "DROP_OPTION_COUNT", "LEGACY_MANIFESTS",
    "LEGACY_SOURCES_FILE", "LegacyRegistry", "LoadedPanel", "SourceRow", "load_legacy_sources",
    "load_panel", "load_rows", "lock_datasets", "normalized_text_hash", "record_inventory_hashes",
    "snapshot_dir", "verify_lock",
]
