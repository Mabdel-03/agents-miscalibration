"""Benchmark loaders -> normalized ``Question`` lists.

All loaders are deterministic given a seed (option order is shuffled with a per-question
RNG so the correct-answer position is not always 'A'). Datasets are pulled via the HF
``datasets`` library; weights/cache live under ``HF_HOME``.

Two historical cached datasets (MMLU-Pro and TruthfulQA) contain valid Arrow data but
metadata that particular ``datasets`` releases cannot deserialize or map back to the
requested configuration.  For those exact recognized errors, this module reads the
already cached Arrow split directly and runs it through the *same* normalization code as
the ordinary loader.  The fallback is intentionally cache-only and narrow; unrelated
exceptions and unsupported benchmarks are never masked.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence

from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import DEFAULT_HF_HOME


_INCOMPATIBLE_DATASET_INFO_TYPE_ERROR = (
    "must be called with a dataclass type or instance"
)
_INCOMPATIBLE_TRUTHFULQA_CONFIG_ERROR = (
    "BuilderConfig 'multiple_choice' not found. Available: ['default']"
)


class CachedBenchmarkLoadError(RuntimeError):
    """A recognized metadata incompatibility could not be recovered safely."""


@dataclass(frozen=True)
class _CachedArrowSpec:
    benchmark: str
    dataset_dir: str
    config_name: str
    dataset_name: str
    split: str
    filename: str
    required_columns: frozenset[str]


@dataclass(frozen=True)
class BenchmarkSourceSpec:
    """Immutable upstream identity for one benchmark split.

    Hugging Face repository default branches are mutable.  The revision is therefore a
    full commit id rather than a branch or tag, and every ordinary ``load_dataset`` call
    below supplies it explicitly.  Normalized Question hashes provide the stronger
    end-to-end contract; this source record preserves where those Questions came from.
    """

    repo_id: str
    config_name: str | None
    split: str
    revision: str
    question_count: int


BENCHMARK_SOURCE_SPECS: dict[str, BenchmarkSourceSpec] = {
    "gpqa": BenchmarkSourceSpec(
        repo_id="Idavidrein/gpqa",
        config_name="gpqa_diamond",
        split="train",
        revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        question_count=198,
    ),
    "mmlu_pro": BenchmarkSourceSpec(
        repo_id="TIGER-Lab/MMLU-Pro",
        config_name=None,
        split="test",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        question_count=12032,
    ),
    "math": BenchmarkSourceSpec(
        repo_id="HuggingFaceH4/MATH-500",
        config_name=None,
        split="test",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        question_count=500,
    ),
    "truthfulqa": BenchmarkSourceSpec(
        repo_id="truthfulqa/truthful_qa",
        config_name="multiple_choice",
        split="validation",
        revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        question_count=817,
    ),
}


def benchmark_source_provenance(name: str) -> dict[str, str | None]:
    """Return the public, JSON-stable source identity for ``name``."""

    try:
        spec = BENCHMARK_SOURCE_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(BENCHMARK_SOURCE_SPECS)}"
        ) from exc
    return {
        "repo_id": spec.repo_id,
        "config_name": spec.config_name,
        "split": spec.split,
        "revision": spec.revision,
    }


def expected_benchmark_question_count(name: str, n: int | None) -> int:
    """Return the exact normalized row count for a pinned split request.

    ``n`` is an upper bound.  This distinction matters for GPQA Diamond: the sweep asks
    for 200 questions, while the complete pinned split contains 198.  Any shorter result
    is a cache/loader failure, not another permissible interpretation of the contract.
    """

    try:
        split_count = BENCHMARK_SOURCE_SPECS[name].question_count
    except KeyError as exc:
        raise KeyError(
            f"unknown benchmark {name!r}; known: {sorted(BENCHMARK_SOURCE_SPECS)}"
        ) from exc
    if n is None:
        return split_count
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("benchmark n must be null or a positive integer")
    return min(n, split_count)


_CACHED_ARROW_SPECS = {
    "mmlu_pro": _CachedArrowSpec(
        benchmark="mmlu_pro",
        dataset_dir="TIGER-Lab___mmlu-pro",
        config_name="default",
        dataset_name="mmlu-pro",
        split="test",
        filename="mmlu-pro-test.arrow",
        required_columns=frozenset({"question", "options", "answer_index"}),
    ),
    "truthfulqa": _CachedArrowSpec(
        benchmark="truthfulqa",
        dataset_dir="truthfulqa___truthful_qa",
        config_name="multiple_choice",
        dataset_name="truthful_qa",
        split="validation",
        filename="truthful_qa-validation.arrow",
        required_columns=frozenset({"question", "mc1_targets"}),
    ),
}

def _shared_hf_home() -> Path:
    """Resolve the one cache root used by every benchmark source read."""

    return Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser().resolve()


def _dataset_cache_files(dataset: Any) -> list[Path]:
    """Extract every materialized Arrow path from a Dataset or DatasetDict."""

    splits = dataset.values() if isinstance(dataset, Mapping) else (dataset,)
    paths: list[Path] = []
    for split in splits:
        cache_files = getattr(split, "cache_files", None)
        if not isinstance(cache_files, list):
            raise CachedBenchmarkLoadError(
                "cache-only benchmark load returned no inspectable cache_files"
            )
        for record in cache_files:
            if not isinstance(record, dict) or not isinstance(
                record.get("filename"), str
            ):
                raise CachedBenchmarkLoadError(
                    "cache-only benchmark load returned malformed cache provenance"
                )
            paths.append(Path(record["filename"]).resolve())
    if not paths:
        raise CachedBenchmarkLoadError(
            "cache-only benchmark load returned no materialized Arrow files"
        )
    return paths


def _validate_loaded_cache_identity(
    dataset: Any,
    *,
    hf_home: Path,
    revision: str,
) -> None:
    """Reject the library's implicit "latest cached revision" fallback.

    In offline mode, ``datasets`` may report that it is using the *latest* processed
    cache even when a revision argument was supplied.  The registered caches for these
    four sources use the full source commit as their processed-cache fingerprint, so we
    additionally require every Arrow file to live below the shared ``HF_HOME`` and in an
    exact ``.../<revision>/<split>.arrow`` directory.
    """

    datasets_root = (hf_home / "datasets").resolve()
    for path in _dataset_cache_files(dataset):
        try:
            relative = path.relative_to(datasets_root)
        except ValueError as exc:
            raise CachedBenchmarkLoadError(
                f"benchmark cache escaped shared HF_HOME: {path}"
            ) from exc
        if revision not in relative.parts or path.parent.name != revision:
            raise CachedBenchmarkLoadError(
                "processed benchmark cache does not match the pinned source revision: "
                f"{path} (expected fingerprint {revision})"
            )


# Lazy import so `import agents_scaling` works without `datasets` installed (e.g. in CI).
def _load_hf(*args, **kwargs):
    """Load one full-commit source strictly from the shared local HF cache."""

    from datasets import DownloadConfig, load_dataset

    revision = kwargs.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CachedBenchmarkLoadError(
            "benchmark source loads require a full lowercase commit revision"
        )
    if "cache_dir" in kwargs or "download_config" in kwargs:
        raise CachedBenchmarkLoadError(
            "benchmark callers may not override the registered shared cache policy"
        )
    hf_home = _shared_hf_home()
    dataset = load_dataset(
        *args,
        **kwargs,
        cache_dir=str(hf_home / "datasets"),
        download_config=DownloadConfig(local_files_only=True, max_retries=0),
    )
    _validate_loaded_cache_identity(dataset, hf_home=hf_home, revision=revision)
    return dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root(spec: _CachedArrowSpec) -> Path:
    return (_shared_hf_home() / "datasets" / spec.dataset_dir / spec.config_name).resolve()


def _cached_arrow_candidates(spec: _CachedArrowSpec) -> tuple[Path, list[Path]]:
    """Return deterministic, contained candidates for one exact cached split.

    A Hugging Face generated cache has the layout
    ``<dataset>/<config>/<version>/<fingerprint>/<split>.arrow``.  Restricting the
    search to that exact shape prevents accidentally ingesting a similarly named file
    elsewhere in ``HF_HOME``.  Symlink files and candidates resolving outside the
    expected cache root are rejected.
    """

    root = _cache_root(spec)
    if not root.is_dir():
        raise CachedBenchmarkLoadError(
            f"cached dataset directory does not exist for {spec.dataset_name!r}: {root}"
        )

    candidates: list[Path] = []
    revision = BENCHMARK_SOURCE_SPECS[spec.benchmark].revision
    for candidate in root.glob(f"*/{revision}/{spec.filename}"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if (
            len(relative.parts) != 3
            or relative.name != spec.filename
            or relative.parent.name != revision
        ):
            continue
        candidates.append(resolved)

    candidates = sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())
    if not candidates:
        raise CachedBenchmarkLoadError(
            "cached benchmark Arrow split not found at the exact expected layout: "
            f"{root}/*/{revision}/{spec.filename}"
        )
    return root, candidates


def _select_cached_arrow(spec: _CachedArrowSpec) -> tuple[Path, str]:
    """Select one cache deterministically, refusing divergent cached revisions."""

    root, candidates = _cached_arrow_candidates(spec)
    hashes = {candidate: _sha256_file(candidate) for candidate in candidates}
    distinct_hashes = set(hashes.values())
    if len(distinct_hashes) != 1:
        details = ", ".join(
            f"{path.relative_to(root)}={digest}" for path, digest in hashes.items()
        )
        raise CachedBenchmarkLoadError(
            "multiple non-identical cached Arrow revisions are present; refusing an "
            f"arbitrary dataset choice ({details})"
        )
    selected = candidates[0]
    return selected, hashes[selected]


def _validate_cached_dataset_info(path: Path, spec: _CachedArrowSpec) -> int:
    """Validate stable cache identity fields without decoding version-sensitive features."""

    info_path = path.parent / "dataset_info.json"
    if info_path.is_symlink() or not info_path.is_file():
        raise CachedBenchmarkLoadError(
            f"cached Arrow split has no regular companion dataset_info.json: {path}"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CachedBenchmarkLoadError(
            f"cannot read cached dataset identity metadata {info_path}: {exc}"
        ) from exc
    if not isinstance(info, dict):
        raise CachedBenchmarkLoadError(
            f"cached dataset identity metadata is not an object: {info_path}"
        )

    expected_identity = {
        "dataset_name": spec.dataset_name,
        "config_name": spec.config_name,
    }
    for field, expected in expected_identity.items():
        if info.get(field) != expected:
            raise CachedBenchmarkLoadError(
                f"cached dataset identity mismatch in {info_path}: "
                f"{field}={info.get(field)!r}, expected {expected!r}"
            )
    split_info = info.get("splits", {}).get(spec.split)
    if not isinstance(split_info, dict) or split_info.get("name") != spec.split:
        raise CachedBenchmarkLoadError(
            f"cached metadata does not describe split {spec.split!r}: {info_path}"
        )
    expected_rows = split_info.get("num_examples")
    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise CachedBenchmarkLoadError(
            f"cached metadata has invalid num_examples for {spec.split!r}: {info_path}"
        )
    return expected_rows


def _read_cached_arrow(benchmark: str) -> tuple[list[dict[str, Any]], Path, str]:
    """Read a validated supported split without invoking ``datasets`` metadata parsing."""

    try:
        spec = _CACHED_ARROW_SPECS[benchmark]
    except KeyError as exc:  # defensive: callers should gate before reaching this helper
        raise CachedBenchmarkLoadError(
            f"no cache-metadata compatibility fallback exists for {benchmark!r}"
        ) from exc

    path, arrow_sha256 = _select_cached_arrow(spec)
    expected_rows = _validate_cached_dataset_info(path, spec)
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
            table = reader.read_all()
    except Exception as exc:
        raise CachedBenchmarkLoadError(
            f"cannot read cached Arrow split {path}: {type(exc).__name__}: {exc}"
        ) from exc

    missing_columns = spec.required_columns.difference(table.column_names)
    if missing_columns:
        raise CachedBenchmarkLoadError(
            f"cached Arrow split {path} lacks required columns {sorted(missing_columns)}"
        )
    if table.num_rows != expected_rows:
        raise CachedBenchmarkLoadError(
            f"cached Arrow row-count mismatch for {path}: read {table.num_rows}, "
            f"metadata declares {expected_rows}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != arrow_sha256:
        raise CachedBenchmarkLoadError(
            f"cached Arrow split changed while it was being read: {path}"
        )
    rows = table.to_pylist()
    _validate_cached_rows(benchmark, rows, path)
    return rows, path, arrow_sha256


def _validate_cached_rows(
    benchmark: str, rows: list[dict[str, Any]], path: Path
) -> None:
    """Reject malformed cache content before canonical question normalization."""

    for index, row in enumerate(rows):
        prefix = f"cached {benchmark} row {index} in {path}"
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise CachedBenchmarkLoadError(f"{prefix} has no string question")
        if benchmark == "mmlu_pro":
            options = row.get("options")
            answer_index = row.get("answer_index")
            if (
                not isinstance(options, list)
                or not options
                or len(options) > 26
                or any(not isinstance(option, str) for option in options)
            ):
                raise CachedBenchmarkLoadError(
                    f"{prefix} has invalid multiple-choice options"
                )
            if (
                not isinstance(answer_index, int)
                or isinstance(answer_index, bool)
                or not 0 <= answer_index < len(options)
            ):
                raise CachedBenchmarkLoadError(f"{prefix} has invalid answer_index")
        elif benchmark == "truthfulqa":
            targets = row.get("mc1_targets")
            if not isinstance(targets, dict):
                raise CachedBenchmarkLoadError(f"{prefix} has invalid mc1_targets")
            choices = targets.get("choices")
            labels = targets.get("labels")
            if (
                not isinstance(choices, list)
                or not choices
                or len(choices) > 26
                or any(not isinstance(choice, str) for choice in choices)
            ):
                raise CachedBenchmarkLoadError(f"{prefix} has invalid MC1 choices")
            if (
                not isinstance(labels, list)
                or len(labels) != len(choices)
                or any(
                    not isinstance(label, int)
                    or isinstance(label, bool)
                    or label not in (0, 1)
                    for label in labels
                )
                or labels.count(1) != 1
            ):
                raise CachedBenchmarkLoadError(
                    f"{prefix} must have exactly one valid MC1 answer label"
                )


def _load_supported_split(
    benchmark: str,
    *dataset_args: str,
    requested_n: int | None,
    seed: int,
    fallback_diagnostics: MutableSequence[dict[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Load a split normally, with one exact legacy-metadata compatibility branch."""

    spec = _CACHED_ARROW_SPECS[benchmark]
    source = BENCHMARK_SOURCE_SPECS[benchmark]
    try:
        dataset = _load_hf(*dataset_args, revision=source.revision)
    except (TypeError, ValueError) as exc:
        recognized = (
            type(exc) is TypeError
            and exc.args == (_INCOMPATIBLE_DATASET_INFO_TYPE_ERROR,)
        ) or (
            benchmark == "truthfulqa"
            and type(exc) is ValueError
            and exc.args == (_INCOMPATIBLE_TRUTHFULQA_CONFIG_ERROR,)
        )
        if not recognized:
            raise
        rows, path, arrow_sha256 = _read_cached_arrow(benchmark)
        if fallback_diagnostics is not None:
            fallback_diagnostics.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "n": requested_n,
                    "cached_arrow_path": str(path),
                    "cached_arrow_sha256": arrow_sha256,
                    "source_revision": source.revision,
                    "fallback_reason": str(exc),
                    "normalization": (
                        "shared canonical loader normalization with deterministic "
                        "per-QID option shuffling"
                    ),
                }
            )
        return rows
    return dataset[spec.split]


def _mcq(
    qid: str,
    bench: str,
    stem: str,
    options: list[str],
    correct_idx: int,
    rng: random.Random,
) -> Question:
    """Build an MCQ Question with options shuffled deterministically."""
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    new_correct = order.index(correct_idx)
    return Question(
        qid=qid,
        benchmark=bench,
        prompt_stem=stem,
        options=shuffled,
        answer_key=chr(ord("A") + new_correct),
        answer_type=AnswerType.MCQ,
    )


def load_gpqa(n: int | None = None, seed: int = 0, subset: str = "gpqa_diamond") -> list[Question]:
    source = BENCHMARK_SOURCE_SPECS["gpqa"]
    if subset != source.config_name:
        raise ValueError(
            f"GPQA subset {subset!r} is outside the frozen source contract "
            f"{source.config_name!r}"
        )
    ds = _load_hf(
        source.repo_id,
        subset,
        revision=source.revision,
    )[source.split]
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        opts = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        out.append(_mcq(f"gpqa-{i}", "gpqa", row["Question"], opts, correct_idx=0, rng=rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_mmlu_pro(
    n: int | None = None,
    seed: int = 0,
    *,
    fallback_diagnostics: MutableSequence[dict[str, Any]] | None = None,
) -> list[Question]:
    ds = _load_supported_split(
        "mmlu_pro",
        "TIGER-Lab/MMLU-Pro",
        requested_n=n,
        seed=seed,
        fallback_diagnostics=fallback_diagnostics,
    )
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        opts = list(row["options"])
        correct_idx = row["answer_index"]
        out.append(_mcq(f"mmlupro-{i}", "mmlu_pro", row["question"], opts, correct_idx, rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_truthfulqa(
    n: int | None = None,
    seed: int = 0,
    *,
    fallback_diagnostics: MutableSequence[dict[str, Any]] | None = None,
) -> list[Question]:
    """TruthfulQA MC1: exactly one correct target among several. Calibration-native."""
    # Use the namespaced repo id; the bare "truthful_qa" alias is rejected by newer
    # huggingface_hub (requires 'namespace/name').
    ds = _load_supported_split(
        "truthfulqa",
        "truthfulqa/truthful_qa",
        "multiple_choice",
        requested_n=n,
        seed=seed,
        fallback_diagnostics=fallback_diagnostics,
    )
    out: list[Question] = []
    for i, row in enumerate(ds):
        rng = random.Random(f"{seed}-{i}")
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 marks the single correct choice
        correct_idx = labels.index(1)
        out.append(_mcq(f"tqa-{i}", "truthfulqa", row["question"], choices, correct_idx, rng))
        if n is not None and len(out) >= n:
            break
    return out


def load_math(n: int | None = None, seed: int = 0) -> list[Question]:
    """MATH-500: free-form numeric, graded by final-answer match.

    Uses HuggingFaceH4/MATH-500 (parquet, no dataset script — the legacy
    hendrycks/competition_math script repo is no longer loadable). It has a clean
    ``answer`` column (the gold final answer); we fall back to the boxed value in
    ``solution`` if ``answer`` is missing.
    """
    from agents_scaling.benchmarks.grading import extract_boxed

    source = BENCHMARK_SOURCE_SPECS["math"]
    ds = _load_hf(source.repo_id, revision=source.revision)[source.split]
    out: list[Question] = []
    for i, row in enumerate(ds):
        gold = row.get("answer") or extract_boxed(row.get("solution", "")) or ""
        out.append(
            Question(
                qid=f"math-{i}",
                benchmark="math",
                prompt_stem=row["problem"],
                answer_key=gold,
                answer_type=AnswerType.NUMERIC,
            )
        )
        if n is not None and len(out) >= n:
            break
    return out


LOADERS = {
    "gpqa": load_gpqa,
    "mmlu_pro": load_mmlu_pro,
    "truthfulqa": load_truthfulqa,
    "math": load_math,
}


def load_benchmark(
    name: str,
    n: int | None = None,
    seed: int = 0,
    *,
    fallback_diagnostics: MutableSequence[dict[str, Any]] | None = None,
) -> list[Question]:
    """Load one benchmark, recording direct-Arrow recovery provenance if requested."""

    if name not in LOADERS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(LOADERS)}")
    if name in _CACHED_ARROW_SPECS:
        return LOADERS[name](
            n=n, seed=seed, fallback_diagnostics=fallback_diagnostics
        )
    return LOADERS[name](n=n, seed=seed)
