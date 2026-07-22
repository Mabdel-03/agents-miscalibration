"""Cache-compatibility contract for canonical benchmark loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from agents_scaling.benchmarks import loaders


METADATA_TYPE_ERROR = "must be called with a dataclass type or instance"


MMLU_ROWS = [
    {
        "question": "MMLU question zero",
        "options": ["zero-a", "zero-b", "zero-c", "zero-d"],
        "answer_index": 2,
    },
    {
        "question": "MMLU question one",
        "options": ["one-a", "one-b", "one-c", "one-d"],
        "answer_index": 0,
    },
    {
        "question": "MMLU question two",
        "options": ["two-a", "two-b", "two-c", "two-d"],
        "answer_index": 3,
    },
]


TRUTHFULQA_ROWS = [
    {
        "question": "Truthful question zero",
        "mc1_targets": {
            "choices": ["zero-a", "zero-b", "zero-c"],
            "labels": [0, 1, 0],
        },
    },
    {
        "question": "Truthful question one",
        "mc1_targets": {
            "choices": ["one-a", "one-b", "one-c", "one-d"],
            "labels": [0, 0, 0, 1],
        },
    },
    {
        "question": "Truthful question two",
        "mc1_targets": {
            "choices": ["two-a", "two-b"],
            "labels": [1, 0],
        },
    },
]


def _write_cache(
    hf_home: Path,
    benchmark: str,
    rows: list[dict],
    *,
    fingerprint: str | None = None,
    version: str = "0.0.0",
) -> Path:
    spec = loaders._CACHED_ARROW_SPECS[benchmark]
    if fingerprint is None:
        fingerprint = loaders.BENCHMARK_SOURCE_SPECS[benchmark].revision
    directory = (
        hf_home
        / "datasets"
        / spec.dataset_dir
        / spec.config_name
        / version
        / fingerprint
    )
    directory.mkdir(parents=True)
    arrow_path = directory / spec.filename
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    info = {
        "dataset_name": spec.dataset_name,
        "config_name": spec.config_name,
        # Deliberately retain an arbitrary feature payload.  The direct reader must not
        # ask ``datasets`` to decode this version-sensitive portion of DatasetInfo.
        "features": {"legacy": {"_type": "List", "feature": None}},
        "splits": {
            spec.split: {
                "name": spec.split,
                "num_examples": len(rows),
                "dataset_name": spec.dataset_name,
            }
        },
    }
    (directory / "dataset_info.json").write_text(json.dumps(info))
    return arrow_path


def _raise_metadata_type_error(*_args, **_kwargs):
    raise TypeError(METADATA_TYPE_ERROR)


@pytest.mark.parametrize(
    ("benchmark", "rows", "split", "qid_prefix"),
    [
        ("mmlu_pro", MMLU_ROWS, "test", "mmlupro-"),
        ("truthfulqa", TRUTHFULQA_ROWS, "validation", "tqa-"),
    ],
)
def test_cached_arrow_fallback_exactly_matches_canonical_normalization(
    monkeypatch,
    tmp_path,
    benchmark,
    rows,
    split,
    qid_prefix,
):
    """Direct Arrow and ordinary HF paths must produce identical Question objects."""

    monkeypatch.setattr(loaders, "_load_hf", lambda *_args, **_kwargs: {split: rows})
    canonical = loaders.load_benchmark(benchmark, n=2, seed=918)

    hf_home = tmp_path / "hf"
    arrow_path = _write_cache(hf_home, benchmark, rows)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)
    diagnostics = []
    recovered = loaders.load_benchmark(
        benchmark,
        n=2,
        seed=918,
        fallback_diagnostics=diagnostics,
    )

    assert recovered == canonical
    assert [question.qid for question in recovered] == [
        f"{qid_prefix}0",
        f"{qid_prefix}1",
    ]
    assert len(recovered) == 2
    for index, question in enumerate(recovered):
        original = rows[index]
        correct_text = (
            original["options"][original["answer_index"]]
            if benchmark == "mmlu_pro"
            else original["mc1_targets"]["choices"][
                original["mc1_targets"]["labels"].index(1)
            ]
        )
        assert question.options[ord(question.answer_key) - ord("A")] == correct_text
    assert diagnostics == [
        {
            "benchmark": benchmark,
            "seed": 918,
            "n": 2,
            "cached_arrow_path": str(arrow_path),
            "cached_arrow_sha256": hashlib.sha256(arrow_path.read_bytes()).hexdigest(),
            "source_revision": loaders.BENCHMARK_SOURCE_SPECS[benchmark].revision,
            "fallback_reason": METADATA_TYPE_ERROR,
            "normalization": (
                "shared canonical loader normalization with deterministic per-QID "
                "option shuffling"
            ),
        }
    ]


def test_supported_remote_loads_pin_full_immutable_revisions(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        repo = args[0]
        if repo == "Idavidrein/gpqa":
            return {
                "train": [
                    {
                        "Question": "GPQA question",
                        "Correct Answer": "yes",
                        "Incorrect Answer 1": "no",
                        "Incorrect Answer 2": "maybe",
                        "Incorrect Answer 3": "unknown",
                    }
                ]
            }
        if repo == "TIGER-Lab/MMLU-Pro":
            return {"test": MMLU_ROWS}
        if repo == "truthfulqa/truthful_qa":
            return {"validation": TRUTHFULQA_ROWS}
        if repo == "HuggingFaceH4/MATH-500":
            return {"test": [{"problem": "1+1", "answer": "2"}]}
        raise AssertionError(repo)

    monkeypatch.setattr(loaders, "_load_hf", fake_load)
    for benchmark in ("gpqa", "mmlu_pro", "truthfulqa", "math"):
        loaders.load_benchmark(benchmark, n=1, seed=0)

    assert len(calls) == 4
    for args, kwargs in calls:
        benchmark = {
            "Idavidrein/gpqa": "gpqa",
            "TIGER-Lab/MMLU-Pro": "mmlu_pro",
            "truthfulqa/truthful_qa": "truthfulqa",
            "HuggingFaceH4/MATH-500": "math",
        }[args[0]]
        revision = kwargs.get("revision")
        assert revision == loaders.BENCHMARK_SOURCE_SPECS[benchmark].revision
        assert len(revision) == 40


def test_hf_loader_enforces_shared_cache_only_and_exact_revision(
    monkeypatch, tmp_path
):
    import datasets

    revision = loaders.BENCHMARK_SOURCE_SPECS["gpqa"].revision
    hf_home = tmp_path / "shared-hf"
    arrow_path = (
        hf_home
        / "datasets"
        / "Idavidrein___gpqa"
        / "gpqa_diamond"
        / "0.0.0"
        / revision
        / "gpqa-train.arrow"
    )
    arrow_path.parent.mkdir(parents=True)
    arrow_path.write_bytes(b"cache identity only")

    class Split:
        cache_files = [{"filename": str(arrow_path)}]

    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"train": Split()}

    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    loaded = loaders._load_hf(
        "Idavidrein/gpqa", "gpqa_diamond", revision=revision
    )

    assert "train" in loaded
    args, kwargs = calls[0]
    assert args == ("Idavidrein/gpqa", "gpqa_diamond")
    assert kwargs["revision"] == revision
    assert kwargs["cache_dir"] == str(hf_home / "datasets")
    assert kwargs["download_config"].local_files_only is True
    assert kwargs["download_config"].max_retries == 0


def test_hf_loader_rejects_latest_cache_from_another_revision(monkeypatch, tmp_path):
    import datasets

    revision = loaders.BENCHMARK_SOURCE_SPECS["gpqa"].revision
    wrong_path = (
        tmp_path
        / "hf"
        / "datasets"
        / "Idavidrein___gpqa"
        / "gpqa_diamond"
        / "0.0.0"
        / ("0" * 40)
        / "gpqa-train.arrow"
    )

    class Split:
        cache_files = [{"filename": str(wrong_path)}]

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(datasets, "load_dataset", lambda *_args, **_kwargs: {"train": Split()})
    with pytest.raises(
        loaders.CachedBenchmarkLoadError, match="pinned source revision"
    ):
        loaders._load_hf(
            "Idavidrein/gpqa", "gpqa_diamond", revision=revision
        )


def test_gpqa_200_request_exhausts_exact_198_row_pinned_split(monkeypatch):
    rows = [
        {
            "Question": f"GPQA question {index}",
            "Correct Answer": "yes",
            "Incorrect Answer 1": "no",
            "Incorrect Answer 2": "maybe",
            "Incorrect Answer 3": "unknown",
        }
        for index in range(198)
    ]
    monkeypatch.setattr(loaders, "_load_hf", lambda *_args, **_kwargs: {"train": rows})
    questions = loaders.load_gpqa(n=200, seed=7)
    assert len(questions) == 198
    assert questions[0].qid == "gpqa-0"
    assert questions[-1].qid == "gpqa-197"


def test_cached_fallback_preserves_seeded_shuffle_and_full_n_contract(
    monkeypatch, tmp_path
):
    hf_home = tmp_path / "hf"
    _write_cache(hf_home, "mmlu_pro", MMLU_ROWS)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)

    seed_a = loaders.load_benchmark("mmlu_pro", n=None, seed=11)
    seed_a_repeat = loaders.load_benchmark("mmlu_pro", n=None, seed=11)
    seed_b = loaders.load_benchmark("mmlu_pro", n=None, seed=12)

    assert seed_a == seed_a_repeat
    assert len(seed_a) == len(MMLU_ROWS)
    assert [question.qid for question in seed_a] == [
        f"mmlupro-{index}" for index in range(len(MMLU_ROWS))
    ]
    assert any(
        first.options != second.options for first, second in zip(seed_a, seed_b)
    )


def test_fallback_is_exact_message_only(monkeypatch):
    expected = TypeError("wrapper: must be called with a dataclass type or instance")

    def fail(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(loaders, "_load_hf", fail)
    monkeypatch.setattr(
        loaders,
        "_read_cached_arrow",
        lambda *_args: pytest.fail("an unrelated TypeError must not inspect the cache"),
    )
    with pytest.raises(TypeError) as caught:
        loaders.load_benchmark("mmlu_pro", n=1, seed=0)
    assert caught.value is expected


def test_fallback_does_not_mask_typeerror_subclasses(monkeypatch):
    class ApplicationTypeError(TypeError):
        pass

    expected = ApplicationTypeError(METADATA_TYPE_ERROR)

    def fail(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(loaders, "_load_hf", fail)
    monkeypatch.setattr(
        loaders,
        "_read_cached_arrow",
        lambda *_args: pytest.fail("a TypeError subclass must not inspect the cache"),
    )
    with pytest.raises(ApplicationTypeError) as caught:
        loaders.load_benchmark("truthfulqa", n=1, seed=0)
    assert caught.value is expected


def test_fallback_is_not_applied_to_unsupported_benchmark(monkeypatch):
    expected = TypeError(METADATA_TYPE_ERROR)

    def fail(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(loaders, "_load_hf", fail)
    monkeypatch.setattr(
        loaders,
        "_read_cached_arrow",
        lambda *_args: pytest.fail("GPQA must not use another dataset's cache"),
    )
    with pytest.raises(TypeError) as caught:
        loaders.load_benchmark("gpqa", n=1, seed=0)
    assert caught.value is expected


def test_recognized_error_fails_clearly_when_exact_cache_is_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)
    with pytest.raises(
        loaders.CachedBenchmarkLoadError, match="cached dataset directory does not exist"
    ):
        loaders.load_benchmark("truthfulqa", n=1, seed=0)


def test_identical_cache_revisions_choose_lexicographically(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    expected = _write_cache(
        hf_home, "mmlu_pro", MMLU_ROWS, version="0.0.0"
    )
    _write_cache(hf_home, "mmlu_pro", MMLU_ROWS, version="1.0.0")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)
    diagnostics = []

    loaders.load_benchmark(
        "mmlu_pro", n=1, seed=0, fallback_diagnostics=diagnostics
    )

    assert diagnostics[0]["cached_arrow_path"] == str(expected)


def test_divergent_cache_revisions_are_rejected_as_ambiguous(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    _write_cache(hf_home, "mmlu_pro", MMLU_ROWS, version="0.0.0")
    divergent = [dict(row) for row in MMLU_ROWS]
    divergent[0] = {**divergent[0], "question": "different revision"}
    _write_cache(hf_home, "mmlu_pro", divergent, version="1.0.0")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)

    with pytest.raises(
        loaders.CachedBenchmarkLoadError,
        match="multiple non-identical cached Arrow revisions",
    ):
        loaders.load_benchmark("mmlu_pro", n=1, seed=0)


def test_fallback_rejects_an_unpinned_processed_cache(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    _write_cache(hf_home, "mmlu_pro", MMLU_ROWS, fingerprint="a" * 40)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)

    with pytest.raises(
        loaders.CachedBenchmarkLoadError, match="exact expected layout"
    ):
        loaders.load_benchmark("mmlu_pro", n=1, seed=0)


def test_cached_identity_and_row_count_are_validated(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf"
    path = _write_cache(hf_home, "truthfulqa", TRUTHFULQA_ROWS)
    info_path = path.parent / "dataset_info.json"
    info = json.loads(info_path.read_text())
    info["splits"]["validation"]["num_examples"] += 1
    info_path.write_text(json.dumps(info))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(loaders, "_load_hf", _raise_metadata_type_error)

    with pytest.raises(loaders.CachedBenchmarkLoadError, match="row-count mismatch"):
        loaders.load_benchmark("truthfulqa", n=1, seed=0)
