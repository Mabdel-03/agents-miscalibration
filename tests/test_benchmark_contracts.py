from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agents_scaling.benchmarks import contracts
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog
from agents_scaling.config import (
    ContextShareLevel,
    ExperimentCell,
    ReasoningLevel,
    Topology,
)
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest
from scripts import freeze_benchmark_contracts as freeze_contract_script
from scripts import init_run_manifest
from scripts.init_run_manifest import initialize


def _cell(*, seed: int = 0, n_questions: int = 2) -> ExperimentCell:
    return ExperimentCell(
        model_size="0.6B",
        context_share_level=ContextShareLevel.ARTIFACT_ONLY,
        prompt_complexity_level=0,
        reasoning_level=ReasoningLevel.OFF,
        topology=Topology.SINGLE_AGENT,
        benchmark="gpqa",
        n_agents=1,
        rounds=1,
        n_samples=1,
        temperature=0.0,
        n_questions=n_questions,
        seed=seed,
    )


def _question(
    index: int,
    *,
    benchmark: str = "gpqa",
    stem_suffix: str = "",
    answer_key: str = "A",
) -> Question:
    return Question(
        qid=f"{benchmark}-{index}",
        benchmark=benchmark,
        prompt_stem=f"Question {index}{stem_suffix}",
        options=[f"correct-{index}", f"wrong-{index}"],
        answer_key=answer_key,
        answer_type=AnswerType.MCQ,
    )


def _loader(name: str, n: int | None = None, seed: int = 0) -> list[Question]:
    del seed
    count = 3 if n is None else n
    return [_question(index, benchmark=name) for index in range(count)]


def _write_manifest(run_root: Path, cells: list[ExperimentCell]):
    run_root.mkdir(parents=True)
    (run_root / "cells.json").write_text(
        json.dumps([cell.to_dict() for cell in cells], indent=2) + "\n",
        encoding="utf-8",
    )
    freeze_manifest(run_root)
    return load_manifest(run_root)


def _rewrite_sidecar(run_root: Path, payload: dict) -> None:
    path = run_root / contracts.BENCHMARK_CONTRACTS_FILENAME
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    (run_root / contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME).write_text(
        f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def _rewrite_raw_sidecar(run_root: Path, raw: bytes) -> None:
    path = run_root / contracts.BENCHMARK_CONTRACTS_FILENAME
    path.write_bytes(raw)
    (run_root / contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME).write_text(
        f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def test_question_hash_binds_prompt_options_gold_and_answer_type():
    baseline = _question(0)
    baseline_hash = contracts.canonical_question_sha256(baseline)
    assert baseline_hash == contracts.canonical_question_sha256(_question(0))

    variants = [
        _question(0, stem_suffix=" changed"),
        Question(
            qid=baseline.qid,
            benchmark=baseline.benchmark,
            prompt_stem=baseline.prompt_stem,
            options=["different", baseline.options[1]],
            answer_key=baseline.answer_key,
            answer_type=baseline.answer_type,
        ),
        _question(0, answer_key="B"),
        Question(
            qid=baseline.qid,
            benchmark=baseline.benchmark,
            prompt_stem=baseline.prompt_stem,
            options=[],
            answer_key="1",
            answer_type=AnswerType.NUMERIC,
        ),
    ]
    assert all(
        contracts.canonical_question_sha256(variant) != baseline_hash
        for variant in variants
    )


def test_ordered_contract_hash_rejects_index_preserving_content_and_order_drift():
    key = contracts.BenchmarkContractKey("gpqa", 2, 7)
    questions = [_question(0), _question(1)]
    baseline = contracts.build_question_contract(key, questions)
    changed = contracts.build_question_contract(
        key, [_question(0, stem_suffix=" changed"), _question(1)]
    )
    reordered = contracts.build_question_contract(key, list(reversed(questions)))

    assert changed["ordered_qids"] == baseline["ordered_qids"]
    assert changed["question_contract_sha256"] != baseline["question_contract_sha256"]
    assert reordered["question_contract_sha256"] != baseline["question_contract_sha256"]


def test_requested_count_is_an_upper_bound_for_short_fixed_split():
    key = contracts.BenchmarkContractKey("gpqa", 200, 0)
    questions = [_question(index) for index in range(198)]
    entry = contracts.build_question_contract(key, questions)
    assert entry["question_count"] == 198

    with pytest.raises(contracts.BenchmarkContractError, match="exactly 198"):
        contracts.build_question_contract(
            key, [_question(index) for index in range(197)]
        )
    with pytest.raises(contracts.BenchmarkContractError, match="requested at most 200"):
        contracts.build_question_contract(
            key, [_question(index) for index in range(201)]
        )


def test_runtime_catalog_requires_sidecar_and_caches_verified_questions(tmp_path):
    cell = _cell()
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [cell])
    with pytest.raises(contracts.BenchmarkContractError, match="sidecar is missing"):
        VerifiedQuestionCatalog(run_root, snapshot=snapshot, benchmark_loader=_loader)

    contracts.freeze_benchmark_contracts(
        run_root,
        snapshot=snapshot,
        benchmark_loader=_loader,
    )
    calls = 0

    def counting_loader(name: str, n: int | None = None, seed: int = 0):
        nonlocal calls
        calls += 1
        return _loader(name, n=n, seed=seed)

    catalog = VerifiedQuestionCatalog(
        run_root,
        snapshot=snapshot,
        benchmark_loader=counting_loader,
    )
    first = catalog.questions_for(cell)
    second = catalog.questions_for(cell)
    assert first is second
    assert calls == 1


def test_frozen_sidecar_binds_manifest_and_never_modifies_cells_files(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell(seed=0), _cell(seed=1)])
    cells_before = (run_root / "cells.json").read_bytes()
    checksum_before = (run_root / "cells.sha256").read_bytes()

    frozen = contracts.freeze_benchmark_contracts(
        run_root,
        snapshot=snapshot,
        benchmark_loader=_loader,
    )
    loaded = contracts.load_frozen_benchmark_contracts(
        run_root, snapshot=snapshot
    )

    assert loaded.sidecar_sha256 == frozen.sidecar_sha256
    assert loaded.manifest_sha256 == snapshot.sha256
    assert len(loaded.contracts_by_id) == 2
    assert (run_root / "cells.json").read_bytes() == cells_before
    assert (run_root / "cells.sha256").read_bytes() == checksum_before
    assert (
        hashlib.sha256(frozen.path.read_bytes()).hexdigest()
        == frozen.sidecar_sha256
    )


def test_contract_for_cell_returns_detached_plain_json_without_mutating_frozen_state(
    tmp_path,
):
    run_root = tmp_path / "run"
    cell = _cell()
    snapshot = _write_manifest(run_root, [cell])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )

    first = frozen.contract_for_cell(cell)
    assert type(first) is dict
    assert type(first["source"]) is dict
    assert type(first["ordered_qids"]) is list
    first["source"]["revision"] = "0" * 40
    first["ordered_qids"][0] = "tampered"

    second = frozen.contract_for_cell(cell)
    assert second["source"]["revision"] != "0" * 40
    assert second["ordered_qids"][0] == "gpqa-0"
    frozen.verify_questions(cell, _loader("gpqa", n=2, seed=0))

    contract_id = contracts.BenchmarkContractKey.from_cell(cell).contract_id
    with pytest.raises(TypeError):
        frozen.contracts_by_id[contract_id]["question_count"] = 1


def test_sidecar_checksum_and_manifest_binding_fail_closed(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    frozen.path.write_text(frozen.path.read_text() + " ", encoding="utf-8")
    with pytest.raises(contracts.BenchmarkContractError, match="checksum mismatch"):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=snapshot)


@pytest.mark.parametrize(
    "checksum_bytes",
    [None, b"not-a-checksum\n", b"0" * 64 + b"  wrong-name.json\n"],
)
def test_contract_freeze_requires_a_strict_frozen_manifest_checksum(
    tmp_path, checksum_bytes
):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    cells_before = (run_root / "cells.json").read_bytes()
    checksum_path = run_root / "cells.sha256"
    if checksum_bytes is None:
        checksum_path.unlink()
        expected = "missing or not regular"
    else:
        checksum_path.write_bytes(checksum_bytes)
        expected = "invalid frozen cells checksum"

    with pytest.raises(contracts.BenchmarkContractError, match=expected):
        contracts.freeze_benchmark_contracts(
            run_root, snapshot=snapshot, benchmark_loader=_loader
        )
    assert (run_root / "cells.json").read_bytes() == cells_before
    assert not (run_root / contracts.BENCHMARK_CONTRACTS_FILENAME).exists()
    assert not (run_root / contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME).exists()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{", "cannot parse"),
        (b'{"schema_version":1,"schema_version":1}\n', "duplicate JSON key"),
        (b'{"schema_version":NaN}\n', "non-finite JSON number"),
        (b"{} trailing\n", "cannot parse"),
    ],
)
def test_malformed_duplicate_and_nonfinite_sidecars_fail_closed(
    tmp_path, raw, message
):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    _rewrite_raw_sidecar(run_root, raw)
    with pytest.raises(contracts.BenchmarkContractError, match=message):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=snapshot)


def test_internal_question_hash_drift_fails_even_with_recomputed_file_checksum(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    payload = json.loads(frozen.path.read_text())
    payload["contracts"][0]["question_sha256s"][0] = "0" * 64
    _rewrite_sidecar(run_root, payload)

    with pytest.raises(
        contracts.BenchmarkContractError, match="question_contract_sha256"
    ):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=snapshot)


def test_source_drift_fails_even_when_source_and_file_hashes_are_recomputed(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    payload = json.loads(frozen.path.read_text())
    entry = payload["contracts"][0]
    entry["source"]["revision"] = "0" * 40
    entry["source_identity_sha256"] = contracts._sha256(entry["source"])
    _rewrite_sidecar(run_root, payload)

    with pytest.raises(
        contracts.BenchmarkContractError, match="source implementation drift"
    ):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=snapshot)


def test_duplicate_contract_entries_fail_with_a_recomputed_file_checksum(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    payload = json.loads(frozen.path.read_text())
    payload["contracts"].append(payload["contracts"][0])
    _rewrite_sidecar(run_root, payload)

    with pytest.raises(contracts.BenchmarkContractError, match="duplicate benchmark"):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=snapshot)


def test_manifest_drift_rejects_both_stale_snapshot_and_current_manifest(tmp_path):
    run_root = tmp_path / "run"
    stale = _write_manifest(run_root, [_cell(seed=0)])
    contracts.freeze_benchmark_contracts(
        run_root, snapshot=stale, benchmark_loader=_loader
    )
    (run_root / "cells.json").write_text(
        json.dumps([_cell(seed=1).to_dict()], indent=2) + "\n",
        encoding="utf-8",
    )
    freeze_manifest(run_root, overwrite=True)

    with pytest.raises(
        contracts.BenchmarkContractError, match="supplied manifest snapshot"
    ):
        contracts.load_frozen_benchmark_contracts(run_root, snapshot=stale)
    with pytest.raises(
        contracts.BenchmarkContractError, match="manifest_sha256"
    ):
        contracts.load_frozen_benchmark_contracts(run_root)


def test_missing_checksum_recovery_preserves_existing_sidecar_bytes(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    original = frozen.path.read_bytes()
    frozen.checksum_path.unlink()

    recovered = contracts.freeze_benchmark_contracts(
        run_root,
        snapshot=snapshot,
        benchmark_loader=_loader,
    )
    assert recovered.path.read_bytes() == original
    assert recovered.sidecar_sha256 == hashlib.sha256(original).hexdigest()


def test_missing_checksum_recovery_does_not_seal_content_drift(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    original = frozen.path.read_bytes()
    frozen.checksum_path.unlink()

    def drifted(name: str, n: int | None = None, seed: int = 0):
        questions = _loader(name, n=n, seed=seed)
        questions[0] = _question(0, benchmark=name, stem_suffix=" changed")
        return questions

    with pytest.raises(
        contracts.BenchmarkContractError, match="normalized Question contract drift"
    ):
        contracts.freeze_benchmark_contracts(
            run_root,
            snapshot=snapshot,
            benchmark_loader=drifted,
        )
    assert frozen.path.read_bytes() == original
    assert not frozen.checksum_path.exists()


def test_checksum_without_sidecar_is_not_treated_as_recoverable(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    checksum = run_root / contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME
    checksum.write_text(
        f"{'0' * 64}  {contracts.BENCHMARK_CONTRACTS_FILENAME}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        contracts.BenchmarkContractError, match="checksum exists without"
    ):
        contracts.freeze_benchmark_contracts(
            run_root, snapshot=snapshot, benchmark_loader=_loader
        )


def test_malformed_retained_sidecar_is_never_regenerated_during_recovery(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell()])
    cells_before = (run_root / "cells.json").read_bytes()
    manifest_checksum_before = (run_root / "cells.sha256").read_bytes()
    sidecar = run_root / contracts.BENCHMARK_CONTRACTS_FILENAME
    sidecar.write_bytes(b"{malformed")

    with pytest.raises(contracts.BenchmarkContractError, match="cannot parse"):
        contracts.freeze_benchmark_contracts(
            run_root,
            snapshot=snapshot,
            benchmark_loader=lambda *_args, **_kwargs: pytest.fail(
                "retained sidecar must never be regenerated"
            ),
        )
    assert sidecar.read_bytes() == b"{malformed"
    assert not (run_root / contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME).exists()
    assert (run_root / "cells.json").read_bytes() == cells_before
    assert (run_root / "cells.sha256").read_bytes() == manifest_checksum_before


def test_verified_loader_rejects_same_qids_and_gold_with_changed_prompt(tmp_path):
    run_root = tmp_path / "run"
    cell = _cell()
    snapshot = _write_manifest(run_root, [cell])
    contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )

    def drifted(name: str, n: int | None = None, seed: int = 0):
        questions = _loader(name, n=n, seed=seed)
        questions[0] = _question(0, benchmark=name, stem_suffix=" revised")
        return questions

    with pytest.raises(
        contracts.BenchmarkContractError, match="normalized Question contract drift"
    ):
        contracts.load_verified_questions(
            run_root,
            cell,
            snapshot=snapshot,
            benchmark_loader=drifted,
        )


def test_freeze_script_verification_reloads_normalized_question_content(tmp_path):
    run_root = tmp_path / "run"
    snapshot = _write_manifest(run_root, [_cell(seed=0)])
    frozen = contracts.freeze_benchmark_contracts(
        run_root, snapshot=snapshot, benchmark_loader=_loader
    )
    calls = []

    def loader(name: str, n: int | None = None, seed: int = 0):
        calls.append((name, n, seed))
        questions = _loader(name, n=n, seed=seed)
        questions[0] = _question(
            0, benchmark=name, stem_suffix=" source drift"
        )
        return questions

    with pytest.raises(
        contracts.BenchmarkContractError, match="normalized Question contract drift"
    ):
        freeze_contract_script.verify_current_questions(
            snapshot, frozen, benchmark_loader=loader
        )
    assert calls == [("gpqa", 2, 0)]


def test_initializer_resolves_contracts_before_publishing_manifest(tmp_path):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        """
axes:
  model_size: [0.6B]
  topology: [single_agent]
  context_share_level: [artifact_only]
  prompt_complexity_level: [0]
  reasoning_level: ["off"]
  benchmark: [gpqa]
  seed: [0]
fixed:
  n_questions: 2
  n_samples: 1
include_sas_baselines: false
""".lstrip(),
        encoding="utf-8",
    )
    failed_root = tmp_path / "failed"

    def fail_loader(*_args, **_kwargs):
        raise RuntimeError("dataset unavailable")

    with pytest.raises(RuntimeError, match="dataset unavailable"):
        initialize(
            config=config,
            run_root=failed_root,
            expected_cells=1,
            benchmark_loader=fail_loader,
        )
    assert not (failed_root / "cells.json").exists()
    assert not (failed_root / "cells.sha256").exists()

    run_root = tmp_path / "complete"
    count, manifest_sha256 = initialize(
        config=config,
        run_root=run_root,
        expected_cells=1,
        benchmark_loader=_loader,
    )
    snapshot = load_manifest(run_root)
    frozen = contracts.load_frozen_benchmark_contracts(
        run_root, snapshot=snapshot
    )
    assert count == 1
    assert manifest_sha256 == snapshot.sha256 == frozen.manifest_sha256
    assert len(frozen.contracts_by_id) == 1


def test_initializer_publication_is_atomic_and_retryable_after_rename_failure(
    monkeypatch, tmp_path
):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        """
axes:
  model_size: [0.6B]
  topology: [single_agent]
  context_share_level: [artifact_only]
  prompt_complexity_level: [0]
  reasoning_level: ["off"]
  benchmark: [gpqa]
  seed: [0]
fixed:
  n_questions: 2
  n_samples: 1
include_sas_baselines: false
""".lstrip(),
        encoding="utf-8",
    )
    run_root = tmp_path / "atomic-run"
    real_rename = init_run_manifest.os.rename

    def fail_publication(source, destination):
        if Path(destination) == run_root:
            raise OSError("simulated publication crash")
        return real_rename(source, destination)

    monkeypatch.setattr(init_run_manifest.os, "rename", fail_publication)
    with pytest.raises(OSError, match="simulated publication crash"):
        initialize(
            config=config,
            run_root=run_root,
            expected_cells=1,
            benchmark_loader=_loader,
    )
    assert not run_root.exists()
    assert not [
        path for path in tmp_path.glob(".atomic-run.initialize.*") if path.is_dir()
    ]

    monkeypatch.setattr(init_run_manifest.os, "rename", real_rename)
    initialize(
        config=config,
        run_root=run_root,
        expected_cells=1,
        benchmark_loader=_loader,
    )
    contracts.load_frozen_benchmark_contracts(run_root)


def test_initializer_refuses_complete_run_without_reading_or_mutating_it(tmp_path):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        """
axes:
  model_size: [0.6B]
  topology: [single_agent]
  context_share_level: [artifact_only]
  prompt_complexity_level: [0]
  reasoning_level: ["off"]
  benchmark: [gpqa]
  seed: [0]
fixed:
  n_questions: 2
  n_samples: 1
include_sas_baselines: false
""".lstrip(),
        encoding="utf-8",
    )
    run_root = tmp_path / "immutable-run"
    initialize(
        config=config,
        run_root=run_root,
        expected_cells=1,
        benchmark_loader=_loader,
    )
    protected = {
        name: (run_root / name).read_bytes()
        for name in (
            "cells.json",
            "cells.sha256",
            contracts.BENCHMARK_CONTRACTS_FILENAME,
            contracts.BENCHMARK_CONTRACTS_CHECKSUM_FILENAME,
        )
    }

    with pytest.raises(FileExistsError, match="immutable run"):
        initialize(
            config=config,
            run_root=run_root,
            expected_cells=1,
            benchmark_loader=lambda *_args, **_kwargs: pytest.fail(
                "an existing run must be rejected before dataset loading"
            ),
        )
    assert {
        name: (run_root / name).read_bytes() for name in protected
    } == protected
