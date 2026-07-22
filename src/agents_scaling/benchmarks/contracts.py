"""Frozen normalized-question contracts for immutable sweep manifests.

An ``ExperimentCell`` names a benchmark, question count, and shuffle seed, but its
manifest entry does not contain the benchmark text or gold answers.  Those live in an
external dataset and can otherwise drift independently of ``cells.json``.  This module
provides two layers of identity:

* a canonical SHA-256 for every normalized :class:`~agents_scaling.benchmarks.schema.Question`;
* an order-sensitive contract SHA-256 for the exact Question sequence selected by one
  ``(benchmark, n_questions, seed)`` key.

Each run stores the unique contracts used by its frozen manifest in
``benchmark_contracts.v1.json``.  A separate checksum covers the exact sidecar bytes, and
the sidecar binds both the manifest SHA-256 and an ordered cell-to-contract index hash.
Loading or verification is fail-closed: malformed JSON, duplicate keys, a missing
checksum, manifest drift, source drift, QID reordering, or any Question-content change is
an explicit error rather than a new implicit benchmark revision.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from agents_scaling.benchmarks.loaders import (
    benchmark_source_provenance,
    expected_benchmark_question_count,
    load_benchmark,
)
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest


BENCHMARK_CONTRACT_SCHEMA_VERSION = 1
BENCHMARK_CONTRACTS_FILENAME = "benchmark_contracts.v1.json"
BENCHMARK_CONTRACTS_CHECKSUM_FILENAME = "benchmark_contracts.v1.sha256"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUESTION_HASH_DOMAIN = "agents_scaling.normalized_question.v1"
_ORDERED_CONTRACT_HASH_DOMAIN = "agents_scaling.ordered_question_contract.v1"
_CONTRACT_KEY_HASH_DOMAIN = "agents_scaling.benchmark_contract_key.v1"
_MANIFEST_INDEX_HASH_DOMAIN = "agents_scaling.manifest_contract_index.v1"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_filename",
        "manifest_sha256",
        "manifest_cell_count",
        "manifest_contract_index_sha256",
        "contracts",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "contract_id",
        "key",
        "source",
        "source_identity_sha256",
        "question_count",
        "ordered_qids",
        "question_sha256s",
        "question_contract_sha256",
    }
)
_KEY_FIELDS = frozenset({"benchmark", "n_questions", "seed"})
_SOURCE_FIELDS = frozenset({"repo_id", "config_name", "split", "revision"})


class BenchmarkContractError(RuntimeError):
    """A frozen benchmark contract is absent, malformed, or does not match."""


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONNumber(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise _NonFiniteJSONNumber(f"non-finite JSON number {value!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one contract value with the single registered canonical JSON form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json(value: Any) -> Any:
    """Recursively remove mutation aliases from a validated JSON value."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a detached ordinary-JSON copy of an immutable contract value."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _domain_hash(domain: str, payload: Any) -> str:
    return _sha256({"domain": domain, "payload": payload})


def canonical_question_payload(question: Question) -> dict[str, Any]:
    """Return the complete normalized scientific content of ``question``.

    The payload deliberately includes prompt text, ordered option text, gold answer, and
    answer type.  QID and answer-key validation alone cannot detect an upstream wording
    change or an index-preserving dataset reorder.
    """

    if not isinstance(question, Question):
        raise BenchmarkContractError("question contract requires Question instances")
    if not isinstance(question.qid, str) or not question.qid:
        raise BenchmarkContractError("question qid must be non-empty text")
    if not isinstance(question.benchmark, str) or not question.benchmark:
        raise BenchmarkContractError("question benchmark must be non-empty text")
    if not isinstance(question.prompt_stem, str) or not question.prompt_stem:
        raise BenchmarkContractError(
            f"question {question.qid!r} prompt_stem must be non-empty text"
        )
    if not isinstance(question.answer_key, str) or not question.answer_key:
        raise BenchmarkContractError(
            f"question {question.qid!r} answer_key must be non-empty text"
        )
    if not isinstance(question.answer_type, AnswerType):
        raise BenchmarkContractError(
            f"question {question.qid!r} has an unregistered answer_type"
        )
    if not isinstance(question.options, list) or any(
        not isinstance(option, str) or not option for option in question.options
    ):
        raise BenchmarkContractError(
            f"question {question.qid!r} options must be non-empty text values"
        )
    if question.answer_type is AnswerType.MCQ:
        if not question.options:
            raise BenchmarkContractError(
                f"MCQ question {question.qid!r} must contain options"
            )
        if question.answer_key not in question.option_letters:
            raise BenchmarkContractError(
                f"MCQ question {question.qid!r} answer_key is outside its options"
            )
    elif question.options:
        raise BenchmarkContractError(
            f"numeric question {question.qid!r} must not contain MCQ options"
        )
    return {
        "qid": question.qid,
        "benchmark": question.benchmark,
        "prompt_stem": question.prompt_stem,
        "options": list(question.options),
        "answer_key": question.answer_key,
        "answer_type": question.answer_type.value,
    }


def canonical_question_sha256(question: Question) -> str:
    """Hash the complete normalized Question content with domain separation."""

    return _domain_hash(_QUESTION_HASH_DOMAIN, canonical_question_payload(question))


@dataclass(frozen=True)
class BenchmarkContractKey:
    benchmark: str
    n_questions: int | None
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark:
            raise BenchmarkContractError("contract benchmark must be non-empty text")
        if self.n_questions is not None and (
            not isinstance(self.n_questions, int)
            or isinstance(self.n_questions, bool)
            or self.n_questions < 1
        ):
            raise BenchmarkContractError(
                "contract n_questions must be null or a positive integer"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise BenchmarkContractError("contract seed must be an integer")

    @classmethod
    def from_cell(cls, cell: ExperimentCell) -> "BenchmarkContractKey":
        return cls(cell.benchmark, cell.n_questions, cell.seed)

    @classmethod
    def from_dict(cls, value: Any) -> "BenchmarkContractKey":
        if not isinstance(value, dict) or set(value) != _KEY_FIELDS:
            raise BenchmarkContractError("benchmark contract key has the wrong fields")
        return cls(
            benchmark=value["benchmark"],
            n_questions=value["n_questions"],
            seed=value["seed"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "n_questions": self.n_questions,
            "seed": self.seed,
        }

    @property
    def contract_id(self) -> str:
        return _domain_hash(_CONTRACT_KEY_HASH_DOMAIN, self.to_dict())

    @property
    def sort_key(self) -> tuple[str, int, int]:
        return (
            self.benchmark,
            -1 if self.n_questions is None else self.n_questions,
            self.seed,
        )


def _validate_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_FIELDS:
        raise BenchmarkContractError("benchmark source identity has the wrong fields")
    for field in ("repo_id", "split", "revision"):
        if not isinstance(value[field], str) or not value[field]:
            raise BenchmarkContractError(
                f"benchmark source {field} must be non-empty text"
            )
    config_name = value["config_name"]
    if config_name is not None and (
        not isinstance(config_name, str) or not config_name
    ):
        raise BenchmarkContractError(
            "benchmark source config_name must be non-empty text or null"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise BenchmarkContractError(
            "benchmark source revision must be a full lowercase commit id"
        )
    return dict(value)


def _ordered_contract_sha256(
    key: BenchmarkContractKey,
    ordered_qids: Sequence[str],
    question_sha256s: Sequence[str],
) -> str:
    ordered = [
        {"qid": qid, "question_sha256": digest}
        for qid, digest in zip(ordered_qids, question_sha256s, strict=True)
    ]
    return _domain_hash(
        _ORDERED_CONTRACT_HASH_DOMAIN,
        {"key": key.to_dict(), "ordered_questions": ordered},
    )


def build_question_contract(
    key: BenchmarkContractKey,
    questions: Iterable[Question],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-stable, order-sensitive normalized Question contract."""

    retained = tuple(questions)
    if not retained:
        raise BenchmarkContractError(f"benchmark contract {key.to_dict()} is empty")
    if key.n_questions is not None and len(retained) > key.n_questions:
        raise BenchmarkContractError(
            f"benchmark contract {key.to_dict()} loaded {len(retained)} questions; "
            f"requested at most {key.n_questions}"
        )
    expected_count = expected_benchmark_question_count(
        key.benchmark, key.n_questions
    )
    if len(retained) != expected_count:
        raise BenchmarkContractError(
            f"benchmark contract {key.to_dict()} loaded {len(retained)} questions; "
            f"the pinned split requires exactly {expected_count}"
        )
    payloads = [canonical_question_payload(question) for question in retained]
    if any(payload["benchmark"] != key.benchmark for payload in payloads):
        raise BenchmarkContractError(
            f"benchmark contract {key.to_dict()} contains a different benchmark"
        )
    ordered_qids = [payload["qid"] for payload in payloads]
    if len(ordered_qids) != len(set(ordered_qids)):
        raise BenchmarkContractError(
            f"benchmark contract {key.to_dict()} contains duplicate QIDs"
        )
    question_sha256s = [
        _domain_hash(_QUESTION_HASH_DOMAIN, payload) for payload in payloads
    ]
    source_payload = _validate_source(
        dict(source)
        if source is not None
        else benchmark_source_provenance(key.benchmark)
    )
    registered_source = benchmark_source_provenance(key.benchmark)
    if source_payload != registered_source:
        raise BenchmarkContractError(
            f"benchmark source implementation drift for {key.benchmark!r}: "
            f"registered {registered_source}, observed {source_payload}"
        )
    return {
        "contract_id": key.contract_id,
        "key": key.to_dict(),
        "source": source_payload,
        "source_identity_sha256": _sha256(source_payload),
        "question_count": len(retained),
        "ordered_qids": ordered_qids,
        "question_sha256s": question_sha256s,
        "question_contract_sha256": _ordered_contract_sha256(
            key, ordered_qids, question_sha256s
        ),
    }


def _manifest_contract_assignments(
    snapshot: ManifestSnapshot,
) -> list[dict[str, str]]:
    return [
        {
            "cell_id": cell.cell_id,
            "contract_id": BenchmarkContractKey.from_cell(cell).contract_id,
        }
        for cell in snapshot.cells
    ]


def manifest_contract_index_sha256(snapshot: ManifestSnapshot) -> str:
    """Hash the manifest-ordered mapping from every cell to its contract key."""

    return _domain_hash(
        _MANIFEST_INDEX_HASH_DOMAIN,
        _manifest_contract_assignments(snapshot),
    )


BenchmarkLoader = Callable[..., list[Question]]


def build_run_benchmark_contracts(
    snapshot: ManifestSnapshot,
    *,
    benchmark_loader: BenchmarkLoader = load_benchmark,
) -> dict[str, Any]:
    """Materialize every unique benchmark contract used by ``snapshot`` once."""

    keys = {
        BenchmarkContractKey.from_cell(cell)
        for cell in snapshot.cells
    }
    contracts = []
    for key in sorted(keys, key=lambda item: item.sort_key):
        questions = benchmark_loader(
            key.benchmark,
            n=key.n_questions,
            seed=key.seed,
        )
        contracts.append(build_question_contract(key, questions))
    payload = {
        "schema_version": BENCHMARK_CONTRACT_SCHEMA_VERSION,
        "manifest_filename": snapshot.path.name,
        "manifest_sha256": snapshot.sha256,
        "manifest_cell_count": len(snapshot.cells),
        "manifest_contract_index_sha256": manifest_contract_index_sha256(snapshot),
        "contracts": contracts,
    }
    # Exercise strict JSON serialization before anything is published.
    canonical_json_bytes(payload)
    return payload


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BenchmarkContractError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_contract_entry(value: Any) -> tuple[BenchmarkContractKey, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
        raise BenchmarkContractError("benchmark contract entry has the wrong fields")
    key = BenchmarkContractKey.from_dict(value["key"])
    if value["contract_id"] != key.contract_id:
        raise BenchmarkContractError("benchmark contract_id does not match its key")
    source = _validate_source(value["source"])
    registered_source = benchmark_source_provenance(key.benchmark)
    if source != registered_source:
        raise BenchmarkContractError(
            f"benchmark source implementation drift for {key.benchmark!r}: frozen "
            f"{source}, registered {registered_source}"
        )
    if value["source_identity_sha256"] != _sha256(source):
        raise BenchmarkContractError(
            "benchmark source_identity_sha256 does not match its source"
        )
    count = value["question_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise BenchmarkContractError("benchmark question_count must be positive")
    if key.n_questions is not None and count > key.n_questions:
        raise BenchmarkContractError(
            "benchmark question_count exceeds its requested contract limit"
        )
    expected_count = expected_benchmark_question_count(
        key.benchmark, key.n_questions
    )
    if count != expected_count:
        raise BenchmarkContractError(
            "benchmark question_count does not match the pinned split request: "
            f"expected {expected_count}, got {count}"
        )
    qids = value["ordered_qids"]
    digests = value["question_sha256s"]
    if (
        not isinstance(qids, list)
        or len(qids) != count
        or any(not isinstance(qid, str) or not qid for qid in qids)
        or len(qids) != len(set(qids))
    ):
        raise BenchmarkContractError("benchmark ordered_qids are invalid")
    if (
        not isinstance(digests, list)
        or len(digests) != count
        or any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in digests
        )
    ):
        raise BenchmarkContractError("benchmark question_sha256s are invalid")
    expected_contract_hash = _ordered_contract_sha256(key, qids, digests)
    if value["question_contract_sha256"] != expected_contract_hash:
        raise BenchmarkContractError(
            "question_contract_sha256 does not match ordered Question hashes"
        )
    return key, dict(value)


@dataclass(frozen=True)
class FrozenBenchmarkContracts:
    path: Path
    checksum_path: Path
    sidecar_sha256: str
    manifest_sha256: str
    manifest_contract_index_sha256: str
    contracts_by_id: Mapping[str, dict[str, Any]]

    def contract_for_cell(self, cell: ExperimentCell) -> dict[str, Any]:
        key = BenchmarkContractKey.from_cell(cell)
        try:
            # Never return aliases into the frozen in-memory scientific contract.
            return _thaw_json(self.contracts_by_id[key.contract_id])
        except KeyError as exc:
            raise BenchmarkContractError(
                f"no frozen benchmark contract for {key.to_dict()}"
            ) from exc

    def verify_questions(
        self,
        cell: ExperimentCell,
        questions: Iterable[Question],
    ) -> dict[str, Any]:
        """Verify exact normalized content and order for one cell, then return its entry."""

        expected = self.contract_for_cell(cell)
        observed = build_question_contract(
            BenchmarkContractKey.from_cell(cell),
            questions,
            source=expected["source"],
        )
        if observed != expected:
            raise BenchmarkContractError(
                f"normalized Question contract drift for {cell.cell_id}: expected "
                f"{expected['question_contract_sha256']}, observed "
                f"{observed['question_contract_sha256']}"
            )
        return expected


def verify_frozen_benchmark_questions(
    snapshot: ManifestSnapshot,
    frozen: FrozenBenchmarkContracts,
    *,
    benchmark_loader: BenchmarkLoader = load_benchmark,
) -> int:
    """Re-read and verify every unique normalized Question contract in a run."""

    representatives: dict[str, ExperimentCell] = {}
    for cell in snapshot.cells:
        key = BenchmarkContractKey.from_cell(cell)
        representatives.setdefault(key.contract_id, cell)
    for contract_id in sorted(representatives):
        cell = representatives[contract_id]
        questions = benchmark_loader(
            cell.benchmark,
            n=cell.n_questions,
            seed=cell.seed,
        )
        frozen.verify_questions(cell, questions)
    return len(representatives)


def _decode_sidecar(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJSONKey,
        _NonFiniteJSONNumber,
    ) as exc:
        raise BenchmarkContractError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{path} must contain one JSON object")
    return value


def _verified_manifest_snapshot(
    root: Path,
    supplied: ManifestSnapshot | None,
) -> ManifestSnapshot:
    """Load the on-disk frozen manifest and reject a stale/injected snapshot.

    ``snapshot`` is a performance hint, not an authority.  Trusting it without re-reading
    ``cells.json`` would allow a caller to verify a sidecar against stale in-memory cells
    after the manifest changed on disk.
    """

    cells_path = root / "cells.json"
    checksum_path = root / "cells.sha256"
    if cells_path.is_symlink() or not cells_path.is_file():
        raise BenchmarkContractError(
            f"frozen cells manifest is missing or not regular: {cells_path}"
        )
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise BenchmarkContractError(
            f"frozen cells checksum is missing or not regular: {checksum_path}"
        )
    try:
        checksum_fields = checksum_path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkContractError(f"cannot read {checksum_path}: {exc}") from exc
    if (
        len(checksum_fields) != 2
        or _SHA256_RE.fullmatch(checksum_fields[0]) is None
        or checksum_fields[1] != "cells.json"
    ):
        raise BenchmarkContractError(
            f"invalid frozen cells checksum record: {checksum_path}"
        )
    try:
        current = load_manifest(root)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(
            f"cannot verify frozen cells manifest in {root}: {exc}"
        ) from exc
    if supplied is None:
        return current
    try:
        same_path = supplied.path.resolve() == current.path.resolve()
    except OSError:
        same_path = False
    if (
        not same_path
        or supplied.sha256 != current.sha256
        or supplied.cells != current.cells
    ):
        raise BenchmarkContractError(
            "supplied manifest snapshot does not match the current frozen cells manifest"
        )
    return current


def _validate_sidecar_payload(
    value: Any,
    snapshot: ManifestSnapshot,
    *,
    path: Path,
    checksum_path: Path,
    sidecar_sha256: str,
) -> FrozenBenchmarkContracts:
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise BenchmarkContractError(f"{path} has the wrong root schema")
    if (
        not isinstance(value["schema_version"], int)
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != BENCHMARK_CONTRACT_SCHEMA_VERSION
    ):
        raise BenchmarkContractError(
            f"unsupported benchmark contract schema {value['schema_version']!r}"
        )
    if value["manifest_filename"] != snapshot.path.name:
        raise BenchmarkContractError(
            "benchmark contract manifest_filename does not match cells manifest"
        )
    if value["manifest_sha256"] != snapshot.sha256:
        raise BenchmarkContractError(
            "benchmark contract manifest_sha256 does not match frozen cells manifest"
        )
    if (
        not isinstance(value["manifest_cell_count"], int)
        or isinstance(value["manifest_cell_count"], bool)
        or value["manifest_cell_count"] != len(snapshot.cells)
    ):
        raise BenchmarkContractError(
            "benchmark contract manifest_cell_count does not match cells manifest"
        )
    expected_index_hash = manifest_contract_index_sha256(snapshot)
    if value["manifest_contract_index_sha256"] != expected_index_hash:
        raise BenchmarkContractError(
            "benchmark contract cell index does not match cells manifest"
        )
    contracts = value["contracts"]
    if not isinstance(contracts, list) or not contracts:
        raise BenchmarkContractError("benchmark contracts must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    sort_keys: list[tuple[str, int, int]] = []
    for entry in contracts:
        key, validated = _validate_contract_entry(entry)
        if key.contract_id in by_id:
            raise BenchmarkContractError("duplicate benchmark contract key")
        by_id[key.contract_id] = validated
        sort_keys.append(key.sort_key)
    if sort_keys != sorted(sort_keys):
        raise BenchmarkContractError("benchmark contracts are not canonically ordered")
    expected_ids = {
        BenchmarkContractKey.from_cell(cell).contract_id for cell in snapshot.cells
    }
    if set(by_id) != expected_ids:
        raise BenchmarkContractError(
            "benchmark contract keys do not exactly cover the cells manifest"
        )
    return FrozenBenchmarkContracts(
        path=path,
        checksum_path=checksum_path,
        sidecar_sha256=sidecar_sha256,
        manifest_sha256=snapshot.sha256,
        manifest_contract_index_sha256=expected_index_hash,
        contracts_by_id=MappingProxyType(
            {
                contract_id: _freeze_json(validated)
                for contract_id, validated in by_id.items()
            }
        ),
    )


def _read_recorded_checksum(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError(
            f"frozen benchmark contract checksum is missing or not regular: {path}"
        )
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkContractError(f"cannot read {path}: {exc}") from exc
    if len(fields) != 2 or fields[1] != BENCHMARK_CONTRACTS_FILENAME:
        raise BenchmarkContractError(f"invalid benchmark contract checksum record: {path}")
    return _validate_sha256(fields[0], "benchmark contract checksum")


def load_frozen_benchmark_contracts(
    run_root: str | Path,
    *,
    snapshot: ManifestSnapshot | None = None,
) -> FrozenBenchmarkContracts:
    """Load and fully verify one run's immutable benchmark-contract sidecar."""

    root = Path(run_root)
    manifest = _verified_manifest_snapshot(root, snapshot)
    path = root / BENCHMARK_CONTRACTS_FILENAME
    checksum_path = root / BENCHMARK_CONTRACTS_CHECKSUM_FILENAME
    if path.is_symlink() or not path.is_file():
        raise BenchmarkContractError(
            f"frozen benchmark contract sidecar is missing or not regular: {path}"
        )
    expected = _read_recorded_checksum(checksum_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BenchmarkContractError(f"cannot read {path}: {exc}") from exc
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise BenchmarkContractError(
            f"benchmark contract checksum mismatch for {path}: expected {expected}, "
            f"got {observed}"
        )
    value = _decode_sidecar(raw, path)
    return _validate_sidecar_payload(
        value,
        manifest,
        path=path,
        checksum_path=checksum_path,
        sidecar_sha256=observed,
    )


def freeze_benchmark_contracts(
    run_root: str | Path,
    *,
    snapshot: ManifestSnapshot | None = None,
    payload: Mapping[str, Any] | None = None,
    benchmark_loader: BenchmarkLoader = load_benchmark,
) -> FrozenBenchmarkContracts:
    """Create the sidecar once, or verify an already frozen identical run contract.

    ``cells.json`` and ``cells.sha256`` are read-only inputs.  The sidecar is published
    first and its checksum second.  If a hard kill lands between those two atomic writes,
    a later call validates the preserved sidecar against the manifest and writes only the
    missing checksum; it never regenerates or replaces existing contract content.
    """

    root = Path(run_root)
    manifest = _verified_manifest_snapshot(root, snapshot)
    path = root / BENCHMARK_CONTRACTS_FILENAME
    checksum_path = root / BENCHMARK_CONTRACTS_CHECKSUM_FILENAME
    if path.exists() and checksum_path.exists():
        return load_frozen_benchmark_contracts(root, snapshot=manifest)
    if checksum_path.exists() and not path.exists():
        raise BenchmarkContractError(
            f"benchmark checksum exists without its immutable sidecar: {checksum_path}"
        )

    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise BenchmarkContractError(
                f"benchmark contract path is not a regular file: {path}"
            )
        raw = path.read_bytes()
        value = _decode_sidecar(raw, path)
        observed = hashlib.sha256(raw).hexdigest()
        unsealed = _validate_sidecar_payload(
            value,
            manifest,
            path=path,
            checksum_path=checksum_path,
            sidecar_sha256=observed,
        )
        # A retained sidecar is only the first half of an interrupted publication.  Bind
        # its normalized hashes back to the current pinned cache before sealing it with a
        # checksum; self-consistent but fabricated hashes must never become authoritative.
        verify_frozen_benchmark_questions(
            manifest,
            unsealed,
            benchmark_loader=benchmark_loader,
        )
        io.atomic_write_text(
            checksum_path,
            f"{observed}  {BENCHMARK_CONTRACTS_FILENAME}\n",
        )
        return load_frozen_benchmark_contracts(root, snapshot=manifest)

    materialized = (
        build_run_benchmark_contracts(manifest, benchmark_loader=benchmark_loader)
        if payload is None
        else dict(payload)
    )
    # Validate a caller-supplied/precomputed payload before its first write.
    serialized = json.dumps(
        materialized,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    raw = serialized.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    _validate_sidecar_payload(
        materialized,
        manifest,
        path=path,
        checksum_path=checksum_path,
        sidecar_sha256=digest,
    )
    io.atomic_write_text(path, serialized)
    io.atomic_write_text(
        checksum_path,
        f"{digest}  {BENCHMARK_CONTRACTS_FILENAME}\n",
    )
    return load_frozen_benchmark_contracts(root, snapshot=manifest)


def load_verified_questions(
    run_root: str | Path,
    cell: ExperimentCell,
    *,
    snapshot: ManifestSnapshot | None = None,
    benchmark_loader: BenchmarkLoader = load_benchmark,
) -> tuple[tuple[Question, ...], dict[str, Any]]:
    """Load Questions and fail unless they match the run's frozen normalized contract."""

    frozen = load_frozen_benchmark_contracts(run_root, snapshot=snapshot)
    expected = frozen.contract_for_cell(cell)
    current_source = benchmark_source_provenance(cell.benchmark)
    if current_source != expected["source"]:
        raise BenchmarkContractError(
            f"benchmark source implementation drift for {cell.benchmark!r}: frozen "
            f"revision {expected['source']['revision']}, current revision "
            f"{current_source['revision']}"
        )
    questions = tuple(
        benchmark_loader(
            cell.benchmark,
            n=cell.n_questions,
            seed=cell.seed,
        )
    )
    return questions, frozen.verify_questions(cell, questions)
