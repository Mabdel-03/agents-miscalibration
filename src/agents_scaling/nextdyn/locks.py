"""Lock files: ``models.lock.json``, ``datasets.lock.json``, ``prompts.lock.json`` (spec 5.4,
17.1, 17.5 ``lock-inputs``; plan §2.1).

Confirmation may not use ``main``, ``latest`` or an unrecorded default (spec 5.4), so every
mutable upstream reference is resolved **once** into a lock whose fields are null until the
``lock-inputs`` verb fills them; the compiler treats a null as a launch blocker.

The **pin** of a lock file is ``study.yaml``'s ``locks.<name>_lock_hash`` (spec 17.4):
``config.lock_hash_checks`` compares every non-null one to sha256 of the lock bytes beside
``study.yaml`` and refuses a stale pin.  The ``.sha256`` sidecar this module writes next to
each lock is a convenience for humans reading the directory; the loader checks it when
present (a drifted sidecar is a :class:`LockError`, not a warning) but it never substitutes
for the study.yaml hash -- a regenerated sidecar cannot launder a tampered lock.

``verify_models_lock`` fails closed: for every enabled model it recomputes the ``config.json``
sha256, the chat-template sha256 (the ``chat_template`` string inside
``tokenizer_config.json``, or ``chat_template.jinja`` when a snapshot ships one) and reads
each safetensors shard's sha256 off the content-addressed HF blob name
(``snapshots/<rev>/<shard> -> ../../blobs/<sha256>``, verified on both pinned snapshots)
under ``$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>/``.  The pattern is
``serving/model_contracts.py:verify_identity`` (fail closed on drift), re-keyed to nextdyn's
model cells.

``lock_inputs_main`` is the verb: models and prompts are locked here; datasets are locked by
``agents_scaling.nextdyn.datasets.lock_datasets`` (WP3) when importable and reported
``blocked`` otherwise.  It is the only lifecycle step that may download anything, and it
runs on the login node.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.config import DEFAULT_HF_HOME
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.cli import Report, common_parser
from agents_scaling.nextdyn.config import (
    ComputeConfig,
    StudyConfig,
    build,
    is_sha256,
    load_compute,
    load_study,
    lock_paths,
)
from agents_scaling.nextdyn.deps import atomic_write_bytes
from agents_scaling.nextdyn.errors import LockError
from agents_scaling.study import identity

LOCK_SCHEMA_VERSION: Final[int] = 1
HF_HOME_ENV: Final[str] = "HF_HOME"
MODEL_STATUS_LOCKED: Final[str] = "locked"
MODEL_STATUS_BLOCKED: Final[str] = "blocked"
TEMPLATES_DIRNAME: Final[str] = "templates"
CONTINUATION_PREFILL_ATTR: Final[str] = "CONTINUATION_PREFILL_TEXT"


# --------------------------------------------------------------------------- lock shapes


@dataclass(frozen=True)
class ModelLockEntry:
    """One model cell's pins.  ``status`` is ``locked`` (every field non-null and verified) or
    ``blocked`` (weights absent / gated; ``blocked_reason`` says why)."""

    status: str
    blocked_reason: str | None
    hf_id: str
    revision: str | None
    tokenizer_revision: str | None
    adapter: str
    config_sha256: str | None
    chat_template_sha256: str | None
    safetensors: Mapping[str, str] | None
    n_layers: int | None
    dtype: str | None
    reasoning_parser: str | None


@dataclass(frozen=True)
class ModelsLock:
    schema_version: int
    locked_at: str | None
    hf_home: str | None
    vllm_version: str | None
    torch_version: str | None
    cuda_version: str | None
    models: Mapping[str, ModelLockEntry]


@dataclass(frozen=True)
class DatasetLockEntry:
    repo_id: str | None
    config_name: str | None
    split: str | None
    revision: str | None
    file_path: str | None
    file_sha256: str | None
    n_rows: int | None
    row_content_sha256: str | None
    filters: Mapping[str, Any] | None
    n_overlap_removed: int | None
    option_count_histogram: Mapping[str, int] | None


@dataclass(frozen=True)
class LegacyRootEntry:
    path: str | None
    manifest_sha256: str | None


@dataclass(frozen=True)
class DatasetsLock:
    schema_version: int
    locked_at: str | None
    hf_home: str | None
    datasets: Mapping[str, DatasetLockEntry]
    legacy_roots: Mapping[str, LegacyRootEntry]
    source_inventory_hash: str | None
    prior_use_inventory_hash: str | None


@dataclass(frozen=True)
class PromptsLock:
    schema_version: int
    locked_at: str | None
    templates: Mapping[str, str]
    continuation_prefill_text: str | None
    continuation_prefill_sha256: str | None


# --------------------------------------------------------------------------- io


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _read_locked_json(path: Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.exists():
        raise LockError(f"{path} does not exist; run lock-inputs")
    raw = path.read_bytes()
    sidecar = _sidecar(path)
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        actual = identity.sha256_hex(raw)
        if expected != actual:
            raise LockError(f"{path}: sidecar sha256 {expected[:12]} != bytes {actual[:12]} (drift)")
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise LockError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, Mapping):
        raise LockError(f"{path}: top level must be an object")
    return data


def _write_locked_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    sha = identity.sha256_hex(raw)
    atomic_write_bytes(Path(path), raw)
    atomic_write_bytes(_sidecar(Path(path)), f"{sha}  {Path(path).name}\n".encode("utf-8"))
    return sha


def _to_jsonable(obj: Any) -> Any:
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def load_models_lock(path: str | os.PathLike) -> ModelsLock:
    data = _read_locked_json(Path(path))
    lock = build(ModelsLock, data, "models.lock")
    if lock.schema_version != LOCK_SCHEMA_VERSION:
        raise LockError(f"{path}: schema_version {lock.schema_version} != {LOCK_SCHEMA_VERSION}")
    return lock


def write_models_lock(path: str | os.PathLike, lock: ModelsLock) -> str:
    return _write_locked_json(Path(path), _to_jsonable(lock))


def load_datasets_lock(path: str | os.PathLike) -> DatasetsLock:
    data = _read_locked_json(Path(path))
    lock = build(DatasetsLock, data, "datasets.lock")
    if lock.schema_version != LOCK_SCHEMA_VERSION:
        raise LockError(f"{path}: schema_version {lock.schema_version} != {LOCK_SCHEMA_VERSION}")
    return lock


def write_datasets_lock(path: str | os.PathLike, lock: DatasetsLock) -> str:
    return _write_locked_json(Path(path), _to_jsonable(lock))


def load_prompts_lock(path: str | os.PathLike) -> PromptsLock:
    data = _read_locked_json(Path(path))
    lock = build(PromptsLock, data, "prompts.lock")
    if lock.schema_version != LOCK_SCHEMA_VERSION:
        raise LockError(f"{path}: schema_version {lock.schema_version} != {LOCK_SCHEMA_VERSION}")
    return lock


def write_prompts_lock(path: str | os.PathLike, lock: PromptsLock) -> str:
    return _write_locked_json(Path(path), _to_jsonable(lock))


# --------------------------------------------------------------------------- HF snapshot facts


def hf_home() -> Path:
    return Path(os.environ.get(HF_HOME_ENV) or DEFAULT_HF_HOME)


def snapshot_dir(hf_home_path: str | os.PathLike, hf_id: str, revision: str) -> Path:
    org, _, name = hf_id.partition("/")
    if not org or not name:
        raise LockError(f"hf_id {hf_id!r} is not '<org>/<name>'")
    return Path(hf_home_path) / "hub" / f"models--{org}--{name}" / "snapshots" / revision


def _chat_template_sha256(snapshot: Path) -> str:
    jinja = snapshot / "chat_template.jinja"
    if jinja.exists():
        return identity.sha256_hex(jinja.read_bytes())
    config = snapshot / "tokenizer_config.json"
    if not config.exists():
        raise LockError(f"{snapshot}: neither chat_template.jinja nor tokenizer_config.json is present")
    template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
    if not isinstance(template, str) or not template:
        raise LockError(f"{config}: no chat_template string")
    return identity.sha256_hex(template.encode("utf-8"))


def _shard_shas(snapshot: Path) -> dict[str, str]:
    """``{shard filename: sha256}`` from the content-addressed blob names."""
    out: dict[str, str] = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        if not shard.is_symlink():
            raise LockError(f"{shard} is not a symlink into blobs/; cannot read its sha off the blob name")
        target = os.path.basename(os.readlink(shard))
        if not is_sha256(target):
            raise LockError(f"{shard} -> {target!r} is not a sha256 blob name")
        out[shard.name] = target
    if not out:
        raise LockError(f"{snapshot}: no safetensors shards")
    return out


def _n_layers(snapshot: Path) -> int:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    n = config.get("num_hidden_layers")
    if not isinstance(n, int):
        raise LockError(f"{snapshot / 'config.json'}: no num_hidden_layers")
    return n


def snapshot_facts(hf_home_path: str | os.PathLike, hf_id: str, revision: str) -> dict[str, Any]:
    snapshot = snapshot_dir(hf_home_path, hf_id, revision)
    if not snapshot.is_dir():
        raise LockError(f"{snapshot} is not a cached snapshot (weights absent or unpinned revision)")
    config = snapshot / "config.json"
    if not config.exists():
        raise LockError(f"{config} is missing")
    return {
        "config_sha256": identity.sha256_hex(config.read_bytes()),
        "chat_template_sha256": _chat_template_sha256(snapshot),
        "safetensors": _shard_shas(snapshot),
        "n_layers": _n_layers(snapshot),
    }


# --------------------------------------------------------------------------- models lock


def lock_models(cfg: StudyConfig, compute: ComputeConfig, hf_home_path: str | os.PathLike) -> ModelsLock:
    """Resolve every model cell against the offline HF cache.  Enabled cells must lock;
    disabled cells whose weights are absent are recorded ``blocked`` with the reason."""
    engine = compute.serving.engine
    entries: dict[str, ModelLockEntry] = {}
    for name, block in cfg.models.items():
        base = dict(hf_id=block.model_id, revision=block.revision, tokenizer_revision=block.tokenizer_revision,
                    adapter=block.adapter, dtype=engine.dtype, reasoning_parser=engine.reasoning_parser)
        try:
            if not block.revision or not block.tokenizer_revision:
                raise LockError(f"model cell {name!r} has no pinned revision")
            if block.tokenizer_revision != block.revision:
                # One snapshot, one tokenizer blob: a tokenizer pinned elsewhere is a second snapshot to verify.
                snapshot_dir(hf_home_path, block.model_id, block.tokenizer_revision)
            facts = snapshot_facts(hf_home_path, block.model_id, block.revision)
        except LockError as exc:
            if block.enabled:
                raise LockError(f"enabled model cell {name!r} cannot be locked: {exc}") from exc
            entries[name] = ModelLockEntry(status=MODEL_STATUS_BLOCKED, blocked_reason=str(exc),
                                           config_sha256=None, chat_template_sha256=None, safetensors=None,
                                           n_layers=None, **base)
            continue
        entries[name] = ModelLockEntry(status=MODEL_STATUS_LOCKED, blocked_reason=None,
                                       config_sha256=facts["config_sha256"],
                                       chat_template_sha256=facts["chat_template_sha256"],
                                       safetensors=facts["safetensors"], n_layers=facts["n_layers"], **base)
    return ModelsLock(schema_version=LOCK_SCHEMA_VERSION,
                      locked_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                      hf_home=str(hf_home_path), vllm_version=engine.vllm, torch_version=engine.torch,
                      cuda_version=engine.cuda, models=entries)


def verify_models_lock(lock: ModelsLock, hf_home_path: str | os.PathLike, *,
                       required: Sequence[str] | None = None) -> None:
    """Fail closed: every ``required`` model (default: every ``locked`` entry) must be
    ``locked`` with non-null pins that match the cache byte for byte / blob for blob."""
    names = list(required) if required is not None else [n for n, e in lock.models.items()
                                                          if e.status == MODEL_STATUS_LOCKED]
    if lock.locked_at is None or lock.vllm_version is None:
        raise LockError("models.lock.json is a skeleton (locked_at / vllm_version null); run lock-inputs")
    for name in names:
        entry = lock.models.get(name)
        if entry is None:
            raise LockError(f"models.lock.json has no entry for model cell {name!r}")
        if entry.status != MODEL_STATUS_LOCKED:
            raise LockError(f"model cell {name!r} is {entry.status}: {entry.blocked_reason}")
        for field_name in ("revision", "tokenizer_revision", "config_sha256", "chat_template_sha256",
                           "safetensors", "n_layers", "dtype", "reasoning_parser"):
            if getattr(entry, field_name) is None:
                raise LockError(f"model cell {name!r}: {field_name} is null (launch blocker)")
        facts = snapshot_facts(hf_home_path, entry.hf_id, entry.revision)
        if facts["config_sha256"] != entry.config_sha256:
            raise LockError(f"model cell {name!r}: config.json drifted ({entry.config_sha256[:12]} -> "
                            f"{facts['config_sha256'][:12]})")
        if facts["chat_template_sha256"] != entry.chat_template_sha256:
            raise LockError(f"model cell {name!r}: chat template drifted")
        if dict(facts["safetensors"]) != dict(entry.safetensors):
            raise LockError(f"model cell {name!r}: safetensors shards drifted "
                            f"(locked {sorted(entry.safetensors)}, cache {sorted(facts['safetensors'])})")
        if facts["n_layers"] != entry.n_layers:
            raise LockError(f"model cell {name!r}: num_hidden_layers drifted")
        if entry.adapter not in K.ADAPTERS:
            raise LockError(f"model cell {name!r}: unknown adapter {entry.adapter!r}")


# --------------------------------------------------------------------------- prompts lock


def templates_dir() -> Path:
    return Path(__file__).resolve().parent / TEMPLATES_DIRNAME


def lock_prompts(templates_path: str | os.PathLike, prefill_text: str) -> PromptsLock:
    """Hash every file under ``templates/`` (sorted relative paths) plus the continuation
    prefill bytes (plan D5)."""
    root = Path(templates_path)
    if not root.is_dir():
        raise LockError(f"{root} is not a directory; the templates (WP2) have not landed")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise LockError(f"{root} holds no template files")
    if not prefill_text:
        raise LockError("the continuation prefill text is empty")
    templates = {str(p.relative_to(root)): identity.sha256_hex(p.read_bytes()) for p in files}
    return PromptsLock(schema_version=LOCK_SCHEMA_VERSION,
                       locked_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                       templates=templates, continuation_prefill_text=prefill_text,
                       continuation_prefill_sha256=identity.sha256_hex(prefill_text.encode("utf-8")))


def verify_prompts_lock(lock: PromptsLock, templates_path: str | os.PathLike, prefill_text: str) -> None:
    if lock.locked_at is None or lock.continuation_prefill_sha256 is None:
        raise LockError("prompts.lock.json is a skeleton; run lock-inputs")
    current = lock_prompts(templates_path, prefill_text)
    if dict(current.templates) != dict(lock.templates):
        raise LockError("prompts.lock.json: template hashes drifted")
    if current.continuation_prefill_sha256 != lock.continuation_prefill_sha256:
        raise LockError("prompts.lock.json: continuation prefill drifted")


def _continuation_prefill_text() -> str:
    """The frozen prefill lives in ``decoding.py`` (WP1); until it lands the prompts lock is blocked."""
    import importlib

    try:
        module = importlib.import_module("agents_scaling.nextdyn.decoding")
    except ImportError as exc:
        raise LockError(f"agents_scaling.nextdyn.decoding is not importable ({exc}); WP1 pending") from exc
    text = getattr(module, CONTINUATION_PREFILL_ATTR, None)
    if not isinstance(text, str) or not text:
        raise LockError(f"decoding.{CONTINUATION_PREFILL_ATTR} is missing or empty")
    return text


# --------------------------------------------------------------------------- the verb


def lock_inputs_main(argv: Sequence[str]) -> Report:
    ap = common_parser("lock-inputs", description="resolve model / prompt / dataset pins into the lock files")
    ap.add_argument("--compute", type=Path, default=None, help="compute.yaml (default beside study.yaml)")
    ap.add_argument("--hf-home", type=Path, default=None, help="HF cache (default $HF_HOME)")
    args = ap.parse_args(list(argv))
    cfg = load_study(args.config)
    compute = load_compute(args.compute or Path(args.config).resolve().parent / "compute.yaml")
    paths = lock_paths(args.config)
    home = args.hf_home or hf_home()
    report = Report()

    # models
    try:
        lock = lock_models(cfg, compute, home)
        verify_models_lock(lock, home, required=[n for n, b in cfg.models.items() if b.enabled])
        write_models_lock(paths.models, lock)
        report = report + Report(completed=1, detail=f"models -> {paths.models}")
    except LockError as exc:
        report = report + Report(failed=1, detail=f"models: {exc}")

    # prompts
    try:
        prompts = lock_prompts(templates_dir(), _continuation_prefill_text())
        write_prompts_lock(paths.prompts, prompts)
        report = report + Report(completed=1, detail=f"prompts -> {paths.prompts}")
    except LockError as exc:
        report = report + Report(blocked=1, detail=f"prompts blocked: {exc}")

    # datasets (WP3)
    try:
        import importlib

        datasets = importlib.import_module("agents_scaling.nextdyn.datasets")
        lock_datasets = getattr(datasets, "lock_datasets")
    except (ImportError, AttributeError) as exc:
        report = report + Report(blocked=1, detail=f"datasets blocked: nextdyn.datasets.lock_datasets not importable ({exc})")
    else:
        try:
            dlock = lock_datasets(cfg, home)
            write_datasets_lock(paths.datasets, dlock)
            report = report + Report(completed=1, detail=f"datasets -> {paths.datasets}")
        except LockError as exc:
            report = report + Report(failed=1, detail=f"datasets: {exc}")
    return report


__all__ = [
    "CONTINUATION_PREFILL_ATTR", "DatasetLockEntry", "DatasetsLock", "HF_HOME_ENV", "LOCK_SCHEMA_VERSION",
    "LegacyRootEntry", "MODEL_STATUS_BLOCKED", "MODEL_STATUS_LOCKED", "ModelLockEntry", "ModelsLock",
    "PromptsLock", "hf_home", "load_datasets_lock", "load_models_lock", "load_prompts_lock",
    "lock_inputs_main", "lock_models", "lock_prompts", "snapshot_dir", "snapshot_facts", "templates_dir",
    "verify_models_lock", "verify_prompts_lock", "write_datasets_lock", "write_models_lock",
    "write_prompts_lock",
]
