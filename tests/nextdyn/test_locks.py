"""Locks: sidecars verified, null fields fail closed, drift fails closed, the verb reports
models / prompts / datasets separately."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import locks as L
from agents_scaling.nextdyn.cli import Report
from agents_scaling.nextdyn.errors import LockError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "nextdyn"
ORG, NAME, REV = "FakeOrg", "Fake-8B", "0123456789abcdef0123456789abcdef01234567"


def _fake_hf_home(tmp_path: Path, *, n_shards: int = 2, chat_template: str = "{{ messages }}") -> Path:
    home = tmp_path / "hf"
    repo = home / "hub" / f"models--{ORG}--{NAME}"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / REV
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 36, "architectures": ["Qwen3ForCausalLM"]}))
    (snap / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))
    for i in range(n_shards):
        content = f"shard-{i}".encode()
        sha = hashlib.sha256(content).hexdigest()
        (blobs / sha).write_bytes(content)
        os.symlink(f"../../blobs/{sha}", snap / f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors")
    return home


def _fake_cfg(tmp_path: Path, *, enabled: bool = True, revision: str | None = REV) -> Path:
    data = yaml.safe_load((CONFIG_DIR / "study.yaml").read_text())
    data["models"] = {
        "primary": {**data["models"]["primary"], "model_id": f"{ORG}/{NAME}", "revision": revision,
                    "tokenizer_revision": revision, "enabled": enabled},
        "scale": {**data["models"]["scale"], "model_id": f"{ORG}/{NAME}", "revision": revision,
                  "tokenizer_revision": revision, "enabled": False},
    }
    data["experiments"]["E3C_FAMILY"]["models"] = ["scale"]
    data["experiments"]["E3c"]["model"] = "scale"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    (tmp_path / "compute.yaml").write_text((CONFIG_DIR / "compute.yaml").read_text())
    return path


def test_lock_round_trip_verifies_the_sidecar(tmp_path):
    """Write, re-read, and refuse a lock whose bytes changed under its sidecar.

    The fixture is built here rather than loaded from ``configs/nextdyn``: the committed lock is
    a moving target once ``lock-inputs`` has run, and what this test is about is the round trip
    and the drift refusal, neither of which depends on which models happen to be pinned today.
    """
    home = _fake_hf_home(tmp_path)
    cfg = C.load_study(_fake_cfg(tmp_path))
    compute = C.load_compute(tmp_path / "compute.yaml")
    lock = L.lock_models(cfg, compute, home)
    assert lock.schema_version == L.LOCK_SCHEMA_VERSION
    assert any(e.status == L.MODEL_STATUS_LOCKED for e in lock.models.values())
    path = tmp_path / "models.lock.json"
    sha = L.write_models_lock(path, lock)
    assert (tmp_path / "models.lock.json.sha256").read_text().split()[0] == sha
    assert L.load_models_lock(path) == lock
    # Any byte change under a sidecar is drift, whatever it touched.
    path.write_text(path.read_text().replace("bfloat16", "float16", 1))
    with pytest.raises(LockError, match="drift"):
        L.load_models_lock(path)
    with pytest.raises(LockError):
        L.load_models_lock(tmp_path / "nope.lock.json")


def test_skeleton_locks_have_null_fields_and_fail_closed(tmp_path):
    """A lock with nothing in it refuses; it never verifies vacuously.

    Asserted on a skeleton this test writes, not on the committed files: those are filled in by
    ``lock-inputs`` as soon as the study is real, and a test that pins them empty would fail the
    moment the study advanced -- and would have been asserting the wrong thing anyway.  What
    must hold forever is the *shape*: an empty lock is a launch blocker.
    """
    skeleton = tmp_path / "models.lock.json"
    skeleton.write_text(json.dumps({"schema_version": L.LOCK_SCHEMA_VERSION, "locked_at": None,
                                    "hf_home": None, "vllm_version": None, "torch_version": None,
                                    "cuda_version": None, "models": {}}), encoding="utf-8")
    models = L.load_models_lock(skeleton)
    assert models.locked_at is None and models.models == {}
    with pytest.raises(LockError):
        L.verify_models_lock(models, tmp_path, required=["primary"])

    empty_prompts = tmp_path / "prompts.lock.json"
    empty_prompts.write_text(json.dumps({"schema_version": L.LOCK_SCHEMA_VERSION, "locked_at": None,
                                         "templates": {}, "continuation_prefill_text": None,
                                         "continuation_prefill_sha256": None}), encoding="utf-8")
    prompts = L.load_prompts_lock(empty_prompts)
    assert prompts.templates == {} and prompts.continuation_prefill_sha256 is None
    with pytest.raises(LockError, match="skeleton"):
        L.verify_prompts_lock(prompts, tmp_path, "x")

    # And the committed locks, whatever state the study has reached, must be self-consistent:
    # a lock that carries entries must carry the facts that make it verifiable.
    committed = L.load_models_lock(CONFIG_DIR / "models.lock.json")
    for name, entry in committed.models.items():
        if entry.status == L.MODEL_STATUS_LOCKED:
            assert entry.revision and entry.config_sha256 and entry.chat_template_sha256, name
            assert entry.safetensors, name
    # The datasets lock is no longer a skeleton: WP3 locked it against the offline cache.  What
    # must hold is that every family sourced from the Hub carries an exact revision (a null one
    # is a launch blocker), while the two families that have no Hub revision by construction --
    # MOD97 is generated and BIGCODEBENCH is a pinned local parquet -- are declared without one.
    datasets = L.load_datasets_lock(CONFIG_DIR / "datasets.lock.json")
    assert set(datasets.datasets) >= {"MATH500", "MMLU_PRO", "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE",
                                      "MATH_EXPANDED", "MOD97"}
    assert set(datasets.legacy_roots) == {"ard36", "study_v4", "full_sweep_v1"}
    hub_pinned = {"MATH500", "MATH_EXPANDED", "MMLU_PRO", "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE"}
    for name in hub_pinned:
        entry = datasets.datasets[name]
        assert entry.revision and len(entry.revision) == 40, name
        assert entry.n_rows and entry.n_rows > 0, name
    assert datasets.datasets["MOD97"].revision is None       # generated, not downloaded
    # The expanded MATH panel is the seven hendrycks subjects minus every MATH-500 item: the
    # disjointness rule of plan section 2.1, visible as a row count rather than asserted in prose.
    assert datasets.datasets["MATH_EXPANDED"].n_rows == 5000 - datasets.datasets["MATH500"].n_rows


def test_lock_models_reads_blob_names_and_verify_detects_drift(tmp_path):
    home = _fake_hf_home(tmp_path)
    cfg = C.load_study(_fake_cfg(tmp_path))
    compute = C.load_compute(tmp_path / "compute.yaml")
    lock = L.lock_models(cfg, compute, home)
    entry = lock.models["primary"]
    assert entry.status == L.MODEL_STATUS_LOCKED and entry.n_layers == 36
    assert entry.safetensors == {f"model-{i + 1:05d}-of-00002.safetensors": hashlib.sha256(f"shard-{i}".encode()).hexdigest()
                                 for i in range(2)}
    assert entry.chat_template_sha256 == hashlib.sha256(b"{{ messages }}").hexdigest()
    assert entry.config_sha256 == hashlib.sha256((home / "hub" / f"models--{ORG}--{NAME}" / "snapshots" / REV / "config.json").read_bytes()).hexdigest()
    assert lock.vllm_version == "0.21.0" and entry.dtype == "bfloat16" and entry.reasoning_parser == "qwen3"
    assert lock.models["scale"].status == L.MODEL_STATUS_LOCKED    # disabled but cached -> locked anyway
    L.verify_models_lock(lock, home)

    # config drift
    snap = home / "hub" / f"models--{ORG}--{NAME}" / "snapshots" / REV
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 36, "x": 1}))
    with pytest.raises(LockError, match="config.json drifted"):
        L.verify_models_lock(lock, home)
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 36, "architectures": ["Qwen3ForCausalLM"]}))
    # shard drift: re-point a symlink at another blob
    shard = snap / "model-00002-of-00002.safetensors"
    shard.unlink()
    other = hashlib.sha256(b"other").hexdigest()
    (snap.parent.parent / "blobs" / other).write_bytes(b"other")
    os.symlink(f"../../blobs/{other}", shard)
    with pytest.raises(LockError, match="safetensors shards drifted"):
        L.verify_models_lock(lock, home)
    # chat template drift
    shard.unlink()
    os.symlink(f"../../blobs/{hashlib.sha256(b'shard-1').hexdigest()}", shard)
    (snap / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ other }}"}))
    with pytest.raises(LockError, match="chat template drifted"):
        L.verify_models_lock(lock, home)
    # a null pin fails closed
    nulled = dataclasses.replace(lock, models={**lock.models, "primary": dataclasses.replace(entry, config_sha256=None)})
    with pytest.raises(LockError, match="config_sha256 is null"):
        L.verify_models_lock(nulled, home)


def test_enabled_model_without_weights_cannot_lock(tmp_path):
    home = tmp_path / "empty_hf"
    home.mkdir()
    cfg = C.load_study(_fake_cfg(tmp_path))
    compute = C.load_compute(tmp_path / "compute.yaml")
    with pytest.raises(LockError, match="enabled model cell 'primary'"):
        L.lock_models(cfg, compute, home)
    disabled = C.load_study(_fake_cfg(tmp_path, enabled=False))
    lock = L.lock_models(disabled, compute, home)
    assert lock.models["primary"].status == L.MODEL_STATUS_BLOCKED and "not a cached snapshot" in lock.models["primary"].blocked_reason
    with pytest.raises(LockError, match="blocked"):
        L.verify_models_lock(lock, home, required=["primary"])
    unpinned = C.load_study(_fake_cfg(tmp_path, enabled=False, revision=None))
    assert "no pinned revision" in L.lock_models(unpinned, compute, home).models["primary"].blocked_reason


def test_lock_prompts_hashes_every_template_and_the_prefill(tmp_path):
    templates = tmp_path / "templates"
    (templates / "sub").mkdir(parents=True)
    (templates / "root.txt").write_text("SYSTEM\nSolve.")
    (templates / "sub" / "filler_words.txt").write_text("alpha beta")
    prefill = '\n\n{\n  "reasoning": "'
    lock = L.lock_prompts(templates, prefill)
    assert set(lock.templates) == {"root.txt", "sub/filler_words.txt"}
    assert lock.templates["root.txt"] == hashlib.sha256(b"SYSTEM\nSolve.").hexdigest()
    assert lock.continuation_prefill_sha256 == hashlib.sha256(prefill.encode()).hexdigest()
    L.verify_prompts_lock(lock, templates, prefill)
    (templates / "root.txt").write_text("SYSTEM\nSolve!")
    with pytest.raises(LockError, match="template hashes drifted"):
        L.verify_prompts_lock(lock, templates, prefill)
    with pytest.raises(LockError):
        L.lock_prompts(tmp_path / "missing", prefill)
    with pytest.raises(LockError):
        L.lock_prompts(templates, "")


def test_lock_inputs_verb_reports_models_prompts_and_datasets_separately(tmp_path, monkeypatch):
    home = _fake_hf_home(tmp_path)
    cfg_path = _fake_cfg(tmp_path)
    report = L.lock_inputs_main(["--config", str(cfg_path), "--hf-home", str(home)])
    assert isinstance(report, Report)
    # models and prompts lock from the repo itself (templates and the continuation prefill are
    # on disk); datasets are blocked because this fake HF home holds no snapshots.  The point of
    # the verb is that the three report separately -- a partially locked study is a blocker, not
    # a silent success.
    assert report.completed == 2                        # models + prompts locked
    assert report.failed == 0
    assert report.blocked == 1                          # datasets: nothing in this HF home
    assert "datasets blocked" in report.detail
    written = L.load_models_lock(tmp_path / "models.lock.json")
    assert written.models["primary"].status == L.MODEL_STATUS_LOCKED
    assert (tmp_path / "models.lock.json.sha256").exists()
    assert report.exit_code("lock-inputs") == 2


def test_hf_home_defaults_to_the_data_filesystem(monkeypatch):
    monkeypatch.delenv(L.HF_HOME_ENV, raising=False)
    assert str(L.hf_home()).startswith("/orcd/data/")
    monkeypatch.setenv(L.HF_HOME_ENV, "/tmp/elsewhere")
    assert L.hf_home() == Path("/tmp/elsewhere")
    assert L.snapshot_dir("/x", "Qwen/Qwen3-8B", "abc") == Path("/x/hub/models--Qwen--Qwen3-8B/snapshots/abc")
    with pytest.raises(LockError):
        L.snapshot_dir("/x", "Qwen3-8B", "abc")
