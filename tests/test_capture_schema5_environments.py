"""Immutable environment capture and ownership-normalization tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from scripts import capture_schema5_environments as capture
from scripts import seal_recovery_evidence as seal_evidence


REPO = Path(__file__).resolve().parent.parent
OWNERSHIP_POLICY = (
    REPO / "configs" / "environment_ownership_policy.v1.json"
)
INTEGRITY_POLICY = (
    REPO
    / "configs"
    / "environment_integrity_normalization_policy.v1.json"
)
SETUPTOOLS_ARTIFACT_SHA256 = (
    "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
)


def _record_hash(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")


def _conda_record(*, name: str, version: str, build: str, digest: str) -> dict:
    return {
        "name": name,
        "version": version,
        "build": build,
        "sha256": digest,
        "url": f"https://conda.example.invalid/noarch/{name}-{version}-{build}.conda",
    }


def _prefix(tmp_path: Path, name: str, *, stale_setuptools_record: bool) -> Path:
    prefix = tmp_path / name
    conda_meta = prefix / "conda-meta"
    site = prefix / "lib" / "python3.11" / "site-packages"
    dist = site / "setuptools-81.0.0.dist-info"
    package = site / "setuptools"
    bin_root = prefix / "bin"
    conda_meta.mkdir(parents=True)
    dist.mkdir(parents=True)
    package.mkdir()
    bin_root.mkdir()
    (bin_root / "python").write_bytes(b"fake python\n")
    (package / "__init__.py").write_bytes(b"__version__ = '81.0.0'\n")
    metadata = b"Metadata-Version: 2.1\nName: setuptools\nVersion: 81.0.0\n\n"
    (dist / "METADATA").write_bytes(metadata)
    (dist / "RECORD").write_text(
        "setuptools/__init__.py,sha256="
        + _record_hash((package / "__init__.py").read_bytes())
        + f",{(package / '__init__.py').stat().st_size}\n"
        "setuptools-81.0.0.dist-info/METADATA,sha256="
        + _record_hash(metadata)
        + f",{len(metadata)}\n"
        "setuptools-81.0.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    python_record = _conda_record(
        name="python",
        version="3.11.13",
        build="h123_0",
        digest="1" * 64,
    )
    (conda_meta / "python-3.11.13-h123_0.json").write_text(
        json.dumps(python_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if stale_setuptools_record:
        setuptools_record = _conda_record(
            name="setuptools",
            version="82.0.1",
            build="pyh332efcf_0",
            digest=SETUPTOOLS_ARTIFACT_SHA256,
        )
        (conda_meta / "setuptools-82.0.1-pyh332efcf_0.json").write_text(
            json.dumps(setuptools_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (prefix / "python-link").symlink_to("bin/python")
    return prefix


def _add_conda_owned_distribution(
    prefix: Path,
    *,
    project: str = "owned_project",
    version: str = "1.2.3",
) -> tuple[Path, Path]:
    site = prefix / "lib" / "python3.11" / "site-packages"
    package = site / project
    dist = site / f"{project}-{version}.dist-info"
    package.mkdir()
    dist.mkdir()
    installed = b"authoritative installed bytes\n"
    metadata = (
        f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n\n"
    ).encode()
    (package / "__init__.py").write_bytes(installed)
    (dist / "METADATA").write_bytes(metadata)
    record_path = dist / "RECORD"
    record_path.write_text(
        f"{project}/__init__.py,sha256={_record_hash(installed)},{len(installed)}\n"
        f"{dist.name}/METADATA,sha256={_record_hash(metadata)},{len(metadata)}\n"
        f"{dist.name}/RECORD,,\n",
        encoding="utf-8",
    )
    conda_record = _conda_record(
        name=project,
        version=version,
        build="py_0",
        digest="2" * 64,
    )
    (prefix / "conda-meta" / f"{project}-{version}-py_0.json").write_text(
        json.dumps(conda_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package / "__init__.py", record_path


def _evidence(
    tmp_path: Path, harness: Path, serving: Path
) -> tuple[Path, Path]:
    recovered = tmp_path / "setuptools-82.0.1-pyh332efcf_0.json"
    recovered.write_text(
        json.dumps(
            _conda_record(
                name="setuptools",
                version="82.0.1",
                build="pyh332efcf_0",
                digest=SETUPTOOLS_ARTIFACT_SHA256,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    recovered.chmod(0o444)
    failed_log = tmp_path / "failed-materialization.log"
    failed_log.write_text("pip distribution view changed\n", encoding="utf-8")
    incident = tmp_path / "CONDA_RECONCILIATION_INCIDENT.json"
    seal_evidence.record_conda_reconciliation_incident(
        output=incident,
        harness_prefix=harness,
        serving_prefix=serving,
        recovered_harness_record=recovered,
        failed_materialization_log=failed_log,
        observed_at="2026-07-23T11:43:55-04:00",
        apply=True,
    )
    return incident, recovered


def _inputs(tmp_path: Path) -> dict:
    harness = _prefix(tmp_path, "harness-source", stale_setuptools_record=False)
    serving = _prefix(tmp_path, "serving-source", stale_setuptools_record=True)
    incident, recovered = _evidence(tmp_path, harness, serving)
    return {
        "output_root": tmp_path / "capture",
        "harness_source": harness,
        "serving_source": serving,
        "ownership_policy": OWNERSHIP_POLICY,
        "integrity_normalization_policy": INTEGRITY_POLICY,
        "reconciliation_incident": incident,
        "recovered_setuptools_record": recovered,
    }


def _rewrite_incident(
    path: Path,
    payload: dict,
    *,
    recompute_id: bool = True,
    canonical: bool = True,
    rewrite_sidecar: bool = True,
) -> None:
    payload = dict(payload)
    if recompute_id:
        payload.pop("incident_id", None)
        payload["incident_id"] = hashlib.sha256(
            capture._incident_canonical_bytes(payload)
        ).hexdigest()
    raw = (
        capture._incident_canonical_bytes(payload)
        if canonical
        else (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    path.chmod(0o644)
    path.write_bytes(raw)
    path.chmod(0o444)
    if rewrite_sidecar:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar.chmod(0o644)
        sidecar.write_text(
            f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n",
            encoding="utf-8",
        )
        sidecar.chmod(0o444)


def _add_pip_record_normalization_fixture(prefix: Path) -> tuple[Path, bytes]:
    site = prefix / "lib" / "python3.11" / "site-packages"
    dist = site / "pip-26.1.1.dist-info"
    dist.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 26.1.1\n\n"
    (dist / "METADATA").write_bytes(metadata)
    (dist / "INSTALLER").write_bytes(b"conda")
    (prefix / "bin" / "pip").write_bytes(b"captured pip launcher\n")
    (prefix / "bin" / "pip3").write_bytes(b"captured pip3 launcher\n")
    launcher_lines = [
        f"{path},{encoded_hash},{size}\n"
        for path, encoded_hash, size in capture.PIP_LAUNCHER_RECORD_ROWS
    ]
    record_bytes = (
        "".join(launcher_lines)
        + "pip-26.1.1.dist-info/INSTALLER,"
        "sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg,4\n"
        + "pip-26.1.1.dist-info/METADATA,sha256="
        + _record_hash(metadata)
        + f",{len(metadata)}\n"
        + "pip-26.1.1.dist-info/RECORD,,\n"
    ).encode()
    record_path = dist / "RECORD"
    record_path.write_bytes(record_bytes)
    return record_path, record_bytes


def _inputs_with_pip_record_normalization(
    tmp_path: Path, monkeypatch
) -> tuple[dict, str, str]:
    inputs = _inputs(tmp_path)
    preimages = []
    for role in capture.ROLES:
        _path, preimage = _add_pip_record_normalization_fixture(
            inputs[f"{role}_source"]
        )
        preimages.append(preimage)
    assert preimages[0] == preimages[1]
    source_sha256 = hashlib.sha256(preimages[0]).hexdigest()
    policy = json.loads(INTEGRITY_POLICY.read_text(encoding="utf-8"))
    pip_policy = next(
        row
        for row in policy["normalizations"]
        if row["normalization_id"] == capture.PIP_RECORD_NORMALIZATION_ID
    )
    launcher_rows = {
        (path, encoded_hash, size)
        for path, encoded_hash, size in capture.PIP_LAUNCHER_RECORD_ROWS
    }
    installer_before = (
        "pip-26.1.1.dist-info/INSTALLER",
        "sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg",
        "4",
    )
    installer_after = (
        "pip-26.1.1.dist-info/INSTALLER",
        "sha256=0O3uFfkbQG8_mXJuROuZC-bjT9A0W1K5EMVo4O72oqg",
        "5",
    )
    post_lines = []
    for line in preimages[0].splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        row = tuple(content.decode().split(","))
        if row in launcher_rows:
            continue
        if row == installer_before:
            ending = line[len(content) :]
            post_lines.append(",".join(installer_after).encode() + ending)
        else:
            post_lines.append(line)
    postimage = b"".join(post_lines)
    normalized_sha256 = hashlib.sha256(postimage).hexdigest()
    pip_policy["source_record_sha256"] = source_sha256
    pip_policy["normalized_record_sha256"] = normalized_sha256
    for role in capture.ROLES:
        source = inputs[f"{role}_source"]
        contracts = []
        for name in ("pip", "pip3"):
            path = source / "bin" / name
            contracts.append(
                {
                    "path": f"bin/{name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                    "state": "regular_file",
                }
            )
        contracts.append({"path": "bin/pip3.12", "state": "missing"})
        pip_policy["role_source_file_contracts"][role] = contracts
    policy_path = tmp_path / "fixture-integrity-policy.json"
    policy_path.write_text(
        json.dumps(policy, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy_path.with_suffix(".sha256").write_text(
        f"{policy_sha256}  {policy_path.name}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(capture, "PIP_RECORD_SOURCE_SHA256", source_sha256)
    monkeypatch.setattr(
        capture, "PIP_RECORD_NORMALIZED_SHA256", normalized_sha256
    )
    inputs["integrity_normalization_policy"] = policy_path
    return inputs, source_sha256, normalized_sha256


def test_capture_is_real_copy_normalizes_only_setuptools_and_is_idempotent(tmp_path):
    inputs = _inputs(tmp_path)
    dry_run = capture.capture_environments(**inputs)

    assert dry_run["status"] == "dry_run"
    assert dry_run["copy_contract"]["hardlinks"] is False
    assert dry_run["copy_contract"]["reflinks"] is False
    assert {
        role: audit["projected_shared_record_path_count"]
        for role, audit in dry_run["source_distribution_audits"].items()
    } == {"harness": 0, "serving": 0}
    assert not inputs["output_root"].exists()

    created = capture.capture_environments(**inputs, apply=True)
    repeated = capture.capture_environments(**inputs, apply=True)

    assert created["status"] == "created"
    assert repeated["status"] == "already_complete"
    assert repeated["capture_id"] == created["capture_id"]
    assert (inputs["output_root"] / capture.COMPLETE_MARKER).is_file()
    for role in capture.ROLES:
        source = inputs[f"{role}_source"]
        seed = inputs["output_root"] / "seeds" / role
        assert not (
            set(_regular_inodes(source)) & set(_regular_inodes(seed))
        )
        assert not list((seed / "conda-meta").glob("setuptools-*.json"))
        assert (seed / "lib/python3.11/site-packages/setuptools").is_dir()
        assert (
            seed / "lib/python3.11/site-packages/setuptools-81.0.0.dist-info"
        ).is_dir()
        assert stat_write_bits(seed) == 0
        receipt = json.loads(
            (
                inputs["output_root"]
                / "normalization"
                / f"{role}.receipt.json"
            ).read_text(encoding="utf-8")
        )
        assert receipt["normalized_identity"]["setuptools_runtime_version"] == "81.0.0"
        assert receipt["normalized_identity"]["setuptools_conda_record_count"] == 0
        assert receipt["archived_preimage"]["record"][
            "artifact_sha256"
        ] == SETUPTOOLS_ARTIFACT_SHA256
    assert json.loads(
        (
            inputs["output_root"] / "normalization" / "harness.receipt.json"
        ).read_text(encoding="utf-8")
    )["source_record_state"] == "absent"
    assert json.loads(
        (
            inputs["output_root"] / "normalization" / "serving.receipt.json"
        ).read_text(encoding="utf-8")
    )["source_record_state"] == "present"


def test_capture_rejects_source_record_state_that_differs_from_sealed_incident(
    tmp_path: Path,
):
    inputs = _inputs(tmp_path)
    incident = inputs["reconciliation_incident"]
    payload = json.loads(incident.read_text(encoding="utf-8"))
    payload["serving_stale_conda_record_present"] = False
    payload["superseded_conda_record"]["serving_path"] = None
    payload["superseded_conda_record"]["serving_sha256"] = None
    _rewrite_incident(incident, payload)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match=(
            "live source Setuptools record state differs from the sealed "
            "reconciliation incident for serving"
        ),
    ):
        capture.capture_environments(**inputs)
    assert not inputs["output_root"].exists()


def test_capture_accepts_exact_incident_v1_emitted_by_sealer(tmp_path):
    inputs = _inputs(tmp_path)
    incident = inputs["reconciliation_incident"]
    payload = json.loads(incident.read_text(encoding="utf-8"))

    assert set(payload["superseded_conda_record"]) == {
        "name",
        "version",
        "build",
        "artifact_sha256",
        "recovered_harness_path",
        "recovered_harness_sha256",
        "serving_path",
        "serving_sha256",
    }
    assert capture.capture_environments(**inputs)["status"] == "dry_run"


def test_capture_rejects_recovered_record_substitution_bound_by_incident(tmp_path):
    inputs = _inputs(tmp_path)
    recovered = inputs["recovered_setuptools_record"]
    payload = json.loads(recovered.read_text(encoding="utf-8"))
    payload["files"] = []
    recovered.chmod(0o644)
    recovered.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    recovered.chmod(0o444)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="recovered-record binding is invalid",
    ):
        capture.capture_environments(**inputs)


@pytest.mark.parametrize("mutation", ("extra_field", "incident_id", "encoding"))
def test_capture_rejects_incident_schema_id_and_encoding_tamper(
    tmp_path, mutation
):
    inputs = _inputs(tmp_path)
    incident = inputs["reconciliation_incident"]
    payload = json.loads(incident.read_text(encoding="utf-8"))
    if mutation == "extra_field":
        payload["unreviewed_authority"] = True
        _rewrite_incident(incident, payload)
        expected = "field inventory drifted"
    elif mutation == "incident_id":
        payload["incident_id"] = "0" * 64
        _rewrite_incident(incident, payload, recompute_id=False)
        expected = "identity or canonical encoding"
    else:
        _rewrite_incident(incident, payload, canonical=False)
        expected = "identity or canonical encoding"

    with pytest.raises(capture.EnvironmentCaptureError, match=expected):
        capture.capture_environments(**inputs)


@pytest.mark.parametrize("mutation", ("missing", "wrong"))
def test_capture_requires_exact_read_only_incident_checksum(tmp_path, mutation):
    inputs = _inputs(tmp_path)
    incident = inputs["reconciliation_incident"]
    sidecar = incident.with_suffix(incident.suffix + ".sha256")
    if mutation == "missing":
        sidecar.unlink()
    else:
        sidecar.chmod(0o644)
        sidecar.write_text(
            f"{'0' * 64}  {incident.name}\n",
            encoding="utf-8",
        )
        sidecar.chmod(0o444)

    with pytest.raises(capture.EnvironmentCaptureError, match="checksum"):
        capture.capture_environments(**inputs)


@pytest.mark.parametrize("preimage", ("harness_metadata", "failure_log"))
def test_capture_revalidates_incident_external_preimages_at_creation(
    tmp_path, preimage
):
    inputs = _inputs(tmp_path)
    incident = json.loads(
        inputs["reconciliation_incident"].read_text(encoding="utf-8")
    )
    if preimage == "harness_metadata":
        target = Path(incident["runtime_owner"]["harness_metadata"])
        expected = "Setuptools METADATA differs"
    else:
        target = Path(incident["failed_materialization_log"])
        expected = "failed materialization log differs"
    target.write_bytes(target.read_bytes() + b"tamper\n")

    with pytest.raises(capture.EnvironmentCaptureError, match=expected):
        capture.capture_environments(**inputs)


def test_capture_requires_recovered_record_to_be_read_only(tmp_path):
    inputs = _inputs(tmp_path)
    inputs["recovered_setuptools_record"].chmod(0o644)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="read-only regular file",
    ):
        capture.capture_environments(**inputs)


def test_capture_normalizes_only_exact_bound_pip_record_rows(
    tmp_path, monkeypatch
):
    inputs, source_sha256, normalized_sha256 = (
        _inputs_with_pip_record_normalization(tmp_path, monkeypatch)
    )
    ownership, _ownership_sha256 = capture._load_policy(
        inputs["ownership_policy"]
    )
    integrity, _integrity_sha256 = capture._load_integrity_policy(
        inputs["integrity_normalization_policy"]
    )
    policy = capture._combined_policy(ownership, integrity)

    for role in capture.ROLES:
        source_report = capture.source_distribution_inventory(
            inputs[f"{role}_source"],
            policy=policy,
            role=role,
        )
        assert source_report["policy_normalized_record_count"] == 4
    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="exact approved postimage",
    ):
        capture.validate_normalized_seed(
            inputs["harness_source"], policy=policy
        )

    result = capture.capture_environments(**inputs, apply=True)
    assert result["status"] == "created"
    for role in capture.ROLES:
        seed = inputs["output_root"] / "seeds" / role
        record_path = (
            seed
            / "lib"
            / "python3.11"
            / "site-packages"
            / "pip-26.1.1.dist-info"
            / "RECORD"
        )
        assert hashlib.sha256(record_path.read_bytes()).hexdigest() == (
            normalized_sha256
        )
        assert all(
            path not in record_path.read_text(encoding="utf-8")
            for path, _encoded_hash, _size in capture.PIP_LAUNCHER_RECORD_ROWS
        )
        assert (seed / "bin" / "pip").read_bytes() == b"captured pip launcher\n"
        assert (seed / "bin" / "pip3").read_bytes() == b"captured pip3 launcher\n"
        paths = capture._normalization_paths(inputs["output_root"], role)
        assert hashlib.sha256(paths["pip_record_archive"].read_bytes()).hexdigest() == (
            source_sha256
        )
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        pip_receipt = receipt["pip_record_normalization"]
        assert pip_receipt["before_sha256"] == source_sha256
        assert pip_receipt["after_sha256"] == normalized_sha256
        assert pip_receipt["removed_row_count"] == 3
        assert pip_receipt["archived_preimage"]["sha256"] == source_sha256
        target_states = {
            row["record_path"]: row for row in pip_receipt["runtime_targets"]
        }
        assert target_states["../../../bin/pip"]["state"] == "regular_file"
        assert target_states["../../../bin/pip3"]["state"] == "regular_file"
        assert target_states["../../../bin/pip3.12"]["state"] == "missing"
        normalized = capture.validate_normalized_seed(seed, policy=policy)
        assert (
            normalized["pip_record_normalization_state"]
            == "exact_normalized_postimage"
        )
        stage = json.loads(
            (
                inputs["output_root"] / capture.ROLE_MARKERS[role]
            ).read_text(encoding="utf-8")
        )
        delta = stage["normalization_inventory_delta"]
        pip_record_relative = (
            "lib/python3.11/site-packages/"
            "pip-26.1.1.dist-info/RECORD"
        )
        expected_changed_paths = {pip_record_relative}
        if role == "serving":
            expected_changed_paths.add(
                "conda-meta/setuptools-82.0.1-pyh332efcf_0.json"
            )
        assert delta["protocol"] == (
            "pathwise-pre-post-full-inventory-diff-v1"
        )
        assert delta["exact_authorized_delta"] is True
        assert delta["runtime_changed_paths"] == []
        assert set(delta["actual_changed_paths"]) == expected_changed_paths
        assert set(delta["expected_changed_paths"]) == expected_changed_paths
        assert delta["record_canonicalization_paths"] == [
            pip_record_relative
        ]

        source_inventory = json.loads(
            (
                inputs["output_root"]
                / f"SOURCE_{role.upper()}_INVENTORY.json"
            ).read_text(encoding="utf-8")
        )
        before = {
            row["path"]: {
                key: value
                for key, value in row.items()
                if key != "mode"
            }
            for row in source_inventory["entries"]
        }
        after = {
            row["path"]: row
            for row in stage["normalized_content_inventory"]["entries"]
        }
        observed_changed_paths = {
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        }
        assert observed_changed_paths == expected_changed_paths


def test_capture_rejects_any_unlisted_runtime_path_change(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    real_normalize = capture._normalize_seed

    def normalize_then_mutate_runtime(**kwargs):
        receipt = real_normalize(**kwargs)
        (kwargs["seed"] / "bin" / "python").write_bytes(
            b"unauthorized runtime mutation\n"
        )
        return receipt

    monkeypatch.setattr(
        capture, "_normalize_seed", normalize_then_mutate_runtime
    )

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="normalization full-inventory delta is unauthorized",
    ):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()


def test_capture_resumes_after_atomic_pip_record_rewrite(
    tmp_path, monkeypatch
):
    inputs, _source_sha256, normalized_sha256 = (
        _inputs_with_pip_record_normalization(tmp_path, monkeypatch)
    )
    real_replace = capture._atomic_replace_bound
    crashed = False

    def crash_after_pip_rewrite(path, payload, **kwargs):
        nonlocal crashed
        status = real_replace(path, payload, **kwargs)
        if (
            not crashed
            and kwargs["description"]
            == (
                "harness "
                f"{capture.PIP_RECORD_NORMALIZATION_ID} normalization"
            )
        ):
            crashed = True
            raise OSError("simulated crash after atomic pip RECORD rewrite")
        return status

    monkeypatch.setattr(
        capture, "_atomic_replace_bound", crash_after_pip_rewrite
    )
    with pytest.raises(OSError, match="simulated crash"):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()

    recovered = capture.capture_environments(**inputs, apply=True)
    assert recovered["status"] == "created"
    harness_record = (
        inputs["output_root"]
        / "seeds"
        / "harness"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "pip-26.1.1.dist-info"
        / "RECORD"
    )
    assert hashlib.sha256(harness_record.read_bytes()).hexdigest() == (
        normalized_sha256
    )


def test_source_record_policy_does_not_hide_non_exempt_mismatch(
    tmp_path, monkeypatch
):
    inputs, _source_sha256, _normalized_sha256 = (
        _inputs_with_pip_record_normalization(tmp_path, monkeypatch)
    )
    ownership, _ownership_sha256 = capture._load_policy(
        inputs["ownership_policy"]
    )
    integrity, _integrity_sha256 = capture._load_integrity_policy(
        inputs["integrity_normalization_policy"]
    )
    policy = capture._combined_policy(ownership, integrity)
    metadata = (
        inputs["harness_source"]
        / "lib"
        / "python3.11"
        / "site-packages"
        / "pip-26.1.1.dist-info"
        / "METADATA"
    )
    original = metadata.read_bytes()
    metadata.write_bytes(b"X" + original[1:])

    with pytest.raises(capture.EnvironmentCaptureError, match="RECORD hash mismatch"):
        capture.source_distribution_inventory(
            inputs["harness_source"],
            policy=policy,
            role="harness",
        )


def test_policy_binds_every_record_repair_and_role():
    ownership, ownership_sha256 = capture._load_policy(OWNERSHIP_POLICY)
    integrity, integrity_sha256 = capture._load_integrity_policy(
        INTEGRITY_POLICY
    )
    assert ownership_sha256 == (
        "807507db3279f031b38ca55ab53c67d5c1fd87da1fd50fdc40897710aef95052"
    )
    assert integrity_sha256 == (
        "3f7c0cc7ccb7a5fec0ba23cefe40e30b4ed62c51c49120d3a1e4d3a8a11f4062"
    )
    assert [
        row["normalization_id"] for row in ownership["normalizations"]
    ] == [capture.SETUPTOOLS_NORMALIZATION_ID]
    policy = capture._combined_policy(ownership, integrity)
    by_id = {
        row["normalization_id"]: row for row in policy["normalizations"]
    }
    assert set(by_id) == {
        capture.SETUPTOOLS_NORMALIZATION_ID,
        *capture.RECORD_NORMALIZATION_IDS,
    }
    assert [
        row["normalization_id"]
        for row in capture._record_normalizations_for_role(
            policy, role="harness"
        )
    ] == [
        capture.PIP_RECORD_NORMALIZATION_ID,
        capture.PACKAGING_RECORD_NORMALIZATION_ID,
        capture.WHEEL_RECORD_NORMALIZATION_ID,
    ]
    assert [
        row["normalization_id"]
        for row in capture._record_normalizations_for_role(
            policy, role="serving"
        )
    ] == list(capture.RECORD_NORMALIZATION_IDS)

    torch = by_id[capture.TORCH_DLPACK_RECORD_NORMALIZATION_ID]
    assert torch["normalized_record_sha256"] == (
        "5f6e642b3bf84eaee6c86cb9de3aa0a6b570347abfe3f1fb0c6248afe607f36f"
    )
    assert set(capture._record_mutations(torch)) == {
        (
            "__pycache__/build_backend.cpython-311.pyc",
            "",
            "",
        ),
        (
            "build_backend.py",
            "sha256=WUgBRce7i-K7htTdoJJ2OVPSLVUAVv8Dv-cEta8gT4k",
            "3802",
        ),
    }
    assert all(after is None for after in capture._record_mutations(torch).values())
    assert torch["other_owner_contract"]["retained_row"] == {
        "path": "build_backend.py",
        "hash": "sha256=0DS7CRZ-adSPYf-93zoPLk9rQjPnbyleqhGzisU1tlc",
        "size": "5797",
    }
    for normalization_id in capture.RECORD_NORMALIZATION_IDS:
        assert not (
            capture._OWNERSHIP_CONFLICT_AUTHORIZATION_FIELDS
            & set(by_id[normalization_id])
        )


def _write_checksummed_policy(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def test_ownership_policy_rejects_a_second_normalization(tmp_path):
    ownership = json.loads(OWNERSHIP_POLICY.read_text(encoding="utf-8"))
    integrity = json.loads(INTEGRITY_POLICY.read_text(encoding="utf-8"))
    ownership["normalizations"].append(integrity["normalizations"][0])
    policy_path = tmp_path / "environment_ownership_policy.v1.json"
    _write_checksummed_policy(policy_path, ownership)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="environment ownership policy is invalid",
    ):
        capture._load_policy(policy_path)


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_integrity_policy_requires_exact_five_repairs(tmp_path, mutation):
    integrity = json.loads(INTEGRITY_POLICY.read_text(encoding="utf-8"))
    if mutation == "missing":
        integrity["normalizations"].pop()
    else:
        integrity["normalizations"].append(
            dict(integrity["normalizations"][0])
        )
    policy_path = (
        tmp_path / "environment_integrity_normalization_policy.v1.json"
    )
    _write_checksummed_policy(policy_path, integrity)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="environment integrity-normalization policy is invalid",
    ):
        capture._load_integrity_policy(policy_path)


def test_policy_rejects_ownership_authorization_on_record_repair(tmp_path):
    policy = json.loads(INTEGRITY_POLICY.read_text(encoding="utf-8"))
    pip_repair = next(
        row
        for row in policy["normalizations"]
        if row["normalization_id"] == capture.PIP_RECORD_NORMALIZATION_ID
    )
    pip_repair["pip_version"] = pip_repair["version"]
    policy_path = (
        tmp_path / "environment_integrity_normalization_policy.v1.json"
    )
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy_path.with_suffix(".sha256").write_text(
        f"{policy_sha256}  {policy_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="RECORD metadata repair cannot authorize",
    ):
        capture._load_integrity_policy(policy_path)


@pytest.mark.parametrize("normalization_id", capture.RECORD_NORMALIZATION_IDS)
def test_record_repairs_cannot_authorize_ownership_conflict(
    tmp_path, normalization_id
):
    ownership, _ownership_sha256 = capture._load_policy(OWNERSHIP_POLICY)
    integrity, _integrity_sha256 = capture._load_integrity_policy(
        INTEGRITY_POLICY
    )
    policy = capture._combined_policy(ownership, integrity)
    record_repair = dict(
        next(
            row
            for row in policy["normalizations"]
            if row["normalization_id"] == normalization_id
        )
    )
    # Even if a future caller synthesizes the fields used by the ownership
    # validator, a checksummed RECORD-repair entry is never an authorization.
    record_repair["pip_version"] = record_repair["version"]
    record_repair["conda_version"] = "999.0"
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="only the approved Setuptools normalization",
    ):
        capture._validate_ownership(
            prefix,
            normalization=record_repair,
            allow_known_conflict=True,
        )


@pytest.mark.parametrize("normalization_id", capture.RECORD_NORMALIZATION_IDS)
def test_record_repair_projects_remain_unlisted_version_conflicts(
    tmp_path, normalization_id
):
    ownership, _ownership_sha256 = capture._load_policy(OWNERSHIP_POLICY)
    integrity, _integrity_sha256 = capture._load_integrity_policy(
        INTEGRITY_POLICY
    )
    policy = capture._combined_policy(ownership, integrity)
    record_repair = next(
        row
        for row in policy["normalizations"]
        if row["normalization_id"] == normalization_id
    )
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    project = str(record_repair["project"])
    version = str(record_repair["version"])
    _runtime, _record = _add_conda_owned_distribution(
        prefix,
        project=project,
        version=version,
    )
    conda_path = prefix / "conda-meta" / f"{project}-{version}-py_0.json"
    conda_payload = json.loads(conda_path.read_text(encoding="utf-8"))
    conda_payload["version"] = "999.0"
    conda_path.write_text(
        json.dumps(conda_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="unlisted Conda/pip ownership version conflict",
    ):
        capture._validate_ownership(
            prefix,
            normalization=next(
                row
                for row in policy["normalizations"]
                if row["normalization_id"]
                == capture.SETUPTOOLS_NORMALIZATION_ID
            ),
            allow_known_conflict=True,
        )


@pytest.mark.parametrize("normalization_id", capture.RECORD_NORMALIZATION_IDS)
def test_record_repair_changes_only_exact_declared_rows(normalization_id):
    _ownership, _ownership_sha256 = capture._load_policy(OWNERSHIP_POLICY)
    policy, _integrity_sha256 = capture._load_integrity_policy(
        INTEGRITY_POLICY
    )
    normalization = next(
        row
        for row in policy["normalizations"]
        if row["normalization_id"] == normalization_id
    )
    mutations = capture._record_mutations(normalization)
    required = capture._required_retained_rows(normalization)
    untouched = (
        f"{normalization['project']}-fixture.dist-info/METADATA,"
        f"sha256={'A' * 43},17"
    )
    source_lines = [
        ",".join(row) for row in mutations
    ] + [
        ",".join(row) for row in required
    ] + [
        untouched,
        f"{normalization['project']}-fixture.dist-info/RECORD,,",
    ]
    expected_lines = [
        ",".join(after)
        for after in mutations.values()
        if after is not None
    ] + [
        ",".join(row) for row in required
    ] + [
        untouched,
        f"{normalization['project']}-fixture.dist-info/RECORD,,",
    ]
    source = ("\r\n".join(source_lines) + "\r\n").encode()
    expected = ("\r\n".join(expected_lines) + "\r\n").encode()
    synthetic = dict(normalization)
    synthetic["normalized_record_sha256"] = hashlib.sha256(expected).hexdigest()

    assert capture._normalized_record_bytes(source, synthetic) == expected

    duplicated = (
        ",".join(next(iter(mutations))).encode() + b"\r\n" + source
    )
    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="each authorized mutation row exactly once",
    ):
        capture._normalized_record_bytes(duplicated, synthetic)


def test_torch_shared_owner_repair_projects_and_produces_zero_shared_paths(
    tmp_path,
):
    prefix = _prefix(tmp_path, "serving", stale_setuptools_record=False)
    site = prefix / "lib" / "python3.11" / "site-packages"
    top_level = site / "build_backend.py"
    bytecode = site / "__pycache__" / "build_backend.cpython-311.pyc"
    bytecode.parent.mkdir()
    top_level.write_bytes(b"FlashInfer authoritative backend\n")
    bytecode.write_bytes(b"FlashInfer authoritative bytecode\n")

    flash_dist = site / "flashinfer_python-0.6.8.post1.dist-info"
    flash_dist.mkdir()
    flash_metadata = (
        b"Metadata-Version: 2.1\n"
        b"Name: flashinfer-python\n"
        b"Version: 0.6.8.post1\n\n"
    )
    (flash_dist / "METADATA").write_bytes(flash_metadata)
    (flash_dist / "RECORD").write_text(
        "build_backend.py,sha256="
        + _record_hash(top_level.read_bytes())
        + f",{top_level.stat().st_size}\n"
        "__pycache__/build_backend.cpython-311.pyc,,\n"
        "flashinfer_python-0.6.8.post1.dist-info/METADATA,sha256="
        + _record_hash(flash_metadata)
        + f",{len(flash_metadata)}\n"
        "flashinfer_python-0.6.8.post1.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    torch_dist = site / "torch_c_dlpack_ext-0.1.5.dist-info"
    torch_dist.mkdir()
    torch_metadata = (
        b"Metadata-Version: 2.1\n"
        b"Name: torch-c-dlpack-ext\n"
        b"Version: 0.1.5\n\n"
    )
    (torch_dist / "METADATA").write_bytes(torch_metadata)
    stale_top = (
        "build_backend.py",
        "sha256=WUgBRce7i-K7htTdoJJ2OVPSLVUAVv8Dv-cEta8gT4k",
        "3802",
    )
    stale_bytecode = (
        "__pycache__/build_backend.cpython-311.pyc",
        "",
        "",
    )
    torch_record = torch_dist / "RECORD"
    torch_record.write_text(
        ",".join(stale_bytecode)
        + "\n"
        + ",".join(stale_top)
        + "\n"
        "torch_c_dlpack_ext-0.1.5.dist-info/METADATA,sha256="
        + _record_hash(torch_metadata)
        + f",{len(torch_metadata)}\n"
        "torch_c_dlpack_ext-0.1.5.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    source = torch_record.read_bytes()
    runtime_preimage = {
        "build_backend.py": top_level.read_bytes(),
        "build_backend.pyc": bytecode.read_bytes(),
    }
    expected = b"".join(source.splitlines(keepends=True)[2:])
    normalization = {
        "normalization_id": capture.TORCH_DLPACK_RECORD_NORMALIZATION_ID,
        "project": "torch-c-dlpack-ext",
        "version": "0.1.5",
        "record_relative_path": (
            "lib/python3.11/site-packages/"
            "torch_c_dlpack_ext-0.1.5.dist-info/RECORD"
        ),
        "source_record_sha256": hashlib.sha256(source).hexdigest(),
        "normalized_record_sha256": hashlib.sha256(expected).hexdigest(),
        "roles": {
            "harness": {"source_record_policy": "forbidden"},
            "serving": {
                "source_record_policy": "exact_preimage_if_present"
            },
        },
        "row_mutations": [
            {
                "before": {
                    "path": row[0],
                    "hash": row[1],
                    "size": row[2],
                },
                "after": None,
            }
            for row in (stale_bytecode, stale_top)
        ],
    }

    source_report = capture.distribution_inventory(
        prefix,
        source_record_normalizations=[normalization],
        role="serving",
    )
    assert source_report["shared_record_path_count"] == 2
    assert source_report["projected_shared_record_path_count"] == 0
    assert source_report["policy_normalized_record_count"] == 2
    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="forbidden for harness",
    ):
        capture._source_record_authorization(
            prefix, normalization, role="harness"
        )

    torch_record.write_bytes(
        capture._normalized_record_bytes(source, normalization)
    )
    assert {
        "build_backend.py": top_level.read_bytes(),
        "build_backend.pyc": bytecode.read_bytes(),
    } == runtime_preimage
    normalized_report = capture.distribution_inventory(prefix)
    assert normalized_report["policy_normalized_record_count"] == 0
    assert normalized_report["shared_record_path_count"] == 0
    assert normalized_report["projected_shared_record_path_count"] == 0


def test_capture_fails_closed_on_nonzero_projected_shared_ownership(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    real_validate = capture._validate_ownership

    def inject_unresolved_shared_owner(*args, **kwargs):
        distributions, conda = real_validate(*args, **kwargs)
        distributions = dict(distributions)
        distributions["projected_shared_record_path_count"] = 1
        return distributions, conda

    monkeypatch.setattr(
        capture, "_validate_ownership", inject_unresolved_shared_owner
    )
    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="source RECORD repairs project 1 shared paths",
    ):
        capture.capture_environments(**inputs)
    assert not inputs["output_root"].exists()


def _regular_inodes(root: Path):
    for directory, _directories, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            if path.is_file() and not path.is_symlink():
                info = path.stat()
                yield info.st_dev, info.st_ino


def stat_write_bits(root: Path) -> int:
    value = root.stat().st_mode & 0o222
    for directory, directories, filenames in os.walk(root, followlinks=False):
        for name in [*directories, *filenames]:
            path = Path(directory) / name
            if not path.is_symlink():
                value |= path.stat().st_mode & 0o222
    return value


def _rewrite_capture_json(path: Path, payload: dict) -> None:
    path.chmod(0o644)
    path.write_bytes(capture._json_bytes(payload))
    path.chmod(0o444)


def _rehash_capture_intent_and_marker(
    root: Path,
    *,
    mutate_intent,
) -> None:
    intent_path = root / capture.INTENT_MARKER
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    mutate_intent(intent)
    intent.pop("intent_id", None)
    intent["intent_id"] = capture._sha256_bytes(
        capture._canonical_bytes(intent)
    )
    _rewrite_capture_json(intent_path, intent)

    marker_path = root / capture.COMPLETE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, value in intent.items():
        marker[key] = value
    marker.pop("capture_id", None)
    marker["capture_id"] = capture._sha256_bytes(
        capture._canonical_bytes(marker)
    )
    _rewrite_capture_json(marker_path, marker)


def test_sealed_capture_rejects_self_consistent_completion_marker_forgery(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    capture.capture_environments(**inputs, apply=True)
    marker_path = inputs["output_root"] / capture.COMPLETE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["source_prefixes"]["harness"] = str(tmp_path / "substituted")
    marker.pop("capture_id")
    marker["capture_id"] = capture._sha256_bytes(
        capture._canonical_bytes(marker)
    )
    _rewrite_capture_json(marker_path, marker)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="intent/marker binding drifted",
    ):
        capture.verify_capture(inputs["output_root"])


def test_sealed_capture_rejects_rehashed_source_distribution_substitution(
    tmp_path,
):
    inputs = _inputs(tmp_path)
    capture.capture_environments(**inputs, apply=True)
    root = inputs["output_root"]
    audit_path = root / capture.SOURCE_DISTRIBUTION_AUDIT_FILES["harness"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["distribution_count"] += 1
    _rewrite_capture_json(audit_path, audit)

    def mutate_intent(intent):
        binding = intent["source_distribution_audits"]["harness"]
        binding["sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        binding["distribution_count"] = audit["distribution_count"]

    _rehash_capture_intent_and_marker(root, mutate_intent=mutate_intent)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="not reconstructible from sealed evidence",
    ):
        capture.verify_capture(root)


def test_sealed_capture_rejects_rehashed_copy_audit_substitution(tmp_path):
    inputs = _inputs(tmp_path)
    capture.capture_environments(**inputs, apply=True)
    root = inputs["output_root"]
    role = "harness"
    stage_path = root / capture.ROLE_MARKERS[role]
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["copy_audit"]["source_regular_file_count"] += 1
    stage.pop("record_sha256")
    stage["record_sha256"] = capture._sha256_bytes(
        capture._canonical_bytes(stage)
    )
    _rewrite_capture_json(stage_path, stage)

    marker_path = root / capture.COMPLETE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["stage_records"][role]["sha256"] = hashlib.sha256(
        stage_path.read_bytes()
    ).hexdigest()
    marker["stage_records"][role]["record_sha256"] = stage[
        "record_sha256"
    ]
    marker.pop("capture_id")
    marker["capture_id"] = capture._sha256_bytes(
        capture._canonical_bytes(marker)
    )
    _rewrite_capture_json(marker_path, marker)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="capture identity drifted",
    ):
        capture.verify_capture(root)


def test_capture_rejects_unlisted_ownership_conflict(tmp_path):
    inputs = _inputs(tmp_path)
    prefix = inputs["harness_source"]
    metadata = (
        prefix / "lib" / "python3.11" / "site-packages" / "python-9.dist-info"
    )
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: python\nVersion: 9\n\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text("python-9.dist-info/RECORD,,\n", encoding="utf-8")

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="unlisted Conda/pip ownership",
    ):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()


def test_capture_rejects_bad_pip_record_hash(tmp_path):
    inputs = _inputs(tmp_path)
    package = (
        inputs["serving_source"]
        / "lib"
        / "python3.11"
        / "site-packages"
        / "setuptools"
        / "__init__.py"
    )
    # Preserve the declared size so this proves the pip81 hash is checked even
    # while the real conda82/pip81 ownership collision is present.
    package.write_text("__version__ = '80.0.0'\n", encoding="utf-8")

    with pytest.raises(
        capture.EnvironmentCaptureError, match="pip RECORD hash mismatch"
    ):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()


def test_conda_owned_distribution_does_not_bypass_record_hashes(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    package_path, _record_path = _add_conda_owned_distribution(prefix)

    inventory = capture.distribution_inventory(prefix)
    row = next(
        item
        for item in inventory["distributions"]
        if item["project"] == "owned-project"
    )
    assert row["conda_owned_same_version"] is True
    assert row["record_validation"] == "all_declared_pip_record_hashes"

    original = package_path.read_bytes()
    package_path.write_bytes(b"X" + original[1:])
    with pytest.raises(capture.EnvironmentCaptureError, match="RECORD hash mismatch"):
        capture.distribution_inventory(prefix)


@pytest.mark.parametrize(
    ("replacement", "error"),
    (
        ("sha1=" + "a" * 43, "invalid or non-SHA256"),
        ("sha256=short", "invalid pip RECORD SHA256 digest"),
        ("sha256=" + "a" * 42 + "*", "invalid pip RECORD SHA256 digest"),
    ),
)
def test_record_rejects_invalid_hash_algorithms_and_digests(
    tmp_path, replacement, error
):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _package_path, record_path = _add_conda_owned_distribution(prefix)
    text = record_path.read_text(encoding="utf-8")
    text = text.replace(
        "sha256=" + _record_hash(b"authoritative installed bytes\n"),
        replacement,
        1,
    )
    record_path.write_text(text, encoding="utf-8")

    with pytest.raises(capture.EnvironmentCaptureError, match=error):
        capture.distribution_inventory(prefix)


@pytest.mark.parametrize(
    ("relative", "error"),
    (
        ("missing-project.py", "references a missing path"),
        ("../../../../../outside", "escapes environment prefix"),
        ("/etc/passwd", "unsafe pip RECORD path"),
        ("owned_project/../owned_project/__init__.py", "non-canonical"),
    ),
)
def test_record_rejects_missing_escaping_and_unsafe_paths(
    tmp_path, relative, error
):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _package_path, record_path = _add_conda_owned_distribution(prefix)
    existing = record_path.read_text(encoding="utf-8")
    record_path.write_text(f"{relative},,\n{existing}", encoding="utf-8")

    with pytest.raises(capture.EnvironmentCaptureError, match=error):
        capture.distribution_inventory(prefix)


def test_record_accepts_spec_unhashed_self_and_generated_bytecode(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _package_path, record_path = _add_conda_owned_distribution(prefix)
    cache = (
        prefix
        / "lib"
        / "python3.11"
        / "site-packages"
        / "owned_project"
        / "__pycache__"
        / "__init__.cpython-311.pyc"
    )
    cache.parent.mkdir()
    cache.write_bytes(b"generated bytecode\n")
    existing = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        "owned_project/__pycache__/__init__.cpython-311.pyc,,\n" + existing,
        encoding="utf-8",
    )

    inventory = capture.distribution_inventory(prefix)
    assert inventory["runtime_cache_record_count"] == 1
    assert inventory["missing_unhashed_runtime_cache_record_count"] == 0
    # One generated bytecode row plus one RECORD self-row for each distribution.
    assert inventory["unhashed_record_count"] == 3


def test_record_audits_missing_unhashed_generated_bytecode(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _package_path, record_path = _add_conda_owned_distribution(prefix)
    missing_relative = (
        "owned_project/__pycache__/__init__.cpython-311.pyc"
    )
    existing = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        f"{missing_relative},,\n{existing}",
        encoding="utf-8",
    )

    inventory = capture.distribution_inventory(prefix)
    row = next(
        item
        for item in inventory["distributions"]
        if item["project"] == "owned-project"
    )
    assert inventory["missing_unhashed_runtime_cache_record_count"] == 1
    assert row["missing_unhashed_runtime_cache_record_count"] == 1
    assert row["missing_unhashed_runtime_cache_paths"] == [missing_relative]


def test_record_rejects_missing_hash_bearing_file(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    package_path, _record_path = _add_conda_owned_distribution(prefix)
    package_path.unlink()

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="pip RECORD references a missing path",
    ):
        capture.distribution_inventory(prefix)


def test_record_rejects_missing_size_bearing_generated_bytecode(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _package_path, record_path = _add_conda_owned_distribution(prefix)
    existing = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        "owned_project/__pycache__/missing.cpython-311.pyc,,17\n" + existing,
        encoding="utf-8",
    )

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="pip RECORD references a missing path",
    ):
        capture.distribution_inventory(prefix)


def test_distribution_inventory_rejects_normalized_duplicate_projects(tmp_path):
    prefix = _prefix(tmp_path, "prefix", stale_setuptools_record=False)
    _add_conda_owned_distribution(prefix, project="owned_project")
    site = prefix / "lib" / "python3.11" / "site-packages"
    duplicate = site / "owned.project-9.9.dist-info"
    duplicate.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: owned.project\nVersion: 9.9\n\n"
    (duplicate / "METADATA").write_bytes(metadata)
    (duplicate / "RECORD").write_text(
        f"{duplicate.name}/METADATA,sha256={_record_hash(metadata)},{len(metadata)}\n"
        f"{duplicate.name}/RECORD,,\n",
        encoding="utf-8",
    )

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="duplicate installed distribution 'owned-project'",
    ):
        capture.distribution_inventory(prefix)


def test_capture_rejects_shared_inode(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_bytes(b"payload")
    os.link(source / "file", destination / "file")

    with pytest.raises(capture.EnvironmentCaptureError, match="shares 1"):
        capture.verify_no_shared_regular_inodes(source, destination)


def test_capture_rejects_source_drift_during_copy(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    real_copy = capture._copy_regular_file
    changed = False

    def mutate_after_copy(source, destination, *, mode):
        nonlocal changed
        real_copy(source, destination, mode=mode)
        if not changed:
            changed = True
            (
                inputs["harness_source"]
                / "lib"
                / "python3.11"
                / "site-packages"
                / "setuptools"
                / "__init__.py"
            ).write_text("drift\n", encoding="utf-8")

    monkeypatch.setattr(capture, "_copy_regular_file", mutate_after_copy)
    with pytest.raises(capture.EnvironmentCaptureError, match="source drifted"):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()


def test_capture_rejects_external_symlink(tmp_path):
    inputs = _inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("outside\n", encoding="utf-8")
    (inputs["serving_source"] / "external").symlink_to(outside)

    with pytest.raises(capture.EnvironmentCaptureError, match="external symlink"):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()


@pytest.mark.parametrize(
    "crash_name",
    ("harness.receipt.json", capture.ROLE_MARKERS["harness"]),
)
def test_capture_resumes_across_normalization_publication_boundaries(
    tmp_path, monkeypatch, crash_name
):
    inputs = _inputs(tmp_path)
    real_publish = capture._atomic_write_once
    crashed = False

    def crash_once(path, payload, **kwargs):
        nonlocal crashed
        if not crashed and Path(path).name == crash_name:
            crashed = True
            raise OSError("simulated normalization publication crash")
        return real_publish(path, payload, **kwargs)

    monkeypatch.setattr(capture, "_atomic_write_once", crash_once)
    with pytest.raises(OSError, match="simulated normalization"):
        capture.capture_environments(**inputs, apply=True)
    assert not (inputs["output_root"] / capture.COMPLETE_MARKER).exists()

    monkeypatch.setattr(capture, "_atomic_write_once", real_publish)
    recovered = capture.capture_environments(**inputs, apply=True)
    assert recovered["status"] == "created"
    assert capture.verify_capture(inputs["output_root"])["capture_id"] == recovered[
        "capture_id"
    ]


def test_capture_publication_never_launders_writable_preimage(tmp_path):
    artifact = tmp_path / "CAPTURE_STAGE.json"
    artifact.write_bytes(b"exact bytes\n")
    artifact.chmod(0o644)

    with pytest.raises(
        capture.EnvironmentCaptureError,
        match="conflicting immutable capture artifact",
    ):
        capture._atomic_write_once(artifact, b"exact bytes\n")

    assert artifact.stat().st_mode & 0o222
