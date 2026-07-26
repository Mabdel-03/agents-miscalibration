"""SLURM array entrypoint: run the cell at ``--index`` from a cells JSON file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.artifact_policy import (
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.runner import Schema5RuntimeProvenance, run_cell
from agents_scaling.experiment.qid_checkpoint import CoordinateAdmissionClosed
from agents_scaling import runtime_integrity
from agents_scaling.serving.model_contracts import ModelContractError, load_model_contracts


def _runtime_integrity_coordinate_guard() -> None:
    """O(1) lease check immediately before each new stochastic coordinate."""

    try:
        runtime_integrity.verify_generation_lease(
            lease_path=Path(os.environ["ASYS_RUNTIME_INTEGRITY_LEASE"]),
            attestation_path=Path(os.environ["ASYS_RUNTIME_ATTESTATION"]),
            attestation_sha256=os.environ["ASYS_RUNTIME_ATTESTATION_SHA256"],
            generation=int(os.environ["ASYS_ROLLOUT_GENERATION"]),
            release_id=os.environ["ASYS_RELEASE_ID"],
            immutable_pins_sha256=os.environ["ASYS_IMMUTABLE_PINS_SHA256"],
            expected_environment_hashes={
                "harness": os.environ["ASYS_HARNESS_ENVIRONMENT_SHA256"],
                "serving": os.environ["ASYS_SERVING_ENVIRONMENT_SHA256"],
            },
        )
    except (KeyError, ValueError, runtime_integrity.RuntimeIntegrityError) as exc:
        raise CoordinateAdmissionClosed(
            f"runtime integrity lease unavailable; no new coordinate admitted: {exc}"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one sweep cell by array index.")
    ap.add_argument("--run-id", required=True)
    pool = ap.add_mutually_exclusive_group()
    pool.add_argument(
        "--server-pool",
        help=(
            "explicit run id or absolute run root whose endpoint registry should serve "
            "this cell; results and locks remain under --run-id"
        ),
    )
    pool.add_argument(
        "--server-run-id",
        help=argparse.SUPPRESS,  # compatibility for already-rendered legacy arrays
    )
    ap.add_argument("--cells-file", required=True, help="JSON list of ExperimentCell dicts")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument(
        "--benchmark-contracts-sha256",
        help="dispatcher-pinned exact benchmark sidecar byte hash",
    )
    ap.add_argument(
        "--artifact-policy-sha256",
        default=os.environ.get("ASYS_ARTIFACT_POLICY_SHA256"),
        help="dispatcher-pinned exact artifact-policy byte hash",
    )
    ap.add_argument(
        "--release-id",
        default=os.environ.get("ASYS_RELEASE_ID"),
        help="immutable production release id",
    )
    ap.add_argument(
        "--environment-hash",
        default=(
            os.environ.get("ASYS_HARNESS_ENVIRONMENT_SHA256")
            or os.environ.get("ASYS_HARNESS_ENVIRONMENT_HASH")
        ),
        help="immutable harness-environment directory hash",
    )
    ap.add_argument(
        "--serving-environment-hash",
        default=(
            os.environ.get("ASYS_SERVING_ENVIRONMENT_SHA256")
            or os.environ.get("ASYS_SERVING_ENVIRONMENT_HASH")
        ),
        help="immutable serving-environment directory hash expected from endpoints",
    )
    ap.add_argument(
        "--model-revision",
        default=os.environ.get("ASYS_MODEL_REVISION"),
        help="exact frozen model commit",
    )
    ap.add_argument(
        "--tokenizer-revision",
        default=os.environ.get("ASYS_TOKENIZER_REVISION"),
        help="exact frozen tokenizer commit",
    )
    ap.add_argument(
        "--model-contract-sha256",
        default=os.environ.get("ASYS_MODEL_CONTRACT_SHA256"),
        help="exact frozen model-contract byte hash",
    )
    ap.add_argument(
        "--release-worktree",
        type=Path,
        help="exact immutable source worktree supplying production runtime resources",
    )
    ap.add_argument(
        "--model-contract",
        type=Path,
        help="exact model-contract file below --release-worktree",
    )
    ap.add_argument(
        "--fleet-contract",
        type=Path,
        help="exact fleet-contract file below --release-worktree",
    )
    ap.add_argument(
        "--prompt-root",
        type=Path,
        help="exact prompt directory below --release-worktree",
    )
    ap.add_argument(
        "--fleet-contract-sha256",
        default=os.environ.get("ASYS_FLEET_CONTRACT_SHA256"),
        help="exact frozen serving-fleet contract byte hash",
    )
    ap.add_argument(
        "--release-fleet-contract-sha256",
        default=os.environ.get("ASYS_RELEASE_FLEET_CONTRACT_SHA256"),
        help="immutable release-bundle fleet hash retained as capacity lineage",
    )
    ap.add_argument(
        "--capacity-generation",
        type=int,
        default=(
            int(os.environ["ASYS_CAPACITY_GENERATION"])
            if os.environ.get("ASYS_CAPACITY_GENERATION")
            else None
        ),
        help="positive controlled fleet/capacity generation",
    )
    ap.add_argument(
        "--rollout-generation",
        type=int,
        default=(
            int(os.environ["ASYS_ROLLOUT_GENERATION"])
            if os.environ.get("ASYS_ROLLOUT_GENERATION")
            else None
        ),
        help="positive desired-state rollout generation",
    )
    ap.add_argument("--judge-prompts", action="store_true", help="score prompt quality with LLM judge")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells_file).read_text())
    if not 0 <= args.index < len(cells):
        raise IndexError(f"index {args.index} out of range for {len(cells)} cells")
    cell = ExperimentCell.from_dict(cells[args.index])
    observed_provenance = {
        "artifact_policy_sha256": args.artifact_policy_sha256,
        "release_id": args.release_id,
        "environment_hash": args.environment_hash,
        "serving_environment_hash": args.serving_environment_hash,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "model_contract_sha256": args.model_contract_sha256,
        "fleet_contract_sha256": args.fleet_contract_sha256,
        "release_fleet_contract_sha256": args.release_fleet_contract_sha256,
        "capacity_generation": args.capacity_generation,
        "rollout_generation": args.rollout_generation,
    }
    run_root = io.results_root() / args.run_id
    try:
        policy = load_artifact_policy(
            run_root,
            expected_file_sha256=args.artifact_policy_sha256,
        )
    except ArtifactPolicyError as exc:
        ap.error(f"schema-5 artifact policy failed closed: {exc}")
    runtime_provenance = None
    coordinate_admission_guard = None
    if policy is not None:
        # Policy and model sidecars are authority.  Scheduler/control values are observed
        # attestations and may only agree with those frozen bytes; they never define the
        # scientific identity of a task.
        resource_values = {
            "release_worktree": args.release_worktree,
            "model_contract": args.model_contract,
            "fleet_contract": args.fleet_contract,
            "prompt_root": args.prompt_root,
        }
        missing_resources = [
            name for name, value in resource_values.items() if value is None
        ]
        if missing_resources:
            ap.error(
                "authoritative schema-5 worker is missing immutable runtime resources: "
                + ", ".join(missing_resources)
            )
        assert args.release_worktree is not None
        assert args.model_contract is not None
        assert args.fleet_contract is not None
        assert args.prompt_root is not None
        if not args.release_worktree.is_absolute():
            ap.error("schema-5 release worktree must be absolute")
        release_worktree = args.release_worktree.resolve()
        expected_resources = {
            "model_contract": release_worktree / "configs" / "model_contracts.v1.json",
            "prompt_root": release_worktree / "configs" / "prompts",
        }
        observed_resources = {
            "model_contract": args.model_contract.resolve(),
            "prompt_root": args.prompt_root.resolve(),
        }
        drifted_resources = [
            name
            for name, expected_path in expected_resources.items()
            if observed_resources[name] != expected_path
        ]
        if drifted_resources:
            ap.error(
                "schema-5 runtime resources are outside the immutable release layout: "
                + ", ".join(drifted_resources)
            )
        if not args.fleet_contract.is_absolute():
            ap.error("schema-5 fleet contract must be an absolute control-pinned path")
        if args.release_worktree.is_symlink() or not release_worktree.is_dir():
            ap.error("schema-5 release worktree is missing or symlinked")
        for name in ("model_contract", "fleet_contract"):
            raw_path = resource_values[name]
            assert isinstance(raw_path, Path)
            if raw_path.is_symlink() or not raw_path.is_file():
                ap.error(f"schema-5 {name} is missing or symlinked")
        if args.prompt_root.is_symlink() or not args.prompt_root.is_dir():
            ap.error("schema-5 prompt root is missing or symlinked")
        for level in range(4):
            prompt = args.prompt_root / f"level{level}.txt"
            if prompt.is_symlink() or not prompt.is_file():
                ap.error(f"schema-5 prompt resource is missing or symlinked: {prompt}")
        try:
            contracts = load_model_contracts(
                args.model_contract,
                expected_sha256=policy.accepted_model_contract_sha256
            )
            identity = contracts.for_size(cell.model_size)
        except ModelContractError as exc:
            ap.error(f"schema-5 model contract failed closed: {exc}")
        required_observed = {
            "release_id": args.release_id,
            "environment_hash": args.environment_hash,
            "rollout_generation": args.rollout_generation,
        }
        missing_observed = [
            name for name, value in required_observed.items() if value is None
        ]
        if missing_observed:
            ap.error(
                "authoritative schema-5 worker is missing control attestations: "
                + ", ".join(missing_observed)
            )
        expected = {
            "artifact_policy_sha256": policy.file_sha256,
            "release_id": policy.release.release_id,
            "environment_hash": policy.environment.harness_sha256,
            "serving_environment_hash": policy.environment.serving_sha256,
            "model_revision": identity.model_revision,
            "tokenizer_revision": identity.tokenizer_revision,
            "model_contract_sha256": policy.accepted_model_contract_sha256,
        }
        if not args.fleet_contract_sha256:
            ap.error(
                "authoritative schema-5 worker is missing fleet_contract_sha256"
            )
        if not args.release_fleet_contract_sha256:
            ap.error(
                "authoritative schema-5 worker is missing "
                "release_fleet_contract_sha256"
            )
        if (
            not isinstance(args.capacity_generation, int)
            or args.capacity_generation < 1
        ):
            ap.error("capacity_generation must be a positive integer")
        for field, expected_value in expected.items():
            observed = observed_provenance[field]
            if observed is not None and observed != expected_value:
                ap.error(
                    f"observed {field} does not match verified schema-5 sidecars"
                )
        if (
            not isinstance(args.rollout_generation, int)
            or args.rollout_generation < 1
        ):
            ap.error("rollout_generation must be a positive integer")
        runtime_provenance = Schema5RuntimeProvenance(
            **expected,
            fleet_contract_sha256=args.fleet_contract_sha256,
            release_fleet_contract_sha256=args.release_fleet_contract_sha256,
            capacity_generation=args.capacity_generation,
            rollout_generation=args.rollout_generation,
            release_worktree=str(release_worktree),
            model_contract_path=str(args.model_contract.resolve()),
            fleet_contract_path=str(args.fleet_contract.resolve()),
            prompt_root=str(args.prompt_root.resolve()),
        )
        # Make the same verified authority available to lower-level schema validators
        # that are also used outside the runner.  This is process-local and is set only
        # after every explicit resource path has matched the immutable release layout.
        os.environ["ASYS_RELEASE_WORKTREE"] = str(release_worktree)
        coordinate_admission_guard = _runtime_integrity_coordinate_guard
    elif any(value is not None for value in observed_provenance.values()) or any(
        value is not None
        for value in (
            args.release_worktree,
            args.model_contract,
            args.fleet_contract,
            args.prompt_root,
        )
    ):
        ap.error(
            "production provenance/control attestations were supplied to a run without "
            "an authoritative schema-5 artifact policy"
        )
    # Use the global cell index as the shard so cells fan out across server endpoints.
    path = run_cell(
        cell,
        args.run_id,
        score_prompt_with_judge=args.judge_prompts,
        shard=args.index,
        server_run_id=args.server_pool or args.server_run_id,
        expected_benchmark_contracts_sha256=args.benchmark_contracts_sha256,
        runtime_provenance=runtime_provenance,
        coordinate_admission_guard=coordinate_admission_guard,
    )
    print(f"[run_one] cell {cell.cell_id} -> {path}")


if __name__ == "__main__":
    main()
