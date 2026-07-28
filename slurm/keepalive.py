"""Server keepalive: maintain a DESIRED REPLICA COUNT per serving profile for a long sweep.

Servers have walltime limits (pi_tpoggio 7-day, ou_bcs 1-day), so over a multi-week run
they get killed and cells then block/round-robin onto dead endpoints. This loop, each pass:
  1. probes a real HTTP /health on every registered endpoint (presence != liveness);
  2. prunes the registry file of any dead endpoint (so cells stop hitting it);
  3. for each managed size, if (live endpoints) + (serve jobs already PD/R) < desired count,
     relaunches enough replicas on that size's designated partition to reach the target.

The target is a **replica spec**: ``profile:count:partition:time`` entries, e.g.
``0.6B:1:pi_tpoggio:7-00:00:00,32B:6:ou_bcs_low:1-00:00:00``. Replica indices are assigned
to keep ports distinct (see launch_server._port_for(size, replica)).

Legacy ``--spec`` operation remains count-based.  Schema-5 ``--fleet-contract``
operation is durable and transactional: it joins squeue+sacct under a pool lock, adopts
crash-window submissions by intent token, and persistently fences genuinely hung jobs.

Usage:
  python slurm/keepalive.py --run-id full_sweep_v1 --interval 600 \
      --spec 0.6B:1:pi_tpoggio:7-00:00:00,1.7B:2:ou_bcs_low:1-00:00:00,...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving import fleet_transactions as fleet_tx
from agents_scaling.serving import protected_capacity, scheduler_safety
from agents_scaling.serving.fleet_contract import (
    FleetContractError,
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.launch_server import (
    _production_release_resources,
    render_sbatch,
    submit as submit_server,
)
from agents_scaling.serving.model_contracts import ModelContractError, load_model_contracts
from agents_scaling.serving.profiles import SERVING_PROFILES, get_serving_profile


@dataclass
class Target:
    size: str
    count: int          # desired number of live endpoints for this size
    partition: str      # where to launch (re)replicas
    time_limit: str
    gpu_type: str = "a100"


# Public compatibility for the existing focused tests and forensic helpers.  Production
# rows now come from a joined squeue+sacct snapshot rather than current squeue alone.
FleetQueueRow = fleet_tx.SchedulerRow


class FleetQueueTruth(tuple):
    """Tuple-compatible rows retaining their exact complete scheduler snapshot."""

    scheduler_snapshot: fleet_tx.SchedulerSnapshot

    def __new__(
        cls, snapshot: fleet_tx.SchedulerSnapshot
    ) -> "FleetQueueTruth":
        material = super().__new__(cls, snapshot.rows)
        material.scheduler_snapshot = snapshot
        return material


@dataclass(frozen=True)
class SpooledServingProvenance:
    """Launch facts recovered from Slurm's immutable copy of one batch script."""

    run_root: str
    server_pool_id: str
    replica_id: str | int
    replica_index: int
    release_id: str | None
    environment_hash: str | None
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_contract_sha256: str
    fleet_contract_sha256: str | None
    release_fleet_contract_sha256: str | None = None
    capacity_generation: int | None = None
    rollout_generation: int | None = None
    spooled_script_sha256: str | None = None


@dataclass(frozen=True)
class ReadOnlyFleetAllocation:
    """One fully verified live allocation from transactional scheduler truth."""

    replica_id: str
    ledger_generation: int
    intent_token: str
    attempt_state: str
    row: FleetQueueRow
    spooled_provenance: SpooledServingProvenance
    sbatch_path: str
    sbatch_sha256: str


@dataclass(frozen=True)
class ReadOnlyFleetSnapshot:
    """A mutation-free readiness view of the canonical production fleet."""

    current_generation: int
    captured_at: float
    allocations: tuple[ReadOnlyFleetAllocation, ...]
    ignored_terminal_job_ids: tuple[str, ...]
    isolated_foreign_job_ids: tuple[str, ...] = ()
    sealed_successful_handoff_terminal_job_ids: tuple[str, ...] = ()


def _validate_live_fleet_scheduler_policy(
    fleet: FrozenFleetContract,
    *,
    expected_policy_contract_id: str | None,
    expected_transport_binding_sha256: str,
    runner=None,
    captured_timestamp: float | None = None,
) -> tuple[dict, dict]:
    """Capture and validate the exact scheduler policy used by one fleet tick."""

    binding_sha256 = scheduler_safety.transport_uncertainty_binding_sha256()
    if expected_transport_binding_sha256 != binding_sha256:
        raise FleetContractError(
            "fleet supervisor transport-uncertainty binding drifted"
        )
    requirements = scheduler_safety.fleet_partition_time_requirements(
        fleet.replicas
    )
    try:
        evidence = scheduler_safety.capture_scheduler_safety_evidence(
            list(requirements),
            runner=runner,
            captured_timestamp=captured_timestamp,
        )
        policy = scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=list(requirements),
            required_time_limits_seconds=requirements,
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise FleetContractError(
            f"live fleet scheduler-safety validation failed: {exc}"
        ) from exc
    if (
        expected_policy_contract_id is not None
        and policy["policy_contract_id"] != expected_policy_contract_id
    ):
        raise FleetContractError(
            "live fleet scheduler policy drifted from attested readiness"
        )
    return evidence, policy


def parse_spec(spec: str, default_gpu: str) -> list[Target]:
    """Parse 'size:count:partition:time[:gpu_type][,...]' into Targets.

    Time contains colons (e.g. 7-00:00:00), so we FIRST peel an optional trailing
    ``:gpu_type`` — recognizable because a time field is digit-led ('7-00:...') while a GPU
    label is alpha-led ('h100','a100'). Then split the rest into exactly 4 fields from the
    left. This lets entries on different partitions request different GPUs (h100 on
    ou_bcs_high, a100 on pi_tpoggio); a 4-field entry stays backward compatible via
    ``default_gpu``."""
    targets: list[Target] = []
    for item in [x for x in spec.split(",") if x.strip()]:
        gpu = default_gpu
        head, _, last = item.rpartition(":")
        if head and last[:1].isalpha():  # trailing alpha token => gpu_type, not a time field
            gpu = last
            item = head
        parts = item.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                f"bad spec entry {item!r}; expected size:count:partition:time[:gpu_type]"
            )
        size, count, partition, time_limit = parts
        targets.append(Target(size, int(count), partition, time_limit, gpu))
    return targets


def _pool_job_name(run_root: str | None, profile_name: str) -> str:
    return (
        registry.serving_job_name(run_root, profile_name)
        if run_root is not None
        else f"asys-serve-{profile_name}"
    )


def _serve_jobs_in_flight(
    model_size: str, partition: str | None = None, *, run_root: str | None = None
) -> int:
    """Count active serve jobs for this size, optionally scoped to ``partition``.

    This is the authoritative "current + coming" server count: every server — running &
    registered, running but still loading, pending, suspended, completing, or in Slurm's
    transient requeue states — has exactly one ``squeue`` row.  Count every matching row
    instead of whitelisting state codes: a preempted ``--requeue`` job briefly reports a
    requeue state, and treating that as absent launches duplicate replicas and consumes
    GPU quota.  Terminal jobs are absent from ``squeue`` already.  The live-endpoint set
    is a subset of these jobs, so we must not add live endpoints to this count.

    ``partition`` scoping lets the SAME size be served from multiple partitions with
    independent target counts (e.g. 1 replica on pi_tpoggio + 2 bonus on ou_bcs_high): each
    target maintains only its own partition's jobs instead of the two targets fighting over
    one size-global count.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%t|%P"],
        capture_output=True, text=True,
    ).stdout
    name = _pool_job_name(run_root, model_size)
    n = 0
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jname, _state, jpart = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if jname == name and (partition is None or jpart == partition):
            n += 1
    return n


def _running_serve_nodes(model_size: str) -> list[str]:
    """Nodes of currently-RUNNING serve jobs for this size."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "R", "-o", "%j %N"],
        capture_output=True, text=True,
    ).stdout
    name = f"asys-serve-{model_size}"
    nodes = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            nodes.append(parts[1])
    return nodes


def _running_serve_endpoints(
    model_size: str, *, run_root: str | None = None
) -> dict[tuple[str, int], str]:
    """Map each running profile endpoint to its authoritative Slurm job id.

    Re-registration used to restore only host/port, losing the job identity needed to
    distinguish a live allocation from a stale registry file.  The rendered sbatch
    command contains the replica id, so the endpoint can be reconstructed without an
    HTTP or filesystem guess.
    """
    from agents_scaling.serving.launch_server import _port_for

    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "R", "-o", "%i|%j|%N|%o"],
        capture_output=True,
        text=True,
    ).stdout
    job_name = _pool_job_name(run_root, model_size)
    filename = re.compile(
        rf"(?:^|/)serve_{re.escape(model_size)}(?:_r(?P<rid>\d+))?"
        rf"(?:\.g\d+\.[0-9a-f]+)?\.sbatch(?:\s|$)"
    )
    endpoints: dict[tuple[str, int], str] = {}
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        job_id, name, node, command = (part.strip() for part in parts)
        if name != job_name or not node or node in {"(null)", "N/A"}:
            continue
        match = filename.search(command)
        if match is None:
            continue
        replica = int(match.group("rid") or 0)
        endpoints[(node, _port_for(model_size, replica))] = job_id
    return endpoints


def _flag_value(tokens: list[str], flag: str) -> str | None:
    """Return one shell-tokenized flag value, rejecting absence and duplicates."""

    positions = [index for index, token in enumerate(tokens) if token == flag]
    if len(positions) != 1:
        return None
    index = positions[0]
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _strip_exact_library_environment(tokens: list[str]) -> list[str] | None:
    """Validate and remove one immutable-prefix LD_LIBRARY_PATH assignment."""

    if len(tokens) < 2 or not tokens[0].startswith("LD_LIBRARY_PATH="):
        return None
    executable = Path(tokens[1])
    if not executable.is_absolute():
        return None
    expected = f"LD_LIBRARY_PATH={executable.parent.parent / 'lib'}"
    if tokens[0] != expected:
        return None
    return tokens[1:]


def _spooled_job_provenance(
    slurm_job_id: str,
    profile_name: str,
    *,
    run_root: str | None = None,
    expected_script: str | None = None,
    runner=None,
) -> SpooledServingProvenance | None:
    """Recover and validate complete launch provenance from Slurm's batch-script copy.

    A registry record is replaceable control state; the batch script spooled for an
    allocation is immutable.  Self-healing therefore trusts only facts recovered from
    that exact copy.  When ``run_root`` is supplied, job name, comment, canonical root,
    pool identity, replica id, and deterministic port must all agree.  Partial schema-5
    provenance is rejected.  A fully unpinned release/environment pair remains readable
    for legacy runs, but production workers will reject it through their frozen policy.
    """

    try:
        profile = get_serving_profile(profile_name)
        invoke = subprocess.run if runner is None else runner
        proc = invoke(
            ["scontrol", "write", "batch_script", str(slurm_job_id), "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (KeyError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    script = proc.stdout
    # Production reconciliation can reconstruct the complete launch script from the
    # frozen fleet, model, environment, and release contracts.  Exact-byte equality is
    # stronger than independently recognizing selected flags: it also binds both
    # interpreters, runtime-version probes, offline environment, resources, logging path,
    # and every vLLM option to the immutable release that owns this supervisor turn.
    if expected_script is not None and script != expected_script:
        return None

    # Collapse shell continuations so each rendered invocation becomes one line.  shlex
    # then gives exact argument boundaries without accepting substring lookalikes.
    normalized = re.sub(r"\\[ \t]*\r?\n", " ", script)
    serve_line = next(
        (
            line.strip()
            for line in normalized.splitlines()
            if " serve " in line and line.rstrip().endswith("&")
        ),
        None,
    )
    register_line = next(
        (
            line.strip()
            for line in normalized.splitlines()
            if " -m agents_scaling.serving.launch_server --register " in line
        ),
        None,
    )
    if serve_line is None or register_line is None:
        return None
    if not serve_line.endswith("&"):
        return None
    try:
        serve_tokens = shlex.split(serve_line[:-1].rstrip(), posix=True)
        register_tokens = shlex.split(register_line, posix=True)
    except ValueError:
        return None
    serve_tokens = _strip_exact_library_environment(serve_tokens)
    register_tokens = _strip_exact_library_environment(register_tokens)
    if serve_tokens is None or register_tokens is None:
        return None
    if (
        len(serve_tokens) < 3
        or Path(serve_tokens[0]).name != "vllm"
        or not Path(serve_tokens[0]).is_absolute()
        or serve_tokens[1:3] != ["serve", profile.hf_id]
    ):
        return None
    try:
        module_index = register_tokens.index("-m")
    except ValueError:
        return None
    if module_index < 1 or register_tokens[module_index - 1] != "-I":
        return None
    if register_tokens[module_index : module_index + 3] != [
        "-m",
        "agents_scaling.serving.launch_server",
        "--register",
    ]:
        return None
    if (
        not register_tokens
        or Path(register_tokens[0]).name != "python"
        or not Path(register_tokens[0]).is_absolute()
        or register_tokens.count("--standby") > 1
    ):
        return None
    standby = "--standby" in register_tokens

    serve_expected = {
        "--served-model-name": profile.served_model_name,
        "--tensor-parallel-size": str(profile.tp_size),
        "--max-model-len": str(profile.max_model_len),
        "--max-logprobs": "20",
        "--reasoning-parser": "qwen3",
        "--port": "$PORT",
    }
    register_expected = {
        "--model-size": profile.model_size,
        "--profile": profile.name,
        "--hf-id": profile.hf_id,
        "--served-model-name": profile.served_model_name,
        "--max-model-len": str(profile.max_model_len),
        "--tp-size": str(profile.tp_size),
        "--port": "$PORT",
    }
    if any(_flag_value(serve_tokens, flag) != value for flag, value in serve_expected.items()):
        return None
    if any(
        _flag_value(register_tokens, flag) != value
        for flag, value in register_expected.items()
    ):
        return None

    observed_root = _flag_value(register_tokens, "--run-root")
    observed_pool_id = _flag_value(register_tokens, "--server-pool-id")
    replica_id = _flag_value(register_tokens, "--replica-id")
    replica_index_text = _flag_value(register_tokens, "--replica-index")
    if (
        observed_root is None
        or observed_pool_id is None
        or replica_id is None
        or replica_index_text is None
    ):
        return None
    try:
        replica_index = int(replica_index_text)
    except ValueError:
        return None
    if replica_index < 0:
        return None

    canonical_observed_root = str(Path(observed_root).expanduser().resolve())
    computed_pool_id = registry.server_pool_id(canonical_observed_root)
    if observed_pool_id != computed_pool_id:
        return None
    if run_root is not None:
        canonical_expected_root = str(Path(run_root).expanduser().resolve())
        if canonical_observed_root != canonical_expected_root:
            return None
        if observed_pool_id != registry.server_pool_id(canonical_expected_root):
            return None
        expected_comment = (
            f"asys-schema5-pool:{observed_pool_id};profile={profile.name};"
            f"replica={replica_id}"
        )
        if not re.search(
            rf"(?m)^#SBATCH\s+--comment={re.escape(expected_comment)}\s*$",
            script,
        ):
            return None

    from agents_scaling.serving.launch_server import _port_for

    port_match = re.search(r"(?m)^PORT=(?P<port>\d+)\s*$", script)
    if (
        port_match is None
        or int(port_match.group("port")) != _port_for(profile.name, replica_index)
        or "#SBATCH --no-requeue" not in script
        or "#SBATCH --export=NONE" not in script
        or "importlib.metadata.version(\"vllm\") == \"0.21.0\"" not in script
        or "export PYTHONDONTWRITEBYTECODE=1" not in script
        or "export PYTHONPATH=" not in script
        or "unset BASH_ENV CDPATH ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD"
        not in script
        or "export PATH=/usr/bin:/bin" not in script
        or "readonly PATH" not in script
        or "export GIT_NO_REPLACE_OBJECTS=1" not in script
        or (
            "unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV "
            "LD_LIBRARY_PATH LD_PRELOAD"
        )
        not in script
        or "${LD_LIBRARY_PATH" in script
        or re.search(r"(?m)^\s*(?:source\s+|mamba\s+activate\s+)", script)
    ):
        return None

    model_revision = _flag_value(register_tokens, "--model-revision")
    tokenizer_id = _flag_value(register_tokens, "--tokenizer-id")
    tokenizer_revision = _flag_value(register_tokens, "--tokenizer-revision")
    model_contract_sha256 = _flag_value(
        register_tokens, "--model-contract-sha256"
    )
    fleet_contract_sha256 = _flag_value(
        register_tokens, "--fleet-contract-sha256"
    )
    release_fleet_contract_sha256 = _flag_value(
        register_tokens, "--release-fleet-contract-sha256"
    )
    capacity_generation_text = _flag_value(
        register_tokens, "--capacity-generation"
    )
    rollout_generation_text = _flag_value(
        register_tokens, "--rollout-generation"
    )
    fleet_contract_path = _flag_value(register_tokens, "--fleet-contract")
    release_worktree = _flag_value(register_tokens, "--release-worktree")
    model_contract_path = _flag_value(register_tokens, "--model-contract")
    release_id = _flag_value(register_tokens, "--release-id")
    environment_hash = _flag_value(register_tokens, "--environment-hash")
    identity_values = (
        model_revision,
        tokenizer_id,
        tokenizer_revision,
        model_contract_sha256,
    )
    if any(not value for value in identity_values):
        return None
    if (
        _flag_value(serve_tokens, "--revision") != model_revision
        or _flag_value(serve_tokens, "--tokenizer") != tokenizer_id
        or _flag_value(serve_tokens, "--tokenizer-revision") != tokenizer_revision
    ):
        return None
    resolved_model_contract: Path | None = None
    resolved_fleet_contract: Path | None = None
    if release_id:
        try:
            (
                resolved_release_worktree,
                resolved_model_contract,
                resolved_fleet_contract,
                _template_path,
            ) = _production_release_resources(
                release_worktree=release_worktree,
                model_contract_path=model_contract_path,
                fleet_contract_path=fleet_contract_path,
            )
        except ValueError:
            return None
        if (
            f'export ASYS_RELEASE_WORKTREE="{resolved_release_worktree}"' not in script
            or f'export ASYS_MODEL_CONTRACT="{resolved_model_contract}"' not in script
        ):
            return None
    elif model_contract_path:
        resolved_model_contract = Path(model_contract_path).expanduser().resolve()
    try:
        contracts = load_model_contracts(
            resolved_model_contract,
            expected_sha256=str(model_contract_sha256),
        )
        contracts.verify_identity(
            size=profile.model_size,
            hf_id=profile.hf_id,
            model_revision=str(model_revision),
            tokenizer_id=str(tokenizer_id),
            tokenizer_revision=str(tokenizer_revision),
        )
    except ModelContractError:
        return None

    if bool(release_id) != bool(environment_hash):
        return None
    if environment_hash and re.fullmatch(r"[0-9a-f]{64}", environment_hash) is None:
        return None
    if release_id and not all(
        marker in script
        for marker in (
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
        )
    ):
        return None
    if release_id:
        if (
            observed_pool_id != "schema5-v1"
            or not replica_id.startswith("schema5-v1--")
            or fleet_contract_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", fleet_contract_sha256) is None
            or release_fleet_contract_sha256 is None
            or re.fullmatch(
                r"[0-9a-f]{64}", release_fleet_contract_sha256
            )
            is None
            or not fleet_contract_path
        ):
            return None
        try:
            capacity_generation = int(str(capacity_generation_text))
            rollout_generation = int(str(rollout_generation_text))
        except ValueError:
            return None
        if (
            capacity_generation < 1
            or rollout_generation < 1
            or f'export ASYS_RELEASE_FLEET_CONTRACT_SHA256="'
            f'{release_fleet_contract_sha256}"' not in script
            or f'export ASYS_CAPACITY_GENERATION="{capacity_generation}"'
            not in script
            or f'export ASYS_ROLLOUT_GENERATION="{rollout_generation}"'
            not in script
        ):
            return None
        try:
            fleet = load_fleet_contract(
                resolved_fleet_contract,
                model_contracts=contracts,
                expected_sha256=fleet_contract_sha256,
                allow_capacity_layout=True,
            )
            fleet.verify_pool_root(canonical_observed_root)
            replica_contract = fleet.for_replica(profile.name, replica_index)
        except FleetContractError:
            return None
        expected_directives = {
            "job-name": replica_contract.scheduler_job_name,
            "partition": replica_contract.partition,
            "gres": f"gpu:{replica_contract.gpu_type}:{replica_contract.gpus_per_replica}",
            "cpus-per-task": str(replica_contract.cpus_per_task),
            "mem": replica_contract.memory,
            "time": replica_contract.time_limit,
        }
        if (
            replica_id != replica_contract.replica_id
            or observed_pool_id != replica_contract.pool_id
            or any(
                re.search(
                    rf"(?m)^#SBATCH\s+--{re.escape(flag)}={re.escape(value)}\s*$",
                    script,
                )
                is None
                for flag, value in expected_directives.items()
            )
        ):
            return None
    else:
        capacity_generation = None
        rollout_generation = None
        release_fleet_contract_sha256 = None
        try:
            legacy_replica_id: str | int = int(replica_id)
        except ValueError:
            legacy_replica_id = replica_id
        replica_id = legacy_replica_id

    return SpooledServingProvenance(
        run_root=canonical_observed_root,
        server_pool_id=observed_pool_id,
        replica_id=replica_id,
        replica_index=replica_index,
        release_id=release_id or None,
        environment_hash=environment_hash or None,
        model_revision=str(model_revision),
        tokenizer_id=str(tokenizer_id),
        tokenizer_revision=str(tokenizer_revision),
        model_contract_sha256=str(model_contract_sha256),
        fleet_contract_sha256=fleet_contract_sha256 or None,
        release_fleet_contract_sha256=release_fleet_contract_sha256,
        capacity_generation=capacity_generation,
        rollout_generation=rollout_generation,
        spooled_script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
    )


def _spooled_job_matches_profile(
    slurm_job_id: str, profile_name: str, *, run_root: str | None = None
) -> bool:
    """Compatibility predicate around complete spooled-provenance recovery."""

    return _spooled_job_provenance(
        slurm_job_id, profile_name, run_root=run_root
    ) is not None


def _replica_ids_in_flight(
    model_size: str, *, run_root: str | None = None
) -> set[int]:
    """Replica ids represented by any nonterminal serve job in ``squeue``.

    Registry entries exist only after vLLM is healthy.  Without also inspecting pending
    and transient requeue states, a second keepalive invocation can reuse replica 0 while
    the first replica-0 job is still active, producing a duplicate allocation or same-node
    port collision later. ``launch_server`` encodes the id in its stable sbatch filename,
    which is visible as ``squeue``'s command.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%t|%o"],
        capture_output=True,
        text=True,
    ).stdout
    job_name = _pool_job_name(run_root, model_size)
    filename = re.compile(
        rf"(?:^|/)serve_{re.escape(model_size)}(?:_r(?P<rid>\d+))?"
        rf"(?:\.g\d+\.[0-9a-f]+)?\.sbatch(?:\s|$)"
    )
    used: set[int] = set()
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3 or parts[0].strip() != job_name:
            continue
        match = filename.search(parts[2].strip())
        if match:
            used.add(int(match.group("rid") or 0))
    return used


def _reregister_running(
    run_root: str,
    model_size: str,
    *,
    running_override: dict[tuple[str, int], str] | None = None,
) -> int:
    """Self-heal: re-register live endpoints whose registry file was lost.

    A running serve job whose registry entry got pruned (e.g. a transient false-dead) would
    otherwise be stranded forever — job count >= target so no relaunch, yet not discoverable.
    For each running serve node, verify the immutable Slurm-spooled launch layout, probe
    the exact replica port, and atomically register a missing endpoint or upgrade a legacy
    or incomplete record at that address.
    """
    import dataclasses as _dc
    import json as _json

    from agents_scaling.serving.registry import ServerEntry, _size_dir

    profile = get_serving_profile(model_size)
    known = {
        (entry.host, entry.port): entry
        for entry in registry.list_servers(run_root, model_size)
    }
    restored = 0
    running = (
        _running_serve_endpoints(model_size, run_root=run_root)
        if running_override is None
        else running_override
    )
    for (node, port), slurm_job_id in running.items():
        existing = known.get((node, port))
        # Job name/path discovery establishes the live allocation and address; Slurm's
        # immutable spooled script establishes the exact model/runtime layout.  Require
        # both for every profile, including standards whose historical registry records
        # were previously accepted with inferred fields.
        provenance = _spooled_job_provenance(
            slurm_job_id, model_size, run_root=run_root
        )
        if provenance is None:
            print(
                f"[keepalive] refusing unverifiable serving profile "
                f"{model_size} @ {node}:{port}"
            )
            continue
        expected_fields = {
            "model_size": profile.model_size,
            "hf_id": profile.hf_id,
            "slurm_job_id": str(slurm_job_id),
            "serving_profile": profile.name,
            "served_model_name": profile.served_model_name,
            "max_model_len": profile.max_model_len,
            "tp_size": profile.tp_size,
            "release_id": provenance.release_id,
            "environment_hash": provenance.environment_hash,
            "model_revision": provenance.model_revision,
            "tokenizer_id": provenance.tokenizer_id,
            "tokenizer_revision": provenance.tokenizer_revision,
            "model_contract_sha256": provenance.model_contract_sha256,
            "fleet_contract_sha256": provenance.fleet_contract_sha256,
            "server_pool_id": provenance.server_pool_id,
            "replica_id": provenance.replica_id,
            "replica_index": provenance.replica_index,
            "release_fleet_contract_sha256": (
                provenance.release_fleet_contract_sha256
            ),
            "capacity_generation": provenance.capacity_generation,
            "rollout_generation": provenance.rollout_generation,
        }
        if existing is not None and all(
            getattr(existing, field) == value
            for field, value in expected_fields.items()
        ) and registry.entry_has_current_provenance(existing, profile.name):
            continue
        # FAST single probe (3s, no retries) so a tick can't stall on flaky/saturated
        # ports — a miss just retries next tick (eventually consistent). Registration
        # is additive and safe; we never remove here.
        if healthcheck.is_alive(node, port, timeout=3.0):
            # Preserve the original process registration instant when repairing fields
            # for the same allocation.  This keeps endpoint_generation stable across a
            # registry-file loss; a different allocation receives a fresh timestamp.
            try:
                existing_started_at = (
                    float(existing.started_at) if existing is not None else 0.0
                )
            except (TypeError, ValueError):
                existing_started_at = 0.0
            started_at = (
                existing_started_at
                if existing is not None
                and str(existing.slurm_job_id or "") == str(slurm_job_id)
                and existing_started_at > 0.0
                else time.time()
            )
            e = ServerEntry(
                model_size=profile.model_size,
                hf_id=profile.hf_id,
                host=node,
                port=port,
                slurm_job_id=slurm_job_id,
                started_at=started_at,
                serving_profile=profile.name,
                served_model_name=profile.served_model_name,
                max_model_len=profile.max_model_len,
                tp_size=profile.tp_size,
                release_id=provenance.release_id,
                environment_hash=provenance.environment_hash,
                model_revision=provenance.model_revision,
                tokenizer_id=provenance.tokenizer_id,
                tokenizer_revision=provenance.tokenizer_revision,
                model_contract_sha256=provenance.model_contract_sha256,
                fleet_contract_sha256=provenance.fleet_contract_sha256,
                server_pool_id=provenance.server_pool_id,
                replica_id=provenance.replica_id,
                replica_index=provenance.replica_index,
                release_fleet_contract_sha256=(
                    provenance.release_fleet_contract_sha256
                ),
                capacity_generation=provenance.capacity_generation,
                rollout_generation=provenance.rollout_generation,
            )
            if (
                provenance.server_pool_id == "schema5-v1"
                and isinstance(provenance.replica_id, str)
            ):
                if registry._production_lineage_complete(e):  # noqa: SLF001
                    try:
                        catalog = registry.collect_endpoint_history_catalog(run_root)
                    except ValueError as exc:
                        print(
                            f"[keepalive] refusing endpoint-history drift for "
                            f"{provenance.replica_id}: {exc}"
                        )
                        continue
                    history = [
                        item
                        for item in catalog.records
                        if item.server_entry.replica_id == provenance.replica_id
                        and str(item.server_entry.slurm_job_id) == str(slurm_job_id)
                        and item.server_entry.serving_profile == profile.name
                    ]
                    if len(history) != 1:
                        print(
                            f"[keepalive] refusing unsealed schema-5 endpoint "
                            f"{slurm_job_id} for {provenance.replica_id}"
                        )
                        continue
                    e = history[0].server_entry
                    if (
                        e.host != node
                        or e.port != port
                        or any(
                            getattr(e, field) != value
                            for field, value in expected_fields.items()
                        )
                    ):
                        print(
                            f"[keepalive] refusing archived endpoint identity drift "
                            f"for {provenance.replica_id} job {slurm_job_id}"
                        )
                        continue
                try:
                    promoted = registry.read_promoted_entry(
                        run_root,
                        profile.registry_key,
                        provenance.replica_id,
                    )
                except ValueError as exc:
                    print(
                        f"[keepalive] refusing unsafe promoted pointer for "
                        f"{provenance.replica_id}: {exc}"
                    )
                    continue
                if (
                    promoted is not None
                    and str(promoted.slurm_job_id or "") != str(slurm_job_id)
                ):
                    # A standby or retiring predecessor can remain RUNNING during a
                    # handoff.  It must never self-heal over the atomic promoted pointer.
                    print(
                        f"[keepalive] refusing stale allocation {slurm_job_id} "
                        f"over promoted job {promoted.slurm_job_id} for "
                        f"{provenance.replica_id}"
                    )
                    continue
                destination = registry.promoted_entry_path(
                    run_root,
                    profile.registry_key,
                    provenance.replica_id,
                )
            else:
                destination = (
                    _size_dir(run_root, profile.registry_key) / f"{node}_{port}.json"
                )
            io.atomic_write_text(
                destination,
                _json.dumps(_dc.asdict(e), indent=2, sort_keys=True) + "\n",
            )
            known[(node, port)] = e
            restored += 1
            action = "upgraded" if existing is not None else "re-registered"
            print(f"[keepalive] {action} live {model_size} @ {node}:{port}")
    return restored


def _robust_alive(host: str, port: int, attempts: int = 5, timeout: float = 10.0) -> bool:
    """True if /health responds 200 on ANY of several attempts.

    Two failure modes make a single probe unreliable: (1) a SATURATED but healthy vLLM is
    slow to answer /health; (2) the login node <-> server-node path is intermittently flaky
    (observed: 5 consecutive 15s timeouts, then 0.1s success). So we retry several times
    with backoff before declaring dead. Even so, a sustained blip can false-prune — the
    self-heal re-registration (_reregister_running) recovers a still-running server on the
    next tick, so a false prune is transient, not permanent.
    """
    for i in range(attempts):
        if healthcheck.is_alive(host, port, timeout=timeout):
            return True
        if i < attempts - 1:
            time.sleep(3.0)
    return False


def _live_dead(run_root: str, model_size: str) -> tuple[list, list]:
    live, dead = [], []
    for e in registry.list_servers(run_root, model_size):
        (live if _robust_alive(e.host, e.port) else dead).append(e)
    return live, dead


def _used_replica_ids(live: list) -> set[int]:
    """Infer which replica indices are live from their ports (port = base + replica)."""
    used = set()
    for e in live:
        from agents_scaling.serving.launch_server import _port_for

        base = _port_for(e.registry_key, 0)
        off = e.port - base
        if 0 <= off < 64:  # plausible replica offset
            used.add(off)
    return used


def _query_fleet_queue() -> tuple[FleetQueueRow, ...]:
    """Return joined live/accounting scheduler truth or fail closed.

    The compatibility name is retained for tests and forensic callers, but production
    admission is intentionally impossible when either squeue or sacct is unavailable.
    """

    try:
        return FleetQueueTruth(fleet_tx.query_scheduler())
    except fleet_tx.FleetTransactionError as exc:
        raise FleetContractError(str(exc)) from exc


def _rollout_generation(launch_options: dict[str, str | int | None]) -> int:
    raw = launch_options.get("rollout_generation")
    if raw is None:
        raw = os.environ.get("ASYS_ROLLOUT_GENERATION")
    try:
        generation = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FleetContractError(
            "canonical fleet admission requires ASYS_ROLLOUT_GENERATION"
        ) from exc
    if generation < 1:
        raise FleetContractError("canonical fleet rollout generation must be positive")
    return generation


def _capacity_generation(launch_options: Mapping[str, str | int | None]) -> int:
    raw = launch_options.get("capacity_generation")
    if raw is None:
        raw = os.environ.get("ASYS_CAPACITY_GENERATION")
    try:
        generation = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FleetContractError(
            "canonical fleet admission requires ASYS_CAPACITY_GENERATION"
        ) from exc
    if generation < 1:
        raise FleetContractError("canonical fleet capacity generation must be positive")
    return generation


def _load_current_fleet_authority(
    control_state_dir: str | Path,
    *,
    fleet: FrozenFleetContract,
    capacity_generation: int,
    rollout_generation: int,
    expected_protected_binding: Mapping[str, str] | None = None,
    expected_protected_capacity_contract: (
        protected_capacity.ProtectedCapacityContract | None
    ) = None,
) -> tuple[dict[str, Any], protected_capacity.ProtectedCapacityContract]:
    """Reload and exactly bind one production fleet turn to current control.

    A positive launch-script generation is not authority by itself.  Every turn and
    every external ``sbatch`` boundary must join it to the current control generations,
    effective fleet, and generation-scoped protected-capacity marker.
    """

    if (
        type(capacity_generation) is not int
        or capacity_generation < 1
        or type(rollout_generation) is not int
        or rollout_generation < 1
    ):
        raise FleetContractError(
            "production fleet generations must be positive integers"
        )
    try:
        from slurm import schema5_control as control_plane
    except (ImportError, OSError) as exc:
        raise FleetContractError(
            f"cannot import current production fleet authority: {exc}"
        ) from exc

    try:
        state_dir = Path(control_state_dir).expanduser().resolve()
        control_state = control_plane.load_control(
            state_dir,
            verify_files=True,
        )
        control_capacity = control_state.get("capacity", {}).get(
            "current_generation"
        )
        control_rollout = control_state.get("rollout_generation")
        fleet_binding = control_plane.effective_fleet_contract_binding(
            control_state,
            verify_files=True,
        )
        protected_binding = (
            control_plane.effective_protected_capacity_binding(
                control_state,
                verify_files=True,
            )
        )
        current_fleet = control_plane.load_effective_fleet_contract(
            control_state,
            verify_files=True,
        )
        current_protected = (
            control_plane.load_effective_protected_capacity_contract(
                control_state,
                verify_files=True,
            )
        )

        expected_fleet_path = fleet.path.expanduser().resolve()
        observed_fleet_path = Path(str(fleet_binding["path"])).resolve()
        observed_contract_path = current_fleet.path.expanduser().resolve()
        if (
            control_capacity != capacity_generation
            or control_rollout != rollout_generation
            or fleet_binding.get("capacity_generation")
            != capacity_generation
            or protected_binding.get("capacity_generation")
            != capacity_generation
            or current_protected.capacity_generation
            != capacity_generation
            or observed_fleet_path != expected_fleet_path
            or observed_contract_path != expected_fleet_path
            or fleet_binding.get("sha256") != fleet.sha256
            or current_fleet.sha256 != fleet.sha256
            or Path(
                str(protected_binding["effective_fleet_contract_path"])
            ).resolve()
            != expected_fleet_path
            or protected_binding.get("effective_fleet_contract_sha256")
            != fleet.sha256
            or current_protected.effective_fleet_contract_path.resolve()
            != expected_fleet_path
            or current_protected.effective_fleet_contract_sha256
            != fleet.sha256
        ):
            raise FleetContractError(
                "production fleet launch authority differs from the current "
                "capacity/rollout generation or effective fleet"
            )

        observed_protected = {
            "path": str(Path(str(protected_binding["path"])).resolve()),
            "sha256": str(protected_binding["sha256"]),
            "marker_id": str(protected_binding["marker_id"]),
        }
        if (
            expected_protected_binding is not None
            and observed_protected
            != {
                "path": str(
                    Path(str(expected_protected_binding["path"]))
                    .expanduser()
                    .resolve()
                ),
                "sha256": str(expected_protected_binding["sha256"]),
                "marker_id": str(expected_protected_binding["marker_id"]),
            }
        ):
            raise FleetContractError(
                "production fleet protected-capacity binding differs from "
                "current control"
            )
        if expected_protected_capacity_contract is not None and (
            current_protected.path.resolve()
            != expected_protected_capacity_contract.path.resolve()
            or current_protected.sha256
            != expected_protected_capacity_contract.sha256
            or current_protected.marker_id
            != expected_protected_capacity_contract.marker_id
            or current_protected.capacity_generation
            != expected_protected_capacity_contract.capacity_generation
            or current_protected.effective_fleet_contract_path.resolve()
            != (
                expected_protected_capacity_contract
                .effective_fleet_contract_path.resolve()
            )
            or current_protected.effective_fleet_contract_sha256
            != (
                expected_protected_capacity_contract
                .effective_fleet_contract_sha256
            )
        ):
            raise FleetContractError(
                "production fleet cached protected-capacity marker is no "
                "longer current"
            )
        protected_capacity.authorize_fleet(fleet, current_protected)
        return control_state, current_protected
    except FleetContractError:
        raise
    except (
        control_plane.ControlError,
        protected_capacity.ProtectedCapacityError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise FleetContractError(
            f"cannot establish current production fleet authority: {exc}"
        ) from exc


def _expected_fleet_script(
    replica,
    run_root: str,
    launch_options: dict[str, str | int | None],
    *,
    standby: bool = False,
) -> str:
    script = render_sbatch(
        replica.model_size,
        run_root,
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(Path(run_root) / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        qos=replica.qos,
        standby=standby,
        **launch_options,
    )
    _validate_fleet_script_contract(
        script,
        replica=replica,
        rollout_generation=_rollout_generation(launch_options),
        fleet_sha256=str(launch_options.get("fleet_contract_sha256") or ""),
        standby=standby,
    )
    return script


def _validate_fleet_script_contract(
    script: str,
    *,
    replica,
    rollout_generation: int,
    fleet_sha256: str,
    standby: bool = False,
) -> None:
    exact_directives = (
        f"#SBATCH --job-name={replica.scheduler_job_name}",
        (
            f"#SBATCH --comment=asys-schema5-pool:{replica.pool_id};"
            f"profile={replica.serving_profile};replica={replica.replica_id}"
        ),
        f"#SBATCH --partition={replica.partition}",
        f"#SBATCH --qos={replica.qos}",
        f"#SBATCH --gres=gpu:{replica.gpu_type}:{replica.gpus_per_replica}",
        f"#SBATCH --cpus-per-task={replica.cpus_per_task}",
        f"#SBATCH --mem={replica.memory}",
        f"#SBATCH --time={replica.time_limit}",
        "#SBATCH --mail-user=mabdel03@mit.edu",
        "#SBATCH --mail-type=FAIL",
        "#SBATCH --no-requeue",
        "#SBATCH --export=NONE",
        f'export ASYS_ROLLOUT_GENERATION="{rollout_generation}"',
    )
    if any(
        re.search(rf"(?m)^{re.escape(directive)}\s*$", script) is None
        for directive in exact_directives
    ):
        raise FleetContractError(
            f"rendered script is not the frozen production contract for "
            f"{replica.replica_id}"
        )
    if (
        not fleet_sha256
        or f'--fleet-contract-sha256 "{fleet_sha256}"' not in script
        or f'--replica-id "{replica.replica_id}"' not in script
        or f'--replica-index {replica.replica_index}' not in script
        or "unset BASH_ENV CDPATH ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD"
        not in script
        or "export PATH=/usr/bin:/bin" not in script
        or "readonly PATH" not in script
        or "export GIT_NO_REPLACE_OBJECTS=1" not in script
    ):
        raise FleetContractError(
            f"rendered script omits frozen identity for {replica.replica_id}"
        )
    release_fleet_match = re.search(
        r'(?m)^export ASYS_RELEASE_FLEET_CONTRACT_SHA256="([0-9a-f]{64})"\s*$',
        script,
    )
    capacity_match = re.search(
        r'(?m)^export ASYS_CAPACITY_GENERATION="([1-9][0-9]*)"\s*$',
        script,
    )
    if release_fleet_match is None or capacity_match is None:
        raise FleetContractError(
            f"rendered script omits endpoint lineage for {replica.replica_id}"
        )
    normalized = re.sub(r"\\[ \t]*\r?\n", " ", script)
    register_line = next(
        (
            line
            for line in normalized.splitlines()
            if " -m agents_scaling.serving.launch_server --register " in line
        ),
        None,
    )
    if register_line is None:
        raise FleetContractError(
            f"rendered script omits registration for {replica.replica_id}"
        )
    try:
        register_tokens = shlex.split(register_line)
    except ValueError as exc:
        raise FleetContractError(
            f"rendered registration cannot be parsed for {replica.replica_id}"
        ) from exc
    if register_tokens.count("--standby") != int(standby):
        raise FleetContractError(
            f"rendered script has wrong standby role for {replica.replica_id}"
        )
    if (
        _flag_value(register_tokens, "--release-fleet-contract-sha256")
        != release_fleet_match.group(1)
        or _flag_value(register_tokens, "--capacity-generation")
        != capacity_match.group(1)
        or _flag_value(register_tokens, "--rollout-generation")
        != str(rollout_generation)
    ):
        raise FleetContractError(
            f"rendered registration omits exact endpoint lineage for "
            f"{replica.replica_id}"
        )


def _validate_active_scheduler_timing(row: FleetQueueRow, *, replica) -> None:
    """Require exact, dependency-free Slurm timing for every active allocation."""

    if fleet_tx.terminal_state(row.state):
        return
    if row.source != "squeue" or row.dependency != "":
        raise FleetContractError(
            f"fleet job {row.job_id} lacks dependency-free squeue timing truth"
        )
    try:
        expected_seconds = fleet_tx._parse_slurm_duration(  # noqa: SLF001
            replica.time_limit,
            source="squeue",
        )
    except fleet_tx.FleetTransactionError as exc:
        raise FleetContractError(str(exc)) from exc
    if (
        expected_seconds is None
        or row.time_limit_seconds != expected_seconds
    ):
        raise FleetContractError(
            f"fleet job {row.job_id} time limit drifted: "
            f"observed={row.time_limit_seconds}, expected={expected_seconds}"
        )
    if row.state.upper() == "RUNNING":
        if row.start_timestamp is None or row.end_timestamp is None:
            raise FleetContractError(
                f"running fleet job {row.job_id} lacks exact start/end timing"
            )
        observed_span = float(row.end_timestamp) - float(row.start_timestamp)
        if abs(observed_span - expected_seconds) > 1.0:
            raise FleetContractError(
                f"running fleet job {row.job_id} start/end span drifted: "
                f"observed={observed_span}, expected={expected_seconds}"
            )


def _attempt_for_token(
    ledgers: list[dict],
    token: str,
) -> tuple[str, dict, dict] | None:
    matches: list[tuple[str, dict, dict]] = []
    for ledger in ledgers:
        for replica_id, record in ledger["replicas"].items():
            for attempt in record["attempts"]:
                if attempt["intent_token"] == token:
                    matches.append((replica_id, attempt, ledger))
    if len(matches) > 1:
        raise FleetContractError(f"fleet intent token {token} is not globally unique")
    return matches[0] if matches else None


def _validate_scheduler_row(
    row: FleetQueueRow,
    *,
    replica,
    attempt: dict,
    fleet: FrozenFleetContract,
    run_root: str,
    expected_script: str,
    scheduler_runner=None,
) -> SpooledServingProvenance | None:
    _validate_active_scheduler_timing(row, replica=replica)
    parsed = fleet_tx.parse_intent_comment(row.comment)
    expected_comment = fleet_tx.intent_comment(
        pool_id=replica.pool_id,
        profile=replica.serving_profile,
        replica_id=replica.replica_id,
        rollout_generation=attempt["rollout_generation"],
        intent_token=attempt["intent_token"],
        fleet_sha256=fleet.sha256,
    )
    if parsed is None or row.comment != expected_comment:
        raise FleetContractError(
            f"scheduler intent comment drift for job {row.job_id}"
        )
    if (
        row.job_name != replica.scheduler_job_name
        or row.partition != replica.partition
        or row.qos != replica.qos
        or parsed["pool"] != replica.pool_id
        or parsed["profile"] != replica.serving_profile
        or parsed["replica"] != replica.replica_id
        or parsed["fleet"] != fleet.sha256
        or attempt.get("submission_transport")
        != fleet_tx.STDIN_EXACT_SUBMISSION_TRANSPORT
        or attempt.get("submission_argv_sha256")
        != fleet_tx.submission_argv_sha256(expected_comment)
        or not fleet_tx.command_binds_stdin_submission(
            row.command, expected_comment
        )
    ):
        raise FleetContractError(
            f"scheduler provenance drift for fleet job {row.job_id}"
        )
    sbatch_path = Path(attempt["sbatch_path"])
    expected_hash = hashlib.sha256(expected_script.encode("utf-8")).hexdigest()
    if (
        attempt["sbatch_sha256"] != expected_hash
        or sbatch_path.is_symlink()
        or not sbatch_path.is_file()
        or hashlib.sha256(sbatch_path.read_bytes()).hexdigest() != expected_hash
        or sbatch_path.stat().st_mode & 0o222
    ):
        raise FleetContractError(
            f"immutable sbatch provenance drift for {replica.replica_id}"
        )
    # Active allocations, including pending ones, retain Slurm's immutable spooled
    # script.  Terminal accounting history can no longer be expected to retain it.
    if not fleet_tx.terminal_state(row.state):
        provenance = _spooled_job_provenance(
            row.job_id,
            replica.serving_profile,
            run_root=run_root,
            expected_script=expected_script,
            runner=scheduler_runner,
        )
        if (
            provenance is None
            or provenance.replica_id != replica.replica_id
            or provenance.replica_index != replica.replica_index
            or provenance.server_pool_id != replica.pool_id
            or provenance.fleet_contract_sha256 != fleet.sha256
            or (
                provenance.release_fleet_contract_sha256 is not None
                and (
                    provenance.rollout_generation
                    != attempt["rollout_generation"]
                    or provenance.spooled_script_sha256 != expected_hash
                )
            )
        ):
            raise FleetContractError(
                f"job {row.job_id} failed immutable fleet provenance validation"
            )
        return provenance
    return None


def _scope_readiness_scheduler_rows(
    rows: Sequence[FleetQueueRow],
    *,
    fleet: FrozenFleetContract,
) -> tuple[tuple[FleetQueueRow, ...], tuple[str, ...]]:
    """Isolate wholly foreign canaries and reject partial production collisions."""

    replicas = {replica.replica_id: replica for replica in fleet.replicas}
    production_names = {
        replica.scheduler_job_name for replica in fleet.replicas
    }
    production: list[FleetQueueRow] = []
    isolated: list[str] = []
    for row in rows:
        parsed = fleet_tx.parse_intent_comment(row.comment)
        if parsed is None:
            claims_production = (
                row.job_name in production_names
                or f"pool={fleet.fleet_id}" in row.comment
                or f"fleet={fleet.sha256}" in row.comment
                or any(
                    f"replica={replica_id}" in row.comment
                    for replica_id in replicas
                )
            )
            if claims_production:
                raise FleetContractError(
                    "malformed scheduler row collides with the production fleet "
                    f"namespace: job {row.job_id}"
                )
            isolated.append(row.job_id)
            continue
        replica = replicas.get(parsed["replica"])
        exact = (
            parsed["pool"] == fleet.fleet_id
            and parsed["fleet"] == fleet.sha256
            and replica is not None
            and parsed["profile"] == replica.serving_profile
            and row.job_name == replica.scheduler_job_name
        )
        wholly_foreign = (
            parsed["pool"] != fleet.fleet_id
            and parsed["fleet"] != fleet.sha256
            and replica is None
            and row.job_name not in production_names
        )
        if exact:
            production.append(row)
        elif wholly_foreign:
            isolated.append(row.job_id)
        else:
            raise FleetContractError(
                "scheduler row partially collides with production fleet identity: "
                f"job {row.job_id}"
            )
    return (
        tuple(sorted(production, key=lambda row: int(row.job_id))),
        tuple(sorted(isolated, key=int)),
    )


def _sealed_handoff_history(
    *,
    current_ledger: Mapping[str, Any],
    logical_allocations: Sequence[fleet_tx.ReconciledFleetAllocation],
) -> set[str]:
    """Validate that every non-live current-generation attempt is sealed history."""

    active_by_replica = {
        allocation.replica_id: allocation for allocation in logical_allocations
    }
    replicas = current_ledger.get("replicas")
    if not isinstance(replicas, Mapping) or set(replicas) != set(active_by_replica):
        raise FleetContractError("current fleet ledger replica identity drifted")
    allowed: set[str] = set()
    for replica_id, record in replicas.items():
        attempts = record.get("attempts") if isinstance(record, Mapping) else None
        active = active_by_replica[replica_id]
        if not isinstance(attempts, list) or not attempts:
            raise FleetContractError(
                f"current fleet generation lacks attempts for {replica_id}"
            )
        active_matches = [
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and attempt.get("intent_token")
            == active.attempt.get("intent_token")
        ]
        if len(active_matches) != 1:
            raise FleetContractError(
                f"current logical allocation is not uniquely ledger-bound: {replica_id}"
            )
        active_attempt = active_matches[0]
        if (
            active_attempt.get("state") != "committed"
            or active_attempt.get("job_id") != str(active.row.job_id)
            or active_attempt.get("committed_at") is None
            or active_attempt.get("last_error") is not None
            or active_attempt.get("retire_error") is not None
            or active_attempt.get("lifecycle") not in {"primary", "promoted"}
        ):
            raise FleetContractError(
                f"current logical allocation is not a clean commit: {replica_id}"
            )
        by_job: dict[str, Mapping[str, Any]] = {}
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise FleetContractError(
                    f"fleet attempt is malformed for {replica_id}"
                )
            job_id = str(attempt.get("job_id") or "")
            if not job_id.isdigit() or job_id in by_job:
                raise FleetContractError(
                    f"fleet attempt job identity is ambiguous for {replica_id}"
                )
            by_job[job_id] = attempt
        terminal = [
            attempt for attempt in attempts if attempt is not active_attempt
        ]
        terminal_ids = {str(attempt["job_id"]) for attempt in terminal}
        for predecessor in terminal:
            predecessor_id = str(predecessor["job_id"])
            successors = [
                candidate
                for candidate in attempts
                if candidate.get("launch_kind") == "handoff"
                and str(candidate.get("predecessor_job_id") or "")
                == predecessor_id
            ]
            if (
                predecessor.get("state") != "terminal"
                or predecessor.get("lifecycle") != "retiring"
                or predecessor.get("terminal_at") is None
                or predecessor.get("last_error") is not None
                or predecessor.get("retire_requested_at") is None
                or not isinstance(predecessor.get("retire_attempts"), int)
                or isinstance(predecessor.get("retire_attempts"), bool)
                or predecessor["retire_attempts"] < 1
                or predecessor.get("last_retire_attempt_at") is None
                or predecessor.get("retire_error") is not None
                or len(successors) != 1
            ):
                raise FleetContractError(
                    "current generation contains failed/orphaned fleet history for "
                    f"{replica_id} job {predecessor_id}"
                )
            successor = successors[0]
            if (
                successor.get("state") not in {"committed", "terminal"}
                or successor.get("lifecycle") not in {"promoted", "retiring"}
                or successor.get("promoted_at") is None
                or not isinstance(successor.get("ready_probe_count"), int)
                or isinstance(successor.get("ready_probe_count"), bool)
                or successor["ready_probe_count"] < 2
                or successor.get("last_error") is not None
                or successor.get("retire_error") is not None
            ):
                raise FleetContractError(
                    "fleet handoff successor is failed/orphaned or not "
                    f"sealed/healthy for {replica_id}"
                )
            allowed.add(predecessor_id)
        active_job_id = str(active_attempt["job_id"])
        for predecessor_id in terminal_ids:
            visited: set[str] = set()
            cursor = predecessor_id
            while cursor != active_job_id:
                if cursor in visited:
                    raise FleetContractError(
                        f"fleet handoff history cycles for {replica_id}"
                    )
                visited.add(cursor)
                successors = [
                    str(candidate["job_id"])
                    for candidate in attempts
                    if candidate.get("launch_kind") == "handoff"
                    and str(candidate.get("predecessor_job_id") or "") == cursor
                ]
                if len(successors) != 1:
                    raise FleetContractError(
                        f"fleet handoff history is disconnected for {replica_id}"
                    )
                cursor = successors[0]
                if cursor not in by_job:
                    raise FleetContractError(
                        f"fleet handoff successor is absent for {replica_id}"
                    )
    return allowed


def reconcile_fleet_read_only(
    run_root: str,
    fleet: FrozenFleetContract,
    *,
    current_generation: int,
    scheduler_runner=None,
    scheduler_now: float | None = None,
) -> ReadOnlyFleetSnapshot:
    """Prove the live fleet from joined Slurm truth without changing control state.

    This is the readiness counterpart to :func:`tick_fleet`.  Both consume the same
    durable intent ledger and :func:`fleet_tx.reconcile_scheduler_rows` join.  The
    readiness view additionally requires every logical replica to be committed,
    RUNNING, non-hung, and byte-bound to Slurm's immutable spooled script.
    """

    canonical_root = fleet.verify_pool_root(run_root)
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 1
    ):
        raise FleetContractError(
            "read-only fleet reconciliation requires a positive generation"
        )
    replica_ids = [replica.replica_id for replica in fleet.replicas]
    replica_by_id = {replica.replica_id: replica for replica in fleet.replicas}
    try:
        with fleet_tx.read_transaction_lock(canonical_root) as directory:
            ledgers = fleet_tx.read_generation_ledgers(
                directory,
                pool_root=canonical_root,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                current_generation=current_generation,
                replica_ids=replica_ids,
            )
            scheduler = fleet_tx.query_scheduler(
                runner=scheduler_runner,
                now=scheduler_now,
            )
            production_rows, isolated_foreign_job_ids = (
                _scope_readiness_scheduler_rows(
                    scheduler.rows,
                    fleet=fleet,
                )
            )
            reconciled = fleet_tx.reconcile_scheduler_rows(
                production_rows,
                ledgers,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                replica_profiles={
                    replica.replica_id: replica.serving_profile
                    for replica in fleet.replicas
                },
                replica_job_names={
                    replica.replica_id: replica.scheduler_job_name
                    for replica in fleet.replicas
                },
                replica_qos={
                    replica.replica_id: replica.qos
                    for replica in fleet.replicas
                },
            )
            physical = reconciled.active_allocations
            active = reconciled.logical_allocations
            if len(physical) != len(active):
                overlaps = sorted(
                    (
                        allocation.replica_id,
                        allocation.row.job_id,
                        str(allocation.attempt.get("lifecycle")),
                    )
                    for allocation in physical
                )
                raise FleetContractError(
                    "preproduction/transition fleet readiness forbids an "
                    f"in-progress handoff overlap: {overlaps}"
                )
            active_ids = [allocation.replica_id for allocation in active]
            missing = sorted(set(replica_ids) - set(active_ids))
            unexpected = sorted(set(active_ids) - set(replica_ids))
            if (
                missing
                or unexpected
                or len(active_ids) != len(set(active_ids))
                or len(active_ids) != len(replica_ids)
            ):
                raise FleetContractError(
                    "canonical fleet is incomplete or duplicated: "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"active={len(active_ids)}, expected={len(replica_ids)}"
                )
            non_running = {
                allocation.replica_id: allocation.row.state
                for allocation in active
                if allocation.row.state.upper() != "RUNNING"
            }
            if non_running:
                raise FleetContractError(
                    f"canonical fleet still has pending/non-running replicas: {non_running}"
                )
            current_ledger = next(
                (
                    ledger
                    for ledger in ledgers
                    if ledger["rollout_generation"] == current_generation
                ),
                None,
            )
            if current_ledger is None:
                raise FleetContractError(
                    "current exact fleet generation ledger is absent"
                )
            sealed_terminal_ids = _sealed_handoff_history(
                current_ledger=current_ledger,
                logical_allocations=active,
            )
            if reconciled.ignored_terminal_job_ids:
                raise FleetContractError(
                    "production scheduler truth contains unmapped terminal rows: "
                    f"{list(reconciled.ignored_terminal_job_ids)}"
                )
            invalid_terminal = []
            for row in production_rows:
                parsed = fleet_tx.parse_intent_comment(row.comment)
                state = row.state.upper().split("+", 1)[0].split(" ", 1)[0]
                if (
                    parsed is not None
                    and parsed["generation"] == str(current_generation)
                    and fleet_tx.terminal_state(row.state)
                    and (
                        row.job_id not in sealed_terminal_ids
                        or state not in {"CANCELLED", "COMPLETED"}
                    )
                ):
                    invalid_terminal.append(row.job_id)
            if invalid_terminal:
                raise FleetContractError(
                    "current fleet generation contains unsealed/failed terminal "
                    f"scheduler rows: {sorted(invalid_terminal, key=int)}"
                )

            verified: list[ReadOnlyFleetAllocation] = []
            for allocation in active:
                replica = replica_by_id[allocation.replica_id]
                attempt = allocation.attempt
                row = allocation.row
                health = allocation.health
                if (
                    attempt.get("state") != "committed"
                    or attempt.get("job_id") != str(row.job_id)
                    or attempt.get("committed_at") is None
                ):
                    raise FleetContractError(
                        f"fleet intent for {replica.replica_id} is not durably committed"
                    )
                if not row.node or row.node in {"(null)", "N/A", "None", "None assigned"}:
                    raise FleetContractError(
                        f"running fleet job {row.job_id} has no scheduler node"
                    )
                if (
                    not isinstance(health, Mapping)
                    or health.get("job_id") != str(row.job_id)
                    or health.get("observer_generation") != current_generation
                ):
                    raise FleetContractError(
                        f"fleet supervisor health state is stale for {replica.replica_id}"
                    )
                if health.get("cancel_state") is not None:
                    raise FleetContractError(
                        f"fleet allocation is hung/fenced for {replica.replica_id}: "
                        f"{health.get('cancel_state')}"
                    )
                script_path = Path(str(attempt["sbatch_path"]))
                try:
                    script = script_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise FleetContractError(
                        f"cannot read immutable fleet script for {replica.replica_id}: "
                        f"{exc}"
                    ) from exc
                _validate_fleet_script_contract(
                    script,
                    replica=replica,
                    rollout_generation=allocation.ledger_generation,
                    fleet_sha256=fleet.sha256,
                    standby=attempt.get("launch_kind") == "handoff",
                )
                provenance = _validate_scheduler_row(
                    row,
                    replica=replica,
                    attempt=dict(attempt),
                    fleet=fleet,
                    run_root=str(canonical_root),
                    expected_script=script,
                    scheduler_runner=scheduler_runner,
                )
                if provenance is None:
                    raise FleetContractError(
                        f"active fleet job {row.job_id} lacks spooled provenance"
                    )
                verified.append(
                    ReadOnlyFleetAllocation(
                        replica_id=replica.replica_id,
                        ledger_generation=allocation.ledger_generation,
                        intent_token=str(attempt["intent_token"]),
                        attempt_state=str(attempt["state"]),
                        row=row,
                        spooled_provenance=provenance,
                        sbatch_path=str(script_path),
                        sbatch_sha256=str(attempt["sbatch_sha256"]),
                    )
                )
    except fleet_tx.FleetTransactionError as exc:
        raise FleetContractError(str(exc)) from exc
    return ReadOnlyFleetSnapshot(
        current_generation=current_generation,
        captured_at=scheduler.captured_at,
        allocations=tuple(sorted(verified, key=lambda item: item.replica_id)),
        ignored_terminal_job_ids=reconciled.ignored_terminal_job_ids,
        isolated_foreign_job_ids=isolated_foreign_job_ids,
        sealed_successful_handoff_terminal_job_ids=tuple(
            sorted(sealed_terminal_ids, key=int)
        ),
    )


def trusted_scientific_fleet_bindings(
    snapshot: ReadOnlyFleetSnapshot,
) -> dict[str, dict[str, Any]]:
    """Export exact fleet IDs only from a fully validated read-only transaction join."""

    bindings: dict[str, dict[str, Any]] = {}
    for allocation in snapshot.allocations:
        job_id = str(allocation.row.job_id)
        parsed = fleet_tx.parse_intent_comment(allocation.row.comment)
        if (
            not job_id.isdigit()
            or allocation.attempt_state != "committed"
            or parsed is None
            or parsed.get("intent") != allocation.intent_token
            or parsed.get("replica") != allocation.replica_id
            or parsed.get("generation")
            != str(allocation.ledger_generation)
            or job_id in bindings
        ):
            raise FleetContractError(
                "trusted fleet export encountered an ambiguous allocation"
            )
        transaction_directory = Path(allocation.sbatch_path).parents[2]
        generation_ledger_path = fleet_tx.ledger_path(
            transaction_directory,
            allocation.ledger_generation,
        )
        try:
            if (
                generation_ledger_path.is_symlink()
                or not generation_ledger_path.is_file()
                or generation_ledger_path.stat().st_nlink != 1
            ):
                raise OSError("fleet generation ledger is unsafe")
            generation_ledger_raw = generation_ledger_path.read_bytes()
        except OSError as exc:
            raise FleetContractError(
                f"cannot bind exact fleet generation ledger: {exc}"
            ) from exc
        bindings[job_id] = {
            "job_name": allocation.row.job_name,
            "comment": allocation.row.comment,
            "sbatch_path": allocation.sbatch_path,
            "sbatch_sha256": allocation.sbatch_sha256,
            "intent_token": allocation.intent_token,
            "replica_id": allocation.replica_id,
            "ledger_generation": allocation.ledger_generation,
            "ledger_path": str(generation_ledger_path.resolve()),
            "ledger_sha256": hashlib.sha256(
                generation_ledger_raw
            ).hexdigest(),
        }
    return dict(sorted(bindings.items(), key=lambda item: int(item[0])))


HUNG_MIN_FAILURES = 3
HUNG_MIN_SPAN_SECONDS = 600.0
HUNG_CANCEL_MAX_ATTEMPTS = 5
HUNG_CANCEL_BACKOFF_SECONDS = 300.0
HANDOFF_LEAD_SECONDS = 12 * 60 * 60
HANDOFF_READY_PROBES = 2
HANDOFF_MAX_OVERLAP_GPUS = 4
HANDOFF_DRAIN_SECONDS = 660.0
HANDOFF_RETIRE_BACKOFF_SECONDS = 300.0
HANDOFF_RETIRE_MAX_ATTEMPTS = 5
HANDOFF_CRITICAL_SECONDS = 60 * 60


def _new_health_record(
    job_id: str, endpoint: str, *, observer_generation: int
) -> dict:
    return {
        "job_id": str(job_id),
        "endpoint": endpoint,
        "observer_generation": observer_generation,
        "first_failure_at": None,
        "last_failure_at": None,
        "last_probe_at": None,
        "consecutive_failures": 0,
        "health_failures": 0,
        "models_failures": 0,
        "cancel_state": None,
        "cancel_requested_at": None,
        "cancel_completed_at": None,
        "cancel_error": None,
        "cancel_attempts": 0,
        "last_cancel_attempt_at": None,
        "next_cancel_eligible_at": None,
        "alert_id": None,
    }


def _probe_health_and_models(
    host: str, port: int, *, timeout: float = 10.0
) -> tuple[bool, bool]:
    """Probe two independent lightweight vLLM paths once per supervisor poll."""

    base = f"http://{host}:{port}"
    health_status, _ = healthcheck._get(f"{base}/health", timeout=timeout)
    models_status, models_body = healthcheck._get(
        f"{base}/v1/models", timeout=timeout
    )
    return health_status == 200, models_status == 200 and b'"id"' in models_body


def _monitor_running_replica(
    *,
    directory: Path,
    ledger: dict,
    replica,
    row: FleetQueueRow,
    observer_generation: int,
    now: float,
    probe=None,
    cancel_runner=None,
) -> None:
    """Fence one persistently hung allocation without reacting to transient load.

    One probe pair is recorded per supervisor poll.  Cancellation is eligible only
    after at least three consecutive *dual* failures spanning ten minutes.  The exact
    job id and endpoint are part of the durable state, so a replacement starts with a
    clean health history and an old alert can never cancel it.
    """

    if row.state.upper() != "RUNNING":
        return
    if not row.node or row.node in {"(null)", "N/A", "None"}:
        raise FleetContractError(f"running fleet job {row.job_id} has no scheduler node")
    from agents_scaling.serving.launch_server import _port_for

    port = _port_for(replica.serving_profile, replica.replica_index)
    endpoint = f"{row.node}:{port}"
    record = ledger["replicas"][replica.replica_id]
    health = record["health"]
    if (
        not isinstance(health, dict)
        or health.get("job_id") != str(row.job_id)
        or health.get("endpoint") != endpoint
        or health.get("observer_generation") != observer_generation
    ):
        health = _new_health_record(
            row.job_id,
            endpoint,
            observer_generation=observer_generation,
        )
        record["health"] = health
        fleet_tx.save_ledger(directory, ledger, now=now)

    # Once Slurm accepted the exact-ID cancellation, only joined scheduler truth may
    # advance this allocation to terminal/replacement.  Exhaustion is likewise an
    # explicit operator-visible state rather than an unbounded mutation loop.
    if health["cancel_state"] in {"accepted", "exhausted"}:
        return
    invoke_probe = probe or _probe_health_and_models
    invoke_cancel = cancel_runner or subprocess.run
    try:
        health_ok, models_ok = invoke_probe(row.node, port)
    except Exception as exc:  # noqa: BLE001 - a probe transport failure is a failed probe
        print(f"[keepalive] probe error for {replica.replica_id}: {exc!r}")
        health_ok, models_ok = False, False
    health["last_probe_at"] = float(now)
    if health_ok or models_ok:
        # Either independent lightweight endpoint proves that the process responds; a
        # saturated engine must recover only once to erase the consecutive-failure run.
        record["health"] = _new_health_record(
            row.job_id,
            endpoint,
            observer_generation=observer_generation,
        )
        fleet_tx.save_ledger(directory, ledger, now=now)
        return
    if health["first_failure_at"] is None:
        health["first_failure_at"] = float(now)
    health["last_failure_at"] = float(now)
    health["consecutive_failures"] += 1
    health["health_failures"] += 1
    health["models_failures"] += 1
    fleet_tx.save_ledger(directory, ledger, now=now)
    failure_span = float(now) - float(health["first_failure_at"])
    if (
        health["consecutive_failures"] < HUNG_MIN_FAILURES
        or failure_span < HUNG_MIN_SPAN_SECONDS
    ):
        return

    if cancel_runner is None and os.environ.get("PYTEST_CURRENT_TEST"):
        raise FleetContractError(
            "pytest fleet cancellation must provide an injected non-Slurm runner"
        )

    if health["cancel_state"] is None:
        fleet_tx.append_alert_once(
            directory,
            ledger,
            replica_id=replica.replica_id,
            health=health,
            now=now,
        )
        health["cancel_state"] = "requested"
        health["cancel_requested_at"] = float(now)
        health["next_cancel_eligible_at"] = float(now)
        fleet_tx.save_ledger(directory, ledger, now=now)
    next_eligible = health["next_cancel_eligible_at"]
    if next_eligible is not None and float(now) < float(next_eligible):
        return
    if health["cancel_attempts"] >= HUNG_CANCEL_MAX_ATTEMPTS:
        health["cancel_state"] = "exhausted"
        health["cancel_error"] = (
            f"exact-ID scancel exhausted after {health['cancel_attempts']} attempts"
        )
        fleet_tx.save_ledger(directory, ledger, now=now)
        return

    # Persist each bounded external-mutation attempt first.  Exact-ID scancel is
    # idempotent; a successor retries only this same fenced job after backoff when a
    # crash may have occurred on either side of the external call.
    health["cancel_state"] = "requested"
    health["cancel_attempts"] += 1
    health["last_cancel_attempt_at"] = float(now)
    backoff = min(
        HUNG_CANCEL_BACKOFF_SECONDS * (2 ** (health["cancel_attempts"] - 1)),
        1800.0,
    )
    health["next_cancel_eligible_at"] = float(now) + backoff
    fleet_tx.save_ledger(directory, ledger, now=now)
    try:
        proc = invoke_cancel(
            ["scancel", str(row.job_id)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except BaseException as exc:
        health["cancel_state"] = "retryable"
        health["cancel_error"] = f"scancel invocation failed: {exc!r}"
        fleet_tx.save_ledger(directory, ledger, now=now)
        raise
    if proc.returncode != 0:
        health["cancel_state"] = (
            "exhausted"
            if health["cancel_attempts"] >= HUNG_CANCEL_MAX_ATTEMPTS
            else "retryable"
        )
        health["cancel_error"] = (
            f"scancel rc={proc.returncode}: {proc.stderr.strip()[:500]}"
        )
        fleet_tx.save_ledger(directory, ledger, now=now)
        print(f"[keepalive] retryable hung-job cancellation: {health['cancel_error']}")
        return
    health["cancel_state"] = "accepted"
    health["cancel_completed_at"] = float(now)
    fleet_tx.save_ledger(directory, ledger, now=now)
    print(
        f"[keepalive] fenced hung {replica.replica_id}: exact job {row.job_id} "
        f"after {health['consecutive_failures']} dual probe failures over {failure_span:.0f}s"
    )


def _attempt_by_job_id(
    ledgers: list[dict],
    *,
    replica_id: str,
    job_id: str,
) -> tuple[dict, dict] | None:
    matches: list[tuple[dict, dict]] = []
    for owner in ledgers:
        for attempt in owner["replicas"][replica_id]["attempts"]:
            if str(attempt.get("job_id") or "") == str(job_id):
                matches.append((owner, attempt))
    if len(matches) > 1:
        raise FleetContractError(
            f"job {job_id} is bound to multiple intents for {replica_id}"
        )
    return matches[0] if matches else None


def _handoff_registry_entry(
    *,
    run_root: str,
    replica,
    row: FleetQueueRow,
    provenance: SpooledServingProvenance,
    attempt: dict,
    now: float,
) -> tuple[registry.ServerEntry, bool]:
    """Return one standby or crash-promoted entry after exact provenance checks."""

    try:
        entry = registry.read_standby_entry(
            run_root,
            replica.serving_profile,
            replica.replica_id,
            str(row.job_id),
        )
        if entry is None:
            promoted = registry.read_promoted_entry(
                run_root,
                replica.serving_profile,
                replica.replica_id,
            )
            if (
                promoted is not None
                and str(promoted.slurm_job_id or "") == str(row.job_id)
            ):
                entry = promoted
    except ValueError as exc:
        raise FleetContractError(str(exc)) from exc
    persisted = entry is not None
    if entry is None:
        profile = get_serving_profile(replica.serving_profile)
        started_at = next(
            (
                float(value)
                for value in (
                    attempt.get("committed_at"),
                    attempt.get("submitted_at"),
                    attempt.get("created_at"),
                    now,
                )
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > 0
            ),
            float(now),
        )
        from agents_scaling.serving.launch_server import _port_for

        entry = registry.ServerEntry(
            model_size=profile.model_size,
            hf_id=profile.hf_id,
            host=row.node,
            port=_port_for(
                replica.serving_profile, replica.replica_index
            ),
            slurm_job_id=str(row.job_id),
            started_at=started_at,
            serving_profile=profile.name,
            served_model_name=profile.served_model_name,
            max_model_len=profile.max_model_len,
            tp_size=profile.tp_size,
            release_id=provenance.release_id,
            environment_hash=provenance.environment_hash,
            model_revision=provenance.model_revision,
            tokenizer_id=provenance.tokenizer_id,
            tokenizer_revision=provenance.tokenizer_revision,
            model_contract_sha256=provenance.model_contract_sha256,
            fleet_contract_sha256=provenance.fleet_contract_sha256,
            server_pool_id=provenance.server_pool_id,
            replica_id=provenance.replica_id,
            replica_index=provenance.replica_index,
            release_fleet_contract_sha256=(
                provenance.release_fleet_contract_sha256
            ),
            capacity_generation=provenance.capacity_generation,
            rollout_generation=provenance.rollout_generation,
        )
    from agents_scaling.serving.launch_server import _port_for

    exact = {
        "host": row.node,
        "port": _port_for(replica.serving_profile, replica.replica_index),
        "slurm_job_id": str(row.job_id),
        "serving_profile": replica.serving_profile,
        "server_pool_id": replica.pool_id,
        "replica_id": replica.replica_id,
        "replica_index": replica.replica_index,
        "release_id": provenance.release_id,
        "environment_hash": provenance.environment_hash,
        "model_revision": provenance.model_revision,
        "tokenizer_id": provenance.tokenizer_id,
        "tokenizer_revision": provenance.tokenizer_revision,
        "model_contract_sha256": provenance.model_contract_sha256,
        "fleet_contract_sha256": provenance.fleet_contract_sha256,
        "release_fleet_contract_sha256": (
            provenance.release_fleet_contract_sha256
        ),
        "capacity_generation": provenance.capacity_generation,
        "rollout_generation": provenance.rollout_generation,
    }
    if (
        not registry.entry_has_current_provenance(
            entry, replica.serving_profile
        )
        or any(getattr(entry, field) != value for field, value in exact.items())
    ):
        raise FleetContractError(
            f"standby registry provenance drift for {replica.replica_id} "
            f"job {row.job_id}"
        )
    return entry, persisted


def _complete_endpoint_lineage(
    provenance: SpooledServingProvenance,
) -> bool:
    return bool(
        isinstance(provenance.release_fleet_contract_sha256, str)
        and re.fullmatch(
            r"[0-9a-f]{64}", provenance.release_fleet_contract_sha256
        )
        and isinstance(provenance.fleet_contract_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", provenance.fleet_contract_sha256)
        and type(provenance.capacity_generation) is int
        and provenance.capacity_generation > 0
        and type(provenance.rollout_generation) is int
        and provenance.rollout_generation > 0
        and isinstance(provenance.spooled_script_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", provenance.spooled_script_sha256)
    )


def _seal_endpoint_lineage(
    *,
    directory: Path,
    run_root: str,
    entry: registry.ServerEntry,
    replica,
    owner_ledger: dict,
    attempt: dict,
    row: FleetQueueRow,
    provenance: SpooledServingProvenance,
    now: float,
) -> registry.EndpointHistoryRecord | None:
    """Seal the exact committed admission before a schema-5 pointer is routable."""

    if not _complete_endpoint_lineage(provenance):
        return None
    try:
        admission = fleet_tx.committed_endpoint_admission(
            directory,
            owner_ledger,
            replica_id=replica.replica_id,
            attempt=attempt,
            slurm_job_id=str(row.job_id),
        )
    except fleet_tx.FleetTransactionError as exc:
        raise FleetContractError(str(exc)) from exc
    if (
        admission.rollout_generation != provenance.rollout_generation
        or admission.scheduler_comment != row.comment
        or str(entry.slurm_job_id or "") != str(row.job_id)
        or entry.release_fleet_contract_sha256
        != provenance.release_fleet_contract_sha256
        or entry.fleet_contract_sha256 != provenance.fleet_contract_sha256
        or entry.capacity_generation != provenance.capacity_generation
        or entry.rollout_generation != provenance.rollout_generation
    ):
        raise FleetContractError(
            f"endpoint/intent generation drift for {replica.replica_id} "
            f"job {row.job_id}"
        )
    script = admission.sbatch_path.read_bytes()
    if hashlib.sha256(script).hexdigest() != provenance.spooled_script_sha256:
        raise FleetContractError(
            f"actual spooled script hash drift for {replica.replica_id} "
            f"job {row.job_id}"
        )
    spooled_provenance = {
        "run_root": str(Path(run_root).expanduser().resolve()),
        "server_pool_id": provenance.server_pool_id,
        "replica_id": provenance.replica_id,
        "replica_index": provenance.replica_index,
        "release_id": provenance.release_id,
        "environment_hash": provenance.environment_hash,
        "model_revision": provenance.model_revision,
        "tokenizer_id": provenance.tokenizer_id,
        "tokenizer_revision": provenance.tokenizer_revision,
        "model_contract_sha256": provenance.model_contract_sha256,
        "release_fleet_contract_sha256": (
            provenance.release_fleet_contract_sha256
        ),
        "fleet_contract_sha256": provenance.fleet_contract_sha256,
        "capacity_generation": provenance.capacity_generation,
        "rollout_generation": provenance.rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": provenance.spooled_script_sha256,
    }
    try:
        return registry.seal_endpoint_history(
            run_root,
            entry,
            release_fleet_contract_sha256=str(
                provenance.release_fleet_contract_sha256
            ),
            capacity_generation=int(provenance.capacity_generation),
            rollout_generation=int(provenance.rollout_generation),
            ledger_generation=admission.rollout_generation,
            intent_token=admission.intent_token,
            intent_state="committed",
            committed_at=admission.committed_at,
            local_script_path=admission.sbatch_path,
            local_script_sha256=admission.sbatch_sha256,
            # _validate_scheduler_row accepted this exact local script only after
            # Slurm's `write batch_script` returned byte-for-byte equality.
            spooled_script=script,
            spooled_provenance=spooled_provenance,
            scheduler_job_name=row.job_name,
            scheduler_comment=row.comment,
            sealed_at=now,
        )
    except ValueError as exc:
        raise FleetContractError(str(exc)) from exc


def _advance_primary_registration(
    *,
    directory: Path,
    run_root: str,
    replica,
    owner_ledger: dict,
    attempt: dict,
    row: FleetQueueRow,
    provenance: SpooledServingProvenance,
    now: float,
    probe=None,
) -> bool:
    """Stage, dual-probe, seal, then atomically route one initial primary."""

    if (
        attempt["lifecycle"] != "primary"
        or row.state.upper() != "RUNNING"
        or not _complete_endpoint_lineage(provenance)
    ):
        return False
    try:
        registered = registry.read_standby_entry(
            run_root,
            replica.serving_profile,
            replica.replica_id,
            str(row.job_id),
        )
        if registered is None:
            promoted = registry.read_promoted_entry(
                run_root,
                replica.serving_profile,
                replica.replica_id,
            )
            if (
                promoted is not None
                and str(promoted.slurm_job_id or "") == str(row.job_id)
            ):
                registered = promoted
    except ValueError as exc:
        raise FleetContractError(str(exc)) from exc
    if registered is None:
        # Never synthesize a production registration timestamp.  The serving process
        # publishes its exact ServerEntry after local vLLM readiness; preserving that
        # byte preimage is part of endpoint lineage.
        return False
    entry, persisted = _handoff_registry_entry(
        run_root=run_root,
        replica=replica,
        row=row,
        provenance=provenance,
        attempt=attempt,
        now=now,
    )
    invoke_probe = probe or _probe_health_and_models
    try:
        health_ok, models_ok = invoke_probe(entry.host, entry.port)
    except Exception as exc:  # noqa: BLE001
        print(f"[keepalive] primary probe error for {replica.replica_id}: {exc!r}")
        health_ok, models_ok = False, False
    attempt["last_ready_probe_at"] = float(now)
    attempt["ready_probe_count"] = (
        int(attempt["ready_probe_count"]) + 1
        if health_ok and models_ok
        else 0
    )
    if health_ok and models_ok and not persisted:
        try:
            registry.write_standby_entry(run_root, entry)
        except ValueError as exc:
            raise FleetContractError(str(exc)) from exc
    fleet_tx.save_ledger(directory, owner_ledger, now=now)
    if attempt["ready_probe_count"] < HANDOFF_READY_PROBES:
        return False
    _seal_endpoint_lineage(
        directory=directory,
        run_root=run_root,
        entry=entry,
        replica=replica,
        owner_ledger=owner_ledger,
        attempt=attempt,
        row=row,
        provenance=provenance,
        now=now,
    )
    try:
        registry.promote_standby_entry(run_root, entry)
        promoted = registry.read_promoted_entry(
            run_root, replica.serving_profile, replica.replica_id
        )
    except ValueError as exc:
        raise FleetContractError(str(exc)) from exc
    if promoted != entry:
        raise FleetContractError(
            f"atomic primary pointer did not bind job {row.job_id}"
        )
    return True


def _advance_handoff(
    *,
    directory: Path,
    ledgers: list[dict],
    run_root: str,
    replica,
    owner_ledger: dict,
    attempt: dict,
    row: FleetQueueRow,
    provenance: SpooledServingProvenance,
    now: float,
    probe=None,
) -> bool:
    """Probe, atomically promote, and crash-resume one typed standby."""

    if attempt["lifecycle"] != "standby" or row.state.upper() != "RUNNING":
        return False
    if _complete_endpoint_lineage(provenance):
        try:
            registered = registry.read_standby_entry(
                run_root,
                replica.serving_profile,
                replica.replica_id,
                str(row.job_id),
            )
            if registered is None:
                promoted = registry.read_promoted_entry(
                    run_root,
                    replica.serving_profile,
                    replica.replica_id,
                )
                if (
                    promoted is not None
                    and str(promoted.slurm_job_id or "") == str(row.job_id)
                ):
                    registered = promoted
        except ValueError as exc:
            raise FleetContractError(str(exc)) from exc
        if registered is None:
            return False
    entry, persisted = _handoff_registry_entry(
        run_root=run_root,
        replica=replica,
        row=row,
        provenance=provenance,
        attempt=attempt,
        now=now,
    )

    invoke_probe = probe or _probe_health_and_models
    try:
        health_ok, models_ok = invoke_probe(entry.host, entry.port)
    except Exception as exc:  # noqa: BLE001 - transport failures are failed probes
        print(f"[keepalive] standby probe error for {replica.replica_id}: {exc!r}")
        health_ok, models_ok = False, False
    attempt["last_ready_probe_at"] = float(now)
    attempt["ready_probe_count"] = (
        int(attempt["ready_probe_count"]) + 1
        if health_ok and models_ok
        else 0
    )
    if health_ok and models_ok and not persisted:
        try:
            registry.write_standby_entry(run_root, entry)
        except ValueError as exc:
            raise FleetContractError(str(exc)) from exc
    fleet_tx.save_ledger(directory, owner_ledger, now=now)
    if attempt["ready_probe_count"] < HANDOFF_READY_PROBES:
        return False

    predecessor_id = str(attempt["predecessor_job_id"])
    matched = _attempt_by_job_id(
        ledgers,
        replica_id=replica.replica_id,
        job_id=predecessor_id,
    )
    if matched is None:
        raise FleetContractError(
            f"handoff for {replica.replica_id} lost predecessor {predecessor_id}"
        )
    predecessor_owner, predecessor = matched
    if predecessor["lifecycle"] not in {"primary", "promoted", "retiring"}:
        raise FleetContractError(
            f"handoff predecessor lifecycle drift for {replica.replica_id}"
        )
    _seal_endpoint_lineage(
        directory=directory,
        run_root=run_root,
        entry=entry,
        replica=replica,
        owner_ledger=owner_ledger,
        attempt=attempt,
        row=row,
        provenance=provenance,
        now=now,
    )
    if predecessor["lifecycle"] != "retiring":
        # Write-ahead: if the supervisor dies here, readiness fails closed and the
        # next supervisor resumes promotion from the durable standby.
        predecessor["lifecycle"] = "retiring"
        fleet_tx.save_ledger(directory, predecessor_owner, now=now)

    try:
        promoted = registry.read_promoted_entry(
            run_root,
            replica.serving_profile,
            replica.replica_id,
        )
        if promoted != entry:
            registry.promote_standby_entry(run_root, entry)
            promoted = registry.read_promoted_entry(
                run_root,
                replica.serving_profile,
                replica.replica_id,
            )
    except ValueError as exc:
        raise FleetContractError(str(exc)) from exc
    if promoted != entry:
        raise FleetContractError(
            f"atomic promoted pointer did not bind job {row.job_id}"
        )

    attempt["lifecycle"] = "promoted"
    if attempt["promoted_at"] is None:
        attempt["promoted_at"] = float(now)
    attempt["retire_error"] = None
    owner_ledger["replicas"][replica.replica_id]["health"] = None
    fleet_tx.save_ledger(directory, owner_ledger, now=now)
    print(
        f"[keepalive] PROMOTE {replica.replica_id}: job {row.job_id} "
        f"replaces exact job {predecessor_id}"
    )
    return True


def _retire_handoff_predecessor(
    *,
    directory: Path,
    ledgers: list[dict],
    replica,
    promoted_attempt: dict,
    scheduler_rows_by_job: dict[str, FleetQueueRow],
    now: float,
    cancellation_runner=None,
) -> None:
    """Drain, then retry a bounded exact-ID retirement with write-ahead state."""

    promoted_at = promoted_attempt.get("promoted_at")
    if promoted_at is None or now < float(promoted_at) + HANDOFF_DRAIN_SECONDS:
        return
    predecessor_id = str(promoted_attempt["predecessor_job_id"])
    matched = _attempt_by_job_id(
        ledgers,
        replica_id=replica.replica_id,
        job_id=predecessor_id,
    )
    if matched is None:
        raise FleetContractError(
            f"promoted handoff lost predecessor {predecessor_id}"
        )
    owner, predecessor = matched
    row = scheduler_rows_by_job.get(predecessor_id)
    if row is None or fleet_tx.terminal_state(row.state):
        return
    if predecessor["lifecycle"] != "retiring":
        raise FleetContractError(
            f"promoted handoff predecessor is not retiring for {replica.replica_id}"
        )
    last = predecessor["last_retire_attempt_at"]
    if (
        last is not None
        and now < float(last) + HANDOFF_RETIRE_BACKOFF_SECONDS
    ):
        return
    if predecessor["retire_attempts"] >= HANDOFF_RETIRE_MAX_ATTEMPTS:
        predecessor["retire_error"] = (
            f"exact-ID retirement exhausted after "
            f"{predecessor['retire_attempts']} attempts"
        )
        fleet_tx.save_ledger(directory, owner, now=now)
        return
    if cancellation_runner is None and os.environ.get("PYTEST_CURRENT_TEST"):
        raise FleetContractError(
            "pytest handoff retirement must provide an injected non-Slurm runner"
        )
    predecessor["retire_requested_at"] = (
        predecessor["retire_requested_at"] or float(now)
    )
    predecessor["last_retire_attempt_at"] = float(now)
    predecessor["retire_attempts"] += 1
    predecessor["retire_error"] = None
    fleet_tx.save_ledger(directory, owner, now=now)
    invoke = cancellation_runner or subprocess.run
    try:
        proc = invoke(
            ["scancel", predecessor_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except BaseException as exc:
        predecessor["retire_error"] = f"scancel invocation failed: {exc!r}"
        fleet_tx.save_ledger(directory, owner, now=now)
        raise
    if proc.returncode != 0:
        predecessor["retire_error"] = (
            f"scancel rc={proc.returncode}: {proc.stderr.strip()[:500]}"
        )
        fleet_tx.save_ledger(directory, owner, now=now)
        return
    fleet_tx.save_ledger(directory, owner, now=now)
    print(
        f"[keepalive] RETIRE {replica.replica_id}: exact predecessor "
        f"job {predecessor_id}"
    )


def tick_fleet(
    run_root: str,
    fleet: FrozenFleetContract,
    *,
    launch_options: dict[str, str | int | None],
    now: float | None = None,
    submission_runner=None,
    health_probe=None,
    cancellation_runner=None,
    scheduler_safety_contract: Mapping[str, str | None] | None = None,
    scheduler_safety_runner=None,
    protected_capacity_contract: (
        protected_capacity.ProtectedCapacityContract | None
    ) = None,
    control_state_dir: str | Path | None = None,
) -> None:
    """Transactionally reconcile, validate, probe, and recover every frozen replica."""

    canonical_root = fleet.verify_pool_root(run_root)
    generation = _rollout_generation(launch_options)
    capacity_generation: int | None = None
    timestamp = time.time() if now is None else float(now)
    try:
        with fleet_tx.transaction_lock(canonical_root) as directory:
            replica_contracts = {
                replica.replica_id: replica for replica in fleet.replicas
            }
            if scheduler_safety_contract is not None:
                if protected_capacity_contract is None:
                    raise FleetContractError(
                        "production fleet lacks a sealed protected-capacity contract"
                    )
                try:
                    protected_capacity.authorize_fleet(
                        fleet, protected_capacity_contract
                    )
                except protected_capacity.ProtectedCapacityError as exc:
                    raise FleetContractError(
                        f"fleet protected-placement authorization failed: {exc}"
                    ) from exc
                if set(scheduler_safety_contract) != {
                    "expected_policy_contract_id",
                    "transport_uncertainty_binding_sha256",
                }:
                    raise FleetContractError(
                        "fleet scheduler-safety contract has the wrong fields"
                    )
                _validate_live_fleet_scheduler_policy(
                    fleet,
                    expected_policy_contract_id=(
                        scheduler_safety_contract[
                            "expected_policy_contract_id"
                        ]
                    ),
                    expected_transport_binding_sha256=str(
                        scheduler_safety_contract[
                            "transport_uncertainty_binding_sha256"
                        ]
                    ),
                    runner=scheduler_safety_runner,
                    captured_timestamp=timestamp,
                )
                capacity_generation = _capacity_generation(
                    launch_options
                )
                if control_state_dir is None or capacity_generation is None:
                    raise FleetContractError(
                        "protected fleet supervision lacks its current "
                        "control/generation authority"
                    )
                _, protected_capacity_contract = _load_current_fleet_authority(
                    control_state_dir,
                    fleet=fleet,
                    capacity_generation=capacity_generation,
                    rollout_generation=generation,
                    expected_protected_capacity_contract=(
                        protected_capacity_contract
                    ),
                )

            def current_fleet_bindings() -> dict[str, dict[str, Any]]:
                bindings: dict[str, dict[str, Any]] = {}
                for replica_id, replica_rows in active_rows.items():
                    replica = replica_contracts.get(replica_id)
                    if replica is None:
                        raise FleetContractError(
                            f"trusted fleet binding references unknown replica "
                            f"{replica_id}"
                        )
                    for row, owner_ledger, attempt, _provenance in replica_rows:
                        job_id = str(row.job_id)
                        if job_id in bindings:
                            raise FleetContractError(
                                f"trusted fleet binding repeats job {job_id}"
                            )
                        owner_generation = int(
                            owner_ledger["rollout_generation"]
                        )
                        owner_ledger_path = fleet_tx.ledger_path(
                            directory,
                            owner_generation,
                        )
                        try:
                            (
                                stable_owner_ledger_path,
                                owner_ledger_raw,
                            ) = fleet_tx._stable_regular_preimage(  # noqa: SLF001
                                owner_ledger_path,
                                description=(
                                    f"trusted fleet generation ledger "
                                    f"g{owner_generation:06d}"
                                ),
                                read_only=False,
                            )
                        except fleet_tx.FleetTransactionError as exc:
                            raise FleetContractError(str(exc)) from exc
                        allocated_gpus = attempt.get("allocated_gpus")
                        if (
                            type(allocated_gpus) is not int
                            or allocated_gpus != replica.gpus_per_replica
                        ):
                            raise FleetContractError(
                                f"trusted fleet attempt GPU allocation drifted "
                                f"for {replica_id}"
                            )
                        bindings[job_id] = {
                            "job_name": row.job_name,
                            "comment": row.comment,
                            "sbatch_path": str(attempt["sbatch_path"]),
                            "sbatch_sha256": str(attempt["sbatch_sha256"]),
                            "intent_token": str(attempt["intent_token"]),
                            "replica_id": str(replica_id),
                            "serving_profile": replica.serving_profile,
                            "ledger_generation": owner_generation,
                            "ledger_path": str(stable_owner_ledger_path),
                            "ledger_sha256": hashlib.sha256(
                                owner_ledger_raw
                            ).hexdigest(),
                            "partition": replica.partition,
                            "qos": replica.qos,
                            "allocated_gpus": allocated_gpus,
                            "gpu_type": replica.gpu_type,
                        }
                return bindings

            def verify_submit_placement(replica) -> None:
                if protected_capacity_contract is None:
                    if scheduler_safety_contract is not None:
                        raise FleetContractError(
                            "protected capacity disappeared before fleet sbatch"
                        )
                    return
                if control_state_dir is None:
                    raise FleetContractError(
                        "protected fleet submission lacks the shared control-state "
                        "directory required for exact dispatcher/fleet occupancy"
                    )
                try:
                    from slurm import schema5_control as control_plane
                except (ImportError, OSError) as exc:
                    raise FleetContractError(
                        "protected fleet submission cannot import its current "
                        f"control authority: {exc}"
                    ) from exc

                try:
                    if capacity_generation is None:
                        raise FleetContractError(
                            "protected fleet submission lacks its capacity generation"
                        )
                    (
                        control_state,
                        current_protected_contract,
                    ) = _load_current_fleet_authority(
                        control_state_dir,
                        fleet=fleet,
                        capacity_generation=capacity_generation,
                        rollout_generation=generation,
                        expected_protected_capacity_contract=(
                            protected_capacity_contract
                        ),
                    )
                    boundary_now = (
                        timestamp if now is not None else time.time()
                    )
                    trusted_provenance = (
                        control_plane.reconcile_trusted_scientific_job_provenance(
                            Path(control_state_dir),
                            fleet_bindings=current_fleet_bindings(),
                            fleet_contract_sha256=fleet.sha256,
                            fleet_generation=generation,
                            now=boundary_now,
                            allow_exact_cell_quiescence=(
                                control_state.get("desired_state") != "running"
                            ),
                        )
                    )
                    protected_capacity.verify_live_placements(
                        current_protected_contract,
                        role="server",
                        placements=[(replica.partition, replica.qos)],
                        required_time_limits_seconds={
                            replica.partition: (
                                protected_capacity.MIN_SCIENTIFIC_WALL_SECONDS
                            )
                        },
                        trusted_scientific_job_provenance=trusted_provenance,
                        runner=scheduler_safety_runner,
                    )
                except (
                    control_plane.ControlError,
                    protected_capacity.ProtectedCapacityError,
                ) as exc:
                    raise FleetContractError(
                        f"live protected placement rejected "
                        f"{replica.replica_id} before sbatch: {exc}"
                    ) from exc
            ledgers = fleet_tx.load_generation_ledgers(
                directory,
                pool_root=canonical_root,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                current_generation=generation,
                replica_ids=[replica.replica_id for replica in fleet.replicas],
                now=timestamp,
            )
            ledger = ledgers[-1]
            queue_truth = _query_fleet_queue()
            rows = tuple(queue_truth)
            scheduler_snapshot = getattr(
                queue_truth, "scheduler_snapshot", None
            )
            reconciled = fleet_tx.reconcile_scheduler_rows(
                rows,
                ledgers,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                replica_profiles={
                    replica.replica_id: replica.serving_profile
                    for replica in fleet.replicas
                },
                replica_job_names={
                    replica.replica_id: replica.scheduler_job_name
                    for replica in fleet.replicas
                },
                replica_qos={
                    replica.replica_id: replica.qos
                    for replica in fleet.replicas
                },
            )
            rows_by_token = {
                str(allocation.attempt["intent_token"]): [allocation.row]
                for allocation in reconciled.allocations
            }

            active_rows: dict[
                str,
                list[
                    tuple[
                        FleetQueueRow,
                        dict,
                        dict,
                        SpooledServingProvenance,
                    ]
                ],
            ] = {}
            grace = float(ledger["visibility_grace_seconds"])
            for replica in fleet.replicas:
                generation_records = [
                    (owner, owner["replicas"][replica.replica_id])
                    for owner in ledgers
                ]
                record = ledger["replicas"][replica.replica_id]
                current_script = _expected_fleet_script(
                    replica, str(canonical_root), launch_options
                )
                for owner_ledger, owner_record in generation_records:
                    changed = False
                    owner_generation = int(owner_ledger["rollout_generation"])
                    for attempt in owner_record["attempts"]:
                        token_rows = rows_by_token.get(attempt["intent_token"], [])
                        if token_rows:
                            row = token_rows[0]
                            attempt_script = Path(attempt["sbatch_path"]).read_text(
                                encoding="utf-8"
                            )
                            _validate_fleet_script_contract(
                                attempt_script,
                                replica=replica,
                                rollout_generation=owner_generation,
                                fleet_sha256=fleet.sha256,
                                standby=attempt["launch_kind"] == "handoff",
                            )
                            provenance = _validate_scheduler_row(
                                row,
                                replica=replica,
                                attempt=attempt,
                                fleet=fleet,
                                run_root=str(canonical_root),
                                expected_script=attempt_script,
                            )
                            if attempt["job_id"] not in {None, str(row.job_id)}:
                                raise FleetContractError(
                                    f"fleet intent {attempt['intent_token']} changed job id"
                                )
                            attempt["job_id"] = str(row.job_id)
                            if attempt["submitted_at"] is None:
                                attempt["submitted_at"] = timestamp
                                changed = True
                            timing_updates = {
                                "last_seen_at": timestamp,
                                "scheduler_start_at": row.start_timestamp,
                                "scheduler_end_at": row.end_timestamp,
                                "scheduler_time_limit_seconds": (
                                    row.time_limit_seconds
                                ),
                            }
                            for field, value in timing_updates.items():
                                if attempt.get(field) != value:
                                    attempt[field] = value
                                    changed = True
                            attempt["missing_since"] = None
                            if fleet_tx.terminal_state(row.state):
                                if attempt["state"] != "terminal":
                                    attempt["state"] = "terminal"
                                    attempt["terminal_at"] = timestamp
                                    changed = True
                                health = owner_record["health"]
                                if (
                                    isinstance(health, dict)
                                    and health.get("job_id") == str(row.job_id)
                                ):
                                    owner_record["health"] = None
                                    changed = True
                            else:
                                # An active allocation from an older rollout remains
                                # admissible when its immutable fleet/release/script
                                # provenance still validates.  Only its replacement is
                                # rendered under the current generation.
                                if attempt["committed_at"] is None:
                                    attempt["committed_at"] = timestamp
                                    changed = True
                                if attempt["state"] != "committed":
                                    attempt["state"] = "committed"
                                    changed = True
                                if provenance is None:
                                    raise FleetContractError(
                                        f"active fleet job {row.job_id} lacks "
                                        "immutable serving provenance"
                                    )
                                active_rows.setdefault(
                                    replica.replica_id, []
                                ).append(
                                    (
                                        row,
                                        owner_ledger,
                                        attempt,
                                        provenance,
                                    )
                                )
                            continue

                        state = attempt["state"]
                        basis = attempt["submit_started_at"] or attempt["created_at"]
                        age = timestamp - float(basis)
                        if state == "prepared" and owner_generation < generation:
                            attempt["state"] = "terminal"
                            attempt["terminal_at"] = timestamp
                            attempt["last_error"] = (
                                "unsubmitted intent superseded by a newer rollout generation"
                            )
                            changed = True
                        elif state == "submitted" and age >= grace:
                            # A numeric job ID returned by sbatch is positive acceptance
                            # evidence.  If that exact ID is absent from both complete
                            # scheduler views, replacing it would turn accounting lag or
                            # scheduler ambiguity into a duplicate GPU allocation.  Keep
                            # the accepted intent fenced and require operator/scheduler
                            # reconciliation instead of converting it into a retry.
                            raise FleetContractError(
                                f"accepted fleet job {attempt.get('job_id')} for "
                                f"{replica.replica_id} disappeared from complete "
                                "squeue+sacct truth; refusing a duplicate replacement"
                            )
                        elif state == "submitting" and age >= grace:
                            # Old-generation scripts are never resubmitted after a
                            # pause/resume.  A current-generation retry must cross the
                            # one typed absence transition so its complete joined
                            # scheduler proof remains independently auditable.
                            if owner_generation != generation:
                                attempt["state"] = "terminal"
                                attempt["terminal_at"] = timestamp
                                attempt["last_error"] = (
                                    "old-generation ambiguous submission absent "
                                    "from complete scheduler truth after visibility "
                                    "grace"
                                )
                                changed = True
                            else:
                                if not isinstance(
                                    scheduler_snapshot,
                                    fleet_tx.SchedulerSnapshot,
                                ):
                                    raise FleetContractError(
                                        "ambiguous fleet submission retry lacks "
                                        "its exact complete scheduler snapshot"
                                    )
                                absence_timestamp = (
                                    time.time()
                                    if now is None
                                    else timestamp
                                )
                                fleet_tx.record_proven_submission_absence(
                                    directory,
                                    owner_ledger,
                                    replica_id=replica.replica_id,
                                    attempt=attempt,
                                    snapshot=scheduler_snapshot,
                                    now=absence_timestamp,
                                )
                        elif state in {"committed", "missing"}:
                            if attempt["missing_since"] is None:
                                attempt["state"] = "missing"
                                attempt["missing_since"] = timestamp
                                changed = True
                            elif timestamp - float(attempt["missing_since"]) >= grace:
                                attempt["state"] = "terminal"
                                attempt["terminal_at"] = timestamp
                                attempt["last_error"] = (
                                    "previously committed job absent from complete "
                                    "scheduler truth for visibility grace"
                                )
                                health = owner_record["health"]
                                if (
                                    isinstance(health, dict)
                                    and health.get("job_id") == str(attempt.get("job_id"))
                                ):
                                    owner_record["health"] = None
                                changed = True
                    if changed:
                        fleet_tx.save_ledger(
                            directory, owner_ledger, now=timestamp
                        )

                current_entries = [
                    (owner, item)
                    for owner, owner_record in generation_records
                    for item in owner_record["attempts"]
                    if item["state"]
                    in {
                        "prepared",
                        "submitting",
                        "submitted",
                        "committed",
                        "missing",
                    }
                ]
                current = [item for _owner, item in current_entries]
                failed = [
                    item
                    for item in record["attempts"]
                    if item["state"] == "submission_failed"
                ]
                if len(current) > 2:
                    raise FleetContractError(
                        f"too many current intents for {replica.replica_id}"
                    )
                prepared = [
                    (owner, item)
                    for owner, item in current_entries
                    if item["state"] == "prepared"
                ]
                for owner, pending in prepared:
                    verify_submit_placement(replica)
                    submission_timestamp = (
                        timestamp if now is not None else time.time()
                    )
                    job_id = fleet_tx.submit_attempt(
                        directory,
                        owner,
                        replica_id=replica.replica_id,
                        attempt=pending,
                        now=submission_timestamp,
                        runner=submission_runner,
                    )
                    print(
                        f"[keepalive] RECOVER pre-sbatch {replica.replica_id} "
                        f"-> job {job_id}"
                    )
                if prepared:
                    continue

                retry_submitted = False
                if failed:
                    retry = failed[-1]
                    retry_basis = retry["submit_started_at"] or retry["created_at"]
                    stable = [
                        item
                        for item in current
                        if item["lifecycle"] in {"primary", "promoted"}
                        and str(item.get("job_id") or "").isdigit()
                    ]
                    retry_allowed = (
                        retry["launch_kind"] == "primary" and not current
                    ) or (
                        retry["launch_kind"] == "handoff"
                        and len(current) == 1
                        and len(stable) == 1
                        and retry["predecessor_job_id"] == stable[0]["job_id"]
                    )
                    if (
                        retry_allowed
                        and timestamp - float(retry_basis) >= grace
                    ):
                        verify_submit_placement(replica)
                        submission_timestamp = (
                            timestamp if now is not None else time.time()
                        )
                        job_id = fleet_tx.submit_attempt(
                            directory,
                            ledger,
                            replica_id=replica.replica_id,
                            attempt=retry,
                            now=submission_timestamp,
                            runner=submission_runner,
                        )
                        print(
                            f"[keepalive] RETRY {replica.replica_id} intent "
                            f"{retry['intent_token']} -> job {job_id}"
                        )
                        retry_submitted = True
                    elif (
                        retry["launch_kind"] == "handoff"
                        and not stable
                    ):
                        retry["state"] = "terminal"
                        retry["terminal_at"] = timestamp
                        retry["last_error"] = (
                            "handoff predecessor ended before retry eligibility"
                        )
                        fleet_tx.save_ledger(directory, ledger, now=timestamp)
                        failed = []
                if retry_submitted:
                    continue
                if not current:
                    # A primary submission failure remains a durable fence throughout
                    # its visibility/backoff window; never create a second token.
                    if failed:
                        continue
                    verify_submit_placement(replica)
                    attempt = fleet_tx.prepare_attempt(
                        directory,
                        ledger,
                        replica_id=replica.replica_id,
                        profile=replica.serving_profile,
                        pool_id=replica.pool_id,
                        fleet_sha256=fleet.sha256,
                        rollout_generation=generation,
                        sbatch_text=current_script,
                        now=timestamp,
                        allocated_gpus=replica.gpus_per_replica,
                    )
                    verify_submit_placement(replica)
                    submission_timestamp = (
                        timestamp if now is not None else time.time()
                    )
                    job_id = fleet_tx.submit_attempt(
                        directory,
                        ledger,
                        replica_id=replica.replica_id,
                        attempt=attempt,
                        now=submission_timestamp,
                        runner=submission_runner,
                    )
                    print(
                        f"[keepalive] LAUNCH {replica.replica_id} on "
                        f"{replica.partition} -> job {job_id}"
                    )

            # Promotion, registration repair, and cancellation run only after every
            # scheduler row and transaction has validated.  Ambiguous scheduler truth
            # therefore cannot mutate a pointer or retire an allocation.
            from agents_scaling.serving.launch_server import _port_for

            scheduler_rows_by_job = {row.job_id: row for row in rows}
            replica_by_id = {
                replica.replica_id: replica for replica in fleet.replicas
            }

            # Initial production registrations are job-specific standbys too.  Make
            # them routable only after two clean probes and an immutable lineage seal.
            for replica in fleet.replicas:
                for (
                    row,
                    owner_ledger,
                    attempt,
                    provenance,
                ) in active_rows.get(replica.replica_id, []):
                    if attempt["lifecycle"] == "primary":
                        _advance_primary_registration(
                            directory=directory,
                            run_root=str(canonical_root),
                            replica=replica,
                            owner_ledger=owner_ledger,
                            attempt=attempt,
                            row=row,
                            provenance=provenance,
                            now=timestamp,
                            probe=health_probe,
                        )

            # First resume/progress every existing standby.  A crash after pointer
            # replacement is adopted from the exact canonical pointer; a lost staging
            # file is reconstructed only after immutable-script validation and a clean
            # dual probe.
            for replica in fleet.replicas:
                for (
                    row,
                    owner_ledger,
                    attempt,
                    provenance,
                ) in active_rows.get(replica.replica_id, []):
                    if attempt["lifecycle"] == "standby":
                        _advance_handoff(
                            directory=directory,
                            ledgers=ledgers,
                            run_root=str(canonical_root),
                            replica=replica,
                            owner_ledger=owner_ledger,
                            attempt=attempt,
                            row=row,
                            provenance=provenance,
                            now=timestamp,
                            probe=health_probe,
                        )

            # Drain and retire only the exact predecessor of a durably promoted
            # successor.  Bounded retries are write-ahead and never target a name.
            for replica in fleet.replicas:
                for (
                    _row,
                    _owner_ledger,
                    attempt,
                    _provenance,
                ) in active_rows.get(replica.replica_id, []):
                    if attempt["lifecycle"] == "promoted":
                        _retire_handoff_predecessor(
                            directory=directory,
                            ledgers=ledgers,
                            replica=replica,
                            promoted_attempt=attempt,
                            scheduler_rows_by_job=scheduler_rows_by_job,
                            now=timestamp,
                            cancellation_runner=cancellation_runner,
                        )

            # Exactly one primary/promoted allocation is routable per logical replica.
            # Standbys and retiring predecessors are never passed to generic self-heal.
            logical_rows: dict[
                str,
                tuple[
                    FleetQueueRow,
                    dict,
                    dict,
                    SpooledServingProvenance,
                ],
            ] = {}
            for replica in fleet.replicas:
                candidates = [
                    item
                    for item in active_rows.get(replica.replica_id, [])
                    if item[2]["lifecycle"] in {"primary", "promoted"}
                ]
                if len(candidates) > 1:
                    raise FleetContractError(
                        f"logical allocation ambiguity for {replica.replica_id}"
                    )
                if not candidates:
                    continue
                logical_rows[replica.replica_id] = candidates[0]
                row, owner_ledger, _attempt, _provenance = candidates[0]
                if row.state.upper() == "RUNNING":
                    _reregister_running(
                        str(canonical_root),
                        replica.serving_profile,
                        running_override={
                            (
                                row.node,
                                _port_for(
                                    replica.serving_profile,
                                    replica.replica_index,
                                ),
                            ): row.job_id
                        },
                    )
                    _monitor_running_replica(
                        directory=directory,
                        ledger=owner_ledger,
                        replica=replica,
                        row=row,
                        observer_generation=generation,
                        now=timestamp,
                        probe=health_probe,
                        cancel_runner=cancellation_runner,
                    )
                else:
                    health = owner_ledger["replicas"][replica.replica_id]["health"]
                    if isinstance(health, dict) and health.get("job_id") != row.job_id:
                        owner_ledger["replicas"][replica.replica_id]["health"] = None
                        fleet_tx.save_ledger(
                            directory, owner_ledger, now=timestamp
                        )

            # Bound physical warm overlap globally.  A submitted/pending standby already
            # consumes its replica's GPU budget even before a node is assigned.
            current_states = {
                "prepared",
                "submitting",
                "submitted",
                "committed",
                "missing",
            }
            standby_replicas: set[str] = set()
            failed_handoff_replicas: set[str] = set()
            for replica in fleet.replicas:
                for owner in ledgers:
                    for attempt in owner["replicas"][replica.replica_id]["attempts"]:
                        if (
                            attempt["launch_kind"] == "handoff"
                            and attempt["state"] in current_states
                        ):
                            standby_replicas.add(replica.replica_id)
                        if (
                            attempt["launch_kind"] == "handoff"
                            and attempt["state"] == "submission_failed"
                        ):
                            failed_handoff_replicas.add(replica.replica_id)
            overlap_gpus = sum(
                replica_by_id[replica_id].gpus_per_replica
                for replica_id in standby_replicas
            )
            if overlap_gpus > HANDOFF_MAX_OVERLAP_GPUS:
                raise FleetContractError(
                    f"warm-handoff overlap is {overlap_gpus} GPUs; "
                    f"maximum is {HANDOFF_MAX_OVERLAP_GPUS}"
                )

            candidates: list[
                tuple[
                    float,
                    str,
                    object,
                    FleetQueueRow,
                    dict,
                ]
            ] = []
            for replica_id, logical in logical_rows.items():
                row, _owner, stable_attempt, _provenance = logical
                replica = replica_by_id[replica_id]
                if (
                    row.state.upper() != "RUNNING"
                    or replica_id in standby_replicas
                    or replica_id in failed_handoff_replicas
                ):
                    continue
                if row.end_timestamp is None:
                    raise FleetContractError(
                        f"running fleet job {row.job_id} lacks a handoff deadline"
                    )
                remaining = float(row.end_timestamp) - timestamp
                if remaining <= 0:
                    raise FleetContractError(
                        f"fleet job {row.job_id} passed its scheduler end before handoff"
                    )
                if remaining <= HANDOFF_LEAD_SECONDS:
                    candidates.append(
                        (
                            float(row.end_timestamp),
                            replica_id,
                            replica,
                            row,
                            stable_attempt,
                        )
                    )

            for (
                predecessor_end,
                replica_id,
                replica,
                row,
                stable_attempt,
            ) in sorted(candidates, key=lambda item: (item[0], item[1])):
                required = replica.gpus_per_replica
                if overlap_gpus + required > HANDOFF_MAX_OVERLAP_GPUS:
                    continue
                verify_submit_placement(replica)
                handoff_script = _expected_fleet_script(
                    replica,
                    str(canonical_root),
                    launch_options,
                    standby=True,
                )
                attempt = fleet_tx.prepare_attempt(
                    directory,
                    ledger,
                    replica_id=replica_id,
                    profile=replica.serving_profile,
                    pool_id=replica.pool_id,
                    fleet_sha256=fleet.sha256,
                    rollout_generation=generation,
                    sbatch_text=handoff_script,
                    now=timestamp,
                    launch_kind="handoff",
                    predecessor_job_id=str(row.job_id),
                    predecessor_end_at=predecessor_end,
                    predecessor_attempt=stable_attempt,
                    allocated_gpus=replica.gpus_per_replica,
                )
                verify_submit_placement(replica)
                submission_timestamp = (
                    timestamp if now is not None else time.time()
                )
                job_id = fleet_tx.submit_attempt(
                    directory,
                    ledger,
                    replica_id=replica_id,
                    attempt=attempt,
                    now=submission_timestamp,
                    runner=submission_runner,
                )
                overlap_gpus += required
                print(
                    f"[keepalive] WARM {replica_id}: standby job {job_id} "
                    f"for predecessor {row.job_id}"
                )
    except fleet_tx.FleetTransactionError as exc:
        raise FleetContractError(str(exc)) from exc


def tick(run_root: str, targets: list[Target]) -> None:
    """One keepalive pass.

    Policy: trust the SLURM JOB as the liveness signal, NOT /health probes. A saturated
    healthy vLLM intermittently fails /health (and the login<->node path is flaky), so
    probe-based pruning caused thrashing/false prunes. Instead:
      * a RUNNING serve job => presumed healthy; ensure it's registered (self-heal) but
        never prune/cancel it on a probe failure alone;
      * only relaunch when the serve-JOB count is below target (a job truly ended).
    This is robust to flaky probes; the cost is we don't auto-recover a true in-job vLLM
    hang (rare; surfaces as that size's cells slowing — handle manually if it occurs).
    """
    # Replica IDs are per-SIZE (ports = base + replica), but a size may now be served from
    # several partitions. Track IDs handed out this tick per size so two same-size targets
    # (e.g. pi_tpoggio + ou_bcs_high) never collide on a replica index / port.
    assigned_rids: dict[str, set[int]] = {}
    for t in targets:
        # Self-heal: re-register any running server whose registry file was lost, so cells
        # can discover it. (Best-effort; uses a probe but only to ADD, never to remove.)
        _reregister_running(run_root, t.size)
        # Relaunch only when the JOB count is below target FOR THIS PARTITION (a server
        # actually ended). Partition-scoped so multi-partition specs for one size don't
        # fight over a single size-global count.
        have = _serve_jobs_in_flight(t.size, t.partition, run_root=run_root)
        if have >= t.count:
            continue
        need = t.count - have
        # Avoid collisions with live replicas AND any IDs already handed out this tick for
        # this size (covers pending servers on other partitions not yet in the registry).
        used = (
            _used_replica_ids(_live_dead(run_root, t.size)[0])
            | _replica_ids_in_flight(t.size, run_root=run_root)
            | assigned_rids.get(t.size, set())
        )
        rid = 0
        for _ in range(need):
            while rid in used:
                rid += 1
            used.add(rid)
            assigned_rids.setdefault(t.size, set()).add(rid)
            try:
                job = submit_server(t.size, run_root, t.partition, t.gpu_type, t.time_limit, replica=rid)
                print(f"[keepalive] RELAUNCH {t.size} r{rid} on {t.partition} ({t.gpu_type}) -> job {job} "
                      f"(have {have}/{t.count} on {t.partition})")
            except Exception as exc:  # noqa: BLE001 — never die on one failure
                print(f"[keepalive] FAILED {t.size} r{rid} on {t.partition}: {exc!r}")
            rid += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Maintain desired vLLM replica counts per size.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", default=None, help="defaults to $ASYS_RESULTS_ROOT/<run-id>")
    ap.add_argument(
        "--spec",
        help="comma-list of profile:count:partition:time, e.g. "
        "0.6B:1:pi_tpoggio:7-00:00:00,32B-long:2:pi_tpoggio:7-00:00:00",
    )
    ap.add_argument(
        "--repair-profile",
        action="append",
        default=[],
        help=(
            "atomically register/upgrade verified live endpoints for this profile and "
            "exit without submitting servers; repeat for multiple profiles"
        ),
    )
    ap.add_argument("--gpu-type", default="a100")
    ap.add_argument("--release-id", default=os.environ.get("ASYS_RELEASE_ID"))
    ap.add_argument(
        "--release-worktree", default=os.environ.get("ASYS_RELEASE_WORKTREE")
    )
    ap.add_argument("--model-contract", default=None)
    ap.add_argument(
        "--model-contract-sha256",
        default=os.environ.get("ASYS_MODEL_CONTRACT_SHA256"),
    )
    ap.add_argument(
        "--fleet-contract", default=os.environ.get("ASYS_FLEET_CONTRACT")
    )
    ap.add_argument(
        "--fleet-contract-sha256",
        default=os.environ.get("ASYS_FLEET_CONTRACT_SHA256"),
    )
    ap.add_argument(
        "--protected-capacity-marker",
        default=os.environ.get("ASYS_PROTECTED_CAPACITY_MARKER"),
    )
    ap.add_argument(
        "--protected-capacity-marker-sha256",
        default=os.environ.get("ASYS_PROTECTED_CAPACITY_MARKER_SHA256"),
    )
    ap.add_argument(
        "--protected-capacity-marker-id",
        default=os.environ.get("ASYS_PROTECTED_CAPACITY_MARKER_ID"),
    )
    ap.add_argument(
        "--control-state-dir",
        default=os.environ.get("ASYS_CONTROL_STATE_DIR"),
        help=(
            "shared schema-5 control/dispatcher state used to reconcile exact "
            "scientific occupancy before every fleet sbatch"
        ),
    )
    ap.add_argument(
        "--harness-environment-prefix",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_PREFIX"),
    )
    ap.add_argument(
        "--serving-environment-prefix",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_PREFIX"),
    )
    ap.add_argument(
        "--harness-environment-manifest",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_MANIFEST"),
    )
    ap.add_argument(
        "--serving-environment-manifest",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_MANIFEST"),
    )
    ap.add_argument(
        "--harness-environment-sha256",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_SHA256"),
    )
    ap.add_argument(
        "--serving-environment-sha256",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_SHA256"),
    )
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--interval", type=float, default=600.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="single pass then exit (for testing)")
    ap.add_argument(
        "--allow-legacy-fleet",
        action="store_true",
        help=(
            "explicit forensic/emergency override for --spec and --repair-profile; "
            "schema-5 production must use the immutable --fleet-contract path"
        ),
    )
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(
        os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT), args.run_id
    )
    if args.fleet_contract is None and not args.allow_legacy_fleet:
        ap.error(
            "legacy keepalive/registry repair is retired: schema-5 production requires "
            "the immutable --fleet-contract command; pass --allow-legacy-fleet only "
            "for an audited forensic or emergency action"
        )
    if args.repair_profile:
        if args.spec is not None or args.fleet_contract is not None:
            ap.error("--repair-profile cannot be combined with --spec/--fleet-contract")
        unknown = sorted(set(args.repair_profile) - set(SERVING_PROFILES))
        if unknown:
            ap.error(
                f"unknown serving profiles: {unknown} (known: {sorted(SERVING_PROFILES)})"
            )
        repaired = 0
        for profile_name in dict.fromkeys(args.repair_profile):
            count = _reregister_running(run_root, profile_name)
            repaired += count
            print(f"[keepalive] repair-only {profile_name}: {count} registry records updated")
        print(f"[keepalive] repair-only complete: {repaired} registry records updated")
        return
    if args.fleet_contract is not None:
        if args.spec is not None:
            ap.error("production --fleet-contract cannot be combined with legacy --spec")
        required = {
            "release_worktree": args.release_worktree,
            "release_id": args.release_id,
            "model_contract": args.model_contract,
            "model_contract_sha256": args.model_contract_sha256,
            "fleet_contract_sha256": args.fleet_contract_sha256,
            "protected_capacity_marker": args.protected_capacity_marker,
            "protected_capacity_marker_sha256": (
                args.protected_capacity_marker_sha256
            ),
            "protected_capacity_marker_id": args.protected_capacity_marker_id,
            "control_state_dir": args.control_state_dir,
            "release_git_commit": os.environ.get("ASYS_RELEASE_GIT_COMMIT"),
            "release_fleet_contract_sha256": os.environ.get(
                "ASYS_RELEASE_FLEET_CONTRACT_SHA256"
            ),
            "transport_censor_protocol_version": os.environ.get(
                "ASYS_TRANSPORT_CENSOR_PROTOCOL_VERSION"
            ),
            "transport_censor_protocol_hash": os.environ.get(
                "ASYS_TRANSPORT_CENSOR_PROTOCOL_HASH"
            ),
            "transport_uncertainty_binding_sha256": os.environ.get(
                "ASYS_TRANSPORT_UNCERTAINTY_BINDING_SHA256"
            ),
            "capacity_generation": os.environ.get(
                "ASYS_CAPACITY_GENERATION"
            ),
            "rollout_generation": os.environ.get(
                "ASYS_ROLLOUT_GENERATION"
            ),
            "harness_environment_prefix": args.harness_environment_prefix,
            "serving_environment_prefix": args.serving_environment_prefix,
            "harness_environment_manifest": args.harness_environment_manifest,
            "serving_environment_manifest": args.serving_environment_manifest,
            "harness_environment_sha256": args.harness_environment_sha256,
            "serving_environment_sha256": args.serving_environment_sha256,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            ap.error("production fleet is missing immutable pins: " + ", ".join(missing))
        transport_binding = (
            scheduler_safety.expected_transport_uncertainty_binding()
        )
        transport_binding_sha256 = (
            scheduler_safety.transport_uncertainty_binding_sha256(
                transport_binding
            )
        )
        if (
            str(required["transport_censor_protocol_version"])
            != str(
                transport_binding["transport_censor_protocol_version"]
            )
            or required["transport_censor_protocol_hash"]
            != transport_binding["transport_censor_protocol_hash"]
            or required["transport_uncertainty_binding_sha256"]
            != transport_binding_sha256
        ):
            ap.error(
                "production fleet transport/checkpoint identity differs from "
                "the frozen release"
            )
        expected_scheduler_policy_contract_id = os.environ.get(
            "ASYS_SCHEDULER_POLICY_CONTRACT_ID"
        )
        if (
            expected_scheduler_policy_contract_id is None
            and not args.once
        ):
            ap.error(
                "continuous production fleet supervision requires the "
                "attested scheduler policy contract"
            )
        if (
            expected_scheduler_policy_contract_id is not None
            and re.fullmatch(
                r"[0-9a-f]{64}",
                expected_scheduler_policy_contract_id,
            )
            is None
        ):
            ap.error(
                "attested scheduler policy contract ID must be SHA-256"
            )
        model_contracts = load_model_contracts(
            args.model_contract,
            expected_sha256=args.model_contract_sha256,
        )
        fleet = load_fleet_contract(
            args.fleet_contract,
            model_contracts=model_contracts,
            expected_sha256=args.fleet_contract_sha256,
            allow_capacity_layout=True,
        )
        # The control plane owns the generation-scoped annotated-tag and
        # frozen-source authority.  Loading the marker directly here would let a
        # syntactically valid self-hashed marker authenticate on commit alone.
        try:
            import sys

            release_root = Path(__file__).resolve().parent.parent
            if str(release_root) not in sys.path:
                sys.path.insert(0, str(release_root))
            from slurm import schema5_control as control_plane
        except (ImportError, OSError) as exc:
            ap.error(
                "production fleet cannot import the frozen control authority: "
                + str(exc)
            )

        try:
            supplied_binding = {
                "path": str(
                    Path(str(args.protected_capacity_marker))
                    .expanduser()
                    .resolve()
                ),
                "sha256": str(args.protected_capacity_marker_sha256),
                "marker_id": str(args.protected_capacity_marker_id),
            }
            capacity_generation = _capacity_generation(
                {
                    "capacity_generation": required[
                        "capacity_generation"
                    ]
                }
            )
            rollout_generation = _rollout_generation(
                {
                    "rollout_generation": required[
                        "rollout_generation"
                    ]
                }
            )
            _control_state, protected_contract = (
                _load_current_fleet_authority(
                    str(required["control_state_dir"]),
                    fleet=fleet,
                    capacity_generation=capacity_generation,
                    rollout_generation=rollout_generation,
                    expected_protected_binding=supplied_binding,
                )
            )
        except (
            protected_capacity.ProtectedCapacityError,
            control_plane.ControlError,
            FleetContractError,
            OSError,
            ValueError,
        ) as exc:
            ap.error(
                "production fleet is not covered by protected capacity: "
                + str(exc)
            )
        fleet.verify_pool_root(run_root)
        launch_options = {
            "release_worktree": args.release_worktree,
            "release_id": args.release_id,
            "environment_hash": args.serving_environment_sha256,
            "model_contract_path": args.model_contract,
            "model_contract_sha256": args.model_contract_sha256,
            "harness_environment_prefix": args.harness_environment_prefix,
            "serving_environment_prefix": args.serving_environment_prefix,
            "harness_environment_manifest_path": args.harness_environment_manifest,
            "serving_environment_manifest_path": args.serving_environment_manifest,
            "harness_environment_hash": args.harness_environment_sha256,
            "fleet_contract_path": args.fleet_contract,
            "fleet_contract_sha256": args.fleet_contract_sha256,
            "release_fleet_contract_sha256": os.environ.get(
                "ASYS_RELEASE_FLEET_CONTRACT_SHA256"
            ),
            "capacity_generation": (
                capacity_generation
            ),
            "rollout_generation": rollout_generation,
            "hf_home": args.hf_home,
        }
        print(
            f"[keepalive] canonical fleet={fleet.sha256} root={Path(run_root).resolve()}"
        )
        while True:
            # Production is intentionally fail-closed.  The outer, cross-supervising
            # controller chain records/restarts a failed turn; this process never hides
            # scheduler ambiguity or immutable-provenance drift in an endless loop.
            tick_fleet(
                run_root,
                fleet,
                launch_options=launch_options,
                scheduler_safety_contract={
                    "expected_policy_contract_id": (
                        expected_scheduler_policy_contract_id
                    ),
                    "transport_uncertainty_binding_sha256": (
                        transport_binding_sha256
                    ),
                },
                protected_capacity_contract=protected_contract,
                control_state_dir=args.control_state_dir,
            )
            if args.once:
                return
            time.sleep(args.interval)
    if args.spec is None:
        ap.error("--spec is required unless --repair-profile is used")
    targets = parse_spec(args.spec, args.gpu_type)
    unknown = [t.size for t in targets if t.size not in SERVING_PROFILES]
    if unknown:
        ap.error(
            f"unknown serving profiles: {unknown} (known: {sorted(SERVING_PROFILES)})"
        )

    print(f"[keepalive] run_root={run_root}")
    for t in targets:
        print(f"[keepalive] target: {t.size} x{t.count} on {t.partition} ({t.time_limit})")
    while True:
        try:
            tick(run_root, targets)
        except Exception as exc:  # noqa: BLE001 — never let a transient error kill the loop
            print(f"[keepalive] tick error (continuing): {exc!r}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
