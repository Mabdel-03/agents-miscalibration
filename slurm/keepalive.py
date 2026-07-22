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

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving import fleet_transactions as fleet_tx
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
        proc = subprocess.run(
            ["scontrol", "write", "batch_script", str(slurm_job_id), "-"],
            capture_output=True,
            text=True,
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
    ):
        return None

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
        or "importlib.metadata.version(\"vllm\") == \"0.21.0\"" not in script
        or "export PYTHONDONTWRITEBYTECODE=1" not in script
        or "export PYTHONPATH=" not in script
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
            or not fleet_contract_path
        ):
            return None
        try:
            fleet = load_fleet_contract(
                resolved_fleet_contract,
                model_contracts=contracts,
                expected_sha256=fleet_contract_sha256,
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
            )
            destination = _size_dir(run_root, profile.registry_key) / f"{node}_{port}.json"
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
        return fleet_tx.query_scheduler().rows
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


def _expected_fleet_script(
    replica,
    run_root: str,
    launch_options: dict[str, str | int | None],
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
        **launch_options,
    )
    _validate_fleet_script_contract(
        script,
        replica=replica,
        rollout_generation=_rollout_generation(launch_options),
        fleet_sha256=str(launch_options.get("fleet_contract_sha256") or ""),
    )
    return script


def _validate_fleet_script_contract(
    script: str,
    *,
    replica,
    rollout_generation: int,
    fleet_sha256: str,
) -> None:
    exact_directives = (
        f"#SBATCH --job-name={replica.scheduler_job_name}",
        (
            f"#SBATCH --comment=asys-schema5-pool:{replica.pool_id};"
            f"profile={replica.serving_profile};replica={replica.replica_id}"
        ),
        f"#SBATCH --partition={replica.partition}",
        f"#SBATCH --gres=gpu:{replica.gpu_type}:{replica.gpus_per_replica}",
        f"#SBATCH --cpus-per-task={replica.cpus_per_task}",
        f"#SBATCH --mem={replica.memory}",
        f"#SBATCH --time={replica.time_limit}",
        "#SBATCH --no-requeue",
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
    ):
        raise FleetContractError(
            f"rendered script omits frozen identity for {replica.replica_id}"
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
) -> None:
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
        or parsed["pool"] != replica.pool_id
        or parsed["profile"] != replica.serving_profile
        or parsed["replica"] != replica.replica_id
        or parsed["fleet"] != fleet.sha256
        or not fleet_tx.command_binds_sbatch(row.command, attempt["sbatch_path"])
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
        )
        if (
            provenance is None
            or provenance.replica_id != replica.replica_id
            or provenance.replica_index != replica.replica_index
            or provenance.server_pool_id != replica.pool_id
            or provenance.fleet_contract_sha256 != fleet.sha256
        ):
            raise FleetContractError(
                f"job {row.job_id} failed immutable fleet provenance validation"
            )


HUNG_MIN_FAILURES = 3
HUNG_MIN_SPAN_SECONDS = 600.0
HUNG_CANCEL_MAX_ATTEMPTS = 5
HUNG_CANCEL_BACKOFF_SECONDS = 300.0


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


def tick_fleet(
    run_root: str,
    fleet: FrozenFleetContract,
    *,
    launch_options: dict[str, str | int | None],
    now: float | None = None,
    submission_runner=None,
    health_probe=None,
    cancellation_runner=None,
) -> None:
    """Transactionally reconcile, validate, probe, and recover every frozen replica."""

    canonical_root = fleet.verify_pool_root(run_root)
    generation = _rollout_generation(launch_options)
    timestamp = time.time() if now is None else float(now)
    try:
        with fleet_tx.transaction_lock(canonical_root) as directory:
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
            rows = _query_fleet_queue()
            expected_by_name = {
                replica.scheduler_job_name: replica for replica in fleet.replicas
            }
            production_rows = [
                row
                for row in rows
                if row.job_name.startswith("asys-s5-serve-")
                or fleet_tx.parse_intent_comment(row.comment) is not None
            ]

            rows_by_token: dict[str, list[FleetQueueRow]] = {}
            active_by_replica: dict[str, list[FleetQueueRow]] = {}
            for row in production_rows:
                parsed = fleet_tx.parse_intent_comment(row.comment)
                if parsed is None:
                    if fleet_tx.terminal_state(row.state):
                        # A retired pre-transaction schema-5 allocation may remain in
                        # accounting history for the seven-day reconciliation window.
                        # It cannot consume capacity or race admission; current/live
                        # rows without an exact intent always fail closed below.
                        continue
                    raise FleetContractError(
                        f"unmappable schema-5 fleet job {row.job_id} lacks "
                        "transactional intent provenance"
                    )
                matched = _attempt_for_token(ledgers, parsed["intent"])
                if matched is None:
                    if fleet_tx.terminal_state(row.state):
                        # Accounting can legitimately retain a sealed/foreign transaction
                        # after its pool state was archived.  Ignore only terminal unknown
                        # tokens; an active unknown token is an admission ambiguity.
                        continue
                    raise FleetContractError(
                        f"scheduler exposes unknown fleet intent {parsed['intent']} "
                        f"as job {row.job_id}"
                    )
                replica_id, _attempt, _owner_ledger = matched
                replica = expected_by_name.get(row.job_name)
                if replica is None or replica.replica_id != replica_id:
                    raise FleetContractError(
                        f"fleet intent {parsed['intent']} has unmappable scheduler name "
                        f"{row.job_name!r}"
                    )
                if parsed["replica"] != replica_id:
                    raise FleetContractError(
                        f"fleet intent {parsed['intent']} changed replica identity"
                    )
                rows_by_token.setdefault(parsed["intent"], []).append(row)
                if not fleet_tx.terminal_state(row.state):
                    active_by_replica.setdefault(replica_id, []).append(row)
            duplicate_tokens = {
                token: [row.job_id for row in token_rows]
                for token, token_rows in rows_by_token.items()
                if len({row.job_id for row in token_rows}) > 1
            }
            duplicate_replicas = {
                replica_id: [row.job_id for row in replica_rows]
                for replica_id, replica_rows in active_by_replica.items()
                if len({row.job_id for row in replica_rows}) > 1
            }
            if duplicate_tokens or duplicate_replicas:
                raise FleetContractError(
                    "ambiguous duplicate fleet jobs: "
                    f"tokens={duplicate_tokens}, replicas={duplicate_replicas}"
                )

            active_rows: dict[str, tuple[FleetQueueRow, dict]] = {}
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
                            )
                            _validate_scheduler_row(
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
                            attempt["last_seen_at"] = timestamp
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
                                active_rows[replica.replica_id] = (row, owner_ledger)
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
                        elif state in {"submitting", "submitted"} and age >= grace:
                            # Old-generation scripts are never resubmitted after a
                            # pause/resume.  Current generation retries the same token and
                            # path after complete joined-scheduler absence.
                            attempt["state"] = (
                                "submission_failed"
                                if owner_generation == generation
                                else "terminal"
                            )
                            attempt["terminal_at"] = (
                                None if owner_generation == generation else timestamp
                            )
                            attempt["last_error"] = (
                                "absent from complete squeue+sacct truth after "
                                "visibility grace"
                            )
                            changed = True
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

                current = [
                    item
                    for _owner, owner_record in generation_records
                    for item in owner_record["attempts"]
                    if item["state"] in {
                        "prepared",
                        "submitting",
                        "submitted",
                        "committed",
                        "missing",
                    }
                ]
                failed = [
                    item
                    for item in record["attempts"]
                    if item["state"] == "submission_failed"
                ]
                if len(current) > 1:
                    raise FleetContractError(
                        f"multiple current intents for {replica.replica_id}"
                    )
                if not current and failed:
                    retry = failed[-1]
                    retry_basis = retry["submit_started_at"] or retry["created_at"]
                    if timestamp - float(retry_basis) >= grace:
                        job_id = fleet_tx.submit_attempt(
                            directory,
                            ledger,
                            replica_id=replica.replica_id,
                            attempt=retry,
                            now=timestamp,
                            runner=submission_runner,
                        )
                        print(
                            f"[keepalive] RETRY {replica.replica_id} intent "
                            f"{retry['intent_token']} -> job {job_id}"
                        )
                    continue
                if not current:
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
                    )
                    job_id = fleet_tx.submit_attempt(
                        directory,
                        ledger,
                        replica_id=replica.replica_id,
                        attempt=attempt,
                        now=timestamp,
                        runner=submission_runner,
                    )
                    print(
                        f"[keepalive] LAUNCH {replica.replica_id} on "
                        f"{replica.partition} -> job {job_id}"
                    )
                elif current[0]["state"] == "prepared":
                    job_id = fleet_tx.submit_attempt(
                        directory,
                        ledger,
                        replica_id=replica.replica_id,
                        attempt=current[0],
                        now=timestamp,
                        runner=submission_runner,
                    )
                    print(
                        f"[keepalive] RECOVER pre-sbatch {replica.replica_id} "
                        f"-> job {job_id}"
                    )

            # Registration repair and hung-allocation fencing run only after every
            # scheduler row and transaction has validated.  A scheduler query or
            # duplicate ambiguity therefore cannot accidentally trigger scancel.
            from agents_scaling.serving.launch_server import _port_for

            for replica in fleet.replicas:
                active = active_rows.get(replica.replica_id)
                if active is None:
                    continue
                row, owner_ledger = active
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
                    # A different job identity will also reset this record when it first
                    # runs; clearing it while pending makes the reset explicit.
                    health = owner_ledger["replicas"][replica.replica_id]["health"]
                    if isinstance(health, dict) and health.get("job_id") != row.job_id:
                        owner_ledger["replicas"][replica.replica_id]["health"] = None
                        fleet_tx.save_ledger(
                            directory, owner_ledger, now=timestamp
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
        model_contracts = load_model_contracts(
            args.model_contract,
            expected_sha256=args.model_contract_sha256,
        )
        fleet = load_fleet_contract(
            args.fleet_contract,
            model_contracts=model_contracts,
            expected_sha256=args.fleet_contract_sha256,
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
            "hf_home": args.hf_home,
        }
        print(
            f"[keepalive] canonical fleet={fleet.sha256} root={Path(run_root).resolve()}"
        )
        while True:
            # Production is intentionally fail-closed.  The outer, cross-supervising
            # controller chain records/restarts a failed turn; this process never hides
            # scheduler ambiguity or immutable-provenance drift in an endless loop.
            tick_fleet(run_root, fleet, launch_options=launch_options)
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
