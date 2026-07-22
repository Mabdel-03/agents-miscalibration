"""Run one ExperimentCell: build agents -> run the topology over N questions -> log JSONL.

The runner is the single place that wires the three axes together for a cell:
  * Axis 1 (capacity): the model served at this cell's ``model_size`` (via the registry).
  * Axis 2 (context): ``cell.context_share_level`` passed to the topology.
  * Axis 3 (prompt): ``cell.prompt_complexity_level`` selects the system prompt.
It writes ``results.jsonl`` (one row per question) and ``meta.json`` (cell config +
prompt token count + quality scores + git commit) into the cell directory.
"""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Iterator

from agents_scaling.agents.base_agent import Agent, SelfConsistencySample
from agents_scaling.agents.topologies import build_topology
from agents_scaling.benchmarks.contracts import (
    FrozenBenchmarkContracts,
    canonical_question_sha256,
    load_frozen_benchmark_contracts,
    load_verified_questions,
)
from agents_scaling.benchmarks.grading import grade
from agents_scaling.calibration.semantic_entropy import semantic_entropy_conf
from agents_scaling.calibration.signals import majority_answer, self_consistency_conf
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.artifact_policy import (
    ArtifactPolicy,
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.completion import (
    CellLockUnavailable,
    CompletionState,
    ExperimentConfigurationError,
    canonicalize_results,
    cell_lock,
    clear_failure,
    get_completion_status,
    quarantine_failure,
    quarantine_metadata,
    reasoning_token_summary,
    summarize_endpoint_generation,
    record_failure,
    summarize_serving_provenance,
    validate_completion_payload,
)
from agents_scaling.experiment.result_schema import (
    SELF_CONSISTENCY_PROTOCOL_HASH,
    SELF_CONSISTENCY_PROTOCOL_VERSION,
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
    CellMeta,
    QuestionResult,
)
from agents_scaling.experiment.qid_checkpoint import (
    CheckpointError,
    CheckpointingAgent,
    CoordinateAdmissionClosed,
    QIDCheckpoint,
    remove_qid_checkpoint,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest
from agents_scaling.models import get_model
from agents_scaling.prompts.prompt_quality import score_prompt_quality
from agents_scaling.prompts.system_prompts import get_prompt, token_count
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    GenerationProtocolCensorError,
    GenerationTruncationError,
    LogprobClient,
)
from agents_scaling.serving.context import tokenizer_for_profile
from agents_scaling.serving.profiles import (
    ServingProfile,
    serving_metadata,
    serving_profile_for_cell,
)
from agents_scaling.serving.registry import (
    ServerEntry,
    entry_matches_frozen_provenance,
    endpoint_instance_id,
    list_live_servers,
    server_pool_id,
    server_pool_generation,
    wait_for_server,
)
from agents_scaling.serving.model_contracts import (
    FrozenModelContracts,
    ModelContractError,
    load_model_contracts,
)
from agents_scaling.serving.fleet_contract import (
    FleetContractError,
    FrozenFleetContract,
    load_fleet_contract,
)


@dataclass(frozen=True)
class Schema5RuntimeProvenance:
    """Dispatcher-pinned identity supplied to an authoritative schema-5 worker."""

    artifact_policy_sha256: str
    release_id: str
    environment_hash: str
    serving_environment_hash: str
    model_revision: str
    tokenizer_revision: str
    model_contract_sha256: str
    fleet_contract_sha256: str
    rollout_generation: int
    release_worktree: str
    model_contract_path: str
    fleet_contract_path: str
    prompt_root: str


@dataclass(frozen=True)
class _ResolvedProductionProvenance:
    policy: ArtifactPolicy
    model_contracts: FrozenModelContracts
    fleet: FrozenFleetContract
    runtime: Schema5RuntimeProvenance
    tokenizer_id: str


@dataclass
class WorkerDrainController:
    """Signal-safe, coordinate-boundary admission control for one cell worker.

    Slurm sends ``USR1`` twenty minutes before the allocation limit.  The signal handler
    only sets a :class:`threading.Event`, which is safe while the main thread is blocked
    in an HTTP request or waiting for a topology thread pool.  Every missing stochastic
    coordinate checks the event immediately before its producer starts.  A producer that
    was already admitted may use the client's bounded 600-second timeout and then fsync
    its exact observation through :class:`QIDCheckpoint`.
    """

    requested: threading.Event = field(default_factory=threading.Event)
    signal_number: int | None = None

    def handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        self.signal_number = int(signum)
        self.requested.set()

    def admit_coordinate(self) -> None:
        if self.requested.is_set():
            signal_label = (
                "USR1"
                if self.signal_number == getattr(signal, "SIGUSR1", None)
                else str(self.signal_number or "operator")
            )
            raise CoordinateAdmissionClosed(
                f"graceful drain requested by {signal_label}; no new coordinate admitted"
            )


@contextmanager
def worker_drain_signals() -> Iterator[WorkerDrainController]:
    """Install and restore the worker's ``USR1`` handler around one cell execution."""

    controller = WorkerDrainController()
    usr1 = getattr(signal, "SIGUSR1", None)
    previous: Any = None
    installed = bool(
        usr1 is not None and threading.current_thread() is threading.main_thread()
    )
    if installed:
        previous = signal.getsignal(usr1)
        signal.signal(usr1, controller.handle_signal)
    try:
        yield controller
    finally:
        if installed:
            signal.signal(usr1, previous)


def _resolve_production_provenance(
    *,
    run_root: Path,
    manifest_snapshot: ManifestSnapshot,
    benchmark_contracts: FrozenBenchmarkContracts,
    cell: ExperimentCell,
    model_hf_id: str,
    runtime: Schema5RuntimeProvenance | None,
) -> _ResolvedProductionProvenance | None:
    """Resolve policy/model pins before any cell artifact or endpoint is touched."""

    try:
        policy = load_artifact_policy(
            run_root,
            expected_file_sha256=(
                None if runtime is None else runtime.artifact_policy_sha256
            ),
        )
    except ArtifactPolicyError as exc:
        raise ExperimentConfigurationError(
            f"schema-5 artifact policy failed closed: {exc}"
        ) from exc
    if policy is None:
        if runtime is not None:
            raise ExperimentConfigurationError(
                "dispatcher supplied schema-5 production provenance to a run without "
                "an authoritative artifact policy"
            )
        return None
    if runtime is None:
        raise ExperimentConfigurationError(
            "authoritative schema-5 run requires dispatcher-pinned runtime provenance"
        )
    if policy.accepted_manifest_sha256 != manifest_snapshot.sha256:
        raise ExperimentConfigurationError(
            "artifact policy manifest pin does not match frozen cells.json"
        )
    if (
        policy.accepted_benchmark_contracts_sha256
        != benchmark_contracts.sidecar_sha256
    ):
        raise ExperimentConfigurationError(
            "artifact policy benchmark-contract pin does not match the frozen sidecar"
        )
    if not isinstance(runtime.rollout_generation, int) or isinstance(
        runtime.rollout_generation, bool
    ) or runtime.rollout_generation < 1:
        raise ExperimentConfigurationError("rollout_generation must be a positive integer")
    exact_runtime = {
        "artifact_policy_sha256": policy.file_sha256,
        "release_id": policy.release.release_id,
        "environment_hash": policy.environment.harness_sha256,
        "serving_environment_hash": policy.environment.serving_sha256,
        "model_contract_sha256": policy.accepted_model_contract_sha256,
    }
    for field_name, expected in exact_runtime.items():
        if getattr(runtime, field_name) != expected:
            raise ExperimentConfigurationError(
                f"dispatcher {field_name} does not match the immutable artifact policy"
            )
    raw_release_worktree = Path(runtime.release_worktree).expanduser()
    if not raw_release_worktree.is_absolute():
        raise ExperimentConfigurationError(
            "schema-5 release worktree must be an absolute path"
        )
    release_worktree = raw_release_worktree.resolve()
    expected_resources = {
        "model_contract_path": release_worktree
        / "configs"
        / "model_contracts.v1.json",
        "fleet_contract_path": release_worktree
        / "configs"
        / "schema5_fleet.v1.json",
        "prompt_root": release_worktree / "configs" / "prompts",
    }
    for field_name, expected_path in expected_resources.items():
        observed_path = Path(getattr(runtime, field_name)).expanduser()
        if not observed_path.is_absolute() or observed_path.resolve() != expected_path:
            raise ExperimentConfigurationError(
                f"dispatcher {field_name} is outside the immutable release layout"
            )
    try:
        contracts = load_model_contracts(
            runtime.model_contract_path,
            expected_sha256=policy.accepted_model_contract_sha256
        )
        identity = contracts.verify_identity(
            size=cell.model_size,
            hf_id=model_hf_id,
            model_revision=runtime.model_revision,
            tokenizer_id=contracts.for_size(cell.model_size).tokenizer_id,
            tokenizer_revision=runtime.tokenizer_revision,
        )
        fleet = load_fleet_contract(
            runtime.fleet_contract_path,
            model_contracts=contracts,
            expected_sha256=runtime.fleet_contract_sha256,
        )
        if fleet.release_id != policy.release.release_id:
            raise FleetContractError(
                "fleet release ID does not match the authoritative run policy"
            )
    except (ModelContractError, FleetContractError) as exc:
        raise ExperimentConfigurationError(
            f"frozen model/tokenizer/fleet contract failed closed: {exc}"
        ) from exc
    return _ResolvedProductionProvenance(
        policy=policy,
        model_contracts=contracts,
        fleet=fleet,
        runtime=runtime,
        tokenizer_id=identity.tokenizer_id,
    )


def _entry_matches_production(
    entry: ServerEntry,
    profile_name: str,
    production: _ResolvedProductionProvenance | None,
    server_root: Path | str | None = None,
) -> bool:
    if production is None:
        return True
    runtime = production.runtime
    if server_root is None:
        return False
    try:
        production.fleet.verify_pool_root(server_root)
        if not isinstance(entry.replica_index, int) or isinstance(
            entry.replica_index, bool
        ):
            return False
        replica = production.fleet.for_replica(profile_name, entry.replica_index)
        from agents_scaling.serving.launch_server import _port_for

        exact_replica = bool(
            entry.server_pool_id == replica.pool_id
            and entry.replica_id == replica.replica_id
            and entry.port == _port_for(profile_name, replica.replica_index)
        )
    except (FleetContractError, KeyError, TypeError, ValueError):
        return False
    return exact_replica and entry_matches_frozen_provenance(
        entry,
        profile_name,
        release_id=runtime.release_id,
        environment_hash=runtime.serving_environment_hash,
        model_revision=runtime.model_revision,
        tokenizer_id=production.tokenizer_id,
        tokenizer_revision=runtime.tokenizer_revision,
        model_contract_sha256=runtime.model_contract_sha256,
        fleet_contract_sha256=runtime.fleet_contract_sha256,
        expected_server_pool_id=server_pool_id(server_root),
    )


def _build_agents(
    cell: ExperimentCell,
    base_url: str,
    served_model: str,
    serving_profile: ServingProfile,
    context_tokenizer: Any | None = None,
    endpoint_generation: str | None = None,
    system_prompt: str | None = None,
) -> list[Agent]:
    # Initialize once on the runner's main thread before any topology thread pool starts.
    # Every client shares this exact tokenizer for preflight and reasoning-token fallback.
    if context_tokenizer is None:
        context_tokenizer = tokenizer_for_profile(serving_profile.name)
    resolved_system_prompt = (
        get_prompt(cell.prompt_complexity_level)
        if system_prompt is None
        else system_prompt
    )
    n = 1 if cell.topology.value == "single_agent" else cell.n_agents
    agents = []
    for i in range(n):
        client = LogprobClient(
            base_url=base_url,
            model=served_model,
            serving_profile=serving_profile,
            context_tokenizer=context_tokenizer,
            endpoint_generation=endpoint_generation,
        )
        agents.append(
            Agent(
                f"agent{i}",
                client,
                resolved_system_prompt,
                temperature=cell.temperature,
                reasoning_level=cell.reasoning_level,  # Axis 4
            )
        )
    return agents


def _current_server_pool_generation(
    server_root: Path,
    registry_key: str,
    production: _ResolvedProductionProvenance | None = None,
) -> str:
    """Snapshot the current live process generation after the cell lock is held."""

    entries = [
        entry
        for entry in list_live_servers(
            server_root,
            registry_key,
            require_current_provenance=True,
        )
        if _entry_matches_production(
            entry, registry_key, production, server_root=server_root
        )
    ]
    return server_pool_generation(entries, profile_name=registry_key)


# Exception classes the runner treats as "this endpoint is unusable, try another".
# Imported lazily because ``openai`` is only available when the harness env is installed.
def _connection_errors() -> tuple[type, ...]:
    from openai import APIConnectionError, APITimeoutError
    return (APIConnectionError, APITimeoutError)


def _pick_endpoint(
    run_root,
    registry_key: str,
    shard: int,
    exclude: set[tuple[str, int]],
    production: _ResolvedProductionProvenance | None = None,
) -> ServerEntry | None:
    """Return a live endpoint for ``registry_key`` not in ``exclude``, shard-aware.

    Calls the non-destructive, profile-validating ``list_live_servers``, filters out
    endpoints already tried by this question, and round-robins by ``shard`` so cells still
    spread across the fleet.
    """
    candidates = [
        entry
        for entry in list_live_servers(
            run_root,
            registry_key,
            require_current_provenance=True,
        )
        if (entry.host, entry.port) not in exclude
        and _entry_matches_production(
            entry, registry_key, production, server_root=run_root
        )
    ]
    if not candidates:
        return None
    return candidates[shard % len(candidates)]


def _wait_for_production_server(
    run_root: Path,
    registry_key: str,
    *,
    shard: int,
    production: _ResolvedProductionProvenance | None,
    timeout_s: float = 300.0,
    poll_s: float = 5.0,
) -> ServerEntry:
    """Wait for a fully pinned endpoint, never adopting a stale release registration."""

    if production is None:
        return wait_for_server(
            run_root,
            registry_key,
            shard=shard,
            require_current_provenance=True,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entry = _pick_endpoint(
            run_root,
            registry_key,
            shard,
            exclude=set(),
            production=production,
        )
        if entry is not None:
            return entry
        time.sleep(poll_s)
    raise TimeoutError(
        f"no endpoint matching frozen schema-5 provenance for {registry_key!r} "
        f"within {timeout_s}s"
    )


def _length_censored_record(
    *,
    cell: ExperimentCell,
    question: Any,
    error: GenerationTruncationError,
    wall_ms: float,
    profile_meta: dict[str, Any],
    benchmark_contract_sha256: str,
    observed_topology_coordinates: list[dict[str, Any]] | None = None,
    production: _ResolvedProductionProvenance | None = None,
) -> dict[str, Any]:
    """Materialize one non-resampled observed censor as a valid schema-5 QID row."""

    censored = error.to_censored_generation()
    # These values come from the frozen manifest/runner rather than exception text.  A
    # mismatch is retained and rejected by completion.py instead of being normalized.
    censored["topology"] = cell.topology.value
    censored["benchmark"] = cell.benchmark
    status = (
        TERMINATION_PROTOCOL_CENSORED
        if isinstance(error, GenerationProtocolCensorError)
        else TERMINATION_LENGTH_CENSORED
    )
    coordinates = list(observed_topology_coordinates or [])
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    rounds: set[int] = set()
    for entry in coordinates:
        request = entry.get("request") if isinstance(entry, dict) else None
        outcome = entry.get("outcome") if isinstance(entry, dict) else None
        if isinstance(request, dict) and isinstance(request.get("round"), int):
            rounds.add(int(request["round"]))
        if not isinstance(outcome, dict):
            continue
        output = outcome.get("agent_output")
        coordinate_censor = outcome.get("censored_generation")
        if isinstance(output, dict):
            prompt_tokens += int(output.get("prompt_tokens", 0))
            completion_tokens += int(output.get("completion_tokens", 0))
            reasoning_tokens += int(output.get("reasoning_tokens", 0))
        elif isinstance(coordinate_censor, dict):
            prompt_tokens += int(coordinate_censor.get("prompt_tokens", 0))
            completion_tokens += int(coordinate_censor.get("completion_tokens", 0))
            reasoning_tokens += _censored_reasoning_tokens(
                coordinate_censor,
                thinking_enabled=cell.reasoning_level.enable_thinking,
            )
    # Direct unit-level construction predates coordinate snapshots.  Production always
    # supplies the checkpoint snapshot; retain truthful one-response cost in this narrow
    # compatibility path while semantic schema-5 validation still requires the snapshot.
    if not coordinates:
        prompt_tokens = int(censored["prompt_tokens"])
        completion_tokens = int(censored["completion_tokens"])
        reasoning_tokens = _censored_reasoning_tokens(
            censored,
            thinking_enabled=cell.reasoning_level.enable_thinking,
        )
    record = QuestionResult(
        cell_id=cell.cell_id,
        qid=question.qid,
        benchmark=question.benchmark,
        question_sha256=canonical_question_sha256(question),
        benchmark_contract_sha256=benchmark_contract_sha256,
        model_size=cell.model_size,
        topology=cell.topology.value,
        context_share_level=cell.context_share_level.value,
        prompt_complexity_level=cell.prompt_complexity_level,
        reasoning_level=cell.reasoning_level.value,
        final_answer=None,
        answer_key=question.answer_key,
        correct=False,
        per_agent=[],
        system_conf={},
        self_consistency={},
        efficiency_raw={
            "n_turns": len(coordinates),
            "n_messages": _observed_topology_message_count(cell, coordinates),
            "n_rounds": len(rounds),
            "n_agents": (
                1 if cell.topology.value == "single_agent" else cell.n_agents
            ),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_reasoning_tokens": reasoning_tokens,
            "wall_ms": wall_ms,
        },
        timestamp=time.time(),
        serving_profile=str(profile_meta["serving_profile"]),
        effective_context_limit=int(profile_meta["effective_context_limit"]),
        tensor_parallel_size=int(profile_meta["tensor_parallel_size"]),
        termination_status=status,
        censored_generation=censored,
        observed_topology_coordinates=coordinates,
    ).to_dict()
    if production is not None:
        runtime = production.runtime
        record.update(
            release_id=runtime.release_id,
            environment_hash=runtime.environment_hash,
            model_revision=runtime.model_revision,
            tokenizer_revision=runtime.tokenizer_revision,
            model_contract_sha256=runtime.model_contract_sha256,
            effective_context=int(profile_meta["effective_context_limit"]),
            rollout_generation=runtime.rollout_generation,
        )
        record["endpoint_generation"] = summarize_endpoint_generation([record])
    return record


def _censored_reasoning_tokens(
    censored: dict[str, Any], *, thinking_enabled: bool
) -> int:
    """Return exact generated reasoning span cost for a retained length censor."""

    if not thinking_enabled:
        return 0
    completion_ids = censored.get("completion_token_ids")
    starts = censored.get("completion_think_start_positions")
    ends = censored.get("completion_think_end_positions")
    if not isinstance(completion_ids, list) or not completion_ids:
        return 0
    if not isinstance(starts, list) or len(starts) != 1:
        return 0
    start = starts[0]
    if not isinstance(start, int) or start < 0 or start >= len(completion_ids):
        return 0
    valid_ends = sorted(
        end
        for end in (ends if isinstance(ends, list) else [])
        if isinstance(end, int) and start < end <= len(completion_ids)
    )
    if valid_ends:
        boundary = valid_ends[0]
    else:
        # Stop-terminated protocol censors retain their terminal EOS token.  Exclude
        # it from the open reasoning span; a length finish has no terminal EOS and
        # therefore retains the whole remaining span as generated reasoning.
        boundary = (
            max(start + 1, len(completion_ids) - 1)
            if censored.get("finish_reason") == "stop"
            else len(completion_ids)
        )
    return max(0, boundary - start - 1)


def _observed_topology_message_count(
    cell: ExperimentCell, coordinates: list[dict[str, Any]]
) -> int:
    """Count peer-output messages actually consumed by observed coordinates."""

    total = 0
    for entry in coordinates:
        request = entry.get("request") if isinstance(entry, dict) else None
        if not isinstance(request, dict):
            continue
        agent_id = request.get("agent_id")
        round_index = request.get("round")
        if (
            not isinstance(agent_id, str)
            or not agent_id.startswith("agent")
            or not isinstance(round_index, int)
            or isinstance(round_index, bool)
        ):
            continue
        try:
            agent_index = int(agent_id.removeprefix("agent"))
        except ValueError:
            continue
        topology = cell.topology.value
        if topology in {"single_agent", "independent"} or round_index == 0:
            continue
        if topology == "decentralized":
            total += max(0, cell.n_agents - 1)
        elif topology == "centralized":
            total += max(0, cell.n_agents - 1) if agent_index == 0 else 1
    return total


def _self_consistency_payload(
    *,
    cell: ExperimentCell,
    outcomes: list[SelfConsistencySample],
    primary_answer: str | None,
    wall_ms: float,
) -> dict[str, Any]:
    """Build the auditable auxiliary-draw payload without conditioning on stopping."""

    completed = [
        outcome for outcome in outcomes if outcome.termination_status == TERMINATION_COMPLETED
    ]
    length_censored = [
        outcome
        for outcome in outcomes
        if outcome.termination_status == TERMINATION_LENGTH_CENSORED
    ]
    protocol_censored = [
        outcome
        for outcome in outcomes
        if outcome.termination_status == TERMINATION_PROTOCOL_CENSORED
    ]
    censored = length_censored + protocol_censored
    has_censor = bool(censored)
    if has_censor:
        majority = None
        consistency = None
        semantic_confidence = None
        semantic_entropy = None
    else:
        sampled_answers = [
            outcome.agent_output.answer_choice
            for outcome in completed
            if outcome.agent_output is not None
        ]
        majority = majority_answer(sampled_answers)
        semantic_confidence, semantic_entropy = semantic_entropy_conf(
            [answer or "" for answer in sampled_answers]
        )
        consistency = self_consistency_conf(
            sampled_answers, primary_answer or ""
        )

    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    for outcome in outcomes:
        if outcome.agent_output is not None:
            prompt_tokens += outcome.agent_output.prompt_tokens
            completion_tokens += outcome.agent_output.completion_tokens
            reasoning_tokens += outcome.agent_output.reasoning_tokens
        elif outcome.censored_generation is not None:
            prompt_tokens += int(outcome.censored_generation["prompt_tokens"])
            completion_tokens += int(outcome.censored_generation["completion_tokens"])
            reasoning_tokens += _censored_reasoning_tokens(
                outcome.censored_generation,
                thinking_enabled=cell.reasoning_level.enable_thinking,
            )

    return {
        "protocol_version": SELF_CONSISTENCY_PROTOCOL_VERSION,
        "protocol_hash": SELF_CONSISTENCY_PROTOCOL_HASH,
        "sample_count": len(outcomes),
        "completed_sample_count": len(completed),
        "length_censored_sample_count": len(length_censored),
        "protocol_censored_sample_count": len(protocol_censored),
        "samples": [outcome.to_dict() for outcome in outcomes],
        "majority": majority,
        "self_consistency_conf": consistency,
        "semantic_entropy_conf": semantic_confidence,
        "semantic_entropy": semantic_entropy,
        "auxiliary_efficiency_raw": {
            "n_samples": len(outcomes),
            "completed_samples": len(completed),
            "length_censored_samples": len(length_censored),
            "protocol_censored_samples": len(protocol_censored),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_reasoning_tokens": reasoning_tokens,
            "wall_ms": wall_ms,
        },
    }


def _publish_qid_record(
    *,
    cell: ExperimentCell,
    cdir: Path,
    results_path: Path,
    record: dict[str, Any],
    expected_qids: tuple[str, ...],
    expected_questions: list[Any],
    verified_benchmark_contracts: FrozenBenchmarkContracts,
    verified_manifest: ManifestSnapshot,
    checkpoint: QIDCheckpoint,
) -> None:
    """Fsync, semantically admit, then delete one QID's stochastic journal."""

    io.append_jsonl(results_path, record)
    admitted = canonicalize_results(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=expected_questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=verified_manifest,
        write=False,
    )
    exact_matches = [
        candidate
        for candidate in admitted.records
        if candidate.get("qid") == record.get("qid")
    ]
    if exact_matches != [record]:
        # Keep the checkpoint: a later repair can discard the rejected row and replay
        # the terminal/coordinates without contacting the model again.
        raise ExperimentConfigurationError(
            f"newly appended QID row {record.get('qid')!r} failed canonical semantic "
            "admission; retaining its durable checkpoint"
        )
    checkpoint.delete()
    # Reaching a semantically admitted QID proves any prior transient cell failure was
    # recovered.  Retry attempts are therefore consecutive failures, never a lifetime
    # count accumulated across unrelated questions.
    clear_failure(cdir)


def run_cell(
    cell: ExperimentCell,
    run_id: str,
    score_prompt_with_judge: bool = False,
    shard: int = 0,
    server_run_id: str | None = None,
    expected_benchmark_contracts_sha256: str | None = None,
    runtime_provenance: Schema5RuntimeProvenance | None = None,
) -> str:
    """Execute one cell end-to-end; returns the path to its results.jsonl.

    ``shard`` (typically the SLURM array task id) round-robins this cell across all
    registered server endpoints for its serving profile.  ``server_run_id`` optionally
    selects an existing serving pool while results and locks remain under ``run_id``.
    """
    # Resolve scientific identity from the frozen run manifest before creating a cell
    # directory, selecting an endpoint, or mutating any artifact.  ``cell_id`` is a
    # compact resume key and intentionally omits some execution fields (for example
    # temperature and n_samples), so matching the id alone is not sufficient.
    run_root = io.results_root() / run_id
    snapshot = load_manifest(run_root)
    manifest_cells = [
        manifest_cell
        for manifest_cell in snapshot.cells
        if manifest_cell.cell_id == cell.cell_id
    ]
    if len(manifest_cells) != 1 or manifest_cells[0] != cell:
        raise ExperimentConfigurationError(
            f"cell {cell.cell_id!r} is not the exact entry in frozen manifest "
            f"{snapshot.path}"
        )
    cell = manifest_cells[0]

    verified_benchmark_contracts = load_frozen_benchmark_contracts(
        run_root, snapshot=snapshot
    )
    if (
        expected_benchmark_contracts_sha256 is not None
        and verified_benchmark_contracts.sidecar_sha256
        != expected_benchmark_contracts_sha256
    ):
        raise ExperimentConfigurationError(
            "dispatcher-pinned benchmark contract does not match the frozen run "
            f"sidecar: expected {expected_benchmark_contracts_sha256}, got "
            f"{verified_benchmark_contracts.sidecar_sha256}"
        )

    spec = get_model(cell.model_size)
    production = _resolve_production_provenance(
        run_root=run_root,
        manifest_snapshot=snapshot,
        benchmark_contracts=verified_benchmark_contracts,
        cell=cell,
        model_hf_id=spec.hf_id,
        runtime=runtime_provenance,
    )
    cdir = io.cell_dir(run_id, cell.cell_id)
    results_path = cdir / "results.jsonl"
    profile = serving_profile_for_cell(cell)
    server_root = io.run_dir(server_run_id or run_id)
    # Installed production packages do not contain a Git worktree.  The run policy is
    # the source of truth for the exact release commit used to gate permanent failures;
    # legacy/editable runs retain the source-tree fingerprint helper.
    code_version = (
        (
            f"{production.policy.release.git_commit}+source."
            f"{production.policy.release.source_tree_sha256[:16]}"
        )
        if production is not None
        else io.git_commit()
    )

    # The lock is acquired before endpoint selection or any result mutation.  Duplicate
    # array tasks therefore turn into cheap no-ops instead of racing on one JSONL file.
    try:
        lock = cell_lock(cdir, blocking=False)
        with lock, worker_drain_signals() as drain:
            pool_generation = _current_server_pool_generation(
                server_root, profile.registry_key, production
            )
            try:
                _run_cell_locked(
                    cell=cell,
                    manifest_snapshot=snapshot,
                    verified_benchmark_contracts=verified_benchmark_contracts,
                    spec_hf_id=spec.hf_id,
                    cdir=cdir,
                    results_path=results_path,
                    server_root=server_root,
                    profile=profile,
                    score_prompt_with_judge=score_prompt_with_judge,
                    shard=shard,
                    code_version=code_version,
                    server_pool_generation=pool_generation,
                    drain=drain,
                    production=production,
                )
            except CoordinateAdmissionClosed as exc:
                # This is a successful scheduler handoff, not an experimental or
                # infrastructure failure.  Every response admitted before USR1 has
                # already passed through the atomic QID checkpoint boundary.  Leave
                # metadata absent unless the whole cell had already completed, release
                # the advisory lock, and let a later task resume only missing work.
                print(f"[run_cell] {cell.cell_id} drained cleanly: {exc}")
            except Exception as exc:
                # A corrupt/mismatched checkpoint, or a failure to persist an already
                # observed coordinate, must never enter ordinary runtime retry.  Doing
                # so could reissue a stochastic trajectory.  Pin the failure to this
                # executable code identity until a researcher explicitly repairs it.
                failure_exc: BaseException = exc
                if isinstance(exc, CheckpointError):
                    failure_exc = ExperimentConfigurationError(
                        "durable QID checkpoint failed closed; refusing replacement "
                        f"sampling: {exc}"
                    )
                # Refresh at failure time so dormancy is pinned to the fleet on which
                # the failed attempt actually ended, not merely the fleet at startup.
                pool_generation = _current_server_pool_generation(
                    server_root, profile.registry_key, production
                )
                failure = record_failure(
                    cdir,
                    cell,
                    failure_exc,
                    serving_profile=profile.name,
                    code_version=code_version,
                    server_pool_generation=pool_generation,
                )
                print(
                    f"[run_cell] {cell.cell_id} failed: {failure.classification} "
                    f"attempt={failure.attempts} disposition={failure.disposition}"
                )
                if failure_exc is not exc:
                    raise failure_exc from exc
                raise
    except CellLockUnavailable:
        print(f"[run_cell] {cell.cell_id} already active; skipping duplicate worker")
    return str(results_path)


def _run_cell_locked(
    *,
    cell: ExperimentCell,
    manifest_snapshot: ManifestSnapshot,
    verified_benchmark_contracts: FrozenBenchmarkContracts,
    spec_hf_id: str,
    cdir: Path,
    results_path: Path,
    server_root: Path,
    profile: ServingProfile,
    score_prompt_with_judge: bool,
    shard: int,
    code_version: str | None,
    server_pool_generation: str,
    drain: WorkerDrainController,
    production: _ResolvedProductionProvenance | None,
) -> None:
    """Execute one cell while the caller owns its advisory lock."""
    meta_path = cdir / "meta.json"
    run_root = cdir.parent.parent
    model_contract_path = (
        production.runtime.model_contract_path if production is not None else None
    )
    questions, benchmark_contract = load_verified_questions(run_root, cell)
    current_benchmark_contracts = load_frozen_benchmark_contracts(
        run_root, snapshot=manifest_snapshot
    )
    if (
        current_benchmark_contracts.sidecar_sha256
        != verified_benchmark_contracts.sidecar_sha256
        or verified_benchmark_contracts.contract_for_cell(cell)
        != benchmark_contract
    ):
        raise ExperimentConfigurationError(
            "benchmark contract changed while the worker was establishing identity"
        )
    questions = list(questions)
    benchmark_contract_sha256 = str(
        benchmark_contract["question_contract_sha256"]
    )
    expected_qids = tuple(question.qid for question in questions)
    if len(expected_qids) != len(set(expected_qids)):
        raise ExperimentConfigurationError("benchmark loader produced duplicate qids")

    status = get_completion_status(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        check_active=False,
        serving_profile=profile.name,
        code_version=code_version,
        server_pool_generation=server_pool_generation,
        model_contract_path=model_contract_path,
    )
    if status.status is CompletionState.COMPLETE:
        # Completion proves every expected QID row is durable and valid.  This also
        # closes the kill window between a final append and checkpoint unlink.
        for qid in expected_qids:
            remove_qid_checkpoint(cdir, qid)
        if status.failure is not None:
            clear_failure(cdir)
        print(f"[run_cell] {cell.cell_id} semantically complete; skipping")
        return
    if status.status is CompletionState.PERMANENT:
        print(f"[run_cell] {cell.cell_id} has a permanent failure; skipping")
        return
    if status.status is CompletionState.RETRYABLE and not status.eligible_for_retry:
        print(
            f"[run_cell] {cell.cell_id} is in retry backoff/dormant state "
            f"(next={status.next_eligible_at}); skipping"
        )
        return

    # Repair malformed, duplicate, unexpected, or invalid legacy rows before resuming.
    parsed = canonicalize_results(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        write=True,
    )
    repaired_status = get_completion_status(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        check_active=False,
        serving_profile=profile.name,
        code_version=code_version,
        server_pool_generation=server_pool_generation,
        model_contract_path=model_contract_path,
    )
    if repaired_status.status is CompletionState.CORRUPT:
        lowered_errors = [error.lower() for error in repaired_status.errors]
        if any("meta" in error for error in lowered_errors):
            quarantined = quarantine_metadata(cdir)
            if quarantined is not None:
                print(f"[run_cell] {cell.cell_id} quarantined invalid metadata: {quarantined}")
        if any("failure" in error for error in lowered_errors):
            quarantined = quarantine_failure(cdir)
            if quarantined is not None:
                print(f"[run_cell] {cell.cell_id} quarantined invalid failure state: {quarantined}")
        repaired_status = get_completion_status(
            cell,
            cdir,
            expected_qids=expected_qids,
            expected_questions=questions,
            verified_benchmark_contracts=verified_benchmark_contracts,
            verified_manifest=manifest_snapshot,
            check_active=False,
            serving_profile=profile.name,
            code_version=code_version,
            server_pool_generation=server_pool_generation,
            model_contract_path=model_contract_path,
        )
    if repaired_status.status is CompletionState.COMPLETE:
        for qid in expected_qids:
            remove_qid_checkpoint(cdir, qid)
        if repaired_status.failure is not None:
            clear_failure(cdir)
        print(f"[run_cell] {cell.cell_id} repaired and semantically complete; skipping inference")
        return
    if repaired_status.status is CompletionState.PERMANENT:
        print(f"[run_cell] {cell.cell_id} repaired results but has a permanent failure; skipping")
        return
    if (
        repaired_status.status is CompletionState.RETRYABLE
        and not repaired_status.eligible_for_retry
    ):
        print(f"[run_cell] {cell.cell_id} repaired results but remains in retry backoff; skipping")
        return
    if repaired_status.status is CompletionState.CORRUPT:
        raise ExperimentConfigurationError(
            "cell remains corrupt after lock-protected canonical repair: "
            + "; ".join(repaired_status.errors[:5])
        )
    records = list(parsed.records)
    done_qids = {record["qid"] for record in records}
    if done_qids:
        # A kill may land after the fsynced QID append and before checkpoint unlink.
        # The canonical row is now authoritative, so stale journals can be removed
        # without inference.  Never remove a checkpoint for a still-missing QID.
        for qid in done_qids:
            remove_qid_checkpoint(cdir, qid)
        print(f"[run_cell] {cell.cell_id} resuming: {len(done_qids)} canonical questions done")

    # Initial endpoint selection happens only after lock/status/repair, and is unnecessary
    # when repair recovered full QID coverage (unless an LLM prompt judge was requested).
    # Runtime profile identity is distinct from scientific model_size, and the server pool
    # can belong to another run (used by the seven-agent extension).
    missing_questions = [question for question in questions if question.qid not in done_qids]
    entry: ServerEntry | None = None
    base_url: str | None = None
    if missing_questions or score_prompt_with_judge:
        drain.admit_coordinate()
        initial = _pick_endpoint(
            server_root,
            profile.registry_key,
            shard,
            exclude=set(),
            production=production,
        )
        if initial is None:
            initial = _wait_for_production_server(
                server_root,
                profile.registry_key,
                shard=shard,
                production=production,
            )
        entry = initial
        base_url = entry.base_url
    served_model = profile.served_model_name

    production_prompt_root = (
        production.runtime.prompt_root if production is not None else None
    )
    system_prompt = get_prompt(
        cell.prompt_complexity_level, prompt_root=production_prompt_root
    )
    context_tokenizer = (
        (
            tokenizer_for_profile(
                profile.name,
                model_contract_path=production.runtime.model_contract_path,
                expected_model_contract_sha256=production.runtime.model_contract_sha256,
            )
            if production is not None
            else tokenizer_for_profile(profile.name)
        )
        if missing_questions or score_prompt_with_judge
        else None
    )
    judge = (
        LogprobClient(
            base_url=base_url or "",
            model=served_model,
            serving_profile=profile,
            context_tokenizer=context_tokenizer,
            endpoint_generation=(
                endpoint_instance_id(entry) if entry is not None else None
            ),
        )
        if score_prompt_with_judge
        else None
    )
    pq = score_prompt_quality(system_prompt, judge=judge)
    profile_meta = serving_metadata(profile)
    started_at = min(
        [time.time()]
        + [float(record["timestamp"]) for record in records if record.get("timestamp")]
    )
    meta = CellMeta(
        cell_id=cell.cell_id,
        config=cell.to_dict(),
        config_hash=cell.config_hash(),
        benchmark_contract_sha256=benchmark_contract_sha256,
        model_hf_id=spec_hf_id,
        served_model_name=served_model,
        prompt_token_count=token_count(
            cell.prompt_complexity_level, prompt_root=production_prompt_root
        ),
        prompt_quality={
            "heuristic": pq.heuristic,
            "llm_judge": pq.llm_judge,
            "features": pq.features,
        },
        serving_profile=str(profile_meta["serving_profile"]),
        effective_context_limit=int(profile_meta["effective_context_limit"]),
        tensor_parallel_size=int(profile_meta["tensor_parallel_size"]),
        git_commit=code_version,
        started_at=started_at,
    )
    if production is not None:
        runtime = production.runtime
        meta.git_commit = production.policy.release.git_commit
        meta.release_id = runtime.release_id
        meta.environment_hash = runtime.environment_hash
        meta.serving_environment_hash = runtime.serving_environment_hash
        meta.model_revision = runtime.model_revision
        meta.tokenizer_revision = runtime.tokenizer_revision
        meta.model_contract_sha256 = runtime.model_contract_sha256
        meta.artifact_policy_sha256 = runtime.artifact_policy_sha256
        meta.effective_context = int(profile_meta["effective_context_limit"])
        meta.rollout_generation = runtime.rollout_generation

    agents = (
        _build_agents(
            cell,
            base_url or "",
            served_model,
            profile,
            context_tokenizer=context_tokenizer,
            endpoint_generation=(
                endpoint_instance_id(entry) if entry is not None else None
            ),
            system_prompt=system_prompt,
        )
        if missing_questions
        else []
    )
    conn_errors = _connection_errors()
    max_endpoint_fails_per_q = 4

    for q in questions:
        if q.qid in done_qids:
            continue
        # Do not begin a new QID (including its deterministic calibration probes) after
        # USR1.  If the signal arrives later, the checkpoint proxy applies the same gate
        # immediately before every missing stochastic coordinate.
        drain.admit_coordinate()
        checkpoint = QIDCheckpoint(
            cdir,
            cell,
            q,
            code_version=code_version,
            serving_profile=profile,
            benchmark_contract_sha256=benchmark_contract_sha256,
        )
        tried: set[tuple[str, int]] = set()
        tr = None
        censored_error: GenerationTruncationError | None = None
        sc: dict = {}
        wall_ms = 0.0
        fails = 0
        terminal = checkpoint.topology_terminal()
        if terminal is not None:
            wall_ms = terminal.wall_ms
            tr = terminal.topology_result
            censored_error = terminal.censored_error
        else:
            # A prior process may already have committed part of this topology.  Start
            # from its durable, concurrency-aware critical-path lower bound instead of
            # reporting only the final replay attempt.
            wall_ms = checkpoint.topology_critical_path_wall_ms()
            while True:
                attempt_started = time.monotonic()
                # Proxies share one synchronized journal.  Distinct siblings can still
                # batch concurrently; a completed sibling is replayed exactly on an
                # endpoint retry or process restart.
                checkpoint_agents = [
                    CheckpointingAgent(
                        agent,
                        checkpoint,
                        agent_id=f"agent{index}",
                        admit_coordinate=drain.admit_coordinate,
                    )
                    for index, agent in enumerate(agents)
                ]
                topo = build_topology(
                    cell.topology,
                    checkpoint_agents,
                    cell.context_share_level,
                    cell.rounds,
                    max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
                    seed=cell.seed,
                    context_tokenizer=context_tokenizer,
                )
                try:
                    # Complete every participant's question-only option probe before the
                    # topology may issue its first stochastic chat.  Calibration itself
                    # is deterministic; stochastic calls below are checkpointed before
                    # control returns to the topology.
                    for agent in checkpoint_agents:
                        agent.prepare_calibration(q)
                    tr = topo.run(q)
                    wall_ms += (time.monotonic() - attempt_started) * 1000.0
                    # Persist the fully assembled result after topology peer-context
                    # audit fields and system confidence have been attached.
                    checkpoint.record_topology_result(tr, wall_ms=wall_ms)
                    break
                except GenerationTruncationError as exc:
                    # Protocol v4 has already used the full exact profile envelope in
                    # one draw.  Persist the censor before publishing the QID row.
                    wall_ms += (time.monotonic() - attempt_started) * 1000.0
                    censored_error = exc
                    checkpoint.record_topology_censor(exc, wall_ms=wall_ms)
                    break
                except conn_errors as exc:
                    # Preserve time already spent in failed endpoint attempts.  A
                    # replayed checkpoint coordinate is fast, but the earlier retained
                    # sibling work and transport wait are still part of this QID's
                    # observed execution cost.
                    wall_ms += (time.monotonic() - attempt_started) * 1000.0
                    fails += 1
                    assert entry is not None
                    tried.add((entry.host, entry.port))
                    print(
                        f"[run_cell] {cell.cell_id} q={q.qid}: endpoint "
                        f"{entry.host}:{entry.port} failed ({type(exc).__name__}); "
                        "trying the next validated endpoint"
                    )
                    nxt = _pick_endpoint(
                        server_root,
                        profile.registry_key,
                        shard,
                        exclude=tried,
                        production=production,
                    )
                    if nxt is None or fails >= max_endpoint_fails_per_q:
                        raise
                    entry = nxt
                    base_url = entry.base_url
                    agents = _build_agents(
                        cell,
                        base_url,
                        served_model,
                        profile,
                        context_tokenizer=context_tokenizer,
                        endpoint_generation=endpoint_instance_id(entry),
                        system_prompt=system_prompt,
                    )

        if censored_error is not None:
            record = _length_censored_record(
                cell=cell,
                question=q,
                error=censored_error,
                wall_ms=wall_ms,
                profile_meta=dict(profile_meta),
                benchmark_contract_sha256=benchmark_contract_sha256,
                observed_topology_coordinates=(
                    checkpoint.observed_topology_coordinates()
                ),
                production=production,
            )
            _publish_qid_record(
                cell=cell,
                cdir=cdir,
                results_path=results_path,
                record=record,
                expected_qids=expected_qids,
                expected_questions=questions,
                verified_benchmark_contracts=verified_benchmark_contracts,
                verified_manifest=manifest_snapshot,
                checkpoint=checkpoint,
            )
            records.append(record)
            done_qids.add(q.qid)
            print(
                f"[run_cell] {cell.cell_id} q={q.qid}: retained one "
                f"{record['termination_status']} outcome; no resampling"
            )
            continue

        assert tr is not None
        if cell.topology.value == "single_agent" and cell.n_samples > 1:
            # The primary terminal and each auxiliary coordinate are durable.  A kill
            # or endpoint failure therefore resumes only missing sample indices.
            outcomes: list[SelfConsistencySample] = []
            for sample_index in range(cell.n_samples):
                sample_fails = 0
                sample_tried: set[tuple[str, int]] = set()
                while True:
                    try:
                        outcomes.append(
                            CheckpointingAgent(
                                agents[0],
                                checkpoint,
                                agent_id="agent0",
                                admit_coordinate=drain.admit_coordinate,
                            ).sample_one(
                                q,
                                sample_index=sample_index,
                                base_seed=cell.seed + 1000,
                            )
                        )
                        break
                    except conn_errors as exc:
                        sample_fails += 1
                        assert entry is not None
                        sample_tried.add((entry.host, entry.port))
                        print(
                            f"[run_cell] {cell.cell_id} q={q.qid} "
                            f"aux_sample={sample_index}: endpoint "
                            f"{entry.host}:{entry.port} failed "
                            f"({type(exc).__name__}); retrying only this auxiliary "
                            "coordinate"
                        )
                        nxt = _pick_endpoint(
                            server_root,
                            profile.registry_key,
                            shard,
                            exclude=sample_tried,
                            production=production,
                        )
                        if nxt is None or sample_fails >= max_endpoint_fails_per_q:
                            raise
                        entry = nxt
                        base_url = entry.base_url
                        agents = _build_agents(
                            cell,
                            base_url,
                            served_model,
                            profile,
                            context_tokenizer=context_tokenizer,
                            endpoint_generation=endpoint_instance_id(entry),
                            system_prompt=system_prompt,
                        )
            sc = _self_consistency_payload(
                cell=cell,
                outcomes=outcomes,
                primary_answer=tr.final_answer,
                wall_ms=checkpoint.self_consistency_wall_ms(),
            )

        correct = grade(q, tr.final_answer or "") if tr.final_answer is not None else False
        total_reasoning_tokens = sum(output.reasoning_tokens for output in tr.per_agent)
        record = QuestionResult(
            cell_id=cell.cell_id,
            qid=q.qid,
            benchmark=q.benchmark,
            question_sha256=canonical_question_sha256(q),
            benchmark_contract_sha256=benchmark_contract_sha256,
            model_size=cell.model_size,
            topology=cell.topology.value,
            context_share_level=cell.context_share_level.value,
            prompt_complexity_level=cell.prompt_complexity_level,
            reasoning_level=cell.reasoning_level.value,
            final_answer=tr.final_answer,
            answer_key=q.answer_key,
            correct=correct,
            per_agent=[output.to_dict() for output in tr.per_agent],
            system_conf=tr.system_conf,
            self_consistency=sc,
            efficiency_raw={
                "n_turns": tr.n_turns,
                "n_messages": tr.n_messages,
                "n_rounds": tr.n_rounds,
                "n_agents": tr.n_agents,
                "total_prompt_tokens": tr.total_prompt_tokens,
                "total_completion_tokens": tr.total_completion_tokens,
                "total_reasoning_tokens": total_reasoning_tokens,
                "wall_ms": wall_ms,
            },
            timestamp=time.time(),
            serving_profile=str(profile_meta["serving_profile"]),
            effective_context_limit=int(profile_meta["effective_context_limit"]),
            tensor_parallel_size=int(profile_meta["tensor_parallel_size"]),
            termination_status=TERMINATION_COMPLETED,
            censored_generation=None,
            observed_topology_coordinates=[],
        ).to_dict()
        if production is not None:
            runtime = production.runtime
            record.update(
                release_id=runtime.release_id,
                environment_hash=runtime.environment_hash,
                model_revision=runtime.model_revision,
                tokenizer_revision=runtime.tokenizer_revision,
                model_contract_sha256=runtime.model_contract_sha256,
                effective_context=int(profile_meta["effective_context_limit"]),
                rollout_generation=runtime.rollout_generation,
            )
            record["endpoint_generation"] = summarize_endpoint_generation([record])
        _publish_qid_record(
            cell=cell,
            cdir=cdir,
            results_path=results_path,
            record=record,
            expected_qids=expected_qids,
            expected_questions=questions,
            verified_benchmark_contracts=verified_benchmark_contracts,
            verified_manifest=manifest_snapshot,
            checkpoint=checkpoint,
        )
        records.append(record)
        done_qids.add(q.qid)

    # Publish one canonical JSONL atomically, then derive metadata statistics from the
    # entire full record set (including rows retained from earlier attempts).
    io.write_jsonl(results_path, records)
    canonical = canonicalize_results(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        write=False,
    )
    records = list(canonical.records)
    if canonical.has_corruption or len(records) != len(expected_qids):
        raise ExperimentConfigurationError(
            "canonical result validation failed before completion metadata publication"
        )

    meta.n_questions = len(expected_qids)
    meta.completed_question_count = sum(
        record.get("termination_status", TERMINATION_COMPLETED)
        == TERMINATION_COMPLETED
        for record in records
    )
    meta.length_censored_question_count = sum(
        record.get("termination_status") == TERMINATION_LENGTH_CENSORED
        for record in records
    )
    meta.protocol_censored_question_count = sum(
        record.get("termination_status") == TERMINATION_PROTOCOL_CENSORED
        for record in records
    )
    schema_counts: dict[str, int] = {}
    for record in records:
        schema = str(record.get("schema_version", "legacy"))
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
    meta.artifact_schema_counts = {
        schema: schema_counts[schema] for schema in sorted(schema_counts)
    }
    reasoning_summary = reasoning_token_summary(records)
    meta.mean_reasoning_tokens = reasoning_summary.mean_reasoning_tokens
    meta.mean_reasoning_tokens_exact = reasoning_summary.mean_reasoning_tokens_exact
    meta.exact_reasoning_question_count = reasoning_summary.exact_question_count
    meta.nonexact_reasoning_question_count = reasoning_summary.nonexact_question_count
    meta.mean_reasoning_word_count_legacy = (
        reasoning_summary.mean_reasoning_word_count_legacy
    )
    provenance_meta = summarize_serving_provenance(cell, records)
    meta.serving_profile = provenance_meta["serving_profile"]
    meta.effective_context_limit = provenance_meta["effective_context_limit"]
    meta.tensor_parallel_size = provenance_meta["tensor_parallel_size"]
    meta.serving_profile_counts = provenance_meta["serving_profile_counts"]
    meta.serving_profile_inferred_counts = provenance_meta[
        "serving_profile_inferred_counts"
    ]
    if production is not None:
        meta.endpoint_generation = summarize_endpoint_generation(records)
        meta.effective_context = meta.effective_context_limit
    meta.finished_at = time.time()
    meta_payload = meta.to_dict()
    validation_errors = validate_completion_payload(
        cell,
        expected_qids,
        records,
        meta_payload,
        expected_questions=questions,
        cell_directory=cdir,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        model_contract_path=model_contract_path,
    )
    if validation_errors:
        raise ExperimentConfigurationError(
            "completion payload is invalid: " + "; ".join(validation_errors[:5])
        )

    # meta.json is the success record.  It is emitted only after semantic validation.
    io.write_json(meta_path, meta_payload)
    final_status = get_completion_status(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=manifest_snapshot,
        check_active=False,
        serving_profile=profile.name,
        code_version=code_version,
        server_pool_generation=_current_server_pool_generation(
            server_root, profile.registry_key, production
        ),
        model_contract_path=model_contract_path,
    )
    if final_status.status is not CompletionState.COMPLETE:
        raise ExperimentConfigurationError(
            "published cell failed semantic completion validation: "
            + "; ".join(final_status.errors[:5])
        )
    clear_failure(cdir)
