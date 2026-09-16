"""Strict compiler from ``configs/nextdyn/*.yaml`` into frozen dataclasses (spec 17.4; rule R2).

Why a compiler and not a dict: ARD36's ``max_inflight`` was declared, plumbed and never
read (D-03), and nothing could tell.  Here every YAML key maps onto a dataclass field with
the **same name** (an unknown key is a :class:`ConfigError` naming its path, a missing key
too), and :data:`CONSUMERS` declares, for every leaf key path of ``study.yaml``,
``compute.yaml``, ``analysis.yaml``, ``protocols.yaml``, ``compositions.yaml`` and
``legacy_sources.yaml``, the module (or slurm template) that reads it.
``tests/nextdyn/test_guards.py`` walks the package AST and checks that the declared consumer
actually reads an attribute of that name; a consumer module that does not exist yet is
reported as *pending*, never silently accepted.  There is no runtime read-recording.  Where
this module only null-checks or cross-checks a knob (the launch blockers, the E2b grid), the
declared consumer is the module that *acts* on it (``plan``, ``freeze``, ``release``,
``cells``); the check here is an extra, never the consumer.

**Every verb obtains its config through** :func:`load_study` **with** ``check=True`` (the
default), which runs :func:`compile_checks` and raises :class:`ContractViolation` listing
every violation at once.  Only tests that build a violating config on purpose pass
``check=False`` and call :func:`compile_checks` directly.

:func:`compile_checks` implements the spec-17.4 rejections plus the plan's rules:

1. a native thinking budget (``models.*.thinking_tokens``) on a ``total_output_envelope``
   adapter is the spec-17.4 error; on a native-thinking cell it is refused too, because v1 has
   exactly one native cap (``budgets.thinking_tokens_main``) and a per-cell override that
   nothing reads would be a silent no-op (the D-03 shape).  ``total_output_tokens`` must be
   set iff the adapter is the envelope (it rides on ``contract.ModelCell``);
2. a non-deployable selector (``oracle``, ``oracle_analysis``) in ``selection.primary`` or
   ``selection.secondary_deployed``;
3. ``mode: confirmation`` with any null among ``locks.*``, ``budgets.campaign_gpu_hours``,
   ``budgets.campaign_cpu_hours``, ``budgets.storage_limit_bytes``,
   ``analysis.confirmatory_sample_size_lock``, ``analysis.analysis_lock_hash`` — or without
   ``confirmation_authorized: true``;
4. every non-null ``locks.*_lock_hash`` equals sha256 of the corresponding lock file beside
   ``study.yaml`` (:func:`lock_paths`), and every non-null inventory hash equals the field of
   the same name in ``datasets.lock.json`` -- the study.yaml hash is the pin the spec mandates;
   the ``.sha256`` sidecars the lock writer adds are a convenience for humans and never
   substitute for it;
5. the E2b grid declared in ``experiments.E2b.{endorsement, checker_access, initial_draft}``
   equals the set of enabled two-phase, packet-carrying receiver conditions in
   ``protocols.yaml`` through the one alias map :data:`E2B_ENDORSEMENT_SUPPORT` (R1: the
   grid is declared once; the registry instantiates it);
6. no configured path under ``/orcd/scratch`` (plan D3: data FS only).

Null lock and budget fields are intentional launch blockers, not values to guess.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import itertools
import json
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation
from agents_scaling.study import identity

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CONFIG_DIR: Final[Path] = REPO_ROOT / "configs" / "nextdyn"
DEFAULT_STUDY_CONFIG: Final[Path] = CONFIG_DIR / "study.yaml"
DEFAULT_COMPUTE_CONFIG: Final[Path] = CONFIG_DIR / "compute.yaml"
DEFAULT_ANALYSIS_CONFIG: Final[Path] = CONFIG_DIR / "analysis.yaml"
DEFAULT_PROTOCOLS_CONFIG: Final[Path] = CONFIG_DIR / "protocols.yaml"
SCRATCH_PREFIX: Final[str] = "/orcd/scratch"

#: The one alias map between the spec's E2b endorsement vocabulary (``experiments.E2b``)
#: and ``ReceiverSpec.support`` in ``protocols.yaml`` (spec 9.4: the correct candidate's
#: support cue).  Names, not knobs.
E2B_ENDORSEMENT_SUPPORT: Final[Mapping[str, str]] = {
    "correct_minority": "1:4",
    "correct_majority": "4:1",
    "hidden": "hidden",
}

# --------------------------------------------------------------------------- study.yaml blocks


@dataclass(frozen=True)
class StudyBlock:
    id: str
    schema_version: int
    spec_version: str
    contract_version: int
    mode: str
    confirmation_authorized: bool


@dataclass(frozen=True)
class SeedsBlock:
    study_seed_hex: str
    split_salt_hex: str


@dataclass(frozen=True)
class LocksBlock:
    """Every field is a sha256 (or a mapping for container digests) or ``null``; null blocks
    confirmation (spec 17.4).  The lock *files* sit beside ``study.yaml`` (:func:`lock_paths`)."""

    models_lock_hash: str | None
    datasets_lock_hash: str | None
    prompts_lock_hash: str | None
    source_inventory_hash: str | None
    prior_use_inventory_hash: str | None
    git_commit: str | None
    container_digests: Mapping[str, str] | None


@dataclass(frozen=True)
class ModelBlock:
    model_id: str
    revision: str | None
    tokenizer_revision: str | None
    adapter: str
    profile: str
    tp: int
    served_model_name: str
    lane: str
    enabled: bool
    thinking_tokens: int | None
    total_output_tokens: int | None


@dataclass(frozen=True)
class SamplingBlock:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float


@dataclass(frozen=True)
class BudgetsBlock:
    thinking_tokens_main: int
    final_phase_tokens_main: int
    receiver_thinking_tokens: tuple[int, ...]
    receiver_plan_final_tokens: int
    receiver_decision_final_tokens: int
    served_window_tokens: int
    context_reserve_tokens: int
    public_packet_token_cap: int
    public_rationale_token_cap: int
    claims_max: int
    claim_token_cap: int
    campaign_gpu_hours: float | None
    campaign_cpu_hours: float | None
    campaign_api_currency_limit: float | None
    storage_limit_bytes: int | None
    compute_reserve_fraction: float
    storage_reserve_fraction: float


@dataclass(frozen=True)
class SubPanelsBlock:
    """Hash-selected sub-panels that have no other home.  The E3a, capture, E2a and natural
    protection panels are sized by ``experiments.E3a.target_items_per_task``,
    ``instrumentation.target_items_per_task``, ``experiments.E2a.items_per_task`` and
    ``cohorts.natural_target_per_family`` -- one declaration each (R1)."""

    second_root: int
    bridge: int
    fragmented: int
    e3a_fill: int
    capture_natural: int


@dataclass(frozen=True)
class CohortsBlock:
    families: tuple[str, ...]
    pilot_items_per_family: int
    pilot_eligibility_extensions: tuple[int, ...]
    pilot_min_paired_items: int
    natural_target_per_family: int
    constructed_target_per_family: int
    paired_compositions: tuple[str, ...]
    fragmented_composition: str
    allow_natural_constructed_item_overlap: bool
    allow_pilot_confirmation_overlap: bool
    donor_initial_draws: int
    donor_max_draws: int
    difficulty_draws: int
    natural_root_draws: int
    natural_frame_prior_use: tuple[str, ...]
    sub_panels: SubPanelsBlock


@dataclass(frozen=True)
class ProtocolBlock:
    n_agents: int
    root_round: int
    revision_rounds: int
    score_rounds: tuple[int, ...]
    protocols: tuple[str, ...]
    synchronous: bool
    early_stop_on_consensus: bool
    own_previous_public_draft: bool
    visible_history: str
    native_thinking_shared: bool
    confidence_shared: bool
    final_phase_constrained_decoding: bool
    full_archive_stored_externally: bool


@dataclass(frozen=True)
class SelectionBlock:
    primary: str
    pool_scope: str
    tie_break: str
    synthesis_allowed: bool
    gold_access: bool
    secondary_deployed: tuple[str, ...]
    analysis_only: tuple[str, ...]
    audit_model: str
    audit_thinking_tokens: int
    audit_final_tokens: int
    audit_failure_fallback: str
    archive_audit_enabled: bool


@dataclass(frozen=True)
class E0Block:
    enabled: bool
    bcb_array_cells: int


@dataclass(frozen=True)
class E1Block:
    """E1 runs ``protocol.protocols`` on the natural and constructed cohorts (one declaration
    of the core protocol set, spec 17.4) plus the answer-only bridge."""

    enabled: bool
    bridge_protocol: str


@dataclass(frozen=True)
class E2aBlock:
    enabled: bool
    items_per_task: int
    tasks: tuple[str, ...]
    replicates: int
    models: tuple[str, ...]
    #: WP5: how many advisers the spec-9.1 constructed endorsement panel holds.  The 1:4 / 4:1
    #: levels are derived from it (``receiver.support_counts``), and E2b displays the same
    #: panel, so this one leaf is the single declaration for both assays.
    support_panel_size: int


@dataclass(frozen=True)
class E2bBlock:
    enabled: bool
    source_problems: int
    total_displayed_operations: int
    chain_depths: tuple[int, ...]
    modulus: int
    endorsement: tuple[str, ...]
    checker_access: tuple[bool, ...]
    initial_draft: tuple[str, ...]
    replicates: int
    max_checks: int
    generation_phases: int
    models: tuple[str, ...]


@dataclass(frozen=True)
class E3aBlock:
    enabled: bool
    target_items_per_task: int
    arms: tuple[str, ...]
    filled_arms: tuple[str, ...]
    natural_random_protection: bool
    protection_protocol: str


@dataclass(frozen=True)
class E3bBlock:
    enabled: bool
    ring_protocols: tuple[str, ...]
    team_sizes: tuple[int, ...]
    horizon_rounds: int
    horizon_score_rounds: tuple[int, ...]
    budgets: tuple[int, ...]


@dataclass(frozen=True)
class E3cBlock:
    enabled: bool
    model: str
    native_init: bool


@dataclass(frozen=True)
class E3cFamilyBlock:
    enabled: bool
    models: tuple[str, ...]


@dataclass(frozen=True)
class E4Block:
    enabled: bool


@dataclass(frozen=True)
class ExperimentsBlock:
    E0: E0Block
    E1: E1Block
    E2a: E2aBlock
    E2b: E2bBlock
    E3a: E3aBlock
    E3b: E3bBlock
    E3c: E3cBlock
    E3C_FAMILY: E3cFamilyBlock
    E4: E4Block


@dataclass(frozen=True)
class InstrumentationBlock:
    enabled: bool
    target_items_per_task: int
    capture_all_root_agents: bool
    anchors: tuple[str, ...]
    relative_layers: tuple[float, ...]
    mode: str
    intervention_enabled: bool
    protocols: tuple[str, ...]


@dataclass(frozen=True)
class PilotBlock:
    """Spec 19.1 exact pilot dispatch subset and the ``validate --smoke`` sizes.  Every number
    the pilot planner uses lives here (plan §3.1); nothing is hard-coded in ``plan``."""

    natural_items_per_family: int
    constructed_paired_items_per_family: int
    directional_items_per_family: int
    natural_receiver_items: int
    natural_receiver_tasks: tuple[str, ...]
    synthetic_receiver_problems: int
    adapter_check_problems: int
    capture_items_per_family: int
    smoke_solver_calls_per_model: int
    smoke_receiver_episodes: int


@dataclass(frozen=True)
class GatesBlock:
    """Spec 18.6 pilot-gate operational targets (design-quality targets, never grounds to
    delete confirmation examples).  Rates are fractions; the differential is in pp."""

    max_unresolved_infra_rate: float
    max_format_failure_rate: float
    max_arm_differential_pp: float


@dataclass(frozen=True)
class AnalysisBlock:
    primary_hypotheses: tuple[str, ...]
    primary_endpoint_round: int
    familywise_alpha: float
    multiplicity: str
    cluster: str
    paired: bool
    task_summary_weights: str
    semantic_correctness_states: tuple[str, ...]
    missingness_bounds_required: bool
    power_target: float
    confirmatory_sample_size_lock: Mapping[str, int] | None
    analysis_lock_hash: str | None
    profile: str


@dataclass(frozen=True)
class FailurePolicyBlock:
    transient_retries: int
    retry_same_request_and_seed: bool
    model_output_repair_calls: int
    wrong_answer_retry: bool
    context_overflow: str
    unresolved_semantics: str
    null_item_ids: str
    duplicate_logical_keys: str
    hidden_label_leakage: str


@dataclass(frozen=True)
class TestModeBlock:
    revision_rounds: int
    score_rounds: tuple[int, ...]


@dataclass(frozen=True)
class StudyConfig:
    study: StudyBlock
    seeds: SeedsBlock
    locks: LocksBlock
    models: Mapping[str, ModelBlock]
    sampling: SamplingBlock
    budgets: BudgetsBlock
    cohorts: CohortsBlock
    protocol: ProtocolBlock
    selection: SelectionBlock
    experiments: ExperimentsBlock
    instrumentation: InstrumentationBlock
    pilot: PilotBlock
    gates: GatesBlock
    analysis: AnalysisBlock
    failure_policy: FailurePolicyBlock
    test_mode: TestModeBlock
    # -- not YAML keys: provenance ---------------------------------------------------
    path: Path = dataclasses.field(default=Path(), metadata={"provenance": True})
    raw_sha256: str = dataclasses.field(default="", metadata={"provenance": True})

    @property
    def contract(self) -> K.StudyContract:
        return contract_of(self)

    def enabled_models(self) -> dict[str, ModelBlock]:
        return {name: block for name, block in self.models.items() if block.enabled}


# --------------------------------------------------------------------------- compute.yaml blocks


@dataclass(frozen=True)
class EngineBlock:
    vllm: str
    torch: str
    cuda: str
    dtype: str
    reasoning_parser: str


@dataclass(frozen=True)
class ServingBlock:
    engine: EngineBlock
    max_num_seqs_by_gpu: Mapping[str, Mapping[str, int]]
    gpu_memory_utilization: float
    exclude_nodes: tuple[str, ...]


@dataclass(frozen=True)
class PoolBlock:
    port_offset: int
    preemptable_allowed: bool
    spec: str


@dataclass(frozen=True)
class PoolsBlock:
    deadline: PoolBlock
    low: PoolBlock
    audit: PoolBlock
    assignment: Mapping[str, str]


@dataclass(frozen=True)
class CaptureFallbackBlock:
    partition: str
    gpu_types: tuple[str, ...]


@dataclass(frozen=True)
class CaptureBlock:
    partition: str
    gpu_type: str
    gpus_per_job: int
    fallback: CaptureFallbackBlock


@dataclass(frozen=True)
class PartitionBlock:
    gpu: Mapping[str, int]
    gpu_total: int | None
    cpu: int | None
    mem_gb: int | None
    max_submit: int
    preemptable: bool
    max_walltime: str


@dataclass(frozen=True)
class AssociationBlock:
    max_submit_jobs: int


@dataclass(frozen=True)
class DispatchBlock:
    throttle_per_endpoint: Mapping[str, int]
    lane_submit_caps: Mapping[str, int]
    bank_submit_cap: int
    global_submit_cap: int
    planner_pass_s: int
    cpu_driver_pass_s: int
    lane_driver_pass_s: int
    pending_alert_min: int
    stall_alert_min: int
    http_error_rate_max: float
    signal_seconds_before_walltime: int
    inflight_target_8b: tuple[int, int]
    inflight_target_32b: tuple[int, int]


@dataclass(frozen=True)
class KeepaliveBlock:
    tick_s: int
    max_launch: int
    cpus: int
    mem: str
    time: str
    partitions: tuple[str, ...]


@dataclass(frozen=True)
class WorkerBlock:
    cpus: int
    mem: str
    time: str
    partition: str
    overflow_partition: str | None


@dataclass(frozen=True)
class WorkersBlock:
    deadline: WorkerBlock
    loss_tolerant: WorkerBlock


@dataclass(frozen=True)
class ReferenceShape:
    n_agents: int
    revision_rounds: int
    thinking_cap: int


@dataclass(frozen=True)
class CellBlock:
    """Per-kind cell shape; cpus/mem/time/partition are null for endpoint kinds (they come
    from ``workers``), ``max_inflight`` may be the literal ``n_agents``."""

    items_per_cell: int
    parallel_items: int
    max_inflight: int | str
    cpus: int | None
    mem: str | None
    time: str | None
    partition: str | None


@dataclass(frozen=True)
class CellsBlock:
    reference: ReferenceShape
    kinds: Mapping[str, CellBlock]


@dataclass(frozen=True)
class ComputeConfig:
    serving: ServingBlock
    pools: PoolsBlock
    capture: CaptureBlock
    partitions: Mapping[str, PartitionBlock]
    association: AssociationBlock
    dispatch: DispatchBlock
    keepalive: KeepaliveBlock
    workers: WorkersBlock
    cells: CellsBlock
    path: Path = dataclasses.field(default=Path(), metadata={"provenance": True})
    raw_sha256: str = dataclasses.field(default="", metadata={"provenance": True})


# --------------------------------------------------------------------------- analysis.yaml blocks


@dataclass(frozen=True)
class ConfidenceWeightedBlock:
    mapping: str
    fallback: str
    min_task_support: int
    missing_confidence_weight: str
    complete_confidence_sensitivity: bool


@dataclass(frozen=True)
class EfficiencyBlock:
    reference_cost: Mapping[str, float] | None
    budget_levels: tuple[float, ...]
    truncation_rule: str


@dataclass(frozen=True)
class PowerBlock:
    """``alpha`` is not a knob: the planning alpha is ``study.analysis.familywise_alpha /
    len(study.analysis.primary_hypotheses)`` computed by ``power`` (R1)."""

    n_values: tuple[int, ...]
    n_sims: int
    target: float


@dataclass(frozen=True)
class AnalysisProfile:
    """Holm's ``n_primary`` is ``len(study.analysis.primary_hypotheses)`` and its alpha is
    ``study.analysis.familywise_alpha``, both computed by ``stats`` (never re-typed here)."""

    n_resamples: int
    bootstrap_seed_label_rule: str
    families: Mapping[str, float]
    difficulty_cutpoints: tuple[float, ...]
    reliability_bins: tuple[float, ...] | None
    confidence_weighted: ConfidenceWeightedBlock
    equivalence_margin_pp: float
    secondary_fdr_q: float
    min_clusters_for_test: int
    webb_below_clusters: int
    posterior_draws: int
    passk_ks: tuple[int, ...]
    efficiency: EfficiencyBlock
    power: PowerBlock


@dataclass(frozen=True)
class AnalysisConfig:
    profile_name: str
    profile: AnalysisProfile
    path: Path = dataclasses.field(default=Path(), metadata={"provenance": True})
    raw_sha256: str = dataclasses.field(default="", metadata={"provenance": True})


# --------------------------------------------------------------------------- strict builder


def _is_provenance(f: dataclasses.Field) -> bool:
    return bool(f.metadata.get("provenance"))


def _scalar(path: str, value: Any, hint: Any) -> Any:
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an int, got {type(value).__name__}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {type(value).__name__}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
        return value
    if hint is Any:
        return value
    raise ConfigError(f"{path}: unsupported field type {hint!r}")


def _convert(path: str, value: Any, hint: Any) -> Any:
    origin = typing.get_origin(hint)
    if origin is types.UnionType or origin is typing.Union:
        args = typing.get_args(hint)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(path, value, arg)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: no admissible type ({'; '.join(errors)})")
    if value is None:
        raise ConfigError(f"{path}: null is not allowed")
    if dataclasses.is_dataclass(hint):
        return build(hint, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        inner = typing.get_args(hint)
        if len(inner) == 2 and inner[1] is Ellipsis:
            return tuple(_convert(f"{path}[{i}]", v, inner[0]) for i, v in enumerate(value))
        if len(value) != len(inner):
            raise ConfigError(f"{path}: expected exactly {len(inner)} entries, got {len(value)}")
        return tuple(_convert(f"{path}[{i}]", v, h) for i, (v, h) in enumerate(zip(value, inner)))
    if origin in (collections.abc.Mapping, dict):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_hint, val_hint = typing.get_args(hint)
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{path}: mapping keys must be strings, got {key!r}")
            out[key] = _convert(f"{path}.{key}", item, val_hint)
        return out
    return _scalar(path, value, hint)


def build(cls: type, data: Any, path: str):
    """Instantiate a config dataclass from a mapping: unknown and missing keys are errors."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = [f for f in dataclasses.fields(cls) if not _is_provenance(f)]
    names = {f.name for f in fields}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path}.{unknown[0]}: unknown key (declared keys of {cls.__name__}: {sorted(names)})")
    missing = sorted(names - set(data))
    if missing:
        raise ConfigError(f"{path}.{missing[0]}: missing key (every {cls.__name__} key is required; "
                          "write null explicitly when a value is not yet known)")
    kwargs = {f.name: _convert(f"{path}.{f.name}", data[f.name], hints[f.name]) for f in fields}
    return cls(**kwargs)


def _read_yaml(path: Path) -> tuple[Any, str]:
    raw = Path(path).read_bytes()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data, identity.sha256_hex(raw)


# --------------------------------------------------------------------------- loaders


def sibling_config(cfg_path: Path, name: str) -> Path | None:
    """``<dir of study.yaml>/<name>`` when it exists (the spec-17.1 layout puts every config
    beside ``study.yaml``); ``None`` otherwise, so a temporary study.yaml compiles alone."""
    candidate = Path(cfg_path).resolve().parent / name
    return candidate if candidate.exists() else None


def load_study(path: str | Path | None = None, *, mode: str | None = None, check: bool = True) -> StudyConfig:
    """Compile ``study.yaml``.  ``mode`` overrides ``study.mode`` (tests pass ``"test"``, which
    also applies the ``test_mode`` block); the result is validated by :func:`validate_study`
    and, with ``check=True`` (the default every verb uses), held to :func:`compile_checks`
    against ``compute.yaml``, ``analysis.yaml`` and ``protocols.yaml`` beside it: any violation
    raises :class:`ContractViolation` listing all of them."""
    cfg_path = Path(path) if path is not None else DEFAULT_STUDY_CONFIG
    data, sha = _read_yaml(cfg_path)
    cfg = build(StudyConfig, data, "study")
    cfg = dataclasses.replace(cfg, path=cfg_path, raw_sha256=sha)
    if mode is not None:
        if mode not in K.MODES:
            raise ConfigError(f"unknown mode {mode!r}; expected one of {K.MODES}")
        cfg = dataclasses.replace(cfg, study=dataclasses.replace(cfg.study, mode=mode))
    if cfg.study.mode == K.MODE_TEST:
        cfg = dataclasses.replace(cfg, protocol=dataclasses.replace(
            cfg.protocol, revision_rounds=cfg.test_mode.revision_rounds,
            score_rounds=cfg.test_mode.score_rounds))
    validate_study(cfg)
    if check:
        compute_path = sibling_config(cfg_path, DEFAULT_COMPUTE_CONFIG.name)
        analysis_path = sibling_config(cfg_path, DEFAULT_ANALYSIS_CONFIG.name)
        violations = compile_checks(
            cfg,
            compute=load_compute(compute_path) if compute_path else None,
            analysis=load_analysis(analysis_path, profile=cfg.analysis.profile) if analysis_path else None,
            protocols=sibling_config(cfg_path, DEFAULT_PROTOCOLS_CONFIG.name),
        )
        if violations:
            raise ContractViolation(f"{cfg_path} does not compile ({len(violations)} violation(s)):\n  "
                                    + "\n  ".join(str(v) for v in violations))
    return cfg


def load_compute(path: str | Path | None = None) -> ComputeConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_COMPUTE_CONFIG
    data, sha = _read_yaml(cfg_path)
    cfg = build(ComputeConfig, data, "compute")
    cfg = dataclasses.replace(cfg, path=cfg_path, raw_sha256=sha)
    validate_compute(cfg)
    return cfg


def load_analysis(path: str | Path | None = None, *, profile: str) -> AnalysisConfig:
    """``analysis.yaml`` holds a complete ``confirmatory`` profile and override profiles
    (``test``) that replace leaves of it.  ``profile`` selects which one is compiled."""
    cfg_path = Path(path) if path is not None else DEFAULT_ANALYSIS_CONFIG
    data, sha = _read_yaml(cfg_path)
    unknown = sorted(set(data) - {"profiles"})
    if unknown:
        raise ConfigError(f"analysis.{unknown[0]}: unknown key")
    profiles = data.get("profiles")
    if not isinstance(profiles, Mapping) or "confirmatory" not in profiles:
        raise ConfigError("analysis.profiles must be a mapping holding at least 'confirmatory'")
    if profile not in profiles:
        raise ConfigError(f"analysis.profiles.{profile}: no such profile (have {sorted(profiles)})")
    merged = _deep_merge(profiles["confirmatory"], profiles[profile], f"analysis.profiles.{profile}") \
        if profile != "confirmatory" else profiles["confirmatory"]
    compiled = build(AnalysisProfile, merged, f"analysis.profiles.{profile}")
    return AnalysisConfig(profile_name=profile, profile=compiled, path=cfg_path, raw_sha256=sha)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Override leaves of ``base``; a key absent from ``base`` is an error (no new knobs in a
    test profile)."""
    out = dict(base)
    for key, value in override.items():
        if key not in base:
            raise ConfigError(f"{path}.{key}: override of a key the confirmatory profile does not declare")
        if isinstance(value, Mapping) and isinstance(base[key], Mapping):
            out[key] = _deep_merge(base[key], value, f"{path}.{key}")
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- validation


def validate_study(cfg: StudyConfig) -> None:
    if cfg.study.mode not in K.MODES:
        raise ConfigError(f"study.study.mode {cfg.study.mode!r} not in {K.MODES}")
    for name, block in cfg.models.items():
        if block.adapter not in K.ADAPTERS:
            raise ConfigError(f"study.models.{name}.adapter {block.adapter!r} not in {K.ADAPTERS}")
    if cfg.selection.audit_model not in cfg.models:
        raise ConfigError(f"study.selection.audit_model {cfg.selection.audit_model!r} is not a model cell")
    for block_name, names in (("E2a", cfg.experiments.E2a.models), ("E2b", cfg.experiments.E2b.models),
                              ("E3C_FAMILY", cfg.experiments.E3C_FAMILY.models)):
        for name in names:
            if name not in cfg.models:
                raise ConfigError(f"study.experiments.{block_name}.models names unknown model cell {name!r}")
    if cfg.experiments.E3c.model not in cfg.models:
        raise ConfigError(f"study.experiments.E3c.model {cfg.experiments.E3c.model!r} is not a model cell")
    if set(cfg.experiments.E2a.tasks) - set(cfg.cohorts.families):
        raise ConfigError("study.experiments.E2a.tasks names a family outside cohorts.families")
    if cfg.cohorts.fragmented_composition in cfg.cohorts.paired_compositions:
        raise ConfigError("study.cohorts.fragmented_composition must differ from the paired compositions")
    if cfg.analysis.profile not in ("confirmatory", "test"):
        raise ConfigError(f"study.analysis.profile {cfg.analysis.profile!r} unknown")
    if not cfg.protocol.synchronous or cfg.protocol.early_stop_on_consensus:
        raise ConfigError("study.protocol: v1 is synchronous with no early stop (spec 6.4, 8.4)")
    if cfg.protocol.visible_history != "previous_round_only":
        raise ConfigError("study.protocol.visible_history must be 'previous_round_only' (spec 6.4)")
    if cfg.protocol.final_phase_constrained_decoding:
        raise ConfigError("study.protocol.final_phase_constrained_decoding must be false: guided "
                          "decoding is not enforced on vLLM 0.21 (plan D6)")
    if cfg.protocol.native_thinking_shared or cfg.protocol.confidence_shared:
        raise ConfigError("study.protocol: native thinking and confidence are never shared (spec 6.1)")
    if cfg.instrumentation.intervention_enabled or cfg.instrumentation.mode != "read_only_replay":
        raise ConfigError("study.instrumentation: capture is read-only replay in v1 (plan D11)")
    if set(cfg.pilot.natural_receiver_tasks) - set(cfg.experiments.E2a.tasks):
        raise ConfigError("study.pilot.natural_receiver_tasks must be a subset of experiments.E2a.tasks")
    if cfg.experiments.E1.bridge_protocol in cfg.protocol.protocols:
        raise ConfigError("study.experiments.E1.bridge_protocol must not be one of protocol.protocols")
    for name, value in (("max_unresolved_infra_rate", cfg.gates.max_unresolved_infra_rate),
                        ("max_format_failure_rate", cfg.gates.max_format_failure_rate)):
        if not 0 < value < 1:
            raise ConfigError(f"study.gates.{name} is a fraction in (0, 1), got {value}")
    if cfg.gates.max_arm_differential_pp <= 0:
        raise ConfigError("study.gates.max_arm_differential_pp must be > 0 (percentage points)")


def validate_compute(cfg: ComputeConfig) -> None:
    for kind, pool in cfg.pools.assignment.items():
        if pool not in ("deadline", "low", "audit"):
            raise ConfigError(f"compute.pools.assignment.{kind}: unknown pool {pool!r}")
    for partition in (cfg.capture.partition, cfg.capture.fallback.partition,
                      cfg.workers.deadline.partition, cfg.workers.loss_tolerant.partition):
        if partition not in cfg.partitions:
            raise ConfigError(f"compute: partition {partition!r} is not in compute.partitions")
    for partition in cfg.keepalive.partitions:
        if partition not in cfg.partitions:
            raise ConfigError(f"compute.keepalive.partitions names unknown partition {partition!r}")
    for kind, block in cfg.cells.kinds.items():
        if isinstance(block.max_inflight, str) and block.max_inflight != "n_agents":
            raise ConfigError(f"compute.cells.kinds.{kind}.max_inflight: an int or the literal 'n_agents'")
        if block.partition is not None and block.partition not in cfg.partitions:
            raise ConfigError(f"compute.cells.kinds.{kind}.partition {block.partition!r} unknown")
    if "capture" in cfg.cells.kinds and cfg.cells.kinds["capture"].partition is not None:
        raise ConfigError("compute.cells.kinds.capture.partition must be null: the capture partition is "
                          "declared once as compute.capture.partition (R1)")
    total = sum(cfg.dispatch.lane_submit_caps.values()) + cfg.dispatch.bank_submit_cap
    if cfg.dispatch.global_submit_cap > cfg.association.max_submit_jobs:
        raise ConfigError("compute.dispatch.global_submit_cap exceeds association.max_submit_jobs")
    if total < cfg.dispatch.global_submit_cap:
        raise ConfigError("compute.dispatch: lane caps sum below the global cap; the global cap would never bind")


# --------------------------------------------------------------------------- contract


def _token_contract(cfg: StudyConfig, *, thinking: int, final: int) -> K.TokenContract:
    b = cfg.budgets
    return K.TokenContract(
        thinking_cap=thinking, final_cap=final, served_window=b.served_window_tokens,
        reserve=b.context_reserve_tokens, packet_cap=b.public_packet_token_cap,
        rationale_cap=b.public_rationale_token_cap, claims_max=b.claims_max, claim_cap=b.claim_token_cap,
    )


def model_cells(cfg: StudyConfig) -> dict[str, K.ModelCell]:
    return {
        name: K.ModelCell(name=name, hf_id=block.model_id, revision=block.revision,
                          tokenizer_revision=block.tokenizer_revision, adapter=block.adapter,
                          profile=block.profile, tp=block.tp, served_model_name=block.served_model_name,
                          lane=block.lane, total_output_tokens=block.total_output_tokens)
        for name, block in cfg.models.items()
    }


def contract_of(cfg: StudyConfig) -> K.StudyContract:
    """The parametric :class:`StudyContract` this config compiles to (every value from YAML)."""
    s = cfg.sampling
    receiver = {}
    for budget in cfg.budgets.receiver_thinking_tokens:
        if budget % 2:
            raise ContractViolation(f"receiver budget {budget} is not divisible into two B/2 phases")
        receiver[f"{budget}/plan"] = _token_contract(cfg, thinking=budget // 2,
                                                     final=cfg.budgets.receiver_plan_final_tokens)
        receiver[f"{budget}/decision"] = _token_contract(cfg, thinking=budget // 2,
                                                         final=cfg.budgets.receiver_decision_final_tokens)
        receiver[f"{budget}/one_call"] = _token_contract(cfg, thinking=budget,
                                                         final=cfg.budgets.receiver_decision_final_tokens)
    return K.StudyContract(
        study_id=cfg.study.id,
        study_seed_hex=cfg.seeds.study_seed_hex,
        split_salt_hex=cfg.seeds.split_salt_hex,
        contract_version=cfg.study.contract_version,
        families=tuple(cfg.cohorts.families),
        models=model_cells(cfg),
        sampling=K.Sampling.from_floats(
            temperature=s.temperature, top_p=s.top_p, top_k=s.top_k, min_p=s.min_p,
            repetition_penalty=s.repetition_penalty, presence_penalty=s.presence_penalty,
            frequency_penalty=s.frequency_penalty),
        tokens=_token_contract(cfg, thinking=cfg.budgets.thinking_tokens_main,
                               final=cfg.budgets.final_phase_tokens_main),
        receiver_tokens=receiver,
        audit_tokens=_token_contract(cfg, thinking=cfg.selection.audit_thinking_tokens,
                                     final=cfg.selection.audit_final_tokens),
        banks=K.BankSpec(difficulty_draws=cfg.cohorts.difficulty_draws,
                         donor_initial=cfg.cohorts.donor_initial_draws,
                         donor_max=cfg.cohorts.donor_max_draws,
                         natural_root_draws=cfg.cohorts.natural_root_draws),
        team=K.TeamShape(n_agents=cfg.protocol.n_agents, revision_rounds=cfg.protocol.revision_rounds,
                         score_rounds=cfg.protocol.score_rounds),
        selection=K.SelectionSpec(
            primary=cfg.selection.primary, pool_scope=cfg.selection.pool_scope,
            tie_break=cfg.selection.tie_break, secondary_deployed=cfg.selection.secondary_deployed,
            analysis_only=cfg.selection.analysis_only, audit_model=cfg.selection.audit_model,
            audit_thinking_tokens=cfg.selection.audit_thinking_tokens,
            audit_final_tokens=cfg.selection.audit_final_tokens,
            audit_failure_fallback=cfg.selection.audit_failure_fallback,
            synthesis_allowed=cfg.selection.synthesis_allowed, gold_access=cfg.selection.gold_access,
            archive_audit_enabled=cfg.selection.archive_audit_enabled),
        failure_policy=K.FailurePolicy(
            transient_retries=cfg.failure_policy.transient_retries,
            retry_same_request_and_seed=cfg.failure_policy.retry_same_request_and_seed,
            model_output_repair_calls=cfg.failure_policy.model_output_repair_calls,
            wrong_answer_retry=cfg.failure_policy.wrong_answer_retry,
            context_overflow=cfg.failure_policy.context_overflow,
            unresolved_semantics=cfg.failure_policy.unresolved_semantics,
            null_item_ids=cfg.failure_policy.null_item_ids,
            duplicate_logical_keys=cfg.failure_policy.duplicate_logical_keys,
            hidden_label_leakage=cfg.failure_policy.hidden_label_leakage),
    )


# --------------------------------------------------------------------------- compile checks

CONFIRMATION_REQUIRED: Final[tuple[str, ...]] = (
    "locks.models_lock_hash", "locks.datasets_lock_hash", "locks.prompts_lock_hash",
    "locks.source_inventory_hash", "locks.prior_use_inventory_hash", "locks.git_commit",
    "locks.container_digests", "budgets.campaign_gpu_hours", "budgets.campaign_cpu_hours",
    "budgets.storage_limit_bytes", "analysis.confirmatory_sample_size_lock", "analysis.analysis_lock_hash",
)


def _leaf(cfg: Any, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def compile_checks(cfg: StudyConfig, *, compute: ComputeConfig | None = None,
                   analysis: AnalysisConfig | None = None, protocols: Path | None = None,
                   extra_paths: Sequence[Path] = ()) -> list[ContractViolation]:
    """Every spec-17.4 rejection plus the plan's rules, all at once (empty list = compiles).

    ``protocols`` is the ``protocols.yaml`` to cross-check the E2b grid against (``None``
    skips that rule, for a study.yaml compiled alone); :func:`load_study` passes the one
    beside ``study.yaml`` whenever it exists.
    """
    out: list[ContractViolation] = []
    for name, block in cfg.models.items():
        if block.thinking_tokens is not None:
            if block.adapter == K.ADAPTER_TOTAL_OUTPUT_ENVELOPE:
                out.append(ContractViolation(
                    f"study.models.{name}: a native thinking budget ({block.thinking_tokens}) on the "
                    f"{K.ADAPTER_TOTAL_OUTPUT_ENVELOPE} adapter is an error (spec 17.4, 5.6)"))
            else:
                out.append(ContractViolation(
                    f"study.models.{name}: thinking_tokens={block.thinking_tokens} is a per-cell override "
                    "nothing implements in v1; the one native cap is budgets.thinking_tokens_main "
                    "(a value here would be a silent no-op, the D-03 shape) -- write null"))
        if block.adapter == K.ADAPTER_TOTAL_OUTPUT_ENVELOPE and block.total_output_tokens is None:
            out.append(ContractViolation(
                f"study.models.{name}: the {K.ADAPTER_TOTAL_OUTPUT_ENVELOPE} adapter needs total_output_tokens"))
        if block.adapter == K.ADAPTER_NATIVE_THINKING and block.total_output_tokens is not None:
            out.append(ContractViolation(
                f"study.models.{name}: total_output_tokens belongs to the envelope adapter, not native thinking"))
    if not out:
        try:
            contract_of(cfg)
        except ContractViolation as exc:
            out.append(exc)
    for selector in (cfg.selection.primary, *cfg.selection.secondary_deployed):
        if selector in K.NON_DEPLOYABLE_SELECTORS:
            out.append(ContractViolation(
                f"study.selection: {selector!r} is an evaluation-only selector and may not appear in "
                "primary or secondary_deployed (spec 6.5, 17.4)"))
    if cfg.study.mode == K.MODE_CONFIRMATION:
        if not cfg.study.confirmation_authorized:
            out.append(ContractViolation("study.study.confirmation_authorized must be true in confirmation mode"))
        for dotted in CONFIRMATION_REQUIRED:
            if _leaf(cfg, dotted) is None:
                out.append(ContractViolation(
                    f"study.{dotted} is null: confirmation needs every lock, resource ceiling, "
                    "sample-size lock and analysis lock (spec 16.4, 17.4)"))
    out.extend(lock_hash_checks(cfg))
    if protocols is not None:
        out.extend(e2b_grid_checks(cfg, protocols))
    for offending in scratch_paths(cfg.path, *(c.path for c in (compute, analysis) if c is not None),
                                   *((protocols,) if protocols is not None else ()), *extra_paths):
        out.append(ContractViolation(f"{offending}: paths under {SCRATCH_PREFIX} are refused (plan D3)"))
    return out


def lock_hash_checks(cfg: StudyConfig) -> list[ContractViolation]:
    """Rule 4: a non-null ``locks.*_lock_hash`` must equal sha256 of the lock file beside
    ``study.yaml``; a non-null inventory hash must equal the same-named field of
    ``datasets.lock.json``.  A stale or absent file is a violation, never a warning."""
    out: list[ContractViolation] = []
    paths = lock_paths(cfg.path)
    for key, path in (("models_lock_hash", paths.models), ("datasets_lock_hash", paths.datasets),
                      ("prompts_lock_hash", paths.prompts)):
        declared = getattr(cfg.locks, key)
        if declared is None:
            continue
        if not is_sha256(declared):
            out.append(ContractViolation(f"study.locks.{key} is not a sha256 hex string"))
            continue
        if not path.exists():
            out.append(ContractViolation(f"study.locks.{key} is set but {path} does not exist"))
            continue
        actual = identity.sha256_hex(path.read_bytes())
        if actual != declared:
            out.append(ContractViolation(
                f"study.locks.{key} {declared[:12]} != sha256({path.name}) {actual[:12]}: the lock file "
                "changed after study.yaml pinned it (stale pin); re-run lock-inputs and copy the hash"))
    for key in ("source_inventory_hash", "prior_use_inventory_hash"):
        declared = getattr(cfg.locks, key)
        if declared is None:
            continue
        if not paths.datasets.exists():
            out.append(ContractViolation(f"study.locks.{key} is set but {paths.datasets} does not exist"))
            continue
        try:
            recorded = json.loads(paths.datasets.read_text(encoding="utf-8")).get(key)
        except (ValueError, AttributeError) as exc:
            out.append(ContractViolation(f"{paths.datasets}: not a JSON object ({exc})"))
            continue
        if recorded != declared:
            out.append(ContractViolation(
                f"study.locks.{key} {str(declared)[:12]} != datasets.lock.json {key} {str(recorded)[:12]}"))
    return out


def e2b_grid_checks(cfg: StudyConfig, protocols_path: Path) -> list[ContractViolation]:
    """Rule 5: the E2b grid ``endorsement x checker_access x initial_draft`` (spec 17.4 names,
    one declaration) equals the set of enabled two-phase packet-carrying receivers in
    ``protocols.yaml`` mapped through :data:`E2B_ENDORSEMENT_SUPPORT`."""
    out: list[ContractViolation] = []
    e2b = cfg.experiments.E2b
    unknown = sorted(set(e2b.endorsement) - set(E2B_ENDORSEMENT_SUPPORT))
    if unknown:
        return [ContractViolation(f"study.experiments.E2b.endorsement names {unknown}; the alias map "
                                  f"E2B_ENDORSEMENT_SUPPORT knows {sorted(E2B_ENDORSEMENT_SUPPORT)}")]
    declared = {(E2B_ENDORSEMENT_SUPPORT[e], bool(c), d)
                for e, c, d in itertools.product(e2b.endorsement, e2b.checker_access, e2b.initial_draft)}
    data, _ = _read_yaml(Path(protocols_path))
    receivers = data.get("receivers")
    if not isinstance(receivers, Mapping):
        return [ContractViolation(f"{protocols_path}: no 'receivers' mapping")]
    instantiated = set()
    for name, entry in receivers.items():
        if not isinstance(entry, Mapping):
            out.append(ContractViolation(f"{protocols_path}: receivers.{name} is not a mapping"))
            continue
        if not entry.get("enabled") or entry.get("packet_content") == "none":
            continue
        if entry.get("phases") != e2b.generation_phases:
            continue
        instantiated.add((entry.get("support"), bool(entry.get("checker")), entry.get("draft_role")))
    if declared != instantiated:
        out.append(ContractViolation(
            "study.experiments.E2b grid != protocols.yaml enabled receivers with phases="
            f"{e2b.generation_phases}: declared-only {sorted(declared - instantiated)}, "
            f"registry-only {sorted(instantiated - declared)}"))
    return out


def scratch_paths(*paths: Path) -> list[str]:
    """``"<file>:<key path>"`` for every string leaf under ``/orcd/scratch`` in the given YAML files."""
    out: list[str] = []
    for path in paths:
        if not path or not Path(path).exists():
            continue
        data, _ = _read_yaml(Path(path))
        for key_path, value in leaf_paths(data):
            if isinstance(value, str) and SCRATCH_PREFIX in value:
                out.append(f"{Path(path).name}:{key_path}")
    return out


def leaf_paths(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """``(dotted key path, value)`` for every leaf of a YAML tree; lists are leaves."""
    if isinstance(data, Mapping):
        out: list[tuple[str, Any]] = []
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(leaf_paths(value, child))
        return out
    return [(prefix, data)]


# --------------------------------------------------------------------------- consumers (R2)

_MOD = "agents_scaling.nextdyn."
_SLURM_KEEPALIVE = "slurm/nextdyn_keepalive.sbatch.tmpl"
_SLURM_CELLS = "slurm/nextdyn_cell_array.sbatch.tmpl"
_SLURM_CAPTURE = "slurm/nextdyn_capture.sbatch.tmpl"

#: leaf key path pattern -> consuming module (dotted) or slurm template (path).  ``*`` matches
#: one data-keyed segment (a model name, a partition, a kind).  For a pattern ending in ``*``
#: the consumer must read the last named segment.  Covers all six YAML files under
#: ``configs/nextdyn`` (list leaves included).  Checked by
#: ``test_every_config_leaf_has_a_declared_consumer_that_reads_it``; a consumer that does not
#: exist yet is *pending* (``test_pending_consumers_are_listed``).  The consumer is the module
#: that ACTS on the value; where this module merely null-checks or cross-checks a knob, that
#: is not the declaration (see the module docstring).
CONSUMERS: Final[Mapping[str, str]] = {
    # ---- study.yaml ---------------------------------------------------------------
    "study.study.id": _MOD + "cli",
    "study.study.schema_version": _MOD + "freeze",
    "study.study.spec_version": _MOD + "freeze",
    "study.study.contract_version": _MOD + "config",
    "study.study.mode": _MOD + "config",
    "study.study.confirmation_authorized": _MOD + "config",
    "study.seeds.study_seed_hex": _MOD + "config",
    "study.seeds.split_salt_hex": _MOD + "config",
    "study.locks.models_lock_hash": _MOD + "freeze",
    "study.locks.datasets_lock_hash": _MOD + "freeze",
    "study.locks.prompts_lock_hash": _MOD + "freeze",
    "study.locks.source_inventory_hash": _MOD + "freeze",
    "study.locks.prior_use_inventory_hash": _MOD + "freeze",
    "study.locks.git_commit": _MOD + "freeze",
    "study.locks.container_digests": _MOD + "freeze",
    "study.models.*.model_id": _MOD + "config",
    "study.models.*.revision": _MOD + "config",
    "study.models.*.tokenizer_revision": _MOD + "config",
    "study.models.*.adapter": _MOD + "config",
    "study.models.*.profile": _MOD + "config",
    "study.models.*.tp": _MOD + "config",
    "study.models.*.served_model_name": _MOD + "config",
    "study.models.*.lane": _MOD + "config",
    "study.models.*.enabled": _MOD + "config",
    "study.models.*.thinking_tokens": _MOD + "config",       # the spec-17.4 trap field: refused unless null
    "study.models.*.total_output_tokens": _MOD + "adapters",  # TotalOutputEnvelopeAdapter cap (via ModelCell)
    "study.sampling.temperature": _MOD + "config",
    "study.sampling.top_p": _MOD + "config",
    "study.sampling.top_k": _MOD + "config",
    "study.sampling.min_p": _MOD + "config",
    "study.sampling.repetition_penalty": _MOD + "config",
    "study.sampling.presence_penalty": _MOD + "config",
    "study.sampling.frequency_penalty": _MOD + "config",
    "study.budgets.thinking_tokens_main": _MOD + "config",
    "study.budgets.final_phase_tokens_main": _MOD + "config",
    "study.budgets.receiver_thinking_tokens": _MOD + "config",
    "study.budgets.receiver_plan_final_tokens": _MOD + "config",
    "study.budgets.receiver_decision_final_tokens": _MOD + "config",
    "study.budgets.served_window_tokens": _MOD + "config",
    "study.budgets.context_reserve_tokens": _MOD + "config",
    "study.budgets.public_packet_token_cap": _MOD + "config",
    "study.budgets.public_rationale_token_cap": _MOD + "config",
    "study.budgets.claims_max": _MOD + "config",
    "study.budgets.claim_token_cap": _MOD + "config",
    "study.budgets.campaign_gpu_hours": _MOD + "plan",          # ceiling admission (spec 16.4)
    "study.budgets.campaign_cpu_hours": _MOD + "plan",
    "study.budgets.campaign_api_currency_limit": _MOD + "plan",
    "study.budgets.storage_limit_bytes": _MOD + "plan",
    "study.budgets.compute_reserve_fraction": _MOD + "plan",
    "study.budgets.storage_reserve_fraction": _MOD + "plan",
    "study.cohorts.families": _MOD + "config",
    "study.cohorts.pilot_items_per_family": _MOD + "inventory",
    "study.cohorts.pilot_eligibility_extensions": _MOD + "inventory",
    "study.cohorts.pilot_min_paired_items": _MOD + "gates",
    "study.cohorts.natural_target_per_family": _MOD + "inventory",
    "study.cohorts.constructed_target_per_family": _MOD + "plan",
    "study.cohorts.paired_compositions": _MOD + "construct",
    "study.cohorts.fragmented_composition": _MOD + "construct",
    "study.cohorts.allow_natural_constructed_item_overlap": _MOD + "inventory",
    "study.cohorts.allow_pilot_confirmation_overlap": _MOD + "inventory",
    "study.cohorts.donor_initial_draws": _MOD + "config",
    "study.cohorts.donor_max_draws": _MOD + "config",
    "study.cohorts.difficulty_draws": _MOD + "config",
    "study.cohorts.natural_root_draws": _MOD + "config",
    "study.cohorts.natural_frame_prior_use": _MOD + "inventory",
    "study.cohorts.sub_panels.second_root": _MOD + "inventory",
    "study.cohorts.sub_panels.bridge": _MOD + "inventory",
    "study.cohorts.sub_panels.fragmented": _MOD + "inventory",
    "study.cohorts.sub_panels.e3a_fill": _MOD + "inventory",
    "study.cohorts.sub_panels.capture_natural": _MOD + "inventory",
    "study.protocol.n_agents": _MOD + "config",
    "study.protocol.root_round": _MOD + "episode",
    "study.protocol.revision_rounds": _MOD + "config",
    "study.protocol.score_rounds": _MOD + "config",
    "study.protocol.protocols": _MOD + "cells",
    "study.protocol.synchronous": _MOD + "config",
    "study.protocol.early_stop_on_consensus": _MOD + "config",
    "study.protocol.own_previous_public_draft": _MOD + "protocols",
    "study.protocol.visible_history": _MOD + "config",
    "study.protocol.native_thinking_shared": _MOD + "config",
    "study.protocol.confidence_shared": _MOD + "config",
    "study.protocol.final_phase_constrained_decoding": _MOD + "config",
    "study.protocol.full_archive_stored_externally": _MOD + "materialize",
    "study.selection.primary": _MOD + "config",
    "study.selection.pool_scope": _MOD + "config",
    "study.selection.tie_break": _MOD + "config",
    "study.selection.synthesis_allowed": _MOD + "config",
    "study.selection.gold_access": _MOD + "config",
    "study.selection.secondary_deployed": _MOD + "config",
    "study.selection.analysis_only": _MOD + "config",
    "study.selection.audit_model": _MOD + "config",
    "study.selection.audit_thinking_tokens": _MOD + "config",
    "study.selection.audit_final_tokens": _MOD + "config",
    "study.selection.audit_failure_fallback": _MOD + "config",
    "study.selection.archive_audit_enabled": _MOD + "config",
    "study.experiments.E0.enabled": _MOD + "plan",
    "study.experiments.E0.bcb_array_cells": _MOD + "e0.main",
    "study.experiments.E1.enabled": _MOD + "plan",
    "study.experiments.E1.bridge_protocol": _MOD + "cells",
    "study.experiments.E2a.enabled": _MOD + "plan",
    "study.experiments.E2a.items_per_task": _MOD + "cells",
    "study.experiments.E2a.tasks": _MOD + "config",
    "study.experiments.E2a.replicates": _MOD + "cells",
    "study.experiments.E2a.models": _MOD + "config",
    "study.experiments.E2a.support_panel_size": _MOD + "receiver",
    "study.experiments.E2b.enabled": _MOD + "plan",
    "study.experiments.E2b.source_problems": _MOD + "cells",
    "study.experiments.E2b.total_displayed_operations": _MOD + "mod97",
    "study.experiments.E2b.chain_depths": _MOD + "mod97",
    "study.experiments.E2b.modulus": _MOD + "mod97",
    "study.experiments.E2b.endorsement": _MOD + "cells",      # the declared grid; config cross-checks it
    "study.experiments.E2b.checker_access": _MOD + "cells",
    "study.experiments.E2b.initial_draft": _MOD + "cells",
    "study.experiments.E2b.replicates": _MOD + "cells",
    "study.experiments.E2b.max_checks": _MOD + "checker",
    "study.experiments.E2b.generation_phases": _MOD + "receiver",
    "study.experiments.E2b.models": _MOD + "config",
    "study.experiments.E3a.enabled": _MOD + "plan",
    "study.experiments.E3a.target_items_per_task": _MOD + "cells",
    "study.experiments.E3a.arms": _MOD + "cells",
    "study.experiments.E3a.filled_arms": _MOD + "cells",
    "study.experiments.E3a.natural_random_protection": _MOD + "cells",
    "study.experiments.E3a.protection_protocol": _MOD + "cells",
    "study.experiments.E3b.enabled": _MOD + "plan",
    "study.experiments.E3b.ring_protocols": _MOD + "cells",
    "study.experiments.E3b.team_sizes": _MOD + "cells",
    "study.experiments.E3b.horizon_rounds": _MOD + "cells",
    "study.experiments.E3b.horizon_score_rounds": _MOD + "cells",
    "study.experiments.E3b.budgets": _MOD + "cells",
    "study.experiments.E3c.enabled": _MOD + "plan",
    "study.experiments.E3c.model": _MOD + "config",
    "study.experiments.E3c.native_init": _MOD + "cells",
    "study.experiments.E3C_FAMILY.enabled": _MOD + "plan",
    "study.experiments.E3C_FAMILY.models": _MOD + "config",
    "study.experiments.E4.enabled": _MOD + "plan",
    "study.instrumentation.enabled": _MOD + "plan",
    "study.instrumentation.target_items_per_task": _MOD + "cells",
    "study.instrumentation.capture_all_root_agents": _MOD + "capture",
    "study.instrumentation.anchors": _MOD + "capture",
    "study.instrumentation.relative_layers": _MOD + "capture",
    "study.instrumentation.mode": _MOD + "config",
    "study.instrumentation.intervention_enabled": _MOD + "config",
    "study.instrumentation.protocols": _MOD + "cells",
    "study.pilot.natural_items_per_family": _MOD + "plan",
    "study.pilot.constructed_paired_items_per_family": _MOD + "plan",
    "study.pilot.directional_items_per_family": _MOD + "plan",
    "study.pilot.natural_receiver_items": _MOD + "plan",
    "study.pilot.natural_receiver_tasks": _MOD + "plan",
    "study.pilot.synthetic_receiver_problems": _MOD + "plan",
    "study.pilot.adapter_check_problems": _MOD + "plan",
    "study.pilot.capture_items_per_family": _MOD + "plan",
    "study.pilot.smoke_solver_calls_per_model": _MOD + "gates",
    "study.pilot.smoke_receiver_episodes": _MOD + "gates",
    "study.gates.max_unresolved_infra_rate": _MOD + "gates",
    "study.gates.max_format_failure_rate": _MOD + "gates",
    "study.gates.max_arm_differential_pp": _MOD + "gates",
    "study.analysis.primary_hypotheses": _MOD + "analysis",
    "study.analysis.primary_endpoint_round": _MOD + "analysis",
    "study.analysis.familywise_alpha": _MOD + "analysis",
    "study.analysis.multiplicity": _MOD + "analysis",
    "study.analysis.cluster": _MOD + "analysis",
    "study.analysis.paired": _MOD + "analysis",
    "study.analysis.task_summary_weights": _MOD + "analysis",
    "study.analysis.semantic_correctness_states": _MOD + "outcomes",
    "study.analysis.missingness_bounds_required": _MOD + "analysis",
    "study.analysis.power_target": _MOD + "power",
    "study.analysis.confirmatory_sample_size_lock": _MOD + "freeze",
    "study.analysis.analysis_lock_hash": _MOD + "release",
    "study.analysis.profile": _MOD + "config",
    "study.failure_policy.transient_retries": _MOD + "config",
    "study.failure_policy.retry_same_request_and_seed": _MOD + "config",
    "study.failure_policy.model_output_repair_calls": _MOD + "config",
    "study.failure_policy.wrong_answer_retry": _MOD + "config",
    "study.failure_policy.context_overflow": _MOD + "config",
    "study.failure_policy.unresolved_semantics": _MOD + "config",
    "study.failure_policy.null_item_ids": _MOD + "config",
    "study.failure_policy.duplicate_logical_keys": _MOD + "config",
    "study.failure_policy.hidden_label_leakage": _MOD + "config",
    "study.test_mode.revision_rounds": _MOD + "config",
    "study.test_mode.score_rounds": _MOD + "config",
    # ---- compute.yaml -------------------------------------------------------------
    "compute.serving.engine.vllm": _MOD + "locks",
    "compute.serving.engine.torch": _MOD + "locks",
    "compute.serving.engine.cuda": _MOD + "locks",
    "compute.serving.engine.dtype": _MOD + "locks",
    "compute.serving.engine.reasoning_parser": _MOD + "locks",
    "compute.serving.max_num_seqs_by_gpu.*.*": _MOD + "fleet",
    "compute.serving.gpu_memory_utilization": _MOD + "fleet",
    "compute.serving.exclude_nodes": _MOD + "fleet",
    "compute.pools.*.port_offset": _MOD + "fleet",
    "compute.pools.*.preemptable_allowed": _MOD + "kinds",
    "compute.pools.*.spec": _MOD + "fleet",
    "compute.pools.assignment.*": _MOD + "kinds",
    "compute.capture.partition": _MOD + "kinds",
    "compute.capture.gpu_type": _SLURM_CAPTURE,
    "compute.capture.gpus_per_job": _SLURM_CAPTURE,
    "compute.capture.fallback.partition": _MOD + "dispatch",
    "compute.capture.fallback.gpu_types": _MOD + "dispatch",
    "compute.partitions.*.gpu.*": _MOD + "fleet",
    "compute.partitions.*.gpu_total": _MOD + "fleet",
    "compute.partitions.*.cpu": _MOD + "dispatch",
    "compute.partitions.*.mem_gb": _MOD + "dispatch",
    "compute.partitions.*.max_submit": _MOD + "dispatch",
    "compute.partitions.*.preemptable": _MOD + "fleet",
    "compute.partitions.*.max_walltime": _MOD + "fleet",
    "compute.association.max_submit_jobs": _MOD + "config",
    "compute.dispatch.throttle_per_endpoint.*": _MOD + "dispatch",
    "compute.dispatch.lane_submit_caps.*": _MOD + "dispatch",
    "compute.dispatch.bank_submit_cap": _MOD + "dispatch",
    "compute.dispatch.global_submit_cap": _MOD + "dispatch",
    "compute.dispatch.planner_pass_s": _MOD + "plan",
    "compute.dispatch.cpu_driver_pass_s": _MOD + "dispatch",
    "compute.dispatch.lane_driver_pass_s": _MOD + "dispatch",
    "compute.dispatch.pending_alert_min": _MOD + "status",
    "compute.dispatch.stall_alert_min": _MOD + "status",
    "compute.dispatch.http_error_rate_max": _MOD + "status",
    "compute.dispatch.signal_seconds_before_walltime": _SLURM_CELLS,
    "compute.dispatch.inflight_target_8b": _MOD + "status",
    "compute.dispatch.inflight_target_32b": _MOD + "status",
    "compute.keepalive.tick_s": _MOD + "fleet",
    "compute.keepalive.max_launch": _MOD + "fleet",
    "compute.keepalive.cpus": _SLURM_KEEPALIVE,
    "compute.keepalive.mem": _SLURM_KEEPALIVE,
    "compute.keepalive.time": _SLURM_KEEPALIVE,
    "compute.keepalive.partitions": _SLURM_KEEPALIVE,
    "compute.workers.*.cpus": _MOD + "kinds",
    "compute.workers.*.mem": _MOD + "kinds",
    "compute.workers.*.time": _MOD + "kinds",
    "compute.workers.*.partition": _MOD + "kinds",
    "compute.workers.*.overflow_partition": _MOD + "dispatch",
    "compute.cells.reference.n_agents": _MOD + "kinds",
    "compute.cells.reference.revision_rounds": _MOD + "kinds",
    "compute.cells.reference.thinking_cap": _MOD + "kinds",
    "compute.cells.kinds.*.items_per_cell": _MOD + "kinds",
    "compute.cells.kinds.*.parallel_items": _MOD + "kinds",
    "compute.cells.kinds.*.max_inflight": _MOD + "kinds",
    "compute.cells.kinds.*.cpus": _MOD + "kinds",
    "compute.cells.kinds.*.mem": _MOD + "kinds",
    "compute.cells.kinds.*.time": _MOD + "kinds",
    "compute.cells.kinds.*.partition": _MOD + "kinds",
    # ---- analysis.yaml ------------------------------------------------------------
    "analysis.profiles.*.n_resamples": _MOD + "stats",
    "analysis.profiles.*.bootstrap_seed_label_rule": _MOD + "stats",
    "analysis.profiles.*.families.*": _MOD + "analysis",
    "analysis.profiles.*.difficulty_cutpoints": _MOD + "banks",
    "analysis.profiles.*.reliability_bins": _MOD + "calib",
    "analysis.profiles.*.confidence_weighted.mapping": _MOD + "selectors",
    "analysis.profiles.*.confidence_weighted.fallback": _MOD + "selectors",
    "analysis.profiles.*.confidence_weighted.min_task_support": _MOD + "selectors",
    "analysis.profiles.*.confidence_weighted.missing_confidence_weight": _MOD + "selectors",
    "analysis.profiles.*.confidence_weighted.complete_confidence_sensitivity": _MOD + "selectors",
    "analysis.profiles.*.equivalence_margin_pp": _MOD + "stats",
    "analysis.profiles.*.secondary_fdr_q": _MOD + "stats",
    "analysis.profiles.*.min_clusters_for_test": _MOD + "stats",
    "analysis.profiles.*.webb_below_clusters": _MOD + "stats",
    "analysis.profiles.*.posterior_draws": _MOD + "analysis",
    "analysis.profiles.*.passk_ks": _MOD + "panels",
    "analysis.profiles.*.efficiency.reference_cost": _MOD + "analysis",
    "analysis.profiles.*.efficiency.budget_levels": _MOD + "analysis",
    "analysis.profiles.*.efficiency.truncation_rule": _MOD + "analysis",
    "analysis.profiles.*.power.n_values": _MOD + "power",
    "analysis.profiles.*.power.n_sims": _MOD + "power",
    "analysis.profiles.*.power.target": _MOD + "power",
    # ---- protocols.yaml (WP2 loads it; conditions are data, plan D7) ---------------
    "protocols.episodes.*.own_draft": _MOD + "protocols",
    "protocols.episodes.*.graph.kind": _MOD + "protocols",
    "protocols.episodes.*.graph.ref.name": _MOD + "protocols",
    "protocols.episodes.*.graph.ref.params.*": _MOD + "graphs",
    "protocols.episodes.*.packet_content": _MOD + "protocols",
    "protocols.episodes.*.filler": _MOD + "protocols",
    "protocols.episodes.*.template": _MOD + "protocols",
    "protocols.episodes.*.schedule": _MOD + "protocols",
    "protocols.episodes.*.display_label": _MOD + "protocols",
    "protocols.episodes.*.legacy_alias": _MOD + "protocols",
    "protocols.episodes.*.equivalent_to": _MOD + "plan",
    "protocols.episodes.*.enabled": _MOD + "plan",
    "protocols.receivers.*.packet_content": _MOD + "receiver",
    "protocols.receivers.*.support": _MOD + "receiver",
    "protocols.receivers.*.checker": _MOD + "receiver",
    "protocols.receivers.*.draft_role": _MOD + "receiver",
    "protocols.receivers.*.phases": _MOD + "receiver",
    "protocols.receivers.*.templates": _MOD + "receiver",
    "protocols.receivers.*.display_label": _MOD + "protocols",
    "protocols.receivers.*.legacy_alias": _MOD + "protocols",
    "protocols.receivers.*.equivalent_to": _MOD + "plan",
    "protocols.receivers.*.enabled": _MOD + "plan",
    # ---- compositions.yaml (WP3) -------------------------------------------------
    "compositions.compositions.*.n": _MOD + "construct",
    "compositions.compositions.*.c0": _MOD + "construct",
    "compositions.compositions.*.wrong_pattern": _MOD + "construct",
    "compositions.compositions.*.wrong_pattern.fragmented": _MOD + "construct",
    "compositions.compositions.*.enabled": _MOD + "plan",
    "compositions.pairs.*": _MOD + "construct",
    "compositions.extension_rule.name": _MOD + "banks",
    "compositions.extension_rule.min_correct_resolved": _MOD + "banks",
    "compositions.extension_rule.min_wrong_resolved": _MOD + "banks",
    "compositions.construction_rule_version": _MOD + "construct",
    # ---- legacy_sources.yaml (WP7) -----------------------------------------------
    "legacy_sources.ard36.*": _MOD + "e0.registry",
    "legacy_sources.study_v4.*": _MOD + "e0.registry",
    "legacy_sources.full_sweep_v1.*": _MOD + "e0.registry",
    "legacy_sources.full_sweep_v1.cache.*": _MOD + "e0.registry",
    "legacy_sources.prior_roots": _MOD + "inventory",
    "legacy_sources.bcb_repair.*": _MOD + "e0.bcb_repair",
}


def consumer_of(leaf_path: str) -> str | None:
    """The most specific ``CONSUMERS`` pattern matching ``leaf_path`` (``*`` = one segment)."""
    parts = leaf_path.split(".")
    best: tuple[int, str] | None = None
    for pattern, consumer in CONSUMERS.items():
        pparts = pattern.split(".")
        if len(pparts) != len(parts):
            continue
        if all(p == "*" or p == q for p, q in zip(pparts, parts)):
            score = sum(1 for p in pparts if p != "*")
            if best is None or score > best[0]:
                best = (score, consumer)
    return None if best is None else best[1]


@dataclass(frozen=True)
class LockPaths:
    models: Path
    datasets: Path
    prompts: Path


def lock_paths(config_path: str | Path) -> LockPaths:
    """The lock files live beside ``study.yaml`` (spec 17.1 layout)."""
    base = Path(config_path).resolve().parent
    return LockPaths(models=base / "models.lock.json", datasets=base / "datasets.lock.json",
                     prompts=base / "prompts.lock.json")


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


__all__ = [
    "AnalysisConfig", "AnalysisProfile", "CONFIG_DIR", "CONFIRMATION_REQUIRED", "CONSUMERS",
    "CellBlock", "CellsBlock", "ComputeConfig", "DEFAULT_ANALYSIS_CONFIG", "DEFAULT_COMPUTE_CONFIG",
    "DEFAULT_PROTOCOLS_CONFIG", "DEFAULT_STUDY_CONFIG", "DispatchBlock", "E2B_ENDORSEMENT_SUPPORT",
    "GatesBlock", "LockPaths", "ModelBlock", "PartitionBlock", "PilotBlock", "PoolBlock", "REPO_ROOT",
    "ReferenceShape", "SCRATCH_PREFIX", "StudyConfig", "WorkerBlock", "build", "compile_checks",
    "consumer_of", "contract_of", "e2b_grid_checks", "is_sha256", "leaf_paths", "load_analysis",
    "load_compute", "load_study", "lock_hash_checks", "lock_paths", "model_cells", "scratch_paths",
    "sibling_config", "validate_compute", "validate_study",
]
