#!/usr/bin/env python3
"""Read-only, exact context-capacity audit for routed sweep cells.

By default the audit retains its focused ``32B-long`` selection.  The opt-in
``--all-routed-profiles`` mode instead checks every filtered manifest cell against the
profile selected by ``serving_profile_for_cell(cell)``.  In either mode it reconstructs
the largest runtime peer message, renders every actual benchmark question through the
same Agent/user-message and Hugging Face chat-template paths used by inference.  Under
protocol v4, the treatment-specific generation allowance is a minimum output-capacity
floor, while the one submitted request uses every safely available output token:
``max_tokens = served context - prompt - 128``.  The audit rejects a request only when
that exact remaining capacity is below its floor.  It prints one JSON report to stdout
and never modifies the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.agents.base_agent import (  # noqa: E402
    AgentOutput,
    render_agent_user_prompt,
    requested_generation_tokens,
)
from agents_scaling.agents.message_builder import (  # noqa: E402
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_COT_CHAR_LIMIT,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
    render_peer_context,
)
from agents_scaling.benchmarks.loaders import load_benchmark  # noqa: E402
from agents_scaling.benchmarks.schema import Question  # noqa: E402
from agents_scaling.config import (  # noqa: E402
    ContextShareLevel,
    DEFAULT_HF_HOME,
    DEFAULT_RESULTS_ROOT,
    ExperimentCell,
    ReasoningLevel,
    Topology,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest  # noqa: E402
from agents_scaling.prompts.system_prompts import get_prompt  # noqa: E402
from agents_scaling.serving.context import (  # noqa: E402
    CONTEXT_RESERVE_TOKENS,
    ContextCapacityError,
    build_chat_messages,
    preflight_chat_context,
    tokenizer_for_profile,
)
from agents_scaling.serving.client import (  # noqa: E402
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    UNLIMITED_THINKING_TOKEN_ALLOWANCE,
)
from agents_scaling.serving.profiles import (  # noqa: E402
    LONG_32B_PROFILE,
    get_serving_profile,
    serving_profile_for_cell,
)


AUDIT_SCHEMA_VERSION = 3
# Repeated digits are a deterministic high-token-density ASCII boundary fixture for
# Qwen3, unlike whitespace words that can substantially underestimate token density.
SYNTHETIC_COT_UNIT = "0123456789"
SYNTHETIC_INTERMEDIATE_UNIT = "9876543210"
SYNTHETIC_INTERMEDIATE = (SYNTHETIC_INTERMEDIATE_UNIT * 401)[:4000]
SYNTHETIC_FINAL_TEXT = "Final answer: A"


class AuditError(RuntimeError):
    """The audit input is invalid or cannot be inspected safely."""


class CacheOnlyBenchmarkLoader:
    """Use the canonical cache-compatible loader while recording audit provenance."""

    def __init__(self) -> None:
        self.diagnostics: list[dict[str, Any]] = []

    def __call__(self, name: str, *, n: int | None, seed: int) -> list[Question]:
        return load_benchmark(
            name,
            n=n,
            seed=seed,
            fallback_diagnostics=self.diagnostics,
        )


@dataclass(frozen=True)
class AuditFilters:
    n_agents: frozenset[int] = frozenset()
    reasoning: frozenset[str] = frozenset()
    prompt_levels: frozenset[int] = frozenset()
    topologies: frozenset[str] = frozenset()
    context_levels: frozenset[str] = frozenset()

    def matches(self, cell: ExperimentCell) -> bool:
        return not any(
            (
                self.n_agents and cell.n_agents not in self.n_agents,
                self.reasoning and cell.reasoning_level.value not in self.reasoning,
                self.prompt_levels
                and cell.prompt_complexity_level not in self.prompt_levels,
                self.topologies and cell.topology.value not in self.topologies,
                self.context_levels
                and cell.context_share_level.value not in self.context_levels,
            )
        )

    def to_dict(self) -> dict[str, list[int] | list[str]]:
        return {
            "n_agents": sorted(self.n_agents),
            "reasoning": sorted(self.reasoning),
            "prompt_levels": sorted(self.prompt_levels),
            "topologies": sorted(self.topologies),
            "context_levels": sorted(self.context_levels),
        }


def _exact_length_text(unit: str, length: int) -> str:
    if not unit:
        raise ValueError("synthetic text unit must be non-empty")
    return (unit * (length // len(unit) + 1))[:length]


def synthetic_peer_outputs(count: int) -> list[AgentOutput]:
    """Construct deterministic, fully populated peer outputs at the registered CoT cap."""
    if count < 0:
        raise ValueError("peer count must be non-negative")
    cot = _exact_length_text(SYNTHETIC_COT_UNIT, PEER_COT_CHAR_LIMIT)
    assert len(cot) == PEER_COT_CHAR_LIMIT
    return [
        AgentOutput(
            agent_id=f"agent{index + 1}",
            round=0,
            answer_choice="A",
            raw_text=SYNTHETIC_FINAL_TEXT,
            cot_text=cot,
            intermediate_results=SYNTHETIC_INTERMEDIATE,
            verbalized_conf=1.0,
        )
        for index in range(count)
    ]


def maximum_peer_count(cell: ExperimentCell) -> int:
    """Largest peer list passed to one runtime request for the configured topology."""
    if cell.topology is Topology.DECENTRALIZED:
        return max(0, cell.n_agents - 1) if cell.rounds > 1 else 0
    if cell.topology is Topology.CENTRALIZED:
        # The orchestrator sees every sub-agent even in round zero.  Later sub-agents see
        # one orchestrator synthesis, which is no larger for every routed MAS cell.
        return max(0, cell.n_agents - 1)
    return 0


def selected_cells(
    snapshot: ManifestSnapshot,
    filters: AuditFilters,
    *,
    all_routed_profiles: bool = False,
) -> list[ExperimentCell]:
    return [
        cell
        for cell in snapshot.cells
        if filters.matches(cell)
        and (
            all_routed_profiles
            or serving_profile_for_cell(cell).registry_key == LONG_32B_PROFILE
        )
    ]


def selected_long_cells(
    snapshot: ManifestSnapshot, filters: AuditFilters
) -> list[ExperimentCell]:
    """Backward-compatible focused selector used by the default audit mode."""
    return selected_cells(snapshot, filters, all_routed_profiles=False)


PromptTokenCache = dict[tuple[str, str, bool], int]


@dataclass(frozen=True)
class RequestCapacityAudit:
    """Exact protocol-v4 capacity accounting for one rendered chat request.

    ``output_capacity_floor_tokens`` is the minimum capacity needed to preserve the
    configured reasoning treatment.  ``requested_output_tokens`` is the full remaining
    safe context envelope that the runtime submits when the floor fits.  It can be
    negative in a rejected audit record when the prompt plus reserve alone exceeds the
    served window; no HTTP request is submitted in that case.
    """

    profile_name: str
    prompt_tokens: int
    output_capacity_floor_tokens: int
    reserve_tokens: int
    served_context: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.output_capacity_floor_tokens < 0:
            raise ValueError("output_capacity_floor_tokens must be non-negative")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if self.served_context <= 0:
            raise ValueError("served_context must be positive")

    @property
    def requested_output_tokens(self) -> int:
        return self.served_context - self.prompt_tokens - self.reserve_tokens

    @property
    def floor_required_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.output_capacity_floor_tokens
            + self.reserve_tokens
        )

    @property
    def required_tokens(self) -> int:
        """Context occupied by the full dynamic envelope (exactly the served limit)."""
        return self.prompt_tokens + self.requested_output_tokens + self.reserve_tokens

    @property
    def output_capacity_headroom_tokens(self) -> int:
        """Available output capacity beyond the treatment-specific minimum."""
        return self.requested_output_tokens - self.output_capacity_floor_tokens

    @property
    def fits(self) -> bool:
        return self.output_capacity_headroom_tokens >= 0

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "serving_profile": self.profile_name,
            "prompt_tokens": self.prompt_tokens,
            "output_capacity_floor_tokens": self.output_capacity_floor_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "available_output_tokens": self.requested_output_tokens,
            "context_reserve_tokens": self.reserve_tokens,
            "floor_required_context_tokens": self.floor_required_tokens,
            "required_context_tokens": self.required_tokens,
            "effective_context_limit": self.served_context,
            "context_preflight_fits": self.fits,
            "output_capacity_headroom_tokens": self.output_capacity_headroom_tokens,
            # Retain the v2 name and meaning: spare capacity after satisfying the
            # minimum output requirement, not the (always-zero) full-envelope margin.
            "context_headroom_tokens": self.output_capacity_headroom_tokens,
            "full_envelope_context_headroom_tokens": (
                self.served_context - self.required_tokens
            ),
        }


def _rendered_prompt_digest(system: str, user: str) -> str:
    """Hash exact message contents without retaining every large peer prompt in RAM."""
    digest = hashlib.sha256()
    for text_value in (system, user):
        encoded = text_value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _preflight_request(
    *,
    tokenizer: Any,
    profile_name: str,
    served_context: int,
    system: str,
    user: str,
    output_capacity_floor_tokens: int,
    enable_thinking: bool,
    prompt_token_cache: PromptTokenCache | None = None,
    cache_stats: dict[str, int] | None = None,
) -> RequestCapacityAudit:
    cache_key = (profile_name, _rendered_prompt_digest(system, user), enable_thinking)
    if prompt_token_cache is not None and cache_key in prompt_token_cache:
        if cache_stats is not None:
            cache_stats["hits"] = cache_stats.get("hits", 0) + 1
        return RequestCapacityAudit(
            profile_name=profile_name,
            prompt_tokens=prompt_token_cache[cache_key],
            output_capacity_floor_tokens=output_capacity_floor_tokens,
            reserve_tokens=CONTEXT_RESERVE_TOKENS,
            served_context=served_context,
        )
    try:
        preflight = preflight_chat_context(
            tokenizer,
            build_chat_messages(system, user),
            requested_output_tokens=output_capacity_floor_tokens,
            served_context=served_context,
            profile_name=profile_name,
            enable_thinking=enable_thinking,
            reserve_tokens=CONTEXT_RESERVE_TOKENS,
        )
    except ContextCapacityError as exc:
        preflight = exc.preflight
    if prompt_token_cache is not None:
        prompt_token_cache[cache_key] = preflight.prompt_tokens
        if cache_stats is not None:
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
    return RequestCapacityAudit(
        profile_name=profile_name,
        prompt_tokens=preflight.prompt_tokens,
        output_capacity_floor_tokens=output_capacity_floor_tokens,
        reserve_tokens=CONTEXT_RESERVE_TOKENS,
        served_context=served_context,
    )


def _request_record(
    *,
    cell: ExperimentCell,
    question: Question,
    peer_count: int,
    preflight: RequestCapacityAudit,
) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "qid": question.qid,
        "benchmark": cell.benchmark,
        "model_size": cell.model_size,
        "seed": cell.seed,
        "n_agents": cell.n_agents,
        "topology": cell.topology.value,
        "reasoning_level": cell.reasoning_level.value,
        "prompt_complexity_level": cell.prompt_complexity_level,
        "peer_count": peer_count,
        **preflight.to_dict(),
        "prompt_plus_reserve_tokens": (
            preflight.prompt_tokens + preflight.reserve_tokens
        ),
    }


def _larger_request(
    current: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if current is None:
        return dict(candidate)
    current_key = (int(current[field]), str(current["cell_id"]), str(current["qid"]))
    candidate_key = (int(candidate[field]), str(candidate["cell_id"]), str(candidate["qid"]))
    return dict(candidate) if candidate_key > current_key else dict(current)


def _smaller_request(
    current: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if current is None:
        return dict(candidate)
    current_key = (int(current[field]), str(current["cell_id"]), str(current["qid"]))
    candidate_key = (int(candidate[field]), str(candidate["cell_id"]), str(candidate["qid"]))
    return dict(candidate) if candidate_key < current_key else dict(current)


def _group_cell_reports(cell_reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate capacity evidence by scientific model, route, and reasoning rung."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    failed_cell_ids: dict[tuple[str, str, str], set[str]] = {}
    for cell in cell_reports:
        key = (
            str(cell["model_size"]),
            str(cell["serving_profile"]),
            str(cell["reasoning_level"]),
        )
        if key not in grouped:
            grouped[key] = {
                "model_size": key[0],
                "serving_profile": key[1],
                "reasoning_level": key[2],
                "effective_context_limit": int(cell["effective_context_limit"]),
                "selected_cells": 0,
                "audited_requests": 0,
                "failed_requests": 0,
                "maximum_prompt_tokens": 0,
                "maximum_output_capacity_floor_tokens": 0,
                "minimum_requested_output_tokens": None,
                "maximum_requested_output_tokens": None,
                "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
                "maximum_floor_required_context_tokens": 0,
                "maximum_required_context_tokens": 0,
                "maximum_full_envelope_context_tokens": 0,
                "minimum_output_capacity_headroom_tokens": None,
                "minimum_context_headroom_tokens": None,
                "maximum_prompt_request": None,
                "maximum_required_request": None,
                "maximum_floor_required_request": None,
                "minimum_headroom_request": None,
            }
            failed_cell_ids[key] = set()
        group = grouped[key]
        group["selected_cells"] += 1
        group["audited_requests"] += int(cell["audited_requests"])
        group["failed_requests"] += int(cell["failed_requests"])
        group["maximum_output_capacity_floor_tokens"] = max(
            int(group["maximum_output_capacity_floor_tokens"]),
            int(cell["output_capacity_floor_tokens"]),
        )
        prior_min_requested = group["minimum_requested_output_tokens"]
        group["minimum_requested_output_tokens"] = (
            int(cell["minimum_requested_output_tokens"])
            if prior_min_requested is None
            else min(
                int(prior_min_requested),
                int(cell["minimum_requested_output_tokens"]),
            )
        )
        prior_max_requested = group["maximum_requested_output_tokens"]
        group["maximum_requested_output_tokens"] = (
            int(cell["maximum_requested_output_tokens"])
            if prior_max_requested is None
            else max(
                int(prior_max_requested),
                int(cell["maximum_requested_output_tokens"]),
            )
        )
        max_prompt = cell["maximum_prompt_request"]
        max_required = cell["maximum_required_request"]
        min_headroom = cell["minimum_headroom_request"]
        group["maximum_prompt_request"] = _larger_request(
            group["maximum_prompt_request"], max_prompt, field="prompt_tokens"
        )
        group["maximum_required_request"] = _larger_request(
            group["maximum_required_request"],
            max_required,
            field="floor_required_context_tokens",
        )
        group["maximum_floor_required_request"] = dict(
            group["maximum_required_request"]
        )
        group["minimum_headroom_request"] = _smaller_request(
            group["minimum_headroom_request"],
            min_headroom,
            field="output_capacity_headroom_tokens",
        )
        group["maximum_prompt_tokens"] = int(
            group["maximum_prompt_request"]["prompt_tokens"]
        )
        group["maximum_floor_required_context_tokens"] = int(
            group["maximum_required_request"]["floor_required_context_tokens"]
        )
        group["maximum_required_context_tokens"] = max(
            int(group["maximum_required_context_tokens"]),
            int(max_required["required_context_tokens"]),
        )
        group["maximum_full_envelope_context_tokens"] = int(
            group["maximum_required_context_tokens"]
        )
        headroom = int(min_headroom["output_capacity_headroom_tokens"])
        previous_headroom = group["minimum_context_headroom_tokens"]
        group["minimum_context_headroom_tokens"] = (
            headroom
            if previous_headroom is None
            else min(int(previous_headroom), headroom)
        )
        group["minimum_output_capacity_headroom_tokens"] = group[
            "minimum_context_headroom_tokens"
        ]
        if int(cell["failed_requests"]):
            failed_cell_ids[key].add(str(cell["cell_id"]))

    reports: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        group["failed_cells"] = len(failed_cell_ids[key])
        group["passed"] = int(group["failed_requests"]) == 0
        reports.append(group)
    return reports


def audit_snapshot(
    snapshot: ManifestSnapshot,
    *,
    run_id: str,
    filters: AuditFilters = AuditFilters(),
    benchmark_loader: Callable[..., list[Question]] | None = None,
    tokenizer_loader: Callable[[str], Any] | None = None,
    failure_example_limit: int = 100,
    all_routed_profiles: bool = False,
) -> dict[str, Any]:
    """Audit selected cells without reading or writing result artifacts."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    if benchmark_loader is None:
        benchmark_loader = CacheOnlyBenchmarkLoader()
    if tokenizer_loader is None:
        tokenizer_loader = tokenizer_for_profile
    if failure_example_limit < 0:
        raise ValueError("failure_example_limit must be non-negative")
    cells = selected_cells(
        snapshot, filters, all_routed_profiles=all_routed_profiles
    )
    if not cells:
        selection = "all routed profiles" if all_routed_profiles else "32B-long"
        raise AuditError(
            f"no manifest cells match the requested filters in {selection} mode"
        )

    tokenizer_cache: dict[str, Any] = {}
    prompt_token_cache: PromptTokenCache = {}
    prompt_cache_stats = {"hits": 0, "misses": 0}
    question_cache: dict[tuple[str, int | None, int], list[Question]] = {}
    cell_reports: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    failure_count = 0
    audited_requests = 0
    maximum_floor_required: dict[str, Any] | None = None
    maximum_full_envelope: dict[str, Any] | None = None
    maximum_prompt: dict[str, Any] | None = None
    minimum_headroom: dict[str, Any] | None = None
    minimum_requested: dict[str, Any] | None = None
    maximum_requested: dict[str, Any] | None = None

    for cell in cells:
        cell_profile = serving_profile_for_cell(cell)
        if cell_profile.registry_key not in tokenizer_cache:
            tokenizer_cache[cell_profile.registry_key] = tokenizer_loader(
                cell_profile.registry_key
            )
        tokenizer = tokenizer_cache[cell_profile.registry_key]
        contract = (cell.benchmark, cell.n_questions, cell.seed)
        if contract not in question_cache:
            question_cache[contract] = benchmark_loader(
                cell.benchmark, n=cell.n_questions, seed=cell.seed
            )
        questions = question_cache[contract]
        if not questions:
            raise AuditError(
                f"benchmark loader returned no questions for {cell.benchmark} seed={cell.seed}"
            )

        peer_count = maximum_peer_count(cell)
        peers = synthetic_peer_outputs(peer_count)
        rendered_peer_context = render_peer_context(
            peers,
            cell.context_share_level,
            tokenizer=tokenizer,
        )
        peer_context = rendered_peer_context.text
        system = get_prompt(cell.prompt_complexity_level)
        output_capacity_floor = requested_generation_tokens(
            cell.reasoning_level, ANSWER_GENERATION_TOKEN_ALLOWANCE
        )
        cell_worst_required: dict[str, Any] | None = None
        cell_max_prompt: dict[str, Any] | None = None
        cell_min_headroom: dict[str, Any] | None = None
        cell_min_requested: dict[str, Any] | None = None
        cell_max_requested: dict[str, Any] | None = None
        cell_failures = 0

        for question in questions:
            user = render_agent_user_prompt(
                question,
                cell.reasoning_level,
                peer_context=peer_context,
                elicit_cot=True,
            )
            preflight = _preflight_request(
                tokenizer=tokenizer,
                profile_name=cell_profile.name,
                served_context=cell_profile.max_model_len,
                system=system,
                user=user,
                output_capacity_floor_tokens=output_capacity_floor,
                enable_thinking=cell.reasoning_level.enable_thinking,
                prompt_token_cache=prompt_token_cache,
                cache_stats=prompt_cache_stats,
            )
            request = _request_record(
                cell=cell,
                question=question,
                peer_count=peer_count,
                preflight=preflight,
            )
            audited_requests += 1
            cell_worst_required = _larger_request(
                cell_worst_required,
                request,
                field="floor_required_context_tokens",
            )
            cell_max_prompt = _larger_request(
                cell_max_prompt, request, field="prompt_tokens"
            )
            cell_min_headroom = _smaller_request(
                cell_min_headroom,
                request,
                field="output_capacity_headroom_tokens",
            )
            cell_min_requested = _smaller_request(
                cell_min_requested, request, field="requested_output_tokens"
            )
            cell_max_requested = _larger_request(
                cell_max_requested, request, field="requested_output_tokens"
            )
            maximum_floor_required = _larger_request(
                maximum_floor_required,
                request,
                field="floor_required_context_tokens",
            )
            maximum_full_envelope = _larger_request(
                maximum_full_envelope, request, field="required_context_tokens"
            )
            maximum_prompt = _larger_request(
                maximum_prompt, request, field="prompt_tokens"
            )
            minimum_headroom = _smaller_request(
                minimum_headroom,
                request,
                field="output_capacity_headroom_tokens",
            )
            minimum_requested = _smaller_request(
                minimum_requested, request, field="requested_output_tokens"
            )
            maximum_requested = _larger_request(
                maximum_requested, request, field="requested_output_tokens"
            )
            if not preflight.fits:
                failure_count += 1
                cell_failures += 1
                if len(failure_examples) < failure_example_limit:
                    failure_examples.append(request)

        assert (
            cell_worst_required is not None
            and cell_max_prompt is not None
            and cell_min_headroom is not None
            and cell_min_requested is not None
            and cell_max_requested is not None
        )
        cell_reports.append(
            {
                "cell_id": cell.cell_id,
                "config_hash": cell.config_hash(),
                "model_size": cell.model_size,
                "benchmark": cell.benchmark,
                "seed": cell.seed,
                "n_questions": len(questions),
                "n_agents": cell.n_agents,
                "rounds": cell.rounds,
                "topology": cell.topology.value,
                "context_share_level": cell.context_share_level.value,
                "reasoning_level": cell.reasoning_level.value,
                "prompt_complexity_level": cell.prompt_complexity_level,
                "serving_profile": cell_profile.name,
                "effective_context_limit": cell_profile.max_model_len,
                "output_capacity_floor_tokens": output_capacity_floor,
                "minimum_requested_output_tokens": cell_min_requested[
                    "requested_output_tokens"
                ],
                "maximum_requested_output_tokens": cell_max_requested[
                    "requested_output_tokens"
                ],
                "minimum_output_capacity_headroom_tokens": cell_min_headroom[
                    "output_capacity_headroom_tokens"
                ],
                "minimum_context_headroom_tokens": cell_min_headroom[
                    "context_headroom_tokens"
                ],
                "peer_count": peer_count,
                "audited_requests": len(questions),
                "failed_requests": cell_failures,
                "fits": cell_failures == 0,
                "maximum_prompt_request": cell_max_prompt,
                "maximum_required_request": cell_worst_required,
                "maximum_floor_required_request": cell_worst_required,
                "minimum_headroom_request": cell_min_headroom,
                "peer_context_tokens": rendered_peer_context.token_count,
                "peer_block_token_counts": list(
                    rendered_peer_context.block_token_counts
                ),
                "peer_truncation_marker_count": (
                    rendered_peer_context.truncation_marker_count
                ),
            }
        )

    assert (
        maximum_floor_required is not None
        and maximum_full_envelope is not None
        and maximum_prompt is not None
        and minimum_headroom is not None
        and minimum_requested is not None
        and maximum_requested is not None
    )
    groups = _group_cell_reports(cell_reports)
    failure_groups = [group for group in groups if not group["passed"]]
    cot_fixture = _exact_length_text(SYNTHETIC_COT_UNIT, PEER_COT_CHAR_LIMIT)
    audit_name = (
        "all_routed_profiles_context_capacity"
        if all_routed_profiles
        else "routed_32b_long_context_capacity"
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": audit_name,
        "run_id": run_id,
        "manifest": {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "cells": len(snapshot.cells),
        },
        "filters": filters.to_dict(),
        "all_routed_profiles": all_routed_profiles,
        "assumptions": {
            "read_only": True,
            "network_downloads_disabled": True,
            "hf_home": os.environ["HF_HOME"],
            "selection": (
                "all filtered manifest cells evaluated against "
                "serving_profile_for_cell(cell)"
                if all_routed_profiles
                else "serving_profile_for_cell(cell).registry_key == '32B-long'"
            ),
            "runtime_answer_generation_allowance_tokens": (
                ANSWER_GENERATION_TOKEN_ALLOWANCE
            ),
            "output_capacity_floor_formula": (
                f"{ANSWER_GENERATION_TOKEN_ALLOWANCE} when reasoning is off; otherwise "
                f"{ANSWER_GENERATION_TOKEN_ALLOWANCE} + thinking budget; unlimited "
                f"uses the runtime {UNLIMITED_THINKING_TOKEN_ALLOWANCE}-token allowance "
                f"=> {ANSWER_GENERATION_TOKEN_ALLOWANCE + UNLIMITED_THINKING_TOKEN_ALLOWANCE}"
            ),
            "requested_output_formula": (
                "effective context limit - exact rendered prompt tokens - context reserve"
            ),
            "admission_rule": (
                "requested_output_tokens >= output_capacity_floor_tokens"
            ),
            "successful_request_invariant": (
                "prompt_tokens + requested_output_tokens + context_reserve_tokens "
                "== effective_context_limit"
            ),
            "headroom_definition": (
                "requested_output_tokens - output_capacity_floor_tokens; this is "
                "capacity beyond the treatment minimum, while the submitted full "
                "envelope itself has zero context margin"
            ),
            "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
            "chat_template": {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
                "enable_thinking_from_cell": True,
            },
            "tokenizers": [
                {
                    "serving_profile": profile_key,
                    "hf_id": get_serving_profile(profile_key).hf_id,
                    "cache_function": (
                        "agents_scaling.serving.context.tokenizer_for_profile"
                    ),
                    "local_files_only": True,
                    "loaded_class": type(tokenizer_cache[profile_key]).__name__,
                    "is_fast": bool(
                        getattr(tokenizer_cache[profile_key], "is_fast", False)
                    ),
                }
                for profile_key in sorted(tokenizer_cache)
            ],
            "prompt_token_cache": {
                "key": (
                    "serving profile + SHA-256 of exact system/user messages + "
                    "enable_thinking"
                ),
                "unique_rendered_prompts": len(prompt_token_cache),
                **prompt_cache_stats,
            },
            "question_rendering": (
                "render_agent_user_prompt -> render_question(..., confidence=True); "
                "reasoning-off includes the runtime CoT hint"
            ),
            "peer_context": {
                "builder": "agents_scaling.agents.message_builder.build_peer_context",
                "protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
                "protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
                "cot_character_limit_per_peer": PEER_COT_CHAR_LIMIT,
                "rendered_block_token_limit_per_peer": (
                    PEER_RENDERED_BLOCK_TOKEN_LIMIT
                ),
                "cot_fixture_characters": len(cot_fixture),
                "cot_fixture_sha256": hashlib.sha256(cot_fixture.encode()).hexdigest(),
                "cot_fill_pattern": repr(SYNTHETIC_COT_UNIT),
                "cot_fixture_interpretation": (
                    "deterministic high-token-density ASCII boundary fixture, not an "
                    "exhaustive adversarial search over arbitrary Unicode"
                ),
                "intermediate_fixture_characters": len(SYNTHETIC_INTERMEDIATE),
                "intermediate_fixture_sha256": hashlib.sha256(
                    SYNTHETIC_INTERMEDIATE.encode()
                ).hexdigest(),
                "intermediate_fill_pattern": repr(SYNTHETIC_INTERMEDIATE_UNIT),
                "answer_choice": "A",
                "raw_final_text": SYNTHETIC_FINAL_TEXT,
                "verbalized_confidence": 1.0,
                "peer_count_rule": (
                    "decentralized: n_agents-1 only when rounds>1; centralized: "
                    "n_agents-1 orchestrator input"
                ),
            },
            "excluded_request_class": (
                "the MCQ forced-answer completion probe has no peer context and requests "
                "one output token; this audit targets the larger generation chat request"
            ),
            "benchmark_loader_fallbacks": list(
                getattr(benchmark_loader, "diagnostics", [])
            ),
        },
        "summary": {
            "passed": failure_count == 0,
            "selected_cells": len(cells),
            "question_contracts_loaded": len(question_cache),
            "audited_requests": audited_requests,
            "failed_requests": failure_count,
            "failed_cells": sum(not report["fits"] for report in cell_reports),
            "maximum_prompt_tokens": maximum_prompt["prompt_tokens"],
            "maximum_output_capacity_floor_tokens": max(
                report["output_capacity_floor_tokens"] for report in cell_reports
            ),
            "minimum_requested_output_tokens": minimum_requested[
                "requested_output_tokens"
            ],
            "maximum_requested_output_tokens": maximum_requested[
                "requested_output_tokens"
            ],
            "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
            "maximum_floor_required_context_tokens": maximum_floor_required[
                "floor_required_context_tokens"
            ],
            "maximum_required_context_tokens": maximum_full_envelope[
                "required_context_tokens"
            ],
            "maximum_full_envelope_context_tokens": maximum_full_envelope[
                "required_context_tokens"
            ],
            "minimum_output_capacity_headroom_tokens": minimum_headroom[
                "output_capacity_headroom_tokens"
            ],
            "minimum_context_headroom_tokens": minimum_headroom[
                "context_headroom_tokens"
            ],
            "maximum_prompt_request": maximum_prompt,
            "maximum_required_request": maximum_floor_required,
            "maximum_floor_required_request": maximum_floor_required,
            "maximum_full_envelope_request": maximum_full_envelope,
            "minimum_headroom_request": minimum_headroom,
        },
        "groups": groups,
        "failure_groups": failure_groups,
        "failure_examples": failure_examples,
        "cells": cell_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--results-root",
        default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--run-root",
        help="exact run root override; basename must equal --run-id",
    )
    parser.add_argument("--n-agents", type=int, nargs="+")
    parser.add_argument(
        "--reasoning", choices=[level.value for level in ReasoningLevel], nargs="+"
    )
    parser.add_argument("--prompt-level", type=int, choices=range(4), nargs="+")
    parser.add_argument(
        "--topology", choices=[topology.value for topology in Topology], nargs="+"
    )
    parser.add_argument(
        "--context-share-level",
        choices=[level.value for level in ContextShareLevel],
        nargs="+",
    )
    parser.add_argument(
        "--all-routed-profiles",
        action="store_true",
        help=(
            "audit every filtered cell against serving_profile_for_cell(cell); "
            "default audits only cells routed to 32B-long"
        ),
    )
    parser.add_argument("--failure-example-limit", type=int, default=100)
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Both the tokenizer and benchmark loader must remain cache-only.  Set these before
    # datasets performs its lazy import inside load_benchmark.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    try:
        run_root = (
            Path(args.run_root).expanduser().resolve()
            if args.run_root
            else (Path(args.results_root).expanduser().resolve() / args.run_id)
        )
        if run_root.name != args.run_id:
            raise AuditError(
                f"run root basename {run_root.name!r} does not match run id {args.run_id!r}"
            )
        checksum_path = run_root / "cells.sha256"
        if not checksum_path.is_file():
            raise AuditError(
                f"immutable manifest checksum is missing: {checksum_path}; freeze it first"
            )
        snapshot = load_manifest(run_root, verify_frozen=True)
        filters = AuditFilters(
            n_agents=frozenset(args.n_agents or ()),
            reasoning=frozenset(args.reasoning or ()),
            prompt_levels=frozenset(args.prompt_level or ()),
            topologies=frozenset(args.topology or ()),
            context_levels=frozenset(args.context_share_level or ()),
        )
        report = audit_snapshot(
            snapshot,
            run_id=args.run_id,
            filters=filters,
            failure_example_limit=args.failure_example_limit,
            all_routed_profiles=args.all_routed_profiles,
        )
    except Exception as exc:
        error = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit": (
                "all_routed_profiles_context_capacity"
                if args.all_routed_profiles
                else "routed_32b_long_context_capacity"
            ),
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(error, indent=None if args.compact else 2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
