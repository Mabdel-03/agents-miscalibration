"""Cells JSON → the seven spec §18 tables and the plan §5.8 descriptive tables (WP4).

Spec §10 "Required round outputs" (correct-count distribution, selected accuracy, current
and archive coverage, agreement/entropy/validity, correction and destruction probabilities
with their denominators, loss/recovery/censoring, persistence versus dissemination, novel
correct answers, confidence panel), §9 (bank and selection panel), §18 (the seven tables).
Plan §5.8 is the executable restatement this module implements verbatim, including its two
keying rules:

* ``round_summary`` is keyed by **episode_id × round** with ``cohort``, ``constructed_C0``,
  ``team_id``, ``reasoning_allowance`` and ``parent_ref`` as *columns* — an
  item × protocol × checkpoint key would collide for the two challenge teams of one item and
  for the 1,024- and 4,096-token E2d arms of one item.
* selected correctness is **long form over selector × pool**.  ``contract.ALL_TABLES`` is
  frozen, so that long form is written as the documented companion
  ``round_selection.parquet`` beside ``round_summary.parquet`` rather than by re-keying it.

``transitions`` is at agent-round grain and carries CC/CW/WC/WW, the three invalidity
classes and the answer-change flag under **both** definitions (canonical class and report
content hash) — the PEER-versus-PRIVATE contrast of that rate is ARD36's prospective
analogue of the atlas 39.2%/8.5% pair.  The accounting identity
``C_{t+1} - C_t == #wrong→correct - #correct→wrong`` is asserted at compaction with a
residual column that must be zero.

Parquet writing reuses ``study/aggregate.py:257-267 _write_parquet`` verbatim — note that it
writes a ``<name>.empty`` sentinel instead of an empty parquet file, which the export step
must handle (plan §10 "all §18 tables ... present with ``.empty`` sentinels handled").

Labels are joined from the evaluator lane's output only; nothing here reads gold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from agents_scaling.ard import contract as K
from agents_scaling.ard import resources as RES
from agents_scaling.ard.types import ArdCellSpec, DecisionRef, EpisodeRecord
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study import metrics_reference as MR
from agents_scaling.study.aggregate import _write_parquet  # reused verbatim (.empty sentinel)

TABLES_DIR: Final[str] = "tables"
GRADES_DIR: Final[str] = "eval/grades"
#: The long-form companion of ``round_summary`` (see the module docstring).
SELECTION_TABLE: Final[str] = "round_selection"

#: Column set of every table this module can write.  ``contract.ALL_TABLES`` fixes the names
#: of the seven spec tables and the twelve descriptive ones; the tables owned by WP6/WP7
#: (``calibration_panel``, ``neural_*``, ``challenge_teams``, ``suffix_summary``) are declared
#: here so the schema is one artifact, and are written as ``.empty`` sentinels until their
#: owner fills them.
SCHEMAS: Final[Mapping[str, tuple[str, ...]]] = {
    # ---- spec §18 -------------------------------------------------------------------
    "items": ("item_id", "family", "dataset_revision", "source_id", "split", "group_id",
              "prompt_hash", "option_permutation", "gold_location", "grader_hash", "subject",
              "declared_difficulty_metadata"),
    "episodes": ("episode_id", "item_id", "cohort", "protocol", "checkpoint",
                 "checkpoint_revision", "root_candidate_ids", "root_seed_ids", "N", "horizon",
                 "reasoning_allowance", "final_allowance", "packet_policy", "memory_policy",
                 "leader_id", "team_id", "parent_ref", "condition_manifest_hash", "cell_id",
                 "split", "family", "fallback"),
    "decisions": ("decision_id", "episode_id", "parent_state_id", "agent_id", "round", "phase",
                  "action", "checkpoint", "seed", "request_hash", "prompt_token_hash",
                  "completion_token_hash", "root_report_id", "prior_report_id",
                  "seen_message_ids", "message_order", "answer_raw", "answer_class",
                  "evidence_text", "confidence", "parse_status", "thinking_tokens",
                  "final_tokens", "prompt_tokens", "cap_hit", "boundary_injected",
                  "logical_decision_id", "physical_request_ids", "latency", "endpoint_id",
                  "final_overrun", "answer_logprob", "packet_tokens", "n_packets",
                  "cached_prefill_tokens", "flops_prefill", "flops_decode", "flops_total",
                  "flops_oracle", "protocol", "cohort", "protected", "content_sha256",
                  "aliased", "fallback", "arrival_ts", "budget_tokens"),
    "labels": ("decision_id", "is_correct", "evaluator_status", "grader_version",
               "hidden_test_or_reference_hash", "grading_duration", "answer_class", "item_id"),
    "features": ("decision_or_state_id", "feature_time", "last_allowed_round",
                 "feature_schema_version", "behavior_features", "neural_feature_file",
                 "model_revision", "training_eligibility"),
    "forks": ("fork_id", "shared_prefix_hash", "focal_agent", "action", "branch_seed",
              "common_suffix_policy", "suffix_seed", "parent_decision_ids", "child_decision_ids",
              "episode_id", "item_id", "round"),
    "resources": ("timestamp", "job_id", "node", "gpu_uuid", "gpu_type", "memory_total",
                  "allocated_gpu_seconds", "utilization", "memory_used", "power_if_available",
                  "server_batch_id", "physical_request_ids", "prefill_tokens", "decode_tokens",
                  "retries", "errors", "endpoint_id", "attributed_gpu_seconds",
                  "mean_concurrency", "attribution_rule"),
    # ---- plan §5.8 descriptive --------------------------------------------------------
    "round_summary": (
        "episode_id", "round", "item_id", "family", "split", "cohort", "constructed_C0",
        "team_id", "reasoning_allowance", "parent_ref", "protocol", "checkpoint",
        "C_t", "m_t", "delta_C", "delta_m", "n_wrong_to_correct", "n_correct_to_wrong",
        "accounting_residual", "a_correction", "d_destruction", "n_initially_wrong",
        "n_initially_correct", "coverage_current", "coverage_archive", "class_multiset",
        "class_entropy", "n_valid", "n_unique_classes", "n_unparsed", "agreement_fraction",
        "correct_plurality_status", "unanimity_correct", "unanimity_wrong",
        "entered_unanimity", "left_unanimity", "unanimous_wrong_escape",
        "first_correct_loss_round", "recovery_round", "right_censored", "survives_to_round3",
        "persistence_same_agent", "dissemination_new_agents",
        "persistence_without_original_holder", "wrong_minority_persists",
        "wrong_class_disseminated", "wrong_unanimity_reached", "novel_correct_on_c0_zero",
        "new_correct_class_after_loss", "selection_failed", "parse_status_histogram",
        "cap_hit_rate", "boundary_injected_rate", "final_overrun_rate", "thinking_tokens_mean",
        "thinking_tokens_max", "mean_confidence", "conf_missing_rate", "mean_answer_logprob",
        "selected_correct_plurality_current", "fallback"),
    SELECTION_TABLE: ("episode_id", "round", "item_id", "protocol", "checkpoint", "cohort",
                      "selector", "pool", "selected_decision_id", "selected_class",
                      "selected_correct", "expected_correct_under_ties", "tied_winning_classes",
                      "valid_count", "planned_count", "pool_size", "repeated_descendants"),
    "transitions": ("episode_id", "item_id", "agent_id", "round_from", "round_to", "protocol",
                    "checkpoint", "cohort", "class_from", "class_to", "valid_from", "valid_to",
                    "correct_from", "correct_to", "transition", "invalidity_class",
                    "changed_class", "changed_content", "protected", "n_packets",
                    "confidence_from", "confidence_to"),
    "c_dist": ("protocol", "checkpoint", "family", "cohort", "C0", "round", "c", "p",
               "n_episodes"),
    "drift": ("protocol", "checkpoint", "family", "cohort", "round", "m", "n_episodes",
              "a_of_m", "d_of_m", "realized_delta_m", "predicted_delta_m", "residual"),
    "bank_summary": ("item_id", "checkpoint", "family", "split", "n_draws", "c_correct",
                     "class_histogram", "correct_plurality_status", "subset_id",
                     "plurality_correct", "confidence_weighted_correct", "oracle_coverage",
                     "candidate_mean", "selection_gap") + tuple(f"pass_at_{k}" for k in K.PASS_KS),
    "calibration_panel": ("protocol", "checkpoint", "cohort", "round", "signal", "n", "valid_n",
                          "conf_missing_rate", "mean_confidence", "accuracy", "signed_gap",
                          "brier", "brier_reliability", "brier_resolution", "nll", "ece_primary",
                          "ece_debiased_l2", "ece_15bin_equal_width", "mce", "cox_intercept",
                          "cox_slope", "within_item_auc", "eligible_items"),
    "cost_summary": ("episode_id", "round", "protocol", "checkpoint", "cohort", "item_id",
                     "logical_decisions", "prompt_tokens", "prefill_tokens",
                     "cached_prefill_tokens", "thinking_tokens", "final_tokens", "logical_flops",
                     "cumulative_logical_flops", "physical_requests", "attributed_gpu_seconds",
                     "wall_clock_s", "critical_path_s"),
    "cost_accuracy": ("arm", "protocol", "checkpoint", "cohort", "reasoning_allowance", "round",
                      "logical_decisions", "mean_prompt_tokens", "mean_prefill_tokens",
                      "mean_decode_tokens", "mean_logical_flops", "total_logical_flops",
                      "attributed_gpu_seconds", "round3_selected_accuracy", "ci_low", "ci_high",
                      "n_episodes", "decision_matched", "flop_matched"),
    "neural_geometry": ("item_id", "protocol", "round", "layer", "anchor", "participation_ratio",
                        "entropy_rank", "leading_eigenvalue_share", "total_variance",
                        "mean_pairwise", "min_pairwise", "max_pairwise", "cosine_distance",
                        "normalised", "feature_time", "last_allowed_round"),
    "neural_agent": ("item_id", "protocol", "round", "layer", "anchor", "agent_id", "state_norm",
                     "distance_to_peer_centroid", "distance_to_nearest_peer",
                     "own_displacement", "displacement_alignment", "near_zero_denominator",
                     "feature_time", "last_allowed_round"),
    "challenge_teams": ("team_id", "item_id", "checkpoint", "constructed_C0",
                        "eligible_correct_roots", "eligible_incorrect_roots",
                        "wrong_concentration_class", "root_candidate_ids", "overlap_with_sibling"),
    "suffix_summary": ("item_id", "parent_protocol", "suffix_seed", "checkpoint", "J0", "J5",
                       "coverage_before", "coverage_after", "regeneration_after_loss",
                       "archive_selected_correct"),
}

#: Tables whose rows another work package produces; written as ``.empty`` sentinels here.
DEFERRED_TABLES: Final[frozenset[str]] = frozenset({
    "features", "calibration_panel", "neural_geometry", "neural_agent", "challenge_teams",
    "suffix_summary",
})

PLURALITY_STATUSES: Final[tuple[str, ...]] = (
    "strict_majority", "unique_plurality", "tied_plurality", "below_plurality", "none",
)


class CompactionError(ValueError):
    """A compaction the recorded data does not support (e.g. a nonzero accounting residual)."""


# --------------------------------------------------------------------------- inputs


def iter_item_files(run_root: str | os.PathLike, cell_ids: Sequence[str] | None = None) -> Iterator[Path]:
    """Item files of the named cells (or of every cell directory when ``cell_ids`` is None).

    Always pass the manifest's cell ids on a shared filesystem: it turns a directory walk
    into a bounded per-cell listing.
    """
    cells = Path(run_root) / "cells"
    names = list(cell_ids) if cell_ids is not None else sorted(
        p.name for p in cells.iterdir() if p.is_dir()) if cells.exists() else []
    for name in names:
        items = cells / name / "items"
        if not items.exists():
            continue
        for path in sorted(items.glob("*.json")):
            yield path


def load_episodes(
    run_root: str | os.PathLike, cell_ids: Sequence[str] | None = None
) -> list[tuple[ArdCellSpec, str, EpisodeRecord]]:
    """``(cell, item_id, episode)`` triples from the committed item files."""
    out: list[tuple[ArdCellSpec, str, EpisodeRecord]] = []
    for path in iter_item_files(run_root, cell_ids):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = ArdCellSpec.from_dict(payload["cell"])
        for entry in payload.get("episodes") or []:
            out.append((cell, payload["source_id"], EpisodeRecord.from_dict(entry)))
    return out


def load_labels(run_root: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """``{decision_id: label row}`` from the evaluator lane (never from gold)."""
    labels: dict[str, dict[str, Any]] = {}
    grades = Path(run_root) / GRADES_DIR
    if not grades.exists():
        return labels
    for cell_dir in sorted(p for p in grades.iterdir() if p.is_dir()):
        for path in sorted(cell_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("rows") or []:
                decision_id = row.get("decision_id")
                if decision_id:
                    labels[str(decision_id)] = dict(row)
    return labels


def _is_correct(labels: Mapping[str, Mapping[str, Any]], decision: DecisionRef) -> bool | None:
    row = labels.get(decision.decision_id)
    if row is None:
        return None
    value = row.get("is_correct")
    return None if value is None else bool(value)


# --------------------------------------------------------------------------- spec tables


def items_rows(run_root: str | os.PathLike) -> list[dict[str, Any]]:
    from agents_scaling.ard.items import load_ard_tasks

    try:
        tasks = load_ard_tasks(run_root)
    except (FileNotFoundError, OSError):
        return []
    return [{key: getattr(task, key) for key in SCHEMAS["items"]} for task in tasks.values()]


def episode_rows(triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
                 families: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for cell, item_id, episode in triples:
        rows.append({
            "episode_id": episode.episode_id, "item_id": item_id, "cohort": episode.cohort,
            "protocol": episode.protocol, "checkpoint": episode.checkpoint,
            "checkpoint_revision": episode.checkpoint_revision,
            "root_candidate_ids": list(episode.root_candidate_ids),
            "root_seed_ids": list(episode.root_seed_ids), "N": episode.N,
            "horizon": episode.horizon, "reasoning_allowance": episode.reasoning_allowance,
            "final_allowance": episode.final_allowance, "packet_policy": episode.packet_policy,
            "memory_policy": episode.memory_policy, "leader_id": episode.leader_id,
            "team_id": episode.team_id, "parent_ref": episode.parent_ref,
            "condition_manifest_hash": episode.condition_manifest_hash,
            "cell_id": cell.cell_id, "split": cell.split,
            "family": families.get(item_id, ""), "fallback": _is_fallback(episode),
        })
    return rows


def _is_fallback(episode: EpisodeRecord) -> bool:
    return str(episode.packet_policy).startswith("fallback:") or any(
        decision.fallback for decision in episode.decisions)


def _content_hash(decision: DecisionRef) -> str:
    if decision.content_sha256:
        return str(decision.content_sha256)
    return identity.sha256_hex(json.dumps(
        [decision.answer_raw, decision.evidence_text], sort_keys=True, ensure_ascii=False))


def decision_rows(triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]]) -> list[dict[str, Any]]:
    rows = []
    for cell, _item_id, episode in triples:
        for decision in episode.decisions:
            flops = decision.flops or {}
            rows.append({
                "decision_id": decision.decision_id, "episode_id": decision.episode_id,
                "parent_state_id": decision.parent_state_id, "agent_id": decision.agent_id,
                "round": decision.round, "phase": decision.phase, "action": decision.phase,
                "checkpoint": decision.checkpoint, "seed": None,
                "request_hash": decision.request_id,
                "prompt_token_hash": None, "completion_token_hash": None,
                "root_report_id": decision.root_report_id,
                "prior_report_id": decision.prior_report_id,
                "seen_message_ids": list(decision.seen_message_ids),
                "message_order": list(decision.message_order),
                "answer_raw": decision.answer_raw, "answer_class": decision.answer_class,
                "evidence_text": decision.evidence_text, "confidence": decision.confidence,
                "parse_status": decision.parse_status,
                "thinking_tokens": decision.thinking_tokens, "final_tokens": decision.final_tokens,
                "prompt_tokens": decision.prompt_tokens, "cap_hit": decision.cap_hit,
                "boundary_injected": decision.boundary_injected,
                "logical_decision_id": decision.decision_id,
                "physical_request_ids": list(decision.physical_request_ids
                                             or (decision.request_id,)),
                "latency": decision.latency_s,
                "endpoint_id": decision.endpoint_id, "final_overrun": decision.final_overrun,
                "answer_logprob": decision.answer_logprob,
                "packet_tokens": decision.packet_tokens, "n_packets": decision.n_packets,
                "cached_prefill_tokens": decision.cached_prefill_tokens,
                "flops_prefill": flops.get("prefill"), "flops_decode": flops.get("decode"),
                "flops_total": flops.get("total"), "flops_oracle": flops.get("oracle"),
                "protocol": decision.protocol, "cohort": episode.cohort,
                "protected": decision.protected, "content_sha256": _content_hash(decision),
                "aliased": decision.aliased, "fallback": decision.fallback,
                "arrival_ts": decision.arrival_ts, "budget_tokens": decision.budget_tokens,
            })
    return rows


def label_rows(labels: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for decision_id, row in labels.items():
        rows.append({
            "decision_id": decision_id, "is_correct": row.get("is_correct"),
            "evaluator_status": row.get("evaluator_status"),
            "grader_version": row.get("grader_version"),
            "hidden_test_or_reference_hash": row.get("hidden_test_or_reference_hash"),
            "grading_duration": row.get("grading_duration"),
            "answer_class": row.get("answer_class"), "item_id": row.get("item_id"),
        })
    return rows


def fork_rows(triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]]) -> list[dict[str, Any]]:
    """Fork and suffix branches (spec §18 ``forks.parquet``)."""
    from agents_scaling.ard.types import ArdCellKind

    rows = []
    for cell, item_id, episode in triples:
        if cell.kind not in (ArdCellKind.FORK, ArdCellKind.SUFFIX):
            continue
        for decision in episode.decisions:
            if decision.round == 0:
                continue
            rows.append({
                "fork_id": f"{episode.episode_id}|a{decision.agent_id}|r{decision.round}",
                "shared_prefix_hash": decision.parent_state_id or episode.condition_manifest_hash,
                "focal_agent": decision.agent_id, "action": decision.phase,
                "branch_seed": cell.episode_rep,
                "common_suffix_policy": cell.options.get("parent_protocol"),
                "suffix_seed": cell.options.get("suffix_seed"),
                "parent_decision_ids": [decision.parent_decision_id],
                "child_decision_ids": [decision.decision_id],
                "episode_id": episode.episode_id, "item_id": item_id, "round": decision.round,
            })
    return rows


def resource_rows(
    decisions: Sequence[Mapping[str, Any]],
    *,
    gpu_samples: Sequence[Mapping[str, Any]] = (),
    sacct_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Per-request attribution rows plus the retained hardware samples (spec §18)."""
    attributed, totals = RES.attribute_gpu_seconds(RES.request_resources_from_decisions(decisions))
    allocated = {str(row.get("node")): row.get("allocated_gpu_seconds") for row in sacct_rows}
    rows: list[dict[str, Any]] = []
    for row in attributed:
        rows.append({
            "timestamp": row["timestamp"], "job_id": None, "node": None, "gpu_uuid": None,
            "gpu_type": None, "memory_total": None, "allocated_gpu_seconds": None,
            "utilization": None, "memory_used": None, "power_if_available": None,
            "server_batch_id": row["server_batch_id"],
            "physical_request_ids": [row["request_id"]],
            "prefill_tokens": row["prefill_tokens"], "decode_tokens": row["decode_tokens"],
            "retries": row["retries"], "errors": row["errors"],
            "endpoint_id": row["endpoint_id"],
            "attributed_gpu_seconds": row["attributed_gpu_seconds"],
            "mean_concurrency": row["mean_concurrency"],
            "attribution_rule": row["attribution_rule"],
        })
    for sample in gpu_samples:
        rows.append({
            "timestamp": sample.get("timestamp"), "job_id": sample.get("job_id"),
            "node": sample.get("node"), "gpu_uuid": sample.get("gpu_uuid"),
            "gpu_type": sample.get("gpu_type"), "memory_total": sample.get("memory_total"),
            "allocated_gpu_seconds": allocated.get(str(sample.get("node"))),
            "utilization": sample.get("utilization"), "memory_used": sample.get("memory_used"),
            "power_if_available": sample.get("power_if_available"), "server_batch_id": None,
            "physical_request_ids": [], "prefill_tokens": None, "decode_tokens": None,
            "retries": None, "errors": None, "endpoint_id": None,
            "attributed_gpu_seconds": None, "mean_concurrency": None,
            "attribution_rule": RES.ATTRIBUTION_RULE,
        })
    if totals:
        io_totals = {"server_totals": totals}
        rows.append({**{key: None for key in SCHEMAS["resources"]},
                     "timestamp": None, "server_batch_id": "SERVER_TOTALS",
                     "physical_request_ids": [], "attribution_rule": json.dumps(io_totals)})
    return rows


# --------------------------------------------------------------------------- round panel


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -math.fsum((n / total) * math.log(n / total) for n in counts if n > 0)


def _plurality_status(classes: Sequence[str | None], correct_classes: set[str]) -> str:
    valid = [c for c in classes if c is not None]
    if not valid or not correct_classes:
        return "none"
    counts = Counter(valid)
    best = max(counts.values())
    winners = [cls for cls, n in counts.items() if n == best]
    correct_best = max((counts[c] for c in counts if c in correct_classes), default=0)
    if correct_best == 0:
        return "none"
    if correct_best > len(valid) / 2:
        return "strict_majority"
    if correct_best == best and len(winners) == 1:
        return "unique_plurality"
    if correct_best == best:
        return "tied_plurality"
    return "below_plurality"


def _round_decisions(episode: EpisodeRecord) -> dict[int, list[DecisionRef]]:
    by_round: dict[int, list[DecisionRef]] = defaultdict(list)
    for decision in episode.decisions:
        by_round[int(decision.round)].append(decision)
    for refs in by_round.values():
        refs.sort(key=lambda d: d.agent_id)
    return dict(by_round)


def round_summary_rows(
    triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    families: Mapping[str, str],
    challenge_c0: Mapping[str, int] | None = None,
    strict_residual: bool = True,
) -> list[dict[str, Any]]:
    """The plan §5.8 round-wise outcome and dynamics panel, keyed episode_id × round."""
    rows: list[dict[str, Any]] = []
    challenge_c0 = challenge_c0 or {}
    for cell, item_id, episode in triples:
        by_round = _round_decisions(episode)
        rounds = sorted(by_round)
        if not rounds:
            continue
        correct: dict[int, list[bool | None]] = {
            r: [_is_correct(labels, d) for d in by_round[r]] for r in rounds}
        classes: dict[int, list[str | None]] = {
            r: [d.answer_class if d.valid else None for d in by_round[r]] for r in rounds}
        correct_classes: set[str] = set()
        for r in rounds:
            for decision, is_ok in zip(by_round[r], correct[r]):
                if is_ok and decision.answer_class:
                    correct_classes.add(decision.answer_class)
        archive: set[str] = set()
        first_loss: int | None = None
        recovery: int | None = None
        c_prev: int | None = None
        root_holders: set[int] = set()
        prev_unanimous: str | None = None
        cumulative_flops = 0
        for index, round_ in enumerate(rounds):
            refs = by_round[round_]
            n = len(refs)
            round_correct = correct[round_]
            c_t = sum(1 for value in round_correct if value)
            m_t = c_t / n if n else 0.0
            holders = {ref.agent_id for ref, ok in zip(refs, round_correct) if ok}
            if round_ == 0:
                root_holders = set(holders)
            covered = c_t > 0
            archive |= {cls for cls, ok in zip(classes[round_], round_correct) if ok and cls}
            valid_classes = [cls for cls in classes[round_] if cls is not None]
            counts = Counter(valid_classes)
            multiset = dict(sorted(counts.items()))
            wtc = ctw = None
            residual = None
            a_correction = d_destruction = None
            n_initially_wrong = n_initially_correct = None
            if index > 0:
                previous = correct[rounds[index - 1]]
                # Pair by agent id, never by position: a lost decision must show up as an
                # uncounted transition (a nonzero residual), not as a silently shifted pair.
                before_by_agent = {ref.agent_id: value for ref, value
                                   in zip(by_round[rounds[index - 1]], previous)}
                after_by_agent = {ref.agent_id: value for ref, value in zip(refs, round_correct)}
                pairs = [(before_by_agent[agent], after_by_agent[agent])
                         for agent in sorted(set(before_by_agent) & set(after_by_agent))]
                wtc = sum(1 for before, after in pairs if before is False and after is True)
                ctw = sum(1 for before, after in pairs if before is True and after is False)
                n_initially_wrong = sum(1 for before in previous if before is False)
                n_initially_correct = sum(1 for before in previous if before is True)
                a_correction = wtc / n_initially_wrong if n_initially_wrong else None
                d_destruction = ctw / n_initially_correct if n_initially_correct else None
                # The identity is only *computable* once both rounds are fully graded; a
                # partially graded run leaves the residual null rather than crashing the
                # compaction, and the hour-30 pass (grading closed) is where it must be zero.
                fully_labelled = all(value is not None for value in previous + round_correct)
                residual = ((c_t - (c_prev or 0)) - (wtc - ctw)) if fully_labelled else None
                if strict_residual and residual:
                    raise CompactionError(
                        f"{episode.episode_id} round {round_}: C_t - C_(t-1) = {c_t - (c_prev or 0)} "
                        f"but #wrong→correct - #correct→wrong = {wtc - ctw} (residual {residual})")
            if covered is False and first_loss is None and index > 0 and (c_prev or 0) > 0:
                first_loss = round_
            if first_loss is not None and recovery is None and c_t > 0 and round_ > first_loss:
                recovery = round_
            unanimous_class = valid_classes[0] if valid_classes and len(counts) == 1 and len(valid_classes) == n else None
            unanimity_correct = bool(unanimous_class and unanimous_class in correct_classes)
            unanimity_wrong = bool(unanimous_class and unanimous_class not in correct_classes)
            entered = bool(unanimous_class and prev_unanimous is None)
            left = bool(prev_unanimous is not None and unanimous_class is None)
            escape = bool(prev_unanimous is not None and prev_unanimous not in correct_classes
                          and unanimous_class != prev_unanimous)
            n_unparsed = sum(1 for ref in refs if not ref.valid)
            confidences = [ref.confidence for ref in refs if ref.confidence is not None]
            logprobs = [ref.answer_logprob for ref in refs if ref.answer_logprob is not None]
            thinking = [ref.thinking_tokens for ref in refs]
            cumulative_flops += sum(int((ref.flops or {}).get("total") or 0) for ref in refs)
            wrong_classes = [cls for cls in valid_classes if cls not in correct_classes]
            rows.append({
                "episode_id": episode.episode_id, "round": round_, "item_id": item_id,
                "family": families.get(item_id, ""), "split": cell.split,
                "cohort": episode.cohort,
                "constructed_C0": challenge_c0.get(str(episode.team_id or "")),
                "team_id": episode.team_id,
                "reasoning_allowance": episode.reasoning_allowance,
                "parent_ref": episode.parent_ref, "protocol": episode.protocol,
                "checkpoint": episode.checkpoint,
                "C_t": c_t, "m_t": m_t,
                "delta_C": None if index == 0 else c_t - (c_prev or 0),
                "delta_m": None if index == 0 else (c_t - (c_prev or 0)) / n,
                "n_wrong_to_correct": wtc, "n_correct_to_wrong": ctw,
                "accounting_residual": residual, "a_correction": a_correction,
                "d_destruction": d_destruction, "n_initially_wrong": n_initially_wrong,
                "n_initially_correct": n_initially_correct,
                "coverage_current": 1.0 if covered else 0.0,
                "coverage_archive": 1.0 if archive else 0.0,
                "class_multiset": json.dumps(multiset, sort_keys=True),
                "class_entropy": _entropy(list(counts.values())),
                "n_valid": len(valid_classes), "n_unique_classes": len(counts),
                "n_unparsed": n_unparsed,
                "agreement_fraction": (max(counts.values()) / n) if counts and n else 0.0,
                "correct_plurality_status": _plurality_status(classes[round_], correct_classes),
                "unanimity_correct": unanimity_correct, "unanimity_wrong": unanimity_wrong,
                "entered_unanimity": entered, "left_unanimity": left,
                "unanimous_wrong_escape": escape,
                "first_correct_loss_round": first_loss, "recovery_round": recovery,
                "right_censored": round_ == rounds[-1] and first_loss is not None and recovery is None,
                "survives_to_round3": c_t > 0 and round_ == K.UPDATE_ROUNDS,
                "persistence_same_agent": len(holders & root_holders),
                "dissemination_new_agents": len(holders - root_holders),
                "persistence_without_original_holder": bool(holders and not (holders & root_holders)),
                "wrong_minority_persists": bool(wrong_classes) and len(wrong_classes) < n,
                "wrong_class_disseminated": bool(index > 0 and wrong_classes and
                                                 len(wrong_classes) > len([
                                                     cls for cls in classes[rounds[index - 1]]
                                                     if cls is not None and cls not in correct_classes])),
                "wrong_unanimity_reached": unanimity_wrong,
                "novel_correct_on_c0_zero": bool(c_t > 0 and correct[rounds[0]].count(True) == 0),
                "new_correct_class_after_loss": bool(recovery == round_),
                "selection_failed": not valid_classes,
                "parse_status_histogram": json.dumps(
                    dict(sorted(Counter(ref.parse_status for ref in refs).items()))),
                "cap_hit_rate": sum(1 for ref in refs if ref.cap_hit) / n if n else 0.0,
                "boundary_injected_rate": sum(1 for ref in refs if ref.boundary_injected) / n if n else 0.0,
                "final_overrun_rate": sum(1 for ref in refs if ref.final_overrun) / n if n else 0.0,
                "thinking_tokens_mean": (math.fsum(thinking) / len(thinking)) if thinking else 0.0,
                "thinking_tokens_max": max(thinking) if thinking else 0,
                "mean_confidence": (math.fsum(confidences) / len(confidences)) if confidences else None,
                "conf_missing_rate": (n - len(confidences)) / n if n else 0.0,
                "mean_answer_logprob": (math.fsum(logprobs) / len(logprobs)) if logprobs else None,
                "selected_correct_plurality_current": _selected_correct(
                    refs, labels, item_id=item_id, context=f"r{round_}|current"),
                "fallback": _is_fallback(episode),
            })
            c_prev = c_t
            prev_unanimous = unanimous_class
    return rows


def _tie_seed(item_id: str, context: str) -> bytes:
    """Frozen HMAC tie seed for one (item, pool) recomputation (spec §7 tie rule).

    The context string names the pool the vote is over — ``"r2|current"`` — so two pools of
    one item resolve their ties independently and reproducibly, and never through anything
    correctness-dependent.
    """
    return identity.sha256_hex(f"{K.ORDER_VOTE_TIE}|{item_id}|{context}").encode("ascii")


def _selected_correct(
    refs: Sequence[DecisionRef], labels: Mapping[str, Mapping[str, Any]], *, item_id: str,
    context: str,
) -> bool | None:
    selection = _plurality(refs, item_id=item_id, context=context)
    if selection.candidate_id is None:
        return None
    chosen = {ref.decision_id: ref for ref in refs}.get(selection.candidate_id)
    return None if chosen is None else _is_correct(labels, chosen)


def _plurality(refs: Sequence[DecisionRef], *, item_id: str, context: str) -> MR.Selection:
    candidates = [
        MR.PublicCandidate(candidate_id=ref.decision_id,
                           vote_key=ref.answer_class if ref.valid else None,
                           valid=bool(ref.valid))
        for ref in refs
    ]
    if not candidates:
        return MR.Selection(None, 0, 0, 0, 0)
    return MR.plurality_vote(candidates, source_id=item_id,
                             tie_seed=_tie_seed(item_id, context))


def selection_rows(
    triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Long form over selector × pool (plan §5.8), the companion of ``round_summary``.

    ``plurality`` and ``confidence_weighted`` are computed here; the remaining registered
    selectors (``CURRENT``, ``dynamics_only``, ``llm_audit``) are WP5's and appear with a
    null ``selected_decision_id`` until their cells run, so the panel's shape is stable.
    ``repeated_descendants`` flags archive rows of interacting protocols, which must never
    be read as independent votes.
    """
    rows: list[dict[str, Any]] = []
    for _cell, item_id, episode in triples:
        by_round = _round_decisions(episode)
        rounds = sorted(by_round)
        archive: list[DecisionRef] = []
        root = by_round.get(0, [])
        for round_ in rounds:
            refs = by_round[round_]
            archive = archive + list(refs)
            pools = {"root": root, "current": refs, "archive": archive}
            for pool_name, pool in pools.items():
                for selector in K.SELECTORS:
                    context = f"r{round_}|{pool_name}"
                    if selector == "plurality":
                        selection = _plurality(pool, item_id=item_id, context=context)
                        chosen = {ref.decision_id: ref for ref in pool}.get(selection.candidate_id)
                    elif selector == "confidence_weighted":
                        chosen = _confidence_weighted(pool)
                        selection = _plurality(pool, item_id=item_id, context=context)
                    else:
                        chosen = None
                        selection = _plurality(pool, item_id=item_id, context=context)
                    rows.append({
                        "episode_id": episode.episode_id, "round": round_, "item_id": item_id,
                        "protocol": episode.protocol, "checkpoint": episode.checkpoint,
                        "cohort": episode.cohort, "selector": selector, "pool": pool_name,
                        "selected_decision_id": None if chosen is None else chosen.decision_id,
                        "selected_class": None if chosen is None else chosen.answer_class,
                        "selected_correct": None if chosen is None else _is_correct(labels, chosen),
                        "expected_correct_under_ties": _expected_under_ties(pool, labels, item_id),
                        "tied_winning_classes": selection.tied_winning_classes,
                        "valid_count": selection.valid_count,
                        "planned_count": selection.planned_count, "pool_size": len(pool),
                        "repeated_descendants": pool_name == "archive" and episode.protocol not in
                        (K.Protocol.PRIVATE.value, K.Protocol.FRESH.value),
                    })
    return rows


def _confidence_weighted(pool: Sequence[DecisionRef]) -> DecisionRef | None:
    weights: dict[str, float] = defaultdict(float)
    best_by_class: dict[str, DecisionRef] = {}
    for ref in pool:
        if not ref.valid or ref.answer_class is None:
            continue
        weight = 1.0 if ref.confidence is None else float(ref.confidence)
        weights[ref.answer_class] += weight
        current = best_by_class.get(ref.answer_class)
        if current is None or (ref.confidence or 0.0) > (current.confidence or 0.0):
            best_by_class[ref.answer_class] = ref
    if not weights:
        return None
    winner = max(sorted(weights), key=lambda cls: weights[cls])
    return best_by_class[winner]


def _expected_under_ties(
    pool: Sequence[DecisionRef], labels: Mapping[str, Mapping[str, Any]], item_id: str
) -> float | None:
    """Analytic expected correctness under a uniform draw over the tied winning classes."""
    valid = [ref for ref in pool if ref.valid and ref.answer_class]
    if not valid:
        return None
    counts = Counter(ref.answer_class for ref in valid)
    best = max(counts.values())
    winners = [cls for cls, n in counts.items() if n == best]
    correct_classes = {ref.answer_class for ref in valid if _is_correct(labels, ref)}
    hits = sum(1 for cls in winners if cls in correct_classes)
    return hits / len(winners)


# --------------------------------------------------------------------------- transitions


def transition_rows(
    triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Agent-round transitions with CC/CW/WC/WW, invalidity classes and both change rules."""
    rows: list[dict[str, Any]] = []
    for _cell, item_id, episode in triples:
        by_round = _round_decisions(episode)
        rounds = sorted(by_round)
        for index in range(len(rounds) - 1):
            before_round, after_round = rounds[index], rounds[index + 1]
            before = {ref.agent_id: ref for ref in by_round[before_round]}
            for after in by_round[after_round]:
                prior = before.get(after.agent_id)
                if prior is None:
                    continue
                correct_from = _is_correct(labels, prior)
                correct_to = _is_correct(labels, after)
                if correct_from is None or correct_to is None:
                    transition = None
                else:
                    transition = ("C" if correct_from else "W") + ("C" if correct_to else "W")
                invalidity = ("valid" if prior.valid else "invalid") + "_to_" + (
                    "valid" if after.valid else "invalid")
                rows.append({
                    "episode_id": episode.episode_id, "item_id": item_id,
                    "agent_id": after.agent_id, "round_from": before_round,
                    "round_to": after_round, "protocol": episode.protocol,
                    "checkpoint": episode.checkpoint, "cohort": episode.cohort,
                    "class_from": prior.answer_class, "class_to": after.answer_class,
                    "valid_from": prior.valid, "valid_to": after.valid,
                    "correct_from": correct_from, "correct_to": correct_to,
                    "transition": transition, "invalidity_class": invalidity,
                    "changed_class": prior.answer_class != after.answer_class,
                    "changed_content": _content_hash(prior) != _content_hash(after),
                    "protected": after.protected, "n_packets": after.n_packets,
                    "confidence_from": prior.confidence, "confidence_to": after.confidence,
                })
    return rows


# --------------------------------------------------------------------------- c_dist / drift


def c_dist_rows(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``P(C_t = c | C0, protocol, family, checkpoint, cohort)`` — a descriptive transition
    summary, never a Markov law (plan §5.8)."""
    c0_by_episode = {row["episode_id"]: row["C_t"] for row in summary if row["round"] == 0}
    buckets: dict[tuple, Counter] = defaultdict(Counter)
    for row in summary:
        c0 = c0_by_episode.get(row["episode_id"])
        if c0 is None:
            continue
        key = (row["protocol"], row["checkpoint"], row["family"], row["cohort"], int(c0),
               int(row["round"]))
        buckets[key][int(row["C_t"])] += 1
    rows = []
    for key, counter in sorted(buckets.items(), key=lambda kv: [str(part) for part in kv[0]]):
        total = sum(counter.values())
        for c in range(K.N_AGENTS + 1):
            rows.append({
                "protocol": key[0], "checkpoint": key[1], "family": key[2], "cohort": key[3],
                "C0": key[4], "round": key[5], "c": c,
                "p": counter.get(c, 0) / total if total else 0.0, "n_episodes": total,
            })
    return rows


def drift_rows(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """State-conditional ``a(m)``, ``d(m)`` and realized versus predicted ``(1-m)a - m d``."""
    buckets: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary:
        if row["round"] == 0 or row["delta_m"] is None:
            continue
        key = (row["protocol"], row["checkpoint"], row["family"], row["cohort"],
               int(row["round"]), round(float(row["m_t"]) - float(row["delta_m"]), 4))
        buckets[key].append(row)
    rows = []
    for key, group in sorted(buckets.items(), key=lambda kv: [str(part) for part in kv[0]]):
        m = key[5]
        wtc = sum(int(row["n_wrong_to_correct"] or 0) for row in group)
        ctw = sum(int(row["n_correct_to_wrong"] or 0) for row in group)
        n_wrong = sum(int(row["n_initially_wrong"] or 0) for row in group)
        n_correct = sum(int(row["n_initially_correct"] or 0) for row in group)
        a = wtc / n_wrong if n_wrong else None
        d = ctw / n_correct if n_correct else None
        realized = math.fsum(float(row["delta_m"]) for row in group) / len(group)
        predicted = None if a is None or d is None else (1 - m) * a - m * d
        rows.append({
            "protocol": key[0], "checkpoint": key[1], "family": key[2], "cohort": key[3],
            "round": key[4], "m": m, "n_episodes": len(group), "a_of_m": a, "d_of_m": d,
            "realized_delta_m": realized, "predicted_delta_m": predicted,
            "residual": None if predicted is None else realized - predicted,
        })
    return rows


# --------------------------------------------------------------------------- bank panel


def bank_summary_rows(
    triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    families: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Per item × checkpoint bank and selection panel (spec §9).

    ``pass@k`` uses the unbiased finite-bank estimator ``metrics_reference.pass_at_k`` for
    ``contract.PASS_KS`` (which contains 9 and 16; the repo's ``aggregate.PASS_KS`` does not
    and is deliberately not reused).  Plurality, confidence-weighted plurality and the
    oracle are computed on the *identical* drawn subset, sharing one ``subset_id``, so the
    generation-selection gap is measured on the same pool.
    """
    from agents_scaling.ard.types import ArdCellKind

    banks: dict[tuple[str, str], list[DecisionRef]] = defaultdict(list)
    splits: dict[tuple[str, str], str] = {}
    for cell, item_id, episode in triples:
        if cell.kind is not ArdCellKind.ROOTS:
            continue
        key = (item_id, episode.checkpoint)
        banks[key].extend(ref for ref in episode.decisions if ref.round == 0)
        splits[key] = cell.split
    rows = []
    for (item_id, checkpoint), refs in sorted(banks.items()):
        refs = sorted(refs, key=lambda ref: ref.decision_id)
        outcomes = [bool(_is_correct(labels, ref)) for ref in refs]
        n, c = len(refs), sum(outcomes)
        classes = [ref.answer_class for ref in refs if ref.valid]
        correct_classes = {ref.answer_class for ref, ok in zip(refs, outcomes)
                           if ok and ref.answer_class}
        selection = _plurality(refs, item_id=item_id, context=f"bank|{checkpoint}")
        chosen = {ref.decision_id: ref for ref in refs}.get(selection.candidate_id)
        weighted = _confidence_weighted(refs)
        oracle_coverage = 1.0 if c > 0 else 0.0
        candidate_mean = c / n if n else 0.0
        selected_correct = float(bool(chosen is not None and _is_correct(labels, chosen)))
        row = {
            "item_id": item_id, "checkpoint": checkpoint, "family": families.get(item_id, ""),
            "split": splits.get((item_id, checkpoint), ""), "n_draws": n, "c_correct": c,
            "class_histogram": json.dumps(dict(sorted(Counter(classes).items()))),
            "correct_plurality_status": _plurality_status(classes, correct_classes),
            "subset_id": identity.sha256_hex("|".join(ref.decision_id for ref in refs))[:16],
            "plurality_correct": selected_correct,
            "confidence_weighted_correct": float(bool(
                weighted is not None and _is_correct(labels, weighted))),
            "oracle_coverage": oracle_coverage, "candidate_mean": candidate_mean,
            "selection_gap": oracle_coverage - selected_correct,
        }
        for k in K.PASS_KS:
            row[f"pass_at_{k}"] = MR.pass_at_k(n, c, k) if n >= k >= 1 else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- cost panel


def cost_summary_rows(
    triples: Sequence[tuple[ArdCellSpec, str, EpisodeRecord]],
    attributed: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Per episode-round cost panel (plan §5.8 "Compute cost")."""
    attributed = attributed or {}
    rows: list[dict[str, Any]] = []
    for _cell, item_id, episode in triples:
        by_round = _round_decisions(episode)
        cumulative = 0
        for round_ in sorted(by_round):
            refs = by_round[round_]
            logical = sum(int((ref.flops or {}).get("total") or 0) for ref in refs)
            cumulative += logical
            latencies = [float(ref.latency_s) for ref in refs]
            hub = episode.protocol == K.Protocol.HUB5.value
            critical = (max(latencies, default=0.0) if not hub
                        else (max(latencies[:1], default=0.0) + max(latencies[1:], default=0.0)))
            rows.append({
                "episode_id": episode.episode_id, "round": round_, "protocol": episode.protocol,
                "checkpoint": episode.checkpoint, "cohort": episode.cohort, "item_id": item_id,
                "logical_decisions": len(refs),
                "prompt_tokens": sum(ref.prompt_tokens for ref in refs),
                "prefill_tokens": sum(max(0, ref.prompt_tokens - ref.cached_prefill_tokens)
                                      for ref in refs),
                "cached_prefill_tokens": sum(ref.cached_prefill_tokens for ref in refs),
                "thinking_tokens": sum(ref.thinking_tokens for ref in refs),
                "final_tokens": sum(ref.final_tokens for ref in refs),
                "logical_flops": logical, "cumulative_logical_flops": cumulative,
                # A cap-hit draw costs two physical requests, not one (spec §6.2).
                "physical_requests": sum(
                    0 if ref.aliased else max(1, len(ref.physical_request_ids or ()))
                    for ref in refs),
                "attributed_gpu_seconds": math.fsum(
                    attributed.get(ref.decision_id, 0.0) for ref in refs),
                "wall_clock_s": math.fsum(latencies), "critical_path_s": critical,
            })
    return rows


def cost_accuracy_rows(
    cost_summary: Sequence[Mapping[str, Any]],
    round_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Per arm × round accuracy-versus-cost table.

    Explicitly **decision-matched, not FLOP-matched** (plan §5.8): GATE, PEER,
    RANDOM_PROTECT and FRESH@20 are compared at equal logical decisions, and the realized
    FLOPs are reported beside them rather than equalised.
    """
    arms: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cost_summary:
        arms[(row["protocol"], row["checkpoint"], row["cohort"], int(row["round"]))].append(row)
    allowance = {row["episode_id"]: row["reasoning_allowance"] for row in round_summary}
    final_accuracy: dict[tuple, list[bool]] = defaultdict(list)
    for row in round_summary:
        if int(row["round"]) == K.UPDATE_ROUNDS and row["selected_correct_plurality_current"] is not None:
            final_accuracy[(row["protocol"], row["checkpoint"], row["cohort"])].append(
                bool(row["selected_correct_plurality_current"]))
    rows = []
    for key, group in sorted(arms.items(), key=lambda kv: [str(part) for part in kv[0]]):
        protocol, checkpoint, cohort, round_ = key
        decisions = sum(int(row["logical_decisions"]) for row in group)
        total_flops = sum(int(row["logical_flops"]) for row in group)
        outcomes = final_accuracy.get((protocol, checkpoint, cohort), [])
        accuracy = (sum(outcomes) / len(outcomes)) if outcomes else None
        half = (1.96 * math.sqrt(accuracy * (1 - accuracy) / len(outcomes))
                if accuracy is not None and outcomes else None)
        budgets = {allowance.get(row["episode_id"]) for row in group}
        rows.append({
            "arm": f"{protocol}.{checkpoint}.{cohort}", "protocol": protocol,
            "checkpoint": checkpoint, "cohort": cohort,
            "reasoning_allowance": next(iter(budgets - {None}), None) if budgets else None,
            "round": round_, "logical_decisions": decisions,
            "mean_prompt_tokens": _mean(row["prompt_tokens"] for row in group),
            "mean_prefill_tokens": _mean(row["prefill_tokens"] for row in group),
            "mean_decode_tokens": _mean(row["thinking_tokens"] + row["final_tokens"] for row in group),
            "mean_logical_flops": total_flops / decisions if decisions else 0.0,
            "total_logical_flops": total_flops,
            "attributed_gpu_seconds": math.fsum(float(row["attributed_gpu_seconds"]) for row in group),
            "round3_selected_accuracy": accuracy,
            "ci_low": None if half is None else max(0.0, accuracy - half),
            "ci_high": None if half is None else min(1.0, accuracy + half),
            "n_episodes": len(group), "decision_matched": True, "flop_matched": False,
        })
    return rows


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return math.fsum(data) / len(data) if data else 0.0


# --------------------------------------------------------------------------- driver


def compact(
    run_root: str | os.PathLike,
    *,
    cell_ids: Sequence[str] | None = None,
    out_dir: str = TABLES_DIR,
    allow_fallback: bool = False,
    strict_residual: bool = True,
    gpu_samples: Sequence[Mapping[str, Any]] = (),
    sacct_rows: Sequence[Mapping[str, Any]] = (),
    challenge_c0: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write every table under ``<run_root>/<out_dir>`` and return the row counts.

    Episodes produced by the WP4 baseline executor (``fallback=True``) are compacted into
    ``episodes``/``decisions`` — where they are visible and flagged — but are kept out of
    every scientific table unless ``allow_fallback`` is set, so a dry run can never be
    mistaken for protocol evidence.
    """
    run_root = Path(run_root)
    triples = load_episodes(run_root, cell_ids)
    labels = load_labels(run_root)
    from agents_scaling.ard.items import load_ard_tasks

    try:
        families = {item_id: task.family for item_id, task in load_ard_tasks(run_root).items()}
    except (FileNotFoundError, OSError):
        families = {}
    scientific = [t for t in triples if allow_fallback or not _is_fallback(t[2])]

    decisions = decision_rows(triples)
    attributed_rows, _totals = RES.attribute_gpu_seconds(
        RES.request_resources_from_decisions(decisions))
    attributed = {str(row["decision_id"]): float(row["attributed_gpu_seconds"])
                  for row in attributed_rows if row.get("decision_id")}
    summary = round_summary_rows(scientific, labels, families=families,
                                 challenge_c0=challenge_c0, strict_residual=strict_residual)
    costs = cost_summary_rows(scientific, attributed)
    tables: dict[str, list[dict[str, Any]]] = {
        "items": items_rows(run_root),
        "episodes": episode_rows(triples, families),
        "decisions": decisions,
        "labels": label_rows(labels),
        "features": [],
        "forks": fork_rows(triples),
        "resources": resource_rows(decisions, gpu_samples=gpu_samples, sacct_rows=sacct_rows),
        "round_summary": summary,
        SELECTION_TABLE: selection_rows(scientific, labels),
        "transitions": transition_rows(scientific, labels),
        "c_dist": c_dist_rows(summary),
        "drift": drift_rows(summary),
        "bank_summary": bank_summary_rows(scientific, labels, families=families),
        "calibration_panel": [],
        "cost_summary": costs,
        "cost_accuracy": cost_accuracy_rows(costs, summary),
        "neural_geometry": [],
        "neural_agent": [],
        "challenge_teams": [],
        "suffix_summary": [],
    }
    missing = set(K.ALL_TABLES) - set(tables)
    if missing:  # pragma: no cover - guarded by tests
        raise CompactionError(f"compaction produced no entry for {sorted(missing)}")
    tables_dir = run_root / out_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        _write_parquet(tables_dir / f"{name}.parquet", [_project(name, row) for row in rows])
    counts = {name: len(rows) for name, rows in tables.items()}
    io.write_json(tables_dir / "compaction.json", {
        "study_id": K.STUDY_ID, "n_episodes": len(triples), "n_scientific_episodes": len(scientific),
        "n_labels": len(labels), "rows": counts, "allow_fallback": allow_fallback,
        "deferred_tables": sorted(DEFERRED_TABLES),
        "attribution_rule": RES.ATTRIBUTION_RULE,
    })
    return counts


def _project(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Every emitted row carries exactly its declared schema (missing keys become ``None``)."""
    schema = SCHEMAS[table]
    unknown = set(row) - set(schema)
    if unknown:
        raise CompactionError(f"{table}: rows carry undeclared columns {sorted(unknown)}")
    return {key: row.get(key) for key in schema}


def run_compact_cell(runner: Any, item_id: str) -> None:
    """COMPACT cell worker: compact the run root, then publish the cell's item file."""
    options = dict(runner.ard_cell.options)
    counts = compact(
        runner.run_root,
        cell_ids=options.get("cells"),
        allow_fallback=bool(options.get("allow_fallback", False)),
        strict_residual=bool(options.get("strict_residual", True)),
    )
    runner.event("compacted", rows=counts)
    task = runner.tasks.get(item_id)
    if task is None:  # the synthetic "ALL" item of an analysis cell
        io.write_json(runner.paths.item(item_id), {
            "schema_version": 1, "cell": runner.ard_cell.to_dict(), "source_id": item_id,
            "family": "", "split": runner.ard_cell.split, "status": "complete",
            "episodes": [], "timing": {"finished_at": runner.clock()},
            "worker": runner._worker_block(), "grades": [],
            "code_version": runner.code_version, "rows": counts,
        })
        return
    runner.write_item(task, ())


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.compact",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-root", required=True)
    p.add_argument("--cells-file", default=None, help="restrict the scan to this manifest's cells")
    p.add_argument("--out-dir", default=TABLES_DIR)
    p.add_argument("--allow-fallback", action="store_true",
                   help="compact WP4 baseline-executor episodes into the scientific tables (dry runs only)")
    p.add_argument("--no-strict-residual", action="store_true",
                   help="report instead of raising on a nonzero C_t accounting residual")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cell_ids = None
    if args.cells_file:
        from agents_scaling.ard.cells import load_ard_cells_file

        cell_ids = [cell.cell_id for cell in load_ard_cells_file(args.cells_file)]
    counts = compact(args.run_root, cell_ids=cell_ids, out_dir=args.out_dir,
                     allow_fallback=args.allow_fallback,
                     strict_residual=not args.no_strict_residual)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "CompactionError", "DEFERRED_TABLES", "GRADES_DIR", "PLURALITY_STATUSES", "SCHEMAS",
    "SELECTION_TABLE", "TABLES_DIR", "bank_summary_rows", "c_dist_rows", "compact",
    "cost_accuracy_rows", "cost_summary_rows", "decision_rows", "drift_rows", "episode_rows",
    "fork_rows", "items_rows", "iter_item_files", "label_rows", "load_episodes", "load_labels",
    "resource_rows", "round_summary_rows", "run_compact_cell", "selection_rows",
    "transition_rows",
]
