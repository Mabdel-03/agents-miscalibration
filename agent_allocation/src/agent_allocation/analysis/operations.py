"""Operational descriptives: prompt growth per protocol, headroom utilization and truncation, cache usage,
parse and transport failures, latency, billed retries, and recorded stop reasons.

All are descriptive, computed per partition x benchmark x generation kind (x round where prompts grow).
Missing usage fields stay unknown and are counted, never read as zero.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from agent_allocation.analysis.load import RunData


def _share(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def generation_operations(candidates: pd.DataFrame, keys: tuple[str, ...] = ("partition_id", "benchmark", "kind",
                                                                            "team_size", "update_round")
                          ) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    done = candidates[candidates["terminal"]]
    keys = list(keys)
    for group_keys, group in done.groupby(keys, sort=True, dropna=False):
        prompt = pd.to_numeric(group["prompt_tokens"], errors="coerce")
        completion = pd.to_numeric(group["completion_tokens"], errors="coerce")
        reasoning = pd.to_numeric(group["reasoning_tokens"], errors="coerce")
        cached = pd.to_numeric(group["cached_prompt_tokens"], errors="coerce")
        cap = group["max_completion_tokens"].astype(float)
        budget = group["reasoning_budget"].astype(float)
        latency = pd.to_numeric(group["latency_seconds"], errors="coerce")
        rows.append({
            **dict(zip(keys, group_keys)),
            "generations": int(len(group)),
            "mean_prompt_tokens": _mean(prompt), "n_prompt_unknown": int(prompt.isna().sum()),
            "mean_completion_tokens": _mean(completion),
            "mean_reasoning_tokens": _mean(reasoning), "n_reasoning_unknown": int(reasoning.isna().sum()),
            "mean_requested_reasoning": _mean(budget),
            "mean_reasoning_over_requested": _mean(reasoning / budget),
            "mean_completion_over_cap": _mean(completion / cap),
            "share_at_completion_cap": _share(completion >= cap) if completion.notna().any() else None,
            "share_finish_length": _share(group["completion_status"] == "length"),
            "share_empty": _share(group["completion_status"] == "empty"),
            "share_parse_invalid": _share(group["parse_status"] != "valid"),
            "share_transport_not_ok": _share(group["transport_status"] != "ok"),
            "share_response_cache_hit": _share(group["cache_status"].astype(str).str.upper() == "HIT"),
            "mean_cached_prompt_share": _mean(cached / prompt),
            "median_latency_seconds": float(latency.median()) if latency.notna().any() else None,
            "p90_latency_seconds": float(latency.quantile(0.9)) if latency.notna().any() else None,
            "public_check_cpu_seconds": _sum(group["public_check_cpu_seconds"]),
            "private_grading_cpu_seconds": _sum(group["grade_cpu_seconds"]),
        })
    return pd.DataFrame(rows)


def _mean(series: pd.Series) -> float | None:
    values = series.replace([np.inf, -np.inf], np.nan).astype(float)
    return float(values.mean()) if values.notna().any() else None


def _sum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.sum()) if values.notna().any() else None


def retry_table(run: RunData, charges: dict, attempts_by_node: dict) -> pd.DataFrame:
    rows = []
    specs = {nid: (part, spec) for part, specs in run.jobs.items() for nid, spec in specs.items()}
    for nid, attempts in sorted(attempts_by_node.items()):
        accepted = run.parsed[nid][0].attempt_id if nid in run.parsed else None
        extra = [a for a in attempts if a != accepted]
        if not extra:
            continue
        part, spec = specs.get(nid, (None, None))
        rows.append({"partition_id": part, "node_id": nid, "benchmark": spec.benchmark if spec else None,
                     "kind": spec.kind.value if spec else None, "retried_attempts": len(extra),
                     "retry_charges_usd": str(sum((charges.get(a, 0) for a in extra), start=0)),
                     "accepted": accepted is not None})
    return pd.DataFrame(rows)


def stop_reasons(run: RunData) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    status = run.paths.status_json
    if status.is_file():
        try:
            data = json.loads(status.read_text())
        except ValueError:
            data = {}
        for key in ("stop_reason", "stopped_reason", "state"):
            if key in data:
                out.append({"source": "state/status.json", "field": key, "value": str(data[key])})
    for gate in run.gates:
        out.append({"source": f"gates/{gate.gate}_{gate.stage}.json", "field": "decision",
                    "value": f"{gate.decision} (passed={gate.passed})"})
    return out
