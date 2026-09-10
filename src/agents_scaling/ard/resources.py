"""Cost and hardware accounting: FLOP wiring, GPU sampling, GPU-second attribution (WP4).

Spec §16 (compute ledger), §18 ``resources.parquet`` (`timestamp, job_id, node, gpu_uuid,
gpu_type, memory_total, allocated_gpu_seconds, utilization, memory_used, power_if_available,
server_batch_id, physical_request_ids, prefill_tokens, decode_tokens, retries, errors`) and
its two rules: *do not claim exact per-request FLOPs from token count alone; report logical
request costs separately from physical cached execution*, and *for batched GPU time
attribution use a documented approximation and retain server-level totals*.  Plan §5.8
("Compute cost").

The three pieces this module owns:

1. **FLOP wiring.**  ``oracle_factory=FlopOracle.from_checkpoint`` + ``make_cost_fn`` are
   attached to the ARD client exactly as ``study/runner.py:254-261`` attaches them.  Without
   it every ``DecisionRecord.flops`` is the ``None`` placeholder and the run produces no FLOP
   number at all.  :func:`assert_flops_populated` is the guard the hour-2 gate calls.
2. **Hardware sampling.**  The ``nvidia-smi`` sampler line each serve job writes, and the
   ``sacct`` GPU-second capture at teardown, joined to the registry on
   ``(host, slurm_job_id)``.
3. **Batched GPU-time attribution.**  The OpenAI-compatible API exposes no batch id, so
   ``server_batch_id = (endpoint_id, one-second arrival bucket)`` and a decision receives its
   request wall time divided by the number of concurrently in-flight requests on that
   replica.  The approximation is documented here, the un-attributed server-level totals are
   retained verbatim, and the substitution is logged as an amendment.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from agents_scaling.ard import contract as K
from agents_scaling.ard.client import cost_fn_for, oracle_for, oracle_tables, verify_oracle_hash
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn
from agents_scaling.study.types import ProtocolError

# --------------------------------------------------------------------------- FLOP wiring

#: The frozen attachment point (contract §2): pass this as ``oracle_factory``.
oracle_factory = FlopOracle.from_checkpoint


def flop_cost_fn(checkpoints: Mapping[str, Any], home: str | os.PathLike | None = None):
    """``cost_fn`` over the pinned snapshots — never a placeholder (contract §2)."""
    return cost_fn_for(checkpoints, home)


def assert_flops_populated(flops: Mapping[str, Any] | None, *, where: str = "decision") -> None:
    """Refuse a record whose ``flops`` block is the ``None`` placeholder or is incomplete."""
    if not flops:
        raise ProtocolError(
            f"{where}: flops is the None placeholder — the FLOP oracle was not attached "
            "(ard/resources.oracle_factory + make_cost_fn, study/runner.py:254-261)")
    missing = [key for key in ("prefill", "decode", "total") if flops.get(key) is None]
    if missing:
        raise ProtocolError(f"{where}: flops block is missing {missing}")


# --------------------------------------------------------------------------- nvidia-smi

NVIDIA_SMI_FIELDS: Final[tuple[str, ...]] = (
    "uuid", "name", "memory.total", "memory.used", "utilization.gpu", "power.draw",
)
#: Column names the sampler writes (``timestamp`` is prepended by the sampler itself).
NVIDIA_SMI_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp", "gpu_uuid", "gpu_type", "memory_total", "memory_used", "utilization",
    "power_if_available",
)
NVIDIA_SMI_INTERVAL_S: Final[int] = 15


def nvidia_smi_argv(interval_s: int = NVIDIA_SMI_INTERVAL_S) -> list[str]:
    """The exact query the serve job runs (spec §18 resources field list)."""
    return [
        "nvidia-smi",
        "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS),
        "--format=csv,noheader,nounits",
        f"--loop={int(interval_s)}",
    ]


def sampler_line(out_path: str, interval_s: int = NVIDIA_SMI_INTERVAL_S) -> str:
    """The single background line dropped into ``slurm/ard36_serve_qwen.sbatch.tmpl``.

    Emits one CSV row per GPU per ``interval_s`` seconds, prefixed with an epoch timestamp,
    so the sample joins to the registry on ``(host, slurm_job_id)``.
    """
    return (
        f"( {' '.join(nvidia_smi_argv(interval_s))} "
        f"| awk -v OFS=, '{{ print systime(), $0 }}' >> \"{out_path}\" ) &"
    )


def parse_nvidia_smi_csv(text: str, *, host: str = "", job_id: str | None = None) -> list[dict[str, Any]]:
    """Parse the sampler file into ``resources.parquet``-shaped rows."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(NVIDIA_SMI_COLUMNS):
            continue
        row = dict(zip(NVIDIA_SMI_COLUMNS, parts[: len(NVIDIA_SMI_COLUMNS)]))
        rows.append({
            "timestamp": _as_float(row["timestamp"]),
            "job_id": job_id,
            "node": host,
            "gpu_uuid": row["gpu_uuid"],
            "gpu_type": row["gpu_type"],
            "memory_total": _as_float(row["memory_total"]),
            "memory_used": _as_float(row["memory_used"]),
            "utilization": _as_float(row["utilization"]),
            "power_if_available": _as_float(row["power_if_available"]),
        })
    return rows


def _as_float(text: str) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- sacct

SACCT_FORMAT: Final[str] = "JobID,JobName,Partition,NodeList,AllocTRES,ElapsedRaw,State"


def sacct_argv(job_ids: Sequence[str]) -> list[str]:
    """``sacct`` capture of allocated GPU-seconds at teardown (plan §5.8)."""
    return ["sacct", "-n", "-P", f"--format={SACCT_FORMAT}", "-j", ",".join(str(j) for j in job_ids)]


def parse_sacct_gpu_seconds(text: str) -> list[dict[str, Any]]:
    """``allocated_gpu_seconds = gres/gpu of AllocTRES × ElapsedRaw`` per accounting row."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        job_id, job_name, partition, nodes, alloc_tres, elapsed, state = parts[:7]
        gpus = 0
        for token in alloc_tres.split(","):
            if token.startswith("gres/gpu="):
                try:
                    gpus = int(token.split("=", 1)[1])
                except ValueError:
                    gpus = 0
        try:
            seconds = int(elapsed)
        except ValueError:
            seconds = 0
        rows.append({
            "job_id": job_id, "job_name": job_name, "partition": partition, "node": nodes,
            "gpus": gpus, "elapsed_s": seconds, "allocated_gpu_seconds": gpus * seconds,
            "state": state,
        })
    return rows


def capture_sacct_gpu_seconds(job_ids: Sequence[str]) -> list[dict[str, Any]]:  # pragma: no cover - needs Slurm
    if not job_ids:
        return []
    out = subprocess.run(sacct_argv(job_ids), capture_output=True, text=True).stdout
    return parse_sacct_gpu_seconds(out)


# --------------------------------------------------------------------------- attribution

#: One-second arrival bucket: the batch proxy of the documented approximation.
ARRIVAL_BUCKET_S: Final[float] = 1.0
ATTRIBUTION_RULE: Final[str] = (
    "server_batch_id = (endpoint_id, floor(arrival_ts / 1 s)); a decision receives its "
    "request wall time divided by the time-weighted number of concurrently in-flight "
    "requests on that replica; un-attributed server-level busy seconds are retained "
    "verbatim (spec §18, plan §5.8)."
)


def server_batch_id(endpoint_id: str, arrival_ts: float) -> str:
    """``(endpoint_id, one-second arrival bucket)`` — the batch proxy, as a string key."""
    return f"{endpoint_id}@{int(math.floor(float(arrival_ts) / ARRIVAL_BUCKET_S))}"


@dataclass(frozen=True)
class RequestResource:
    """One physical request as the attribution sees it."""

    request_id: str
    endpoint_id: str
    arrival_ts: float
    latency_s: float
    prompt_tokens: int = 0
    cached_prefill_tokens: int = 0
    decode_tokens: int = 0
    retries: int = 0
    errors: int = 0
    decision_id: str | None = None

    @property
    def prefill_tokens(self) -> int:
        """Physical prefill: the logical prompt minus what the prefix cache already held."""
        return max(0, int(self.prompt_tokens) - int(self.cached_prefill_tokens))

    @property
    def end_ts(self) -> float:
        return float(self.arrival_ts) + max(0.0, float(self.latency_s))


def _mean_concurrency(target: RequestResource, peers: Sequence[RequestResource]) -> float:
    """Time-weighted mean number of in-flight requests on the replica over ``target``'s span."""
    start, end = float(target.arrival_ts), target.end_ts
    span = end - start
    if span <= 0:
        overlapping = sum(1 for peer in peers if peer.arrival_ts <= start < peer.end_ts)
        return float(max(1, overlapping))
    edges = {start, end}
    for peer in peers:
        for point in (float(peer.arrival_ts), peer.end_ts):
            if start < point < end:
                edges.add(point)
    ordered = sorted(edges)
    weighted = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = right - left
        if width <= 0:
            continue
        middle = left + width / 2.0
        n = sum(1 for peer in peers if peer.arrival_ts <= middle < peer.end_ts)
        weighted += width * max(1, n)
    return weighted / span if span > 0 else 1.0


def attribute_gpu_seconds(
    requests: Iterable[RequestResource],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """``(per-request rows, per-replica server totals)`` under :data:`ATTRIBUTION_RULE`.

    Server totals are the *un-attributed* truth kept beside the approximation: busy seconds
    are the union of the request intervals on that replica (never their sum), so a batched
    replica's totals stay comparable with ``sacct``'s allocated GPU-seconds.
    """
    by_endpoint: dict[str, list[RequestResource]] = {}
    for request in requests:
        by_endpoint.setdefault(request.endpoint_id, []).append(request)
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {}
    for endpoint, group in by_endpoint.items():
        ordered = sorted(group, key=lambda r: (r.arrival_ts, r.request_id))
        busy = _union_seconds(ordered)
        summed = math.fsum(max(0.0, r.latency_s) for r in ordered)
        attributed_total = 0.0
        for request in ordered:
            concurrency = _mean_concurrency(request, ordered)
            attributed = max(0.0, float(request.latency_s)) / concurrency
            attributed_total += attributed
            rows.append({
                "request_id": request.request_id,
                "decision_id": request.decision_id,
                "endpoint_id": endpoint,
                "server_batch_id": server_batch_id(endpoint, request.arrival_ts),
                "timestamp": float(request.arrival_ts),
                "latency_s": float(request.latency_s),
                "mean_concurrency": concurrency,
                "attributed_gpu_seconds": attributed,
                "prompt_tokens": int(request.prompt_tokens),
                "cached_prefill_tokens": int(request.cached_prefill_tokens),
                "prefill_tokens": request.prefill_tokens,
                "decode_tokens": int(request.decode_tokens),
                "retries": int(request.retries),
                "errors": int(request.errors),
                "attribution_rule": ATTRIBUTION_RULE,
            })
        totals[endpoint] = {
            "n_requests": float(len(ordered)),
            "server_busy_seconds": busy,
            "summed_request_seconds": summed,
            "attributed_seconds": attributed_total,
            "first_arrival": float(ordered[0].arrival_ts) if ordered else 0.0,
            "last_completion": max((r.end_ts for r in ordered), default=0.0),
        }
    return rows, totals


def _union_seconds(requests: Sequence[RequestResource]) -> float:
    """Wall seconds during which the replica had at least one request in flight."""
    intervals = sorted(((float(r.arrival_ts), r.end_ts) for r in requests if r.end_ts > r.arrival_ts))
    total = 0.0
    current_start: float | None = None
    current_end = 0.0
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
            continue
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def request_resources_from_decisions(decisions: Iterable[Mapping[str, Any]]) -> list[RequestResource]:
    """Build the attribution inputs from ``decisions.parquet``-shaped rows."""
    out: list[RequestResource] = []
    for row in decisions:
        out.append(RequestResource(
            request_id=str(row.get("request_id") or row.get("decision_id") or ""),
            endpoint_id=str(row.get("endpoint_id") or ""),
            arrival_ts=float(row.get("arrival_ts") or 0.0),
            latency_s=float(row.get("latency_s") or 0.0),
            prompt_tokens=int(row.get("prompt_tokens") or 0),
            cached_prefill_tokens=int(row.get("cached_prefill_tokens") or 0),
            decode_tokens=int(row.get("thinking_tokens") or 0) + int(row.get("final_tokens") or 0),
            retries=int(row.get("retries") or 0),
            errors=int(row.get("errors") or 0),
            decision_id=row.get("decision_id"),
        ))
    return out


__all__ = [
    "ARRIVAL_BUCKET_S", "ATTRIBUTION_RULE", "NVIDIA_SMI_COLUMNS", "NVIDIA_SMI_FIELDS",
    "NVIDIA_SMI_INTERVAL_S", "RequestResource", "SACCT_FORMAT", "assert_flops_populated",
    "attribute_gpu_seconds", "capture_sacct_gpu_seconds", "flop_cost_fn", "make_cost_fn",
    "nvidia_smi_argv", "oracle_factory", "oracle_for", "oracle_tables",
    "parse_nvidia_smi_csv", "parse_sacct_gpu_seconds", "request_resources_from_decisions",
    "sacct_argv", "sampler_line", "server_batch_id", "verify_oracle_hash",
]
