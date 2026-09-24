"""WP-D: stage reconciliation against OpenRouter's free account endpoints (contract §8; plan §9 Gates 1/2/4).

``before`` snapshots ``/key`` and ``/credits`` (never the key or its label) plus the ledger's settled total into
``<run_root>/state/reconcile_<stage>_before.json``; ``after`` snapshots again, compares the key-usage delta with the
ledger's settled + charged-at-maximum delta, fetches ``/generation`` metadata for a seeded 2 % sample of the stage's
generation ids (tokens counted exactly once, cost within $0.01 or 5 %, provider == pinned) and writes
``<run_root>/state/reconciliation_<stage>.json``. An upstream HTTP 402 (``PAYMENT_REQUIRED``: key limit or balance)
anywhere in the stage is an infrastructure ``stop_notify``, never a harness exhaustion (plan B6). Every printed or
journaled byte goes through ``apikey.redact``.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.apikey import redact
from agent_allocation.contracts import LedgerEventKind
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic

LAYOUT = "dynorch/reconciliation/v1"
SAMPLE_SHARE = Decimal("0.02")
COST_TOLERANCE_USD = Decimal("0.01")
COST_TOLERANCE_SHARE = Decimal("0.05")
ZERO = Decimal("0")
CLEAN, MISMATCH, STOP_NOTIFY = "clean", "mismatch", "stop_notify"


class ReconcileError(RuntimeError):
    """An input the reconciliation cannot interpret (never a silent default)."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dec(value: Any, what: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ReconcileError(f"{what} is {value!r}, not a number")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        raise ReconcileError(f"{what}={value!r} is not a decimal") from None


def _opt_dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


# ------------------------------------------------------------------------------------------------------------------
# Account snapshots
# ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountSnapshot:
    recorded_at: str
    key_usage_usd: Decimal
    key_limit_usd: Decimal | None
    key_limit_remaining_usd: Decimal | None
    is_free_tier: bool | None
    total_credits_usd: Decimal | None
    total_usage_usd: Decimal | None
    payment_required: bool          # a 402 from the account endpoints themselves

    def to_json(self) -> dict[str, Any]:
        return {"recorded_at": self.recorded_at, "key": {"usage": str(self.key_usage_usd),
                                                          "limit": None if self.key_limit_usd is None else str(self.key_limit_usd),
                                                          "limit_remaining": None if self.key_limit_remaining_usd is None else str(self.key_limit_remaining_usd),
                                                          "is_free_tier": self.is_free_tier},
                "credits": {"total_credits": None if self.total_credits_usd is None else str(self.total_credits_usd),
                            "total_usage": None if self.total_usage_usd is None else str(self.total_usage_usd)},
                "payment_required": self.payment_required}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AccountSnapshot":
        key, credits = data["key"], data["credits"]
        return cls(str(data["recorded_at"]), _dec(key["usage"], "key.usage"), _opt_dec(key.get("limit")),
                   _opt_dec(key.get("limit_remaining")), key.get("is_free_tier"), _opt_dec(credits.get("total_credits")),
                   _opt_dec(credits.get("total_usage")), bool(data.get("payment_required", False)))


async def snapshot_account(client: Any) -> AccountSnapshot:
    """``client`` is ``api/client.py::OpenRouterClient`` (``key_info`` / ``credits``; ``label`` already dropped).
    A 402 from the account endpoints is recorded as ``payment_required`` instead of raising."""
    from agent_allocation.api.client import OpenRouterAccountError
    from agent_allocation.contracts import TransportStatus
    payment_required = False
    try:
        key = await client.key_info()
    except OpenRouterAccountError as exc:
        if exc.transport_status is not TransportStatus.PAYMENT_REQUIRED:
            raise
        key, payment_required = {"usage": 0}, True
    try:
        credits = await client.credits()
    except OpenRouterAccountError as exc:
        if exc.transport_status is not TransportStatus.PAYMENT_REQUIRED:
            raise
        credits, payment_required = {}, True
    if "label" in key:
        raise ReconcileError("key_info returned the key label; refusing to record it")
    return AccountSnapshot(utcnow(), _dec(key.get("usage", 0), "key.usage"), _opt_dec(key.get("limit")),
                           _opt_dec(key.get("limit_remaining")), key.get("is_free_tier"),
                           _opt_dec(credits.get("total_credits")), _opt_dec(credits.get("total_usage")), payment_required)


# ------------------------------------------------------------------------------------------------------------------
# Ledger view
# ------------------------------------------------------------------------------------------------------------------

def ledger_totals(paths: RunPaths) -> dict[str, Any]:
    """Settled / charged-at-maximum / released / open totals from the fsync'd ledger journal (durable artifacts only)."""
    from agent_allocation.runtime.ledger import journal_attempts
    if not paths.ledger_journal.exists():
        return {"attempts": 0, "settled_usd": "0", "charged_max_usd": "0", "released_unsent_usd": "0",
                "open_reservations_usd": "0", "open_attempts": []}
    attempts = journal_attempts(paths.ledger_journal)
    settled = charged = released = open_total = ZERO
    open_ids: list[str] = []
    for aid, info in sorted(attempts.items()):
        if info.resolution is None:
            open_total += info.reserved_usd
            open_ids.append(aid)
        elif info.resolution is LedgerEventKind.SETTLE:
            settled += info.resolved_usd or ZERO
        elif info.resolution is LedgerEventKind.CHARGE_MAX:
            charged += info.resolved_usd or ZERO
        elif info.resolution is LedgerEventKind.RELEASE_UNSENT:
            released += info.resolved_usd or ZERO
    return {"attempts": len(attempts), "settled_usd": str(settled), "charged_max_usd": str(charged),
            "released_unsent_usd": str(released), "open_reservations_usd": str(open_total), "open_attempts": open_ids}


def ledger_charged(totals: Mapping[str, Any]) -> Decimal:
    return Decimal(totals["settled_usd"]) + Decimal(totals["charged_max_usd"])


# ------------------------------------------------------------------------------------------------------------------
# Generation sample
# ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CallGenerations:
    """One proxy call's journaled facts for the /generation cross-check. ``completion_tokens`` counts the grace."""
    run_key: str
    call_no: int
    generation_ids: tuple[str, ...]
    completion_tokens: int | None
    cost_usd: Decimal | None
    expected_provider: str | None


def calls_from_table(calls: Any, runs: Any, locks: Mapping[str, Any]) -> list[CallGenerations]:
    """The population from WP-C's ``dynorch_calls.parquet`` joined to ``dynorch_runs.parquet`` for the shard
    (``locks``: ``{shard_id: EndpointLock}`` gives the pinned provider name)."""
    for column in ("run_key", "call_no", "generation_ids", "completion_tokens", "grace_tokens", "cost_usd"):
        if column not in calls.columns:
            raise ReconcileError(f"dynorch_calls table lacks column {column!r}")
    for column in ("run_key", "shard_id"):
        if column not in runs.columns:
            raise ReconcileError(f"dynorch_runs table lacks column {column!r}")
    shard_of = dict(zip(runs["run_key"], runs["shard_id"]))
    rows: list[CallGenerations] = []
    for rec in calls.to_dict("records"):
        ids = _generation_ids(rec["generation_ids"], f"{rec['run_key']}#{rec['call_no']}")
        if not ids:
            continue
        shard = shard_of.get(rec["run_key"])
        lock = locks.get(shard) if shard is not None else None
        completion = rec["completion_tokens"]
        grace = rec["grace_tokens"]
        total = None if completion is None or _isnan(completion) else int(completion) + (0 if grace is None or _isnan(grace) else int(grace))
        cost = rec["cost_usd"]
        rows.append(CallGenerations(str(rec["run_key"]), int(rec["call_no"]), ids, total,
                                    None if cost is None or (isinstance(cost, float) and _isnan(cost)) else Decimal(str(cost)),
                                    getattr(lock, "provider_name", None)))
    return rows


def _isnan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _generation_ids(value: Any, where: str) -> tuple[str, ...]:
    """WP-C's ``dynorch_calls.generation_ids`` column is the JSON text of the id list (a parquet-friendly string,
    contract §7 / requests/wpC.md); a list / array is accepted too. Anything else is a malformed table."""
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ReconcileError(f"{where}: generation_ids is not a JSON list: {value[:80]!r}") from None
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        items = list(value.tolist()) if hasattr(value, "tolist") else list(value)
        if not all(isinstance(g, str) for g in items):
            raise ReconcileError(f"{where}: generation_ids must be strings, got {items!r}")
        return tuple(g for g in items if g)
    raise ReconcileError(f"{where}: generation_ids has an unexpected type {type(value).__name__}")


def calls_from_journal(run_root: Path) -> list[CallGenerations]:
    """Fallback population from the proxy's own call journals (``raw/**/call_*.json.gz``, layout ``dynorch/v1``):
    generation ids from the phases, tokens from ``budget.realized`` (+ ``reply.grace_tokens`` when present); no
    cost (the ledger, not the journal, owns dollars) and no provider expectation."""
    rows: list[CallGenerations] = []
    raw = Path(run_root) / "raw"
    if not raw.exists():
        return rows
    for path in sorted(raw.rglob("call_*.json.gz")):
        with gzip.open(path, "rb") as handle:
            doc = json.loads(handle.read().decode("utf-8"))
        if doc.get("layout") != "dynorch/v1":
            raise ReconcileError(f"{path} is not a dynorch/v1 call journal")
        ids = tuple(str(p["generation_id"]) for p in doc.get("phases", []) if p.get("generation_id"))
        if not ids:
            continue
        budget = doc.get("budget") or {}
        realized = budget.get("realized")
        grace = (doc.get("reply") or {}).get("grace_tokens") or 0
        rows.append(CallGenerations(str(doc["run_key"]), int(doc["call_no"]), ids,
                                    None if realized is None else int(realized) + int(grace), None, None))
    return rows


def sample_calls(population: Sequence[CallGenerations], *, seed: int, share: Decimal = SAMPLE_SHARE,
                 minimum: int = 1) -> list[CallGenerations]:
    """Seeded sample of ``max(minimum, ceil(share·n))`` calls, deterministic in (seed, population order)."""
    if not population:
        return []
    ordered = sorted(population, key=lambda c: (c.run_key, c.call_no))
    n = max(minimum, int(-(-len(ordered) * share // 1)))     # ceil
    n = min(n, len(ordered))
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, n), key=lambda c: (c.run_key, c.call_no))


def cost_within(journaled: Decimal | None, metadata: Decimal | None) -> bool | None:
    if journaled is None or metadata is None:
        return None
    tolerance = max(COST_TOLERANCE_USD, abs(journaled) * COST_TOLERANCE_SHARE)
    return abs(journaled - metadata) <= tolerance


async def fetch_generations(client: Any, ids: Iterable[str], *, attempts: int = 6, delay_s: float = 5.0) -> dict[str, Mapping[str, Any] | None]:
    """``/generation`` for every id, retrying 404s (the endpoint lags a few seconds). Missing after ``attempts``
    → ``None`` (reported, never invented)."""
    wanted = [g for g in dict.fromkeys(ids) if g]
    found: dict[str, Mapping[str, Any] | None] = {g: None for g in wanted}
    for round_no in range(attempts):
        pending = [g for g in wanted if found[g] is None]
        if not pending:
            break
        if round_no:
            await asyncio.sleep(delay_s)
        for gid in pending:
            data = await client.generation_metadata(gid)
            if data is not None:
                found[gid] = data
    return found


def compare_call(call: CallGenerations, metadata: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """One sample row: journaled vs summed ``/generation`` facts for the call's ids (a forced-close pair counts once
    on both sides)."""
    docs = [metadata.get(g) for g in call.generation_ids]
    fetched = all(d is not None for d in docs)
    tokens = cost = None
    providers: set[str] = set()
    if fetched:
        tokens = 0
        cost = ZERO
        for doc in docs:
            assert doc is not None
            tokens += int(doc.get("tokens_completion") or 0)
            cost += _dec(doc.get("total_cost", 0), "total_cost")
            if doc.get("provider_name") is not None:
                providers.add(str(doc["provider_name"]))
    provider_ok: bool | None
    if not fetched or call.expected_provider is None:
        provider_ok = None
    else:
        provider_ok = providers == {call.expected_provider}
    return {
        "run_key": call.run_key, "call_no": call.call_no, "generation_ids": list(call.generation_ids),
        "fetched": fetched,
        "journaled_completion_tokens": call.completion_tokens, "metadata_completion_tokens": tokens,
        "tokens_match": None if not fetched or call.completion_tokens is None else tokens == call.completion_tokens,
        "journaled_cost_usd": None if call.cost_usd is None else str(call.cost_usd),
        "metadata_cost_usd": None if cost is None else str(cost),
        "cost_within_tolerance": cost_within(call.cost_usd, cost),
        "provider_names": sorted(providers), "expected_provider": call.expected_provider, "provider_ok": provider_ok,
    }


# ------------------------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------------------------

def before_path(paths: RunPaths, stage: str) -> Path:
    return paths.state_dir / f"reconcile_{stage}_before.json"


def report_path(paths: RunPaths, stage: str) -> Path:
    return paths.state_dir / f"reconciliation_{stage}.json"


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    text = json.dumps(doc, indent=1, sort_keys=True, default=str)
    replace_atomic(path, (redact(text) or "").encode("utf-8"))


def write_before(paths: RunPaths, stage: str, snapshot: AccountSnapshot) -> Path:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    totals = ledger_totals(paths)
    doc = {"layout": LAYOUT, "stage": stage, "phase": "before", "account": snapshot.to_json(), "ledger": totals,
           "ledger_charged_usd": str(ledger_charged(totals))}
    target = before_path(paths, stage)
    _write(target, doc)
    return target


def read_before(paths: RunPaths, stage: str) -> dict[str, Any]:
    target = before_path(paths, stage)
    if not target.exists():
        raise ReconcileError(f"no before-snapshot for stage {stage!r} ({target}); run dynorch-reconcile --phase before first")
    doc = json.loads(target.read_text())
    if doc.get("layout") != LAYOUT or doc.get("phase") != "before":
        raise ReconcileError(f"{target} is not a {LAYOUT} before-snapshot")
    return doc


def build_report(*, stage: str, before: Mapping[str, Any], after: AccountSnapshot, ledger_after: Mapping[str, Any],
                 sample_rows: Sequence[Mapping[str, Any]], population: int, seed: int, population_source: str,
                 upstream_402: bool, key_tolerance_usd: Decimal) -> dict[str, Any]:
    """The reconciliation document (pure; ``upstream_402`` = any run/call in the stage saw an upstream 402)."""
    before_snap = AccountSnapshot.from_json(before["account"])
    key_delta = after.key_usage_usd - before_snap.key_usage_usd
    ledger_delta = ledger_charged(ledger_after) - Decimal(before["ledger_charged_usd"])
    diff = key_delta - ledger_delta
    open_usd = Decimal(ledger_after["open_reservations_usd"])
    problems: list[str] = []
    if abs(diff) > key_tolerance_usd:
        problems.append(f"key delta {key_delta} vs ledger delta {ledger_delta} differ by {diff} (> {key_tolerance_usd})")
    if open_usd != ZERO:
        problems.append(f"open reservations {open_usd}")
    for row in sample_rows:
        if not row["fetched"]:
            problems.append(f"{row['run_key']}#{row['call_no']}: /generation missing for {row['generation_ids']}")
        if row["tokens_match"] is False:
            problems.append(f"{row['run_key']}#{row['call_no']}: tokens {row['journaled_completion_tokens']} != {row['metadata_completion_tokens']}")
        if row["cost_within_tolerance"] is False:
            problems.append(f"{row['run_key']}#{row['call_no']}: cost {row['journaled_cost_usd']} vs {row['metadata_cost_usd']}")
        if row["provider_ok"] is False:
            problems.append(f"{row['run_key']}#{row['call_no']}: provider {row['provider_names']} != {row['expected_provider']}")
    payment = upstream_402 or after.payment_required or before_snap.payment_required
    if payment:
        decision = STOP_NOTIFY
    elif problems:
        decision = MISMATCH
    else:
        decision = CLEAN
    return {
        "layout": LAYOUT, "stage": stage, "phase": "after", "recorded_at": after.recorded_at,
        "before": before["account"], "after": after.to_json(),
        "key_delta_usd": str(key_delta), "ledger_before": before["ledger"], "ledger_after": dict(ledger_after),
        "ledger_delta_usd": str(ledger_delta), "key_minus_ledger_usd": str(diff), "key_tolerance_usd": str(key_tolerance_usd),
        "generation_sample": {"share": str(SAMPLE_SHARE), "seed": seed, "population": population,
                              "population_source": population_source, "sampled": len(sample_rows), "rows": list(sample_rows)},
        "upstream_402": bool(upstream_402), "problems": problems, "decision": decision,
    }


def read_report(paths: RunPaths, stage: str) -> dict[str, Any]:
    target = report_path(paths, stage)
    if not target.exists():
        raise ReconcileError(f"no reconciliation for stage {stage!r} ({target})")
    doc = json.loads(target.read_text())
    if doc.get("layout") != LAYOUT or doc.get("phase") != "after":
        raise ReconcileError(f"{target} is not a {LAYOUT} reconciliation")
    return doc


def upstream_402_seen(runs: Any) -> bool:
    """True when any run row records an upstream payment refusal (``status == "transport_failed"`` with an error
    naming ``payment_required`` / 402, or a boolean ``upstream_402`` column)."""
    if runs is None or len(runs) == 0:
        return False
    if "upstream_402" in runs.columns:
        return bool(runs["upstream_402"].fillna(False).astype(bool).any())
    if "error" not in runs.columns:
        return False
    errors = runs["error"].dropna().astype(str).str.lower()
    return bool((errors.str.contains("payment_required") | errors.str.contains("402")).any())


async def reconcile_after(cfg: Mapping[str, Any], paths: RunPaths, stage: str, *, client: Any,
                          population: Sequence[CallGenerations], population_source: str, upstream_402: bool,
                          key_tolerance_usd: Decimal, seed: int | None = None) -> dict[str, Any]:
    """After-snapshot + generation sample + report, written to ``state/reconciliation_<stage>.json``."""
    before = read_before(paths, stage)
    after = await snapshot_account(client)
    seed = int(cfg.get("seed", 0)) if seed is None else seed
    sample = sample_calls(population, seed=seed)
    metadata = await fetch_generations(client, (g for c in sample for g in c.generation_ids))
    rows = [compare_call(c, metadata) for c in sample]
    report = build_report(stage=stage, before=before, after=after, ledger_after=ledger_totals(paths), sample_rows=rows,
                          population=len(population), seed=seed, population_source=population_source,
                          upstream_402=upstream_402, key_tolerance_usd=key_tolerance_usd)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    _write(report_path(paths, stage), report)
    return report


# ------------------------------------------------------------------------------------------------------------------
# Population loading for the verb
# ------------------------------------------------------------------------------------------------------------------

def load_population(paths: RunPaths, *, stage: str) -> tuple[list[CallGenerations], str, bool]:
    """(population, source, upstream_402): the tables when WP-C wrote them, else the call journals."""
    runs_path = paths.tables_dir / "dynorch_runs.parquet"
    calls_path = paths.tables_dir / "dynorch_calls.parquet"
    if runs_path.exists() and calls_path.exists():
        import pandas as pd
        from agent_allocation.dynorch.endpoint_rule import read_locks
        runs = pd.read_parquet(runs_path)
        calls = pd.read_parquet(calls_path)
        if "partition" in runs.columns:
            runs = runs[runs["partition"] == stage]
            calls = calls[calls["run_key"].isin(set(runs["run_key"]))]
        locks = read_locks(paths.run_dir) if paths.endpoint_lock.exists() else {}
        return calls_from_table(calls, runs, locks), "tables", upstream_402_seen(runs)
    return calls_from_journal(paths.run_dir), "journal", False


KEY_TOLERANCE_BY_STAGE: dict[str, Decimal] = {"canary": Decimal("0.02"), "pilot": Decimal("0.02")}


def key_tolerance_for(stage: str, ledger_delta_hint: Decimal | None = None) -> Decimal:
    """Plan §9: canary/pilot ``/key`` delta within $0.02; main/extensions within $0.50 or 2 % (Gate 4)."""
    if stage in KEY_TOLERANCE_BY_STAGE:
        return KEY_TOLERANCE_BY_STAGE[stage]
    base = Decimal("0.50")
    if ledger_delta_hint is not None:
        base = max(base, abs(ledger_delta_hint) * Decimal("0.02"))
    return base


def make_client() -> Any:
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    return OpenRouterClient(api_key=load_api_key())


def cmd_dynorch_reconcile(argv: list[str]) -> int:
    """``allocation dynorch-reconcile --config CFG --run-dir RUN --stage S --phase before|after``.
    Exit 0 clean, 3 stop_notify (an upstream 402), 1 mismatch."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-reconcile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["canary", "pilot", "main", "ext_e0", "ext_e1", "ext_e2", "ext_e3"])
    parser.add_argument("--phase", choices=["before", "after"], default="after")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    client = make_client()
    try:
        if args.phase == "before":
            snap = asyncio.run(snapshot_account(client))
            target = write_before(paths, args.stage, snap)
            print(json.dumps({"written": str(target), "payment_required": snap.payment_required}))
            return 3 if snap.payment_required else 0
        population, source, upstream_402 = load_population(paths, stage=args.stage)
        totals = ledger_totals(paths)
        before = read_before(paths, args.stage)
        hint = ledger_charged(totals) - Decimal(before["ledger_charged_usd"])
        report = asyncio.run(reconcile_after(cfg, paths, args.stage, client=client, population=population,
                                             population_source=source, upstream_402=upstream_402,
                                             key_tolerance_usd=key_tolerance_for(args.stage, hint)))
    finally:
        asyncio.run(client.aclose())
    summary = {k: report[k] for k in ("stage", "key_delta_usd", "ledger_delta_usd", "key_minus_ledger_usd", "decision", "problems")}
    summary["sampled"] = report["generation_sample"]["sampled"]
    print(redact(json.dumps(summary, indent=1, sort_keys=True)), file=sys.stdout)
    return {CLEAN: 0, STOP_NOTIFY: 3}.get(report["decision"], 1)
