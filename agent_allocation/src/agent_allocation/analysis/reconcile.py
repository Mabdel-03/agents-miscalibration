"""Collection-ledger reconciliation (MASTER_SPEC §7.1) for COST_RECONCILIATION.md.

Three independent views of the same money:

1. the ledger journal (``state/ledger_journal.jsonl``): reservations, settlements, conservative max charges,
   releases, and what remains unresolved;
2. the accepted generation records (``usage.actual_cost_usd`` of every parsed result), compared attempt by
   attempt with the ledger settlement of that attempt;
3. provider ``/key`` usage snapshots stored by the lead under ``state/`` (see ``load.load_provider_snapshots``):
   the usage delta between the first and last snapshot versus ledger charges recorded in that window. A shared
   account's unrelated spending would show here as an unexplained positive difference, never as study cost.

The journal replay also checks the atomic admission invariant
``settled + in_flight + unresolved <= credit ceiling`` after every event.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd

from agent_allocation.analysis.load import RunData
from agent_allocation.contracts import LedgerEventKind

TOLERANCE_USD = Decimal("0.01")
TOLERANCE_FRACTION = Decimal("0.01")


def ledger_state(run: RunData) -> dict[str, Any]:
    ceiling = Decimal(str(run.config.get("budget", {}).get("credit_ceiling_usd", "0")))
    reserved: dict[str, Decimal] = {}
    sent: set[str] = set()
    closed: set[str] = set()
    settled = Decimal(0)
    charged_max = Decimal(0)
    released = Decimal(0)
    in_flight = Decimal(0)        # reserved, not sent, not closed
    unresolved = Decimal(0)       # sent, not closed
    per_partition: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    violations: list[str] = []
    peak_exposure = Decimal(0)
    last_seq = 0
    exceeded = 0

    def close(attempt: str) -> None:
        nonlocal in_flight, unresolved
        if attempt in closed or attempt not in reserved:
            closed.add(attempt)
            return
        if attempt in sent:
            unresolved -= reserved[attempt]
        else:
            in_flight -= reserved[attempt]
        closed.add(attempt)

    for event in run.ledger:
        if event.seq <= last_seq:
            violations.append(f"ledger seq {event.seq} not increasing after {last_seq}")
        last_seq = max(last_seq, event.seq)
        attempt = event.attempt_id or ""
        part = event.partition_id or "unknown"
        if event.kind == LedgerEventKind.RESERVE:
            if attempt in reserved:
                violations.append(f"attempt {attempt} reserved twice")
            else:
                reserved[attempt] = event.amount_usd
                in_flight += event.amount_usd
            per_partition[part]["reserved"] += event.amount_usd
        elif event.kind == LedgerEventKind.SENT:
            if attempt in reserved and attempt not in sent and attempt not in closed:
                in_flight -= reserved[attempt]
                unresolved += reserved[attempt]
            elif attempt not in reserved:
                violations.append(f"attempt {attempt} sent without a reservation")
            sent.add(attempt)
        elif event.kind == LedgerEventKind.SETTLE:
            settled += event.amount_usd
            close(attempt)
            per_partition[part]["settled"] += event.amount_usd
        elif event.kind == LedgerEventKind.CHARGE_MAX:
            charged_max += event.amount_usd
            close(attempt)
            per_partition[part]["charged_max"] += event.amount_usd
        elif event.kind == LedgerEventKind.RELEASE_UNSENT:
            if attempt in sent:
                violations.append(f"attempt {attempt} released as unsent after a SENT event")
            released += reserved.get(attempt, Decimal(0))
            per_partition[part]["released"] += reserved.get(attempt, Decimal(0))
            close(attempt)
        exposure = settled + charged_max + in_flight + unresolved
        peak_exposure = max(peak_exposure, exposure)
        if ceiling > 0 and exposure > ceiling:
            exceeded += 1
            if exceeded == 1:
                violations.append(f"after ledger seq {event.seq}: exposure {exposure} exceeds ceiling {ceiling}")
    if exceeded > 1:
        violations.append(f"exposure exceeded the ceiling after {exceeded} ledger events in total")
    open_attempts = [a for a in reserved if a not in closed]
    return {
        "events": len(run.ledger),
        "settled_usd": settled,
        "charged_max_usd": charged_max,
        "charged_total_usd": settled + charged_max,
        "released_unsent_usd": released,
        "in_flight_usd": in_flight,
        "unresolved_usd": unresolved,
        "open_attempts": len(open_attempts),
        "peak_exposure_usd": peak_exposure,
        "ceiling_usd": ceiling,
        "invariant_violations": violations,
        "per_partition": {p: dict(v) for p, v in sorted(per_partition.items())},
    }


def usage_vs_ledger(run: RunData) -> dict[str, Any]:
    settle_by_attempt: dict[str, Decimal] = defaultdict(Decimal)
    for event in run.ledger:
        if event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX) and event.attempt_id:
            settle_by_attempt[event.attempt_id] += event.amount_usd
    mismatches: list[str] = []
    usage_total = Decimal(0)
    unknown_usage = 0
    missing_settlement = 0
    for node_id, (result, _cand) in sorted(run.parsed.items()):
        cost = result.usage.actual_cost_usd if result.usage else None
        if cost is None:
            unknown_usage += 1
            continue
        usage_total += cost
        ledger_amount = settle_by_attempt.get(result.attempt_id)
        if ledger_amount is None:
            missing_settlement += 1
            continue
        if abs(ledger_amount - cost) > Decimal("1e-9"):
            mismatches.append(f"{result.attempt_id}: usage {cost} vs ledger {ledger_amount}")
    accepted = {r.attempt_id for r, _ in run.parsed.values()}
    retry_total = sum((v for a, v in settle_by_attempt.items() if a not in accepted), Decimal(0))
    return {"accepted_usage_usd": usage_total, "accepted_unknown_usage": unknown_usage,
            "accepted_missing_settlement": missing_settlement, "attempt_mismatches": mismatches,
            "non_accepted_attempt_charges_usd": retry_total}


def parse_instant(value: Any) -> datetime | None:
    """ISO-8601 instant as an aware UTC datetime (``Z`` or offsets, any fractional precision); None if unparseable."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def provider_reconciliation(run: RunData) -> dict[str, Any]:
    snaps = [s for s in run.provider_snapshots if str(s.get("endpoint", "key")) == "key"
             and isinstance(s.get("data"), dict) and s["data"].get("usage") is not None
             and parse_instant(s.get("fetched_at")) is not None]
    snaps.sort(key=lambda s: parse_instant(s["fetched_at"]))
    if len(snaps) < 2:
        return {"status": "unavailable", "detail": f"{len(snaps)} timestamped provider /key usage snapshot(s) under "
                                                    "state/; need a baseline and a later snapshot"}
    first, last = snaps[0], snaps[-1]
    provider_delta = Decimal(str(last["data"]["usage"])) - Decimal(str(first["data"]["usage"]))
    lo, hi = parse_instant(first["fetched_at"]), parse_instant(last["fetched_at"])
    unparsed = 0
    window = Decimal(0)
    for e in run.ledger:
        if e.kind not in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX):
            continue
        at = parse_instant(e.recorded_at)
        if at is None:
            unparsed += 1
            continue
        if lo <= at <= hi:
            window += e.amount_usd
    diff = provider_delta - window
    tolerance = max(TOLERANCE_USD, abs(window) * TOLERANCE_FRACTION)
    status = "reconciled" if abs(diff) <= tolerance and not unparsed else "discrepancy"
    return {"status": status, "first_fetched_at": first["fetched_at"], "last_fetched_at": last["fetched_at"],
            "first_stage": first.get("stage"), "last_stage": last.get("stage"),
            "provider_usage_delta_usd": provider_delta, "ledger_charges_in_window_usd": window,
            "difference_usd": diff, "tolerance_usd": tolerance, "snapshots": len(snaps),
            "ledger_charges_with_unparseable_time": unparsed}


def reconciliation(run: RunData) -> dict[str, Any]:
    state = ledger_state(run)
    usage = usage_vs_ledger(run)
    provider = provider_reconciliation(run)
    reconciled = (provider["status"] == "reconciled" and state["unresolved_usd"] == 0 and state["in_flight_usd"] == 0
                  and not usage["attempt_mismatches"] and not state["invariant_violations"]
                  and usage["accepted_missing_settlement"] == 0)
    return {"ledger": state, "usage": usage, "provider": provider, "reconciled": reconciled}


def reconciliation_frame(rec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in rec["ledger"].items():
        if key in ("per_partition", "invariant_violations"):
            continue
        rows.append({"section": "ledger", "quantity": key, "value": str(value)})
    rows.append({"section": "ledger", "quantity": "invariant_violations",
                 "value": str(len(rec["ledger"]["invariant_violations"]))})
    for part, values in rec["ledger"]["per_partition"].items():
        for key, value in sorted(values.items()):
            rows.append({"section": f"ledger:{part}", "quantity": key, "value": str(value)})
    for key, value in rec["usage"].items():
        rows.append({"section": "usage_vs_ledger", "quantity": key,
                     "value": str(len(value)) if isinstance(value, list) else str(value)})
    for key, value in rec["provider"].items():
        rows.append({"section": "provider_key_usage", "quantity": key, "value": str(value)})
    rows.append({"section": "summary", "quantity": "reconciled", "value": str(rec["reconciled"])})
    return pd.DataFrame(rows)
