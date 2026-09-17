"""Ledger reconciliation report: the fsync'd journal against the persisted responses (``raw/``, ``parsed/``).

``ledger_reconciliation(paths)`` answers, from durable artifacts only:

* totals by resolution kind (settled from usage, charged at maximum, released unsent, still open) and by partition;
* successful-generation charges (the accepted attempt of each node) versus retry/failed-attempt charges;
* discrepancies - a settlement that differs from the cost recorded in the attempt's raw response, a raw response
  with no journal reservation, an accepted result whose attempt the journal never saw, open reservations.

A clean run has ``discrepancies == []`` and ``open_reservations_usd == "0"``. Provider-side figures (``/key`` usage,
``/credits``) are account-level and may include unrelated spending; pass them as ``provider`` to have the report
show ``provider_minus_study_usd`` without treating the difference as study cost. Config keys consumed: none.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Mapping

from agent_allocation.contracts import LedgerEventKind
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.ledger import confirmed_cost, journal_attempts
from agent_allocation.storage.raw import iter_raw_files, read_raw_response, result_from_raw, sha256_file
from agent_allocation.storage.state import iter_parsed_files, read_parsed

ZERO = Decimal("0")


def ledger_reconciliation(paths: RunPaths, *, provider: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attempts = journal_attempts(paths.ledger_journal)
    raw_costs: dict[str, Decimal | None] = {}
    raw_status: dict[str, str] = {}
    for file in iter_raw_files(paths):
        relative = paths.relative(file)
        result = result_from_raw(read_raw_response(paths, relative), relative_path=relative, sha256=sha256_file(file))
        if result is None:
            continue
        raw_costs[result.attempt_id] = confirmed_cost(result)
        raw_status[result.attempt_id] = result.transport_status.value
    accepted: dict[str, str] = {}
    for file in iter_parsed_files(paths):
        result, _ = read_parsed(file)
        accepted[result.node_id] = result.attempt_id
    accepted_attempts = set(accepted.values())

    by_kind: dict[str, Decimal] = defaultdict(lambda: ZERO)
    by_partition: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(lambda: ZERO))
    successful = retry = ZERO
    discrepancies: list[dict[str, Any]] = []
    open_total = ZERO
    for aid, info in sorted(attempts.items()):
        part = info.partition_id or "unknown"
        if info.resolution is None:
            open_total += info.reserved_usd
            by_partition[part]["open_reservations"] += info.reserved_usd
            discrepancies.append({"attempt_id": aid, "issue": "open_reservation", "reserved_usd": str(info.reserved_usd)})
            continue
        amount = info.resolved_usd or ZERO
        by_kind[info.resolution.value] += amount
        if info.resolution is LedgerEventKind.RELEASE_UNSENT:
            by_partition[part]["released_unsent"] += amount
            continue
        by_partition[part]["charged"] += amount
        if aid in accepted_attempts:
            successful += amount
        else:
            retry += amount
        if info.resolution is LedgerEventKind.SETTLE and aid in raw_costs:
            recorded = raw_costs[aid]
            if recorded is not None and recorded != amount:
                discrepancies.append({"attempt_id": aid, "issue": "settled_differs_from_raw_usage",
                                      "settled_usd": str(amount), "raw_cost_usd": str(recorded)})
    for aid in sorted(set(raw_costs) - set(attempts)):
        discrepancies.append({"attempt_id": aid, "issue": "raw_response_without_reservation",
                              "status": raw_status.get(aid)})
    for node_id, aid in sorted(accepted.items()):
        if aid not in attempts:
            discrepancies.append({"attempt_id": aid, "issue": "accepted_result_without_reservation", "node_id": node_id})

    charged_total = by_kind.get(LedgerEventKind.SETTLE.value, ZERO) + by_kind.get(LedgerEventKind.CHARGE_MAX.value, ZERO)
    report: dict[str, Any] = {
        "attempts": len(attempts),
        "raw_responses": len(raw_costs),
        "accepted_results": len(accepted),
        "charged_usd": str(charged_total),
        "settled_from_usage_usd": str(by_kind.get(LedgerEventKind.SETTLE.value, ZERO)),
        "charged_at_maximum_usd": str(by_kind.get(LedgerEventKind.CHARGE_MAX.value, ZERO)),
        "released_unsent_usd": str(by_kind.get(LedgerEventKind.RELEASE_UNSENT.value, ZERO)),
        "open_reservations_usd": str(open_total),
        "successful_generation_charges_usd": str(successful),
        "retry_and_failed_attempt_charges_usd": str(retry),
        "by_partition": {p: {k: str(v) for k, v in d.items()} for p, d in sorted(by_partition.items())},
        "discrepancies": discrepancies,
    }
    if provider is not None and provider.get("usage_usd") is not None:
        provider_usage = Decimal(str(provider["usage_usd"]))
        report["provider_usage_usd"] = str(provider_usage)
        report["provider_minus_study_usd"] = str(provider_usage - charged_total)
    return report
