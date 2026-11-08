# Cost reconciliation: `openrouter_forced_budget_v2`

**INTERIM REPORT: PARTIAL COLLECTION.** This report is partial and not a final result:

- 80 checkpoint rows lack a known delivered outcome (unrun, unselected, unsealed or ungraded)
- costs are not reconciled against provider usage (see COST_RECONCILIATION.md)

- Run directory: `merged`
- Input digest (sha256 over every input file): `2169fc79139832bf79d587f161e8ed4576d190ae225b591eefd432eaced42595`
- Config source: base study config; merged from shards ['mancer', 'parasail']
- Bootstrap replicates: 10000 (item-clustered within benchmark strata)

**Status: NOT RECONCILED.** A final run report requires reconciled costs.

## Collection ledger (journal replay)

| quantity | value |
|---|---|
| events | 42814 |
| settled_usd | 70.06987436 |
| charged_max_usd | 2.194846940 |
| charged_total_usd | 72.264721300 |
| released_unsent_usd | 0.873922280 |
| in_flight_usd | 0E-9 |
| unresolved_usd | 0E-9 |
| open_attempts | 0 |
| peak_exposure_usd | 72.382838788 |
| ceiling_usd | 95 |
| invariant_violations | 0 |

### Per partition

| partition | reserved | settled | charged_max | released |
|---|---|---|---|---|
| calibration | 1.808866048 | 0.45780884 | unknown | unknown |
| calibration_natural | 2.696545984 | 1.30021108 | unknown | unknown |
| canary | 1.261405200 | 0.70369408 | unknown | unknown |
| gate0 | 0.218909372 | 0.06510122 | 0.01151106 | unknown |
| main | 211.406847432 | 65.24760268 | 1.017333768 | 0.71484820 |
| pilot | 8.811440616 | 2.29545646 | 1.166002112 | 0.159074080 |

## Accepted generation usage versus ledger settlements

| quantity | value |
|---|---|
| accepted_usage_usd | 69.99915064 |
| accepted_unknown_usage | 0 |
| accepted_missing_settlement | 0 |
| attempt_mismatches | 0 |
| non_accepted_attempt_charges_usd | 2.265570660 |

## Provider `/key` usage snapshots

| quantity | value |
|---|---|
| status | discrepancy |
| first_fetched_at | 2026-09-18T09:22:08.127396+00:00 |
| last_fetched_at | 2026-09-18T17:17:22.370258+00:00 |
| first_stage | canary_before |
| last_stage | final_after_all_shards |
| provider_usage_delta_usd | 70.32789552 |
| ledger_charges_in_window_usd | 72.264721300 |
| difference_usd | -1.936825780 |
| tolerance_usd | 0.72264721300 |
| snapshots | 20 |
| ledger_charges_with_unparseable_time | 0 |

A shared account's unrelated spending appears as an unexplained provider-side difference; it is never attributed to study policies. Policy costs in RUN_REPORT use each checkpoint's own ancestors, not this collection total.
