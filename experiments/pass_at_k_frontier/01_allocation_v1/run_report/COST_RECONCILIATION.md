# Cost reconciliation: `openrouter_agent_allocation_v1`

**Final: every planned checkpoint is known and costs are reconciled.**

- Run directory: `openrouter_agent_allocation_v1`
- Input digest (sha256 over every input file): `9fbd6a3a15330a35e26415537bd2478a3af956c3dfb2efe0e0a1da19ca0a2875`
- Config source: configs/balanced150.json
- Bootstrap replicates: 10000 (item-clustered within benchmark strata)

**Status: RECONCILED.** A final run report requires reconciled costs.

## Collection ledger (journal replay)

| quantity | value |
|---|---|
| events | 205985 |
| settled_usd | 47.62460042 |
| charged_max_usd | 0.01239601 |
| charged_total_usd | 47.63699643 |
| released_unsent_usd | 0 |
| in_flight_usd | 0E-8 |
| unresolved_usd | 0E-8 |
| open_attempts | 0 |
| peak_exposure_usd | 48.09456828 |
| ceiling_usd | 135 |
| invariant_violations | 0 |

### Per partition

| partition | reserved | settled | charged_max |
|---|---|---|---|
| canary | 0.07117022 | 0.0321442 | unknown |
| high | 43.40352038 | 10.25633033 | 0.00378576 |
| main | 123.87264087 | 28.64325413 | 0.00167882 |
| pilot | 3.10979746 | 0.98024193 | 0.00693143 |
| replication | 38.91500415 | 7.71262983 | unknown |

## Accepted generation usage versus ledger settlements

| quantity | value |
|---|---|
| accepted_usage_usd | 46.93803734 |
| accepted_unknown_usage | 0 |
| accepted_missing_settlement | 0 |
| attempt_mismatches | 0 |
| non_accepted_attempt_charges_usd | 0.69895909 |

## Provider `/key` usage snapshots

| quantity | value |
|---|---|
| status | reconciled |
| first_fetched_at | 2026-09-17T07:37:38.730253+00:00 |
| last_fetched_at | 2026-09-17T16:53:15.311969+00:00 |
| first_stage | doctor |
| last_stage | run_phase_c |
| provider_usage_delta_usd | 39.82607779 |
| ledger_charges_in_window_usd | 39.92436660 |
| difference_usd | -0.09828881 |
| tolerance_usd | 0.3992436660 |
| snapshots | 16 |
| ledger_charges_with_unparseable_time | 0 |

A shared account's unrelated spending appears as an unexplained provider-side difference; it is never attributed to study policies. Policy costs in RUN_REPORT use each checkpoint's own ancestors, not this collection total.
