# PILOT_REPORT — openrouter_dynorch_v1

Evaluated 2026-09-24T07:30:04Z; decision **stop_notify** (passed=False).

## Gate 2 criteria

| criterion | observed | threshold | passed |
|---|---|---|---|
| H1_valid_final_share | `{"n": 64, "share": 0.9062, "valid": 58}` | >= 0.9 (>= 0.8: stop_notify; below: stop) | yes |
| H1_tool_call_parse_failure_share | `{"calls": 0, "share": 0.0}` | <= 0.05 | yes |
| H1_cap_overrun_runs | `[]` | 0 | yes |
| H1_grace_overrun_runs | `[]` | 0 | yes |
| H1_spawn_share_k16 | `{"runs_k16": 32, "share": 0.0, "spawning": 0}` | >= 0.1 | NO |
| H1_guard_triggered_share | `0.0` | <= 0.05 | yes |
| H1_evaluator_error_share | `{"graded": 58, "share": 0.0}` | <= 0.01 | yes |
| H1_reinjection_hit_rate | `{"hits": 0, "misses": 0, "rate": null}` | >= 0.95 (vacuous with no assistant tool turns) | yes |
| H1_timeout_share | `{"share": 0.0, "timeouts": 0}` | <= 0.02 | yes |
| H3_valid_final_share | `{"n": 64, "share": 0.9844, "valid": 63}` | >= 0.9 (>= 0.8: stop_notify; below: stop) | yes |
| H3_tool_call_parse_failure_share | `{"calls": 2, "share": 0.0}` | <= 0.05 | yes |
| H3_cap_overrun_runs | `[]` | 0 | yes |
| H3_grace_overrun_runs | `[]` | 0 | yes |
| H3_spawn_share_k16 | `{"runs_k16": 32, "share": 0.0, "spawning": 0}` | >= 0.1 | NO |
| H3_guard_triggered_share | `0.0` | <= 0.05 | yes |
| H3_evaluator_error_share | `{"graded": 63, "share": 0.0}` | <= 0.01 | yes |
| H3_reinjection_hit_rate | `{"hits": 0, "misses": 0, "rate": null}` | >= 0.95 (vacuous with no assistant tool turns) | yes |
| H3_timeout_share | `{"share": 0.0, "timeouts": 0}` | <= 0.02 | yes |
| key_delta_within_tolerance | `{"diff": "0E-8", "key_delta": "2.65154310", "ledger_delta": "2.65154310"}` | |key − ledger| <= 0.02 | yes |
| generation_sample_matches | `{"bad": [], "sampled": 2}` | every sampled call: /generation fetched, tokens counted once, cost within $0.01 or 5 % | yes |
| observed_provider_pinned | `[]` | every sampled generation served by the pinned provider | yes |
| no_upstream_402 | `false` | no PAYMENT_REQUIRED from upstream | yes |
| open_reservations_zero | `"0"` | 0 | yes |
| core_forecast_within_soft_ceiling | `{"core_forecast_usd": "70.4665210000000", "cuts_applied": [], "soft_ceiling_usd": "90", "spent_usd": "3.15013060"}` | forecast (+25 %) <= soft ceiling − spent after at most C1–C3 | yes |

## Harness diagnostics

| arm | runs | valid final | parse-failure share | spawn share k=16 | guard | reinjection | timeouts |
|---|---|---|---|---|---|---|---|
| H0 | 16 | 1.000 | 0.000 | 0.000 (0/16) | 0.000 | n/a | 0 |
| H1 | 64 | 0.906 | 0.000 | 0.000 (0/32) | 0.000 | n/a | 0 |
| H3 | 64 | 0.984 | 0.000 | 0.000 (0/32) | 0.000 | n/a | 0 |

Forced-close kinds per arm: H0/mid_call=7, H0/none=9, H1/mid_call=38, H1/none=26, H3/mid_call=38, H3/none=23, H3/visible_continuation=3

## Accuracy pooled per benchmark (no per-item outcomes)

| arm | rung | benchmark | n | accuracy |
|---|---|---|---|---|
| H0 | 16 | aime | 4 | 0.500 |
| H0 | 16 | gpqa_diamond | 4 | 0.750 |
| H0 | 16 | hmmt | 4 | 1.000 |
| H0 | 16 | omni_math_hard | 4 | 1.000 |
| H0 | 16 | macro | 16 | 0.812 |
| H1 | 2 | aime | 8 | 0.375 |
| H1 | 2 | gpqa_diamond | 8 | 0.250 |
| H1 | 2 | hmmt | 8 | 0.375 |
| H1 | 2 | omni_math_hard | 8 | 0.125 |
| H1 | 2 | macro | 32 | 0.281 |
| H1 | 16 | aime | 8 | 0.750 |
| H1 | 16 | gpqa_diamond | 8 | 0.500 |
| H1 | 16 | hmmt | 8 | 0.750 |
| H1 | 16 | omni_math_hard | 8 | 0.750 |
| H1 | 16 | macro | 32 | 0.688 |
| H3 | 2 | aime | 8 | 0.250 |
| H3 | 2 | gpqa_diamond | 8 | 0.500 |
| H3 | 2 | hmmt | 8 | 0.375 |
| H3 | 2 | omni_math_hard | 8 | 0.000 |
| H3 | 2 | macro | 32 | 0.281 |
| H3 | 16 | aime | 8 | 0.875 |
| H3 | 16 | gpqa_diamond | 8 | 0.750 |
| H3 | 16 | hmmt | 8 | 0.750 |
| H3 | 16 | omni_math_hard | 8 | 0.250 |
| H3 | 16 | macro | 32 | 0.656 |

## Reconciliation

- key delta $2.65154310 vs ledger delta $2.65154310 (diff $0E-8); decision clean
- generation sample: 2 of 146 calls
- upstream 402: False

## Re-fitted core forecast

- core forecast (+25 %): $70.4665210000000; spent so far $3.15013060; soft ceiling $90; cuts applied: []

## Owner go/no-go

The pilot never auto-advances. Record the decision with `live.record_decision` (see RUNBOOK: `allocation dynorch-gates --gate 2 ... --owner-go "<text>"`) before `freeze` and the main run.
