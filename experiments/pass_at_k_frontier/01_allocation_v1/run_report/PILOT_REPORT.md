# PILOT_REPORT — openrouter_agent_allocation_v1

Evaluated 2026-09-17T15:33:45Z. **Gate 2 decision: `stop_notify`.**

## Gate 2 criteria (frozen before the canary)

| Criterion | Observed | Threshold | Pass |
|---|---|---|---|
| math500_valid_final | `{"n": 108, "share": 0.907}` | >= 0.95 | **no** |
| math500_truncated_without_final | `{"n": 108, "share": 0.093}` | <= 0.03 | **no** |
| mmlu_pro_valid_final | `{"n": 108, "share": 0.972}` | >= 0.95 | yes |
| mmlu_pro_truncated_without_final | `{"n": 108, "share": 0.028}` | <= 0.03 | yes |
| bbh_logic4_valid_final | `{"n": 108, "share": 1.0}` | >= 0.95 | yes |
| bbh_logic4_truncated_without_final | `{"n": 108, "share": 0.0}` | <= 0.03 | yes |
| mbpp_plus_valid_final | `{"n": 108, "share": 0.963}` | >= 0.95 | yes |
| mbpp_plus_truncated_without_final | `{"n": 108, "share": 0.019}` | <= 0.03 | yes |
| zero_cache_hits | `0` | 0 | yes |
| zero_provider_drift | `0` | 0 | yes |
| resume_no_duplicate_terminal_nodes | `0` | 0 | yes |
| budget_stop_never_over_reserved | `"-0.00008077"` | <= 0 | yes |
| selector_rows_populated | `0` | 0 missing | yes |
| code_public_and_private_rows | `{"candidates": 113, "private": 113, "public": 113}` | every code candidate has both | yes |
| evaluator_error_share | `0.0` | <= 0.01 | yes |
| packet_envelope_respected | `0` | 0 | yes |
| core_forecast_within_soft_ceiling | `"69.74931756"` | <= 125 | yes |

## Measured forecast and admission plan

```json
{
 "account_balance_usd": "81.475436339",
 "active_ceiling_usd": "81.494753899",
 "admission": {
  "high_waves": 6,
  "main_waves": 25
 },
 "available_for_core_usd": "80.475436339",
 "core_forecast_usd": "58.73",
 "core_plus_spent_plus_reserve_usd": "69.74931756",
 "per_item_costs": {
  "bbh_logic4": {
   "extension": "0.00712806815578817703450",
   "high": "0.13405971005747126407335",
   "main": "0.08088938178571428297745"
  },
  "math500": {
   "extension": "0.01144686704946633806600",
   "high": "0.23390465977011494388025",
   "main": "0.13965259928571428500375"
  },
  "mbpp_plus": {
   "extension": "0.00791315457922824347300",
   "high": "0.16888369810344827591025",
   "main": "0.08704941976190476493775"
  },
  "mmlu_pro": {
   "extension": "0.0087692910098522159040",
   "high": "0.1350609965517241378775",
   "main": "0.1014355419047618980125"
  }
 },
 "reserve_usd": "10",
 "spent_usd": "1.01931756"
}
```

## Integrity checks

```json
{
 "amendments_used": "['headroom_text', 'headroom_text_owner_4096', 'prompt_system_v2']",
 "budget_stop_max_overshoot_usd": "-0.00008077",
 "code_candidates": "113",
 "code_private_grades": "113",
 "code_public_rows": "113",
 "duplicate_terminal_nodes": "0",
 "evaluator_errors": "0",
 "graded_total": "452",
 "grader_returncode": "0",
 "packet_envelope_violations": "0"
}
```

## Selection

```json
{
 "checkpoints_missing_candidates": 0,
 "pending_policy_ids_sample": [],
 "seal": "47d3976eb3f7447ca7890e64a9d889cdbac83bf8210de4e85445632f0fdb240f",
 "selections": 800
}
```

Floor/ceiling effects and per-benchmark pilot accuracy are informational and never block.
