# Audit: `openrouter_agent_allocation_v1`

- Input digest: `9fbd6a3a15330a35e26415537bd2478a3af956c3dfb2efe0e0a1da19ca0a2875`
- Checks: 29 pass, 0 fail, 1 skip

## Checks

| check | status | detail | evidence | n_violations |
|---|---|---|---|---|
| fixed_population | pass | every checkpoint in each claim partition x benchmark x view covers the same planned items | manifests/policies/*.jsonl, manifests/waves.json | 0 |
| sealed_before_join | pass | batch seals verify for ['high (1 batches)', 'main (1 batches)', 'pilot (1 batches)', 'replication (1 batches)']; grades joined only to verified batches | selections/{partition}/batch_NNNN.jsonl + .seal.json | 0 |
| selector_legality | pass | 101088 sealed selections use a legal concrete selector for their declared family and checkpoint shape | manifests/policies/, selections/ | 0 |
| unknown_outcomes_not_coerced | pass | 101088 rows: unknown and unrun states stay NaN | tables/policy_outcomes | 0 |
| delivered_within_coverage_candidate_only | pass | 100704 candidate-only rows checked; synthesis rows exempt (0 POOLED rows exceed input-bank coverage, which is allowed) | tables/policy_outcomes | 0 |
| archive_coverage_monotone | pass | 42320 consecutive archive pairs, 50544 archive/current pairs | tables/policy_outcomes | 0 |
| pass_at_1_equals_mean_draw_correctness | pass | 576 fully known banks | tables/sample_bank | 0 |
| pass_at_bank_size_equals_full_bank_coverage | pass | 576 fully known banks | tables/sample_bank | 0 |
| transition_identity | pass | 29632 known transitions | tables/transitions | 0 |
| policy_ancestor_counts | pass | 50544 checkpoints charge n(1+R)-style ancestor counts | tables/policy_costs | 0 |
| cumulative_costs_monotone | pass | 253920 consecutive-prefix comparisons | tables/policy_costs | 0 |
| primary_populations_fixed | pass | primary populations equal the planned item sets | tables/primary_contrasts | 0 |
| price_units | pass | google/gemini-2.5-flash-lite: in 0.10/M out 0.40/M; google/gemini-2.5-flash: in 0.30/M out 2.50/M; median actual/standardized = 1.0000 over 67513 generations (a unit error shows as ~1e6 or ~1e-6; cache discounts and price drift move it moderately) | planning_prices, endpoint lock | 0 |
| ledger_invariants | pass | 205985 ledger events; peak exposure 48.09456828 <= ceiling 135 | state/ledger_journal.jsonl | 0 |
| agent06_metric_tables_agree | skip | no agent 06 metric tables under tables/metrics/ | tables/metrics/ | 0 |
| independent_recompute_contrasts | pass | 2 contrast(s) compared | tests/integration/independent_recompute.py | 0 |
| independent:manifest_node_identity | pass | 67514 job records hash to their node_id | tests/integration/independent_recompute.py | 0 |
| independent:raw_matches_parsed | pass | 67514 parsed records re-verified against raw | tests/integration/independent_recompute.py | 0 |
| independent:ancestor_closure | pass | required_node_ids equal the parent closure of every checkpoint pool | tests/integration/independent_recompute.py | 0 |
| independent:selection_seals | pass | 101088 sealed selection rows verified | tests/integration/independent_recompute.py | 0 |
| independent:selections_within_pool | pass | every candidate-only selection names a member of its declared pool | tests/integration/independent_recompute.py | 0 |
| independent:selector_rows_present | pass | every fully generated current checkpoint has a selector row | tests/integration/independent_recompute.py | 0 |
| independent:delivered_le_coverage | pass | candidate-only delivered accuracy never exceeds pool coverage (POOLED exempt) | tests/integration/independent_recompute.py | 0 |
| independent:fixed_population | pass | each checkpoint family covers one fixed item set per partition | tests/integration/independent_recompute.py | 0 |
| independent:pass_at_k_identities | pass | pass@1 = mean correctness; pass@25 = coverage | tests/integration/independent_recompute.py | 0 |
| independent:transition_identity | pass | C_next - C_prev = W_to_C - C_to_W | tests/integration/independent_recompute.py | 0 |
| independent:archive_coverage_monotone | pass | archive coverage never decreases | tests/integration/independent_recompute.py | 0 |
| independent:invalid_not_graded | pass | unparseable outputs are INVALID_OUTPUT, never a graded wrong answer | tests/integration/independent_recompute.py | 0 |
| independent:ledger_replay | pass | journal replays within ceiling 135 | tests/integration/independent_recompute.py | 0 |
| independent:no_credentials_in_artifacts | pass | no OpenRouter key pattern in any run artifact | tests/integration/independent_recompute.py | 0 |

