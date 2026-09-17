"""Report generation for openrouter_agent_allocation_v1 (agent 07): ``analyze`` and ``audit`` verbs.

Exported tables (``paths.tables_dir``, each as ``.parquet`` and ``.csv``; key columns):

* ``policy_outcomes`` - one row per planned checkpoint: ``partition_id, policy_id, item_id, benchmark, subtask,
  stratum, name, view, family, n_pool, n_required, n_required_terminal, coverage, input_coverage,
  selection_status, selection_batch, selector_concrete, selected_candidate_id, selected_in_pool, sealed,
  delivered_state, y`` (``y`` 1/0/NaN; unknown and unrun states are NaN, never 0).
* ``candidates`` - one row per planned generation node with usage, statuses, grade state and render report.
* ``accuracy_summary`` - ``partition_id, name, view, level (stratum|benchmark|macro), benchmark, stratum,
  n_planned, n_correct, n_wrong, n_op_failure, n_unknown, n_not_run, acc, acc_lower, acc_upper, acc_known_only,
  ci_lo, ci_hi, ci_basis, coverage, selection_gap``.
* ``primary_contrasts`` - ``contrast, role (primary|descriptive|exploratory), panel, level, benchmark, n_items,
  n_complete, status (complete|partial|not_estimable*), estimate, se, ci95_lo, ci95_hi, ci95_percentile_lo,
  ci95_percentile_hi, bound_lower, bound_upper, p_value, p_holm``.
* ``high_vs_main_shared_items`` - the same columns plus ``variant`` (high|replication), shared ids only.
* ``policy_costs`` - one row per checkpoint: nominal generations, requested reasoning/caps, realized tokens,
  ``success_cost_usd, retry_cost_usd, total_charged_usd, standardized_uncached_cost_usd``, CPU and latency.
* ``cost_summary``, ``frontier_descriptive``, ``pass_at_k_vs_readout``, ``sample_bank_items``,
  ``transitions_items``, ``transitions_summary``, ``budget_replay_items``, ``budget_replay_curves``,
  ``operations``, ``billed_retries``, ``cost_reconciliation``, ``price_unit_checks``, ``completeness``,
  ``node_completeness``, ``populations``; ``audit_checks`` (from ``audit``).
"""
