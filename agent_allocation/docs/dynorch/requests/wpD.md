# WP-D interface requests (ops → WP-A/B/C)

WP-D codes against the contract and imports every WP-A/B/C module lazily inside its verbs; a missing entry point
reports `blocked` (exit 2) with the name below. Each item names the caller, the signature WP-D calls, and why.

## To WP-B (`dynorch/runner.py`, `dynorch/plan.py`, `dynorch/grade.py`)

1. **`runner.launch_stage`** — called by `dynorch/cli.py::launch_stage` (verb `dynorch-run`):
   ```python
   async def launch_stage(cfg, paths: RunPaths, *, stage: str, cells: list[dict], max_usd: Decimal,
                          active_ceiling_usd: Decimal, resume: bool, concurrency: int | None, client: OpenRouterClient,
                          account_auto_reload: bool, proxy_host: str, proxy_port: int,
                          on_bound: Callable[[str, int], Any]) -> dict
   ```
   Expected behaviour: build `ProxyCore` + `Upstream` (locks from `endpoint_rule.read_locks(paths.run_dir)`, per-item
   shard from `locks/provider_map.json`), bind `httpd` on `proxy_host:proxy_port` (`127.0.0.1:0`), call
   `on_bound(host, actual_port)` **before the first harness process** (WP-D writes `state/proxy.json` there), run the
   `cells` (each `{arm, rung, item_id, benchmark, repeat, partition}`; `plan_runs` may re-derive `RunSpec`s from them)
   with `Runner.run_stage(stage, resume=resume)`, and return a JSON-able summary containing at least
   `{"runs": int, "done": int, "failed": int, "timeout": int, "transport_failed": int, "upstream_402": bool,
   "settled_usd": str}`. `active_ceiling_usd` is `live.active_ceiling(max_usd, settled, key_snapshot,
   account_auto_reload)` computed by WP-D at launch; the ledger's `set_active_ceiling` refresh every 5 min stays in
   the runner. Drain on SIGTERM (`--signal=B:TERM@900`) and on `control/STOP` must return normally (exit 0 path).
2. **`plan.plan_runs(cfg, *, stage, arms, rungs, repeats, items)`** — called by `dynorch-plan --write` exactly as in
   contract §6; WP-D writes the returned specs to `manifests/dynorch_plan_<stage>.jsonl` via `dataclasses.asdict`.
   The cell enumeration WP-D uses for dry runs and Gate 4 (`cli.enumerate_cells`) is: canary = one dev item per
   `dynorch.canary.items` benchmark × {H1,H3,H0} @ rung 4 + the same items × {H1,H3} @ rung 16 (10 cells); pilot =
   8 dev items per benchmark × {H1,H3} × {2,16} + 4 dev items per benchmark × H0 @ 16 (144); main = 96 × {H1,H3,H0}
   × 6 rungs (1,728); E0 = 96 × {H1c,H3c} × {4,16}; E1 = 96 × {H1,H3} × 6 rungs, repeat 2; E2 = {H1n,H3n} @ {4,16};
   E3 = {H2} @ {4,16}. Please keep `plan_runs` cell-for-cell identical (Gate 4 checks planned cells against rows).
3. **`grade.candidate_from_text(text, task, *, node_id)`** — used by `cli.regrade_mismatches` (Gate 4 grader re-run)
   with `grading/text/grade.py::grade_text`; the contract §6 signature suffices.

## To WP-C (`dynorch/tables.py`, `dynorch/forecast.py`)

4. **`tables.build_tables(cfg, paths: RunPaths) -> dict`** — called by `dynorch-tables` and at the end of
   `dynorch-run`. Writes `tables/dynorch_runs.parquet`, `tables/dynorch_calls.parquet`,
   `tables/policy_outcomes_dynorch.parquet`, `tables/policy_costs_dynorch.parquet`. WP-D's gates read these columns:
   - runs: `run_key, arm, harness, item_id, benchmark, rung, cap_tokens, repeat, attempt, partition, status, calls,
     spawns, completion_total, grace_tokens, forced_close_kind, exhausted, guard_triggered, tool_parse_failures,
     reinjection_hits, reinjection_misses, final_message, parse_status, grade_status, correct, wall_seconds,
     cost_usd, shard_id, endpoint_tag` (+ optional `error`, `upstream_402`); `partition` = the stage name.
   - calls: `run_key, call_no, conversation_class, granted_max_tokens, completion_tokens, prompt_tokens,
     grace_tokens, forced_close_kind, tool_calls, spawn_calls, tool_parse, reinjection, wire, duplicate_served,
     generation_ids (list[str]), cost_usd`.
   Semantics assumed (from plan B5): `completion_tokens` / `completion_total` exclude the grace; `grace_tokens` is
   separate; a terminal call's `granted_max_tokens` is G; `tool_parse ∈ {none, ok, failed}`; `parse_status == "VALID"`
   for a delivered `<final>`; `grade_status == "error"` for an evaluator error; `status == "timeout"` for wall
   timeouts.
5. **`forecast.gate2_refit(cfg, runs: pd.DataFrame, *, spent_usd: Decimal) -> dict`** returning
   `{"core_forecast_usd": str, "spent_usd": str, "cuts_applied": list[str]}` with the 25 % contingency included and at
   most cuts C1–C3 applied (plan §8); WP-D's Gate 2 criterion is `core_forecast_usd <= soft_ceiling − spent_usd`.
6. **`forecast.stage_forecast(cfg, cells: list[dict]) -> dict`** (optional) for the dry-run plan's cost line.

## To WP-A (`dynorch/journal.py`)

7. The call journal payload (contract §5) is read by `reconcile.calls_from_journal` as the fallback population when
   the tables are absent: `layout == "dynorch/v1"`, `run_key`, `call_no`, `phases[].generation_id`,
   `budget.realized` (counted completion), and — request — `reply.grace_tokens` so the journal fallback can count a
   forced-close pair once. Every journaled byte must already be `apikey.redact`-ed (Gate 1 scans the run tree,
   gzip included, for key-shaped strings).

## Lock file layout WP-D writes (for WP-A/B readers)

`<run_root>/locks/openrouter_endpoints.json` = `{"layout": "dynorch/v1", "model", "decision": {action, reason,
shard_to_tag, verdicts}, "locks": {shard_id: EndpointLock record}, "lock_hashes": {shard_id: hash}, "updated_at"}`
(read with `endpoint_rule.read_locks(run_root) -> {shard_id: EndpointLock}`), and `locks/provider_map.json` =
`{"layout": "dynorch/v1", "items": {item_id: {forced_v2_shard, shard_id, endpoint_tag}}, "decision"}`. After a
switch every item's `shard_id` is the passing shard; `forced_v2_shard` keeps the base assignment as the covariate.
`state/proxy.json` = `{host, port, pid, stage, bound_at, base_url_template}`.

## Test-tree note

`tests/unit/dynorch/__init__.py` (WP-A owned) is an empty placeholder in this worktree so WP-D's tests import as a
package; WP-A's version supersedes it at merge (an empty-file conflict).
