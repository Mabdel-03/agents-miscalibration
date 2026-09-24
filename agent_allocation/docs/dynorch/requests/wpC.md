# WP-C interface requests (tables, forecast, report)

Everything WP-C codes against that another WP owns, with the shape WP-C assumes. Deviations from these
assumptions break `dynorch/tables.py` loudly (`TablesError`), never silently.

## R1. Plan manifest (WP-B, `dynorch/plan.py`): `<run_root>/manifests/dynorch_plan_<stage>.jsonl`

One JSON object per planned run, one line each, in plan order. WP-C reads it with `tables.read_plan` and accepts
either shape:

```json
{"__record__": "RunSpec", "run_key": "...", "arm": "H1", "harness": "openclaude", "item_id": "aime/2026/11",
 "benchmark": "aime", "stratum": "aime", "shard_id": "parasail", "endpoint_tag": "parasail/fp8", "rung": 16,
 "cap_tokens": 16384, "repeat": 1, "attempt": 1, "generation": 1, "partition": "main",
 "allowed_tools": ["Agent"], "spawn_tools": ["Agent"], "root_call_ceiling": null, "sampling_hash": "...",
 "template_sha256": "...", "prompt_version": "dynorch-v1", "proxy_version": "...", "harness_version": "..."}
```

i.e. `contracts.dump_record(RunSpec)` (a `{"spec": {...}}` wrapper is also accepted). Required keys: `run_key, arm,
harness, item_id, benchmark, stratum, shard_id, endpoint_tag, rung, cap_tokens, repeat, attempt, partition`
(the rest is optional for tables). Rules WP-C relies on:

- exactly one line per (arm, item_id, rung, repeat) cell (attempt 1); rerun attempts do NOT get their own plan
  line (they are matched to the cell by arm/item/rung/repeat from their RunRecord) -- if WP-B prefers to append
  rerun lines, say so and WP-C will de-duplicate by cell instead of refusing duplicates;
- the manifest is written BEFORE the stage starts (every planned cell gets a `pending` outcome row);
- `partition` is `main` for the core, `ext_e0`, `ext_e1`, ... for extensions; `stratum == benchmark` for the four
  text benchmarks.

## R2. RunRecord files (WP-A, `dynorch/journal.py::RunStore`): `<run_root>/runs/<arm>/<rung>/<repeat>/<item_slug>.json`

`contracts.dump_record(RunRecord)` with nested `spec` (RunSpec) and `calls_detail` (list of CallRecord), every
field of contract section 5 present (strict), tuples as lists, `cost_usd` a decimal string, `correct` bool | None,
`parse_status` one of the extraction vocabulary (`VALID` is the only value that yields a delivered answer).
WP-C scans `runs/` recursively, so superseded attempts may stay anywhere below it (e.g. `superseded/`); each
attempt of a cell must carry its own `spec.attempt` (duplicates raise). The cell's current record is the highest
attempt.

Two quantities the sealed cost schema needs that the contract does not provide: `reasoning_total` (used as
`policy_costs.reasoning_tokens` when not None; 0 and counted otherwise) -- if the proxy can split reasoning
tokens from the raw completion, please set it; and no grading CPU seconds (`private_grading_cpu_seconds` is NaN).

## R3. `tests/unit/dynorch/__init__.py` and `conftest.py` (WP-A)

WP-C's tests live in `tests/unit/dynorch/test_{tables,forecast,report_pipeline}.py` and use only stdlib + pytest
fixtures; they run with or without the package `__init__.py` (basenames are unique in the tree). No conftest
fixture is required.

## R4. `dynorch-tables` verb (WP-D, `dynorch/cli.py`)

Call `tables.build_tables(run_root, cfg, stage=<stage>)`; it writes `<run_root>/tables/*.parquet` and returns the
frames plus `reasoning_unknown_rows`, `n_planned`, `n_records`. For Gate 4 completeness use
`policy_outcomes_dynorch.delivered_state` (`pending`/`unknown` are the incomplete states) and the merged Frame
`docs/dynorch_report/core.py::Frame` for "sealed ladder point estimates reproduced" (`ladder_check` in
`numbers.json`). Gate 2's re-fit is `forecast.refit(<tables/dynorch_runs.parquet>, cfg,
planned_runs=forecast.planned_runs_core(cfg))`.

## R5. Open question for the lead: E0/E2 ceilings in plan section 8

`forecast.plan_table` reproduces the core rows of the section 8 table under the re-injection ceiling (per-call
input = 5.5k + the run's output): core H1+H3 $57.97 (plan $58), core H0 $14.17 ($14), pilot $6.73 ($7), core total
$79.3 ($80). The E0 and E2 rows ($12 for 384 runs at rungs {4, 16}) are reproduced only by the flat-input model
($12.56); under the same ceiling model as the core they are $19.9 (+25% = $24.9). Admission of E0/E2 should use
the higher figure or the plan's number should be re-derived.

## R6. Vocabulary note

The sealed ladder's `delivered_state` values are `correct | wrong | op_invalid_output | unknown_evaluator_error`;
the contract's harness vocabulary is `delivered | op_no_output | unknown | pending`. WP-C follows the contract.
Report code never keys on `delivered_state` across the two families (it uses `y` and, for the harness side, the
contract vocabulary), so the mismatch is cosmetic in the merged frame; say if the lead wants the sealed words.
