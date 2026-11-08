# WP-B (wire layer, runner, adapters) — interface requests and notes

Everything WP-B ships codes against `docs/dynorch/00_contract.md` §2/§3/§5/§6 as amended by §10 (lead commit
`76e8858`, merged into `dynorch-wpB`). Items below are additive; nothing in the frozen signatures was changed.

## R-B1 `RunStore` surface the runner uses (WP-A `journal.py`)

`runner.Runner` calls `store.save(record)`, `store.scan() -> {run_key: RunRecord}` and `store.is_terminal(record)`,
and — when present — `store.archive_previous(spec)` before opening a rerun attempt and `journal.supersede(spec)`
(passed as `Runner(..., journal=CallJournal)`) for an interrupted attempt's raw files. WP-A's `journal.py` already
exposes exactly these names; nothing to do beyond keeping them. Pre-merge the e2e uses `tests/unit/dynorch/stubs.py::
StubRunStore` (delete at merge).

## R-B2 `close_run` and `final_message` (WP-A R-A6, confirmed)

The runner keeps a wire-level `ReplyLog` (`httpd.ReplyLog`, the replies and errors actually served per run) and
applies B7 from it. If `close_run` returns a record with `final_message` already filled (R-A6), the runner keeps the
core's value and only cross-checks it against the reply log (`harness_reported.final_message_disagreement = True`
on a mismatch, logged). If the core leaves it `None`, the runner fills `final_message` / `final_source` from the log.
The runner then sets `cli_result`, `cli_result_mismatch`, `harness_exit_code`, `harness_reported`, `wall_seconds`,
`rerun_reason`, `error` (only when it has one), the grade fields (`grade.finalize_run`) and re-saves.

## R-B3 `budget_view` keys (WP-A `core.py`)

The exhaustion watchdog (plan §2 B4: SIGTERM 60 s after the last served reply once exhausted, SIGKILL 30 s later)
reads `budget_view(run_key)["calls"]` and `["remaining"]`; it treats `remaining < terminal_threshold_tokens` (64)
as exhausted and also honours an optional `exhausted: bool` key if the core adds one (recommended: after a grace
the root may still have `remaining >= 64` in principle, and only the core knows a grace happened).

## R-B4 Transport-failure signal for the rerun rule (WP-A R-A6, confirmed)

The rule "last proxy reply was a transport/ledger error" is decided from the last served event of the run in the
reply log (`ProxyError` codes `upstream_failed` / `ledger_refused` / `internal_error`, or any 5xx) and,
belt-and-braces, from `core.record_of(run_key).error.startswith("upstream_failed:")` when the core offers
`record_of`. Refused 402s are never rerun.

## R-B5 Runner construction (WP-D `cli.py`)

`Runner(core, adapters, settings, *, proxy_url, store, inputs, reply_log, cfg, base_env=None, journal=None)` with
`settings = runner.RunnerSettings.from_config(cfg, run_root=..., project_root=..., scratch_root=None, concurrency=None)`
(`DynorchSettings` was not frozen when WP-B started; `RunnerSettings` reads the same config keys and is trivially
buildable from a `DynorchSettings` if WP-D prefers), `adapters = harness.adapters_from_config(cfg)`,
`inputs = plan.BaseInputs.load(cfg["base_run_root"], model=cfg["primary_model"])`, `reply_log = server.reply_log`
where `server = httpd.ProxyServer(core, host=..., port=..., keepalive_seconds=...)` is started first
(`await server.start()`; `server.base_url` is the `proxy_url`). `runner.install_drain_handler(runner)` traps SIGTERM.
`await runner.run_stage(stage, resume=..., arms=..., rungs=...)` returns a `StageSummary`. On `--resume`, run
`journal.resolve_inherited(ledger, store)` (WP-A R-A7) BEFORE `run_stage(resume=True)`.

## R-B6 `harness_version` / `proxy_version` stamping (WP-A `config.py`, WP-D)

`plan_runs` stamps `proxy_version` from its keyword (the runner passes `getattr(core, "proxy_version",
"unrecorded")`; WP-A's core module exposes `PROXY_VERSION` — please also expose it as `ProxyCore.proxy_version` or
tell WP-D to pass it) and `harness_version` from `adapter.version()` (`openclaude --version` / `codex --version`,
cached; `"absent"` when the binary is missing).

## R-B7 Pre-registration file for WP-C tables

`plan.write_plan(specs, run_root, stage)` writes `<run_root>/manifests/dynorch_plan_<stage>.jsonl` (one JSON
`RunSpec` per line, `plan.read_plan` reads it back) BEFORE any cell starts; the runner writes it at the top of
`run_specs`. WP-C's `policy_outcomes_dynorch` should emit one `pending` row per line and finalise from
`RunStore.scan()` (`grade.delivered_state(record)` / `grade.outcome_y(record)` give the sealed vocabulary and y).
Every attempt (including superseded ones) is appended to `<run_root>/state/dynorch_attempts.jsonl` (full
`RunRecord` JSON, key-redacted) for the "one row per run and attempt" table.

## R-B8 `pytest.mark.slow` and the `tests/unit/dynorch/__init__.py` package file (lead)

The 330 s / 660 s delayed-reply e2e cases are marked `@pytest.mark.slow` + `@pytest.mark.timeout(1500)` and skipped
unless `DYNORCH_SLOW=1` (they take 17 minutes together; both pass). Please register `slow` in `pyproject.toml`
`markers` to silence the unknown-mark warning. `tests/unit/dynorch/__init__.py` and `conftest.py` are WP-A files;
WP-B's tests import `tests.unit.dynorch.stubs` as a namespace package until they land (works today).

## R-B9 Allowlist of extra Codex tools (lead / WP-E Gate 0)

`plan_runs` stamps `allowed_tools` from the frozen config arm (§10.3). If Gate 0 shows Codex offering `sleep` /
`current_time` / `get_context_remaining` despite the config, the proxy will refuse with `tool_surface_violation`
(fail closed): widen the config arm's list deliberately, never the adapter.

## R-B10 Contract §10.4 buffering (WP-A `core.py`) — nothing to do

The Responses route emits `response.output_item.done` items and `response.completed` back-to-back only after the
core returned the full reply, so a stream can never carry a partial item; an error after the headers is a single
`response.failed` event (`rate_limit_exceeded` when `ProxyError.retryable`, `context_length_exceeded` for
`input_limit`, else `server_error`). `ledger_refused` (503) is the only retryable code today; if the core wants a
Codex retry to be served from the duplicate cache, raise it with `retryable=True`.
