# GATE 0 REPORT — `openrouter_dynorch_v1` (offline, $0; WP-E, 2026-09-23)

Plan §9 Gate 0; contract §8, §10.18. Worktree `agents_scaling_dynorch_v1` branch `dynorch-v1`; run root
`RUN=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts`. No paid call, no network beyond
loopback, no key read; nothing written under `/home` by any run (rule in §3.4).

## 1. Verdict

| item | result | evidence |
|---|---|---|
| `dynorch-doctor` preconditions | **exit 0** (`{"blocking_failures": [], "warnings": []}`) with the runbook shell header (`npm_config_cache` exported; unset it is a blocking `state_paths_outside_home`, requests/wpE.md item 17) | §2 |
| `tests/unit/dynorch` | **370 passed** (23 s) | §2 |
| real-CLI smoke, OpenClaude 0.13.0 | **all cases pass** through the REAL proxy stack (see §3.1) | `gates/gate0_smoke.json` |
| real-CLI smoke, Codex 0.156.1 | **all cases pass**; no Codex fallback needed (see §3.2) | `gates/gate0_smoke.json` |
| `dynorch-gates --gate 0` | **advance, exit 0** (`gates/gate0_offline.json`) | §5 |

Two proxy-side fixes came out of the smoke (§4); each is committed with its pinning test. The plan's Codex fallback
rule (H3 := OpenClaw) is **not** triggered: every Codex disagreement was fixed in the wire layer or is a verified,
harmless behaviour recorded in `docs/dynorch/requests/wpE.md` items 14-18 (proposed contract §10.19-10.21).

## 2. Doctor and unit suite

```
$ export npm_config_cache=/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/npm-cache TMPDIR=/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/tmp
$ .venv/bin/allocation dynorch-doctor --config configs/dynorch_v1.json --run-dir $RUN     -> exit 0
   home_file_count < 990k; key file 0600; no sk-or- in the worktree; state paths outside /home;
   codex-cli 0.156.1 in the scratch prefix; openclaude 0.13.0 on PATH; template sha c3cf9e34...; base run
   root inputs present; jinja2 3.1.6; prompt files present
$ uv run --frozen --no-sync pytest tests/unit/dynorch -q                                  -> 370 passed
$ uv run --frozen --no-sync pytest tests/integration/test_dynorch_e2e_fake_harness.py -m "not slow" -q -> 6 passed
```

## 3. Real-CLI smoke ($0) — what ran and what it proved

Stand: `tests/integration/dynorch_smoke_support.py`. `runner.build_stack` (the real `ProxyCore`, `CollectionLedger`,
`CallJournal`, `RunStore`, `Upstream`) + the real `httpd.ProxyServer` on `127.0.0.1:0` + the real
`OpenclaudeAdapter` / `CodexAdapter` launched by the real `Runner` (scrubbed env, work dir under `tmp_path`); the
only substitution is `api/fake.py`'s transport as the upstream, scripted per conversation class from the coordinates
the core tags on every `/completions` call. Every case records its checks into `$RUN/gates/gate0_smoke.json`
(`{passed, failures, cases}`), the file `dynorch-gates --gate 0` reads. Commands:

```
export DYNORCH_GATE0_SMOKE_OUT=$RUN/gates/gate0_smoke.json
uv run --frozen --no-sync pytest tests/integration/test_dynorch_openclaude_smoke.py tests/integration/test_dynorch_codex_smoke.py -m "not slow" -q   # ~4 min
uv run --frozen --no-sync pytest tests/integration/test_dynorch_real_cli_smoke.py -m slow -q                                                        # canary rehearsal, ~10 s
uv run --frozen --no-sync pytest tests/integration/test_dynorch_openclaude_smoke.py -m slow -q   # 330 s + 660 s stalls
uv run --frozen --no-sync pytest tests/integration/test_dynorch_codex_smoke.py -m slow -q        # 330 s + 660 s stalls
```
(`-m 'not live'` is the only default deselection in `pyproject.toml`; `slow` must be deselected / selected explicitly.)

### 3.1 OpenClaude 0.13.0 (H1) — plan §9 list, all VERIFIED through the real proxy

| Gate 0 item | case | observed |
|---|---|---|
| first request's tool list exactly `{Agent}` | `delegation_rung4` | `tools == ["Agent"]`, wire `chat` |
| `messages[0]` = harness prefix + system text (contract §10.2) | `delegation_rung4` | byte-equal to `OPENCLAUDE_PREFIX + system_dynorch.txt.rstrip("\n")` |
| no `budget_usd` text anywhere | every case | no hit in journaled requests, `launch.json`, stdout, stderr |
| a scripted `Agent` call yields a `solver` conversation with no tools | `delegation_rung4` | 3 spawns -> 3 solver conversations, each `tools == []`, system text = prefix + `subagent_dynorch.txt` + OpenClaude's `Notes:` suffix; `non_solver_spawns == []` |
| 3-turn tool loop re-injects cached reasoning, hit rate ≥ 0.95 | `delegation_rung4` | ids `call_1_0`, `call_3_0`, `call_5_0` echoed exactly on the assistant turns and tool results; `reinjection_hits = 6`, misses 0, rate 1.0 |
| `other` calls == 0 | every case | `calls_by_class["other"] == 0`; classes `root, solver, root, solver, root, solver, root` |
| 402 ends the run within 10 s with ≤ 1 further request | `forced_close_then_402` | rung 1: call 1 thinks past the cap -> `mid_call` grace whose continuation spawns; the solver's call gets HTTP 402, the root's next request is the one further 402; process exit **0.07 s** after the first 402, exit 1, `is_error: true`, `result: "API Error: 402 ..."`, no rerun, watchdog not involved, zero upstream calls after the grace |
| fake delays produce exactly one upstream call and `subtype == success` | `delayed_35s` (two keepalive comments), `delayed_330s`, `delayed_660s` | one upstream call, `subtype: success`, `is_error: false`, `duplicates_served == 0` (stall results in §3.3) |
| rung-25 run ends with `subtype == success` | `rung25` | cap 25,600 granted, `subtype: success`, graded, wall timeout 3,600 s |
| nothing written under `/home` | every case | §3.4 |
| canary cells through `dynorch-run --rehearsal` | `canary_rehearsal/H1` (pre-existing smoke) | 4 runs done, reconciliation clean |

Also pinned per case: every grant is `C − spent` (B2), `finish_reason ∈ {stop, tool_calls}`, the delivered answer is
the proxy's (`final_source == proxy_root`) and cross-checks the CLI result, no key material in the run tree or
scratch, `harness_version` stamped (`0.13.0`).

### 3.2 Codex CLI 0.156.1 (H3) — plan §9 list, all VERIFIED through the real proxy

| Gate 0 item | case | observed |
|---|---|---|
| install + `codex --version` | `install` | `codex-cli 0.156.1` from `<scratch>/npm-prefix/bin/codex` |
| bundled bwrap runs on this kernel | — | not exercised: with `shell_tool = false` no sandbox is ever launched (probe §2.4); moot for H3 as configured |
| `wire_api = "chat"` refused | `install` | `--strict-config` refuses the config before any request (exit ≠ 0, stderr names `wire_api` / `chat`, no `turn.started`) |
| first request's tools exactly `update_plan` + `multi_agent_v1`, none of `exec_command`/`shell`/`apply_patch`/`web_search`/`view_image` | `delegation_rung4` | `["update_plan", "spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent"]`, namespace `multi_agent_v1` on the five, none of the forbidden names (nor `request_user_input` / goal tools) |
| `instructions == system_text` (§10.2) | `delegation_rung4` | `messages[0]` = the system text followed by Codex's `<skills_instructions>` developer block (joined by the wire layer, contract §10.7) |
| scripted `spawn_agent` -> child request with `x-openai-subagent` at the same base URL | `delegation_rung4` | the child arrives under the same run key, `subagent: true`, classified `solver`, tools = `update_plan` only (no spawn tool), user turn = the spawn message; `wait_agent` on the minted agent id returns the child's text; events `collab_tool_call spawn_agent`, `wait` completed; `reinjection == "explicit"` (Codex re-sends its reasoning items) |
| scripted `exec_command` answered as unknown tool, no process | `exec_command` | proxy emits the call with `unknown_function:exec_command` noted; Codex answers `function_call_output: "unsupported call: exec_command"`; no `command_execution` item; the leak file the command would have created is absent |
| missing `response.completed` -> retry then `turn.failed` | `missing_completed` | a fault-injecting `ProxyServer` subclass cuts the stream after `response.created`: `Reconnecting... 1/2`, `2/2`, then `turn.failed`, exit 1, no `-o` file; the two retries are identical and served from the proxy's cache (`duplicates_served == 2`, **one** upstream call, B9) |
| the 660 s delay | `delayed_660s` (+ `delayed_35s`, `delayed_330s`) | one upstream call, `turn.completed`, no `error` event (§3.3) |
| `timeout` kills the whole tree | `timeout` | `timeout --kill-after=5s 12s codex ...` with a 40 s upstream delay: exit 124 at ~12 s, the whole process group (timeout -> node -> native codex) gone afterwards, the abandoned upstream call finished and settled (ledger `in_flight == unresolved == 0`) |
| 402 after the grace | `forced_close_then_402` | Codex reports `unexpected status 402 Payment Required` and re-sends the identical request twice (`stream_max_retries = 2`) per conversation, each refused in-process (no upstream call), then `turn.failed` 0.7 s after the first 402; exit 1, no rerun (requests/wpE.md item 15) |
| rung-25 run | `rung25` | `turn.completed`, graded, argv wall `3600s` |
| nothing under `/home` | every case | §3.4 |
| canary cells through `dynorch-run --rehearsal` | `canary_rehearsal/H3` | 4 runs done, reconciliation clean |

### 3.3 The stalls (`-m slow`)

| harness | delay | upstream calls | harness outcome | wall |
|---|---|---|---|---|
| OpenClaude | 35 s (2 keepalive comments) | 1 | `subtype: success`, `is_error: false` | 37.1 s |
| OpenClaude | 330 s (22 keepalives) | 1 | `subtype: success` | 331.5 s |
| OpenClaude | 660 s (44 keepalives) | 1 | `subtype: success` | 661.5 s |
| Codex | 35 s | 1 | `turn.completed`, no `error` event | 35.5 s |
| Codex | 330 s | 1 | `turn.completed`, no reconnect | 330.5 s |
| Codex | 660 s | 1 | `turn.completed`, no reconnect | 660.5 s |

Every stall: one journaled call, `duplicates_served == 0`, the record `done` / `parse_status valid`, the CLI result
cross-checking the proxy's delivered answer. The proxy's headers went out within the 2 s refusal grace and the
`: dynorch keepalive` comment every 15 s kept both clients on their single request (B9 holds with
`API_TIMEOUT_MS = 3600000` / `stream_idle_timeout_ms = 1800000`).

### 3.4 The `/home` rule and what it saw

`$HOME` is snapshotted before each run (depth ≤ 3, the session's own `.claude`, `.cursor-server`, `.vscode-server`,
`.cache/uv` pruned) and walked again after it. A NEW file fails unless a live foreign harness instance accounts for
it; a pre-existing file touched under a harness state dir (`.codex`, `.claude`, `.openclaude*`, `.npm`, ...) fails
unless attributed; other churn is reported. Attribution reads `/proc/<pid>/environ` of every `codex` / `claude` /
`openclaude` process of this user that is not a descendant of the test (pid + command recorded in the report). On
this node during the smoke: cursor-server's Codex app-server (pid 3543847, `CODEX_HOME=$HOME/.codex`, writes
`logs_2.sqlite*` / `models_cache.json` continuously) and two Claude Code sessions (`~/.claude.json`); a root-owned
`orcd/.quota` and `.humming/tmp/lock/launcher.lock` are periodic writers unrelated to any harness. **No run created
a file under `/home`; every harness state file stayed under the per-run work dir** (`home/cfg/...` for OpenClaude,
`state/codex/...` for Codex), as `harness.base.check_env` requires of the environment.

## 4. Fixes made from the smoke (each committed with its evidence)

1. **`httpd.py` — a refusal decided before the headers carries its real HTTP status** (commit `726e0e4`). The
   streaming path sent `200 text/event-stream` before calling the core, so a budget 402 (decided in microseconds,
   no upstream call) reached OpenClaude as an error frame inside a 200 stream, which it retried once per
   conversation (four 402s in 70 ms). `ProxyServer` now waits `refusal_grace_seconds = min(2 s, keepalive)` for an
   outright refusal and answers it with its status + JSON body (both routes); errors after the headers stay
   in-stream. Pinned by `test_httpd.py::test_streaming_refusal_before_headers_carries_the_real_http_status` and the
   OpenClaude `forced_close_then_402` case. Plan §2 B9's "headers at once" is now "headers within the refusal
   grace" (proposed contract §10.19).
2. **Test stand** (`dynorch_smoke_support.py`): the scratch key scan skips symlinks and binaries — Codex leaves
   `state/codex/tmp/arg0/*` symlinks to its own binary on a failed exit and that binary carries a `sk-or-OPENAI...`
   substring (proposed §10.21); `ps -ww` without `COLUMNS` for process attribution (pytest exports `COLUMNS`, `ps`
   truncates); `flock` around the report merge.

No adapter or core change was needed: the plan §6 flags, the §10.3 Codex config keys, the §10.10 classification
and the §10.4/10.5 result parsing all held against the real CLIs through the real stack.

## 5. `dynorch-gates --gate 0`

```
$ .venv/bin/allocation dynorch-gates --config configs/dynorch_v1.json --run-dir $RUN --gate 0
{"decision": "advance", "failing": [], "gate": "gate0", "stage": "offline", "written": ".../gates/gate0_offline.json"}
   doctor_preconditions  passed  (no blocking failure)
   unit_suite_green      passed  (pytest tests/unit/dynorch: 370 passed)
   real_cli_smoke        passed  ({"failures": [], "passed": true}; 18 cases: 2 canary-rehearsal, 7 OpenClaude, 9 Codex)
   evaluated_at 2026-09-24T02:23:04Z   exit 0
```
`gates/gate0_smoke.json` (18 cases, every check true) and `gates/gate0_offline.json` are in the run root.
**Gate 0 passes; nothing blocks it.** Gate 1 (the paid canary, ≤ $2) is the next step and is an owner action
(`slurm/launch_dynorch_v1.sh canary`); nothing paid was run here.

## 6. Not done / caveats

- The bundled bwrap is not exercised (nothing to sandbox with `shell_tool = false`); the kernel-4.18 question stays
  open and moot for H3 as configured.
- "No process spawned" for `exec_command` rests on the absent `command_execution` item, the router's
  `unsupported call` answer and the absent leak file (ptrace is denied on this host, as in the probe).
- The stall cases were run once each (16.5 min per harness); they are not in the default `pytest` selection unless
  `-m slow` is given, and `-m 'not slow'` must be passed to skip them (the `addopts` only deselect `live`).
- The Codex 402 retry loop (`unexpected status 402`, 2 free retries, then `turn.failed`) cannot be disabled without
  also disabling the wanted retries on transient failures; it costs nothing and ends within a second.
- The contract (`00_contract.md`) is lead-only: the proposed §10.19-10.21 texts are in `docs/dynorch/requests/wpE.md`.
- The RUNBOOK's Gate 0 block (WP-D) still lists only `test_dynorch_real_cli_smoke.py`; the commands in §3 above are
  the complete set (requests/wpE.md item 19).
