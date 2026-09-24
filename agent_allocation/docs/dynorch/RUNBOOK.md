# RUNBOOK — `openrouter_dynorch_v1` (WP-D; plan §9 gates, contract §8)

Worktree `/orcd/data/tpoggio/001/mabdel03/agents_scaling_dynorch_v1/agent_allocation` (branch `dynorch-v1`), run root
`RUN=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts`, config `CFG=configs/dynorch_v1.json`,
scratch `/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/` (every harness state dir, temp dir, npm cache and the Codex
prefix). The sealed base run root `.../forced_budget_v2/artifacts` is read-only. Key file
`~/.config/agents_scaling/openrouter_allocation_v1.key` (0600; never printed, never copied). Every shell starts with

```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_dynorch_v1/agent_allocation && source slurm/env.sh
export TMPDIR=/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/tmp npm_config_cache=/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/npm-cache
export PATH=/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/npm-prefix/bin:$PATH
A=.venv/bin/allocation; CFG=configs/dynorch_v1.json; RUN=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts
```

Exit codes of every `dynorch-*` verb: **0 advance / clean**, **3 stop_notify** (an owner must look before anything
continues), **1 stop / refused** (an invariant broke or the verb refused), **2 blocked** (a WP-A/B/C entry point has
not landed: see `docs/dynorch/requests/wpD.md`). Nothing auto-raises a ceiling, switches a provider or edits a prompt.

## Owner actions before Gate 3 (main)

1. Raise the OpenRouter key limit so `limit_remaining ≥ 95` (dashboard). The ledger ceiling is
   `min(--max-usd, settled + limit_remaining − $1)`; below that the main run stops cleanly at the key limit.
2. Free `/home` inodes below 990k: `rm -rf ~/.npm/_cacache ~/.cache/uv ~/.cache/pip` (the doctor refuses above 990k;
   `tools/install_codex.sh` warns while `~/.npm` still holds files).
3. Auto-reload statement: the account auto-refills, so the balance is not a ceiling. It is recorded in
   `state/run_decisions.jsonl` by the `--account-auto-reload` flag of every `dynorch-run` (kind `dynorch_run_start`,
   `account_auto_reload: true`); freeze re-states it (step 5).

## 0. Gate 0 — offline, $0

```bash
tools/install_codex.sh                       # idempotent; verifies codex-cli 0.156.1 + the bundled bwrap
$A dynorch-doctor --config $CFG --run-dir $RUN            # exit 0 required
uv run --frozen --no-sync pytest tests/unit/dynorch -q     # green required
$A dynorch-plan --config $CFG --stage canary               # dry run: 10 cells; pilot 144; main 1728
# the real-CLI smoke ($0: both CLIs through the real proxy over the fake upstream; writes gates/gate0_smoke.json)
DYNORCH_GATE0_SMOKE_OUT=$RUN/gates/gate0_smoke.json uv run --frozen --no-sync pytest \
    tests/integration/test_dynorch_real_cli_smoke.py -m slow -q
uv run --frozen --no-sync pytest tests/integration/test_dynorch_e2e_fake_harness.py -m slow -q   # 330 s / 660 s stalls (17 min)
$A dynorch-gates --config $CFG --run-dir $RUN --gate 0     # doctor + suite + gates/gate0_smoke.json
```
Or as a job: `slurm/launch_dynorch_v1.sh gate0`. Gate 0 advances only with `gates/gate0_smoke.json`
(`passed: true`, written by the smoke test above; the scripted-spawn / 402 / long-delay items of the Gate 0 list are
covered by the fake-harness e2e, see contract §10.18). **Codex fallback rule** (plan §9): a Codex smoke failure no config change fixes → H3 := OpenClaw
(`config.dynorch.extension_arms.H2`), owner notified, its own smoke list must pass.

## 1. Gate 1 — canary (≤ $2 allowance, plan expects ≈ $0.7)

```bash
slurm/launch_dynorch_v1.sh canary            # VERB=canary: dynorch-run --stage canary --allow-paid --max-usd 2
squeue --start -j <jobid>                    # printed by the launcher; resubmit on the other partition if hours away
```
What `dynorch-run --stage canary` does, in order: validates the config; refuses without `--allow-paid`; refuses
`--max-usd` above the canary allowance ($2) or the preset ceiling ($95) unless `--approval "<owner text>"` is given
(recorded via `live.record_decision`); fetches `GET /models/qwen/qwen3.8-27b/endpoints` **once** and applies plan
§4 (`locks/openrouter_endpoints.json`, `locks/provider_map.json`, `locks/provider_decision.json`; a `stop` decision
exits 3 and writes nothing else); snapshots `/key` + `/credits` (`state/reconcile_canary_before.json`); binds the
proxy on `127.0.0.1:0` (`state/proxy.json`); runs the 10 canary cells; builds the tables; reconciles
(`state/reconciliation_canary.json`); writes `state/status.json`.

Then evaluate:
```bash
$A dynorch-tables --config $CFG --run-dir $RUN [--stage canary]   # rebuild the tables (every pre-registered stage by default)
$A dynorch-gates --config $CFG --run-dir $RUN --gate 1     # writes gates/gate1_canary.json + locks/harness_prompts.json
```
Gate 1 criteria (all in `gates/gate1_canary.json`): endpoint rule locked; every call `completion ≤ granted` and
`granted == C − spent`; usage present; `/key` delta within $0.02 of the ledger; the 2 % `/generation` sample matches
(tokens once, cost within a cent or 5 %, provider pinned); no upstream 402; `grace_truncated_share ≤ 0.02`; a forced
close with non-empty text per harness; a Codex Responses tool loop; the k=16 ceiling probe recorded (**0/4 spawns →
exit 3: stop and ask the owner, option A**); `S_H1`, `S_H3` written; no key-shaped string in the run tree; nothing
new under `$HOME` (the walk takes minutes; `--no-home-scan` fails that criterion on purpose).

## 2. Gate 2 — pilot (≤ $7 allowance, plan expects ≈ $4)

```bash
slurm/launch_dynorch_v1.sh pilot             # VERB=pilot: --stage pilot --allow-paid --max-usd 7 --resume
$A dynorch-gates --config $CFG --run-dir $RUN --gate 2     # writes gates/gate2_pilot.json + reports/PILOT_REPORT.md
```
Per harness: `valid_final_share ≥ 0.90` (0.80–0.90 → exit 3, below → exit 1); tool-call parse failures ≤ 5 %;
`cap_overrun_runs == 0`, `grace_overrun_runs == 0` (exit 1); `spawn_share(k=16) ≥ 0.10` (else 3); guard ≤ 5 %;
evaluator errors ≤ 1 %; reinjection hit rate ≥ 0.95; timeouts ≤ 2 %; reconciliation as Gate 1; re-fitted core
forecast (+25 %) ≤ soft ceiling ($90) − spent after at most cuts C1–C3 (C4 is the owner's: drop a harness or the
seeded 48-item subpanel). `PILOT_REPORT.md` pools accuracies per benchmark only.

**Owner go/no-go is mandatory** and is what unlocks `main`:
```bash
$A dynorch-gates --config $CFG --run-dir $RUN --gate 2 --owner-go "owner 2026-..: go, cuts none"      # or --owner-no-go "..."
```

## 3. Freeze and Gate 3 — main ($95 ceiling, $90 soft, expected $50–60)

Freeze = the endpoint lock (already write-once), the prompt/config hashes (WP-A locks), `proxy_version` and harness
versions stamped on every record, and the owner's auto-reload statement:
```bash
$A dynorch-run ... --account-auto-reload     # every main/extension invocation carries the flag; recorded at start
slurm/launch_dynorch_v1.sh run               # VERB=run: --stage main --allow-paid --max-usd 95 --resume $RUN_FLAGS
RUN_FLAGS="--account-auto-reload --concurrency 12" slurm/launch_dynorch_v1.sh run
```
`main` refuses to start unless `gates/gate2_pilot.json` records `advance`, the owner go is in
`state/run_decisions.jsonl`, and the endpoint lock exists. The job is 12 h on `mit_normal` / 2 d on `pi_lhtsai`; the
runner traps the drain signal (`--signal=B:TERM@900`), stops admitting, waits ≤ 540 s for in-flight runs, terminates the survivors, finishes its drain by 660 s and exits 0 (contract §10.33) — **resubmit the
same command with `--resume`** (already in `VERB=run`) until `state/status.json` says `finished`.

Monitor every hour:
```bash
squeue -u $USER --start; tail -5 $(dirname $RUN)/logs/dyn-run-main_<jobid>.out
python - <<'EOF'
import json; s=json.load(open("$RUN/state/status.json")); print(s["state"], s.get("runner"), s.get("active_ceiling_usd"))
EOF
$A dynorch-reconcile --config $CFG --run-dir $RUN --stage main --phase after   # spend vs /key any time (free endpoints)
```
**Stop rules** (plan §9 Gate 3; `touch $RUN/control/STOP` drains cleanly, `scancel` is the hard stop). **Remove
the sentinel before the next job** (`rm $RUN/control/STOP $RUN/control/DRAIN`): a stage now REFUSES to start while
one exists (it would otherwise drain itself at the first 5 s poll and execute nothing). Rules: parse-failure
share over the trailing 50 runs > 0.15; timeout share > 0.05; harness crash share > 0.10; provider mismatch on any
call; the endpoint failing §4 for > 30 min (`GET /models/.../endpoints` by hand — the lock is never re-evaluated);
a ledger invariant violation; unresolved exposure > $2; any upstream 402 (`status.json → reconciliation.decision ==
stop_notify`). After a stop: fix, record the decision (`live.record_decision`), resume. A post-freeze proxy fix that
affects scoring or the cap forces re-execution of every affected cell (never a silent patch).

## 4. Gate 4 — finalize, seal, report

```bash
slurm/launch_dynorch_v1.sh finalize          # reconcile --phase after → dynorch-tables → dynorch-gates --gate 4
$A dynorch-gates --config $CFG --run-dir $RUN --gate 4 --stage main   # by hand: gates/gate4_main.json, tables/SEAL.json
bash docs/dynorch_report/build.sh            # WP-C; copy the report to the main checkout docs/dynorch_v1/report/
```
Gate 4: a finalised row per planned cell; completeness ≥ 0.95 per (arm, rung) (partial cells excluded from the
primary p-values; a partial H1/16 or H3/16 → exit 3); `/key` delta within $0.50 or 2 %; grader re-run identical;
sealed ladder point estimates reproduced (deep/16 = 0.7917, sample/16 = 0.4167); tables sealed.

## 5. Extensions (Gate 5) — only after Gate 4 advances, in this order, each `remaining ≥ 1.25 × forecast`

```bash
STAGE=ext_e0 slurm/launch_dynorch_v1.sh run --stage ext_e0   # E0 capped H1c/H3c @ {4,16}, ≈ $8 (ceiling $12)
STAGE=ext_e1 slurm/launch_dynorch_v1.sh run --stage ext_e1   # E1 second repeat, rung order 16, 4, 25, 8, 2, 1
STAGE=ext_e2 slurm/launch_dynorch_v1.sh run --stage ext_e2   # E2 spawn-disabled H1n/H3n @ {4,16}
STAGE=ext_e3 slurm/launch_dynorch_v1.sh run --stage ext_e3   # E3 OpenClaw @ {4,16} (unless already substituted)
slurm/launch_dynorch_v1.sh finalize --stage ext_e0            # each extension reconciled and sealed on its own
```
Restrict a stage with `RUN_FLAGS="--arms H1 --rungs 16"` (a restriction never adds cells; the pre-registered
manifest is write-once, so a filtered relaunch never shrinks the planned cell set). **`RUN_FLAGS` must not contain
a comma** — `sbatch --export` splits on it, so `--arms H1,H3` would silently run one arm; the launcher refuses it.
Each extension also needs the previous one sealed (`gate4_ext_e0` before `ext_e1`, ...) and
`remaining ≥ 1.25 × forecast`; `dynorch-run` refuses otherwise and names the shortfall.

Every VERB passes `--resume`. Re-running a stage without it is refused as soon as any cell has an unfinished or
rerun-eligible record (it would overwrite paid records in place and conflict in the write-once call journal).
Submitting the sbatch directly (without the launcher) needs the log dir first:
`mkdir -p $(dirname $RUN)/logs && sbatch --export=ALL,VERB=run,... slurm/dynorch_v1.sbatch`.

## Where things are

| what | path |
|---|---|
| verbs | `allocation dynorch-plan / dynorch-run / dynorch-tables / dynorch-gates / dynorch-reconcile / dynorch-doctor` |
| gate results | `$RUN/gates/gate{0,1,2,4}_<stage>.json` (`contracts.GateResult`) |
| locks | `$RUN/locks/{openrouter_endpoints,provider_map,provider_decision,harness_prompts}.json` |
| account/ledger reconciliation | `$RUN/state/reconcile_<stage>_before.json`, `$RUN/state/reconciliation_<stage>.json` |
| proxy / status / decisions | `$RUN/state/{proxy.json,status.json,run_decisions.jsonl}` |
| pilot report | `$RUN/reports/PILOT_REPORT.md` |
| Slurm | `slurm/dynorch_v1.sbatch` (VERB=gate0\|canary\|pilot\|run\|finalize), `slurm/launch_dynorch_v1.sh`, logs `$(dirname $RUN)/logs/` |
| Codex | `tools/install_codex.sh` → `/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/npm-prefix/bin/codex` |
