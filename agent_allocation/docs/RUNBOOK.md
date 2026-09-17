# RUNBOOK — `openrouter_agent_allocation_v1`

From `/orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation` on branch `allocation-v1`.
Always start with `source slurm/env.sh`. Study run dir: `artifacts/openrouter_agent_allocation_v1` (symlink to
`/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1`).
Key: `~/.config/agents_scaling/openrouter_allocation_v1.key` (0600). Logs: `.../openrouter_agent_allocation_v1/logs/`.

## FINAL — the study is complete (2026-09-17 14:52 EDT)

Collection ran 11:35:58 → 13:01:49; finalize (seal → analyze → audit) finished 14:52:44. Nothing is pending.

- **67,514 generations terminal, 101,088 checkpoint outcomes known, 0 unfinished partitions.** All four selection
  batches sealed and verified before any gold join.
- **Spend $47.64** (main $28.64, high $10.26, Flash replication $7.71, pilot $0.98, canary $0.03) against the $135
  ledger ceiling; ledger vs `/key` difference −$0.098 on a $39.83 window, inside tolerance; 0 invariant violations.
- **`audit`: 29 pass / 0 fail / 1 skip** (the skip is the optional agent-06 duplicate metrics table, never produced).
  0 cache hits, 0 provider drift, 0 rate-limit responses, 0 duplicate dispatches across the run.
- Reports: `artifacts/openrouter_agent_allocation_v1/reports/{RUN_REPORT,AUDIT,COST_RECONCILIATION,COMPLETENESS}.md`
  (+ `.html`); the 14:23 first edition is kept at `reports_1423_backup/` for diffing.
- **Read `docs/FINDINGS.md` before quoting any contrast.** The MMLU-Pro primary zeros are genuine but fragile
  (degenerate bootstrap, one item from non-zero), and `cp/0 - peer/4/1` is a tautology (cp/0 *is* peer/4/1).
- Results headline: PEER n=4 r=4 is **worse** than PRIVATE_TEAM by 2.5 pts macro (95% CI −4.6 to −0.8,
  Holm p=0.027), driven by MBPP+ (−7.0) and MATH (−3.0); the CP contrast is null (−0.5 pts, p=0.57).

### Deviations during finalization (both recorded in `state/run_decisions.jsonl`)
1. Pending finalize job 22899228 was moved `mit_normal` → `pi_lhtsai` (Slurm had it starting the next morning).
   The sbatch files now list both partitions; still check `squeue --start -j <id>` after submitting.
2. The final cached grading pass was terminated after 32 min (`grade rc=143`). It re-reads ~71k cached grade files
   over NFS and writes only for candidates lacking a grade; the follower grader had already reported
   `candidates: 0`, no parsed record was newer than 13:02:38, and `audit` confirmed every checkpoint has a selector
   row and every candidate a grade. Seal/analyze/audit then ran normally.

### Re-running analysis (free: reads sealed artifacts, no API calls)
```bash
.venv/bin/allocation analyze --run-dir artifacts/openrouter_agent_allocation_v1 --config configs/balanced150.json
.venv/bin/allocation audit   --run-dir artifacts/openrouter_agent_allocation_v1 --config configs/balanced150.json
```
`analyze` takes ~25 min (10,000 bootstrap replicates over 101k rows).

## History — where things stood (2026-09-17 05:00 EDT)

- Gate 1 (canary) **passed** 03:39; $0.0321442 = key usage delta.
- Gate 2 (pilot) **stopped for an owner decision**: after the single pre-authorized amendment (text headroom
  1,024 -> 2,048), truncation-without-final is MATH 7.4% and MMLU-Pro 5.6% (limit 3%). Every other criterion passes
  (BBH 2.8%, MBPP+ 0%, no cache hits, no drift, no duplicate dispatches, budget stop never over-reserved, zero evaluator
  errors, forecast $65.70). Truncated answers are genuine visible derivations (not loops); at A=4,096 (5,120-token
  cap) none truncated in the pilot. Cumulative spend $0.51237945.
- Account balance ~$81.7 (other projects share the account). The core forecast fits it; the optional Flash block
  (~$11) needs a top-up.
- Decisions logged in `state/run_decisions.jsonl`; amendments in `amendments.jsonl`; journal repair (duplicate
  charge_max from the kill test) documented there with the original kept read-only.

## Owner decision -> commands

### A. Raise text headroom to 4,096 (recommended)
```bash
.venv/bin/allocation pilot --config configs/balanced150.json --allow-paid --max-usd 5 \
    --phase owner-headroom --family text --headroom 4096 --approval "<owner message quote + timestamp>"
.venv/bin/allocation gates --stage pilot            # exit 0 = advance; 3 = stop again (ask the owner)
```

### B. Accept the remaining truncation as a documented limitation
```bash
.venv/bin/allocation pilot --config configs/balanced150.json --allow-paid --max-usd 5 \
    --phase owner-accept --approval "<owner message quote + timestamp>"
```

## After Gate 2 advances: freeze and launch (batch jobs survive the interactive session)
```bash
.venv/bin/allocation freeze --config configs/balanced150.json
DJ=$(sbatch --parsable slurm/dispatch.sbatch run --config configs/balanced150.json --allow-paid --max-usd 135 --resume)
GJ=$(sbatch --parsable slurm/grade.sbatch --run-dir artifacts/openrouter_agent_allocation_v1 --resume --follow --workers 8 --config configs/balanced150.json)
FJ=$(sbatch --parsable --dependency=afterany:$DJ --export=GRADER_JOB=$GJ slurm/finalize.sbatch)
echo "dispatch=$DJ grade=$GJ finalize=$FJ"
```
Monitor: `.venv/bin/allocation status --run-dir artifacts/openrouter_agent_allocation_v1`,
`squeue -u $USER`, `tail -f logs/dispatch_$DJ.out`. Graceful stop: `.venv/bin/allocation stop --run-dir ... --drain`.
A STOP/DRAIN sentinel persists: clear with `allocation stop --run-dir ... --clear` before any resume.

## Stop rules (spec §16) — pause, never improvise
Provider/tier drift, cache hits on independent draws, missing budget support, widespread extraction failure, faulty
grading, ledger discrepancy (journal refuses to replay), leaked gold, or source-version change: stop, preserve raw data,
tell the owner. Never raise ceilings, switch providers/models, edit prompts, or buy credits.

## State at 11:35 EDT (session ending 11:41)
Prompt amendment `prompt_system_v2` (owner-approved) fixed MMLU-Pro (97.2% valid). Text headroom is 4,096 (owner-approved).
Gate 2 re-pilot stops only on MATH: truncation 9.3% on two items (`intermediate_algebra/2022` x8, `/362` x2), valid 90.7%.
Everything else passes; forecast $69.75; spend $1.02. **Owner decision pending: accept the MATH truncation?**
If accepted, one command does acceptance + freeze + launch as a batch job:
```bash
sbatch --export=APPROVAL="<owner quote + timestamp>" slurm/accept_freeze_launch.sbatch
```
