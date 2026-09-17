# RUNBOOK — `openrouter_agent_allocation_v1`

From `/orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation` on branch `allocation-v1`.
Always start with `source slurm/env.sh`. Study run dir: `artifacts/openrouter_agent_allocation_v1` (symlink to
`/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1`).
Key: `~/.config/agents_scaling/openrouter_allocation_v1.key` (0600). Logs: `.../openrouter_agent_allocation_v1/logs/`.

## Where things stand (2026-09-17 05:00 EDT)

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
