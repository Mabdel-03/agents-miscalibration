# passk_cont_v1 — runbook (WP-G: generation + ops)

Everything below runs from the merged checkout `CODE=/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation`
with its own venv (`.venv/bin/...`). Never the consortium env; nothing under /home.

```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation
source slurm/env.sh
A=.venv/bin/allocation; PY=.venv/bin/python; CFG=configs/passk_cont_v1.json
STUDY=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1; RUN=$STUDY/artifacts
```

## 0. Preconditions (lead, Gate S)

`$RUN/manifests/` holds `splits.json` (contract §5.1), `panels.json` (dev = D32, main = L96 in legacy order,
reserve = T200 in frozen rank order), `waves.json` = `[]`, `public/*.jsonl` for every L96/D32/T200 item (L96/D32
records byte-identical to forced_v2's), `private/*.jsonl` (gold), `locks/datasets.json`; and `$STUDY/gates/gate_s.json`
with `{"passed": true, "splits_sha256": <sha256 of splits.json>}`.

## 1. Shards (writes `$RUN/shards/{assignment.json,mancer,parasail}`; $0)

```bash
$A shard-plan --config $CFG --run-dir $RUN            # print: counts per panel (reserve = T200 1:1)
$A shard-plan --config $CFG --run-dir $RUN --write
for s in mancer parasail; do                          # node ids carry the endpoint lock hash: adopt the forced_v2
  $A passk-run --config $CFG --run-dir $RUN --shard $s --partitions smoke --adopt-legacy-lock   # lock (write-once)
done
```
or `sbatch --export=ALL,VERB=shard-plan slurm/passk_cont.sbatch`. The reserve (T200) assignment comes from
`splits.json:assignment_t200` when present, else the seeded hash with `passk.t200_shard_shares` (1:1); dev/main keep
seed 20260917 and 3:2, which reproduces the legacy assignment for L96 ∪ D32 (tested).

## 2. Plan check ($0, writes nothing)

```bash
for s in mancer parasail; do
  $A passk-run --config $CFG --run-dir $RUN --shard $s --plan-only \
     --partitions smoke+c48_extra+e1_fill+t200_diag+v48_bank+t200_bank
done
```
Prints items/nodes per partition, the waves, each partition's jobs sha256 (and whether it matches `gates/gate0.json`)
and the unit-cost forecast. Expected totals: smoke 12, e1_fill 1,056, c48_extra 816, v48_bank 1,488, t200_bank 6,200,
t200_diag 1,000 (both shards together); forecast ≈ Mancer $56.4 / Parasail $40.8 (+ probes).

## 3. Gate 0 ($0; the conductor runs it first)

```bash
$PY -m agent_allocation.passk.gates gate0 --config $CFG --study-root $STUDY --check-key --run-tests --run-e0
```
Writes `$STUDY/gates/gate0.json` (and `gates/e0_gate.json`): Gate S + splits sha; planned counts; node ids unique and
none a legacy id; prompt_version / decoding hash == forced_v2; THINK-body parity on every L96/D32 item (new body from
`ForcedClient.validate_request` == legacy raw `request_body.think` bar `max_tokens`, same `provider.only`); the E0
subset; slices 87/82 + 63/60; forecast ≤ $115 and ≤ each soft slice; `/key`: 150 ≤ limit_remaining − 3; nothing under
/home; the passk tests + mocked e2e green. It records `jobs_sha256[shard][partition]`: every later `passk-run` refuses
to dispatch a list whose sha differs. Exit 0 = passed and complete. `--overwrite` exists for an owner decision only.

## 4. The conductor (everything after Gate 0)

```bash
sbatch slurm/passk_conductor.sbatch                                   # Gate 0 -> ... -> report
sbatch --export=ALL,START_AT=main slurm/passk_conductor.sbatch        # resume (gate0|gate1|main|analysis|gate4)
cat $STUDY/STATUS.md $STUDY/conductor.state; squeue -u $USER | grep passk
```
Chain: Gate 0 (retried ×3, 10 min apart, only when it crashes) → both canaries together (a first-shard exit 3 —
union key reconciliation pending — is accepted; a failed forced endpoint screen is retried ×3 when transient and stops
at once when semantic: banana / `</think>` / exactness) → `canary --evaluate-only` on both → `passk-run --partitions
smoke` on both → `grade` → `checkpoints gate1-smoke --fetch-metadata` (exact 100%, permaslug `…-20260814`, provider ==
pin, zero cache hits) → main `passk-run` per shard + `grade --follow` per shard; every `POLL` s Gate 3 (stop rules;
drains) and Gate 2 (≥150 nodes or $4, then every 500) per shard; when stage 1 (c48_extra, e1_fill, t200_diag) is
parsed and graded on both shards: `passk-analyze --stage extract, pairs, e1-lock, e2-fit, e5-fit, seal` (WP-A) →
Gate 4 per shard → `release, score, report` → `control/STOP` ends the grade-follow jobs. Any stop drains both shards
during generation, writes STATUS.md, exits non-zero (Slurm mails FAIL).

## 5. Single steps by hand

```bash
sbatch --export=ALL,VERB=canary,SHARD=mancer slurm/passk_cont.sbatch
sbatch --export=ALL,VERB=canary-eval,SHARD=mancer slurm/passk_cont.sbatch
sbatch --export=ALL,VERB=smoke,SHARD=parasail slurm/passk_cont.sbatch
sbatch --export=ALL,VERB=run,SHARD=mancer,PARTITIONS=c48_extra+e1_fill slurm/passk_cont.sbatch   # '+' never ','
sbatch --export=ALL,VERB=grade-follow,SHARD=mancer slurm/passk_cont.sbatch
sbatch --export=ALL,VERB=checkpoint,SHARD=mancer,STAGE=gate2 slurm/passk_cont.sbatch
sbatch --export=ALL,VERB=analyze,STAGE=extract slurm/passk_cont.sbatch
$PY -m agent_allocation.passk.checkpoints gate3 --config $CFG --study-root $STUDY --shard mancer   # inline, $0
$PY -m agent_allocation.passk.e0_gate --legacy-root <forced_v2>/artifacts --study-root $STUDY      # ~40 s
```

Direct CLI (what the sbatch runs): `$A passk-run --config $CFG --run-dir $RUN --shard <s> --partitions <a+b>
--allow-paid --max-usd <shard slice> --account-auto-reload --resume`. The ledger ceiling is
`min(active_ceiling(max_usd, settled, /key snapshot, account_auto_reload), shard credit slice)`; the dispatcher holds
new attempts under the shard's soft slice itself. It never grades; `allocation grade --run-dir <shard dir> --resume
[--follow] --config <shard dir>/state/shard_config.json` grades, skipping held-out ids on every pass.

## 6. Control

- Drain a shard (finish in-flight, admit nothing new): `touch $RUN/shards/<s>/control/DRAIN` (the sbatch's SIGTERM
  trap does the same; Slurm sends it 15 min before the time limit). Remove it before resubmitting.
- Hard stop: `touch $RUN/shards/<s>/control/STOP` (also ends `grade --follow`).
- Status: `$A status --run-dir $RUN/shards/<s>`; decisions: `$RUN/shards/<s>/state/run_decisions.jsonl`
  (`passk_run_start` records the ceiling, slice, limit_remaining and the jobs shas).
- Held-out fence: `v48_bank`, `t200_bank`, `t200_diag` are never graded until `$STUDY/sealed/SEAL_predictions.json`
  verifies (every listed artifact's sha256 matches; an optional `"splits": ["v48"|"t200"]` releases a subset). The
  running `grade --follow` jobs pick them up within one poll (default 60 s) of a valid seal.
- Never: raise ceilings, switch providers, edit prompts or templates, add GenerationSpec fields, replace failed
  generations, expand T200.
