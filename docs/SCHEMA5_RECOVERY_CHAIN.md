# Schema-5 v1.1 recovery chain

The production recovery is rendered only after the repository-wide test suite passes,
the tree is clean, and the annotated tag `sweep-recovery-schema5-v1.1` resolves to
`HEAD`. The renderer refuses a lightweight tag, dirty checkout, existing schema-5 run
root, existing control state, prior v1.1 release, or reused job namespace.

Rendering is dry-run by default:

```bash
repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
recovery="$results/recovery/schema5-v1"
dev_python=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python

"$dev_python" "$repo/scripts/render_schema5_recovery_chain.py" render \
  --repository "$repo" \
  --results-root "$results" \
  --recovery-root "$recovery" \
  --hf-home /orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  --dev-python /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --conda-executable /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda
```

Repeat the exact command with `--apply`. Success publishes read-only sbatch files under
`$recovery/jobs/schema5-v1.1/`, a separate generation-specific log directory, and
`$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1.json` last. Verify it independently:

```bash
"$dev_python" "$repo/scripts/render_schema5_recovery_chain.py" verify \
  --chain-manifest "$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1.json"
```

The first job makes a fresh `git clone --no-local --no-checkout`, requires an annotated
tag, detaches at the manifest-pinned commit, rejects object alternates, runs `git fsck`,
and refuses to adopt an existing checkout. The chain never refers to the failed v1
checkout or release directory.

The heavyweight filesystem passes are one strict ancestry chain:

```text
pre-repair snapshot
  -> independent snapshot verification/attestation
  -> release materialization
  -> release freeze/verification
  -> legacy consolidation
  -> consolidated snapshot
  -> independent consolidated verification/attestation
```

`legacy_consolidate` is the first job allowed to mutate legacy data. It cannot run until
both the snapshot and release jobs succeed, and it re-verifies the immutable release
immediately before running the complete consolidation dry-run and apply. Retirement,
new-run initialization, readiness, smoke runs, the controller kill drill, and production
resume remain downstream.

Every recovery-wrapper allocation uses the non-preempting `mit_normal` partition and
declares `#SBATCH --no-requeue`; the self-healing GPU fleet retains its separately
pinned serving partition. Fleet readiness uses an 11-hour
allocation with a bounded 10-hour polling deadline, rather than consuming the complete
effective 12-hour cell limit. The smoke job reads paused control immediately before
launch and passes exactly `rollout_generation + 1`; the smoke runner's control lock and
own generation check close the remaining race.

Submission is also dry-run by default:

```bash
"$dev_python" "$repo/scripts/render_schema5_recovery_chain.py" submit \
  --chain-manifest "$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1.json"
```

Inspect the symbolic plan, then repeat with `--apply`. The submitter requires the live
cluster policy `DependencyParameters=kill_invalid_depend`, uses exact `afterok` job IDs,
passes `--no-requeue` again at the command line, and persists a durable intent before
each `sbatch`. If it dies across that boundary, the next invocation joins both `squeue`
and `sacct` by a unique generation-scoped comment and adopts exactly one accepted job.
Because this cluster does not retain `JobComment` in accounting, reconciliation also
extracts and cross-checks that exact comment from `sacct`'s immutable `SubmitLine`;
zero or multiple ambiguous matches fail closed. It publishes
`RECOVERY_CHAIN_SCHEMA5_V1_1_SUBMISSION.json` only after all 20 jobs are mapped.

If a terminal job fails, inspect the exact failed/cancelled suffix before resubmitting:

```bash
"$dev_python" "$repo/scripts/render_schema5_recovery_chain.py" repair \
  --chain-manifest "$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1.json"
```

The command refuses active, missing, ambiguous, or unclassified scheduler state. Repeat
with `--apply` to create the next contiguous repair generation. Completed ancestors keep
their exact job IDs; only terminal failed/cancelled jobs receive generation-scoped
comments and new `afterok` submissions. Each generation has its own durable intent
journal and immutable receipt under `$recovery/recovery_chain_repairs/gNNNN/`.

If `release_materialize` failed after creating a partial release tree, repair deliberately
stops. Preserve the complete tree before retrying, first as a dry run and then with
`--apply`:

```bash
"$dev_python" "$repo/scripts/render_schema5_recovery_chain.py" \
  quarantine-materialization \
  --chain-manifest "$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1.json"
```

This operation requires the exact materialization job and all chain jobs to be terminal,
rejects either completion marker, records the source device and inode in an immutable
intent, and performs one same-filesystem atomic rename into
`$recovery/releases/quarantine/`. It never deletes or rewrites the partial tree. A crash
after the rename is recoverable by rerunning the same command; an immutable completion
record is stored under `$recovery/materialization_quarantines/`. Run `repair --apply`
only after that record is complete.

Do not submit until the full suite, tag, fresh-render verification, scheduler quiescence,
and operator review are complete. The final `production_resume` job receives the full
11.5-hour wrapper because the first rollout performs a byte-level verification of both
large recovery snapshots before publishing its generation seal. It still relies on the
controller's fail-closed readiness validation; it cannot set running state unless all
seven gates and the marker-last two-role kill drill verify.
