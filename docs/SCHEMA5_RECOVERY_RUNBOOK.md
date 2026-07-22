# Durable schema-5 recovery and production runbook

This is the authoritative operator procedure for the clean schema-5 rerun. It is ordered
to preserve the legacy evidence before mutation and to keep production fail-closed until
every readiness gate is backed by a checksummed artifact. Do not substitute the retired
`.dispatcher-v3`, chunk drivers, editable environments, or old run IDs.

## Fixed identities and invariants

```bash
repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
recovery="$results/recovery/schema5-v1"
release="$recovery/releases/sweep-recovery-schema5-v1"
worktree="$release/worktree"
identity="$release/identity"
harness="$release/environments/harness"
serving="$release/environments/serving"
state="$results/.dispatcher-schema5-v1"
pool="$results/server_pools/schema5-v1"
hf_home=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
dev_python=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python
conda_exe=/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda
```

The three authoritative run IDs and no others are:

| run ID | cells | expected QIDs |
| --- | ---: | ---: |
| `full_sweep_schema5_v1` | 4,680 | 933,660 |
| `full_sweep_agent_counts_schema5_v1` | 14,400 | 2,872,800 |
| `full_sweep_agent_count_7_schema5_v1` | 3,600 | 718,200 |
| **total** | **22,680** | **4,524,660** |

At all times before `resume`, require all of the following:

- `desired_state=paused` or no schema-5 control state yet;
- no legacy cell or dispatcher jobs and no held cell locks;
- no mutation of a legacy run before the pre-repair snapshot marker and external
  attestation both verify;
- no production execution from the development checkout;
- no manual editing of a pin, readiness envelope, registry record, manifest, result, or
  controller sbatch;
- exact-ID cancellation only. Never cancel by a broad job-name pattern.

## 1. Establish maintenance and seal the pre-repair snapshot

Capture `squeue` and `sacct`, cancel only the resolved legacy job IDs, and publish the
maintenance interlock. The consolidation tool independently checks scheduler quiescence
and every advisory lock, so an operator assertion alone is insufficient.

Create the independent snapshot in a durable CPU allocation with `--no-requeue`:

```bash
cd "$repo"
"$dev_python" scripts/create_recovery_snapshot.py \
  --snapshot-root "$recovery/pre_repair" \
  --source "full_sweep_v1=$results/full_sweep_v1" \
  --source "full_sweep_agent_counts_v1=$results/full_sweep_agent_counts_v1" \
  --source "full_sweep_agent_count_7_v1=$results/full_sweep_agent_count_7_v1" \
  --source "dispatcher_v3=$results/.dispatcher-v3" \
  --source "recovery_evidence=$recovery/pre_repair_inventory"
```

The command is resumable and removes only its own exact incomplete temporary files. It
still rejects symlinks, shared source/destination inodes, hardlinked temporaries,
lookalikes, unexplained destination entries, and source drift. Success exists only when
`SNAPSHOT_COMPLETE.json` was published last. Then perform a separate verification pass
and write its envelope outside the read-only snapshot:

```bash
"$dev_python" scripts/create_recovery_snapshot.py \
  --verify-only \
  --snapshot-root "$recovery/pre_repair" \
  --attestation-path "$recovery/pre_repair.attestation.json"
```

Archive both Slurm job records and logs. Do not proceed to legacy cleanup if verification
re-reads a single byte differently.

## 2. Test, tag, materialize, and seal the release

From the development checkout, require a clean full suite and clean whitespace check:

```bash
cd "$repo"
"$dev_python" -m pytest -q
git diff --check
```

Review the intended source/configuration/documentation/test set, commit it, and create the
annotated tag `sweep-recovery-schema5-v1`. The tag must resolve to `HEAD`, and the checkout
must have no tracked or untracked release inputs. Do not move or recreate the tag after an
environment has been materialized.

Run materialization first without `--apply`, then repeat the exact command with `--apply`
inside a durable, non-requeued CPU job:

```bash
"$dev_python" scripts/materialize_schema5_release.py materialize \
  --output-root "$release" \
  --source-repository "$repo" \
  --release-worktree "$worktree" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --harness-prefix "$harness" \
  --serving-prefix "$serving" \
  --conda-executable "$conda_exe"
```

Verify `MATERIALIZATION_COMPLETE.json`, including the isolated installed-package probe,
zero shared source/destination inodes, and exact tagged worktree. Next run the freezer as
a dry run and then with
`--apply --seal-worktree --seal-environments --seal-output-root`:

```bash
"$harness/bin/python" -I "$worktree/scripts/freeze_schema5_release.py" create \
  --output-root "$identity" \
  --release-worktree "$worktree" \
  --harness-prefix "$harness" \
  --serving-prefix "$serving" \
  --model-contract "$worktree/configs/model_contracts.v1.json" \
  --fleet-contract "$worktree/configs/schema5_fleet.v1.json" \
  --conda-executable "$conda_exe"
```

Finally verify the marker-last bundle:

```bash
"$harness/bin/python" -I "$worktree/scripts/freeze_schema5_release.py" verify \
  --output-root "$identity"
```

Only `RELEASE_COMPLETE.json` makes this a production release.

## 3. Consolidate legacy evidence in the fixed order

Both invocations below must use the frozen interpreter and frozen script. First inspect
the dry-run plan; it names every would-change cell and checkpoint with before/after hashes.
Then repeat with `--apply`:

```bash
"$harness/bin/python" -I "$worktree/scripts/consolidate_legacy_recovery.py" \
  --results-root "$results" \
  --recovery-root "$recovery" \
  --pre-repair-attestation "$recovery/pre_repair.attestation.json"
```

The transaction order is fixed: archive/reset 22 discarded-response incidents, migrate
47 remaining schema-1 checkpoints, archive three obsolete permanent ledgers, run the
zero-rewrite canonical audit, and perform the complete semantic audit. Acceptance is:

- 740 complete legacy cells;
- 888,068 active validated QIDs;
- 1,064 newly sealed incident QIDs, with the 355 earlier incident QIDs retained as sealed
  history;
- 47 migrated checkpoints preserving sampled coordinates;
- zero corrupt, unresolved permanent, or canonical-repair mutations.

The success marker is `$recovery/LEGACY_CLEANUP_COMPLETE.json`.

Create and externally attest the independent consolidated snapshot:

```bash
"$harness/bin/python" -I "$worktree/scripts/create_recovery_snapshot.py" \
  --snapshot-root "$recovery/legacy_consolidated" \
  --source "full_sweep_v1=$results/full_sweep_v1" \
  --source "full_sweep_agent_counts_v1=$results/full_sweep_agent_counts_v1" \
  --source "full_sweep_agent_count_7_v1=$results/full_sweep_agent_count_7_v1" \
  --source "dispatcher_v3=$results/.dispatcher-v3" \
  --source "legacy_cleanup_evidence=$recovery/operations/legacy_consolidation" \
  --source "legacy_cleanup_complete=$recovery/LEGACY_CLEANUP_COMPLETE.json"

"$harness/bin/python" -I "$worktree/scripts/create_recovery_snapshot.py" \
  --verify-only \
  --snapshot-root "$recovery/legacy_consolidated" \
  --attestation-path "$recovery/legacy_consolidated.attestation.json"
```

Retirement is dry-run first and then `--apply`. It verifies that exact six-source
snapshot before removing write permission:

```bash
"$harness/bin/python" -I "$worktree/scripts/retire_legacy_runs.py" \
  --results-root "$results" --recovery-root "$recovery"
```

Build the supplementary cache only in `supplementary-legacy` mode. It must be labelled
mixed-protocol; schema-less rows cannot support exact-token claims. Historical notebooks,
tables, and reports remain supplementary even after this cache refresh.

## 4. Clone the authoritative runs and initialize paused control

Use `release_identity.schema5-v1.json/control_pin_fragment` as the sole source of the
release ID, Git/source identity, contract paths, and environment-manifest hashes. Invoke
`clone_schema5_manifests.py` first as a dry run and then with `--apply`. Do not type or
copy hashes from terminal output. The resulting roots must be empty of result rows and
must contain byte-identical copies of the legacy scientific contracts plus immutable
lineage and artifact-policy sidecars.

Initialize the three estimand-excluded smoke suites by the same dry-run/`--apply` pattern
with `init_schema5_smokes.py`. They contain exactly 15 + 20 + 6 cells.

After all six roots verify, derive rather than hand-author the control pins:

```bash
"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" prepare-pins \
  --release-bundle-root "$identity" \
  --hf-home "$hf_home" \
  --output "$recovery/immutable_pins.schema5-v1.json"

"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" init \
  --pins-json "$recovery/immutable_pins.schema5-v1.json"

"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" reconcile --all --no-admit
```

`prepare-pins` validates all 22,680 cells and 4,524,660 expected QIDs, creates only the
canonical empty pool directory, and publishes read-only JSON plus SHA-256. `init` must
leave `desired_state=paused`. Reconciliation must show no unmappable job, ambiguous
intent, validation error, or missing profile.

## 5. Bootstrap the fleet and build all seven readiness gates

Load the exact `fleet_supervisor_command` argv array from the immutable pins and execute
it once with an added `--once`; do not reconstruct the flags in shell. This submits the
22 logical replicas/24 GPUs while control remains paused. The expected allocation is:

- two standard replicas each for 0.6B, 1.7B, 4B, and 14B;
- three standard 8B and four standard 32B replicas;
- one long replica for each 0.6B through 14B;
- two TP=2 `32B-long` replicas.

Every registered endpoint must match its Slurm-spooled script, release/model/tokenizer/
environment/fleet identities, replica ID, port, served context, and live HTTP probe.

Generate gate envelopes with `build_schema5_readiness.py` and attach them with
`schema5_control.py attest`. The required gates are:

1. `snapshot`: both externally attested snapshots;
2. `migrations`: exact cleanup/migration evidence;
3. `semantic_audit`: the accepted 740/888,068/1,064 legacy accounting;
4. `fleet`: all 22 logical replicas and 24 GPUs live and provenance-valid;
5. `context_audit`: 43,092 selected dense-peer requests and all 57,456 selected
   seven-agent plus-CoT/unlimited requests fit, with zero failed preflights and positive
   margin;
6. `smoke_runs`: all 41 schema-5 smoke cells complete at concurrency one with exact
   provenance and zero unresolved context, protocol, or truncation incident;
7. `email_test`: a real test delivered to `mabdel03@mit.edu`.

Context audits must run from the frozen release with offline model/tokenizer revisions.
The dense audit selects seven-agent decentralized, prompt-level 3, plus-CoT cells at
`b2048`, `b8192`, and `unlimited`; the second selects every seven-agent plus-CoT,
unlimited cell. Pass `--all-routed-profiles` in both cases. Feed their complete JSON
artifacts into the typed context-evidence builder rather than transcribing totals.

Run `run_schema5_smokes.py --apply` with the immutable pins SHA-256 and current rollout
generation. It is resumable, uses the production worker boundary, and contacts one cell
at a time. Its final schema-2 report is the `smoke_runs` evidence.

Before production, perform one exact-ID dispatcher-controller kill drill and one exact-ID
fleet-controller kill drill. Each successor must recover within 15 minutes with unchanged
fairness state and no duplicate mutation. Re-run scheduler reconciliation afterward.

## 6. Transactional resume and staged ramp

`resume` refuses to change desired state unless all readiness envelopes still verify and
the scheduler join is unambiguous:

```bash
"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" resume

"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" status --live
```

The initial ceiling is 24. Advance only with the explicit `set-ceiling` command and the
following observed clean windows:

| ceiling | promotion requirement |
| ---: | --- |
| 24 | all three production runs make validated-QID progress for one clean hour |
| 96 | six additional clean hours |
| 192 | twelve additional clean hours |
| 384 | every preceding gate and live-health check remains green |

The dispatcher enforces the 448 submitted-job QOS ceiling, reserves 64 jobs, uses arrays
of at most 24, and requests 1 CPU/4 GB for each HTTP client. A cell receives `USR1` twenty
minutes before the effective 12-hour limit, stops starting new coordinates, journals the
current request, and exits without inventing a failure.

For a controlled stop:

```bash
"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" pause --drain
```

If both controller jobs disappear, leave desired state unchanged and run the idempotent
`repair-chain`; never launch another state directory. Account-wide cancellation cannot
self-restart on this cluster, so the alert email and this command are the recovery path.

## 7. Continuous acceptance and finalization

The live monitor checks controllers, successors, heartbeats, admissions, endpoints,
scheduler holds, disk, and new failures every five minutes; semantic/QID progress every
six hours; and full manifest validation daily. Missing controllers, stale heartbeats,
starvation, corrupt/permanent states, untrusted responses, QOS holds, and gate failures
must create a durable alert and email.

The first fully healthy production poll starts the throughput epoch. Controller
successors do not reset it; pauses and material fleet changes close it and append a new
epoch. After 48 continuous hours, every scientific stratum must have measured throughput
and the projection must be no more than 28 days. The aggregate requirement is at least
161,595 validated QIDs/day. Add saturated-profile capacity—one TP=2 `32B-long` replica at
a time where indicated—and begin a new epoch after a material change. Never trim the
scientific design implicitly.

Final success requires all 22,680 cells and exactly 4,524,660 expected QID outcomes, with
zero missing, partial, retryable, permanent, corrupt, malformed, duplicate, unexpected,
or stale-ingested records. Every censor appears exactly once and is reported explicitly.
Publish a checksummed final snapshot and a primary analysis cache restricted to the three
schema-5 manifests.
