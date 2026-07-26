# Retired schema-5 v1.1-r1 recovery runbook

> **Immutable historical record—do not execute these commands.** The r1 chain failed
> deterministically during environment materialization and is sealed with
> `requires_superseding_release`. Never submit its proposed `g0001` repair. The
> authoritative procedure is
> [SCHEMA5_V12_RECOVERY_RUNBOOK.md](SCHEMA5_V12_RECOVERY_RUNBOOK.md).

This was the operator procedure for the superseded r1 attempt. It is ordered
to preserve the legacy evidence before mutation and to keep production fail-closed until
every readiness gate is backed by a checksummed artifact. Do not substitute the retired
`.dispatcher-v3`, chunk drivers, editable environments, or old run IDs.

Production release/artifact ID `sweep-recovery-schema5-v1.1` supersedes the failed-closed
v1 release attempt. Operational retry tag `sweep-recovery-schema5-v1.1-r1` supersedes
the safely cancelled v1.1 recovery wrapper; the production release ID, run IDs, and
scientific configuration remain unchanged. The v1 environment clone reached package
verification but never published `MATERIALIZATION_COMPLETE.json` or
`RELEASE_COMPLETE.json`; it is forensic evidence, not a production input. The cancelled
v1.1 chain paths are also immutable operational evidence: do not reuse, rename, or
remove its source checkout, jobs, logs, manifest, journal, receipt, locks, or repair
directory. The v1.1-r1 wrapper uses fresh identities while retaining the v1.1 artifact
ID. Never adopt or resume the v1 output directory.

## Fixed identities and invariants

```bash
repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
recovery="$results/recovery/schema5-v1"
release="$recovery/releases/sweep-recovery-schema5-v1.1"
source_checkout="$recovery/release_source_checkout_v1_1_r1"
worktree="$release/worktree"
identity="$release/identity"
harness="$release/environments/harness"
serving="$release/environments/serving"
state="$results/.dispatcher-schema5-v1"
pool="$results/server_pools/schema5-v1"
hf_home=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
dev_python=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python
conda_exe=/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda
jobs="$recovery/jobs/schema5-v1.1-r1"
logs="$recovery/logs/schema5-v1.1-r1"
chain_manifest="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
submission_journal="$recovery/.RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submission.json"
submission_receipt="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_1_R1_SUBMISSION.json"
render_lock="$recovery/.RECOVERY_CHAIN_SCHEMA5_V1_1_R1.render.lock"
submission_lock="$recovery/.RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submit.lock"
repair_root="$recovery/recovery_chain_repairs_v1_1_r1"
```

The Slurm comment namespace is `asys:s5-recovery-v1.1-r1:<chain-id>:gNNNN:<job>`
and generated job names begin `asys-s5v11r1-`. None of these paths or scheduler
identities may alias the cancelled v1.1 chain.

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

## Transactional recovery-DAG execution

The preferred execution path for Sections 1–6 is the immutable 20-job recovery DAG.
Render it only after the annotated v1.1-r1 tag points at a clean `HEAD`; the commands in
the numbered sections below describe the gates implemented by those jobs and remain the
manual audit reference. First inspect the render without publishing anything, then
repeat the exact invocation with `--apply`:

```bash
renderer="$repo/scripts/render_schema5_recovery_chain.py"

"$dev_python" -I "$renderer" render \
  --repository "$repo" --results-root "$results" --recovery-root "$recovery" \
  --hf-home "$hf_home" --dev-python "$dev_python" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --conda-executable "$conda_exe" --partition mit_normal --slurm-user "$USER"

"$dev_python" -I "$renderer" render \
  --repository "$repo" --results-root "$results" --recovery-root "$recovery" \
  --hf-home "$hf_home" --dev-python "$dev_python" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --conda-executable "$conda_exe" --partition mit_normal --slurm-user "$USER" \
  --apply

"$dev_python" -I "$renderer" verify --chain-manifest "$chain_manifest"
"$dev_python" -I "$renderer" submit --chain-manifest "$chain_manifest"
"$dev_python" -I "$renderer" submit --chain-manifest "$chain_manifest" --apply
```

Publication is marker-last: the read-only job namespace is written before the immutable
chain manifest. Repeating `render --apply` after a crash adopts only byte-exact scripts
and an empty canonical log namespace. Submission writes an intent before each `sbatch`,
reconciles its generation-specific comment through both `squeue` and `sacct`, recovers
the exact value from accounting `SubmitLine` when the cluster leaves `JobComment` blank,
and seals the completed journal into the read-only receipt. A missing scheduler observation after
any attempted `sbatch` remains ambiguous for five minutes; do not delete the journal or
manually resubmit during that interval.

All recovery wrappers use `mit_normal`, `--no-requeue`, exact `afterok` dependencies,
and require cluster `kill_invalid_depend`. If a submitted generation has terminal
failures, inspect and then apply a suffix-only repair:

```bash
"$dev_python" -I "$renderer" repair --chain-manifest "$chain_manifest"
"$dev_python" -I "$renderer" repair --chain-manifest "$chain_manifest" --apply
```

Repair refuses active jobs, creates a new contiguous generation, and reuses only the
completed parent jobs. If `release_materialize` failed after creating a partial release
tree, preserve that exact tree first; this is a recoverable same-filesystem rename with
marker-first intent evidence, never a deletion:

```bash
"$dev_python" -I "$renderer" quarantine-materialization \
  --chain-manifest "$chain_manifest"
"$dev_python" -I "$renderer" quarantine-materialization \
  --chain-manifest "$chain_manifest" --apply
"$dev_python" -I "$renderer" repair --chain-manifest "$chain_manifest" --apply
```

The DAG deliberately orders `supplementary_cache -> fleet_bootstrap -> fleet_readiness
-> smoke_readiness`: CPU/context work finishes before GPUs launch, and no parallel join
can strand a short-lived lease. Fleet bootstrap and every fleet-readiness poll hold the
control lock while creating or validating the paused control's exact next-generation
runtime attestation and integrity lease; polls are at most five minutes apart. Because
Slurm may delay the dependent smoke allocation arbitrarily, that allocation re-adopts,
probes, and re-attests the exact fleet before its first draw. The smoke parent then
renews the same g+1 lease immediately before every worker and once per minute while it
runs; a renewal failure sends `USR1` and fails the smoke gate. Production `resume` must
adopt the same generation and cannot bypass this provenance chain.

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
annotated operational retry tag `sweep-recovery-schema5-v1.1-r1`. The tag must resolve
to `HEAD`, and the checkout must have no tracked or untracked release inputs. Do not
move or recreate the tag after an environment has been materialized. Create the fresh
detached `release_source_checkout_v1_1_r1` checkout from that tag; do not reuse the
cancelled `release_source_checkout_v1_1` path or materialize from the development or
failed v1 checkout. The materialized artifact continues to declare release ID
`sweep-recovery-schema5-v1.1`.

Run materialization first without `--apply`, then repeat the exact command with `--apply`
inside a durable, non-requeued CPU job:

```bash
"$dev_python" scripts/materialize_schema5_release.py materialize \
  --output-root "$release" \
  --source-repository "$source_checkout" \
  --release-worktree "$worktree" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --harness-prefix "$harness" \
  --serving-prefix "$serving" \
  --conda-executable "$conda_exe"
```

The apply invocation is identical except for the final `--apply` flag. The completed
clone must contain no regular-file inode shared with either source prefix and no symlink
whose dependency path ever leaves the clone. External, source-owned, broken, and cyclic
symlinks all fail the release.

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

Fleet admission is transactional under the canonical pool root. The supervisor holds
`.fleet-transactions-v1/fleet.lock`, publishes an immutable
`sbatch/gNNNNNN/<replica>.<intent>.sbatch` and a durable intent before `sbatch`, and
commits a replica only after joining complete `squeue` and `sacct` truth with the exact
scheduler comment, command path, partition, and Slurm-spooled bytes. Generation ledgers
live at `ledgers/gNNNNNN.json`; hash-bound `CURRENT.json` advances atomically. A normal
pause/resume may leave a valid older-generation server running: the next supervisor
adopts it only after the old ledger, scheduler token, and spooled script all validate,
while any eventual replacement is rendered under the new generation. Duplicate tokens
or active jobs fail closed. Do not delete these ledgers or reconstruct a missing intent.

HTTP failure is evidence, not immediate permission to recycle a GPU allocation. For an
exact scheduler-active job and endpoint, both `/health` and `/v1/models` must fail on at
least three supervisor polls spanning at least ten minutes. The supervisor then records
a durable alert and issues only `scancel <exact-job-id>`. Boundary crashes and rejected
cancellations retry that same fenced ID at bounded exponential backoff (at most five
attempts); intermittent success or a replacement job resets the evidence. The five-minute
monitor imports this state as `monitor:fleet-hung`, sends the first deduplicated email,
and excludes the interval from successful throughput polls until scheduler truth shows
the old allocation terminal and its replacement is healthy.

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

Snapshot readiness has a generation-scoped cache, not a permanent trust shortcut.
Attestation and every paused-to-new-generation `resume` fully hash both payload
inventories, reject writable/symlinked/hardlinked controls and payloads, and prove each
`sealed_snapshot_member` logical path, hash, snapshot ID, and root against the addressed
inventory. That pass publishes an immutable compact seal plus a complete lstat metadata
baseline under `$state/snapshot_integrity/`. Both controllers attempt renewal on their
60-second heartbeat under one cross-node lock; only one metadata scan can run, scans are
at most once per 300 seconds, and the lease expires after 420 seconds. Dispatcher polls,
spec rendering, array admission, controller successors, and live production hooks hash
only the compact seal, lease, evidence envelopes, selected members, and five snapshot
controls—not the tens-of-GiB payload. Any chmod, write, inode replacement, hardlink, or
directory-shape change prevents renewal, so new admission fails closed no later than
420 seconds after the last clean scan. A later rollout generation performs a new full
byte verification rather than inheriting the prior generation's seal.

Context audits must run from the frozen release with offline model/tokenizer revisions.
The dense audit selects seven-agent decentralized, prompt-level 3, plus-CoT cells at
`b2048`, `b8192`, and `unlimited`; the second selects every seven-agent plus-CoT,
unlimited cell. Pass `--all-routed-profiles` in both cases. Feed their complete JSON
artifacts into the typed context-evidence builder rather than transcribing totals.

Run `run_schema5_smokes.py --apply` with the immutable pins SHA-256 and the next planned
rollout generation, exactly `paused_control.rollout_generation + 1` (generation 1 for a
fresh control). The runner holds the cross-node control lock, rejects running/resuming/
draining state, and contacts one cell at a time. Its resumable final schema-2 report is
the `smoke_runs` evidence.

Before production, perform the two-controller drill while production remains paused and
admission-disabled. These commands launch isolated drill controllers only; they never
launch dispatcher/fleet children or admit a cell:

```bash
control=("$harness/bin/python" -I "$worktree/slurm/schema5_control.py" --state-dir "$state")

"${control[@]}" reconcile --all --no-admit
"${control[@]}" drill start
until "${control[@]}" drill status --live; do sleep 5; done

"${control[@]}" drill kill --role dispatcher
"${control[@]}" drill wait --role dispatcher --timeout 900
until "${control[@]}" drill status --live; do sleep 5; done

"${control[@]}" drill kill --role fleet_supervisor
"${control[@]}" drill wait --role fleet_supervisor --timeout 900
"${control[@]}" drill finish
"${control[@]}" drill status --live
```

Require each exact-ID successor to recover within 15 minutes, two consecutive clean
scheduler reads after cleanup, unchanged fairness/run state, and exact equality between
the drill journal and final state history. Inspect and archive
`$state/CONTROLLER_KILL_DRILL_COMPLETE.json`; `resume` rejects a missing, drifted, or
stale drill proof.

## 6. Transactional resume and staged ramp

`resume` refuses to change desired state unless all readiness envelopes still verify and
the scheduler join is unambiguous:

```bash
"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" resume

"$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state" status --live
```

The initial ceiling is 24. Promotion is automatic and can occur only when the serialized
monitor commits a read-only, checksummed evidence chain satisfying these windows:

| ceiling | promotion requirement |
| ---: | --- |
| 24 | all three production runs make validated-QID progress for one clean hour |
| 96 | six additional clean hours |
| 192 | twelve additional clean hours |
| 384 | every preceding gate and live-health check remains green |

Each semantic evidence file records exact validated-QID totals for all three production
runs, immutable-control and rollout identities, material fleet generation, current
ceiling, integrity result, and critical findings. Five-minute health evidence makes the
clean interval continuous; a gap longer than 660 seconds or any critical/unclean report
resets the current stage window. A pause or material fleet-generation change also closes
the throughput epoch and returns the ceiling to 24. Every promotion revalidates the
checksummed readiness files and is appended to control history with the complete evidence
list and digest. `set-ceiling` is retained only for an idempotent hold or manual decrease;
it refuses every increase.

The exact all-three-run QID-increase test applies to 24-to-96 only. Later promotions
require non-regressing exact per-run totals plus their six- and twelve-hour clean windows;
a run that has already completed cannot deadlock the ramp merely because it has no QID
left to add. `monitor:qos-memory` and `monitor:starvation` remain warning-class alerts for
operations, but are explicitly ramp-blocking alongside every critical alert: any such
active finding returns an elevated ceiling to 24 and requires a new clean ramp.

The supervised semantic cadence remains six hours. Its immediate post-resume scan creates
the exact three-run baseline, and the five-minute health scans prove continuity, but only
the next semantic scan can prove exact per-run QID gains. Consequently the ordinary
24-to-96 promotion occurs at the first qualifying semantic poll—up to roughly six clean
hours after the baseline rather than exactly at one hour—even though its minimum window
is one hour; cached dispatcher counts are never substituted for semantic QID validation
merely to promote earlier.

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
transactional fleet/hung-allocation state, scheduler holds, disk, and new failures every
five minutes; semantic/QID progress every six hours; and full manifest validation daily.
Missing controllers, stale heartbeats, starvation, corrupt/permanent states, untrusted
responses, QOS holds, hung fleet allocations, and gate failures must create a durable
alert and email.

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
