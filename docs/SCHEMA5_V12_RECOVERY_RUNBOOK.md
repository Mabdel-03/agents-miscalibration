# Schema-5 v1.2-r2 recovery and production runbook

This is the authoritative operator entry point for the clean schema-5 production
rerun. The scientific run IDs and manifests remain unchanged; the operational
release is `sweep-recovery-schema5-v1.2-r2`, the logical release is
`sweep-recovery-schema5-v1.2`, and the chain namespace is `schema5-v1.2-r2`.

The v1.1-r1 chain is sealed forensic evidence. Never submit, repair, or reuse its
`g0001` proposal, partial release, checkout, jobs, logs, intents, or repair namespace.
Use `verify_schema5_recovery_evidence.py` to inspect r1 without executing its code.

## Fixed identities

```bash
repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
recovery="$results/recovery/schema5-v1"
state="$results/.dispatcher-schema5-v1"
pool="$results/server_pools/schema5-v1"
tag=sweep-recovery-schema5-v1.2-r2
release_id=sweep-recovery-schema5-v1.2
slurm_user=mabdel03
chain_manifest="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R2.json"
pilot_checkout="$recovery/materialization_pilot_source_checkout_v1_2_r2"
pilot_root="$recovery/materialization_pilots/schema5-v1.2-r2"
canary_root="$recovery/slurm_canaries/schema5-v1.2-r2"
dev_python="$(realpath -e /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python)"
conda_exe=/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda
sealed_python="$pilot_root/materialization/harness-environment/bin/python"
durable_remote=origin
durable_commit_ref=refs/heads/schema5-v1.2-r2
durable_marker="$recovery/DURABLE_GIT_RELEASE_COMPLETE.json"
```

The three authoritative run IDs contain exactly 4,680, 14,400, and 3,600 cells,
respectively: 22,680 cells and 4,524,660 expected QIDs in total. Production remains
paused until every gate below passes.

## Preconditions

Require all of the following before tagging:

- the r1 quarantine seal, r1 failure envelope, and compact r1 evidence snapshot
  verify;
- the exact-a5cd930 `r1_acceptance/QUARANTINE_IDEMPOTENCY_RECEIPT.json` and
  `ZERO_RESULT_MUTATION_RECEIPT.json` are read-only and verify;
- the original 201,528-file pre-repair snapshot and external attestation verify;
- no legacy/schema-5 worker or controller jobs and no held cell locks;
- a clean full test suite, checksum checks, and Git diff check;
- read-only source-prefix distribution audits pass under the independently
  checksummed ownership and integrity-normalization policies;
- protected non-preemptible scientific server/client capacity for the full
  384-cell ceiling plus reserve has completed its sealed scheduler proof before
  rendering; the independent forced-command watchdog must be provisioned for the
  stage-19 deployment, liveness, and exact-namespace recovery gate before production
  resume; and
- no Conda command has been run against either live developer prefix.

The ownership policy permits exactly one checksummed ownership normalization—the
Setuptools 82/81 collision. A separate integrity-normalization policy permits exactly
five checksummed `RECORD`-only canonicalizations: pip 26.1.1 installer/launcher
metadata; packaging 26.2's Conda-owned `INSTALLER`; wheel 0.47.0's generated launcher;
NumPy 2.3.5's duplicate generated-bytecode row; and Torch C-DLPack's stale
`build_backend.py`/bytecode claims, with FlashInfer retained as owner.
The five canonicalizations cannot authorize a Conda/pip ownership collision and never
rewrite runtime files. Normalization happens only inside copied seeds, archives
complete preimages before atomic changes, and proves pathwise that the full pre/post
inventory delta contains only the stale Setuptools Conda record (when present) and
the applicable `RECORD` paths. It must project to zero shared `RECORD` paths.
[SCHEMA5_RELEASE.md](SCHEMA5_RELEASE.md) records the exact hashes.

## Freeze and test the r2 source

Create an annotated tag only from the reviewed, clean commit:

```bash
git -C "$repo" status --short
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
git -C "$repo" tag -a "$tag" -m "Durable schema-5 v1.2 recovery release"
commit="$(git -C "$repo" rev-list -n 1 "$tag")"
test "$(git -C "$repo" cat-file -t "refs/tags/$tag")" = tag
test "$(git -C "$repo" rev-parse "$tag^{commit}")" = "$commit"
git -C "$repo" fsck --full --strict

# The push is explicit. The evidence publisher itself never changes a remote.
git -C "$repo" push --atomic "$durable_remote" \
  "$commit:$durable_commit_ref" \
  "refs/tags/$tag:refs/tags/$tag"
durable_publisher="$repo/scripts/publish_schema5_durable_git_release.py"
"$dev_python" -I "$durable_publisher" \
  --repository "$repo" --recovery-root "$recovery" \
  --remote "$durable_remote" --remote-commit-ref "$durable_commit_ref"
"$dev_python" -I "$durable_publisher" \
  --repository "$repo" --recovery-root "$recovery" \
  --remote "$durable_remote" --remote-commit-ref "$durable_commit_ref" \
  --apply
test -f "$durable_marker" && test ! -L "$durable_marker"
test ! -w "$durable_marker"
jq -e --arg commit "$commit" \
  '.passed == true and .clean_checkout == true and
   .annotated_tag == true and .remote_query_read_only == true and
   .release_git_commit == $commit and .remote_commit == $commit and
   .remote_peeled_commit == $commit' "$durable_marker"
```

Create the prerequisite pilot checkout in its own namespace. The production DAG owns
the distinct `release_source_checkout_v1_2_r2` path and requires that path to be
absent when rendered.

```bash
git clone --no-local --no-checkout "$repo" "$pilot_checkout"
git -C "$pilot_checkout" checkout --detach "$commit"
test ! -e "$pilot_checkout/.git/objects/info/alternates"
test "$(git -C "$pilot_checkout" rev-parse "refs/tags/$tag")" = \
  "$(git -C "$repo" rev-parse "refs/tags/$tag")"
git -C "$pilot_checkout" fsck --full --strict
test -z "$(git -C "$pilot_checkout" status --porcelain=v1 --untracked-files=all)"
```

Run the genuine isolated schema-4 composite Slurm canary from that tagged checkout.
The default mode combines the admission transaction, a three-job dependency-cascade
experiment, and two executable warm turnovers. Do not use the diagnostic
`--transaction-only`, `--dependency-only`, or `--turnover-only` modes for release
acceptance:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"$dev_python" -I "$pilot_checkout/scripts/run_schema5_slurm_fleet_canary.py" \
  --canary-root "$canary_root" \
  --partition mit_normal --time-limit 00:08:00 --scheduler-user "$slurm_user" \
  --turnover-cycles 2 --turnover-drain-seconds 5 \
  --dependency-alert-latency-seconds 180 \
  --durable-git-release-marker "$durable_marker"

"$dev_python" -I "$pilot_checkout/scripts/run_schema5_slurm_fleet_canary.py" \
  --canary-root "$canary_root" \
  --partition mit_normal --time-limit 00:08:00 --scheduler-user "$slurm_user" \
  --turnover-cycles 2 --turnover-drain-seconds 5 \
  --dependency-alert-latency-seconds 180 \
  --visibility-timeout 900 --terminal-timeout 900 --probe-timeout 10 \
  --durable-git-release-marker "$durable_marker" \
  --apply

"$dev_python" -I "$pilot_checkout/scripts/run_schema5_slurm_fleet_canary.py" \
  --canary-root "$canary_root" \
  --durable-git-release-marker "$durable_marker" --verify
```

The first invocation is dry-run only. The apply command submits one transaction
allocation, three dependency allocations, and three short-walltime turnover
allocations. It must prove
durable pre-`sbatch` intent, complete `squeue` plus `sacct` reconciliation, exact
spooled script bytes, and effective `Requeue=0` for every allocation. Each of the
two turnovers requires two simultaneous predecessor/standby endpoint probes, atomic
promotion, routed-endpoint continuity through the drain, no more than two physical
allocations and zero GPU overlap, exact-ID retirement, and terminal accounting.
The dependency root is submitted with `--hold`; only after all three jobs and their
immutable submission receipt exist is that exact root released. It deliberately
exits 42, its `afterok` child must be cancelled as never started with
`DependencyNeverSatisfied`, and its aggregate `afterany` sentinel must start no more
than 180 seconds after the root terminal timestamp. The canary records and reparses
the raw live `scontrol show config` output and requires
`DependencyParameters=kill_invalid_depend`.

`CANARY_COMPLETE.json` is published last at the canonical root only after all three
sealed component markers verify. Its schema-4 composite evidence hash-binds the
transaction, dependency, and turnover roots, markers, canary IDs, bounded alert
latency, two completed turnover cycles, overlap and continuity measurements, the
exact annotated tag object and commit, and the SHA-256 and size of both the executing
canary and `fleet_transactions.py`. Earlier-schema or component-only evidence is not
a release prerequisite. Production
CLI use derives code identity from the exact clean tagged checkout; only unit tests
may inject it.

## Required two-prefix materialization pilot

Follow [SCHEMA5_RELEASE.md](SCHEMA5_RELEASE.md) to render and submit the durable
11.5-hour, `--no-requeue` pilot job from the exact tagged checkout. The pilot must
publish `PILOT_COMPLETE.json`, join its exact terminal `COMPLETED|0:0` scheduler
record through `accept-scheduler`, publish
`PILOT_SCHEDULER_ACCEPTED.json` last, and independently verify both markers. It
proves:

- unchanged bytes in both live prefixes;
- real buffered copies with zero shared regular-file inodes;
- one exact archived Setuptools ownership normalization plus the five
  `RECORD`-only canonicalizations, with a zero-runtime-path inventory delta;
- one Setuptools 81 distribution and no Setuptools 82 record/files;
- repeatable offline materialization of both prefixes; and
- sealed release verification without mutable sources or an external Conda
  executable.

Do not render the production chain until both pilot markers verify. Interactive
`run --apply` output and a semantic pilot marker without scheduler acceptance are
deliberately insufficient.
The schema-4 verifier report exposes both independent policy hashes, the pilot's exact
harness and serving
`live_source_inventory_sha256` values and the Conda executable's canonical path,
SHA-256, size, absolute shebang interpreter, and repeated full base-runtime inventory
(excluding only caches and child environment roots). These are production admission
inputs, not informational metadata. It also binds the pilot's sealed materialized harness: its canonical
prefix, frozen environment-manifest hash and directory-inventory hash, exact resolved
Python path/hash/size, and library path. The renderer compares the first verification
report with a second report produced by that sealed harness. All subsequent
chain/source/submit verification uses the sealed pilot harness; it does not import
the full verifier graph in the stdlib-only bootstrap.

Use the exact durable marker in the immutable pilot receipt. Run the dry render,
publish the batch files, dry-run and apply the transactional submitter, wait for
queue absence and exact terminal accounting, accept scheduler truth, repeat
acceptance to prove idempotency, and verify in this order:

```bash
pilot_script="$pilot_checkout/scripts/run_schema5_materialization_pilot.py"
pilot_sbatch="$recovery/jobs/schema5-v1.2-r2-materialization-pilot.sbatch"
pilot_receipt="$pilot_sbatch.receipt.json"
pilot_logs="$recovery/logs/materialization-pilot-r2"
ownership_policy="$pilot_checkout/configs/environment_ownership_policy.v1.json"
integrity_policy="$pilot_checkout/configs/environment_integrity_normalization_policy.v1.json"
reconciliation_incident="$recovery/post_snapshot_incidents/2026-07-23_conda_pip_interop_source_metadata_reconciliation.json"
recovered_setuptools="$recovery/releases/quarantine/sweep-recovery-schema5-v1.1.partial-job-18555913/environments/harness/conda-meta/setuptools-82.0.1-pyh332efcf_0.json"

pilot_render=(
  "$dev_python" -I "$pilot_script" render-sbatch
  --sbatch-path "$pilot_sbatch" --log-dir "$pilot_logs"
  --partition mit_normal --python-executable "$dev_python"
  --pilot-root "$pilot_root" --release-checkout "$pilot_checkout"
  --expected-tag "$tag" --expected-commit "$commit"
  --harness-source /orcd/home/002/mabdel03/conda_envs/asys_env
  --serving-source /orcd/home/002/mabdel03/conda_envs/serve_env
  --ownership-policy "$ownership_policy"
  --integrity-normalization-policy "$integrity_policy"
  --reconciliation-incident "$reconciliation_incident"
  --recovered-setuptools-record "$recovered_setuptools"
  --conda-executable "$conda_exe"
  --durable-git-release-marker "$durable_marker"
)
"${pilot_render[@]}"
"${pilot_render[@]}" --apply
pilot_sha="$(jq -er '.sbatch_sha256' "$pilot_receipt")"
test "$(sha256sum -- "$pilot_sbatch" | awk '{print $1}')" = "$pilot_sha"
test "$(awk '{print $1}' "$pilot_sbatch.sha256")" = "$pilot_sha"

pilot_submit_cmd=(
  "$dev_python" -I "$pilot_script" submit-sbatch
  --pilot-root "$pilot_root"
  --sbatch-receipt "$pilot_receipt"
  --scheduler-user mabdel03
  --visibility-timeout 900
)
"${pilot_submit_cmd[@]}"
pilot_submission_json="$("${pilot_submit_cmd[@]}" --apply)"
jq -e '
  .status == "submitted" or
  .status == "adopted" or
  .status == "already_submitted"
' <<<"$pilot_submission_json"
pilot_job_id="$(
  jq -er '.job_id | select(type == "string" and test("^[0-9]+$"))' \
    <<<"$pilot_submission_json"
)"

pilot_squeue="$(squeue -h -j "$pilot_job_id" -o "%i")" || exit 1
while test -n "$pilot_squeue"; do
  sleep 30
  pilot_squeue="$(squeue -h -j "$pilot_job_id" -o "%i")" || exit 1
done
pilot_sacct_deadline=$((SECONDS + 900))
while :; do
  pilot_sacct="$(
    sacct -X -n -P -j "$pilot_job_id" --format=JobIDRaw,State,ExitCode
  )" || exit 1
  if (( SECONDS > pilot_sacct_deadline )); then
    echo "timed out waiting for top-level pilot sacct accounting" >&2
    exit 1
  fi
  pilot_sacct_rows=()
  while IFS= read -r pilot_sacct_row; do
    test -z "$pilot_sacct_row" || pilot_sacct_rows+=("$pilot_sacct_row")
  done <<<"$pilot_sacct"

  case "${#pilot_sacct_rows[@]}" in
    0)
      pilot_sacct_remaining=$((pilot_sacct_deadline - SECONDS))
      if (( pilot_sacct_remaining <= 0 )); then
        echo "timed out waiting for top-level pilot sacct accounting" >&2
        exit 1
      fi
      if (( pilot_sacct_remaining < 10 )); then
        sleep "$pilot_sacct_remaining"
      else
        sleep 10
      fi
      ;;
    1)
      pilot_terminal="${pilot_sacct_rows[0]}"
      if test "$pilot_terminal" != "$pilot_job_id|COMPLETED|0:0"; then
        echo "unexpected top-level pilot terminal accounting: $pilot_terminal" >&2
        exit 1
      fi
      break
      ;;
    *)
      echo "expected exactly one top-level pilot sacct row, got ${#pilot_sacct_rows[@]}" >&2
      exit 1
      ;;
  esac
done

pilot_accept=(
  "$dev_python" -I "$pilot_script" accept-scheduler
  --pilot-root "$pilot_root" --sbatch-receipt "$pilot_receipt"
  --job-id "$pilot_job_id"
)
"${pilot_accept[@]}"
"${pilot_accept[@]}" --apply
"${pilot_accept[@]}" --apply
"$dev_python" -I "$pilot_script" verify --pilot-root "$pilot_root"
```

The submitter dry-run makes no changes. Apply publishes immutable marker-first
intent and attempt records, invokes only the receipt-bound submit argv, and waits up
to 900 seconds for complete `squeue` plus `sacct` visibility before publishing
`PILOT_SUBMISSION_ACCEPTED.json`. Its JSON output is the only source of
`pilot_job_id`; do not parse raw `sbatch` output or submit the script manually.
Replaying apply returns `already_submitted`, while a crash across the scheduler
boundary is recovered by adopting one exact receipt/comment/SubmitLine-matching job
and returning `adopted`. Multiple or conflicting jobs, receipt or identity drift,
successful submit output without complete scheduler truth, and unresolved ambiguity
all fail closed without resubmission. Even after submission is accepted, the
operator must observe exact `squeue` absence, then bounded-poll `sacct` for one
top-level row. An empty result alone is retried; multiple rows, a mismatched
`JobIDRaw`, or a terminal row other than
`"$pilot_job_id|COMPLETED|0:0"` fails immediately, and absence through the 900-second
deadline fails. `accept-scheduler` may run only after the exact success row.

If an accepted pilot instead terminates in one of the explicitly allowed
scheduler/node transient states, run the three-step quarantine sequence in
`SCHEMA5_RELEASE.md` and append its exact marker-last seal before retrying. If
`PILOT_SUBMISSION_ACCEPTED.json` is absent, replay `pilot_submit_cmd --apply` first:
it must adopt that exact terminal job without a second `sbatch` before quarantine.

```bash
pilot_quarantine_seal="$(
  dirname "$pilot_root"
)/quarantine_evidence/partial-job-${pilot_job_id}.sealed.json"
test -r "$pilot_quarantine_seal"
pilot_submit_cmd+=(--prior-quarantine-seal "$pilot_quarantine_seal")
```

Accumulate every earlier seal on later retries. There is no implicit discovery or
state-based suppression: the submitter recursively reverifies each explicit
quarantine and excludes only its exact sealed terminal job ID from scheduler
candidate reconciliation. Any unsealed, mismatched, or newly duplicated job remains
a hard ambiguity.

## Protected capacity before render; external watchdog at stage 19

Do not render the chain until the canonical, marker-last protected-capacity file has
been published by its approved evidence builder:

```bash
protected_capacity="$recovery/PROTECTED_CAPACITY_COMPLETE.json"
protected_canary_root="$recovery/protected_capacity/schema5-v1.2-r2"
protected_builder="$pilot_checkout/scripts/build_schema5_protected_capacity_evidence.py"
protected_publisher="$pilot_checkout/scripts/publish_schema5_protected_capacity.py"
fleet_contract="$pilot_checkout/configs/schema5_fleet.v1.json"
model_contract="$pilot_checkout/configs/model_contracts.v1.json"
fleet_contract_sha256="$(sha256sum -- "$fleet_contract" | awk '{print $1}')"
model_contract_sha256="$(sha256sum -- "$model_contract" | awk '{print $1}')"
release_git_commit="$(git -C "$pilot_checkout" rev-parse "$tag^{commit}")"
release_tag_object="$(git -C "$pilot_checkout" rev-parse "refs/tags/$tag")"
capacity_token="$(openssl rand -hex 16)"

test "$release_git_commit" = "$commit"
test "$(git -C "$pilot_checkout" cat-file -t "refs/tags/$tag")" = tag
test "${#capacity_token}" -eq 32

# Non-mutating plan. Review all 43 arrays and their exact resource shapes.
"$sealed_python" -I "$protected_builder" run \
  --root "$protected_canary_root" \
  --recovery-root "$recovery" \
  --release-git-commit "$release_git_commit" \
  --release-tag-object "$release_tag_object" \
  --partition ou_bcs_normal \
  --qos normal \
  --scheduler-user "$slurm_user" \
  --token "$capacity_token" \
  --fleet-contract "$fleet_contract" \
  --fleet-contract-sha256 "$fleet_contract_sha256" \
  --model-contract "$model_contract" \
  --model-contract-sha256 "$model_contract_sha256" \
  --release-worktree "$pilot_checkout"

# Real transaction. Reuse the exact token and arguments from the reviewed plan.
"$sealed_python" -I "$protected_builder" run \
  --root "$protected_canary_root" \
  --recovery-root "$recovery" \
  --release-git-commit "$release_git_commit" \
  --release-tag-object "$release_tag_object" \
  --partition ou_bcs_normal \
  --qos normal \
  --scheduler-user "$slurm_user" \
  --token "$capacity_token" \
  --fleet-contract "$fleet_contract" \
  --fleet-contract-sha256 "$fleet_contract_sha256" \
  --model-contract "$model_contract" \
  --model-contract-sha256 "$model_contract_sha256" \
  --release-worktree "$pilot_checkout" \
  --apply

"$sealed_python" -I "$protected_publisher" verify \
  --recovery-root "$recovery" \
  --release-git-commit "$release_git_commit" \
  --release-tag-object "$release_tag_object"
test -f "$protected_capacity" && test ! -L "$protected_capacity"
test ! -w "$protected_capacity"
unset capacity_token
```

The `--apply` invocation is the only command above that submits jobs. It resumes its
marker-first transaction after interruption and publishes the completion marker
last; never choose a new token for a retry.

It is a schema-2, regular, non-symlink, read-only JSON object bound to the exact
release ID, annotated tag, commit, tag object, and `schema5-v1.2-r2` namespace. It
uses a semantic canonical-JSON `marker_id`; the chain additionally binds its raw
SHA-256 and size. Never hand-author, copy forward, or rehash this marker.

`PROTECTED_CAPACITY_COMPLETE.json` uses protocol
`schema5-v1.2-r2-protected-capacity-v2` and capacity source
`sealed_protected_canary+partition_inventory+association`. It must prove at least
24 active GPUs plus
four warm-headroom GPUs, a protected 384-cell ceiling plus a 64-job reserve, submit
headroom of at least 448 jobs, at least 384 CPUs and 1,572,864 MiB for clients, and
complete `squeue` plus `sacct` reconciliation. Both scientific server and scientific
client placement must attest `PreemptMode=OFF`. The marker binds the exact scheduler
cluster, user, account, QOS association, association `MaxSubmitJobs`, and client
partition CPU/memory/GPU inventory. Blank inherited QOS resource limits are never
interpreted as infinity: CPU and memory authority comes from the sealed simultaneous
canary, while submit authority comes from the exact association. A transport-censor
protocol does not turn a preemptible scientific allocation into protected capacity.

The 64-job reserve is inclusive, not additive. Its sealed category accounting is
exactly 22 active server elements, three warm-turnover elements, and 39 held
controller/monitor/other placeholders. Together with the 384 client elements, the
real canary therefore submits exactly 448 elements. Before its first `sbatch`, two
complete `squeue`/exact-ID `sacct` occupancy observations 60 seconds apart must be
stable and must prove `existing user elements + 448 <= min(association, QOS limit)`.
An accepted association limit of exactly 448 therefore requires exact user-job
quiescence; a larger limit permits only its proven residual occupancy and does not
enlarge the canary. Any other residual, category total, occupancy drift, or raw
`squeue`/`sacct` element count fails publication.

The watchdog evidence cannot be a render prerequisite because it is bound to the
paused control plane created by stage 10. Stage 19 waits for sealed external
deployment and liveness evidence, runs the isolated exact-namespace cancellation
drill through the forced command, and publishes
`EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json` followed by `WATCHDOG_READY.json`.
The drill binds the deployment, watchdog code, immutable release, control, commit,
and annotated-tag object; it proves recovery within 900 seconds with zero duplicate
jobs, duplicate admission intents, or fairness mutation. `WATCHDOG_READY.json`,
protocol `schema5-v1.2-r2-external-watchdog-v1`, additionally binds a
forced-command-only restriction, an exact 300-second timer, two scheduler
observations at least 60 seconds apart, and an acknowledged liveness email.
Production resume fails closed until the renderer's post-initialization watchdog
verifier accepts both markers against that exact paused control.

After the submitted chain has completed `schema5_initialize` (stage 10), provision
the watchdog. Do not run these commands earlier: the bundle is bound to the exact
paused `control.json`. The VM login and cluster host are deliberate operator inputs;
the shell aborts if either is absent. `WATCHDOG_CLUSTER_KNOWN_HOSTS` must be a
separately verified, single-host OpenSSH known-hosts file.

```bash
: "${WATCHDOG_VM_LOGIN:?set WATCHDOG_VM_LOGIN to user@institutional-vm}"
: "${WATCHDOG_CLUSTER_HOST:?set WATCHDOG_CLUSTER_HOST to the cluster SSH host}"
: "${WATCHDOG_CLUSTER_KNOWN_HOSTS:?set a verified known-hosts file}"
test -f "$WATCHDOG_CLUSTER_KNOWN_HOSTS"

watchdog_release="$recovery/releases/$release_id/worktree"
watchdog_harness="$recovery/releases/$release_id/environments/harness/bin/python"
watchdog_tool="$watchdog_release/scripts/build_schema5_watchdog_deployment.py"
watchdog_bundle="$recovery/readiness/external_watchdog/deployment_bundle"
watchdog_evidence="$recovery/readiness/external_watchdog"
watchdog_public_key="$watchdog_evidence/watchdog_vm_ed25519.pub"
watchdog_vm_stage=/var/tmp/schema5-watchdog-v1.2-r2
watchdog_vm_python=/opt/agents-scaling-watchdog/python
watchdog_vm_release=/opt/agents-scaling-watchdog/release
watchdog_vm_config=/etc/agents-scaling-watchdog/watchdog.json
watchdog_vm_identity=/etc/agents-scaling-watchdog/id_ed25519
watchdog_vm_known_hosts=/etc/agents-scaling-watchdog/known_hosts
watchdog_vm_state=/var/lib/agents-scaling-watchdog
control_sha256="$(jq -er '.immutable_sha256' "$state/control.json")"

test "$(jq -er '.desired_state' "$state/control.json")" = paused
test "$(jq -er '.drain_requested' "$state/control.json")" = false
test -x "$watchdog_harness"
mkdir -p -- "$watchdog_evidence"

# Generate the VM-only private key once. Only its public half leaves the VM.
ssh "$WATCHDOG_VM_LOGIN" \
  "sudo install -d -o root -g root -m 0755 /etc/agents-scaling-watchdog; \
   test -f $watchdog_vm_identity || \
     sudo ssh-keygen -q -t ed25519 -N '' -f $watchdog_vm_identity; \
   sudo chmod 0600 $watchdog_vm_identity; \
   sudo cat $watchdog_vm_identity.pub" > "$watchdog_public_key"
chmod 0444 "$watchdog_public_key"

"$sealed_python" -I "$watchdog_tool" bundle \
  --output-root "$watchdog_bundle" \
  --release-root "$watchdog_release" \
  --harness-python "$watchdog_harness" \
  --control-state-dir "$state" \
  --git-commit "$commit" --tag-object "$release_tag_object" \
  --control-sha256 "$control_sha256" \
  --remote-host "$WATCHDOG_CLUSTER_HOST" --remote-user "$slurm_user" \
  --identity-file "$watchdog_vm_identity" \
  --known-hosts-file "$watchdog_vm_known_hosts" \
  --external-state-root "$watchdog_vm_state" \
  --public-key-file "$watchdog_public_key" \
  --vm-python "$watchdog_vm_python" \
  --vm-release-root "$watchdog_vm_release" \
  --vm-config-path "$watchdog_vm_config" \
  --service-user agents-scaling-watchdog \
  --liveness-email mabdel03@mit.edu

# Install exactly one forced-command line in the cluster account.
install -d -m 0700 -- "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 0600 "$HOME/.ssh/authorized_keys"
(
  flock -x 9
  forced_line="$(cat "$watchdog_bundle/authorized_keys.line")"
  grep -Fqx -- "$forced_line" "$HOME/.ssh/authorized_keys" ||
    printf '%s\n' "$forced_line" >> "$HOME/.ssh/authorized_keys"
  test "$(grep -Fxc -- "$forced_line" "$HOME/.ssh/authorized_keys")" -eq 1
) 9>"$HOME/.ssh/.schema5-watchdog-authorized-keys.lock"
install -m 0444 -- "$watchdog_bundle/authorized_keys.line" \
  "$watchdog_evidence/authorized_keys.installation.snapshot"

# Transfer only the sealed, relocatable bundle and verified host-key input.
ssh "$WATCHDOG_VM_LOGIN" "install -d -m 0700 $watchdog_vm_stage"
rsync -a -- "$watchdog_bundle/" \
  "$WATCHDOG_VM_LOGIN:$watchdog_vm_stage/"
scp -- "$WATCHDOG_CLUSTER_KNOWN_HOSTS" \
  "$WATCHDOG_VM_LOGIN:$watchdog_vm_stage/known_hosts"
scp -- "$watchdog_evidence/authorized_keys.installation.snapshot" \
  "$WATCHDOG_VM_LOGIN:$watchdog_vm_stage/authorized_keys.installation.snapshot"

# On the VM, install a real interpreter copy and the exact three-file runtime.
ssh "$WATCHDOG_VM_LOGIN" "\
  set -euo pipefail; \
  id agents-scaling-watchdog >/dev/null 2>&1 || \
    sudo useradd --system --home $watchdog_vm_state \
      --shell /usr/sbin/nologin agents-scaling-watchdog; \
  sudo install -d -o agents-scaling-watchdog -g agents-scaling-watchdog \
    -m 0700 $watchdog_vm_state; \
  sudo install -d -o root -g root -m 0755 \
    /opt/agents-scaling-watchdog $watchdog_vm_release/scripts \
    $watchdog_vm_release/src/agents_scaling/serving; \
  vm_python_source=\$(readlink -e \$(command -v python3)); \
  sudo install -o root -g root -m 0555 \
    \$vm_python_source $watchdog_vm_python; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/release/scripts/build_schema5_watchdog_deployment.py \
    $watchdog_vm_release/scripts/build_schema5_watchdog_deployment.py; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/release/scripts/schema5_external_watchdog.py \
    $watchdog_vm_release/scripts/schema5_external_watchdog.py; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/release/src/agents_scaling/serving/external_watchdog.py \
    $watchdog_vm_release/src/agents_scaling/serving/external_watchdog.py; \
  sudo install -o agents-scaling-watchdog -g agents-scaling-watchdog -m 0600 \
    $watchdog_vm_stage/known_hosts $watchdog_vm_known_hosts; \
  sudo chown agents-scaling-watchdog:agents-scaling-watchdog \
    $watchdog_vm_identity $watchdog_vm_identity.pub; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/watchdog.json $watchdog_vm_config; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/agents-scaling-schema5-watchdog.service \
    /etc/systemd/system/agents-scaling-schema5-watchdog.service; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/agents-scaling-schema5-watchdog.timer \
    /etc/systemd/system/agents-scaling-schema5-watchdog.timer; \
  sudo systemctl daemon-reload; \
  sudo systemctl start agents-scaling-schema5-watchdog.service; \
  sudo systemctl enable --now agents-scaling-schema5-watchdog.timer; \
  sudo systemctl show agents-scaling-schema5-watchdog.service \
    -p LoadState -p Result -p ExecMainStatus; \
  sudo install -d -o root -g root -m 0755 $watchdog_vm_state/evidence; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_state/WATCHDOG_HEARTBEAT.json \
    $watchdog_vm_state/evidence/heartbeat-1.json; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_stage/authorized_keys.installation.snapshot \
    $watchdog_vm_state/evidence/authorized_keys.installation.snapshot; \
  sudo $watchdog_vm_python -I \
    $watchdog_vm_release/scripts/build_schema5_watchdog_deployment.py \
    deployment-evidence \
    --bundle-manifest $watchdog_vm_stage/BUNDLE.json \
    --installed-release-root $watchdog_vm_release \
    --vm-python $watchdog_vm_python \
    --installed-config $watchdog_vm_config \
    --installed-service \
      /etc/systemd/system/agents-scaling-schema5-watchdog.service \
    --installed-timer \
      /etc/systemd/system/agents-scaling-schema5-watchdog.timer \
    --installed-authorized-keys \
      $watchdog_vm_state/evidence/authorized_keys.installation.snapshot \
    --service-heartbeat $watchdog_vm_state/evidence/heartbeat-1.json \
    --output $watchdog_vm_state/evidence/DEPLOYMENT_EVIDENCE.json"

# Obtain a second independent service heartbeat. The explicit gap is part of
# acceptance; do not copy the same heartbeat twice.
sleep 60
ssh "$WATCHDOG_VM_LOGIN" "\
  sudo systemctl start agents-scaling-schema5-watchdog.service; \
  sudo install -o root -g root -m 0444 \
    $watchdog_vm_state/WATCHDOG_HEARTBEAT.json \
    $watchdog_vm_state/evidence/heartbeat-2.json"

for name in DEPLOYMENT_EVIDENCE.json heartbeat-1.json heartbeat-2.json; do
  scp -- "$WATCHDOG_VM_LOGIN:$watchdog_vm_state/evidence/$name" \
    "$watchdog_evidence/$name"
  chmod 0444 "$watchdog_evidence/$name"
done

# Run this acknowledgement only after personally confirming the liveness email.
"$sealed_python" -I "$watchdog_tool" acknowledge-liveness \
  --deployment-evidence "$watchdog_evidence/DEPLOYMENT_EVIDENCE.json" \
  --heartbeat "$watchdog_evidence/heartbeat-1.json" \
  --heartbeat "$watchdog_evidence/heartbeat-2.json" \
  --operator "$USER" --confirm-email-received \
  --output "$watchdog_evidence/LIVENESS_ACKNOWLEDGEMENT.json"
"$sealed_python" -I "$watchdog_tool" liveness-evidence \
  --deployment-evidence "$watchdog_evidence/DEPLOYMENT_EVIDENCE.json" \
  --heartbeat "$watchdog_evidence/heartbeat-1.json" \
  --heartbeat "$watchdog_evidence/heartbeat-2.json" \
  --acknowledgement "$watchdog_evidence/LIVENESS_ACKNOWLEDGEMENT.json" \
  --output "$watchdog_evidence/LIVENESS_EVIDENCE.json"
test ! -w "$watchdog_evidence/DEPLOYMENT_EVIDENCE.json"
test ! -w "$watchdog_evidence/LIVENESS_EVIDENCE.json"
```

The service and timer files above come only from the bundle renderer; there is no
second static unit source. Deployment acceptance verifies the complete installed
runtime inventory, the exact non-symlink interpreter, a successful isolated
`python -I` import probe, `Result=success`, `ExecMainStatus=0`, and a sealed
successful service heartbeat. Do not manually publish `WATCHDOG_READY.json`; stage
19 combines these inputs with its exact-namespace cancellation drill and publishes
that marker last.

## Render and submit the superseding chain

Rendering and submission are dry-run by default:

```bash
renderer="$pilot_checkout/scripts/render_schema5_recovery_chain_v12.py"
sealed_python="$(realpath -e "$sealed_python")"
test -x "$sealed_python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

"$sealed_python" -I "$renderer" render \
  --repository "$repo" \
  --results-root "$results" \
  --recovery-root "$recovery" \
  --hf-home /orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  --dev-python "$dev_python" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --conda-executable "$conda_exe" \
  --materialization-pilot-root "$pilot_root" \
  --slurm-canary-root "$canary_root" \
  --partition mit_normal --slurm-user "$slurm_user"

"$sealed_python" -I "$renderer" render \
  --repository "$repo" \
  --results-root "$results" \
  --recovery-root "$recovery" \
  --hf-home /orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  --dev-python "$dev_python" \
  --source-harness-prefix /orcd/home/002/mabdel03/conda_envs/asys_env \
  --source-serving-prefix /orcd/home/002/mabdel03/conda_envs/serve_env \
  --conda-executable "$conda_exe" \
  --materialization-pilot-root "$pilot_root" \
  --slurm-canary-root "$canary_root" \
  --partition mit_normal --slurm-user "$slurm_user" \
  --apply

"$sealed_python" -I "$renderer" verify --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" submit --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" submit --chain-manifest "$chain_manifest" --apply

chain_receipt="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R2_SUBMISSION.json"
root_release="$recovery/RECOVERY_CHAIN_ROOT_RELEASE_COMPLETE.json"
launch_complete="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R2_LAUNCHED.json"
jq -e --arg manifest "$chain_manifest" \
  '.passed == true and .no_requeue == true and .manifest == $manifest and
   .root_initial_hold == true and
   .dependency_canary.kill_invalid_depend == true and
   .dependency_canary.alert_latency_bound_seconds == 180 and
   (.jobs | length) == 43 and
   ([.jobs[].job_id] | length == (unique | length))' \
  "$chain_receipt"
jq -e --arg receipt "$chain_receipt" \
  '.root_no_longer_held == true and .receipt == $receipt and
   .dependency_canary.kill_invalid_depend == true and
   (.release_attempts | length) >= 1' "$root_release"
jq -e --arg receipt "$chain_receipt" --arg release "$root_release" \
  '.receipt == $receipt and .root_release == $release and
   .root_initial_hold == true and .alert_latency_bound_seconds == 180' \
  "$launch_complete"
while IFS=$'\t' read -r job_id name comment script; do
  expected_name="$(jq -er --arg name "$name" \
    '.jobs[] | select(.name == $name) | .job_name' "$chain_manifest")"
  job_record="$(scontrol show job -o "$job_id")"
  [[ "$job_record" == *" JobName=$expected_name "* ]]
  [[ "$job_record" == *" Requeue=0 "* ]]
  [[ "$job_record" == *" Comment=$comment "* ]]
  [[ "$job_record" == *" Command=$script "* ]]
done < <(jq -r '.jobs[] | [.job_id, .name, .comment, .script] | @tsv' \
  "$chain_receipt")
```

The DAG uses immutable generation-addressed scripts, `--no-requeue`, exact
`afterok` dependencies, durable submission intents, one fail-fast `afterany` observer
for each of the 21 production stages, and one aggregate `afterany` classifier over
all stages and observers. The source-checkout root is initially held. The submitter
publishes all 43 jobs and the read-only receipt, reparses and persists the live
dependency policy, then records a marker-first exact-ID release intent before
`scontrol release`.
`RECOVERY_CHAIN_SCHEMA5_V1_2_R2_LAUNCHED.json` is published last. A crash after
release is reconciled by the exact job/comment instead of issuing a second release.
The same transaction holds the first resubmitted job of each repair generation until
that generation's complete repair receipt exists. A repaired production stage always
brings its generation-scoped observer with it. Deterministic contract failures require
a superseding release; use same-generation suffix repair only for the aggregate
classifier's verified scheduler or node transient outcome.

The generation-scoped `throughput_qualification` stage 18 is ordered after
`smoke_readiness`. It directly executes the frozen
`run_schema5_throughput_qualification.py execute` producer with a ten-hour deadline,
then executes the immutable renderer's full verifier over
`readiness/schema5_throughput_qualification_v1/THROUGHPUT_QUALIFICATION_COMPLETE.json`.
It does not wait for or adopt a pre-existing qualification marker.
`controller_drill` stage 19 depends on qualification and the independent email
branch; it publishes and verifies the external-watchdog evidence before running the
controller and fleet-supervisor drills. `production_resume` names both stages 18 and
19 as exact `afterok` dependencies and re-verifies `WATCHDOG_READY.json`.

Scheduler release and marker-last filesystem publication cannot be one atomic
operation. Consequently, every production allocation begins with a sealed-bootstrap
launch-authorization gate before any stage mutation. The gate waits at most 180
seconds for the generation-local submission receipt, exact-root release completion,
and launch completion; it then revalidates their canonical identities and hashes,
the current scheduler comment/job ID, every immutable sbatch, the live dependency
policy receipts, the bounded dependency canary, and protected capacity. Stage 19
then validates the forced-command external watchdog against initialized control and
publishes its completed drill/readiness markers; production resume validates them
again. This applies equally to the first resubmitted suffix job in a repair
generation. If publication crashes after
`scontrol release`, the root fails closed without entering its body. All 21 fail-fast
observers and the aggregate failure sentinel are intentionally not launch-gated so
they can record and alert on that boundary failure.

Each stage observer authenticates its sealed sentinel executable, exact generation
comment, own Slurm job ID, immutable manifest, and generation receipt. It reparses
complete `squeue` plus `sacct` truth and requires its target stage to be terminal and
its own exact `afterany:<target-job-id>` allocation to be running. It publishes
`STAGE_SCHEDULER_EVIDENCE.json`, persists bounded email-delivery attempts for every
non-success terminal state, and publishes `STAGE_SENTINEL_COMPLETE.json` last under
`recovery_chain_stage_sentinels/schema5-v1.2-r2/gNNNN/<stage>/`. A stage observation
explicitly has no repair authority. The aggregate sentinel remains the sole causal
classifier and suffix-repair authority; it also verifies that every stage observer
terminated successfully. Thus a failure in either parallel readiness branch is
recorded and alerted as soon as that branch terminates, without waiting for the
long-running email-acknowledgement branch or for aggregate classification.

The durable Git release marker, pilot and canary roots, and protected-capacity
marker are required
pre-render inputs at their canonical r2 paths. Rendering fails if any of those
inputs is missing, writable, symlinked, tampered, belongs to another
tag/commit/chain namespace, or fails its sealed verifier. Chain schema 9 binds the
durable bundle/checksum marker and the protected-capacity path, raw SHA-256, size,
and self-hash identity, as well as the
pilot/canary verifier reports, annotated tag object, commit, prerequisite source
hashes, and the real dependency-cascade proof. Watchdog drill/readiness markers are
stage-19 outputs and therefore are deliberately absent from the render prerequisite
contract.
The prerequisite evidence binds the tagged
`environment_ownership_policy.v1.json` and
`environment_integrity_normalization_policy.v1.json` bytes separately and rejects a
pilot or completed production capture whose corresponding hashes differ.
Creation-time render and both pre-lock and in-lock `submit --apply` checks
rerun the live gates. Sealed `verify` uses only the immutable job-tool bundle,
bootstrap, pilot/canary markers, and sealed pilot runtime; it requires neither the
mutable repository nor the external Conda installation.

The immediate receipt loop verifies the effective Slurm value `Requeue=0`, exact
generation comment, job name, and submitted script for every accepted job while
Slurm retains completed records. Do not infer submission from terminal output alone:
the read-only 43-job receipt is published only after every durable intent has
reconciled to exactly one scheduler job. No chain job can execute before that point,
because the sole dependency-free root remains under the user hold. The root release
then fails closed unless a fresh, checksummed raw scheduler-policy check still
contains `kill_invalid_depend`. The sealed canary's measured 180-second bound is the
release acceptance bound for early-failure sentinel alert eligibility.

The source-checkout allocation verifies the bound marker hashes before any early
exit or checkout mutation. It independently checks the bundled verifier's regular
file type, non-symlink path, read-only mode, size, and exact tagged SHA-256 before
executing it; after cloning, it applies the same checks to the target tagged verifier
before rerunning both full prerequisite verifiers. It then seals the entire checkout
read-only and publishes `SOURCE_CHECKOUT_SCHEMA5_V1_2_R2_COMPLETE.json` last; jobs
02–05 recheck the annotated tag object, commit, clean status, worktree/index diff,
and full `git fsck` immediately before use. Every fail-fast and aggregate `afterany`
sentinel performs the same independent checks on its bundled sentinel executable
before use.

No pre-freeze allocation executes the mutable developer Python. Source checkout,
maintenance, snapshot verification, and environment capture use the complete
read-only Python/stdlib/native-library bootstrap inventoried at render time.
Maintenance itself is an immutable inline stdlib-only check: it validates the exact
interlock, runs the user-scoped full `squeue` query and legacy job filters, and probes
every existing legacy cell lock with the original nonblocking `flock` semantics. It
does not import the experiment/serving client graph.
Environment capture computes the current full harness and serving inventories before
copy, requires both to equal the pilot hashes, and checks the marker-last capture
inventories and both independent policy hashes after copy. A completed capture may be
resumed without rereading live sources, but it must still prove it captured the pilot
bytes and used both pilot-bound policies.
This capture is the last production read of either live prefix.

Materialization first re-verifies the capture and pilot-inventory binding with the
sealed bootstrap, then executes from the read-only captured-harness Python. It checks
the Conda executable's canonical regular executable path, size, and SHA-256 against
the pilot immediately before both the dry-run and apply invocations. Freeze and all
later work execute from the materialized immutable harness. Any source-prefix, Conda,
bootstrap, or captured-interpreter drift fails closed before release publication.

If the sentinel classifies `release_materialize` as a repairable scheduler/node
transient and that failed job left an incomplete release tree, quarantine and seal
the exact partial tree before any suffix repair. Do not use this command for a
deterministic contract failure, an active chain, a completed materialization, or when
no partial release tree exists:

```bash
"$sealed_python" -I "$renderer" quarantine-materialization \
  --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" quarantine-materialization \
  --chain-manifest "$chain_manifest" --apply
"$sealed_python" -I "$renderer" quarantine-materialization \
  --chain-manifest "$chain_manifest" --apply
```

The first command is non-mutating. The first `--apply` publishes the marker-first
intent, performs the same-filesystem inode-preserving rename, recursively inventories
and seals the quarantine, and publishes its completion evidence last. Repeating
`--apply` proves crash recovery, sealing, and idempotency. The command rechecks that
every chain job is terminal and that the exact failed materialization job remains
repairable; it rejects either materialization/release completion marker. Only after
the repeated call reports `already_quarantined_and_sealed` may the following
dry-run/apply suffix repair proceed:

```bash
"$sealed_python" -I "$renderer" repair-chain --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" repair-chain --chain-manifest "$chain_manifest" --apply
```

## Email acknowledgement and production

The email gate sends a one-time token and an exact acknowledgement command to
`mabdel03@mit.edu`. It remains blocked until that received token is explicitly
acknowledged. There are no fixed `email_ack_request.json` or
`email_acknowledgement.json` paths: each repair generation publishes a fresh,
request-ID-scoped challenge. Prefer running the exact command in the received email
once. To perform a non-mutating validation immediately before consuming the token,
derive only the active request and output paths from the marker-last pointer:

```bash
active_challenge="$recovery/readiness/email_challenges/CURRENT.json"
request="$(jq -er '.request' "$active_challenge")"
output="$(jq -er '.acknowledgement' "$active_challenge")"
ack_tool="$(jq -er '.acknowledgement_tool.path' "$request")"
release_python="$recovery/releases/$release_id/environments/harness/bin/python"

read -r -s -p "One-time token from the received email: " email_token
echo
"$release_python" -I "$ack_tool" \
  --request "$request" --output "$output" --token "$email_token"
"$release_python" -I "$ack_tool" \
  --request "$request" --output "$output" --token "$email_token" --apply
unset email_token
```

The acknowledgement tool rejects a stale/superseded `CURRENT.json`, a mismatched
request or output path, an expired challenge, and replay. Do not synthesize a token,
copy a token from logs or durable state, or acknowledge a challenge from an earlier
repair generation.

Production first resumes at ceiling 24 only after static, context, email, fleet,
41-cell smoke, full-capacity throughput qualification, the external-watchdog drill,
and exact-ID controller/fleet-supervisor kill-drill gates pass. Use
`schema5_control.py status --live` for scheduler-joined state. Any integrity hold sets
the admission ceiling to zero; clear it only through the documented semantic scan and
`ack-hold` transaction.

Smoke readiness is generation- and attempt-scoped. The three 15/20/6 manifests are
materialized only under
`$results/schema5-smoke-attempt-runs-v1/<attempt-id>/`; fixed top-level smoke run
roots are forbidden. Attempt pointers and evidence live under
`$recovery/readiness/schema5-smoke-readiness-v1/`. A killed worker, incomplete
request, or transport-censored attempt is inventoried and recursively sealed before
a fresh empty successor is allowed to draw. `CURRENT.json` is published last only
after exactly 41 clean cells, and the renderer resolves and verifies that selector
before the control plane attests its selected immutable `smoke_runs.json`. Never
manually repoint `CURRENT.json` or copy rows between attempts.

An additive qualification capacity transition must replay the exact readiness suffix
`fleet_readiness -> smoke_readiness -> throughput_qualification`. The replacement
smoke attempt carries the new fleet contract, capacity generation, rollout
generation, and trusted endpoint-catalog ID. Reusing a prior-generation success or a
fixed smoke evidence path fails closed.

The r2 controller implements the complete `24 -> 96 -> 192 -> 384` rollout state
machine, and launch authorization now requires protected placement for that full
design ceiling before the first ceiling-24 cell is admitted. Scientific cell arrays
and every scientific serving allocation must use authorization-bound partitions with
`PreemptMode=OFF`; neither clients nor servers may use `mit_preemptable`,
`ou_bcs_low`, or any other preemptible placement. The protected-capacity marker must
already prove the 384-cell ceiling, 64-job reserve, 448-job submit headroom, full
CPU/memory reservation, 24 active GPUs, and four warm-headroom GPUs.

Live admission preserves the same inclusive contract:
`min(384 - active/pending cells, 448 - all live user job elements,
client CPU/memory headroom)`. The global term includes every partition. Protected
servers are excluded from the client-only CPU/memory term only when their exact job
ID, name, comment, fleet intent, immutable sbatch, and sealed endpoint history all
agree; they still count toward the 448 total. Any unknown or mismatched job on the
scientific-client placement is charged as a client consumer, so a copied job name
cannot create capacity.

The dispatcher and fleet-supervisor controller processes may use a controller-only
placement because they issue no scientific HTTP requests. Their durable intent,
successor, heartbeat, and exact-ID takeover protocols are independently covered by
the forced-command external watchdog. The watchdog's five-minute scheduler
observation and completed namespace kill drill are launch prerequisites, not
best-effort monitoring after admission.

Each stochastic coordinate is still fsynced as a pending intent before its one SDK
call. An ambiguous connection loss, timeout, or process interruption becomes one
terminal reported transport censor and is never redrawn. Any new transport censor is
a scientific-integrity incident: admission drains to zero and remains held until a
clean semantic scan and explicit `ack-hold`. This censor contract preserves estimator
semantics; it never authorizes a preemptible scientific server or client.

At every dispatcher poll and again under the admission lock immediately before
`sbatch`, the dispatcher re-queries the authorization-bound partition and QOS,
computes headroom from all visible CPU/memory/job/GPU reservations, and renders those
exact protected placements into the immutable microbatch. Any live drift from the
sealed limits aborts submission and drains admission to zero. There is no
generation-1 96-cell fallback and no policy path that treats ceilings 192 or 384 as
optional.

The clean-window and stall contracts are:

| configured ceiling | automatic next stage | continuous clean evidence | hold if unresolved |
| ---: | ---: | ---: | ---: |
| 24 | 96 | one hour, with useful trusted-QID progress in every run | eight hours |
| 96 | 192 | six hours | thirteen hours |
| 192 | 384 | twelve hours | nineteen hours |
| 384 | terminal | continuous monitoring | n/a |

Because semantic scans run every six hours, promotion occurs on the first qualifying
semantic or daily observation after the minimum interval, with five-minute health
observations no more than 660 seconds apart. The preproduction qualification exercises
768 cells and 15,360 QIDs across ceilings `[24, 96, 192, 384]`, holds ceiling 384
steadily for at least 7,200 seconds, requires useful progress in every stratum, and
requires at least 201,994 trusted QIDs/day with zero integrity or transport-censor
incidents. Its schema-1
`THROUGHPUT_QUALIFICATION_COMPLETE.json` self-hash binds the chain manifest and the
protected-capacity prerequisite identity and raw hash. Watchdog evidence is produced
and verified in stage 19 after qualification.

At 48 continuous production hours every stratum must still have measured throughput,
aggregate throughput must remain at least 201,994 trusted QIDs/day, and both overall
and worst-stratum ETAs must be at most 28 days. A failed capacity gate requires
explicit capacity work; treatments are never trimmed and scientific work is never
routed to preemptible capacity.

## Controlled capacity transition

Capacity changes publish a new operational generation; they do not mutate the frozen
release fleet contract, manifests, run IDs, model contracts, or artifact policy.
Serving changes use an additive fleet overlay that retains every base replica unchanged.
A different or larger client placement additionally requires the separate sealed
client-placement authorization described below; a serving overlay by itself never
authorizes a partition or QOS change, and no transition may reduce the protected
384-cell/64-reserve launch floor. Use the immutable release harness and worktree:

```bash
release_root="$recovery/releases/$release_id"
harness="$release_root/environments/harness"
worktree="$release_root/worktree"
control=("$harness/bin/python" -I "$worktree/slurm/schema5_control.py" \
  --state-dir "$state")
overlay=/absolute/path/to/schema5_fleet.capacity-gNNNN.json
overlay_sha="$(sha256sum "$overlay" | awk '{print $1}')"

"${control[@]}" pause --drain

# The omitted action is a non-mutating dry run.
"${control[@]}" capacity-transition \
  --fleet-contract "$overlay" \
  --fleet-contract-sha256 "$overlay_sha"

"${control[@]}" capacity-transition apply \
  --fleet-contract "$overlay" \
  --fleet-contract-sha256 "$overlay_sha"
"${control[@]}" capacity-transition status
```

`apply` records the intent before mutation, archives and seals the old transaction
state, retires only the exact recorded old job IDs, launches the replacement fleet
transactionally, and leaves admission held. It is safe to repeat after a crash: the
same transition is adopted, never duplicated. A different overlay or any removal,
replacement, reordered base identity, or other non-additive change fails closed and
requires a superseding release.

While the capacity hold remains active, rebuild and attest the `fleet`,
`smoke_runs`, and `scheduler_reconciliation` gates for the new capacity generation.
Fleet readiness must bind the overlay's canonical path and SHA-256 and prove the
effective replica/GPU totals, complete `squeue` plus `sacct` reconciliation, and two
provenance-correct probes per endpoint. Rerun the affected smoke suites under the
new effective fleet and planned rollout generation. Do not reuse base-generation
evidence.

For any replacement protected client placement, use the marker-last builder below.
Do not hand-author any of its evidence files or invoke `sbatch` yourself. `dry-run`
is read-only. `prepare` records the immutable build intent first, captures a sealed
scheduler preflight, and emits the exact immutable no-requeue client canary sbatch
path; it never submits a job. `submit` is the only submission entry point. Before
its first `sbatch` it persists a marker-first submission intent and attempt, then
requires complete `squeue` plus `sacct` truth and exact comment, immutable path,
script hash, job name, partition, and job-ID identity. A replay adopts exactly one
matching visible or terminal job and never submits a duplicate. It publishes the
accepted-submission marker last. An ambiguous retry requires two independently
sealed, source-verified absence observations at least 60 seconds apart; the next
attempt recursively binds their sealed retry authorization. Any complete or partial
candidate-bearing observation permanently forbids a second submission and is adopted
when its complete evidence is available. `apply` derives the completed canary ID only from
that sealed accepted marker, recaptures live scheduler/QOS truth, revalidates all
generation-bound post-transition readiness gates, and publishes
`CLIENT_CAPACITY_COMPLETE.json` last. The authorization protocol is
`schema5-v1.2-r2-client-placement-capacity-generation-v1`; it binds the immutable
control hash, current capacity contract/generation, exact target ceiling,
`PreemptMode=OFF`, CPU, memory, submit headroom, apply-time capture timestamp, and all
four rehashed evidence identities.

```bash
"${control[@]}" reconcile --all --no-admit

client_partition=schema5_clients_gNNNN
target_ceiling=384 # 384 plus the 64-job reserve is the minimum admissible generation
capacity_dry_run="$recovery/client-capacity-dry-run.json"
capacity_prepare="$recovery/client-capacity-prepare.json"
capacity_submit="$recovery/client-capacity-submit.json"
capacity_apply="$recovery/client-capacity-apply.json"

"${control[@]}" build-client-capacity dry-run \
  --partition "$client_partition" --target-ceiling "$target_ceiling" \
  > "$capacity_dry_run"
"${control[@]}" build-client-capacity prepare \
  --partition "$client_partition" --target-ceiling "$target_ceiling" \
  > "$capacity_prepare"

capacity_submit_next="${capacity_submit}.next"
while true; do
  "${control[@]}" build-client-capacity submit \
    --partition "$client_partition" --target-ceiling "$target_ceiling" \
    > "$capacity_submit_next"
  jq -e . "$capacity_submit_next" >/dev/null
  mv -- "$capacity_submit_next" "$capacity_submit"
  client_canary_state="$(jq -er '.state' "$capacity_submit")"
  case "$client_canary_state" in
    adopted|already_submitted) break ;;
    submission_pending) sleep 60 ;;
    *) exit 1 ;;
  esac
done
client_canary_job_id="$(jq -er '.job_id' "$capacity_submit")"
[[ "$client_canary_job_id" =~ ^[0-9]+$ ]]

# Wait for this exact accepted ID to reach COMPLETED/0:0. Repeating submit adopts
# it through complete scheduler truth. apply independently proves the sacct row,
# SubmitLine, immutable script hash, 1 CPU, 4 GiB, and --no-requeue.
"${control[@]}" build-client-capacity apply \
  --partition "$client_partition" --target-ceiling "$target_ceiling" \
  > "$capacity_apply"

client_capacity="$(jq -er '.authorization' "$capacity_apply")"
client_capacity_sha="$(jq -er '.sha256' "$capacity_apply")"
"${control[@]}" attest-client-capacity \
  --evidence "$client_capacity" --sha256 "$client_capacity_sha"

# The monitor resolves monitor:capacity-gate after the attestation. Require a fresh
# clean semantic scan, then acknowledge only that scientific-integrity layer.
"${control[@]}" ack-hold \
  --note "capacity generation and post-transition semantic scan verified"
"${control[@]}" capacity-transition complete
"${control[@]}" status --live
```

Repeat `complete` if an operator process exits after controller submission but before
the marker-last commit; it resumes idempotently only after all generation-bound gates
and scheduler truth are clean. `complete` rejects an unresolved client-capacity request
even if the replacement serving fleet is healthy. Successful completion clears only
the transition's operator hold, resumes at ceiling 24, and opens a new throughput
epoch; the normal one-hour/six-hour/twelve-hour state machine must prove its stages
again. Any concurrent critical or scientific-integrity hold remains at ceiling zero
and requires its own clean semantic scan and, where required, explicit `ack-hold`.

Workers and artifacts carry both the immutable release fleet hash and the active
overlay hash, plus capacity, rollout, and endpoint generations. A QID interrupted
mid-request retains per-coordinate provenance in its durable checkpoint; a partial
cell may therefore span generations without redraw. Canonical result rows and final
metadata store exact per-generation coordinate counts. Mixed active generations are
represented explicitly rather than collapsed into a false scalar identity.

### Qualification-only additive transition

A stage-18 exit `76` is reserved for a sealed throughput-only shortfall. The
aggregate sentinel accepts that disposition only when the current immutable attempt
and run are recursively read-only, the failure self-hash and attempt pointer agree,
and the marker requests exactly one replica (two GPUs only for `32B-long`). Any
integrity, transport-censor, context, semantic, protocol, or unsealed failure remains
`requires_superseding_release`.

Keep production paused. Apply the exact additive fleet overlay named by the failure,
advance capacity and rollout generations, rebuild the trusted endpoint catalog and
readiness evidence, then seal the post-transition authority:

```bash
qualification="$release_worktree/scripts/run_schema5_throughput_qualification.py"
"$harness/bin/python" -I "$qualification" publish-capacity-transition \
  --chain-manifest "$chain_manifest" \
  --submission-receipt "$failed_generation_receipt" \
  --failed-job-id "$failed_stage18_job_id" \
  --failed-comment "$failed_stage18_comment" \
  --apply
```

This command performs no scheduler mutation. It verifies paused control, unchanged
r2 release/scientific identities, the original sealed failure and receipt, the
authoritative additive contract, and strictly newer capacity, rollout, catalog,
fleet, and readiness bindings. Only then may `repair-chain --apply` resubmit the
stage-18 suffix. The suffix includes stage 18, its observer, `controller_drill`,
`production_resume`, their affected observers, and the aggregate sentinel; completed
upstream jobs and fairness state are reused. Successful stage 18 publishes the
fixed-root marker last and attests the control-visible
`throughput_qualification` gate. Never delete or rewrite a failed attempt, pointer,
or transition receipt.

Final completion is published only by the marker-last finalizer after drain, zero
writers/locks, exact semantic validation of all 22,680 cells and 4,524,660 trusted
outcomes, a checksummed final snapshot, and a schema-5-only analysis cache.
