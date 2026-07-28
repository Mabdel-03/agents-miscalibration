# Schema-5 production release

Schema-5 production must run from a clean, exact Git tag and two independent,
read-only Conda prefixes. Development environments are inputs only: neither their
editable checkout nor their hardlinked package files are accepted as production
identity.

## Superseding operational lineage

The current operational tag is `sweep-recovery-schema5-v1.2-r9` and its chain
namespace is `schema5-v1.2-r9`. The r2 through r8 tags, remote branches, durable
bundles/checksums, and completion markers are immutable historical evidence and must
not be moved, rewritten, or copied into r9 paths. The deterministic r2 live-canary
scheduler-capture failure has disposition `requires_superseding_release`; its
original partial tree remains recursively read-only in place and is bound by
`$schema5_recovery/canary_failures/schema5-v1.2-r2/CANARY_FAILURE_SEALED.json`
using protocol `schema5-v1.2-r2-partial-canary-failure-seal-v1`.

The immutable r3 release (`acd723ba9a99d88e77f7d752268bc31205c3a808`, annotated
tag object `fa87b283974afc1fa48fcb22b61e6ae7eec4bb36`) exposed two deterministic
prelaunch defects: the shared Miniforge runtime contains a recorded broken internal
compiler-tool symlink, and its release-local offline clone cache was not seeded.
No r3 scheduler job or scientific mutation followed. The marker-last evidence at
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r3/PRELAUNCH_FAILURE_SEALED.json`
must verify under `schema5-v1.2-r3-prelaunch-failure-seal-v1`, contain both exact
failure classifications, set `retry_in_place=false` and
`requires_superseding_release=true`, and report an empty scheduler-job list.

The immutable r4 release (`f23cf1c4b2d2bf606afd013123b9afe9c614bb6c`) reached no
scheduler or scientific-result boundary. Its canonical production path required a
157-byte absolute Python shebang, causing Miniforge to install `bin/conda` with
`#!/usr/bin/env python`. The r4 verifier rejected that fallback. The tagged r9
failure sealer preserves both r4 attempts and publishes
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r4-toolchain/TOOLCHAIN_FAILURE_SEALED.json`
under protocol `schema5-v1.2-r4-overlong-conda-prefix-failure-seal-v1`.

The immutable r5 release (`fa50d619579d6ffde9ab768d342bb65f1b98d844`,
annotated tag object `51267bb1bbe59833bfadc96b4d3ad74355ce1d39`) built and
verified its isolated short-prefix toolchain, then failed before either r3 probe ran.
Its tagged CLI used the same argparse destination for the subcommand selector and the
remainder probe argv, causing deterministic verifier-branch misdispatch and
`AttributeError` before any envelope, scheduler job, or scientific mutation. The r9
failure sealer binds the exact traceback reproduction, unchanged probe tree, r5
source/toolchain identities, and publishes
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r5-cli/PRELAUNCH_CLI_FAILURE_SEALED.json`
under protocol `schema5-v1.2-r5-prelaunch-cli-dispatch-failure-seal-v1`.

The immutable r6 release (`9b40414a0d90f569cf021ac152503746bedda6c7`,
annotated tag object `e6b9c2f2548f7ef58ad5163b65a25b164013a4e6`) built and
verified its isolated toolchain. Its first r3 recorder dry run then rejected that
toolchain before probe execution because the binding producer returned the valid
`portable_shebang` field while the recorder's exact field set omitted it. The r9
failure sealer binds the one-field set difference, empty probe tree, exact exit-2
diagnostic, r6 source/toolchain identities, scheduler quiescence, and zero scientific
mutation, and publishes
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r6-toolchain-binding/PRELAUNCH_TOOLCHAIN_BINDING_FAILURE_SEALED.json`
under protocol
`schema5-v1.2-r6-prelaunch-toolchain-binding-failure-seal-v1`.

The immutable r7 release (`acc0beb98cfbb5017d5d5b5e60567aa43229e248`,
annotated tag object `95f8f3852498c4ab8e3026f29c183ecdd58678f4`) built and
verified its isolated toolchain. Its first r3 recorder dry run then detected that the
recorder's own Git identity query had replaced the freshly cloned `.git/index`.
Index bytes, size, and device were unchanged, but the inode changed; no probe,
scheduler job, or scientific mutation followed. The r9 failure sealer reproduces
that exact replacement, proves that `GIT_OPTIONAL_LOCKS=0` preserves the index for
the same read-only query sequence, binds the r7 source/toolchain identities and
quiescent namespaces, and publishes
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r7-git-index-refresh/PRELAUNCH_GIT_INDEX_REFRESH_FAILURE_SEALED.json`
under protocol
`schema5-v1.2-r7-prelaunch-git-index-refresh-failure-seal-v1`.

The immutable r8 release (`ed18bb969c6632d2b2643024064a2b4d1b63f6de`,
annotated tag object `c31692215b98c176bdd138109b0c2b6ec0a63e1f`) built and
verified its isolated toolchain. Its first exact broken-link recorder apply launched
the full r3 materialization-pilot CLI under that minimal toolchain and failed while
importing the unrelated project dependency `backoff`, before reaching the stdlib-only
runtime-identity subcommand or publishing an envelope. No scheduler job or scientific
mutation followed. The r9 failure sealer binds both exact failed commands, the empty
probe tree, r8 source/toolchain identities, and quiescent namespaces, and publishes
`$schema5_recovery/prelaunch_failures/schema5-v1.2-r8-r3-probe-import/PRELAUNCH_R3_PROBE_IMPORT_FAILURE_SEALED.json`
under protocol
`schema5-v1.2-r8-prelaunch-r3-probe-import-failure-seal-v1`.

The r9 recovery renderer must independently verify and bind all seven exact historical
seals before render and submission. All r9 pilot, canary, source-checkout, batch-job/log, and
protected-capacity roots are new. Every newly created r9 artifact uses a
`schema5-v1.2-r9-*` protocol identity and binds the r9 tag and namespace. An r2
or r3/r4/r5/r6/r7/r8 protocol is accepted only while verifying the explicitly named immutable
historical evidence above; it is never emitted for fresh production state.

## Contract

The release pipeline has three distinct marker-last steps:

1. `capture_schema5_environments.py` copies both mutable developer prefixes with
   explicit buffered read/write I/O, validates before/after inventories, normalizes
   only the checksummed metadata defects authorized below, and seals immutable seeds.
2. `materialize_schema5_release.py` creates an independent tagged repository clone and
   independent harness/serving prefixes from those sealed seeds. It records
   `MATERIALIZATION_COMPLETE.json` only after all stages validate.
3. `freeze_schema5_release.py` inventories and seals those live inputs and publishes
   `RELEASE_COMPLETE.json` only after re-reading every input and artifact.

Prelaunch protected capacity is deliberately the tagged base fleet, not a speculative
scale-up: generation one has 22 logical replicas on 24 active GPUs, zero additive
replicas/GPUs, three warm-turnover jobs using four GPUs, and exactly 28 attested GPUs.
The inclusive 64-job non-cell reserve is 22 active servers + 3 warm jobs + 39 held
controller/monitor/other slots; with 384 clients this is exactly 448 submitted job
elements. The zero-QID admission certificate truthfully records the base fleet's
278/384 selected cells and 106-cell shortfall. Additive replicas are permitted only
after qualification fails and a controlled capacity transition publishes a new
generation and reruns the affected readiness and smoke gates.

No program invokes Conda against either developer prefix. Capture uses neither
hardlinks nor reflinks and requires equal source-before, source-after, and copied
inventories plus zero shared regular-file inodes. It validates all installed
distribution identities and hash-bearing pip `RECORD` rows. Every unlisted Conda/pip
version conflict fails.

The ownership normalization preserves the observed `setuptools==81.0.0` runtime and
removes the stale `setuptools 82.0.1` Conda record whose artifact SHA-256 is
`82088a6e…4e9e1`. A complete record preimage is archived before removal. If the live
record is already absent, capture succeeds only when a read-only Conda-reconciliation
incident and the recovered exact record bind that absence. The policy is frozen in
`configs/environment_ownership_policy.v1.json`.

The sealed July 23 incident records the stale Conda record as absent from both current
live prefixes and identifies the recovered harness preimage in the quarantined r1
release. Capture requires each prefix's observed presence/absence to equal the
role-specific boolean in that incident before copying; merely mentioning the prefix is
not sufficient authorization. A reappearing or newly missing record therefore fails
closed rather than being silently normalized.

Ownership authority ends with that one Setuptools collision. A second, independently
checksummed integrity policy,
`configs/environment_integrity_normalization_policy.v1.json`, authorizes exactly five
captured-seed-only RECORD repairs:

| Role | Distribution | Exact repair | Source → postimage SHA-256 |
|---|---|---|---|
| Both | `pip==26.1.1` | Remove the three stale `pip`, `pip3`, and absent `pip3.12` launcher claims; rewrite `INSTALLER` from the stale four-byte claim to the observed five-byte `conda` file. | `e6838013…e14bfc1` → `aaf8210b…a279c9` |
| Both | `packaging==26.2` | Rewrite `INSTALLER` from the stale four-byte claim to the Conda-owned six-byte `conda\n` file. | `8548d2f3…363c88` → `6096a53c…0ce189` |
| Both | `wheel==0.47.0` | Remove the stale generated `bin/wheel` launcher claim while pinning each role's actual launcher bytes. | `2f377d00…1f605b` → `8b1b6702…21f90e` |
| Serving | `numpy==2.3.5` | Remove only the stale hash-bearing duplicate generated-bytecode row and retain its canonical unhashed wheel row. | `1eec6aac…723f5` → `c158387d…00187` |
| Serving | `torch-c-dlpack-ext==0.1.5` | Remove its stale claims on both top-level `build_backend.py` and its generated bytecode; retain FlashInfer's authoritative claims. | `b12cd315…716655` → `5f6e642b…07f36f` |

The integrity policy cannot contain Conda/pip ownership-authority fields, and the
ownership policy cannot contain a second normalization. Every source RECORD, target
runtime file, role-specific launcher, Conda ownership
record, other-owner RECORD, and normalized postimage is pinned. The source audit
validates every non-mutated declared hash and size and additionally projects the
post-repair owner graph; that graph must contain zero shared paths. Capture first
publishes one role-scoped marker-first intent that binds both policy hashes and
archives every complete RECORD
preimage, then applies only the exact declared row mutations atomically. Each
role-specific receipt binds the source hash, postimage hash, archived preimage,
mutation list, and ownership evidence. The actual launcher/runtime bytes are never
changed and remain pinned by the complete installed-file inventory. Interrupted
rewrites restore every archived source RECORD before the inventory-bound copy
resumes. Sealed-seed verification requires exact normalized postimages, zero policy
exemptions, and zero shared RECORD paths; it never accepts a source RECORD or a
broader validation exemption.

Wheel-spec unhashed and unsized generated `.pyc`/`.pyo` entries may name a cache that
is no longer present. Capture records each such absence explicitly. A missing path
with either a declared hash or size remains fatal, including for bytecode and
Conda-owned distributions.

The materializer uses `conda create --copy --clone --offline` with Conda pip
interoperability and default packages disabled. Its source prefixes must be exactly
the normalized, read-only captured seeds. It uses a release-local checksummed package
cache, compares canonical pip views and direct `conda-meta` locks, runs `pip check`,
and requires zero shared regular-file inodes. An interrupted clone has no trusted
stage marker and is never adopted automatically; quarantine that exact partial prefix
before retrying.

The materializer resolves every destination symlink one dependency hop at a time.
Only links whose complete resolution stays inside the cloned prefix are permitted;
external, source-owned, broken, and cyclic links fail both the clone stage and final
verification. This makes the sealed environment independent of mutable source prefixes
and unpinned system paths even when Conda represents compatibility paths as symlinks.

The cloned harness's editable `agents_scaling` package is uninstalled and reinstalled
with no index, dependencies, build isolation, or generated bytecode from the exact
tagged worktree. An
isolated (`python -I`) probe must import it from inside the new harness prefix. Its
`direct_url.json`, installed-file inventory, version, Git commit, and source-tree hash
are all recorded. The serving prefix is cloned independently but does not install the
harness package; server control code runs under the separately pinned harness Python.
Setuptools's permitted in-place build byproducts,
`src/agents_scaling.egg-info`, `build/`, and `dist/`, are atomically moved into
`build_evidence/`; any other
tracked, untracked, or ignored source mutation fails the materialization. The tagged
worktree is revalidated clean immediately after installation.

## Required two-prefix pilot

Before the recovery chain may consume a release, run the complete transaction once
through `run_schema5_materialization_pilot.py` in a new isolated pilot root. The pilot
requires the expected commit explicitly and rejects a lightweight tag even when it
points at the right commit. It executes each capture, materialization, and freeze
phase as dry-run, apply, and verify; retains full before/after inventories for both
live prefixes; and publishes `PILOT_COMPLETE.json` only after:

- both live source inventories remain byte-for-byte equal across the transaction;
- all four captured/materialized prefixes contain exactly one Setuptools 81
  distribution and no Setuptools 82 Conda record or versioned path;
- every source, seed, clone, package-cache, and worktree copy boundary has zero shared
  regular-file inodes; and
- capture, materialization, and release verification each return identical results
  twice from sealed artifacts.

Dry-run first:

```bash
schema5_recovery=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/recovery/schema5-v1
schema5_pilot="$schema5_recovery/materialization_pilots/schema5-v1.2-r9"
schema5_conda_toolchain_root="$schema5_recovery/toolchains/r9/conda"
schema5_source_package_cache=/orcd/home/002/mabdel03/.conda/pkgs
# The production DAG owns release_source_checkout_v1_2_r9 and requires it to be
# absent at render time.  Keep the prerequisite pilot checkout in its own namespace.
schema5_checkout="$schema5_recovery/materialization_pilot_source_checkout_v1_2_r9"
schema5_commit="$(git -C "$schema5_checkout" rev-parse HEAD)"
test "$(git -C "$schema5_checkout" rev-parse refs/tags/sweep-recovery-schema5-v1.2-r9^{commit})" = "$schema5_commit"
test -z "$(git -C "$schema5_checkout" status --porcelain=v1 --untracked-files=all)"
test ! -e "$schema5_checkout/.git/objects/info/alternates"
schema5_dev_python="$(realpath -e /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python)"
schema5_pilot_script="$schema5_checkout/scripts/run_schema5_materialization_pilot.py"

"$schema5_dev_python" -I "$schema5_checkout/scripts/provision_schema5_conda_toolchain.py" verify \
  --toolchain-root "$schema5_conda_toolchain_root"

"$schema5_dev_python" -I "$schema5_pilot_script" run \
  --pilot-root "$schema5_pilot" \
  --release-checkout "$schema5_checkout" \
  --expected-tag sweep-recovery-schema5-v1.2-r9 \
  --expected-commit "$schema5_commit" \
  --harness-source /orcd/home/002/mabdel03/conda_envs/asys_env \
  --serving-source /orcd/home/002/mabdel03/conda_envs/serve_env \
  --ownership-policy "$schema5_checkout/configs/environment_ownership_policy.v1.json" \
  --integrity-normalization-policy "$schema5_checkout/configs/environment_integrity_normalization_policy.v1.json" \
  --reconciliation-incident "$schema5_recovery/post_snapshot_incidents/2026-07-23_conda_pip_interop_source_metadata_reconciliation.json" \
  --recovered-setuptools-record "$schema5_recovery/releases/quarantine/sweep-recovery-schema5-v1.1.partial-job-18555913/environments/harness/conda-meta/setuptools-82.0.1-pyh332efcf_0.json" \
  --conda-toolchain-root "$schema5_conda_toolchain_root" \
  --source-package-cache "$schema5_source_package_cache" \
  --durable-git-release-marker "$schema5_recovery/DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R9_COMPLETE.json"
```

Do not add `--apply` to that interactive command. The immutable batch job rendered
below is the only production apply path. It passes its own checksummed sbatch receipt
to the pilot, and the running allocation binds its numeric `SLURM_JOB_ID`, complete
`squeue` identity, effective `scontrol` fields, and scheduler-spooled script bytes
before capture begins. The evidence retains the exact argv, return code, raw
stdout/stderr, and hashes for each scheduler query; sealed verification reparses
those raw records instead of trusting derived fields. After terminal accounting is
accepted separately, the verification command uses only sealed
capture/materialization/release and scheduler evidence; it does not require either
live prefix, the source checkout, or Conda. The pilot passes the Conda executable only
to the normalized-seed clone stage; no Conda command is issued against a live
developer prefix.

The superseding recovery-chain renderer requires this exact canonical pilot root as
`--materialization-pilot-root`; it does not discover or accept a nearby marker. It
runs the complete tagged pilot verifier, then binds the canonical root, marker hash
and size, full report and report hash, pilot ID, release tag object and commit, and
the exact tagged pilot/capture/materialize/freeze/runtime-inventory source hashes into
chain schema 11 and prerequisite-evidence schema 12. Creation-time gates rerun
immediately before scheduler submission;
sealed chain verification uses the immutable bundle and sealed evidence without the
mutable source checkout or external Conda installation. Pilot schema 5 also exports
the exact full live harness and serving inventory hashes and the tested Conda
toolchain marker, verified entrypoint, absolute shebang interpreter, and repeated
full base-runtime inventory
(excluding only package caches and child environment roots). Production capture must
match both live-prefix inventory hashes before and after copying; production
materialization checks both the entrypoint and complete runtime-toolchain identity
immediately before each dry/apply invocation and runs from the verified
captured-harness interpreter rather than the mutable developer prefix.

Render the durable batch job from the same exact annotated checkout rather than
running the apply phase interactively:

```bash
schema5_pilot_sbatch="$schema5_recovery/jobs/schema5-v1.2-r9-materialization-pilot.sbatch"
schema5_pilot_receipt="$schema5_pilot_sbatch.receipt.json"
schema5_pilot_logs="$schema5_recovery/logs/materialization-pilot-r9"

"$schema5_dev_python" -I "$schema5_pilot_script" render-sbatch \
  --sbatch-path "$schema5_pilot_sbatch" \
  --log-dir "$schema5_pilot_logs" \
  --partition mit_normal \
  --python-executable "$schema5_dev_python" \
  --pilot-root "$schema5_pilot" \
  --release-checkout "$schema5_checkout" \
  --expected-tag sweep-recovery-schema5-v1.2-r9 \
  --expected-commit "$schema5_commit" \
  --harness-source /orcd/home/002/mabdel03/conda_envs/asys_env \
  --serving-source /orcd/home/002/mabdel03/conda_envs/serve_env \
  --ownership-policy "$schema5_checkout/configs/environment_ownership_policy.v1.json" \
  --integrity-normalization-policy "$schema5_checkout/configs/environment_integrity_normalization_policy.v1.json" \
  --reconciliation-incident "$schema5_recovery/post_snapshot_incidents/2026-07-23_conda_pip_interop_source_metadata_reconciliation.json" \
  --recovered-setuptools-record "$schema5_recovery/releases/quarantine/sweep-recovery-schema5-v1.1.partial-job-18555913/environments/harness/conda-meta/setuptools-82.0.1-pyh332efcf_0.json" \
  --conda-toolchain-root "$schema5_conda_toolchain_root" \
  --source-package-cache "$schema5_source_package_cache" \
  --durable-git-release-marker "$schema5_recovery/DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R9_COMPLETE.json"
```

Repeat the render with `--apply`, review the immutable `.sbatch`, `.sha256`, and
`.receipt.json` files. Validate the receipt against the exact script before crossing
the transactional scheduler submission boundary:

```bash
schema5_pilot_sha="$(jq -er '.sbatch_sha256' "$schema5_pilot_receipt")"
test "$(sha256sum "$schema5_pilot_sbatch" | awk '{print $1}')" = "$schema5_pilot_sha"
test "$(awk '{print $1}' "$schema5_pilot_sbatch.sha256")" = "$schema5_pilot_sha"
jq -e --arg path "$schema5_pilot_sbatch" \
  '.submit_command == ["sbatch", $path] and
   .slurm.no_requeue == true and
   .slurm.time_limit == "11:30:00" and
   .slurm.partition == "mit_normal"' \
  "$schema5_pilot_receipt"

schema5_pilot_submit_cmd=(
  "$schema5_dev_python" -I "$schema5_pilot_script" submit-sbatch
  --pilot-root "$schema5_pilot"
  --sbatch-receipt "$schema5_pilot_receipt"
  --scheduler-user mabdel03
  --visibility-timeout 900
)
"${schema5_pilot_submit_cmd[@]}"
schema5_pilot_submission_json="$("${schema5_pilot_submit_cmd[@]}" --apply)"
jq -e '
  .status == "submitted" or
  .status == "adopted" or
  .status == "already_submitted"
' <<<"$schema5_pilot_submission_json"
schema5_pilot_job_id="$(
  jq -er '.job_id | select(type == "string" and test("^[0-9]+$"))' \
    <<<"$schema5_pilot_submission_json"
)"
```

The generated job is fixed at `11:30:00`, which is below `mit_normal`'s 12-hour
limit, includes `--no-requeue`, explicit output/error paths, an identity-bearing
Slurm comment, and executes the pilot script from the exact tagged checkout. It is
not submitted by `render-sbatch`. The `submit-sbatch` dry-run is non-mutating. Its
apply path writes and seals the global and attempt intents before invoking the exact
receipt-bound submit argv, then polls complete `squeue` plus `sacct` truth for up to
900 seconds and publishes `PILOT_SUBMISSION_ACCEPTED.json` only after exactly one
matching allocation is visible. Save the job ID from its JSON output; never parse raw
`sbatch` text or submit the script manually.

Replaying the same apply command returns `already_submitted` without issuing another
submission. If the process stops across the scheduler boundary, replay adopts the
one exact receipt/comment/SubmitLine-matching allocation and reports `adopted`.
Multiple or conflicting candidates, receipt or identity drift, successful `sbatch`
output without complete scheduler visibility, and any unresolved attempt fail closed
without a second submission.

After the allocation appears terminal, explicitly wait until its exact job ID is
absent from `squeue`. Then poll `sacct -X` for at most 900 seconds for exactly one
nonempty top-level `JobIDRaw` row. An empty result is the only retryable state. More
than one row, a different `JobIDRaw`, or any observed terminal result other than
exact `COMPLETED|0:0` fails immediately; reaching the accounting deadline without a
row also fails. Only after both checks succeed may `accept-scheduler` run:

```bash
schema5_pilot_squeue="$(
  squeue -h -j "$schema5_pilot_job_id" -o "%i"
)" || exit 1
while test -n "$schema5_pilot_squeue"; do
  sleep 30
  schema5_pilot_squeue="$(
    squeue -h -j "$schema5_pilot_job_id" -o "%i"
  )" || exit 1
done
schema5_pilot_sacct_deadline=$((SECONDS + 900))
while :; do
  schema5_pilot_sacct="$(
    sacct -X -n -P -j "$schema5_pilot_job_id" \
      --format=JobIDRaw,State,ExitCode
  )" || exit 1
  if (( SECONDS > schema5_pilot_sacct_deadline )); then
    echo "timed out waiting for top-level pilot sacct accounting" >&2
    exit 1
  fi
  schema5_pilot_sacct_rows=()
  while IFS= read -r schema5_pilot_sacct_row; do
    test -z "$schema5_pilot_sacct_row" ||
      schema5_pilot_sacct_rows+=("$schema5_pilot_sacct_row")
  done <<<"$schema5_pilot_sacct"

  case "${#schema5_pilot_sacct_rows[@]}" in
    0)
      schema5_pilot_sacct_remaining=$((schema5_pilot_sacct_deadline - SECONDS))
      if (( schema5_pilot_sacct_remaining <= 0 )); then
        echo "timed out waiting for top-level pilot sacct accounting" >&2
        exit 1
      fi
      if (( schema5_pilot_sacct_remaining < 10 )); then
        sleep "$schema5_pilot_sacct_remaining"
      else
        sleep 10
      fi
      ;;
    1)
      schema5_pilot_terminal="${schema5_pilot_sacct_rows[0]}"
      if test "$schema5_pilot_terminal" != \
        "$schema5_pilot_job_id|COMPLETED|0:0"; then
        echo "unexpected top-level pilot terminal accounting: $schema5_pilot_terminal" >&2
        exit 1
      fi
      break
      ;;
    *)
      echo "expected exactly one top-level pilot sacct row, got ${#schema5_pilot_sacct_rows[@]}" >&2
      exit 1
      ;;
  esac
done

"$schema5_dev_python" -I "$schema5_pilot_script" accept-scheduler \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id"
"$schema5_dev_python" -I "$schema5_pilot_script" accept-scheduler \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id" --apply
"$schema5_dev_python" -I "$schema5_pilot_script" accept-scheduler \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id" --apply

"$schema5_dev_python" -I "$schema5_pilot_script" verify \
  --pilot-root "$schema5_pilot"
schema5_sealed_python="$schema5_pilot/materialization/harness-environment/bin/python"
env LD_LIBRARY_PATH="$schema5_pilot/materialization/harness-environment/lib" \
  "$schema5_sealed_python" -I "$schema5_pilot_script" verify \
  --pilot-root "$schema5_pilot"
```

The first acceptance command is non-mutating. The two apply calls must report
`accepted` and `already_accepted`, respectively. `PILOT_COMPLETE.json` alone is not
a production prerequisite: `PILOT_SCHEDULER_ACCEPTED.json` is published last only
after exact terminal `COMPLETED|0:0`, `Requeue=0`, job identity, submit command,
receipt, raw scheduler-query evidence, and scheduler-spooled script evidence all join
successfully. Both verifier
invocations must reject the pilot before that acceptance marker exists.

If the allocation ends after durable stage markers, the exact same immutable sbatch
may adopt those completed stages. If an interrupted Conda clone left an unmarked
partial prefix, a direct rerun fails closed. Same-tag retry is allowed only when
accounting proves `BOOT_FAIL`, `NODE_FAIL`, `PREEMPTED`, `REVOKED`, or an explicit
`CANCELLED by UID` external cancellation. `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, an
unattributed cancellation, or any contract error requires a superseding release.
For an allowed transient, first require
`"$schema5_pilot/PILOT_SUBMISSION_ACCEPTED.json"`. If the submitter stopped after
Slurm accepted the job but before publishing that marker, replay the same
`"${schema5_pilot_submit_cmd[@]}" --apply` command so it adopts the exact terminal
job and publishes acceptance without resubmitting. Only then preserve the entire
canonical pilot root:

```bash
"$schema5_dev_python" -I "$schema5_pilot_script" quarantine \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id"
"$schema5_dev_python" -I "$schema5_pilot_script" quarantine \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id" --apply
"$schema5_dev_python" -I "$schema5_pilot_script" quarantine \
  --pilot-root "$schema5_pilot" \
  --sbatch-receipt "$schema5_pilot_receipt" \
  --job-id "$schema5_pilot_job_id" --apply
```

The tool joins the exact read-only sbatch receipt to complete `squeue`/`sacct`
truth, binds state, exit code, reason, job name, and SubmitLine, writes intent first,
performs a same-filesystem inode-preserving rename, inventories and recursively seals
the complete partial tree, and publishes completion last. The repeated apply must
report `already_quarantined_and_sealed`. Bind that exact marker-last seal into the
next transactional submission:

```bash
schema5_pilot_quarantine_seal="$(
  dirname "$schema5_pilot"
)/quarantine_evidence/partial-job-${schema5_pilot_job_id}.sealed.json"
test -r "$schema5_pilot_quarantine_seal"
schema5_pilot_submit_cmd+=(
  --prior-quarantine-seal "$schema5_pilot_quarantine_seal"
)
```

Only then may the same immutable sbatch recreate the canonical pilot root. Every
subsequent transient retry appends its own explicit seal while retaining all earlier
`--prior-quarantine-seal` arguments. Never discover these markers with a glob or omit
an earlier one on replay. The submitter recursively reverifies every sealed tree and
its completion/intent lineage, binds the sorted unique retired job IDs and hashes
into the new marker-first submission intent, and ignores only those exact terminal
accounting rows. A writable, tampered, wrong-root, wrong-receipt, non-transient, or
unsealed historical job remains fatal; an unrelated matching job is never hidden.
A quarantine created without transactionally accepted submission evidence remains a
valid archive, but its seal is deliberately ineligible to suppress any scheduler
candidate.

## Materialize

First finish and test the source, create the exact tag
`sweep-recovery-schema5-v1.2-r9`, and make the fresh
`release_source_checkout_v1_2_r9` checkout containing only that tag's tracked files.
The operational retry tag is distinct from the production artifact ID
`sweep-recovery-schema5-v1.2`. The materializer rejects a source checkout whose `HEAD`,
status, or source-tree hash differs from the tag. Do not reuse the cancelled v1.1
checkout or use the development checkout, whose supplementary analysis artifacts are
intentionally retained. Then choose empty destination paths.

First capture the environments. The following is a dry run because it omits
`--apply`:

```bash
schema5_recovery=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/recovery/schema5-v1
schema5_release_base="$schema5_recovery/releases/sweep-recovery-schema5-v1.2"
schema5_capture="$schema5_recovery/environment_captures/sweep-recovery-schema5-v1.2"
schema5_conda_toolchain_root="$schema5_recovery/toolchains/r9/conda"
schema5_source_package_cache=/orcd/home/002/mabdel03/.conda/pkgs
schema5_checkout="$schema5_recovery/materialization_pilot_source_checkout_v1_2_r9"
schema5_pilot="$schema5_recovery/materialization_pilots/schema5-v1.2-r9"
schema5_pilot_python="$schema5_pilot/materialization/harness-environment/bin/python"

python scripts/capture_schema5_environments.py capture \
  --output-root "$schema5_capture" \
  --harness-source /orcd/home/002/mabdel03/conda_envs/asys_env \
  --serving-source /orcd/home/002/mabdel03/conda_envs/serve_env \
  --ownership-policy configs/environment_ownership_policy.v1.json \
  --integrity-normalization-policy configs/environment_integrity_normalization_policy.v1.json \
  --reconciliation-incident "$schema5_recovery/post_snapshot_incidents/2026-07-23_conda_pip_interop_source_metadata_reconciliation.json" \
  --recovered-setuptools-record "$schema5_recovery/releases/quarantine/sweep-recovery-schema5-v1.1.partial-job-18555913/environments/harness/conda-meta/setuptools-82.0.1-pyh332efcf_0.json"
```

Repeat the exact command with `--apply`, then verify it with
`capture_schema5_environments.py verify --output-root "$schema5_capture"`. Next
extract the exact cache-input contract from the already verified schema-5 pilot.
Do not recompute or hand-author this JSON:

```bash
test -f "$schema5_pilot/PILOT_COMPLETE.json"
test ! -L "$schema5_pilot/PILOT_COMPLETE.json"
schema5_expected_cache_input="$(
  "$schema5_pilot_python" -I -c \
    'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(json.dumps(p["package_cache_seed_input"], sort_keys=True, separators=(",", ":")))' \
    "$schema5_pilot/PILOT_COMPLETE.json"
)"

env LD_LIBRARY_PATH="$schema5_pilot/materialization/harness-environment/lib" \
"$schema5_pilot_python" -I \
"$schema5_checkout/scripts/materialize_schema5_release.py" materialize \
  --output-root "$schema5_release_base" \
  --environment-capture-root "$schema5_capture" \
  --source-repository "$schema5_checkout" \
  --release-worktree "$schema5_release_base/worktree" \
  --source-harness-prefix "$schema5_capture/seeds/harness" \
  --source-serving-prefix "$schema5_capture/seeds/serving" \
  --source-package-cache "$schema5_source_package_cache" \
  --harness-prefix "$schema5_release_base/environments/harness" \
  --serving-prefix "$schema5_release_base/environments/serving" \
  --expected-package-cache-seed-input-json "$schema5_expected_cache_input" \
  --conda-toolchain-root "$schema5_conda_toolchain_root"
```

Review the resolved paths and tag commit, then repeat the exact command with `--apply`.
Because the two physical copies are large, run the apply command inside a durable,
`--no-requeue` batch allocation with enough wall time. Do not run two materializers
against the same destination. Materialization schema 5 first verifies the sealed
release-local toolchain and then copies the exact required package artifacts and
cache metadata from `--source-package-cache` into its checksummed local seed. It
never executes the shared Miniforge base or invokes Conda against either live
developer prefix.

Verify the completed materialization independently with the same tagged tool:

```bash
"$schema5_pilot_python" -I \
"$schema5_checkout/scripts/materialize_schema5_release.py" verify \
  --output-root "$schema5_release_base"
```

## Freeze and seal

Use the materialized paths and the contracts inside the materialized worktree. Dry-run
the freezer first:

```bash
"$schema5_release_base/environments/harness/bin/python" \
  "$schema5_release_base/worktree/scripts/freeze_schema5_release.py" create \
  --output-root "$schema5_release_base/identity" \
  --release-worktree "$schema5_release_base/worktree" \
  --harness-prefix "$schema5_release_base/environments/harness" \
  --serving-prefix "$schema5_release_base/environments/serving" \
  --model-contract "$schema5_release_base/worktree/configs/model_contracts.v1.json" \
  --fleet-contract "$schema5_release_base/worktree/configs/schema5_fleet.v1.json"
```

Repeat with `--apply --seal-worktree --seal-environments --seal-output-root` only after
the dry run succeeds. The freezer does not invoke or require a Conda executable. Then
verify:

```bash
"$schema5_release_base/environments/harness/bin/python" \
  "$schema5_release_base/worktree/scripts/freeze_schema5_release.py" verify \
  --output-root "$schema5_release_base/identity"
```

The freezer refuses arbitrary worktrees or prefixes: it fully verifies the sibling
`MATERIALIZATION_COMPLETE.json`, binds its materialization ID and all stage hashes into
the release identity, then rechecks the live destinations before marker publication.
Later verification uses that sealed proof and the complete environment inventories; it
does not depend on mutable development prefixes, Git repository metadata, or the
external Conda executable remaining available. If a process stops after publishing
`RELEASE_COMPLETE.json` but before the final output-directory chmod, repeating the
same `create --apply --seal-output-root` command finishes that seal idempotently.
Requests to retrofit worktree or environment sealing onto an already-complete release
fail closed because their stored inventories and seal declarations are immutable.

`RELEASE_COMPLETE.json` is the sole release-success marker. A materialization marker
without this freezer marker is not production-ready.

## Lock interpretation

The environment manifest contains two complementary locks:

- `conda_explicit` pins each remote Conda package URL and SHA-256.
- `pip_freeze_all` records the canonical Python distribution view.

The freezer always constructs the explicit lock directly from every
`conda-meta/*.json` record. The materializer records the clone-time Conda executable's
absolute path and SHA-256 as creation provenance, but freezer creation and all future
verification neither execute nor require that external tool. Environment manifests
also bind the seed, ownership-policy, integrity-normalization-policy,
normalization-receipt, installed-file, Conda-artifact, package-cache, source-tree,
model/tokenizer, and fleet hashes.

Some Conda-built packages (currently `packaging` and `pip`) expose builder-local
`file://` provenance through `pip freeze`. The freezer accepts such a row only if the
matching Conda record owns and hashes its `METADATA` and `direct_url.json`; it then
normalizes the row to exact `name==version`. The explicit Conda lock remains the source
of reproduction truth. Editable requirements, VCS URLs, unpinned requirements, remote
direct references, unowned local references, and stale Conda ownership records all
fail closed. The only local direct reference allowed is the non-editable
`agents_scaling` install bound to the exact release worktree.
