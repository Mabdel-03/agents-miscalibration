# Schema-5 v1.2-r9 recovery and production runbook

This is the authoritative operator entry point for the clean schema-5 production
rerun. The scientific run IDs and manifests remain unchanged; the operational
release is `sweep-recovery-schema5-v1.2-r9`, the logical release is
`sweep-recovery-schema5-v1.2`, and the chain namespace is `schema5-v1.2-r9`.

The v1.1-r1 chain is sealed forensic evidence. Never submit, repair, or reuse its
`g0001` proposal, partial release, checkout, jobs, logs, intents, or repair namespace.
Use `verify_schema5_recovery_evidence.py` to inspect r1 without executing its code.

Run every local shell block below, in order, in the same dedicated Bash process.
The first block enables fail-fast and unset-variable handling for that process;
do not paste a later mutation or submission block into an uninitialized shell.

## Fixed identities

```bash
set -euo pipefail

# Establish the command trust boundary before the first executable or command
# substitution. Keep this prelude in the same Bash process as every later block.
unset BASH_ENV ENV CDPATH LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT
unset GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_INDEX_FILE
unset GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_EXEC_PATH
unset GIT_TEMPLATE_DIR GIT_NAMESPACE GIT_SSH GIT_SSH_COMMAND
unset GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS
unset GIT_CEILING_DIRECTORIES GIT_DISCOVERY_ACROSS_FILESYSTEM
unset GIT_REPLACE_REF_BASE GIT_SHALLOW_FILE
unset SLURM_CONF SLURM_CLUSTERS SLURM_TIME_FORMAT
while IFS= read -r ambient_name; do
  case "$ambient_name" in
    BASH_FUNC_*|GIT_CONFIG_KEY_*|GIT_CONFIG_VALUE_*|SBATCH_*|SACCT_*|SCONTROL_*|SQUEUE_*)
      unset "$ambient_name"
      ;;
  esac
done < <(compgen -e)
export PATH=/usr/bin:/bin
readonly PATH
export LC_ALL=C LANG=C
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_ATTR_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1
export GIT_TERMINAL_PROMPT=0

repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
recovery="$results/recovery/schema5-v1"
state="$results/.dispatcher-schema5-v1"
pool="$results/server_pools/schema5-v1"
tag=sweep-recovery-schema5-v1.2-r9
release_id=sweep-recovery-schema5-v1.2
slurm_user=mabdel03
chain_manifest="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R9.json"
pilot_checkout="$recovery/materialization_pilot_source_checkout_v1_2_r9"
pilot_root="$recovery/materialization_pilots/schema5-v1.2-r9"
canary_root="$recovery/slurm_canaries/schema5-v1.2-r9"
dev_python="$(realpath -e /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python)"
conda_toolchain_namespace="$recovery/toolchains/r9"
conda_toolchain_root="$conda_toolchain_namespace/conda"
source_package_cache=/orcd/home/002/mabdel03/.conda/pkgs
sealed_python="$pilot_root/materialization/harness-environment/bin/python"
durable_remote=origin
durable_commit_ref=refs/heads/schema5-v1.2-r9
durable_marker="$recovery/DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R9_COMPLETE.json"
r2_canary_tree="$recovery/slurm_canaries/schema5-v1.2-r2"
r2_canary_failure_root="$recovery/canary_failures/schema5-v1.2-r2"
r2_canary_failure_marker="$r2_canary_failure_root/CANARY_FAILURE_SEALED.json"
r2_release_checkout="$recovery/materialization_pilot_source_checkout_v1_2_r2"
r2_zero_mutation="$recovery/r1_acceptance/ZERO_RESULT_MUTATION_RECEIPT.json"
r2_snapshot_marker="$recovery/pre_repair/SNAPSHOT_COMPLETE.json"
r3_release_checkout="$recovery/materialization_pilot_source_checkout_v1_2_r3"
r3_durable_marker="$recovery/DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R3_COMPLETE.json"
r3_prelaunch_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r3"
r3_prelaunch_failure_marker="$r3_prelaunch_failure_root/PRELAUNCH_FAILURE_SEALED.json"
r4_toolchain_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r4-toolchain"
r4_toolchain_failure_marker="$r4_toolchain_failure_root/TOOLCHAIN_FAILURE_SEALED.json"
r5_prelaunch_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r5-cli"
r5_prelaunch_failure_marker="$r5_prelaunch_failure_root/PRELAUNCH_CLI_FAILURE_SEALED.json"
r6_prelaunch_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r6-toolchain-binding"
r6_prelaunch_failure_marker="$r6_prelaunch_failure_root/PRELAUNCH_TOOLCHAIN_BINDING_FAILURE_SEALED.json"
r7_prelaunch_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r7-git-index-refresh"
r7_prelaunch_failure_marker="$r7_prelaunch_failure_root/PRELAUNCH_GIT_INDEX_REFRESH_FAILURE_SEALED.json"
r8_prelaunch_failure_root="$recovery/prelaunch_failures/schema5-v1.2-r8-r3-probe-import"
r8_prelaunch_failure_marker="$r8_prelaunch_failure_root/PRELAUNCH_R3_PROBE_IMPORT_FAILURE_SEALED.json"
```

The three authoritative run IDs contain exactly 4,680, 14,400, and 3,600 cells,
respectively: 22,680 cells and 4,524,660 expected QIDs in total. Production remains
paused until every gate below passes.

## Immutable r2–r8 to r9 lineage

The annotated r2 tag, remote `refs/heads/schema5-v1.2-r2` branch, durable Git bundle
and checksum, and `DURABLE_GIT_RELEASE_COMPLETE.json` remain immutable historical
evidence. Do not move, overwrite, delete, or republish any of them under an r9 name.
The r2 live composite canary failed deterministically while capturing scheduler
output. Its disposition is `requires_superseding_release`; it is not eligible for an
r2 suffix repair.

Seal the original partial tree in place because its artifacts contain absolute
paths. The sealer recursively removes write bits from that tree and publishes the
separate marker-last evidence at exactly
`$recovery/canary_failures/schema5-v1.2-r2/CANARY_FAILURE_SEALED.json`. The marker
must use protocol
`schema5-v1.2-r2-partial-canary-failure-seal-v1`, report
`classification=deterministic_canary_failure_sealed_fail_closed`, and set
`retry_in_place=false`. Never rename the partial tree to a quarantine directory.

Run the exact marker-first sealer once without `--apply`, once with `--apply`, and
then repeat the identical apply invocation to prove idempotency. The two
`--mutation-evidence` inputs are the already sealed zero-result-mutation receipt and
the independently verified pre-repair snapshot marker; absence is never treated as
proof that scientific results were unchanged.

```bash
r2_sealer="$repo/scripts/seal_recovery_evidence.py"
r2_error_summary='Immutable r2 composite canary raised AttributeError in _dependency_scheduler_snapshot before dependency sbatch because the production default runner invoked subprocess.run without capture_output/text, leaving stdout=None; transaction canary job 18889366 was cancelled before start.'
r2_canary_seal_cmd=(
  "$dev_python" -I "$r2_sealer" seal-canary-failure
  --tree "$r2_canary_tree"
  --evidence-root "$r2_canary_failure_root"
  --release-checkout "$r2_release_checkout"
  --scheduler-user "$slurm_user"
  --error-classification deterministic_missing_subprocess_capture
  --error-summary "$r2_error_summary"
  --mutation-evidence "$r2_zero_mutation"
  --mutation-evidence "$r2_snapshot_marker"
)
"${r2_canary_seal_cmd[@]}"
"${r2_canary_seal_cmd[@]}" --apply
"${r2_canary_seal_cmd[@]}" --apply

test -f "$r2_canary_failure_marker" && test ! -L "$r2_canary_failure_marker"
test ! -w "$r2_canary_failure_marker"
jq -e '
  .passed == true and
  .classification == "deterministic_canary_failure_sealed_fail_closed" and
  .error_classification == "deterministic_missing_subprocess_capture" and
  .retry_in_place == false and
  .no_active_matching_jobs_preseal == true and
  .no_active_matching_jobs_postseal == true and
  .known_scheduler_identity.job_ids == ["18889366"]
' "$r2_canary_failure_marker"
```

The dry-run changes nothing. The first apply invocation may only seal the original
r2 partial canary and publish its separate evidence tree; it neither submits nor
cancels a job and does not mutate experiment results. Repeated apply must report
`already_sealed` after rechecking complete scheduler truth and must not rewrite the
marker, either inventory, or either scheduler-evidence record.

The r9 renderer must verify and bind the exact seal path, raw SHA-256, size, and
`seal_id` into its immutable prerequisite evidence before rendering, and reverify
that binding at submission. A missing, writable, symlinked, malformed, or mismatched
seal fails closed. All r9 pilot, canary, source-checkout, job/log, protected-capacity,
repair, and sentinel paths are fresh r9 paths; no r2 completion marker is copied
forward.

Only immutable prerequisite evidence produced by the historical r2 flow retains
a `schema5-v1.2-r2-*` protocol. Every fresh protected-capacity, watchdog, pilot,
client-capacity, and other active wire payload uses a `schema5-v1.2-r9-*`
protocol, an r9-specific artifact path, and binds the r9 tag and
`chain_namespace=schema5-v1.2-r9` wherever that chain namespace is part of the
payload schema. Operational artifact basenames are r9-specific; r2 names appear
only in the explicit immutable historical lineage and failure-seal inputs above.

The annotated r3 tag and durable bundle are also immutable history. Release r3 is
commit `acd723ba9a99d88e77f7d752268bc31205c3a808`, annotated tag object
`fa87b283974afc1fa48fcb22b61e6ae7eec4bb36`, tag
`sweep-recovery-schema5-v1.2-r3`, and namespace `schema5-v1.2-r3`. Its real
prelaunch probes exposed two deterministic materialization defects before any
scheduler job was submitted: the recorded broken internal compiler-tool symlink in
the shared Miniforge runtime and an unseeded release-local offline package cache.
The shared runtime must remain untouched and must never be executed by r9.

The r9-tagged sealer archives the two canonical probe envelopes marker-first under
`$recovery/prelaunch_failures/schema5-v1.2-r3`, binds the existing explicit
zero-result-mutation receipt, and publishes `$r3_prelaunch_failure_marker` last. The
exact producer, sealing, and `verify-prelaunch-failure` commands appear after the
tagged r9 checkout and toolchain are created below; there is intentionally no
executable verification block here because the seal does not exist yet. The
independently callable verifier ultimately consumes only that sealed root.

It rechecks the exact r3 release identities, both required classifications, compact
canonical envelope and inventory hashes, marker/intent self-hashes, recursive
read-only state, `retry_in_place=false`,
`requires_superseding_release=true`, the explicit zero-result-mutation basis, and
empty scheduler-job lists. The renderer persists the complete portable binding
returned under protocol
`schema5-v1.2-r3-prelaunch-failure-seal-binding-v1`; a substituted, incomplete,
writable, or internally rehashed-but-wrong release seal fails closed.

The immutable r4 release is commit
`f23cf1c4b2d2bf606afd013123b9afe9c614bb6c`. Its production toolchain root made
the required absolute `base/bin/python` interpreter path 154 bytes and its complete
shebang 157 bytes. Miniforge therefore installed `bin/conda` with the fallback
`#!/usr/bin/env python`, which the r4 runtime-identity contract correctly rejected
before any pilot, canary, recovery-chain job, schema-5 run root, server pool, or
controller state existed. Replaying r4 would deterministically reinstall the same
invalid entrypoint, so r4 is not repairable in place.

After the r9 tag and detached pilot checkout exist, the tagged r9 failure sealer
inventories both r4 attempts, the already quarantined g0001 prefix, the unmarked g0002
prefix, raw symlink text, internal hardlink topology, immutable r4 source identity,
empty r4 scheduler namespace, and absence of every schema-5 scientific output. It
then removes write bits from the r4 toolchain and transaction roots and publishes
`$r4_toolchain_failure_marker` last. The r9 renderer and every sentinel bind and
reverify this third historical failure seal.

The immutable r5 release is commit
`fa50d619579d6ffde9ab768d342bb65f1b98d844`, annotated tag object
`51267bb1bbe59833bfadc96b4d3ad74355ce1d39`, tag
`sweep-recovery-schema5-v1.2-r5`, and namespace `schema5-v1.2-r5`. Its isolated
short-prefix Conda toolchain completed and verifies read-only. The first exact
`record-prelaunch-attempt` CLI invocation nevertheless failed before either r3
probe ran because argparse used `command` for both the subparser selector and the
remainder argv. The probe argv replaced the selector, dispatch fell through to the
verification branch, and the tagged program raised
`AttributeError: 'Namespace' object has no attribute 'evidence_root'`. No failure
envelope, scheduler job, r3 seal, schema-5 run root, pool, or controller state was
created.

After the r9 tag exists, the tagged r9 sealer reproduces that exact r5 CLI invocation
once inside a marker-first transaction, requires exit 1 and both exact traceback
signatures, proves the declared `/tmp` probe tree is byte-for-byte unchanged, binds
the clean r5 checkout, durable marker, and independently verified read-only r5
toolchain, and publishes `$r5_prelaunch_failure_marker` last. The r9 renderer and
all sentinels reverify both the r4 and r5 historical seals. The marker protocol is
`schema5-v1.2-r5-prelaunch-cli-dispatch-failure-seal-v1`, and the canonical marker is
`prelaunch_failures/schema5-v1.2-r5-cli/PRELAUNCH_CLI_FAILURE_SEALED.json`.

The immutable r6 release is commit
`9b40414a0d90f569cf021ac152503746bedda6c7`, annotated tag object
`e6b9c2f2548f7ef58ad5163b65a25b164013a4e6`, tag
`sweep-recovery-schema5-v1.2-r6`, and namespace `schema5-v1.2-r6`. Its fresh
short-prefix Conda toolchain completed and independently verifies read-only. The
first r3 recorder dry run stopped before executing either probe because
`verified_conda_toolchain_binding` correctly returned `portable_shebang`, while the
r6 recorder's exact accepted-field set omitted that field. The only set difference
was that one additive field; no field was missing. The recorder exited 2 with
`sealed r6 Conda toolchain verification identity drifted` before publishing an
envelope, submitting a scheduler job, or mutating scientific state.

After the r9 tag exists, the tagged r9 sealer reproduces that exact r6 dry-run
invocation once inside a marker-first transaction. It binds the clean r6 checkout,
durable marker, valid r6 toolchain, exact expected and observed field sets, empty
r6 probe tree, scheduler quiescence, and zero-result state, then publishes
`$r6_prelaunch_failure_marker` last. Its protocol is
`schema5-v1.2-r6-prelaunch-toolchain-binding-failure-seal-v1`; the canonical marker
is
`prelaunch_failures/schema5-v1.2-r6-toolchain-binding/PRELAUNCH_TOOLCHAIN_BINDING_FAILURE_SEALED.json`.
The r9 recorder accepts and validates the complete returned binding, including the
portable-shebang contract. The r9 renderer and every sentinel reverify the r4, r5,
and r6 historical seals.

The immutable r7 release is commit
`acc0beb98cfbb5017d5d5b5e60567aa43229e248`, annotated tag object
`95f8f3852498c4ab8e3026f29c183ecdd58678f4`, tag
`sweep-recovery-schema5-v1.2-r7`, and namespace `schema5-v1.2-r7`. Its durable
bundle and fresh short-prefix Conda toolchain completed and verify independently.
The first r3 recorder dry run then stopped before executing either probe because
its immutable-input inventory included `.git/index` metadata and the recorder's own
read-only Git identity query allowed Git's optional index refresh. In a freshly
cloned checkout, `git status --porcelain=v1 --untracked-files=all` replaced
`.git/index`: content, size, and device stayed equal, but the inode changed. The
input inventory therefore correctly detected a mutation and exited 2 with
`r3 prelaunch immutable-input verification mutated a probe input`. No failure
envelope, scheduler job, r3 seal, schema-5 run root, pool, controller state, or
scientific result was created.

After the r9 tag exists, the tagged r9 sealer clones the exact durable r7 bundle,
reproduces that index replacement once with the exact r7 Git query sequence, and
requires unchanged index bytes/size/device with a changed inode and no other
checkout mutation. It then proves that the same sequence with
`GIT_OPTIONAL_LOCKS=0` preserves the fresh index inode, binds the clean r7 checkout,
durable marker, valid r7 toolchain, empty r7 probe tree, scheduler quiescence, and
zero-result state, and publishes `$r7_prelaunch_failure_marker` last. Its protocol is
`schema5-v1.2-r7-prelaunch-git-index-refresh-failure-seal-v1`; the canonical marker
is
`prelaunch_failures/schema5-v1.2-r7-git-index-refresh/PRELAUNCH_GIT_INDEX_REFRESH_FAILURE_SEALED.json`.
The r9 recorder sets `GIT_OPTIONAL_LOCKS=0` only for its read-only Git identity
queries. The r9 renderer and every sentinel independently bind and reverify the r7
seal alongside the earlier historical seals.

The immutable r8 release is commit
`ed18bb969c6632d2b2643024064a2b4d1b63f6de`, annotated tag object
`c31692215b98c176bdd138109b0c2b6ec0a63e1f`, tag
`sweep-recovery-schema5-v1.2-r8`, and namespace `schema5-v1.2-r8`. Its durable
bundle and isolated short-prefix Conda toolchain completed and independently verify
read-only. The first exact r3 broken-link recorder apply then stopped before
publishing an envelope: the r8 probe launched the full r3 materialization-pilot CLI
under the intentionally minimal r8 toolchain, so Python imported the pilot's project
dependency graph and failed on the undeclared `backoff` dependency before reaching
the stdlib-only runtime-identity subcommand. The recorder rejected that traceback
because it was not the canonical broken-symlink signature. The r8 probe root remained
empty; no envelope, scheduler job, r3 seal, schema-5 run root, pool, controller state,
or scientific result was created.

After the r9 tag exists, the tagged r9 sealer reproduces the exact outer r8 recorder
failure and exact inner r3 import traceback once inside a marker-first transaction.
It binds the clean r8 checkout, durable marker and bundle, independently verified r8
toolchain, empty r8 probe tree, scheduler quiescence, and zero-result state, then
publishes `$r8_prelaunch_failure_marker` last. Its protocol is
`schema5-v1.2-r8-prelaunch-r3-probe-import-failure-seal-v1`; the canonical marker is
`prelaunch_failures/schema5-v1.2-r8-r3-probe-import/PRELAUNCH_R3_PROBE_IMPORT_FAILURE_SEALED.json`.
The r9 recorder launches the pinned r3 stdlib-only
`schema5_conda_runtime_identity.py` directly, avoiding the unrelated pilot import
graph while preserving the exact historical runtime-identity contract. The r9
renderer and every sentinel bind and reverify all seven historical failure seals.

## Before-tag gates

Require all of the following before tagging:

- the r1 quarantine seal, r1 failure envelope, and compact r1 evidence snapshot
  verify;
- the exact-a5cd930 `r1_acceptance/QUARANTINE_IDEMPOTENCY_RECEIPT.json` and
  `ZERO_RESULT_MUTATION_RECEIPT.json` are read-only and verify;
- the original 201,528-file pre-repair snapshot and external attestation verify;
- the deterministic r2 canary failure seal above verifies as read-only,
  scheduler-quiescent evidence requiring a superseding release;
- the immutable r4 toolchain-failure seal verifies from its marker and preserves
  the 157-byte required-shebang failure without permitting an r4 retry;
- the r5 failure sealer dry run proves the exact CLI-dispatch classification,
  zero scheduler/scientific mutation, and the independently valid r5 toolchain;
- the r6 failure sealer dry run proves the one-field toolchain-binding mismatch,
  zero scheduler/scientific mutation, and the independently valid r6 toolchain;
- the r7 failure sealer dry run verifies the durable source/toolchain inputs, empty
  scheduler/scientific namespaces, and the exact fresh-clone index-refresh proof
  contract without performing the reproduction;
- the r8 failure sealer dry run verifies the durable source/toolchain inputs, empty
  scheduler/scientific namespaces, and exact recorder/import-failure proof contract
  without performing the reproduction;
- no legacy/schema-5 worker or controller jobs and no held cell locks;
- a clean full test suite, checksum checks, and Git diff check;
- read-only source-prefix distribution audits pass under the independently
  checksummed ownership and integrity-normalization policies;
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
The r9 production prerequisite and environment-capture job accept only Conda
reconciliation incident SHA-256
`9f588ce4ffc4aeb5a3ac494a35244604e4eb100a7190b18d9eb3c568468fdb09`
with incident ID
`2abfc4fab5828cd1e965ff822e54e55462b1a6a2b280b7f47eebd90cadcfd872`;
both harness and serving stale-record flags must be exactly `false`.
[SCHEMA5_RELEASE.md](SCHEMA5_RELEASE.md) records the exact hashes.

## Freeze and test the r9 source

Create an annotated tag only from the reviewed, clean commit:

```bash
git -C "$repo" status --short
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$repo" for-each-ref --format='%(refname)' refs/replace)"
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
the distinct `release_source_checkout_v1_2_r9` path and requires that path to be
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

Seal the deterministic r4 toolchain failure from the exact tagged r9 checkout before
creating the replacement toolchain:

```bash
r4_failure_sealer="$pilot_checkout/scripts/seal_schema5_r4_toolchain_failure.py"
r4_failure_seal=(
  "$dev_python" -I "$r4_failure_sealer" seal
  --evidence-root "$r4_toolchain_failure_root"
  --recovery-root "$recovery"
  --scheduler-user "$slurm_user"
)
"${r4_failure_seal[@]}"
"${r4_failure_seal[@]}" --apply
"${r4_failure_seal[@]}" --apply
"$dev_python" -I "$r4_failure_sealer" verify \
  --evidence-root "$r4_toolchain_failure_root" \
  --recovery-root "$recovery"
test -f "$r4_toolchain_failure_marker"
test ! -L "$r4_toolchain_failure_marker"
test ! -w "$r4_toolchain_failure_marker"
```

Seal the deterministic r5 prelaunch CLI failure from the exact tagged r9 checkout.
The dry run does not execute the failed command; the first apply invocation records
the single exact reproduction, and the repeated apply is verification-only:

```bash
r5_failure_sealer="$pilot_checkout/scripts/seal_schema5_r5_prelaunch_failure.py"
r5_failure_seal=(
  "$dev_python" -I "$r5_failure_sealer" seal
  --evidence-root "$r5_prelaunch_failure_root"
  --recovery-root "$recovery"
  --scheduler-user "$slurm_user"
)
"${r5_failure_seal[@]}"
"${r5_failure_seal[@]}" --apply
"${r5_failure_seal[@]}" --apply
"$dev_python" -I "$r5_failure_sealer" verify \
  --evidence-root "$r5_prelaunch_failure_root" \
  --recovery-root "$recovery"
test -f "$r5_prelaunch_failure_marker"
test ! -L "$r5_prelaunch_failure_marker"
test ! -w "$r5_prelaunch_failure_marker"
```

Seal the deterministic r6 prelaunch toolchain-binding failure from the exact tagged
r9 checkout. The dry run validates the one-field mismatch without executing the
historical recorder; the first apply performs the single exact failed dry run, and
the repeated apply is verification-only:

```bash
r6_failure_sealer="$pilot_checkout/scripts/seal_schema5_r6_prelaunch_failure.py"
r6_failure_seal=(
  "$dev_python" -I "$r6_failure_sealer" seal
  --evidence-root "$r6_prelaunch_failure_root"
  --recovery-root "$recovery"
  --scheduler-user "$slurm_user"
)
"${r6_failure_seal[@]}"
"${r6_failure_seal[@]}" --apply
"${r6_failure_seal[@]}" --apply
"$dev_python" -I "$r6_failure_sealer" verify \
  --evidence-root "$r6_prelaunch_failure_root" \
  --recovery-root "$recovery"
test -f "$r6_prelaunch_failure_marker"
test ! -L "$r6_prelaunch_failure_marker"
test ! -w "$r6_prelaunch_failure_marker"
```

Seal the deterministic r7 prelaunch Git-index-refresh failure from the exact tagged
r9 checkout. The dry run validates all historical inputs without cloning or running
Git. The first apply clones the immutable r7 bundle and performs the exact failed and
corrected query sequences; the repeated apply is verification-only:

```bash
r7_failure_sealer="$pilot_checkout/scripts/seal_schema5_r7_prelaunch_failure.py"
r7_failure_seal=(
  "$dev_python" -I "$r7_failure_sealer" seal
  --evidence-root "$r7_prelaunch_failure_root"
  --recovery-root "$recovery"
  --scheduler-user "$slurm_user"
)
"${r7_failure_seal[@]}"
"${r7_failure_seal[@]}" --apply
"${r7_failure_seal[@]}" --apply
"$dev_python" -I "$r7_failure_sealer" verify \
  --evidence-root "$r7_prelaunch_failure_root" \
  --recovery-root "$recovery"
test -f "$r7_prelaunch_failure_marker"
test ! -L "$r7_prelaunch_failure_marker"
test ! -w "$r7_prelaunch_failure_marker"
```

Seal the deterministic r8 prelaunch r3-probe import failure from the exact tagged
r9 checkout. The dry run validates all historical inputs without executing either
failed command. The first apply performs the exact outer recorder and inner import
reproductions; the repeated apply is verification-only:

```bash
r8_failure_sealer="$pilot_checkout/scripts/seal_schema5_r8_prelaunch_failure.py"
r8_failure_seal=(
  "$dev_python" -I "$r8_failure_sealer" seal
  --evidence-root "$r8_prelaunch_failure_root"
  --recovery-root "$recovery"
  --scheduler-user "$slurm_user"
)
"${r8_failure_seal[@]}"
"${r8_failure_seal[@]}" --apply
"${r8_failure_seal[@]}" --apply
"$dev_python" -I "$r8_failure_sealer" verify \
  --evidence-root "$r8_prelaunch_failure_root" \
  --recovery-root "$recovery"
test -f "$r8_prelaunch_failure_marker"
test ! -L "$r8_prelaunch_failure_marker"
test ! -w "$r8_prelaunch_failure_marker"
```

Provision and verify the r9-only Conda toolchain from the exact tagged checkout.
This operation uses the pinned cached installer, never the shared Miniforge
executable and never Conda from either developer prefix:

```bash
conda_installer=/orcd/data/lhtsai/001/om2/mabdel03/Miniforge3-Linux-x86_64.sh
conda_provisioner="$pilot_checkout/scripts/provision_schema5_conda_toolchain.py"
if [[ ! -e "$conda_toolchain_namespace" ]]; then
  install -d -m 0755 -- "$conda_toolchain_namespace"
fi
test -d "$conda_toolchain_namespace"
test ! -L "$conda_toolchain_namespace"
"$dev_python" -I "$conda_provisioner" provision \
  --installer "$conda_installer" \
  --namespace-root "$conda_toolchain_namespace"
"$dev_python" -I "$conda_provisioner" provision \
  --installer "$conda_installer" \
  --namespace-root "$conda_toolchain_namespace" \
  --apply
"$dev_python" -I "$conda_provisioner" verify \
  --toolchain-root "$conda_toolchain_root"
```

The provision marker is marker-last, read-only, and bound to the pinned installer
SHA-256. Verification needs neither the installer nor another Conda installation.
Follow [SCHEMA5_R9_CONDA_TOOLCHAIN.md](SCHEMA5_R9_CONDA_TOOLCHAIN.md) for the complete
link, inode, offline-probe, and crash-quarantine contract.

Produce the two r3 prelaunch-failure envelopes only after the r9 toolchain above has
sealed. These are narrow, read-only reproductions of the two defects; they are not
arbitrary commands labeled after the fact. Both commands use one private temporary
root. The recorder rejects a different interpreter, script, argument order, input
root, environment key, output path, exit code, or diagnostic signature.
The tagged sealer itself runs under the already resolved harness Python and imports
the tagged r9 provisioner in-process. That provisioner performs the complete
marker/runtime/inventory binding without executing the sealed toolchain; only after
that independent check may the exact probe command launch the sealed r9 Python or
Conda executable.

The broken-link probe executes the exact r3 stdlib-only
`schema5_conda_runtime_identity.py` at commit
`acd723ba9a99d88e77f7d752268bc31205c3a808` (file SHA-256
`b3a66668b09e2aa5ba67c00878a713ced94b94cb272106b217b4ce84d2c2c64d`) directly
with the sealed r9 Python, `-I`, and `-B`. It reads, but never executes, the recorded
shared Conda base. Direct invocation is required: importing the full r3 materialization
pilot would import unrelated project dependencies before reaching this stdlib-only
probe, which is the deterministic r8 failure sealed above. The offline probe executes
only the sealed r9 Conda and clones that same sealed base into the declared temporary
destination. Its package cache must exist and be completely empty before the first
apply.

```bash
r9_probe_python="$conda_toolchain_root/base/bin/python"
r9_probe_conda="$conda_toolchain_root/base/bin/conda"
r9_sealer="$pilot_checkout/scripts/seal_recovery_evidence.py"
r3_runtime_identity="$r3_release_checkout/scripts/schema5_conda_runtime_identity.py"
recorded_shared_conda_base=/orcd/data/lhtsai/001/om2/mabdel03/miniforge3
recorded_shared_conda="$recorded_shared_conda_base/bin/conda"
r3_probe_root="/tmp/schema5-r3-prelaunch-${slurm_user}-r9"
r3_broken_envelope="$r3_probe_root/FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
r3_offline_envelope="$r3_probe_root/FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json"
r3_offline_cache="$r3_probe_root/empty-conda-pkgs"
r3_offline_destination="$r3_probe_root/offline-clone-destination"

if [[ ! -e "$r3_probe_root" ]]; then
  install -d -m 0700 -- "$r3_probe_root"
fi
test -d "$r3_probe_root" && test ! -L "$r3_probe_root"
test "$(stat -c %u "$r3_probe_root")" = "$(id -u)"
test "$(( 8#$(stat -c %a "$r3_probe_root") & 8#022 ))" -eq 0
if [[ ! -e "$r3_offline_cache" ]]; then
  install -d -m 0700 -- "$r3_offline_cache"
fi
test -d "$r3_offline_cache" && test ! -L "$r3_offline_cache"

r3_probe_common_environment=(
  --environment HOME "$r3_probe_root/home"
  --environment PYTHONDONTWRITEBYTECODE 1
  --environment PYTHONNOUSERSITE 1
  --environment TEMP "$r3_probe_root/tmp"
  --environment TMP "$r3_probe_root/tmp"
  --environment TMPDIR "$r3_probe_root/tmp"
  --environment XDG_CACHE_HOME "$r3_probe_root/xdg-cache"
  --environment XDG_CONFIG_HOME "$r3_probe_root/xdg-config"
  --environment XDG_DATA_HOME "$r3_probe_root/xdg-data"
  --environment XDG_STATE_HOME "$r3_probe_root/xdg-state"
)
r3_broken_probe=(
  "$dev_python" -I -B "$r9_sealer" record-prelaunch-attempt
  --output "$r3_broken_envelope"
  --classification unsafe_recorded_broken_internal_symlink
  --cwd "$r3_release_checkout"
  "${r3_probe_common_environment[@]}"
  --input-root tagged-r3-release-checkout "$r3_release_checkout"
  --input-root tagged-r9-release-checkout "$pilot_checkout"
  --input-root sealed-r9-conda-toolchain "$conda_toolchain_root"
  --input-root recorded-shared-conda-base "$recorded_shared_conda_base"
  --write-root "$r3_probe_root"
)
r3_broken_command=(
  "$r9_probe_python" -I -B "$r3_runtime_identity"
  --conda-executable "$recorded_shared_conda"
)
"${r3_broken_probe[@]}" --command "${r3_broken_command[@]}"
"${r3_broken_probe[@]}" --apply --command "${r3_broken_command[@]}"
"${r3_broken_probe[@]}" --apply --command "${r3_broken_command[@]}"

if [[ ! -f "$r3_offline_envelope" ]]; then
  test -z "$(find "$r3_offline_cache" -mindepth 1 -print -quit)"
  test ! -e "$r3_offline_destination" && test ! -L "$r3_offline_destination"
fi
r3_offline_probe=(
  "$dev_python" -I -B "$r9_sealer" record-prelaunch-attempt
  --output "$r3_offline_envelope"
  --classification offline_clone_unseeded_release_local_cache
  --cwd "$r3_release_checkout"
  "${r3_probe_common_environment[@]}"
  --environment CONDA_ENVS_PATH "$r3_probe_root/conda-envs"
  --environment CONDA_NO_PLUGINS true
  --environment CONDA_OFFLINE true
  --environment CONDA_PIP_INTEROP_ENABLED false
  --environment CONDA_PKGS_DIRS "$r3_offline_cache"
  --input-root tagged-r3-release-checkout "$r3_release_checkout"
  --input-root tagged-r9-release-checkout "$pilot_checkout"
  --input-root sealed-r9-conda-toolchain "$conda_toolchain_root"
  --write-root "$r3_probe_root"
)
r3_offline_command=(
  "$r9_probe_conda" create --yes --offline
  --clone "$conda_toolchain_root/base"
  --prefix "$r3_offline_destination"
)
"${r3_offline_probe[@]}" --command "${r3_offline_command[@]}"
"${r3_offline_probe[@]}" --apply --command "${r3_offline_command[@]}"
"${r3_offline_probe[@]}" --apply --command "${r3_offline_command[@]}"

r3_prelaunch_seal=(
  "$dev_python" -I -B "$r9_sealer" seal-prelaunch-failure
  --evidence-root "$r3_prelaunch_failure_root"
  --durable-release-marker "$r3_durable_marker"
  --release-checkout "$r3_release_checkout"
  --failure-envelope "$r3_broken_envelope"
  --failure-envelope "$r3_offline_envelope"
  --mutation-evidence "$r2_zero_mutation"
  --mutation-evidence "$r2_snapshot_marker"
)
"${r3_prelaunch_seal[@]}"
"${r3_prelaunch_seal[@]}" --apply
"${r3_prelaunch_seal[@]}" --apply
```

The first invocation of each array is a dry-run, the first `--apply` records or seals,
and the repeated `--apply` must adopt the identical immutable artifact without
rerunning either failure. The recorder inventories the tagged r3 checkout, tagged r9
checkout, sealed r9 toolchain, and—only for the runtime-identity probe—the recorded
shared base before and after execution. The exact command shapes bind every mutable
HOME, XDG, Conda cache/environment, temporary, destination, and envelope path below
`$r3_probe_root`; the sealer therefore does not infer zero mutation from an
operator-authored Boolean. The shared-base inventory uses the exact historical
runtime-identity boundary, excluding only top-level `.conda`, `conda-bld`, `envs`,
and `pkgs`; the known broken `libexec` link and every runtime path the probe can read
remain inside the before/after inventory.

Independently verify the resulting immutable r3 prelaunch-failure seal using only the
sealed root and the exact tagged r9 verifier. This is a hard render prerequisite:

```bash
"$dev_python" -I -B "$r9_sealer" \
  verify-prelaunch-failure \
  --evidence-root "$r3_prelaunch_failure_root"
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
  --partition mit_normal --qos normal --time-limit 00:08:00 --scheduler-user "$slurm_user" \
  --turnover-cycles 2 --turnover-drain-seconds 5 \
  --dependency-alert-latency-seconds 180 \
  --durable-git-release-marker "$durable_marker"

"$dev_python" -I "$pilot_checkout/scripts/run_schema5_slurm_fleet_canary.py" \
  --canary-root "$canary_root" \
  --partition mit_normal --qos normal --time-limit 00:08:00 --scheduler-user "$slurm_user" \
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
11.5-hour, `--no-requeue` pilot job from the exact tagged checkout. Pilot schema 5
drives materialization schema 5; neither schema accepts the retired unseeded-cache
or arbitrary shared-Conda interface. The pilot must
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
The schema-5 pilot verifier report exposes both independent policy hashes, the
pilot's exact
harness and serving
`live_source_inventory_sha256` values, the sealed Conda-toolchain binding, the
checksummed source-package-cache seed, the verified entrypoint's canonical path,
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
pilot_sbatch="$recovery/jobs/schema5-v1.2-r9-materialization-pilot.sbatch"
pilot_receipt="$pilot_sbatch.receipt.json"
pilot_logs="$recovery/logs/materialization-pilot-r9"
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
  --conda-toolchain-root "$conda_toolchain_root"
  --source-package-cache "$source_package_cache"
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
protected_canary_root="$recovery/protected_capacity/schema5-v1.2-r9"
protected_builder="$pilot_checkout/scripts/build_schema5_protected_capacity_evidence.py"
protected_publisher="$pilot_checkout/scripts/publish_schema5_protected_capacity.py"
effective_fleet_tool="$pilot_checkout/scripts/materialize_schema5_effective_fleet.py"
effective_fleet_root="$recovery/effective-fleet-schema5-v1.2-r9"
qualification_runner="$pilot_checkout/scripts/run_schema5_throughput_qualification.py"
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

# Publish the generation-one effective-fleet transaction from the tagged
# 22-replica/24-GPU base with a zero additive delta. Its bytes and SHA-256 must
# equal the base contract even though it has its own canonical transaction path.
# The first command is a non-mutating dry run.
effective_fleet_args=(
  --release-worktree "$pilot_checkout"
  --release-git-commit "$release_git_commit"
  --release-tag-object "$release_tag_object"
  --base-fleet-contract "$fleet_contract"
  --base-fleet-contract-sha256 "$fleet_contract_sha256"
  --model-contract "$model_contract"
  --model-contract-sha256 "$model_contract_sha256"
  --output-root "$effective_fleet_root"
)
"$sealed_python" -I "$effective_fleet_tool" materialize \
  "${effective_fleet_args[@]}"
"$sealed_python" -I "$effective_fleet_tool" materialize \
  "${effective_fleet_args[@]}" --apply

# Verification emits ten fixed-order NUL-delimited values. This avoids eval, jq,
# and any hand-authored fleet transformation.
mapfile -d '' -t effective_fleet_inputs < <(
  "$sealed_python" -I "$effective_fleet_tool" verify \
    "${effective_fleet_args[@]}" --format runbook-nul
)
test "${#effective_fleet_inputs[@]}" -eq 10
effective_fleet_contract="${effective_fleet_inputs[0]}"
effective_fleet_contract_sha256="${effective_fleet_inputs[1]}"
additive_overlay_contract="${effective_fleet_inputs[2]}"
additive_overlay_contract_sha256="${effective_fleet_inputs[3]}"
source_tree_sha256="${effective_fleet_inputs[4]}"
dispatcher_source="${effective_fleet_inputs[5]}"
dispatcher_source_sha256="${effective_fleet_inputs[6]}"
qualification_runner_source="${effective_fleet_inputs[7]}"
qualification_runner_source_sha256="${effective_fleet_inputs[8]}"
effective_fleet_marker_id="${effective_fleet_inputs[9]}"
test "$effective_fleet_contract" = \
  "$effective_fleet_root/schema5_fleet.effective.v1.json"
test "$additive_overlay_contract" = "$effective_fleet_contract"
test "$additive_overlay_contract_sha256" = \
  "$effective_fleet_contract_sha256"
test "$effective_fleet_contract_sha256" = "$fleet_contract_sha256"
test "${#effective_fleet_marker_id}" -eq 64

# Seal the exact generation-one base-fleet admission wave before any scientific
# QID is drawn. The truthful base result is 278/384 selected cells, a 106-cell
# shortfall, and 12 microbatches (11x24 plus 14). That shortfall is expected to
# drive the qualification-only controlled capacity transition; it is not
# permission to preinstall additive replicas. The certificate is marker-last and
# the first invocation remains non-mutating.
static_capacity_certificate="$recovery/readiness/PREFLIGHT_CAPACITY_CERTIFICATE.json"
static_capacity_args=(
  --base-fleet-contract "$fleet_contract"
  --effective-fleet-contract "$effective_fleet_contract"
  --additive-overlay-contract "$additive_overlay_contract"
  --capacity-generation 1
  --release-git-commit "$release_git_commit"
  --source-tree-sha256 "$source_tree_sha256"
  --dispatcher-source "$dispatcher_source"
  --qualification-runner-source "$qualification_runner_source"
  --output "$static_capacity_certificate"
)
"$sealed_python" -I "$qualification_runner" preflight-capacity \
  "${static_capacity_args[@]}"
"$sealed_python" -I "$qualification_runner" preflight-capacity \
  "${static_capacity_args[@]}" --apply
test -f "$static_capacity_certificate"
test ! -L "$static_capacity_certificate"
test ! -w "$static_capacity_certificate"
static_capacity_certificate_sha256="$(
  sha256sum -- "$static_capacity_certificate" | awk '{print $1}'
)"
static_capacity_certificate_id="$(
  "$sealed_python" -I -c \
    'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8")); print(v["certificate_id"])' \
    "$static_capacity_certificate"
)"
test "${#static_capacity_certificate_sha256}" -eq 64
test "${#static_capacity_certificate_id}" -eq 64

# Non-mutating plan. Review exactly 448 canary job elements: 384 clients,
# 22 active servers, three warm-turnover allocations, and 39 held reserve elements.
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
  --effective-fleet-contract "$effective_fleet_contract" \
  --effective-fleet-contract-sha256 "$effective_fleet_contract_sha256" \
  --additive-overlay-contract "$additive_overlay_contract" \
  --additive-overlay-contract-sha256 "$additive_overlay_contract_sha256" \
  --static-feasibility-certificate "$static_capacity_certificate" \
  --static-feasibility-certificate-sha256 \
    "$static_capacity_certificate_sha256" \
  --static-feasibility-certificate-id "$static_capacity_certificate_id" \
  --capacity-generation 1 \
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
  --effective-fleet-contract "$effective_fleet_contract" \
  --effective-fleet-contract-sha256 "$effective_fleet_contract_sha256" \
  --additive-overlay-contract "$additive_overlay_contract" \
  --additive-overlay-contract-sha256 "$additive_overlay_contract_sha256" \
  --static-feasibility-certificate "$static_capacity_certificate" \
  --static-feasibility-certificate-sha256 \
    "$static_capacity_certificate_sha256" \
  --static-feasibility-certificate-id "$static_capacity_certificate_id" \
  --capacity-generation 1 \
  --model-contract "$model_contract" \
  --model-contract-sha256 "$model_contract_sha256" \
  --release-worktree "$pilot_checkout" \
  --apply

"$sealed_python" -I "$protected_publisher" verify \
  --recovery-root "$recovery" \
  --release-git-commit "$release_git_commit" \
  --release-tag-object "$release_tag_object" \
  --source-tree-sha256 "$source_tree_sha256" \
  --dispatcher-source-sha256 "$dispatcher_source_sha256" \
  --qualification-runner-source-sha256 \
    "$qualification_runner_source_sha256"
test -f "$protected_capacity" && test ! -L "$protected_capacity"
test ! -w "$protected_capacity"
unset capacity_token
```

The `--apply` invocation is the only command above that submits jobs. It resumes its
marker-first transaction after interruption and publishes the completion marker
last; never choose a new token for a retry.

The preceding generation-one effective-fleet publication uses protocol
`schema5-v1.2-r9-effective-fleet-materialization-v1`. Its create-once intent binds
the exact annotated tag, full source tree, model/base contracts, publisher,
dispatcher, and qualification-runner bytes. It deterministically applies a zero
delta to every serving profile before qualification.
`schema5_fleet.effective.v1.json` and its exact checksum are read-only, and
`EFFECTIVE_FLEET_COMPLETE.json` is published last only after both the runtime fleet
loader and additive-prefix validator prove base = effective = 22/24 and zero
additive replicas/GPUs. Never construct
the effective contract with `jq`, copy a test fixture, or edit its JSON.

`PROTECTED_CAPACITY_COMPLETE.json` is a schema-4, regular, non-symlink, read-only
JSON object bound to the exact release ID, annotated tag, commit, tag object, and
`schema5-v1.2-r9` namespace. It uses a semantic canonical-JSON `marker_id`; the
chain additionally binds its raw SHA-256 and size. Never hand-author, copy forward,
or rehash this marker.

`PROTECTED_CAPACITY_COMPLETE.json` uses protocol
`schema5-v1.2-r9-protected-capacity-v4` and capacity source
`sealed_protected_canary+partition_inventory+association`. It must prove at least
the frozen 22-replica/24-GPU base fleet, a zero-additive generation-one effective
fleet with identical bytes/SHA-256, and four separately retained warm-headroom GPUs
across three warm jobs. It must also prove a protected 384-cell ceiling plus a 64-job
reserve, submit headroom of at least 448 jobs, at least 384 CPUs and 1,572,864 MiB
for clients, and complete `squeue` plus `sacct` reconciliation. Both scientific
server and scientific client placement must attest `PreemptMode=OFF`. The marker
binds the exact scheduler cluster, user, account, QOS association, association
`MaxSubmitJobs`, client partition CPU/memory/GPU inventory, each server
partition's exact aggregate CPU/memory/GPU TRES and node count, and frozen
source-tree/dispatcher/qualification-runner hashes. Blank inherited QOS resource
limits are never interpreted as infinity: CPU and memory authority comes from the
sealed simultaneous canary, while submit authority comes from the exact association.
A transport-censor protocol does not turn a preemptible scientific allocation into
protected capacity.

Immediately before every production server `sbatch`, the live server partition
must still be `UP` and its aggregate CPU, memory, GPU, and node inventory must
exactly equal the sealed placement inventory; the aggregate GPU inventory must
also cover all 24 active GPUs plus the separately retained four-GPU warm envelope,
for exactly 28 attested GPUs before qualification.
This is a live drift gate, not a claim that transiently idle GPUs are reserved.

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
protocol `schema5-v1.2-r9-external-watchdog-v1`, additionally binds a
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
watchdog_vm_stage=/var/tmp/schema5-watchdog-v1.2-r9
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

## Post-tag and pre-render gates

After publishing the annotated r9 tag, exact remote branch, and durable Git marker,
but before rendering the 43-job chain, require all of the following:

- the clean detached r9 pilot checkout resolves to the exact annotated tag object and
  commit and has no object alternates;
- the release-local Conda toolchain verifies from its marker-last sealed root, and
  the source package cache passes the schema-5 copy-plan audit without invoking
  Conda;
- the immutable r3 prelaunch-failure seal independently verifies both deterministic
  failures, zero scheduler jobs, and explicit zero-result-mutation evidence;
- the immutable r4 toolchain-failure seal independently verifies both attempts, the
  157-byte required shebang, fallback entrypoint, sealed trees, zero scheduler jobs,
  and zero schema-5 result mutations;
- the immutable r5 and r6 prelaunch seals independently verify their exact failed
  invocations, valid release-local toolchains, empty scheduler namespaces, and zero
  schema-5 result mutations;
- the immutable r7 prelaunch seal independently verifies its fresh-clone index
  replacement, corrected `GIT_OPTIONAL_LOCKS=0` control, valid release-local
  toolchain, empty scheduler namespace, and zero schema-5 result mutations;
- the immutable r8 prelaunch seal independently verifies its exact outer-recorder
  rejection and inner missing-dependency traceback, valid release-local toolchain,
  empty scheduler/probe namespaces, and zero schema-5 result mutations;
- the genuine r9 composite Slurm canary verifies at its fresh canonical root;
- both the materialization pilot completion and scheduler-acceptance markers verify
  under the sealed pilot harness;
- `PROTECTED_CAPACITY_COMPLETE.json` proves protected non-preemptible scientific
  server/client capacity for the full 384-cell ceiling plus 64-job reserve;
- the bootstrap-watchdog VM login, verified single-host cluster known-hosts file,
  and reserved production-watchdog public key are available for the held-root
  bootstrap transaction; production external-watchdog deployment and liveness
  evidence are created only after `schema5_initialize` publishes the exact paused
  `control.json`, and are consumed by stage 19; and
- the r1 evidence, adopted snapshot, and exact r2 canary failure seal still verify,
  maintenance and scheduler quiescence still hold, and complete `squeue` plus `sacct`
  truth is available and unambiguous.

The canary, pilot, and protected-capacity evidence are deliberately post-tag and
pre-render because they bind the annotated r9 tag and commit. Bootstrap-watchdog
operator inputs are needed before root release but are not render prerequisites.
Production watchdog evidence necessarily binds stage-10 paused control, so it is
produced after initialization and verified by stage 19.
`WATCHDOG_READY.json` remains a stage-19 output.

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
  --conda-toolchain-root "$conda_toolchain_root" \
  --source-package-cache "$source_package_cache" \
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
  --conda-toolchain-root "$conda_toolchain_root" \
  --source-package-cache "$source_package_cache" \
  --materialization-pilot-root "$pilot_root" \
  --slurm-canary-root "$canary_root" \
  --partition mit_normal --slurm-user "$slurm_user" \
  --apply

"$sealed_python" -I "$renderer" verify --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" submit --chain-manifest "$chain_manifest"
submit_result="$("$sealed_python" -I "$renderer" submit \
  --chain-manifest "$chain_manifest" --apply)"
jq -e '.status == "awaiting_bootstrap_watchdog" and
  .launch_marker_complete == false' <<<"$submit_result"

chain_receipt="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R9_SUBMISSION.json"
generation_provenance="$recovery/BOOTSTRAP_GENERATION_PROVENANCE.json"
root_release="$recovery/RECOVERY_CHAIN_ROOT_RELEASE_COMPLETE.json"
launch_complete="$recovery/RECOVERY_CHAIN_SCHEMA5_V1_2_R9_LAUNCHED.json"
jq -e --arg manifest "$chain_manifest" \
  '.passed == true and .no_requeue == true and .manifest == $manifest and
   .root_initial_hold == true and
   .dependency_canary.kill_invalid_depend == true and
   .dependency_canary.alert_latency_bound_seconds == 180 and
   (.jobs | length) == 43 and
   ([.jobs[].job_id] | length == (unique | length))' \
  "$chain_receipt"
test -s "$generation_provenance"
test ! -w "$generation_provenance"

# Keep the exact canonical root held here. The bootstrap VM login, cluster
# host-key input, and separately reserved production-watchdog public key are
# deliberate operator inputs; abort rather than guessing any of them.
: "${BOOTSTRAP_VM_LOGIN:?set to deployer@institutional-vm}"
: "${BOOTSTRAP_CLUSTER_HOST:?set to the cluster SSH host}"
: "${BOOTSTRAP_CLUSTER_KNOWN_HOSTS:?set to a verified single-host file}"
: "${PRODUCTION_WATCHDOG_PUBLIC_KEY:?set to the reserved stage-19 public key}"
test -f "$BOOTSTRAP_CLUSTER_KNOWN_HOSTS"
test -f "$PRODUCTION_WATCHDOG_PUBLIC_KEY"

bootstrap_root="$recovery/readiness/bootstrap_watchdog"
bootstrap_bundle="$bootstrap_root/bundle"
bootstrap_deployment_evidence="$bootstrap_root/DEPLOYMENT_EVIDENCE.json"
bootstrap_drill_evidence="$bootstrap_root/CANCELLATION_DRILL_EVIDENCE.json"
bootstrap_attestation="$bootstrap_root/BOOTSTRAP_ATTESTATION.json"
bootstrap_observation_1="$bootstrap_root/canonical-observation-1.json"
bootstrap_observation_2="$bootstrap_root/canonical-observation-2.json"
bootstrap_authorized_keys_snapshot="$bootstrap_root/authorized_keys.snapshot"
bootstrap_public_key="$bootstrap_root/bootstrap_vm_ed25519.pub"
watchdog_public_key="$(realpath -e "$PRODUCTION_WATCHDOG_PUBLIC_KEY")"
bootstrap_vm_stage=/var/tmp/schema5-bootstrap-watchdog-v1.2-r9
bootstrap_vm_python=/opt/agents-scaling-bootstrap-watchdog/venv/bin/python
bootstrap_vm_release=/opt/agents-scaling-bootstrap-watchdog/release
bootstrap_vm_config=/etc/agents-scaling-bootstrap-watchdog/watchdog.json
bootstrap_vm_identity=/etc/agents-scaling-bootstrap-watchdog/id_ed25519
bootstrap_vm_known_hosts=/etc/agents-scaling-bootstrap-watchdog/known_hosts
bootstrap_vm_state=/var/lib/agents-scaling-bootstrap-watchdog
bootstrap_service=agents-scaling-schema5-bootstrap-watchdog.service
bootstrap_timer=agents-scaling-schema5-bootstrap-watchdog.timer
pilot_marker="$pilot_root/PILOT_COMPLETE.json"
sealed_release_worktree="$(jq -er '.layout.release_worktree' "$pilot_marker")"
sealed_python_alias="$(jq -er '.layout.harness_prefix' "$pilot_marker")/bin/python"
sealed_release_bundle="$(jq -er '.layout.release_bundle' "$pilot_marker")"
sealed_harness_manifest="$sealed_release_bundle/harness_environment.schema5-v1.json"
watchdog_tool="$sealed_release_worktree/scripts/build_schema5_watchdog_deployment.py"
mkdir -p -- "$bootstrap_root"
test "$(realpath -e "$sealed_release_worktree")" = "$sealed_release_worktree"
test -x "$sealed_python_alias"
test -f "$watchdog_tool" && test ! -w "$watchdog_tool"

# Render and submit a separately self-hashed, harmless 43-job sibling. Its scripts
# can only print a diagnostic and exit 78 if accidentally released. Its root remains
# held throughout the drill, and its job IDs/comments must be disjoint from canonical.
isolated_root="$recovery/isolated_cancellation_drill"
isolated_manifest="$isolated_root/RECOVERY_CHAIN_SCHEMA5_V1_2_R9.json"
isolated_receipt="$isolated_root/RECOVERY_CHAIN_SCHEMA5_V1_2_R9_SUBMISSION.json"
isolated_repair_result="$isolated_root/BOOTSTRAP_REPAIR_RESULT.json"
"$sealed_python" -I "$renderer" prepare-isolated-bootstrap-drill \
  --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" prepare-isolated-bootstrap-drill \
  --chain-manifest "$chain_manifest" --apply
"$sealed_python" -I "$renderer" verify \
  --chain-manifest "$isolated_manifest"
"$sealed_python" -I "$renderer" submit \
  --chain-manifest "$isolated_manifest"
isolated_submit="$("$sealed_python" -I "$renderer" submit \
  --chain-manifest "$isolated_manifest" --apply)"
jq -e '.status == "awaiting_bootstrap_watchdog" and
  .root_initial_hold == true and .no_requeue == true and
  (.jobs | length) == 43' <<<"$isolated_submit"
jq -e --slurpfile canonical "$chain_receipt" '
  ([.jobs[].job_id] | unique) as $isolated_ids |
  ([$canonical[0].jobs[].job_id] | unique) as $canonical_ids |
  ([.jobs[].comment] | unique) as $isolated_comments |
  ([$canonical[0].jobs[].comment] | unique) as $canonical_comments |
  (($isolated_ids - $canonical_ids) | length) == ($isolated_ids | length) and
  (($isolated_comments - $canonical_comments) | length) ==
    ($isolated_comments | length)
' "$isolated_receipt"
test ! -e "$recovery/recovery_chain_repairs_v1_2_r9"
test ! -e "$root_release"
test ! -e "$launch_complete"

# Generate the bootstrap VM private key only on that VM. Install the emitted
# restricted line exactly once in the cluster account and snapshot the complete
# authorized_keys file so duplicate unrestricted aliases are detectable.
ssh "$BOOTSTRAP_VM_LOGIN" "\
  sudo install -d -o root -g root -m 0755 \
    /etc/agents-scaling-bootstrap-watchdog; \
  test -f $bootstrap_vm_identity || \
    sudo ssh-keygen -q -t ed25519 -N '' -f $bootstrap_vm_identity; \
  sudo chmod 0600 $bootstrap_vm_identity; \
  sudo cat $bootstrap_vm_identity.pub" > "$bootstrap_public_key"
chmod 0444 "$bootstrap_public_key"

"$sealed_python" -I "$watchdog_tool" bootstrap-bundle \
  --output-root "$bootstrap_bundle" \
  --release-root "$sealed_release_worktree" \
  --harness-python "$sealed_python_alias" \
  --harness-environment-manifest "$sealed_harness_manifest" \
  --materialization-pilot-marker "$pilot_marker" \
  --chain-manifest "$chain_manifest" \
  --submission-receipt "$chain_receipt" \
  --isolated-drill-chain-manifest "$isolated_manifest" \
  --isolated-drill-submission-receipt "$isolated_receipt" \
  --git-commit "$release_git_commit" --tag-object "$release_tag_object" \
  --remote-host "$BOOTSTRAP_CLUSTER_HOST" --remote-user "$slurm_user" \
  --identity-file "$bootstrap_vm_identity" \
  --known-hosts-file "$bootstrap_vm_known_hosts" \
  --external-state-root "$bootstrap_vm_state" \
  --bootstrap-public-key-file "$bootstrap_public_key" \
  --production-public-key-file "$watchdog_public_key"

install -d -m 0700 -- "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 0600 "$HOME/.ssh/authorized_keys"
(
  flock -x 9
  forced_line="$(cat "$bootstrap_bundle/authorized_keys.bootstrap.line")"
  grep -Fqx -- "$forced_line" "$HOME/.ssh/authorized_keys" ||
    printf '%s\n' "$forced_line" >> "$HOME/.ssh/authorized_keys"
  test "$(grep -Fxc -- "$forced_line" "$HOME/.ssh/authorized_keys")" -eq 1
) 9>"$HOME/.ssh/.schema5-bootstrap-watchdog-authorized-keys.lock"
install -m 0444 -- "$HOME/.ssh/authorized_keys" \
  "$bootstrap_authorized_keys_snapshot"

# Transfer only the sealed bundle, verified host-key input, and cluster-side
# authorized_keys snapshot; then install the exact inventory and canonical units.
ssh "$BOOTSTRAP_VM_LOGIN" "install -d -m 0700 $bootstrap_vm_stage"
rsync -a -- "$bootstrap_bundle/" \
  "$BOOTSTRAP_VM_LOGIN:$bootstrap_vm_stage/"
scp -- "$BOOTSTRAP_CLUSTER_KNOWN_HOSTS" \
  "$BOOTSTRAP_VM_LOGIN:$bootstrap_vm_stage/known_hosts"
scp -- "$bootstrap_authorized_keys_snapshot" \
  "$BOOTSTRAP_VM_LOGIN:$bootstrap_vm_stage/authorized_keys.snapshot"
ssh "$BOOTSTRAP_VM_LOGIN" "\
  set -euo pipefail; \
  id agents-scaling-bootstrap-watchdog >/dev/null 2>&1 || \
    sudo useradd --system --home $bootstrap_vm_state \
      --shell /usr/sbin/nologin agents-scaling-bootstrap-watchdog; \
  sudo install -d -o agents-scaling-bootstrap-watchdog \
    -g agents-scaling-bootstrap-watchdog -m 0700 $bootstrap_vm_state; \
  sudo install -d -o root -g root -m 0755 \
    /opt/agents-scaling-bootstrap-watchdog/venv/bin \
    $bootstrap_vm_release/scripts \
    $bootstrap_vm_release/src/agents_scaling/serving; \
  vm_python_source=\$(readlink -e \$(command -v python3)); \
  sudo install -o root -g root -m 0555 \
    \$vm_python_source $bootstrap_vm_python; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/release/scripts/build_schema5_watchdog_deployment.py \
    $bootstrap_vm_release/scripts/build_schema5_watchdog_deployment.py; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/release/scripts/schema5_bootstrap_watchdog.py \
    $bootstrap_vm_release/scripts/schema5_bootstrap_watchdog.py; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/release/src/agents_scaling/serving/external_watchdog.py \
    $bootstrap_vm_release/src/agents_scaling/serving/external_watchdog.py; \
  sudo chmod 0555 $bootstrap_vm_release $bootstrap_vm_release/scripts \
    $bootstrap_vm_release/src $bootstrap_vm_release/src/agents_scaling \
    $bootstrap_vm_release/src/agents_scaling/serving; \
  sudo install -o agents-scaling-bootstrap-watchdog \
    -g agents-scaling-bootstrap-watchdog -m 0400 \
    $bootstrap_vm_stage/known_hosts $bootstrap_vm_known_hosts; \
  sudo chown agents-scaling-bootstrap-watchdog:agents-scaling-bootstrap-watchdog \
    $bootstrap_vm_identity $bootstrap_vm_identity.pub; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/bootstrap-watchdog.json $bootstrap_vm_config; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/$bootstrap_service \
    /etc/systemd/system/$bootstrap_service; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/$bootstrap_timer \
    /etc/systemd/system/$bootstrap_timer; \
  sudo install -d -o root -g root -m 0755 $bootstrap_vm_state/evidence; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_stage/authorized_keys.snapshot \
    $bootstrap_vm_state/evidence/authorized_keys.snapshot; \
  sudo systemctl daemon-reload; \
  sudo systemctl start $bootstrap_service; \
  sudo systemctl enable --now $bootstrap_timer; \
  sudo install -o root -g root -m 0444 \
    $bootstrap_vm_state/BOOTSTRAP_WATCHDOG_HEARTBEAT.json \
    $bootstrap_vm_state/evidence/heartbeat.json; \
  sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    $bootstrap_vm_python -I \
    $bootstrap_vm_release/scripts/build_schema5_watchdog_deployment.py \
    bootstrap-deployment-evidence \
    --bundle-manifest $bootstrap_vm_stage/BUNDLE.json \
    --installed-release-root $bootstrap_vm_release \
    --vm-python $bootstrap_vm_python \
    --installed-config $bootstrap_vm_config \
    --installed-service /etc/systemd/system/$bootstrap_service \
    --installed-timer /etc/systemd/system/$bootstrap_timer \
    --installed-authorized-keys \
      $bootstrap_vm_state/evidence/authorized_keys.snapshot \
    --service-heartbeat $bootstrap_vm_state/evidence/heartbeat.json \
    --output $bootstrap_vm_state/evidence/DEPLOYMENT_EVIDENCE.json"
scp -- \
  "$BOOTSTRAP_VM_LOGIN:$bootstrap_vm_state/evidence/DEPLOYMENT_EVIDENCE.json" \
  "$bootstrap_deployment_evidence"
chmod 0444 "$bootstrap_deployment_evidence"

# Select the two canonical held observations by the IDs emitted by the successful
# VM service heartbeat. No path glob is accepted unless it resolves to exactly one
# sealed artifact for that ID.
bootstrap_heartbeat_local="$bootstrap_root/heartbeat.json"
scp -- "$BOOTSTRAP_VM_LOGIN:$bootstrap_vm_state/evidence/heartbeat.json" \
  "$bootstrap_heartbeat_local"
chmod 0444 "$bootstrap_heartbeat_local"
canonical_observation_root="$recovery/bootstrap_watchdog_observations"
for pair in \
  "first_observation_id:$bootstrap_observation_1" \
  "second_observation_id:$bootstrap_observation_2"; do
  field="${pair%%:*}"
  destination="${pair#*:}"
  observation_id="$(jq -er --arg field "$field" '.[$field]' \
    "$bootstrap_heartbeat_local")"
  mapfile -t matches < <(find "$canonical_observation_root" -maxdepth 1 \
    -type f -name "observation-*-$observation_id.json" -print)
  test "${#matches[@]}" -eq 1
  install -m 0444 -- "${matches[0]}" "$destination"
done

# Cancel only the 43 isolated receipt IDs. The forced command is bound to both
# exact manifest hashes, so its drill selectors cannot observe or repair canonical.
mapfile -t isolated_job_ids < <(jq -er '.jobs[].job_id' "$isolated_receipt")
test "${#isolated_job_ids[@]}" -eq 43
scancel -- "${isolated_job_ids[@]}"
bootstrap_forced_selector() {
  local selector="$1"
  ssh "$BOOTSTRAP_VM_LOGIN" "\
    sudo -u agents-scaling-bootstrap-watchdog \
      ssh -o BatchMode=yes -o IdentitiesOnly=yes \
      -o UserKnownHostsFile=$bootstrap_vm_known_hosts \
      -o StrictHostKeyChecking=yes -i $bootstrap_vm_identity \
      $slurm_user@$BOOTSTRAP_CLUSTER_HOST '$selector'"
}

drill_status_1_stdout="$bootstrap_root/drill-status-1.stdout.json"
drill_status_1_tmp="$drill_status_1_stdout.tmp"
attempt=0
until bootstrap_forced_selector \
  "schema5-bootstrap-watchdog drill-status" > "$drill_status_1_tmp"; do
  attempt=$((attempt + 1))
  test "$attempt" -lt 30
  sleep 10
done
chmod 0444 "$drill_status_1_tmp"
mv -- "$drill_status_1_tmp" "$drill_status_1_stdout"
drill_observation_1="$(jq -er '.observation_artifact' \
  "$drill_status_1_stdout")"
case "$drill_observation_1" in "$isolated_root"/*) ;; *) exit 1 ;; esac
test -f "$drill_observation_1" && test ! -w "$drill_observation_1"

sleep 60
drill_status_2_stdout="$bootstrap_root/drill-status-2.stdout.json"
drill_status_2_tmp="$drill_status_2_stdout.tmp"
bootstrap_forced_selector "schema5-bootstrap-watchdog drill-status" \
  > "$drill_status_2_tmp"
chmod 0444 "$drill_status_2_tmp"
mv -- "$drill_status_2_tmp" "$drill_status_2_stdout"
drill_observation_2="$(jq -er '.observation_artifact' \
  "$drill_status_2_stdout")"
case "$drill_observation_2" in "$isolated_root"/*) ;; *) exit 1 ;; esac
test -f "$drill_observation_2" && test ! -w "$drill_observation_2"

drill_repair_stdout="$bootstrap_root/drill-repair.stdout.json"
drill_repair_tmp="$drill_repair_stdout.tmp"
bootstrap_forced_selector "schema5-bootstrap-watchdog drill-repair" \
  > "$drill_repair_tmp"
chmod 0444 "$drill_repair_tmp"
mv -- "$drill_repair_tmp" "$drill_repair_stdout"
jq -e '.status == "bootstrap_repaired_held" and
  .root_held == true and .root_released == false and
  .watchdog_scientific_jobs_submitted == 0' "$drill_repair_stdout"
test -f "$isolated_repair_result" && test ! -w "$isolated_repair_result"

recovered_status_stdout="$bootstrap_root/drill-recovered.stdout.json"
recovered_status_tmp="$recovered_status_stdout.tmp"
bootstrap_forced_selector "schema5-bootstrap-watchdog drill-status" \
  > "$recovered_status_tmp"
chmod 0444 "$recovered_status_tmp"
mv -- "$recovered_status_tmp" "$recovered_status_stdout"
drill_recovered_observation="$(jq -er '.observation_artifact' \
  "$recovered_status_stdout")"
case "$drill_recovered_observation" in "$isolated_root"/*) ;; *) exit 1 ;; esac
recovery_seconds="$(jq -nr \
  --slurpfile first "$drill_observation_1" \
  --slurpfile recovered "$drill_recovered_observation" \
  '$recovered[0].observed_at_timestamp -
   $first[0].observed_at_timestamp')"

"$sealed_python" -I "$watchdog_tool" bootstrap-drill-evidence \
  --bundle-manifest "$bootstrap_bundle/BUNDLE.json" \
  --deployment-evidence "$bootstrap_deployment_evidence" \
  --cancelled-observation "$drill_observation_1" \
  --cancelled-observation "$drill_observation_2" \
  --repair-result "$isolated_repair_result" \
  --recovered-observation "$drill_recovered_observation" \
  --recovery-seconds "$recovery_seconds" \
  --output "$bootstrap_drill_evidence"

# Publish the attestation marker last, then READY -> ARM_INTENT -> ARMED. The
# canonical root still has no release, launch, or repair marker at this boundary.
test ! -e "$recovery/recovery_chain_repairs_v1_2_r9"
test ! -e "$root_release"
test ! -e "$launch_complete"
"$sealed_python" -I "$watchdog_tool" bootstrap-attestation \
  --bundle-manifest "$bootstrap_bundle/BUNDLE.json" \
  --deployment-evidence "$bootstrap_deployment_evidence" \
  --scheduler-observation "$bootstrap_observation_1" \
  --scheduler-observation "$bootstrap_observation_2" \
  --cancellation-drill-evidence "$bootstrap_drill_evidence" \
  --output "$bootstrap_attestation"
"$sealed_python" -I \
  "$pilot_checkout/scripts/publish_schema5_watchdog_ready.py" \
  --bootstrap --recovery-root "$recovery" \
  --chain-manifest "$chain_manifest" \
  --submission-receipt "$chain_receipt" \
  --bootstrap-attestation "$bootstrap_attestation" \
  --git-commit "$release_git_commit" --tag-object "$release_tag_object"
"$sealed_python" -I \
  "$pilot_checkout/scripts/publish_schema5_watchdog_ready.py" \
  --bootstrap --recovery-root "$recovery" \
  --chain-manifest "$chain_manifest" \
  --submission-receipt "$chain_receipt" \
  --bootstrap-attestation "$bootstrap_attestation" \
  --git-commit "$release_git_commit" --tag-object "$release_tag_object" \
  --apply

"$sealed_python" -I "$renderer" release-root \
  --chain-manifest "$chain_manifest"
"$sealed_python" -I "$renderer" release-root \
  --chain-manifest "$chain_manifest" --apply
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
publishes all 43 jobs, the read-only receipt, and marker-last generation-zero
scheduler/spooled-script provenance; it does not release the root. The separate
bootstrap watchdog is then deployed and attested against the exact tag, commit,
pilot harness, pilot completion, manifest, receipt, job IDs, comments, SubmitLines,
and spooled scripts. Two complete scheduler cuts at least 60 seconds apart and the
isolated cancellation drill must be sealed before `READY`, `ARM_INTENT`, and
`ARMED` are published. Only the explicit `release-root --apply` transaction records
the marker-first exact-ID release intent and invokes `scontrol release`.
`RECOVERY_CHAIN_SCHEMA5_V1_2_R9_LAUNCHED.json` is published last. A crash after
release is reconciled by the exact job/comment instead of issuing a second release.
The same transaction holds the first resubmitted job of each repair generation until
that generation's complete repair receipt and inherited marker-last provenance exist.
Completed jobs that have aged out of Slurm controller memory are authenticated by
their sealed origin-generation provenance plus complete `sacct` truth; every active
or pending job still requires live `scontrol` and spooled-script verification.
Before the initial launch, a reconstructed generation remains held and requires a
separate re-attestation decision. After launch, the bootstrap watchdog may release
only a repair generation descending from the validated immutable initial
`LAUNCHED` marker. It has no direct scientific admission, production-control
mutation, or safety-hold-clearing authority. A repaired production stage always
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
launch-authorization gate before any stage mutation. The gate waits at most 1,200
seconds for the generation-local submission receipt, exact-root release completion,
and launch completion; it then revalidates their canonical identities and hashes,
the current scheduler comment/job ID, every immutable sbatch, the live dependency
policy receipts, the bounded dependency canary, and protected capacity. Stage 19
then validates the forced-command production watchdog against initialized control,
and publishes its completed drill/readiness markers. `production_resume` validates
stage-19 readiness and starts the production chains, but it does not transfer
bootstrap authority. This applies equally to the first resubmitted suffix job in a
repair generation. If publication crashes after
`scontrol release`, the root fails closed without entering its body. All 21 fail-fast
observers and the aggregate failure sentinel are intentionally not launch-gated so
they can record and alert on that boundary failure.

Each stage observer authenticates its sealed sentinel executable, exact generation
comment, own Slurm job ID, immutable manifest, and generation receipt. It reparses
complete `squeue` plus `sacct` truth and requires its target stage to be terminal and
its own exact `afterany:<target-job-id>` allocation to be running. It publishes
`STAGE_SCHEDULER_EVIDENCE.json`, persists bounded email-delivery attempts for every
non-success terminal state, and publishes `STAGE_SENTINEL_COMPLETE.json` last under
`recovery_chain_stage_sentinels/schema5-v1.2-r9/gNNNN/<stage>/`. A stage observation
explicitly has no repair authority. The aggregate sentinel remains the sole causal
classifier and suffix-repair authority; it also verifies that every stage observer
terminated successfully. Thus a failure in either parallel readiness branch is
recorded and alerted as soon as that branch terminates, without waiting for the
long-running email-acknowledgement branch or for aggregate classification.

On the all-success path, the aggregate `failure_sentinel` first completes that
classification and then publishes the bootstrap handoff while holding the shared
submission/repair lock. It requires the latest complete generation receipt with no
pending repair; the exact generation-local root-release and launch markers; its own
live running job ID and comment; terminal `COMPLETED` plus `ExitCode=0:0` truth for
the other 42 jobs; explicit successful `production_resume`; verified production
watchdog readiness; running, healthy control with no drain, alert, or safety hold;
and both exact controller chains live and fresh. It publishes
`RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_HANDOFF_COMPLETE.json` last under protocol
`schema5-v1.2-r9-bootstrap-watchdog-handoff-v1`, binding the generation, receipt,
root release, launch, generation-zero arm, watchdog/control identity, aggregate
sentinel, production resume, and authority-transfer flags. Only after that sealed
marker exists does the bootstrap watchdog refuse further repair and become a no-op.
A success-path crash before the marker is replayable by the still-authoritative
bootstrap chain; a crash after it leaves no required recovery-chain work.

The durable Git release marker, pilot and canary roots, and protected-capacity
marker are required
pre-render inputs at their canonical r9 paths. Rendering fails if any of those
inputs is missing, writable, symlinked, tampered, belongs to another
tag/commit/chain namespace, or fails its sealed verifier. Chain schema 11 and
prerequisite-evidence schema 12 bind the
durable bundle/checksum marker and the protected-capacity path, raw SHA-256, size,
and self-hash identity, as well as the
pilot/canary verifier reports, the complete r3 prelaunch-failure portable binding,
the sealed toolchain and package-cache source identities, annotated tag object,
commit, prerequisite source hashes, and the real dependency-cascade proof. Watchdog
drill/readiness markers are
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
read-only and publishes `SOURCE_CHECKOUT_SCHEMA5_V1_2_R9_COMPLETE.json` last; jobs
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

The r9 controller implements the complete `24 -> 96 -> 192 -> 384` rollout state
machine, and launch authorization now requires protected placement for that full
design ceiling before the first ceiling-24 cell is admitted. Scientific cell arrays
and every scientific serving allocation must use authorization-bound partitions with
`PreemptMode=OFF`; neither clients nor servers may use `mit_preemptable`,
`ou_bcs_low`, or any other preemptible placement. The protected-capacity marker must
already prove the 384-cell ceiling, 64-job reserve, 448-job submit headroom, full
CPU/memory reservation, and the generation-one 22-replica/24-active-GPU effective
fleet with zero additive capacity, three warm jobs, four warm-headroom GPUs, and
28 total attested GPUs. Any additive replicas are authorized only after a failed
qualification attempt publishes a controlled capacity transition and the affected
fleet/readiness/smoke gates are rerun.

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
768 cells and 15,360 QIDs across ceilings `[24, 96, 192, 384]`. It records a
`health_soak_384_seconds` window of at least 7,200 seconds while the configured
ceiling remains 384; that health soak is not represented as loaded steady-state
throughput. Separately, it requires a contiguous interval with at least two
observations at exactly 384 active/pending clients, positive trusted-QID progress,
and backlog that permits saturation. The marker reports that measured interval as
`loaded_384_seconds`, `loaded_384_useful_qids`, and
`loaded_384_observation_count`. Qualification also requires useful progress in every
stratum and at least 201,994 trusted QIDs/day with zero integrity or
transport-censor incidents. Its schema-1
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
`schema5-v1.2-r9-client-placement-capacity-generation-v1`; it binds the immutable
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
r9 release/scientific identities, the original sealed failure and receipt, the
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
