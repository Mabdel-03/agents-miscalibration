# Schema-5 r12 offline Conda toolchain

The schema-5 r12 release must not execute the shared Miniforge base at
`/orcd/data/lhtsai/001/om2/mabdel03/miniforge3`. Its runtime contains a broken
compiler-tool link, and copying its entrypoint would preserve that defect. The two
developer prefixes are scientific inputs only and must never be queried or repaired by
Conda.

The r12 prerequisite instead installs the cached self-contained installer into a fresh
release namespace. The immutable installer contract is:

- file: `/orcd/data/lhtsai/001/om2/mabdel03/Miniforge3-Linux-x86_64.sh`
- release: `Miniforge3-25.11.0-1`
- SHA-256: `be1bad9d4e67a8753eb76fb4940e9a08036786675c7adf060627e55791bf110d`
- resulting Conda version: `25.11.0`

`scripts/provision_schema5_conda_toolchain.py` contains this contract as code; the CLI
does not accept a caller-supplied replacement digest.

## Provision and verify

These blocks are not a standalone entry point. Run them only in the same dedicated
Bash process after the trusted `PATH`, Git, Slurm, locale, and fail-fast prelude in
[SCHEMA5_V12_RECOVERY_RUNBOOK.md](SCHEMA5_V12_RECOVERY_RUNBOOK.md). In particular,
do not run them from an ambient Conda shell.

Create the empty r12 namespace through the recovery transaction, then run the immutable
r12 checkout's provisioner. The first command is a read-only preflight:

```bash
repo=/orcd/data/tpoggio/001/mabdel03/agents_scaling
results=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
namespace="$results/recovery/schema5-v1/toolchains/r12"
installer=/orcd/data/lhtsai/001/om2/mabdel03/Miniforge3-Linux-x86_64.sh
pilot_checkout="$results/recovery/schema5-v1/materialization_pilot_source_checkout_v1_2_r12"
provisioner="$pilot_checkout/scripts/provision_schema5_conda_toolchain.py"
dev_python="$(realpath -e /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python)"

test "$(git -C "$pilot_checkout" describe --tags --exact-match)" = \
  sweep-recovery-schema5-v1.2-r12
test -z "$(git -C "$pilot_checkout" status --porcelain=v1 --untracked-files=all)"
if [[ ! -e "$namespace" ]]; then
  install -d -m 0755 -- "$namespace"
fi
test -d "$namespace"
test ! -L "$namespace"

"$dev_python" -I "$provisioner" provision \
  --installer "$installer" \
  --namespace-root "$namespace"
```

After reviewing the exact paths and installer identity, apply once:

```bash
"$dev_python" -I "$provisioner" provision \
  --installer "$installer" \
  --namespace-root "$namespace" \
  --apply
```

Independent replay neither reads the cached installer nor invokes another Conda:

```bash
toolchain="$namespace/conda"
"$dev_python" -I "$provisioner" verify \
  --toolchain-root "$toolchain"
```

The verified executable is
`$toolchain/base/bin/conda`. Code should obtain it through
`verified_conda_executable(toolchain_root)`, which fails closed unless the full
completion contract verifies.

## Transaction and integrity contract

The provision intent is written before the installer runs. The installer is copied
with ordinary buffered reads and writes, and the copy must have a different inode from
the cache. Installation receives a minimal environment with offline mode enabled and
targets only the exact release-local `base` path. The provisioner never invokes Conda
from the cached installer directory, shared base, harness prefix, or serving prefix.

Every executable-runtime symlink must resolve entirely inside the new base. Miniforge's
extracted `pkgs/` cache contains package-relative links whose targets exist only after
the dependency packages are linked into an environment. Those declared cache entries
are excluded from runtime reachability, but their paths, target text, bytes, and
hardlink topology remain in the complete-prefix inventory. Every regular-file inode
link count must be accounted for by paths inside the base; this permits Miniforge's
internal package-cache hardlinks while rejecting a hardlink dependency on another
prefix. The installer copy is made with explicit buffered reads and writes, so neither
hardlink nor reflink cloning is used. After installation, all write bits are removed. Two isolated
`conda --version` and `conda info --offline --json` probes use writable scratch outside
the toolchain and must report `root_writable=false`, `offline=true`, the exact root, and
no external configuration. Complete prefix inventories and complete runtime identities
before and after the probes must match exactly.

The r12 canonical interpreter path is 110 bytes and its complete shebang is 113 bytes.
The provisioner rejects a namespace before creating an intent or running the installer
when the absolute interpreter shebang would exceed the 127-byte portable limit. This
preflight was added after r4's 154-byte interpreter path caused Miniforge to emit
`#!/usr/bin/env python`, which the absolute base-prefix identity contract rejected.

The identically short r5 toolchain completed successfully and remains sealed
historical evidence. It is not reused for r12 because its marker, protocol, release
tag, and namespace are immutably r5-bound; r12 creates an independent fresh prefix.
The identically short r6 toolchain also completed and remains valid sealed history;
the r6 failure was the recorder's closed binding-field set, not toolchain
provisioning. The r6 prefix is likewise not reused because its marker, protocol,
release tag, and namespace are immutably r6-bound.
The identically short r7 toolchain also completed and remains valid sealed history;
the r7 failure was a Git optional-index refresh during immutable input verification,
not toolchain provisioning. The r7 prefix is not reused because its marker, protocol,
release tag, and namespace are immutably r7-bound.
The identically short r8 toolchain also completed and remains valid sealed history;
the r8 failure was the broken-link recorder importing the full r3 pilot dependency
graph under the minimal toolchain before reaching its stdlib-only runtime-identity
subcommand, not toolchain provisioning. The r8 prefix is not reused because its
marker, protocol, release tag, and namespace are immutably r8-bound.
The identically short r9 toolchain also completed and remains valid sealed history;
the r9 failure was the offline recorder's empty-stdout/single-error diagnostic
contract rejecting genuine bounded Conda 25.11 progress plus multiple canonical
offline-fetch blocks, not toolchain provisioning. The r9 prefix is not reused because
its marker, protocol, release tag, and namespace are immutably r9-bound.

An interrupted, unmarked `base` is never reused. On replay it is atomically moved under
the release namespace's `.conda.provisioning/quarantine`
tree and sealed read-only before a fresh install starts. No partial prefix is deleted.

`CONDA_TOOLCHAIN_COMPLETE.json` is canonical, checksummed, read-only, and published
last. Verification rechecks:

- the marker and marker-first intent identities;
- the sealed installer copy and exact pinned digest;
- the complete base inventory, including package-cache link text and hardlink
  accounting;
- the runtime identity's two-pass safe-link inventory;
- the stricter materialization symlink audit;
- recursive read-only permissions; and
- two fresh offline command probes bracketed by unchanged inventories.

## r12 launch integration

The r12 recovery renderer treats the completion marker as a hard prelaunch
prerequisite, bundles this provisioner and `schema5_conda_runtime_identity.py` from
the exact annotated r12 tag, and invokes `verify_conda_toolchain` at render
verification, scheduler acceptance, materialization start, and release freeze. Run
the provisioner and verifier from the clean detached
`materialization_pilot_source_checkout_v1_2_r12` checkout, never from the mutable
developer checkout. The pilot and production materializer accept
`--conda-toolchain-root` and
`--source-package-cache /orcd/home/002/mabdel03/.conda/pkgs`; they do not accept an
arbitrary Conda executable or the old shared-base path. Materialization schema 5
copies only the exact required extracted packages, available archives, and cache
metadata into a marker-first release-local seed, proves no shared regular-file
inodes, and invokes only the executable obtained from
`verified_conda_executable(toolchain_root)`.

The prerequisite evidence entry should bind at minimum the marker's `marker_id`,
installer contract, intent ID, toolchain root, Conda executable, runtime
`identity_sha256`, complete-prefix `inventory_sha256`, and read-only probe summary.
Any missing marker, writable entry, changed inode/link, unsafe link, failed probe, or
identity mismatch keeps the recovery root held. Provisioning is therefore completed
before the materialization pilot and recovery DAG are allowed to submit.

This toolchain supersedes r3; it does not erase r3. The r12 renderer must also
independently verify
`prelaunch_failures/schema5-v1.2-r3/PRELAUNCH_FAILURE_SEALED.json`, whose two exact
failure classifications record the unsafe shared runtime and unseeded offline cache,
before accepting this replacement toolchain as a production prerequisite.

## Reference real-installer pilot

On 2026-07-27 the exact pinned installer was provisioned and independently verified in
a fresh `/tmp` namespace. The first development attempt deliberately remained
unmarked when a whole-prefix link check encountered two unresolved package-cache
links. Replay preserved that prefix as read-only quarantine `g0001`, installed a fresh
`g0002`, and completed under the explicit runtime/cache boundary above. This exercised
the real interrupted-prefix recovery path as well as normal provisioning.

The completed reference prefix reported:

- 9,896 runtime files, 927 runtime directories, and 1,278 internal runtime symlinks;
- 20,852 complete-prefix files, 2,688 directories, and 2,555 symlinks;
- 8,468 internal hardlink groups and zero externally shared regular-file inodes;
- two inventoried unresolved cache links and zero unresolved runtime links;
- runtime identity
  `f1269812f1db0e5ed35107adc9ebb440f9ce76e8eade4f336f5620f0dda9efdf`;
- complete-prefix inventory
  `972579afd219ead177fe18b4cffe095cdf6eae6c8c78b1c066869dc991e8a92f`;
- marker ID
  `441591457122c687bbc672bdd7573be56bde48f58c4f6f88eb5bfc5cf4f8f344`; and
- canonical marker SHA-256
  `bf9d58f6fe925586f7bcbcc30d34cf8dd78dc67d3a011bb007baf1c10c688356`.

The runtime and complete-prefix digests include the absolute installation prefix and
will therefore differ in the production namespace; their counts and invariant
contracts are the portable reference. The cached installer remained
`be1bad9d…bf110d`, the copied installer used a distinct device and inode, the final
tree contained no writable regular file or directory, and a separate exercising
`verify` pass reproduced the marker byte-for-byte.
