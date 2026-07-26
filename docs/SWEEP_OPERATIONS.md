# Sweep operations

> **Protocol reference for the retired v1 runs.** Sections below that name
> `full_sweep_v1`, `.dispatcher-v3`, `launch_dispatcher.py`, or legacy repair commands are
> evidence/forensics documentation and must not be used for new admission. The
> authoritative clean-rerun procedure is
> [SCHEMA5_V12_RECOVERY_RUNBOOK.md](SCHEMA5_V12_RECOVERY_RUNBOOK.md); it uses three new schema-5
> run IDs, a sealed release, `.dispatcher-schema5-v1`, typed readiness gates, and a
> transactional dual-controller resume.

The retired controlled v1 sweep was admitted by one global dispatcher. The older per-run
chunk drivers are retained only for historical inspection and fail closed unless an
explicit legacy override is supplied.

## Frozen scope

Each run is defined by an immutable `cells.json` plus `cells.sha256` and the checksummed
`benchmark_contracts.v1.json` sidecar. The sidecar binds every unique
`(benchmark, n_questions, seed)` selection to the full pinned source revision, exact QID
order, and hashes of normalized prompt text, option text/order, answer key, and answer
type. Normal run initialization resolves every cache-only source first, builds and
verifies all four files in a hidden sibling directory, then publishes the bundle with one
same-filesystem directory rename. A crash therefore exposes either no run or a complete
run, never a manifest without its Question contract. Never regenerate a run in place.
The intended headline scope is the disjoint union of:

| run | cells | agent counts |
| --- | ---: | --- |
| `full_sweep_v1` | 4,680 | 1 and 3 |
| `full_sweep_agent_counts_v1` | 14,400 | 2, 4, 5, and 6 |
| `full_sweep_agent_count_7_v1` | 3,600 | 7 |
| **total** | **22,680** | **1 through 7** |

Use `scripts/init_run_manifest.py` to create and freeze a new run. It refuses to replace
any existing run path. For a checksum-frozen historical manifest, the audited
`scripts/freeze_benchmark_contracts.py` attachment path never modifies `cells.json` or
`cells.sha256`, never replaces an existing sidecar, and recovers an interrupted
sidecar-before-checksum publication only after reloading and matching every pinned
Question. `--verify-only` also rechecks normalized content against the shared local
`HF_HOME`; source access never falls back to a mutable branch or network download. GPQA
requests for 200 questions deliberately contract to all 198 rows of the complete pinned
Diamond split. Analysis reads only manifest-listed cell IDs by default; stale or pre-trim
directories require an explicit supplementary mode.

## Admission

First run the exact live plan without writing state or submitting jobs:

```bash
python slurm/dispatch_sweeps.py dispatch --dry-run \
  --state-dir "$ASYS_RESULTS_ROOT/.dispatcher-v3" \
  --run full_sweep_v1 \
  --run full_sweep_agent_counts_v1 \
  --run full_sweep_agent_count_7_v1 \
  --server-pool full_sweep_v1=full_sweep_agent_counts_v1 \
  --server-pool full_sweep_agent_count_7_v1=full_sweep_agent_counts_v1
```

The report must have an empty `unmappable_cell_jobs` list and no validation errors. Check
that every backlogged run appears in `selected`. Production defaults enforce a 448-job QOS
ceiling, reserve 64 jobs, submit at most 24 cells per array, and request one CPU plus 4 GB
per HTTP client. The controlled 2 GB production pilot on 2026-07-18 reached 1,561,824 KiB
MaxRSS for a 14B worker, crossing the rollout's 1.5 GB promotion threshold. This remains
far below the legacy 16 GB reservation while retaining tokenizer-initialization headroom.

Current-protocol traffic also requires an explicit serving layout, live Slurm allocation,
and finite positive registration timestamp on every endpoint. Legacy cross-pool records
remain available for historical audit but contribute zero dispatcher capacity and cannot be
selected by a worker. If a live server has an incomplete target-pool record, repair only the
named profiles from immutable Slurm-spooled launch evidence (this mode never enters the
server-submission tick):

```bash
python slurm/keepalive.py \
  --run-id full_sweep_agent_counts_v1 \
  --repair-profile 4B \
  --repair-profile 14B \
  --repair-profile 32B
```

Re-run the endpoint monitor and dispatcher dry run after repair. Every endpoint counted as
`live` must also be counted by `current_provenance_valid_registered`; do not repair a record
by hand or copy a stale registry JSON between pools.

For a new state directory, render the durable singleton only after the dry run passes:

```bash
python slurm/launch_dispatcher.py \
  --state-dir "$ASYS_RESULTS_ROOT/.dispatcher-v3" \
  --run full_sweep_v1 \
  --run full_sweep_agent_counts_v1 \
  --run full_sweep_agent_count_7_v1 \
  --server-pool full_sweep_v1=full_sweep_agent_counts_v1 \
  --server-pool full_sweep_agent_count_7_v1=full_sweep_agent_counts_v1
```

Inspect and checksum the rendered control file, then submit it explicitly. On restart,
reuse the already-audited file and state directory; do not invoke the renderer again because
rendering intentionally replaces `global_dispatcher.sbatch`.

```bash
bash -n "$ASYS_RESULTS_ROOT/.dispatcher-v3/global_dispatcher.sbatch"
sha256sum "$ASYS_RESULTS_ROOT/.dispatcher-v3/global_dispatcher.sbatch"
sbatch --parsable "$ASYS_RESULTS_ROOT/.dispatcher-v3/global_dispatcher.sbatch"
```

The dispatcher holds both a results-root-global advisory lock and a state-directory lock,
atomically records pre-submit intents and Slurm job mappings, and queues an `afterany`
successor only after both locks are owned. Thus a migration to a fresh ledger directory
cannot create a second admission controller. Inspect its durable state with:

```bash
python slurm/dispatch_sweeps.py status \
  --state-dir "$ASYS_RESULTS_ROOT/.dispatcher-v3"
```

The `.dispatcher-v3` name is the durable state directory of this controlled rollout;
artifact schema 5 and the unchanged generation protocol v4 do not require renaming the
ledger. Reuse it on restart so
the three pilot arrays, pre-submit intents, fairness deficits, and per-cell fingerprints
are reconciled rather than forgotten. Never start the same run set from a fresh state
directory.

Use the semantic monitor for per-run health. Shared serving pools use the same explicit
mapping syntax as the dispatcher, so endpoint generation and dormant-failure state are
reported against the fleet that actually serves the run:

```bash
python scripts/monitor_run.py \
  --run-id full_sweep_agent_count_7_v1 \
  --server-pool full_sweep_agent_count_7_v1=full_sweep_agent_counts_v1
```

The report separates connection, context-capacity, generation-truncation, response-shape
protocol, tokenizer, and bad-request log signatures; includes user-scoped Slurm
pending/QOS reasons; reports completed, length-censored, and protocol-censored QID
counts/rates; reports auxiliary self-consistency length/protocol censors under their own
sample denominator (plus explicitly named all-generation health rates); and emits a joint
`model_size × reasoning_level × topology × agent_count` ETA table. The ETA gate
defaults to a 48-hour observation window and fails closed when any remaining stratum has
no measured throughput or projects beyond 28 days. Its start time defaults to the atomic
`throughput_observation_started_at` epoch written by the first live post-remediation
dispatcher poll; dry runs never start the clock. It can be pinned explicitly with
`--eta-started-at`.

Do not add `full_sweep_agent_count_7_v1` until both `32B-long` TP=2 replicas are live and
the context-capacity and live smoke gates below pass. When admitted, map it explicitly to
the agent-count server pool:

```text
--run full_sweep_agent_count_7_v1 \
--server-pool full_sweep_agent_count_7_v1=full_sweep_agent_counts_v1
```

## Completion and repair

`CompletionStatus` is the sole completion contract shared by the runner, dispatcher,
monitor, repair utility, and analysis. A cell is complete only when its metadata matches
the frozen manifest and its canonical JSONL contains exactly one valid row for every
benchmark-derived QID. The validator also reloads the deterministic Question contract:
stored answer keys must equal benchmark truth and stored correctness must equal a fresh
grade of the retained final answer. Supplying an expected QID list does not disable this
truth check; the list must match the derived Question order. For current rows it also
re-extracts each producer answer, validates normalized option probabilities, replays the
topology decision, and recomputes the exact confidence dictionary. A `meta.json` file by
itself is not success.

Audit is read-only by default:

```bash
python scripts/audit_repair_run.py --run-id full_sweep_agent_counts_v1
```

`--apply` acquires a non-blocking per-cell lock, preserves the first valid expected-QID
row, atomically removes malformed/duplicate/unexpected rows, and quarantines invalid
metadata. During migration, also exclude every cell owned by a live legacy worker because
those workers predate the advisory-lock contract. Runners resume only the missing QIDs and
publish metadata after semantic revalidation.

Within a missing QID, stochastic work is also restart-safe. The runner journals each
`(agent, round)` topology response or censor, every scheduled self-consistency sample, and
the assembled topology terminal in `.qid_checkpoints/` using atomic replacement and an
integrity hash. Checkpoint schema 2 is pinned to the full cell, Question, source,
serving-profile, artifact schema, generation-censor protocol, and self-consistency
protocol identity. Endpoint fallback and process restart therefore replay already observed
coordinates and issue only missing requests. A completed output, `length_censored`
response, or `protocol_censored` response is replayed exactly; a corrupt or mismatched
journal is a fail-closed configuration incident, never an invitation to draw a replacement.
The journal is removed only after the canonical QID row is fsynced; stale journals whose
rows survived a kill are removed during lock-protected resume.

Failure state is atomic and classified. Connection/runtime failures back off and become
dormant after repeated attempts. A preflight whose remaining output envelope is smaller
than its treatment-specific floor is a context-capacity/configuration failure and consumes
no retry slots until the serving profile or runtime code version changes. A server response
with missing token IDs, prompt/usage disagreement, parser-content mismatch, or another
untrusted envelope raises `ServerResponseProtocolError` and is configuration-blocked. It
must not be retried automatically: once a response was sampled, accepting a different
response would condition the dataset on parser success. A response whose exact token IDs
and usage are trustworthy but whose terminal/delimiter/budget shape violates generation
protocol v4 is instead retained once as a schema-5 `protocol_censored` outcome and creates
no failure ledger.
Result provenance names the exact serving process as `endpoint_generation`; failure
dormancy instead records `server_pool_generation`, a process- and layout-aware hash of
the whole live profile fleet. Reusing a host and port after restart therefore revives a
dormant transient failure without falsifying which endpoint produced a retained result.

Generation protocol v4 treats exact full-envelope exhaustion differently from a failure.
If the one registered sample consumes every requested output token and returns
`finish_reason="length"`, the runner writes one `length_censored` QID row with
`final_answer=null` and `correct=false`. Artifact schema 5 applies the same one-draw rule to
a post-validated response-shape anomaly and writes `protocol_censored`. Both retain full
token-ID, request, endpoint, and checkpoint-coordinate provenance; neither may be retried,
given a different seed, replaced after endpoint fallback, or admitted from a parseable
prefix. These are semantically valid observed outcomes, not endpoint failures. Cell
metadata must partition all QIDs with `completed_question_count`,
`length_censored_question_count`, and `protocol_censored_question_count`, and must report
the canonical row versions in `artifact_schema_counts`.

The validator continues to read schema-4 artifacts exactly under their frozen contract:
only `completed` and `length_censored` statuses, self-consistency protocol v1, and no
schema-5 coordinate or generation-censor fields. They are not rewritten merely because
schema 5 exists. A schema-4 metadata record cannot certify a `protocol_censored` row;
new production rows and metadata are schema 5, and a schema-5 metadata record explicitly
accounts for any retained schema-4 rows in `artifact_schema_counts`.

The runner establishes a calibration barrier before each topology execution: all agents'
question-only forced-option probes must succeed and are cached before the first stochastic
chat is issued. Endpoint fallback after a probe failure therefore cannot resample an
earlier agent. In single-agent cells with `n_samples>1`, auxiliary draws use the exact
schedule `seed + 1000 + sample_index`. Every coordinate is retained as a stopped output,
an exact-envelope length censor, or an exact response-shape protocol censor; later
coordinates still run after a censor, and no replacement draw is permitted. Any auxiliary
censor makes SC/semantic-entropy aggregates undefined;
the primary topology answer remains completed, while auxiliary token/wall costs are
recorded separately from topology efficiency.

### Generation-protocol v2/v3 incident migration

Explicit schema-v2 and schema-v3 pilot artifacts must not be pooled with protocol-v4
outcomes. Schema v2 was the brief serving-provenance pilot and schema v3 was the retired
multi-phase generation pilot. Run
the evidence-preserving migration only after the affected workers have drained. It
requires each target run to have a checksum-frozen manifest, skips active/locked cells,
and leaves schema-less legacy rows and metadata untouched.

First perform the default read-only audit:

```bash
for run_id in \
  full_sweep_v1 \
  full_sweep_agent_counts_v1 \
  full_sweep_agent_count_7_v1
do
  python scripts/migrate_generation_protocol.py --run-id "$run_id"
done
```

Inspect `errors`, `active_or_locked_cells_skipped`, the artifact counts, and the manifest
checksum in every report. Apply the same target set only when the audit is clean:

```bash
for run_id in \
  full_sweep_v1 \
  full_sweep_agent_counts_v1 \
  full_sweep_agent_count_7_v1
do
  python scripts/migrate_generation_protocol.py --run-id "$run_id" --apply
done
```

For each manifested, lockable cell, the migration removes only rows with an exact
superseded schema-v2 or schema-v3 marker, quarantines matching explicit metadata and
only exact legacy schema-v1 permanent/configuration `GenerationTruncationError` failure
ledgers. Ambiguous or current failure ledgers are retained. Every removed artifact's
original UTF-8 payload and SHA-256 are recorded in the atomic run-level
`generation_protocol_migration_incident_v1.json`. Re-run the dry audit after application;
do not restart admission until it reports no target artifacts, skips, or errors. The
normal completion/repair workflow then resumes the missing QIDs under protocol v4.
Any other explicit schema marker, including a future version, is an error and is never
deleted or reinterpreted by this incident-specific utility.

### QID checkpoint schema-1-to-2 migration

Do not start a schema-5 worker while any retained QID journal is still schema 1. The
current runner deliberately fails closed on that identity rather than silently replaying
an observation under a different artifact or self-consistency contract. Stop dispatcher
admission and wait for every legacy cell worker to drain first; keeping serving processes
alive is safe. First complete the discarded-response discovery/reset in the next section:
a checkpoint belonging to a whole-cell incident must be preserved in that incident archive,
not needlessly rewritten before the cell is reset. Then audit every remaining active
checkpoint in all frozen runs without mutation:

```bash
python scripts/migrate_qid_checkpoints.py \
  --run-id full_sweep_v1 \
  --run-id full_sweep_agent_counts_v1 \
  --run-id full_sweep_agent_count_7_v1
```

The migrator accepts only checksum-frozen manifests and benchmark contracts and the exact
allowlisted schema-1 executable/protocol identity. It validates every coordinate,
terminal, Question, configuration, serving profile, and integrity hash. Any active or
locked cell is skipped, and any source drift or ambiguous identity is an error. Do not
apply while `active_or_locked_cells_skipped`, `unsafe_or_invalid_cells_skipped`, or
`errors` is nonzero.

After reviewing `schema1_candidates`, the target code identity, and both frozen checksums,
apply the same target set explicitly:

```bash
python scripts/migrate_qid_checkpoints.py \
  --run-id full_sweep_v1 \
  --run-id full_sweep_agent_counts_v1 \
  --run-id full_sweep_agent_count_7_v1 \
  --apply
```

Before each atomic replacement, the exact source UTF-8/SHA-256, source integrity and
identity, target identity, and migration record are committed to
`qid_checkpoint_schema_1_to_2_incident_v1.json`. Coordinates, timestamps, completed
outputs, censors, and terminals are preserved; only the executable/protocol identity and
schema wrapper change. The operation is idempotent. Repeat the read-only audit until it
reports zero schema-1 candidates, zero skips/errors, and every migrated journal as
`schema2_already_migrated` (or a native schema-2 journal). This clean audit is a hard
dispatcher-resume gate.

### Discarded-response incident archive and whole-cell reset

Before schema 5, some `ServerResponseProtocolError` responses were sampled, discarded,
and then retried. At least one retry under the same prompt and seed produced a different
accepted completion, so retaining any later row from such a cell would condition the
estimand on parser success. Those cells require a clean whole-cell rerun; labeling the
later response as the original censor is forbidden because the discarded token IDs are
not available.

Discovery joins two evidence sources: current manifested `failure.json` records and
protocol tracebacks mapped through the durable dispatcher ledger, immutable batch
manifest, and task index. Run it read-only first and name every dispatcher state directory
that admitted affected work:

```bash
python scripts/archive_protocol_incidents.py \
  --run-id full_sweep_v1 \
  --run-id full_sweep_agent_counts_v1 \
  --run-id full_sweep_agent_count_7_v1 \
  --dispatcher-state-dir "$ASYS_RESULTS_ROOT/.dispatcher-v3"
```

An unmappable traceback, manifest/config drift, symlink, changed source byte sequence, or
selected cell without exact evidence fails closed. Review every discovered cell and its
failure/dispatcher evidence. Once admission is stopped and all workers have drained,
apply the exact same command with `--apply`. During a controlled drain it is also safe to
archive an already-discovered, unlocked cell: every legacy worker owns its advisory cell
lock for its whole mutation window, and the utility skips locked cells. A final
discovery/application pass after the worker count reaches zero is still mandatory because
draining workers can emit new incidents. For each lockable cell, the utility
first publishes an immutable, hash-verified archive under
`incidents/discarded_server_response_protocol_v4/<cell_id>/`, including the active
`results.jsonl`, `meta.json`, `failure.json`, all QID checkpoints, and relevant dispatcher
evidence. Only that archived byte snapshot is then removed from the active cell directory;
`reset_complete.json` closes the recoverable archive-before-reset transaction.

Re-run the read-only command. Admission remains blocked until every affected cell reports
`already_reset`, with zero `candidate`, `archive_pending_reset`, `locked`, or `error`
outcomes. Later schema-5 rerun artifacts are outside the sealed incident transaction and
are never removed by an idempotent re-run. The dispatcher must schedule these reset cells
from QID one so each receives an unbiased one-draw schema-5 result.

### Production dispatcher resume gate

The paused dispatcher may be resumed only after all of the following are simultaneously
true:

- legacy cell workers have drained and queued legacy successors are absent/canceled, while
  required serving keepalives remain healthy;
- the v2/v3 artifact migration audit has no target artifacts, skips, or errors;
- every discarded-response incident is immutably archived and `already_reset`;
- after those whole-cell resets, the checkpoint audit has no schema-1 candidates,
  active/locked skips, unsafe cells, or errors;
- completion repair reports zero corrupt metadata, malformed/duplicate/unexpected QIDs,
  and unresolved permanent failures outside an explicitly resolved incident;
- both `32B-long` replicas and every required standard/long profile pass provenance and
  context-capacity checks;
- a fresh 15-cell schema-5 smoke run for agent counts 3, 6, and 7 passes the gate below,
  with no unresolved length or protocol censor; and
- a final dispatcher dry run has no validation errors or unmappable jobs, respects the
  384-cell ceiling and 24-cell microbatch cap, and allocates validation/admission fairly
  across every backlogged run.

Resume with the existing audited `.dispatcher-v3` ledger and rendered singleton; never
start a fresh state directory, regenerate the control file, or run a per-run legacy driver.
After submission, require every backlogged run to receive admission within two polling
intervals and require no memory-QOS holds. A failed gate pauses admission again without
canceling healthy servers or already-running cells.

## Long-context gate

All coordinated 32B cells sharing intermediate results or CoT route to `32B-long`
(40,960 tokens, TP=2); artifact-only 32B cells remain on the 16,384-token TP=1 profile.
The implemented 40,960-token window deliberately exceeds the plan's 32K minimum: the
exact seven-agent dense-peer audit showed that 32K cannot preserve every registered
reasoning floor, while 40,960 leaves a positive worst-case floor margin. This is a
serving-capacity amendment only; prompts, peer context, reasoning treatments, and sampled
outcomes are unchanged.
For 0.6B–14B, the one-GPU 40,960-token `-long` profile is selected at the exact dense-peer
risk boundary: at least six agents with B8192/UNLIMITED, or seven agents with B2048, when
the topology shares intermediate results or CoT. Other cells stay on the 32,768-token
standard profiles. Every native answer/judge chat is counted with the cached model
tokenizer and exact chat template before HTTP; the raw one-token option probe is counted
with its literal completion-prompt tokenization.
Each fully rendered peer block is also capped at 4,000 exact profile-tokenizer tokens;
truncation carries an explicit hash-covered marker and is recorded on the consuming
agent output. The original 4,000-character CoT-field cap remains in force.
The invariant is:

```text
output_floor(off, b512, b2048, b8192, unlimited)
    = (4096, 4608, 6144, 12288, 12288)
remaining_output = served_context - exact_rendered_input - 128
require remaining_output >= output_floor
max_tokens = remaining_output
exact_rendered_input + max_tokens + 128 = served_context
```

Run `scripts/audit_context_capacity.py` against the frozen seven-agent manifest, then run
representative live smoke cells for agent counts 3, 6, and 7 across reasoning levels. The
unlimited-reasoning cases must not fail the floor preflight. A stopped result must carry
the exact full-envelope request and its non-empty `endpoint_generation`. If a sample
truthfully reaches that envelope, the smoke report must expose its validated
`length_censored` row; it must not resample, silently truncate, or relabel it as completed.
After exact token/usage validation, a terminal, delimiter, budget, or finish-shape anomaly
must likewise appear exactly once as `protocol_censored`; it must not become endpoint
fallback or replacement sampling. An untrusted response envelope remains a blocked
configuration incident. A censor is a semantically valid retained outcome, but any censor
in this small readiness smoke requires review and fails the dispatcher-resume gate until
its serving/protocol cause is resolved or the scientific design explicitly accepts it.

The live smoke is a fresh immutable run, not part of the 22,680-cell estimand. Initialize
it with `scripts/init_run_manifest.py`, then pin the printed sidecar byte hash on every
manual `run_one` task using `--benchmark-contracts-sha256`. Use the shared agent-count
server pool and the exact 15-cell array range `0-14%1`. Older smoke run IDs must never be
reused. Restart production admission only after all 15 smoke cells pass semantic
schema-5 validation, have the expected schema/protocol provenance, satisfy the censor
readiness rule above, and the smoke array has fully drained.

## Final acceptance

Completion is valid only when all 22,680 manifest cells pass semantic validation with zero
corrupt metadata, missing/duplicate/unexpected QIDs, unresolved permanent failures, or
stale-directory ingestion. Every QID must be partitioned into `completed`, validated
`length_censored`, or validated `protocol_censored`; both censor counts/rates and their
combined rate must be reported. Censored outcomes remain incorrect in accuracy, and
coordination efficiency is withheld for any affected target or baseline cell. Primary
calibration/ECE and deltas are likewise withheld for an affected cell; any completed-only
estimate must be labeled supplementary and conditional on termination. Primary topology
reasoning-token means are likewise withheld when a top-level QID censors; only an
explicitly named `completed_only` statistic may describe the observed terminating
top-level rows. Auxiliary self-consistency costs and censors use their own counters and
denominator and do not redefine that primary topology mean. There must be zero
uncaptured/ambiguous response anomalies and every pre-schema-5 discarded-response incident
must remain linked to its sealed archive and unbiased whole-cell rerun. After 48 hours of
unified dispatch, compute ETA by model size, reasoning level, topology, and agent count. If
the stratified projection exceeds 28 days, add serving capacity rather than trimming the
registered design.
