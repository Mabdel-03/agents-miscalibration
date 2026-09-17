# Agent 02: DAG scheduler, persistence, and global spending

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 7, 9, 10, 13; reference_dag.py; source S07.

## File ownership
`src/agent_allocation/runtime/`, `src/agent_allocation/storage/`, `tests/unit/runtime/`. The lead owns CLI entry points and shared types.

## Implementation tasks
Implement a ready-node DAG scheduler with one async networking coordinator, one serialized SQLite writer on local disk, immutable raw-response artifacts, and restartable node/attempt records. A root completion must unlock its dependents immediately. Semaphores cover HTTP work only, never backoff, dependency waits, or CPU grading.

Build atomic global reservations with settled, in-flight, and unresolved exposure. Reserve before send, settle after confirmed usage, never release a timed-out request as free. Study ceiling is cumulative across pilot/main/replication. Add phase gates, $2 unresolved-exposure circuit breaker, optional-block admission, graceful draining, and explicit reserve usage. Reconcile provider credit information without mistaking unrelated account usage for study costs.

Use balanced waves of 16, no more than four active waves initially. Protect completion of admitted paired panels. Prioritize older waves and long private chains while remaining fair across protocols/benchmarks. Start at 16 concurrency and 2 requests/sec, AIMD toward capped 128 and 16/sec subject to capacity. Honor Retry-After. Do not evade limits with extra keys. Retry transport failures at most three attempts; completed bad answers are terminal model outcomes.

Implement leases/heartbeats and recovery of sent-but-unknown attempts without assuming network exactly-once delivery. Distinguish raw artifact persisted from database committed. Use durable outbox-style ordering. Never run two active attempts for one semantic node.

## Required tests
Stress-test concurrent reservations against a tiny ceiling; mock crash after provider response but before DB commit; timeout with later bill; stale sent lease; 429 burst; slow grader; missing parent; duplicate completion notification; disk failure; graceful stop. On restart, successful nodes make zero new requests. Outstanding costs cannot disappear.

## Deliverable
A mock run that streams dependent completions, a ledger reconciliation report, and measured effective concurrency. Do not benchmark only task creation speed.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
