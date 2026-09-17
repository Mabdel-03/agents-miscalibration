# Agent 08: Independent QA and failure injection

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Full master, especially sections 13 and 15; both user-supplied report caveat summaries in SOURCES.md.

## File ownership
`tests/integration/`, `tests/fixtures/adversarial/`, `docs/audit/`. Submit fixes to owners; do not rewrite their modules without coordinating.

## Implementation tasks
Act as an adversarial verifier of the experiment. Begin before the adapter exists, using the reference DAG and synthetic generation/grade fixtures. Test measurement integrity before scientific output.

Recreate the prior root-bank/private-baseline collision using identical episode labels. Test prompt-identical independent draws, duplicate completion notifications, all-null selectors, unknown grades, malformed finals, context growth, code public/private test leakage, and provider-response caching. Check all ancestor call counts and population joins.

Inject 429, connection timeout after dispatch, delayed bills, worker crash after writing raw response, SQLite commit failure, lease expiry, grading outage, and spending-limit contention. A resume must not double dispatch a known-successful node or release unknown money. Test across the three core profiles and the extension.

Create one deterministic mocked end-to-end four-benchmark run that generates, selects, grades, exports, resumes, and reconciles. Real-model output is not required for unit/integration tests. Verify source code never sees credentials or gold/reference files and that API prompts contain no private fixture canary values. Check the actual production suite, not only the supplied reference/planning tests.

## Required tests
Require all master section 15 checks. Audit one full item of each benchmark through raw requests, outputs, selected candidate, private grade, and policy cost by a separate manual/simple script. Recompute two headline contrasts from raw per-item records and compare with the analysis module.

## Deliverable
AUDIT.md listing passed/failed checks, independent recomputations, unresolved blockers, and explicit release recommendation. Favor a paused run over a fast invalid result.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
