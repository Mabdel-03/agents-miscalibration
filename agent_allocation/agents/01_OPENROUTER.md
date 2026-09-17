# Agent 01: OpenRouter adapter and capability checks

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 4, 7, 9, 10.3, 10.5, and 13; sources S01–S07.

## File ownership
`src/agent_allocation/api/`, `tests/unit/api/`, API response fixtures. Do not edit the scheduler ledger or shared contracts directly.

## Implementation tasks
Implement an async HTTP adapter using one persistent httpx client. Build the documented reasoning.max_tokens request with completion headroom, exact provider lock, no fallbacks, parameter requirements, and response caching disabled. Do not assume support based solely on accepting a JSON field. Add model/endpoint discovery and a capability canary. Resolve provider tags from metadata and store raw snapshots.

Normalize raw usage without filling absent reasoning counts with zero. Preserve total completion tokens, reasoning subset, prompt/cache fields, billed cost, provider, generation ID, finish reason, raw response/hash, and response headers. Do not double-count thinking. Keep Decimal money. Return a typed result to the coordinator; do not mutate its ledger or retry internally.

Categorize 401/402/403, 429, transient upstream errors, completed length/empty responses, refusals, and malformed content. An HTTP 200 containing an error is not success. Disable automatic SDK retries. Expose retry hints but let the scheduler decide. Implement metadata reconciliation without assuming a lost completion body can be recovered. Treat response-cache HIT as an audit failure for independent generation.

## Required tests
Mock success with and without reasoning fields; empty length-limited content; retry hints; provider mismatch; 200 error envelope; malformed usage; streaming disabled; response cache HIT; redacted Authorization. Demonstrate different draw IDs do not collapse even when request bodies match. No real API call outside a lead-authorized dispatcher pilot.

## Deliverable
A tested adapter, capability-report schema, normalized response fixture, and explicit provider/budget blockers. Give the lead one mock-backed integration example.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
