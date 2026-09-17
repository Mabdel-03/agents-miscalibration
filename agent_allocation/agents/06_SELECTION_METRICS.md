# Agent 06: Gold-blind selection, extraction, and derived metrics

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 0, 5.8, 9, 11–12 and 15; contracts and prompt formats.

## File ownership
`src/agent_allocation/selection/`, `src/agent_allocation/extraction/`, `src/agent_allocation/metrics/`, corresponding unit tests.

## Implementation tasks
Extract one explicit final element deterministically without an LLM repair call. Retain parse/finish statuses and raw text. Strip at most one allowed outer code fence. Do not obtain final answers from hidden reasoning, arbitrary rationale letters, or gold-dependent matching.

Implement plurality for normalized text answers and restricted gold-blind mathematical classes. For Math-Verify-based equivalence, guard asymmetry and nontransitive merges; deterministic mutual/complete-link admission is the specified default. Ties use frozen correctness-blind priorities and not return order. Code uses agent 05 public-score records to select; do not label it voting.

Seal candidate pool and selection records before joining private grades. Make it impossible for selectors to accept PrivateEvalSpec/GradeResult. Current pool and archive pool are separate checkpoint views. POOLED is synthesis, not selection from the input pool.

Calculate pass@k from independent banks only, prefix readouts, active/archive coverage, selection gaps, candidate counts, W-to-C/C-to-W, root minority strata, and parse/transport failure rates. Keep semantic correctness tri-state and operational failure separate. Do not treat arbitrary code source differences as semantic answer switches. Export per-item metrics; do not bootstrap agent messages as independent units.

## Required tests
pass@1 and pass@25 identities; stable-combination formula edges; archive monotonicity; candidate-only delivered<=coverage; synthesis exemption; exact transition identity on known panel; unknown statuses stay unknown; no gold dependence in ties/classes; all invalid pool abstains; current accuracy need not be monotone; same population at all k.

## Deliverable
Real populated selector rows, sealed selection hashes, per-item metric tables, and fixtures for agent 07. Empty selector outputs are blockers, not zero-accuracy results.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
