# Agent 05: MBPP+, public selection checks, and isolated grading

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 2.4, 3, 8–9, 10.4, 11.2–11.3, 12–13; sources S12–S14.

## File ownership
`src/agent_allocation/benchmarks/mbpp_plus.py`, `src/agent_allocation/grading/code/`, `sandbox/`, and `tests/unit/code/`. Coordinate only the public-score interface with agent 06.

## Implementation tasks
Resolve a pinned EvalPlus release and its actual MBPP+ inventory. Join official public problem/examples by stable task ID. Never hardcode the inventory size or expose canonical_solution, private base tests, plus_input, or private expected outputs. Produce a prompt/interface plus only examples intended for public display.

Implement two separate services: (a) public-example execution returning the fraction passed for terminal selection; (b) private full EvalPlus verdicts for offline evaluation. No runtime feedback to generation. Preserve official comparison and timeout semantics; do not replace them with source equality, AST voting, repr equality, or a new LLM judge.

Generated programs require an OS sandbox: no network/credentials/host repository/Docker socket, unprivileged user, read-only base, disposable scratch, bounded CPU/memory/processes/output, enforceable termination. Keep private reference programs and expected outputs in the trusted evaluator, not available for candidate introspection. Adapt the execution boundary when the stock tool does not provide this separation, and test verdict parity.

Cache grades by code hash, item, test kind, evaluator version, and environment digest. This may save CPU but must never deduplicate API generation draws. Run CPU work outside the API event loop. Do not scale code generation until reference programs, incorrect mutations, public/private separation, and sandbox tests pass.

## Required tests
Reference code succeeds; off-by-one and constant-output mutants fail private tests; a program hardcoding public cases can pass public selection but fail private grading; public/private calls never share answer files; timeout differs from evaluator outage; network/file/fork/output attacks contained; task-specific comparison parity; no unsupported dependencies silently dropped.

## Deliverable
A working isolated public checker and full grader, pinned environment, cache schema, and safety/parity report. If isolation or source inventory is blocked, notify lead and never run code unsandboxed.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
