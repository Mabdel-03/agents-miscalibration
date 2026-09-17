# Agent 03: Protocol DAGs, prompts, and policy ancestry

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 0, 4–6, 8–9; all prompts; tools/reference_dag.py.

## File ownership
`src/agent_allocation/protocols/`, `src/agent_allocation/prompts/`, `tests/unit/protocols/`. Propose prompt changes through the lead before freeze.

## Implementation tasks
Implement ROOT/SAMPLE, PRIVATE_SINGLE, PRIVATE_TEAM, PEER, DEEP_SINGLE, CP, and POOLED with explicit generation nodes and separate checkpoint views. Use the structural reference as a count oracle, not a production request cache.

Main has 25 roots, 24 updates on root 0, four updates on roots 1–4, and peer teams 2–5 over four update rounds. Create all prefix checkpoints. High A has only two extra deep calls. Reduced model replication follows the 57-call profile. CP branches from existing four-agent PEER round 1 and privately continues three steps; POOLED reads only four initial roots.

Construct prompts using only PublicTask and permitted candidate packets. PEER is synchronous within each team; no same-round visibility. PRIVATE sees only its latest visible solution. Preserve the original question every time. Freeze peer-order permutations independently of return times. Keep hidden thinking, scores, and all non-ancestor candidates out of requests.

Implement ancestry-based policy cost manifests separately from collection cost. Distinct independent draw IDs remain distinct even for identical prompts. PRIVATE_TEAM reuses private roots/updates across team sizes explicitly; different PEER team sizes never alias. No early stop based on repetition/consensus. No horizon in prompts. Preserve all raw histories externally.

## Required tests
Exact counts 125/123/57 and 12+1 extension; root r=0 equality; n(1+R) ancestry; private-team reuse; CP total ancestry 20 at h=3; pooled ancestry 5; no future peers; correct parameterized deep budgets; no hidden labels in rendered prompts; identical prompt does not imply identical node.

## Deliverable
Executable DAG builders, checkpoint manifests, prompt hash locks, and per-item call/cost graphs. Provide a two-item synthetic run to agents 02 and 06.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
