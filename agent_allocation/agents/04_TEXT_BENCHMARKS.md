# Agent 04: MATH-500, MMLU-Pro, and BBH adapters

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 2.1–2.3, 3, 9, 11.1, 12–13; sources S08–S11.

## File ownership
`src/agent_allocation/benchmarks/math500.py`, `mmlu_pro.py`, `bbh_logic4.py`, `text_manifest.py`; `src/agent_allocation/grading/text/`; corresponding unit tests.

## Implementation tasks
Resolve and pin source revisions; inspect live dataset schemas. Build allowlisted PublicTask records and private grading records separately. Generate seeded, source-item-disjoint development/main manifests. Preserve proportions for MATH/MMLU and equal BBH subtask allocation. Construct fixed balanced waves and nested subsets without looking at model outcomes.

Use the four BBH files named in the master, zero shot, with task-specific exact final extraction requirements. Do not load CoT demonstrations or call the subset the full BBH score. Keep MMLU options in their original order and apply frozen outer-whitespace cleanup. Remove all answer/rationale label fields from model input.

Implement pinned Math-Verify grading, preserving argument order and handling exceptions/timeouts. Coordinate with agent 06 on candidate extraction: agent 06 owns output extraction and voting classes; you own gold evaluation. Export a gold-free mathematical parsing helper when needed without exposing the gold record to a selector.

Validate reference answer round trips and source IDs. Preserve test revisions/preprocessing/license notes and label any excluded item with a pre-generation engineering reason. A changed upstream file after freeze must not quietly replace the local version.

## Required tests
No gold/solution/rationale leakage in serialized PublicTask; immutable revision/hash; dev/main disjointness; correct wave balance; correct labels for all four BBH formats; legal MMLU choices only; mathematical-equivalence and asymmetry fixtures; grader exception remains unknown; no per-k denominator shrinkage.

## Deliverable
Three ready adapters with public/private fixtures, locked manifests, deterministic data preparation, and an engineering exclusion audit. Report any benchmark availability blocker immediately.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
