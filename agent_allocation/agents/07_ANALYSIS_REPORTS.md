# Agent 07: Analysis, cost frontiers, completeness, and report generation

## Context
You implement one module of `openrouter_agent_allocation_v1`. Read the referenced sections and shared contracts before editing. Work in your assigned Git worktree. Use mocks immediately so other agents need not finish first. The experiment compares sampling, private reasoning, communication, and final selection on four benchmark families under one shared budget.

## Read
Master sections 6–7, 12, 14–16; supplied budget calculator.

## File ownership
`src/agent_allocation/analysis/`, `src/agent_allocation/reporting/`, analysis tests. Do not change grading, prompts, or selection based on interesting results.

## Implementation tasks
Build deterministic exports and a static Markdown/HTML report from stored outputs. Report each benchmark separately, equal-benchmark macro averages, and equal BBH-subtask weighting. Cluster all methods/rounds for one source item in paired bootstrap resampling. Include independent-item counts and small-panel warnings.

Plot delivered accuracy versus actual dollars, requested allowance, realized tokens, and nominal generations with clear labels. Separate oracle pass@k from achievable vote@k/public_select@k. Show raw measured configurations and a descriptive frontier without claiming an unseen optimal configuration. No test-set winner presented as validated policy. Add budget-admission replay using only prefix-observable information and conservative next-call reservations, or report the curve as descriptive when replay is unavailable.

Implement the two frozen primary contrasts, Holm if reporting their p-values, and descriptive CP improvement decomposition. Preserve all planned/started/ungraded states in a completeness matrix. Unknown-grade bounds and operational failures remain distinguishable. Report per-protocol prompt growth, headroom utilization, truncation, cache usage, stop reasons, latency and billed retries.

Join policy ancestor-cost manifests from agent 03, not total experimental collection spending. Include code public-check CPU cost separately from private research grading. Compare core panel variants on their shared IDs only; do not compare 96-item high A to all 400 low-A items as an unpaired aggregate.

## Required tests
Hand-computed small fixture matches exports; clustered CIs are item-based; shared-subset high/low joins; no outcome-dependent subset removal; candidate/synthesis invariant scope; cumulative costs monotone; current accuracy allowed nonmonotone; broken selector shown missing; price units verified; figure generation needs no API key.

## Deliverable
Report generator, tables, figures, audit/completeness summaries, and reproducible analysis command. Interim plots must identify partial collection; final report requires reconciled costs and grading.

## Coordination rules
Do not edit shared contracts or dependency locks directly; send a minimal requested change to the lead. Commit working increments, document assumptions, and supply a typed mock fixture to downstream agents. Do not add unrequested model calls, benchmarks, judges, or paid services. No independent live API runs. Only the lead's dispatcher may spend the shared allowance. Never modify results or test populations to favor communication.
