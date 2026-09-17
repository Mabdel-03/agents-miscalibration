# Parallel coding-agent handoff

Start with **MASTER_SPEC.md**. Give **agents/00_LEAD.md** to the coordinating coding agent and distribute **agents/01_... through 08_...** to eight agents in separate worktrees. They share one immutable contract and one paid dispatcher, not eight spending budgets.

## Included

- `MASTER_SPEC.md`: complete scientific and implementation contract.
- `agents/`: lead brief and eight disjoint parallel assignments.
- `configs/balanced150.json`: 400 main items across four benchmark families, high-A and continuation subsets, optional model check; $135 credit ceiling.
- `configs/lean100.json`: smaller 320-item plan and no second model; $90 credit ceiling.
- `prompts/`: initial, private, peer, pooled, common-system and task-output templates.
- `shared/contracts.py`: typed interface seed for integration.
- `tools/estimate_budget.py`: working offline cost planner.
- `tools/reference_dag.py`: working structural DAG/reuse reference, without inference.
- `tests/`: offline planner/DAG/identity tests.
- `SOURCES.md`: official API and benchmark sources plus user-report provenance.

## Run the included offline checks

```bash
python tools/estimate_budget.py --preset balanced150
python tools/estimate_budget.py --preset balanced150 --avg-input-tokens 5000
python tools/estimate_budget.py --preset lean100
python tools/reference_dag.py
python -m unittest discover -s tests -v
```

These tools use the Python standard library and spend no money. The full experiment CLI described in the master spec is an implementation deliverable, not a tool already shipped here.

## Default plan

MATH-500, MMLU-Pro, four specified BBH subtasks, and MBPP+; 100 main questions per benchmark family. Flash-Lite at requested A=1,024, high-A subset at 4,096. Retain all sampling/revision checkpoints through 25 and all 2–5 agent / 1–4 peer-round combinations. Include independent private teams, deeper single calls, communicate-once-then-private, and pooled synthesis controls.

The coding benchmark uses public-example selection followed by private full-test evaluation, not syntactic voting or hidden-test selection. The default estimated generation count is 67,040 before pilot/retries, including optional replication. The planning estimate at 2,500 mean input tokens/call is approximately $129.76 credits with pilot and contingency. Costs and concurrency require live preflight; local code grading and applicable credit-purchase fees are not included in the API estimate.

## First message to the lead

> Implement MASTER_SPEC.md using the eight parallel briefs in agents/. Freeze the shared contracts first, then let the agents work against mocks in separate worktrees. Prioritize a minimal safe OpenRouter adapter, global budget ledger, resumable DAG scheduler, and one end-to-end development pilot while other modules finish. All paid requests must pass through one dispatcher; main inference requires a passing pilot and frozen manifests. Use balanced150 unless I explicitly select lean100. Do not spend beyond its cumulative credit ceiling, leak private tests into selection, regenerate shared roots, or call the study complete before selection, grading, and cost reconciliation finish.

The user must provide credentials and authorize the actual live run in their execution environment. No paid API request or benchmark execution was performed to prepare this package.
