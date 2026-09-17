# Lead / integration agent

## Mission
Implement and run `openrouter_agent_allocation_v1` from `MASTER_SPEC.md`. Coordinate eight coding agents in parallel, integrate their modules, and control the only paid dispatcher. The supplied Python helpers are structural/planning references, not the production harness. Never claim full implementation from passing their tests alone.

## Read first
`MASTER_SPEC.md`, `SOURCES.md`, `configs/balanced150.json`, and `shared/contracts.py`. Inspect the actual repository and existing tests before choosing the final namespace. Do not assume a previous assistant's command or filesystem path exists. Preserve previous studies and raw data.

## First actions
Commit the shared contract skeleton and configuration in a new isolated branch. Create one Git worktree per agent, then send agents 01–08 their matching brief plus the master spec and contracts. Ensure they can import contracts and build against mocks immediately. Only you may edit shared contracts, dependency lock files, CLI root, project configuration, or top-level integration tests without coordination.

Own `pyproject.toml`, dependency locking, `src/agent_allocation/contracts.py`, `src/agent_allocation/cli.py`, configuration compilation, and top-level run authorization. Use the namespace already present only when it can preserve this specification's semantics. Copy the supplied reference contracts into the actual package, complete missing types, and publish a small interface fixture rather than waiting for every implementation.

## Paid-operation policy
Only one coordinator process receives the OpenRouter key. All agent unit tests and ordinary smoke tests are offline/mocked. A live command requires `--allow-paid`; the default mode is a plan, not a generation run. Pilot and main share one cumulative ledger. Main requires passing pilot gates and frozen manifests. A configured $135 is the entire study cap, never a per-worker or per-phase allowance. No automatic credits, higher key caps, provider switching, or model replacement.

## Integration order
1. Contracts + 01 adapter + 02 ledger/scheduler + a text adapter from 04 + 08 smoke tests.
2. Integrate 03 protocols and 06 readouts as soon as their mocks pass.
3. Launch a small authorized development canary while 05 finishes code grading and 07 finishes reports.
4. Require each benchmark's extraction/grading gate before that benchmark's main jobs. Code may not scale without sandbox tests.
5. Freeze config and measured cost forecast, then admit balanced waves. Start the long private chains early. Run remaining coding work and offline analysis beside inference, not before all inference.
6. Reconcile costs, finish selectors/grading, audit completeness, and deliver one sealed report.

## CLI and runbook
Implement the command contract in master section 14. `run --resume` must be idempotent for successful nodes. Compile a live endpoint lock with exact standard-tier provider tags; do not leave placeholder slugs or unpinned dataset revisions. Avoid NFS-backed SQLite or a new distributed framework. One local async dispatcher with separate grader processes is the default.

## Review priorities
Key identity and explicit reuse; nominal versus actual cost; no private-label leakage; correct code selector; valid output headroom; actual concurrency rather than nested episode awaits; unknown-cost requests retained across crashes; reference grader parity.

## Definition of done
The actual production test suite passes, all required conditions have generation/selection/grade rows or explicit statuses, budget is reconciled, and the report states the tested subsets and limits. Publish the runnable command, Git commit, manifest hashes, actual spend, realized throughput, unfinished partitions, and the location of the report. Do not predict a favorable scientific result or claim cross-family transfer from the optional within-Gemini check.
