# Package validation

Performed while preparing the handoff, without network inference or benchmark execution:

- 20 standard-library unit tests passed for budget arithmetic, reference-DAG counts/ancestry, explicit root reuse, synchronous peer dependencies, immutable identity, public-task field separation, and preset panel balance.
- Shared contracts and both helper scripts compile.
- Main reference graph: 125 generation nodes per item; high-allowance graph: 123; reduced replication: 57; main plus CP/POOLED: 138.
- Default profile has 67,040 generation requests before pilot and retries, including optional replication.
- At the explicit 2,500-input-token/call assumption, balanced150 estimates $129.76 credits including $5 pilot and $10 contingency. Lean100 estimates $86.54 including the same allowances.
- Doubling assumed input tokens to 5,000 makes the full default profile exceed the $135 credit ceiling ($147.89 estimated). The mandatory pilot forecast and optional-panel admission gate are therefore important. The optional model check must be omitted or future balanced waves reduced when actual costs do not fit.

These tests validate the supplied planning helpers and interface seed. They do not validate OpenRouter behavior, a production scheduler, actual dataset versions, sandbox security, graders, model accuracy, or real wall-clock runtime. Those are explicit release gates for the coding agents.
