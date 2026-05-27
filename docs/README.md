# Documentation index

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — module-by-module map, the execution model
  (servers + chunked cell workers + filesystem registry), and the end-to-end data flow of
  one experiment cell.
- **[AXES.md](AXES.md)** — design of all four scaling axes (capacity, context-sharing,
  prompt complexity, reasoning), the knob-vs-measured-attribute split, and how they combine
  in the sweep. Includes the two-call ECE protocol for the reasoning axis.
- **[OPERATIONS.md](OPERATIONS.md)** — the runbook: cluster facts, env setup, HF auth,
  launching servers + the chunked full sweep, monitoring, resuming, server keepalive,
  cost projection, and the gotchas already fixed.
- **[DATA_SCHEMA.md](DATA_SCHEMA.md)** — exact `results.jsonl` / `meta.json` fields and the
  tidy analysis-table columns.
- **[RESULTS.md](RESULTS.md)** — validation milestones, pilot numbers, and live full-sweep
  status. (Living document.)

Top-level: [README.md](../README.md) (overview + quickstart),
[CONTRIBUTING.md](../CONTRIBUTING.md) (dev workflow, invariants, how to add an axis),
[env/SETUP.md](../env/SETUP.md) (conda envs).
