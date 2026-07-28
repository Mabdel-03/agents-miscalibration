# Documentation Index

- **[EXPERIMENT.md](EXPERIMENT.md)**: research-facing dossier. Objectives, research
  questions, experimental grid, model/agent/benchmark methods, diagrams, and how to read
  the results.
- **[AXES.md](AXES.md)**: detailed design of the four scaling axes, including the
  knob-vs-measured-attribute split and the two-call ECE protocol for reasoning.
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: module map, execution model, server/cell
  orchestration, and end-to-end data flow for one cell.
- **[DATA_SCHEMA.md](DATA_SCHEMA.md)**: exact `results.jsonl`, `meta.json`, and tidy
  analysis-table fields.
- **[SCHEMA5_V12_RECOVERY_RUNBOOK.md](SCHEMA5_V12_RECOVERY_RUNBOOK.md)**: authoritative
  v1.2-r5 recovery, immutable-release, readiness, relaunch, staged-ramp, and
  incident-control procedure.
- **[SCHEMA5_RECOVERY_RUNBOOK.md](SCHEMA5_RECOVERY_RUNBOOK.md)**: retired v1.1-r1
  recovery record; retained for forensic reproducibility and never to be executed.
- **[OPERATIONS.md](OPERATIONS.md)**: development and retired chunk-driver procedures,
  retained for legacy-run forensics rather than schema-5 production.
- **[SCHEMA5_RELEASE.md](SCHEMA5_RELEASE.md)**: exact-tag worktree, independent Conda
  copy, package-provenance, and immutable release-freezing runbook.
- **[RESULTS.md](RESULTS.md)**: validation milestones, pilot numbers, and live full-sweep
  status.
- **[figures/](figures/)**: editable Mermaid diagram sources and exported SVG assets used
  by the experiment dossier.

Top-level references:

- [../README.md](../README.md): concise project overview and quickstart.
- [../analysis/README.md](../analysis/README.md): aggregation workflow and runnable
  analysis examples.
- [../CONTRIBUTING.md](../CONTRIBUTING.md): development workflow and extension invariants.
- [../env/SETUP.md](../env/SETUP.md): conda environment setup.
