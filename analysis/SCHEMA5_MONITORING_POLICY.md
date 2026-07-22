# Schema-5 Sweep Monitoring Contract

The production monitor has three independent cadences.  Each report is written
atomically beneath `.dispatcher-schema5-v1/monitoring/`; alert state remains in the
control plane's append-only alert journal.

| Cadence | Scope | Required checks |
|---|---|---|
| 5 minutes | live health | controllers and successors, heartbeat age, scheduler mapping and holds, endpoint/fleet health, disk headroom, new corrupt/permanent/retryable failures, QID progress and starvation |
| 6 hours | semantic progress | all three manifest/policy pins, completion states, validated top-level QIDs, top-level and auxiliary censor accounting, jointly stratified QID throughput, generation-scoped 28-day projection |
| daily | full acceptance audit | all 22,680 cells and 4,524,660 expected QIDs, zero malformed/duplicate/unexpected/stale-ingested records, every censor represented once under its own denominator |

Run `scripts/schema5_monitor.py --cadence health`, `--cadence semantic`, or
`--cadence daily`.  `--persist` writes the report, records a successful QID-throughput
sample only after a fully successful poll, and raises/resolves durable alerts through
`slurm/schema5_control.py`.  Add `--send-email` to deliver newly raised alerts to the
address frozen in `control.json` (default `mabdel03@mit.edu`).  Report generation without
`--persist` is read-only.

Throughput is measured in validated top-level QID outcomes, never completed-cell count.
At least 48 continuous hours of one fleet generation are required before the ETA gate is
decisive.  Every unfinished scientific stratum must have positive observed throughput,
the overall rate must be at least 161,595 QIDs/day, and the jointly projected remaining
work must finish within 28 days.  A pause or material fleet change closes the epoch; an
ordinary controller successor does not reset it.

Censors are outcomes, not dropped rows.  Top-level `completed`, `length_censored`, and
`protocol_censored` QIDs partition the validated top-level denominator exactly once.
Auxiliary self-consistency draws use a separate denominator and are never added to
top-level QID throughput.
