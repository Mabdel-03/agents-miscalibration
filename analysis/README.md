# Analysis

This directory contains the analysis helpers used after a run has been aggregated from
raw per-question JSONL into a tidy per-cell parquet table.

## Authoritative and Supplementary Data

The default notebook cache is now restricted to the homogeneous schema-5 production
contract.  A refresh accepts exactly these frozen manifests together:

- `full_sweep_schema5_v1`
- `full_sweep_agent_counts_schema5_v1`
- `full_sweep_agent_count_7_schema5_v1`

Each run must have a valid artifact-policy sidecar binding its manifest, benchmark
contracts, release, environments, and model contract.  Only semantically complete
schema-5 cells are ingested.  Unmanifested directories are forbidden.  The cache
manifest records every manifest, benchmark, and policy hash so analyses can verify the
exact source contract.

The preserved v1 runs are supplementary mixed-protocol evidence.  They can never enter
the default cache and must be requested explicitly:

```bash
ANALYSIS_MODE=supplementary-legacy bash analysis/refresh.sh
```

This writes `analysis/cache/supplementary_legacy/` and labels every table and its cache
manifest `mixed_protocol=true`.  To include pre-trim/stale cell directories, additionally
set `INCLUDE_UNMANIFESTED=1`; this option is rejected in primary mode.  Schema-less rows
retain historical reasoning counts only in columns named `legacy_or_mixed` or
`legacy_nonexact`; exact-token columns are null for those rows.

The manifested supplementary item and agent tables retain every canonically valid QID
from `complete`, `partial`, `active`, and `retryable` cells.  Each row carries its cell's
completion state, expected/valid/missing QID counts and exact QID sets.  Only
semantically `complete` cells enter `cells_dedup_v1.parquet`; partial progress is never
silently promoted to a complete-cell estimand.

Discarded response-protocol evidence is kept in two separate, scientifically excluded
tables: `incident_items_scientifically_excluded_v1.parquet` and
`incident_agents_scientifically_excluded_v1.parquet`.  Every incident, reset marker,
archived artifact, and dispatcher-evidence hash is verified before publication.  The
tables retain all 1,419 historical incident QIDs and identify whether each row was one
of the 355 outcomes already sealed before the pre-repair snapshot or one of the 1,064
pre-repair-active outcomes newly sealed by cleanup.  The frozen-baseline acceptance is
therefore exactly 888,068 active QIDs + 1,064 newly sealed QIDs = 889,132; the older 355
remain inspectable but are outside that frozen denominator.

Cache refreshes are fail-closed and generation scoped.  An in-progress marker withdraws
the previous manifest before any parquet is replaced.  The new manifest, which binds
every parquet's SHA-256 and byte size, is published last; notebook loaders refuse an
in-progress generation.  Call `nb_lib.data.verify_cache_integrity()` when a full
cryptographic read-back is required.

`analysis/refresh.sh` resolves the immutable release and harness from
`$ASYS_RESULTS_ROOT/.dispatcher-schema5-v1/control.json`, revalidates those pins through
the frozen control implementation, and runs the frozen ingest script with `python -I`.
The current checkout is only the default cache-output location. Set
`SCHEMA5_CONTROL_STATE_DIR` only when the authoritative control directory lives at a
nondefault path; a missing or invalid control contract aborts before any cache mutation.

## Aggregate A Run

```bash
python scripts/aggregate_results.py --run-id <run_id> --out analysis/<run_id>.parquet
```

The parquet has one row per completed cell. Nested calibration, efficiency, and prompt
quality dictionaries are stored as JSON strings in `calibration_json`, `efficiency_json`,
and `prompt_quality_json`. Top-level QID and auxiliary self-consistency censor counts use
separate denominators. A primary `mean_reasoning_tokens` value is present only when every
top-level topology QID terminates; a terminating-row observation is exposed only under the
explicit `mean_reasoning_tokens_completed_only` name. Auxiliary-only censoring remains a
separate health signal and does not null that primary topology statistic.

## Runnable Examples

Inspect the exact cell grid that a config produces:

```bash
PYTHONPATH=src python analysis/examples/inspect_sweep.py \
  --config configs/full_sweep.yaml
```

Expected output: total cell count, counts by model/topology/context/prompt/reasoning/
benchmark/seed, and the first few deterministic cell IDs.

Summarize an aggregated parquet:

```bash
PYTHONPATH=src python analysis/examples/summarize_run.py \
  --parquet analysis/<run_id>.parquet \
  --system-conf final_producer_logprob
```

Expected output: cell count, total questions, mean accuracy, mean per-agent ECE, mean
system ECE, Delta ECE, and topology-grouped means.

Plot a reliability diagram for one completed cell:

```bash
PYTHONPATH=src python analysis/examples/plot_reliability.py \
  --run-id <run_id> \
  --cell-id <cell_id> \
  --signal per_agent \
  --out analysis/reliability_<cell_id>.png
```

For system confidence, use `--signal system:<confidence_key>`, for example
`system:final_producer_logprob`.

The examples fail with clear missing-file messages if the requested run/cell/parquet has
not been produced yet.

## Fitting Utilities

`fitting.py` provides:

- `power_law_fit(x, y)`: `y = a * x^b` via log-log least squares.
- `kim_regression(df, target)`: standardized OLS in the spirit of Kim et al.'s scaling
  equation, regressing a metric on capacity, capacity squared, context-share rank, prompt
  complexity, and topology indicators with 5-fold CV R2.

## Preserved Legacy Notebook Series

The existing four notebooks and their rendered tables/reports were developed against the
partial v1 rollout. They are preserved in the pre-repair recovery snapshot, excluded
from the immutable production tag, and are supplementary mixed-protocol evidence only.
Their helper module refuses the primary cache and reads
`analysis/cache/supplementary_legacy/` instead. Refresh that cache explicitly:

```bash
ANALYSIS_MODE=supplementary-legacy bash analysis/refresh.sh
```

No notebook is yet designated as a primary schema-5 analysis. The primary cache is
available for integrity and progress analysis during the run; confirmatory downstream
notebooks must be frozen only after representative coverage and the final 22,680-cell
acceptance audit.
