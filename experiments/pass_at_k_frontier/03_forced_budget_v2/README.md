# Study 3: `openrouter_forced_budget_v2`

**Question.** Does study 1's conclusion survive when every generation spends exactly its reasoning budget?

**Run.** 2026-09-18, main run 10:27-12:06 EDT. Qwen3.8-27B, A = 1,024, temperature 1.0, top_p 0.95, top_k 20. 96 items, 24
each from AIME 2025+2026, HMMT February 2025+2026, Omni-MATH difficulty 7+ and GPQA-Diamond. 96 configurations. All 13,344
main generations realized exactly their budget. Two provider shards: Mancer (58 items) and Parasail (38). $70.00.

## Findings

Macro accuracy over the four benchmarks.

| Finding | Value | Source |
|---|---|---|
| `deep/16` against `sample/16` | 0.792 against 0.417, +37.5 pts [+28.1, +46.9] | `tables/policy_outcomes.parquet`, `report/report.pdf` |
| Oracle pass@16, best revision chain, best team at 16A | 0.580, 0.525, 0.479 | `report/report.pdf` |
| `deep/4` against `sample/25`, at far less compute | 0.610 against 0.427, +18.3 pts | `tables/policy_outcomes.parquet` |
| `deep/25` against `deep/16` | 0.750 against 0.792 | same |

At every budget that includes a deep call, only the deep call delivers more than pass@k of 1,024-token samples. Every point
on the dollar frontier is a deep call.

- **GPQA-Diamond is the exception:** depth and serial revision tie there, so it behaves like a selection problem.
- **Selection cannot close the gap from the width side:** a perfect verifier over 16 samples delivers 0.5625, still 22.9 pts
  below `deep/16` ([`verifier_sweep.csv`](../../../agent_allocation/docs/forced_v2_report/tables/verifier_sweep.csv)).
- **Looping is not failure:** deep calls loop more as they get longer (28% at 16A), but looping calls are the ones that had
  already solved the problem (0.96 against 0.73 at 16A).
- **Peer exposure does not hurt here, as it did in study 1.** At n = 4, r = 4 it is +4.6 pts [-2.2, +11.7], p = 0.195, so not
  significant; across 16 cells the mean is +1.4 pts, with 11 positive (`report/numbers.json`, `comm`). The same row in
  `tables/primary_contrasts.csv` reads as `partial` with no interval, because one item is unknown.

## Caveats

- `run_report/RUN_REPORT.md` is stamped **INTERIM** for two documented reasons, and nothing further was collected afterwards.
  80 checkpoint rows have no known outcome: 49 `current` plus 31 `archive`, from Math-Verify timeouts at the pre-registered
  5 s limit. All 49 `current` rows are on `hmmt/2025/18` and `omni_math_hard/473`. And the conservative ledger ($72.26,
  including $2.19 charged at full reservation) exceeds the provider's figure ($70.33).
- Study 5's README classifies this study's analysis as descriptive. Study 5 is the confirmatory test.
- 79.2 here and 78.5 in study 5 are the same statistic on the 96-item and 94-item cohorts.
- `data/DATA_DICTIONARY.csv` covers 31 columns, not all of them. `tables/SCHEMA.md` lists every column with its type.
- No table holds reasoning text. `text_available` in `tables/effective_depth.parquet` only says the raw text was readable when
  that table was built.

## Files

| Path | Contents |
|---|---|
| `report/report.pdf` | Main report (23 pp, 19 figures); `numbers.json` holds its figures; `REPORT_BUILD.md` summarizes it |
| `data/` | **Easiest entry point.** Flat CSVs with `DATA_DICTIONARY.csv`: `forced_budget_v2_item_config.csv` (item x configuration x view), `forced_budget_v2_generations.csv` (one row per generation) |
| `tables/` | Sealed merged tables, with `SCHEMA.md` |
| `run_report/`, `figures/` | Generated run report and figures |
| `gates/<shard>/` | Per-shard gate records and run decisions |
| `calibration.json`, `merge_manifest.json` | Natural-length calibration; merge provenance |
| [`forced_v2_report/tables/`](../../../agent_allocation/docs/forced_v2_report/tables/) | Report-specific tables: verifier sweep, difficulty strata and others |

Code: [`configs/forced27b.json`](../../../agent_allocation/configs/forced27b.json), `src/agent_allocation/shards.py`,
`calibrate.py`, `analysis/merge.py`. Raw transcripts are not on GitHub, see [`RAW_DATA.md`](../RAW_DATA.md).
