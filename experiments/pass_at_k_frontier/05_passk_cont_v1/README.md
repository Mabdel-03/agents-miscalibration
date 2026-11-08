# Study 5: `openrouter_passk_cont_v1`

**The last study in the investigation.** It asks whether study 3's "one deep call wins" holds on fresh problems, and whether
it can be turned into per-problem allocation rules.

**Run.** Core on 2026-09-25, 14:13-20:07 EDT; extensions and package finished 2026-09-26 01:06. Same model and protocol as
study 3. Pre-registered, with held-out predictions sealed before any held-out label was released. $131.56 of a $150 cap.

**Read [`results_package/README.md`](results_package/README.md)** for the full account. It is the entry point for this
study, and it opens with the main findings. `results_package/DATA_DICTIONARY.md` documents every column of every CSV.

## Findings

T200 uses 200 never-used problems and **mixture** weights (AIME .10, HMMT .10, Omni-MATH .50, GPQA .30).

| Experiment | Finding | Status |
|---|---|---|
| E1 | `1x16` 0.705 against locked `4x4` 0.580: -12.5 pts [-18.4, -7.0], Holm p = 0.0008 | Confirmatory |
| E0 | Study 3's headlines reconcile: 79.2 and 78.5 are one statistic on two cohorts | Audit |
| E2 | Per-problem depth-response fits predict fresh draws (plurality Brier 0.066-0.067 against 0.102 shallow-only) | Descriptive |
| E5 | Mechanism-informed predictor P2 is *worse* than feature-based P1: +0.009 [+0.001, +0.018], Holm p = 0.034, under mixture weights. No predictor improved allocation | Confirmatory |
| E8a | The model picking its own allocation chose `1x16` for only 42% of problems: 0.680 and 0.665 against 0.705, Holm p = 0.17 and 0.099 | Post-hoc addendum |
| LEX | Under accuracy-then-consistency-then-latency, per-problem choice does not pay on an honest split-half check; `1x25` (0.836) is the best fixed option on C48 | Exploratory |
| Selector comparison | No statistical, LLM or heuristic selector has accuracy significantly above `1x16` on either panel | Exploratory |
| E8b | Telling the LLM the objective made it worse: 0.610 and 0.593 on T200 | Exploratory |

## Panels

| Panel | Items | Use |
|---|---|---|
| L96 | 96, study 3's main panel | E0, E1 lock, E5 training |
| C48 | 48 from L96 | E2 fitting, LEX banks |
| V48 | The same 48 with new draws | E2 validation, held out until the seal |
| T200 | 200 new (20 AIME, 20 HMMT, 100 Omni-MATH, 60 GPQA) | E1 primary test, E5 prospective test, selector scoring |

## Caveats

- State your weights. Mixture and macro can change conclusions, including P2 against P1.
- `4x4` was locked on the development cohort (n = 95 after an exclusion) under the sealed procedure.
- E8a's prompt was written with T200 outcomes known, as its pre-registration discloses.
- 1x25 reads 0.836, not 0.846. An early LEX pass left 13 invalid 25A draws unknown instead of wrong; this was fixed before
  the final export (`results_package/README.md`, section 10).
- `results_package/MANIFEST.csv` describes the full original package. Eight prompt files and 56 cells differ in this public
  copy; [`REDACTIONS.csv`](../REDACTIONS.csv) lists them. The package README still mentions those prompt files
  (`results_package/prompts/WITHHELD.md` explains) and cites cluster paths (`R/`, `W/`, `MEM`), defined under its "Path
  conventions".

## Files

| Path | Contents |
|---|---|
| `results_package/` | Everything needed for analysis: README, data dictionary, `csv/00_study` to `csv/11_costs`, pre-registrations, prompts, figures |
| `report/` | Two-page core report (E0 audit and the primary tests) and its numbers. The package README is the full account |
| `sealed/` | The seal, predictions and fitted models fixed before any held-out label: `SEAL_predictions.json`, `v48/`, `t200/` |
| `scores/`, `scored/` | E2 and E5 scores, menus, the primary test (`scored/primary.json`) |
| `gates/` | Every gate record |
| `e0/`, `e8a/`, `lex/` | Audit outputs and extension summaries |
| `manifests/` | Item ids per panel and split |

Design: [`agent_allocation/docs/passk_cont/`](../../../agent_allocation/docs/passk_cont/) (`PREREG.md`, the E8a, E8b and LEX
addenda, `00_contract.md`, `RUNBOOK.md`). Spec:
[`remaining_experiments_spec.md`](../../../expermt_agent_specs/Pass@K-Frontier-Cont/remaining_experiments_spec.md). Code:
`src/agent_allocation/passk/`. Raw transcripts are not on GitHub, see [`RAW_DATA.md`](../RAW_DATA.md).
