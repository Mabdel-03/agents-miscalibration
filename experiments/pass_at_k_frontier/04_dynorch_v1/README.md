# Study 4: `openrouter_dynorch_v1`

**Question.** Does an off-the-shelf harness that decides for itself when to spawn sub-agents beat study 3's fixed rule, with
the same model and the same completion-token budget `C = k*A`?

**Run.** 2026-09-23 to 09-24. Qwen3.8-27B on study 3's 96 items at rungs k = 1, 2, 4, 8, 16, 25, with budgets enforced by a
token-metering proxy.

| Arm | What it is |
|---|---|
| `H0` | Plain capped call, no harness (control) |
| `H1` | OpenClaude 0.13.0, offered delegation tools |
| `H3` | Codex CLI 0.156.1, offered delegation tools |
| `H1c`, `H3c` | Root call capped, to create room for delegation (E0) |
| `H1n`, `H3n` | Spawn tool removed (E2) |

Orchestration tools only: no shell, code, web or files. Ledger total $59.86 of a $95 ceiling.

## Findings

Macro accuracy at C = 16A.

| Arm | Accuracy | Against `deep/16` (0.792) | Holm p |
|---|---|---|---|
| H1 OpenClaude | 0.667 | -12.5 pts [-21.3, -5.2] | 0.004 |
| H3 Codex | 0.646 | -14.6 pts [-23.4, -7.1] | 0.003 |
| H0 plain capped call | 0.719 | -7.3 pts [-15.5, -0.3] (descriptive) | none |

- **Harnesses beat sampling.** H1 and H3 beat `sample/16` (0.417) by +25.0 and +22.9 pts, and beat oracle pass@16 (0.580).
- **Harnesses did not delegate in the core.** 0 of the 1,152 core harness runs spawned a sub-agent. H1 made one model call per
  run. H3 made a second call only after `update_plan`, in 12 runs.
- **The gap has two parts.** A natural call stops short of the cap (realized share about 0.6 at k = 16, where `deep/16` is
  forced to 1.0), and the harness scaffolding itself costs accuracy.

| Extension | Runs | Spend | Result |
|---|---|---|---|
| E0: root call capped | 384 | $3.97 | 41 runs spawned, but the harness mostly accepted the capped root's answer as final (`H1c/16`: 72% of ceiling closes). `H1c/16` is -30.2 pts [-40.6, -20.3] against `deep/16` |
| E1: second repeat | 1,152 | $18.16 | Replicates, with 1 run spawning. Mean of both repeats against `deep/16`: H1 -12.0 pts [-19.4, -5.4], H3 -11.5 pts [-18.8, -5.1]. Agreement between repeats per item 0.76-0.83 |
| E2: spawn tool removed | 384 | $6.45 | 0 spawned. Removing the unused tool **raised** accuracy: `H1n/16` +6.8 pts [+0.2, +13.4] over the mean of both offered repeats |
| E3: OpenClaw | none | none | **Not run** |
| E4: RLM | none | none | **Not run** |

## Caveats

- In `dynorch_runs`, keep `status == "done"`, which leaves exactly one row per cell in every partition. `attempt` counts tries
  within a cell, and earlier tries appear as `transport_failed` rows (16 in main, 5 in the extensions).
- Do not filter on `generation`. It is 2 for eight main cells re-executed after provider stream cuts, because a retried 502 had
  received a terminal no-thinking answer that would otherwise be scored as a model outcome. It is also 2 for every canary row,
  since the canary was re-run after a defect fix. Superseded executions are already out of the table.
- Accounting differs by call type: an E0 `ceiling_close` answer is charged inside `C`, terminal calls are granted extra but
  count 0 in `C`, and the run-level `forced_close_kind` never shows `ceiling_close`, so derive E0 stop categories from
  `dynorch_calls`.
- Pilot and main gates advanced under explicit owner waivers (`gates/owner_waivers.json`): non-delegation was treated as the
  finding, not a defect.
- In `tables/dynorch_runs.parquet`, 207 `final_message` and `cli_result` values are withheld because they quote the question.
  The SEAL files hash the originals, so verifying those seals needs the raw data on the cluster.

## Files

| Path | Contents |
|---|---|
| `report/report.pdf` | Main report (13 pp); `numbers.json` holds every figure; `REPORT_BUILD.md` summarizes it |
| `tables/` | `dynorch_runs`, `dynorch_calls`, `policy_outcomes_dynorch`, `policy_costs_dynorch`, the stage `SEAL*.json`, and `SCHEMA.md` |
| `exports/` | OpenClaude results workbook and CSVs |
| `gates/` | Gates, reconciliations, waivers, run decisions |
| `manifests/` | The plan: every run per stage |
| `notes/` | Pre-launch review findings |
| `run_report/PILOT_REPORT.md` | Pilot report |

Design: [`agent_allocation/docs/dynorch/`](../../../agent_allocation/docs/dynorch/), especially `00_contract.md`. Code:
`src/agent_allocation/dynorch/`. Raw transcripts are not on GitHub, see [`RAW_DATA.md`](../RAW_DATA.md).
