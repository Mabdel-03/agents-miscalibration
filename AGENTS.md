# Guide for agents

**Pass@K frontier investigation** (forced reasoning budgets, depth against width, allocation selectors): start at
[`experiments/pass_at_k_frontier/README.md`](experiments/pass_at_k_frontier/README.md). It maps all five studies to their
results, tables, reports and code, and lists the mistakes that have produced wrong numbers.

Everything else in this repository, the earlier agent-scaling and calibration harness, is described in
[`README.md`](README.md).

## Rules

- **Do not run paid stages.** Commands with `--allow-paid` spend real money through OpenRouter. Analyses can be redone from
  the committed tables.
- **Do not publish benchmark text.** This repository is public. Raw prompts and transcripts are deliberately not in it (see
  `experiments/pass_at_k_frontier/RAW_DATA.md`) and must not be copied here.
- Paths under `/orcd/` and the Slurm scripts refer to the MIT ORCD cluster where the studies ran. They will not exist on your
  machine.
