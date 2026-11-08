# Iso-compute allocation report (`openrouter_agent_allocation_v1`)

Re-keys the sealed run by compute budget `C = B * R * A` instead of by protocol
family, and asks which way of spending a fixed `C` pays: more agents (width),
more rounds (depth), a longer reasoning allowance in one call, or simply more
independent samples read out by a vote.

`report.pdf` is the deliverable: abstract, methods, results, interpretation.
13 pages, 11 figures, every result reported per benchmark.

## Headline

- At matched compute the allocation shape barely matters. Of 70 comparisons
  against independent sampling per benchmark, none survives Benjamini-Hochberg
  correction; the smallest macro p-value is 0.017.
- Every allocation plateaus near what two samples and a vote deliver, while
  oracle pass@k keeps climbing. Unrealised headroom grows to 6.7 macro points
  (9.0 on MMLU-Pro). On MBPP+ the public-test readout declines with budget.
- Peer exposure costs accuracy and the damage deepens with rounds: 5.0 macro
  points at B=5, r=4, carried by MBPP+ at 10 points.
- One `deep/8` call matches the best orchestrated configuration to within 0.3
  macro points, using one generation at a twelfth of the cost.

The binding constraint is selection, not generation.

## The prescription

Levers ranked by what they are worth at `C = 25A`, macro over the three
informative benchmarks:

1. **Verification.** A reranker at pairwise AUC 0.70, applied to candidates the
   run already paid for, beats all 95 allocations; AUC 0.80 delivers 0.876
   against an oracle of 0.917. Today's plurality vote is equivalent to AUC 0.63.
2. **Per-draw allowance `A`.** The only generation-side axis that touches the
   binding quantity. Halves the MATH-500 gap, neutral on MMLU-Pro, negative on
   MBPP+, so verify per benchmark.
3. **Fix broken selectors.** MBPP+ public-test selection falls below a random
   draw from k=8 onward.
4. **Consolidate rather than communicate.** Private teams raise the pool's
   correct share from 0.795 to 0.843. The archive readout is free and recovers
   +0.037 on `peer/5/4`.
5. **More draws at fixed `A`.** Close to nothing.

Appendix A of the report specifies a validation run that tests lever 1 against a
real judge on the stored candidates: 400 items, no regeneration, about $2.19.

## Provenance

Source of truth is branch `allocation-v1`, commit `84e3654`, at
`agent_allocation/docs/isocompute/` in the worktree
`/orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1`. This directory is
a copy for convenience in the main checkout; edit it there and re-copy, or keep
the two in sync deliberately.

Inputs are the sealed run tables, read only, at
`/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1/tables/`.

## Build

```bash
./build.sh                 # analysis -> figures -> lint -> compile -> log check
./build.sh --self-check    # compute-accounting checks only
RUN_DIR=/path/to/artifacts/openrouter_agent_allocation_v1 ./build.sh
```

`build.sh` resolves the interpreter to the allocation-v1 worktree venv (it needs
the `agent_allocation` package) and uses TinyTeX on `/home/mabdel03/.local/bin`.
Override with `PY=...` if the worktree moves.

## Layout

| file | role |
|---|---|
| `isocore.py` | loading, the `C = B*R*A` accounting, the self-check, palettes |
| `isoframe.py` | master frame: paired item-clustered bootstrap over every checkpoint |
| `isostyle.py` | shared figure style |
| `fig_*.py` | one module per figure group |
| `isocompute.py` | driver: checks, 11 figures, `numbers.json`, `tables/*.tex` |
| `check_numbers.py` | asserts every number in the prose traces to `numbers.json` |
| `report.tex` | the report source |

## What the build verifies

1. Derived `C` equals the sealed `requested_reasoning_tokens` on every panel, and
   `B*R` equals the sealed generation count except for `deep` (one call by design).
2. Recomputed accuracies match the published run report: `sample/1` 0.835,
   `private_team/5/4` 0.8875, `peer/4/4` 0.855, `deep/8` 0.885, oracle pass@25 0.9375.
3. `private_team/n/0` = `peer/n/0` = `sample/n` on all 400 items, for every `n`.
4. The two frozen contrasts are quoted from `primary_contrasts.parquet`, not recomputed.
5. No em dashes, every quoted number traces to `numbers.json`, no undefined
   references and no overfull boxes.
