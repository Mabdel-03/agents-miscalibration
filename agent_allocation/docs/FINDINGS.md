# Findings and caveats — `openrouter_agent_allocation_v1`

Written after the run completed (2026-09-17 14:52 EDT). Numbers cite the sealed artifacts under
`artifacts/openrouter_agent_allocation_v1/`; nothing here is hand-edited into a generated report.

## F1. The MMLU-Pro zeros on both primary contrasts are genuine, and fragile

**What the report shows.** `primary_i_peer_vs_private_team_n4_r4` and `primary_ii_cp_minus_private_team_gain_h3`
both estimate exactly `0.0000` for `mmlu_pro` with a zero-width CI (`se = 0`, `p_value = NaN`,
`n_resamples = 10000` in `tables/primary_contrasts.parquet`), while `math500` (−0.030) and `mbpp_plus` (−0.070)
move. `bbh_logic4` is at a 1.000 ceiling, which explains its zeros; `mmlu_pro` sits at ~0.72–0.79, which does not.

**It is not a defect.** Per-item correctness differences between `peer/4/r` and `private_team/4/r`
(`tables/policy_outcomes.parquet`, main partition, `view=current`):

| benchmark | r=1 | r=2 | r=3 | r=4 |
|---|---|---|---|---|
| mmlu_pro | 1 | 2 | **0** | **0** |
| math500 | 2 | 6 | 6 | 3 |
| mbpp_plus | 3 | 7 | 8 | 11 |

MMLU-Pro answers do respond to the protocol — just at rounds 1 and 2, not at the frozen headline cell r=4. The
delivered answer string differs between the two protocols in 83/2000 (item, n, r) cells overall (math500: 214),
and 35/100 items deliver more than one distinct answer across the 100 checkpoints. The mechanism is high
self-consensus: 61/100 MMLU-Pro items give the same answer in all 25 independent root draws (math500: 34/100),
and peer rounds are unanimous across the n=4 slots in 96.7% of (item, round) groups (math500: 75.6%), so a
plurality over four near-identical option letters rarely moves.

**How to read it.** The zero-width interval is a degenerate bootstrap — every per-item difference is exactly 0,
so the resampled SE is 0 — not a precisely estimated null. At the headline cell exactly one item
(`mmlu_pro/11794`) delivers a different letter under the two protocols, and it is wrong under both, so the
correctness difference is 0. One coin flip the other way would have produced a non-zero estimate; at n=3 and n=5
the same comparison gives roughly −0.02 to −0.04. **Do not report MMLU-Pro as provably unaffected by peer
communication.** `analyze` now labels such rows (`DEGENERATE_ZERO_NOTE` in `analysis/contrasts.py`).

## F2. `cp/0 - peer/4/1` was a tautology reported as an exploratory result

`protocols/grid.py` sets `cp[0] = peer[CP_TEAM_SIZE, CP_PEER_ROUND]`, i.e. **cp/0 is the peer/4/1 checkpoint**.
The exploratory row `cp/0 - peer/4/1` therefore reads 0.000 with a zero-width CI on all four benchmarks by
construction — a definition, not a measurement — and the 14:23 report listed it beside real comparisons with
nothing to distinguish it. `analysis/contrasts.py:identity_note` now detects any contrast whose terms cancel
(treating `cp/0` and `peer/4/1` as the same term) and labels the row and the report section; regression tests are
in `tests/unit/analysis/test_m2_statistics.py`. The reported primary contrasts are unaffected.

A related structural zero is expected and harmless: `sample/n`, `private_team/n/0` and `peer/n/0` share the same
roots (`grid.py:188`), so every r=0 peer−private_team difference is 0 on every benchmark.

## F3. BBH four-subtask subset is at ceiling

`bbh_logic4` delivers 1.000 at essentially every checkpoint (pass@2 = 1.000 on the high panel), so no protocol
contrast can move it and its zeros carry no information. This is the documented zero-shot four-subtask subset,
not full BBH; it should be replaced with harder subtasks in any follow-up.

## F4. Delivered accuracy falls well short of oracle pass@k

On the main panel, oracle `pass@25` reaches 0.948 macro while the achievable readout (`vote@k` for text,
`public_select@k` for MBPP+) plateaus near 0.87 — a selection gap of roughly 7 points that no amount of extra
sampling closes. MMLU-Pro is the widest: pass@25 = 0.875 against vote@25 = 0.792 on the high panel. The
bottleneck in this study is selection, not generation.
