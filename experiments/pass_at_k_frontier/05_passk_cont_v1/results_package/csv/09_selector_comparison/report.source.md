# passk_cont_v1 LEX selector comparison (post-hoc, exploratory)

Generated 2026-09-26T04:13:23+00:00 by `python -m agent_allocation.passk.selcmp`. Objective: accuracy -> consistency -> latency (incl. selector overhead), ties within 0.02. C48 is scored with the per-item bank estimates on the **no-diagnostic bank** (each item's five diagnostic draws, which P1-P3, the heuristic and the A1/B1 LLM arms read, are removed from its A-level bank; the previous full-bank numbers are kept in the bank-sensitivity section) as expected values, macro over benchmarks; T200 with realized outcomes (one realization per item, so consistency is not measurable and the ranking is accuracy -> latency).

Overheads: `diag` = max of the item's five diagnostic call latencies (macro mean 46 s on C48, 33 s on T200); `llm` = the decision call latency summed over retries (E8a A0: one decision; E8b B0: averaged over the item's 5 decisions); `diag+llm` = both (A1, B1). Mean decision latency: A0/L96 16 s, A0/T200 16 s, A1/L96 16 s, A1/T200 16 s, B0/C48 19 s, B0/T200 18 s, B1/C48 19 s, B1/T200 18 s.

LLM selectors: `LLM:A0/A1` = E8a (asked for the most accurate of the 5 M5 options, one decision per item). `LLM:B0/B1` = E8b (PREREG_E8b_lex_llm.md: told the LEX objective, the 15-option M15 menu with typical times, 5 decisions per item; each item's score is the mean over its decisions, and on C48 its consistency is p_sys^2 + (1 - p_sys)^2 of that mean; invalid after one retry -> 1x16). Prompt sha256 78090c1910f9; 2530 call attempts.

Statistical selectors are cross-fitted (5 problem-grouped folds over L96 = the E5 CV fold assignment; each C48 item's predictions come from a fit that never saw it). The refit on all of L96 reproduces the sealed T200 predictions (max |diff| = 0 over 10800 cells).

`LEX rank` ranks the deployable selectors (fixed M15 options, statistical, heuristic, LLM A0/A1/B0/B1); `ref` rows (1x25, oracles, split-half) are references outside that competition. `P(win)` = share of bootstrap replicates in which the selector is the population LEX winner; on T200 it is given under both the mixture and the macro weights, and the two rankings differ.

## Headline: statistical vs LLM selectors, by panel and weighting

Generated from the numbers in this report (`report.json` → `headline`). The answer depends on the panel and, on T200, on the benchmark weighting; read each block on its own.

**C48, macro weights (expected per-item values on the no-diagnostic bank; ranking uses accuracy -> consistency -> latency).** Best deployable accuracy 0.785, so the LEX tie band starts at 0.765; always-1x16 is 0.785. Inside that band the best consistency is 0.895, so latency decides among 11 rows: fixed:1x16, stat:P0/M15, stat:P0/M5, stat:P2-conc/M15, stat:P3/M15, stat:P3/M5, stat:P2-conc/M5, stat:P1/M15, stat:P1/M5, stat:P1-conc/M15, stat:P1-conc/M5. LEX winner: fixed:1x16.

- Statistical models (accuracy and contrast shown for the rows on the LLMs' menus; band counts and ranks over all of the predictor's rows):
  - P0: 2 of 2 rows inside the top band, 2 in the LEX band, 2 of 2 within 0.02 of 1x16; stat:P0/M5 0.785 (C 0.886), vs 1x16 no difference on any item; stat:P0/M15 0.785 (C 0.886), vs 1x16 no difference on any item; ranks 2-3.
  - P1: 2 of 2 rows inside the top band, 2 in the LEX band, 2 of 2 within 0.02 of 1x16; stat:P1/M5 0.785 (C 0.886), vs 1x16 no difference on any item; stat:P1/M15 0.785 (C 0.886), vs 1x16 no difference on any item; ranks 8-9.
  - P2: 0 of 2 rows inside the top band, 0 in the LEX band, 0 of 2 within 0.02 of 1x16; stat:P2/M5 0.760 (C 0.906), vs 1x16 -2.4 pts [-6.8, +0.7], CI includes 0, p = 0.275; stat:P2/M15 0.763 (C 0.902), vs 1x16 -2.2 pts [-6.3, +0.8], CI includes 0, p = 0.321; ranks 12-14.
  - P3: 2 of 2 rows inside the top band, 2 in the LEX band, 2 of 2 within 0.02 of 1x16; stat:P3/M5 0.777 (C 0.877), vs 1x16 -0.7 pts [-3.3, +1.2], CI includes 0, p = 0.616; stat:P3/M15 0.777 (C 0.877), vs 1x16 -0.7 pts [-3.3, +1.2], CI includes 0, p = 0.616; ranks 5-6.
  - P1-conc: 2 of 2 rows inside the top band, 2 in the LEX band, 2 of 2 within 0.02 of 1x16; stat:P1-conc/M5 0.785 (C 0.886), vs 1x16 no difference on any item; stat:P1-conc/M15 0.785 (C 0.886), vs 1x16 no difference on any item; ranks 10-11.
  - P2-conc: 2 of 2 rows inside the top band, 2 in the LEX band, 2 of 2 within 0.02 of 1x16; stat:P2-conc/M5 0.774 (C 0.895), vs 1x16 -1.0 pts [-3.7, +0.8], CI includes 0, p = 0.483; stat:P2-conc/M15 0.774 (C 0.895), vs 1x16 -1.0 pts [-3.7, +0.8], CI includes 0, p = 0.483; ranks 4-7.
- LLM selectors (E8a A0/A1: asked for accuracy, M5, one decision; E8b B0/B1: told the LEX objective, M15, mean of 5 decisions):
  - LLM:A0 0.731, vs 1x16 -5.4 pts [-9.5, -1.9], CI below 0, p = 0.040, outside the top band, rank 17, P(win) 0.007, C 0.861, latency 207 s.
  - LLM:A1 0.713, vs 1x16 -7.1 pts [-12.9, -2.4], CI below 0, p = 0.053, outside the top band, rank 16, P(win) 0.004, C 0.883, latency 240 s.
  - LLM:B0 0.670, vs 1x16 -11.4 pts [-16.3, -6.8], CI below 0, p < 0.001, outside the top band, rank 21, P(win) 0.000, C 0.785, latency 155 s, choice stability 0.46, 0 invalid -> 1x16.
  - LLM:B1 0.694, vs 1x16 -9.1 pts [-13.1, -5.2], CI below 0, p < 0.001, outside the top band, rank 20, P(win) 0.000, C 0.827, latency 202 s, choice stability 0.50, 1 invalid -> 1x16.
- Reference rows: fixed:1x16 0.785, vs 1x16 --, inside the top band and the LEX band, rank 1, P(win) 0.324, C 0.886, latency 277 s; heuristic/M15 0.751, vs 1x16 -3.3 pts [-7.9, -0.1], CI below 0, p = 0.238, outside the top band, rank 15, P(win) 0.102, C 0.878, latency 262 s; heuristic/M5 0.753, vs 1x16 -3.2 pts [-8.5, +0.0], CI touches 0, p = 0.164, outside the top band, rank 13, P(win) 0.019, C 0.895, latency 273 s.
- Statistical minus LLM accuracy on the E8a allocator's 5-option menu (stat:P0/M5, stat:P1/M5, stat:P2/M5, stat:P3/M5, stat:P1-conc/M5, stat:P2-conc/M5): vs LLM:A0 +3.0 to +5.4 pts, CI above 0 for 5 of 6, below 0 for 0 (includes or touches 0: stat:P2/M5), p 0.040-0.323; vs LLM:A1 +4.7 to +7.1 pts, CI above 0 for 6 of 6, below 0 for 0, p 0.053-0.121.
- Statistical minus LLM accuracy on E8b's 15-option menu (stat:P0/M15, stat:P1/M15, stat:P2/M15, stat:P3/M15, stat:P1-conc/M15, stat:P2-conc/M15): vs LLM:B0 +9.3 to +11.4 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001-0.001; vs LLM:B1 +6.9 to +9.1 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001-0.002.
- Telling the LLM the objective (B minus A; B also has the 15-option menu with typical times and averages 5 decisions): LLM:B0 - LLM:A0: accuracy -6.0 pts [-11.4, -0.8], CI below 0, p = 0.039; C -7.6 pts [-12.4, -3.0], CI below 0, p = 0.005; latency -52 [-81, -23] s | LLM:B0 - LLM:A1: accuracy -4.3 pts [-9.2, +0.5], CI includes 0, p = 0.102; C -9.8 pts [-14.8, -5.0], CI below 0, p = 0.002; latency -85 [-113, -56] s | LLM:B1 - LLM:A1: accuracy -1.9 pts [-6.7, +3.6], CI includes 0, p = 0.499; C -5.6 pts [-9.6, -1.6], CI below 0, p = 0.015; latency -37 [-66, -9] s | LLM:B1 - LLM:A0: accuracy -3.6 pts [-8.4, +1.3], CI includes 0, p = 0.167; C -3.4 pts [-7.6, +0.8], CI includes 0, p = 0.134; latency -5 [-35, +25] s | LLM:B1 - LLM:B0: accuracy +2.4 pts [-1.5, +6.6], CI includes 0, p = 0.284; C +4.2 pts [+0.2, +8.5], CI above 0, p = 0.076; latency +48 [+30, +66] s.
- Ranking: statistical rows rank 2-14; LLM rows rank LLM:A1 16, LLM:A0 17, LLM:B1 20, LLM:B0 21; the best LLM row (LLM:A1) ranks 16; statistical rows ranked below it: none.
- Bootstrap win share: LLM:A0 0.007, LLM:A1 0.004, LLM:B0 0.000, LLM:B1 0.000; highest statistical row stat:P2/M15 0.347.

**T200, macro weights (one realized outcome per item; accuracy -> latency).** Best deployable accuracy 0.798, so the LEX tie band starts at 0.778; always-1x16 is 0.794. Latency decides among the 23 rows in the band. LEX winner: heuristic/M15.

- Statistical models (accuracy and contrast shown for the rows on the LLMs' menus; band counts and ranks over all of the predictor's rows):
  - P0: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P0/M5 0.794, vs 1x16 no difference on any item; stat:P0/M15 0.794, vs 1x16 no difference on any item; ranks 8-11.
  - P1: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P1/M5 0.794, vs 1x16 no difference on any item; stat:P1/M15 0.794, vs 1x16 no difference on any item; ranks 14-17.
  - P2: 0 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P2/M5 0.776, vs 1x16 -1.8 pts [-3.8, -0.1], CI below 0, p = 0.072; stat:P2/M15 0.776, vs 1x16 -1.8 pts [-3.8, -0.1], CI below 0, p = 0.072; ranks 22-25.
  - P3: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P3/M5 0.790, vs 1x16 -0.4 pts [-1.2, +0.0], CI touches 0, p = 0.078; stat:P3/M15 0.790, vs 1x16 -0.4 pts [-1.2, +0.0], CI touches 0, p = 0.078; ranks 3-6.
  - P1-conc: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P1-conc/M5 0.798, vs 1x16 +0.4 pts [+0.0, +1.2], CI touches 0, p = 0.078; stat:P1-conc/M15 0.798, vs 1x16 +0.4 pts [+0.0, +1.2], CI touches 0, p = 0.078; ranks 18-21.
  - P2-conc: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P2-conc/M5 0.794, vs 1x16 no difference on any item; stat:P2-conc/M15 0.798, vs 1x16 +0.4 pts [+0.0, +1.2], CI touches 0, p = 0.078; ranks 12-27.
- LLM selectors (E8a A0/A1: asked for accuracy, M5, one decision; E8b B0/B1: told the LEX objective, M15, mean of 5 decisions):
  - LLM:A0 0.762, vs 1x16 -3.2 pts [-7.6, +0.3], CI includes 0, p = 0.152, outside the top band, rank 28, P(win) 0.207, latency 202 s.
  - LLM:A1 0.731, vs 1x16 -6.3 pts [-11.8, -1.7], CI below 0, p = 0.047, outside the top band, rank 31, P(win) 0.008, latency 233 s.
  - LLM:B0 0.664, vs 1x16 -13.0 pts [-17.7, -8.6], CI below 0, p < 0.001, outside the top band, rank 32, P(win) 0.000, latency 152 s, choice stability 0.43, 6 invalid -> 1x16.
  - LLM:B1 0.666, vs 1x16 -12.8 pts [-16.8, -9.0], CI below 0, p < 0.001, outside the top band, rank 33, P(win) 0.000, latency 184 s, choice stability 0.49, 1 invalid -> 1x16.
- Reference rows: fixed:1x16 0.794, vs 1x16 --, inside the top band, rank 7, P(win) 0.029, latency 285 s; heuristic/M15 0.779, vs 1x16 -1.5 pts [-3.2, -0.2], CI below 0, p = 0.085, inside the top band, rank 1, P(win) 0.429, latency 244 s; heuristic/M5 0.786, vs 1x16 -0.8 pts [-2.1, +0.0], CI touches 0, p = 0.015, inside the top band, rank 2, P(win) 0.195, latency 246 s.
- Statistical minus LLM accuracy on the E8a allocator's 5-option menu (sealed:P0/M5, sealed:P1/M5, sealed:P2/M5, sealed:P3/M5, sealed:P1-conc/M5, sealed:P2-conc/M5): vs LLM:A0 +1.4 to +3.7 pts, CI above 0 for 0 of 6, below 0 for 0 (includes or touches 0: sealed:P0/M5, sealed:P1/M5, sealed:P2/M5, sealed:P3/M5, sealed:P1-conc/M5, sealed:P2-conc/M5), p 0.122-0.513; vs LLM:A1 +4.5 to +6.8 pts, CI above 0 for 5 of 6, below 0 for 0 (includes or touches 0: sealed:P2/M5), p 0.040-0.122.
- Statistical minus LLM accuracy on E8b's 15-option menu (stat:P0/M15, stat:P1/M15, stat:P2/M15, stat:P3/M15, stat:P1-conc/M15, stat:P2-conc/M15): vs LLM:B0 +11.2 to +13.4 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001; vs LLM:B1 +11.0 to +13.2 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001.
- Telling the LLM the objective (B minus A; B also has the 15-option menu with typical times and averages 5 decisions): LLM:B0 - LLM:A0: accuracy -9.8 pts [-14.8, -4.9], CI below 0, p = 0.001; latency -50 [-68, -32] s | LLM:B0 - LLM:A1: accuracy -6.7 pts [-12.9, -0.2], CI below 0, p = 0.050; latency -81 [-102, -59] s | LLM:B1 - LLM:A1: accuracy -6.5 pts [-11.9, -0.6], CI below 0, p = 0.034; latency -49 [-68, -31] s | LLM:B1 - LLM:A0: accuracy -9.5 pts [-14.3, -4.8], CI below 0, p < 0.001; latency -19 [-40, +2] s | LLM:B1 - LLM:B0: accuracy +0.2 pts [-3.7, +4.4], CI includes 0, p = 0.919; latency +31 [+19, +44] s.
- Ranking: statistical rows rank 3-27; LLM rows rank LLM:A0 28, LLM:A1 31, LLM:B0 32, LLM:B1 33; the best LLM row (LLM:A0) ranks 28; statistical rows ranked below it: none.
- Bootstrap win share: LLM:A0 0.207, LLM:A1 0.008, LLM:B0 0.000, LLM:B1 0.000; highest statistical row stat:P3/M15 0.086.

**T200, mixture weights (aime .10, hmmt .10, omni .50, gpqa .30: the study's primary T200 estimand and E8a's pre-registered test; accuracy -> latency).** Best deployable accuracy 0.710, so the LEX tie band starts at 0.690; always-1x16 is 0.705. Latency decides among the 22 rows in the band. LEX winner: heuristic/M5.

- Statistical models (accuracy and contrast shown for the rows on the LLMs' menus; band counts and ranks over all of the predictor's rows):
  - P0: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P0/M5 0.705, vs 1x16 no difference on any item; stat:P0/M15 0.705, vs 1x16 no difference on any item; ranks 7-10.
  - P1: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P1/M5 0.705, vs 1x16 no difference on any item; stat:P1/M15 0.705, vs 1x16 no difference on any item; ranks 18-21.
  - P2: 0 of 4 rows inside the top band, 0 of 4 within 0.02 of 1x16; sealed:P2/M5 0.675, vs 1x16 -3.0 pts [-6.0, -0.5], CI below 0, p = 0.045; stat:P2/M15 0.675, vs 1x16 -3.0 pts [-6.0, -0.5], CI below 0, p = 0.045; ranks 26-29.
  - P3: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P3/M5 0.700, vs 1x16 -0.5 pts [-1.5, +0.0], CI touches 0, p = 0.078; stat:P3/M15 0.700, vs 1x16 -0.5 pts [-1.5, +0.0], CI touches 0, p = 0.078; ranks 2-5.
  - P1-conc: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P1-conc/M5 0.710, vs 1x16 +0.5 pts [+0.0, +1.5], CI touches 0, p = 0.078; stat:P1-conc/M15 0.710, vs 1x16 +0.5 pts [+0.0, +1.5], CI touches 0, p = 0.078; ranks 13-16.
  - P2-conc: 4 of 4 rows inside the top band, 4 of 4 within 0.02 of 1x16; sealed:P2-conc/M5 0.705, vs 1x16 no difference on any item; stat:P2-conc/M15 0.710, vs 1x16 +0.5 pts [+0.0, +1.5], CI touches 0, p = 0.078; ranks 11-23.
- LLM selectors (E8a A0/A1: asked for accuracy, M5, one decision; E8b B0/B1: told the LEX objective, M15, mean of 5 decisions):
  - LLM:A0 0.680, vs 1x16 -2.5 pts [-6.0, +1.0], CI includes 0, p = 0.170, outside the top band, rank 24, P(win) 0.290, latency 208 s.
  - LLM:A1 0.665, vs 1x16 -4.0 pts [-8.0, +0.0], CI touches 0, p = 0.049, outside the top band, rank 25, P(win) 0.039, latency 233 s.
  - LLM:B0 0.610, vs 1x16 -9.5 pts [-13.1, -6.0], CI below 0, p < 0.001, outside the top band, rank 32, P(win) 0.000, latency 158 s, choice stability 0.43, 6 invalid -> 1x16.
  - LLM:B1 0.593, vs 1x16 -11.2 pts [-14.8, -7.7], CI below 0, p < 0.001, outside the top band, rank 35, P(win) 0.000, latency 178 s, choice stability 0.49, 1 invalid -> 1x16.
- Reference rows: fixed:1x16 0.705, vs 1x16 --, inside the top band, rank 6, P(win) 0.028, latency 290 s; heuristic/M15 0.685, vs 1x16 -2.0 pts [-4.0, -0.5], CI below 0, p = 0.074, outside the top band, rank 17, P(win) 0.244, latency 244 s; heuristic/M5 0.695, vs 1x16 -1.0 pts [-2.5, +0.0], CI touches 0, p = 0.015, inside the top band, rank 1, P(win) 0.231, latency 246 s.
- Statistical minus LLM accuracy on the E8a allocator's 5-option menu (sealed:P0/M5, sealed:P1/M5, sealed:P2/M5, sealed:P3/M5, sealed:P1-conc/M5, sealed:P2-conc/M5): vs LLM:A0 -0.5 to +3.0 pts, CI above 0 for 0 of 6, below 0 for 0 (includes or touches 0: sealed:P0/M5, sealed:P1/M5, sealed:P2/M5, sealed:P3/M5, sealed:P1-conc/M5, sealed:P2-conc/M5), p 0.121-0.791; vs LLM:A1 +1.0 to +4.5 pts, CI above 0 for 1 of 6, below 0 for 0 (includes or touches 0: sealed:P0/M5, sealed:P1/M5, sealed:P2/M5, sealed:P3/M5, sealed:P2-conc/M5), p 0.024-0.633.
- Statistical minus LLM accuracy on E8b's 15-option menu (stat:P0/M15, stat:P1/M15, stat:P2/M15, stat:P3/M15, stat:P1-conc/M15, stat:P2-conc/M15): vs LLM:B0 +6.5 to +10.0 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001; vs LLM:B1 +8.2 to +11.7 pts, CI above 0 for 6 of 6, below 0 for 0, p <0.001.
- Telling the LLM the objective (B minus A; B also has the 15-option menu with typical times and averages 5 decisions): LLM:B0 - LLM:A0: accuracy -7.0 pts [-10.9, -3.1], CI below 0, p < 0.001; latency -50 [-68, -32] s | LLM:B0 - LLM:A1: accuracy -5.5 pts [-9.5, -1.4], CI below 0, p = 0.010; latency -81 [-102, -59] s | LLM:B1 - LLM:A1: accuracy -7.2 pts [-11.3, -3.2], CI below 0, p = 0.001; latency -49 [-68, -31] s | LLM:B1 - LLM:A0: accuracy -8.7 pts [-12.9, -4.6], CI below 0, p < 0.001; latency -19 [-40, +2] s | LLM:B1 - LLM:B0: accuracy -1.7 pts [-4.8, +1.3], CI includes 0, p = 0.291; latency +31 [+19, +44] s.
- Ranking: statistical rows rank 2-29; LLM rows rank LLM:A0 24, LLM:A1 25, LLM:B0 32, LLM:B1 35; the best LLM row (LLM:A0) ranks 24; statistical rows ranked below it: sealed:P2/M5, stat:P2/M5, sealed:P2/M9, stat:P2/M15.
- Bootstrap win share: LLM:A0 0.290, LLM:A1 0.039, LLM:B0 0.000, LLM:B1 0.000; highest statistical row stat:P3/M15 0.073.

CI = 95% percentile interval of the 10,000-replicate benchmark-stratified bootstrap; p = the null-centred studentised bootstrap p of the same contrast (unadjusted). The two can disagree at these sample sizes. "CI touches 0" = a bound exactly at 0 (0/1 outcomes), which is not an exclusion of 0.

C48 (no-diagnostic bank estimates, macro; M5 rows are like-for-like with A0/A1, M15 rows with B0/B1):

| selector | menu | overhead | accuracy [95% CI] | consistency [95% CI] | latency s [95% CI] | Δacc vs 1x16 (pts) | ΔC vs 1x16 (pts) | Δlat vs 1x16 (s) | LEX rank | P(win) |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed:1x16 | 1x16 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | -- | -- | -- | 1 | 0.32 |
| stat:P0/M15 | M15 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 2 | 0.00 |
| stat:P0/M5 | M5 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 3 | 0.00 |
| stat:P2-conc/M15 | M15 | diag | 0.774 [0.675, 0.867] | 0.895 [0.845, 0.942] | 278 [254, 304] | -1.0 [-3.7, +0.8] | +0.9 [-1.1, +3.4] | +2 [-14, +18] | 4 | 0.12 |
| stat:P3/M15 | M15 | diag | 0.777 [0.683, 0.866] | 0.877 [0.824, 0.927] | 279 [248, 311] | -0.7 [-3.3, +1.2] | -0.9 [-3.5, +1.3] | +3 [-20, +25] | 5 | 0.08 |
| stat:P3/M5 | M5 | diag | 0.777 [0.683, 0.866] | 0.877 [0.824, 0.927] | 280 [249, 312] | -0.7 [-3.3, +1.2] | -0.9 [-3.5, +1.3] | +4 [-18, +26] | 6 | 0.00 |
| stat:P2-conc/M5 | M5 | diag | 0.774 [0.675, 0.867] | 0.895 [0.845, 0.942] | 285 [259, 311] | -1.0 [-3.7, +0.8] | +0.9 [-1.1, +3.4] | +8 [-9, +27] | 7 | 0.00 |
| stat:P1/M15 | M15 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 320 [298, 345] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +44 [+32, +58] | 8 | 0.00 |
| stat:P1/M5 | M5 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 320 [298, 345] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +44 [+32, +58] | 9 | 0.00 |
| stat:P1-conc/M15 | M15 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 323 [301, 347] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +46 [+35, +60] | 10 | 0.00 |
| stat:P1-conc/M5 | M5 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 323 [301, 347] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +46 [+35, +60] | 11 | 0.00 |
| stat:P2/M15 | M15 | diag | 0.763 [0.662, 0.858] | 0.902 [0.852, 0.947] | 266 [239, 294] | -2.2 [-6.3, +0.8] | +1.6 [-0.6, +4.1] | -11 [-31, +9] | 12 | 0.35 |
| heuristic/M5 | M5 | diag | 0.753 [0.649, 0.852] | 0.895 [0.844, 0.943] | 273 [239, 306] | -3.2 [-8.5, +0.0] | +0.9 [-0.0, +2.8] | -4 [-29, +20] | 13 | 0.02 |
| stat:P2/M5 | M5 | diag | 0.760 [0.658, 0.857] | 0.906 [0.857, 0.952] | 283 [252, 315] | -2.4 [-6.8, +0.7] | +2.0 [-0.4, +5.0] | +6 [-19, +32] | 14 | 0.00 |
| heuristic/M15 | M15 | diag | 0.751 [0.650, 0.847] | 0.878 [0.826, 0.927] | 262 [226, 297] | -3.3 [-7.9, -0.1] | -0.8 [-2.7, +1.0] | -15 [-42, +12] | 15 | 0.10 |
| LLM:A1 | M5 | diag+llm | 0.713 [0.606, 0.818] | 0.883 [0.835, 0.928] | 240 [211, 269] | -7.1 [-12.9, -2.4] | -0.3 [-3.8, +3.6] | -37 [-62, -12] | 16 | 0.00 |
| LLM:A0 | M5 | llm | 0.731 [0.627, 0.827] | 0.861 [0.810, 0.910] | 207 [179, 236] | -5.4 [-9.5, -1.9] | -2.5 [-6.5, +1.6] | -69 [-95, -44] | 17 | 0.01 |
| fixed:1x8 | 1x8 | none | 0.688 [0.581, 0.792] | 0.870 [0.817, 0.919] | 151 [143, 159] | -9.7 [-15.4, -4.5] | -1.6 [-7.3, +4.3] | -126 [-135, -116] | 18 | 0.00 |
| LLM:B1 | M15 | diag+llm | 0.694 [0.595, 0.792] | 0.827 [0.779, 0.873] | 202 [179, 228] | -9.1 [-13.1, -5.2] | -5.9 [-10.2, -1.6] | -74 [-92, -54] | 20 | 0.00 |
| LLM:B0 | M15 | llm | 0.670 [0.578, 0.761] | 0.785 [0.732, 0.837] | 155 [135, 175] | -11.4 [-16.3, -6.8] | -10.0 [-15.4, -5.0] | -122 [-137, -106] | 21 | 0.00 |
| oracle/M15 | M15 | none | 0.829 [0.747, 0.905] | 0.878 [0.821, 0.931] | 128 [100, 159] | +4.4 [+1.2, +8.7] | -0.8 [-4.2, +2.2] | -148 [-174, -120] | ref | -- |
| oracle/M5 | M5 | none | 0.829 [0.747, 0.905] | 0.879 [0.822, 0.932] | 136 [108, 165] | +4.4 [+1.2, +8.7] | -0.7 [-4.1, +2.2] | -141 [-167, -113] | ref | -- |
| fixed:1x25 | 1x25 | none | 0.836 [0.755, 0.910] | 0.894 [0.846, 0.938] | 418 [394, 440] | +5.1 [+1.6, +8.8] | +0.8 [-2.2, +4.1] | +141 [+129, +153] | ref | -- |

T200 (realized outcomes; `sealed:` = the sealed 9-option predictions, `stat:` = post-hoc M15 refit), sorted by the mixture rank:

| selector | menu | overhead | acc mixture [95% CI] | acc macro [95% CI] | latency s (macro) [95% CI] | Δacc mixture vs 1x16 (pts) | Δacc macro vs 1x16 (pts) | Δlat vs 1x16 (s) | rank mixture | rank macro | P(win) mixture | P(win) macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic/M5 | M5 | diag | 0.695 [0.635, 0.750] | 0.786 [0.723, 0.843] | 246 [223, 268] | -1.0 [-2.5, +0.0] | -0.8 [-2.1, +0.0] | -39 [-55, -23] | 1 | 2 | 0.23 | 0.19 |
| stat:P3/M15 | M15 | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 267 [246, 288] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -18 [-32, -5] | 2 | 3 | 0.07 | 0.09 |
| sealed:P3/M9 | M9sealed | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 267 [246, 288] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -18 [-32, -4] | 3 | 4 | 0.00 | 0.00 |
| sealed:P3/M5 | M5sealed | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 269 [247, 289] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -16 [-30, -3] | 4 | 5 | 0.00 | 0.00 |
| fixed:1x16 | 1x16 | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | -- | -- | -- | 6 | 7 | 0.03 | 0.03 |
| sealed:P0/M5 | M5sealed | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 7 | 8 | 0.00 | 0.00 |
| stat:P0/M15 | M15 | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 9 | 10 | 0.00 | 0.00 |
| stat:P2-conc/M15 | M15 | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 309 [293, 324] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +24 [+18, +29] | 11 | 12 | 0.00 | 0.00 |
| stat:P1-conc/M15 | M15 | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [299, 329] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +29 [+25, +32] | 13 | 18 | 0.00 | 0.00 |
| sealed:P1-conc/M5 | M5sealed | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [300, 330] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +30 [+26, +32] | 15 | 20 | 0.00 | 0.00 |
| heuristic/M15 | M15 | diag | 0.685 [0.625, 0.745] | 0.779 [0.716, 0.837] | 244 [221, 266] | -2.0 [-4.0, -0.5] | -1.5 [-3.2, -0.2] | -41 [-58, -25] | 17 | 1 | 0.24 | 0.43 |
| sealed:P1/M5 | M5sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 18 | 14 | 0.00 | 0.00 |
| stat:P1/M15 | M15 | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 20 | 16 | 0.00 | 0.00 |
| sealed:P2-conc/M5 | M5sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 315 [300, 330] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +30 [+26, +33] | 22 | 26 | 0.00 | 0.00 |
| LLM:A0 | M5 | llm | 0.680 [0.625, 0.740] | 0.762 [0.694, 0.825] | 202 [184, 222] | -2.5 [-6.0, +1.0] | -3.2 [-7.6, +0.3] | -83 [-103, -62] | 24 | 28 | 0.29 | 0.21 |
| LLM:A1 | M5 | diag+llm | 0.665 [0.605, 0.725] | 0.731 [0.657, 0.799] | 233 [211, 254] | -4.0 [-8.0, +0.0] | -6.3 [-11.8, -1.7] | -52 [-72, -33] | 25 | 31 | 0.04 | 0.01 |
| sealed:P2/M5 | M5sealed | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [252, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-26, -10] | 26 | 22 | 0.00 | 0.00 |
| stat:P2/M15 | M15 | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [253, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-25, -10] | 29 | 25 | 0.01 | 0.01 |
| fixed:2x8 | 2x8 | none | 0.650 [0.590, 0.710] | 0.725 [0.653, 0.792] | 178 [166, 192] | -5.5 [-10.0, -1.0] | -6.9 [-12.2, -2.3] | -107 [-124, -89] | 31 | 30 | 0.02 | 0.01 |
| LLM:B0 | M15 | llm | 0.610 [0.559, 0.659] | 0.664 [0.601, 0.724] | 152 [139, 165] | -9.5 [-13.1, -6.0] | -13.0 [-17.7, -8.6] | -133 [-147, -119] | 32 | 32 | 0.00 | 0.00 |
| LLM:B1 | M15 | diag+llm | 0.593 [0.541, 0.646] | 0.666 [0.605, 0.724] | 184 [167, 200] | -11.2 [-14.8, -7.7] | -12.8 [-16.8, -9.0] | -101 [-116, -87] | 35 | 33 | 0.00 | 0.00 |
| oracle/M5 | M5 | none | 0.765 [0.710, 0.820] | 0.829 [0.768, 0.883] | 77 [62, 94] | +6.0 [+3.0, +9.5] | +3.5 [+1.7, +5.6] | -208 [-226, -189] | ref | ref | -- | -- |

## E8b LLM selectors: choices, stability, fallbacks

Choice stability = mean over items of the modal option's share of the item's 5 decisions (1.0 = the same option every time).

| arm | panel | decisions | invalid -> 1x16 | decisions retried | choice stability | items with all 5 identical | distinct options per item | mean call latency s | options chosen |
|---|---|---|---|---|---|---|---|---|---|
| B0 | C48 | 240 | 0 | 5 | 0.46 | 0.00 | 3.58 | 18.6 | 1x16 60, 1x4 29, 1x8 27, 1x1 22, 2x8 18, 4x4 16, 16x1 15, 1x2 13, 4x2 12, 2x2 8, 8x2 5, 4x1 5, 2x4 4, 8x1 4, 2x1 2 |
| B0 | T200 | 1000 | 6 | 25 | 0.43 | 0.01 | 3.69 | 17.9 | 1x16 250, 1x8 126, 1x1 99, 4x4 74, 4x2 65, 1x4 64, 2x8 58, 16x1 56, 2x4 48, 8x2 34, 1x2 33, 4x1 29, 2x2 26, 8x1 23, 2x1 15 |
| B1 | C48 | 240 | 1 | 7 | 0.50 | 0.02 | 3.35 | 19.3 | 1x16 68, 1x1 35, 1x8 29, 16x1 17, 4x2 15, 4x4 15, 1x4 14, 2x8 14, 8x1 7, 2x2 6, 1x2 5, 2x4 5, 4x1 4, 2x1 3, 8x2 3 |
| B1 | T200 | 1000 | 1 | 13 | 0.49 | 0.03 | 3.38 | 17.8 | 1x16 210, 1x1 190, 1x8 105, 2x8 87, 4x4 73, 16x1 66, 4x2 52, 1x4 41, 2x4 38, 4x1 29, 8x1 27, 2x2 25, 2x1 22, 8x2 18, 1x2 17 |

## Telling the LLM the objective: E8b minus E8a

C48 (no-diagnostic bank, macro):

| contrast | Δacc (pts) | p (acc) | ΔC (pts) | Δlat (s) |
|---|---|---|---|---|
| LLM:B0 minus LLM:A0 | -6.0 [-11.4, -0.8] | 0.039 | -7.6 [-12.4, -3.0] | -52 [-81, -23] |
| LLM:B0 minus LLM:A1 | -4.3 [-9.2, +0.5] | 0.102 | -9.8 [-14.8, -5.0] | -85 [-113, -56] |
| LLM:B1 minus LLM:A1 | -1.9 [-6.7, +3.6] | 0.499 | -5.6 [-9.6, -1.6] | -37 [-66, -9] |
| LLM:B1 minus LLM:A0 | -3.6 [-8.4, +1.3] | 0.167 | -3.4 [-7.6, +0.8] | -5 [-35, +25] |
| LLM:B1 minus LLM:B0 | +2.4 [-1.5, +6.6] | 0.284 | +4.2 [+0.2, +8.5] | +48 [+30, +66] |

T200 (realized):

| contrast | Δacc mixture (pts) | p mixture | Δacc macro (pts) | p macro | Δlat macro (s) |
|---|---|---|---|---|---|
| LLM:B0 minus LLM:A0 | -7.0 [-10.9, -3.1] | <0.001 | -9.8 [-14.8, -4.9] | 0.001 | -50 [-68, -32] |
| LLM:B0 minus LLM:A1 | -5.5 [-9.5, -1.4] | 0.010 | -6.7 [-12.9, -0.2] | 0.050 | -81 [-102, -59] |
| LLM:B1 minus LLM:A1 | -7.2 [-11.3, -3.2] | 0.001 | -6.5 [-11.9, -0.6] | 0.034 | -49 [-68, -31] |
| LLM:B1 minus LLM:A0 | -8.7 [-12.9, -4.6] | <0.001 | -9.5 [-14.3, -4.8] | <0.001 | -19 [-40, +2] |
| LLM:B1 minus LLM:B0 | -1.7 [-4.8, +1.3] | 0.291 | +0.2 [-3.7, +4.4] | 0.919 | +31 [+19, +44] |

## C48 (no-diagnostic bank, macro), all selectors

| selector | menu | overhead | accuracy [95% CI] | consistency [95% CI] | latency s [95% CI] | Δacc vs 1x16 (pts) | ΔC vs 1x16 (pts) | Δlat vs 1x16 (s) | LEX rank | P(win) |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed:1x16 | 1x16 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | -- | -- | -- | 1 | 0.32 |
| stat:P0/M15 | M15 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 2 | 0.00 |
| stat:P0/M5 | M5 | none | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 277 [262, 291] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 3 | 0.00 |
| stat:P2-conc/M15 | M15 | diag | 0.774 [0.675, 0.867] | 0.895 [0.845, 0.942] | 278 [254, 304] | -1.0 [-3.7, +0.8] | +0.9 [-1.1, +3.4] | +2 [-14, +18] | 4 | 0.12 |
| stat:P3/M15 | M15 | diag | 0.777 [0.683, 0.866] | 0.877 [0.824, 0.927] | 279 [248, 311] | -0.7 [-3.3, +1.2] | -0.9 [-3.5, +1.3] | +3 [-20, +25] | 5 | 0.08 |
| stat:P3/M5 | M5 | diag | 0.777 [0.683, 0.866] | 0.877 [0.824, 0.927] | 280 [249, 312] | -0.7 [-3.3, +1.2] | -0.9 [-3.5, +1.3] | +4 [-18, +26] | 6 | 0.00 |
| stat:P2-conc/M5 | M5 | diag | 0.774 [0.675, 0.867] | 0.895 [0.845, 0.942] | 285 [259, 311] | -1.0 [-3.7, +0.8] | +0.9 [-1.1, +3.4] | +8 [-9, +27] | 7 | 0.00 |
| stat:P1/M15 | M15 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 320 [298, 345] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +44 [+32, +58] | 8 | 0.00 |
| stat:P1/M5 | M5 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 320 [298, 345] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +44 [+32, +58] | 9 | 0.00 |
| stat:P1-conc/M15 | M15 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 323 [301, 347] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +46 [+35, +60] | 10 | 0.00 |
| stat:P1-conc/M5 | M5 | diag | 0.785 [0.688, 0.875] | 0.886 [0.831, 0.936] | 323 [301, 347] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +46 [+35, +60] | 11 | 0.00 |
| stat:P2/M15 | M15 | diag | 0.763 [0.662, 0.858] | 0.902 [0.852, 0.947] | 266 [239, 294] | -2.2 [-6.3, +0.8] | +1.6 [-0.6, +4.1] | -11 [-31, +9] | 12 | 0.35 |
| heuristic/M5 | M5 | diag | 0.753 [0.649, 0.852] | 0.895 [0.844, 0.943] | 273 [239, 306] | -3.2 [-8.5, +0.0] | +0.9 [-0.0, +2.8] | -4 [-29, +20] | 13 | 0.02 |
| stat:P2/M5 | M5 | diag | 0.760 [0.658, 0.857] | 0.906 [0.857, 0.952] | 283 [252, 315] | -2.4 [-6.8, +0.7] | +2.0 [-0.4, +5.0] | +6 [-19, +32] | 14 | 0.00 |
| heuristic/M15 | M15 | diag | 0.751 [0.650, 0.847] | 0.878 [0.826, 0.927] | 262 [226, 297] | -3.3 [-7.9, -0.1] | -0.8 [-2.7, +1.0] | -15 [-42, +12] | 15 | 0.10 |
| LLM:A1 | M5 | diag+llm | 0.713 [0.606, 0.818] | 0.883 [0.835, 0.928] | 240 [211, 269] | -7.1 [-12.9, -2.4] | -0.3 [-3.8, +3.6] | -37 [-62, -12] | 16 | 0.00 |
| LLM:A0 | M5 | llm | 0.731 [0.627, 0.827] | 0.861 [0.810, 0.910] | 207 [179, 236] | -5.4 [-9.5, -1.9] | -2.5 [-6.5, +1.6] | -69 [-95, -44] | 17 | 0.01 |
| fixed:1x8 | 1x8 | none | 0.688 [0.581, 0.792] | 0.870 [0.817, 0.919] | 151 [143, 159] | -9.7 [-15.4, -4.5] | -1.6 [-7.3, +4.3] | -126 [-135, -116] | 18 | 0.00 |
| fixed:2x8 | 2x8 | none | 0.688 [0.581, 0.792] | 0.870 [0.817, 0.919] | 166 [158, 174] | -9.7 [-15.4, -4.5] | -1.6 [-7.3, +4.3] | -111 [-121, -100] | 19 | 0.00 |
| LLM:B1 | M15 | diag+llm | 0.694 [0.595, 0.792] | 0.827 [0.779, 0.873] | 202 [179, 228] | -9.1 [-13.1, -5.2] | -5.9 [-10.2, -1.6] | -74 [-92, -54] | 20 | 0.00 |
| LLM:B0 | M15 | llm | 0.670 [0.578, 0.761] | 0.785 [0.732, 0.837] | 155 [135, 175] | -11.4 [-16.3, -6.8] | -10.0 [-15.4, -5.0] | -122 [-137, -106] | 21 | 0.00 |
| fixed:4x4 | 4x4 | none | 0.645 [0.521, 0.764] | 0.919 [0.875, 0.958] | 102 [98, 106] | -14.0 [-21.1, -7.3] | +3.3 [-3.0, +9.5] | -175 [-188, -160] | 22 | 0.00 |
| fixed:8x2 | 8x2 | none | 0.597 [0.477, 0.715] | 0.904 [0.856, 0.948] | 60 [56, 64] | -18.7 [-26.8, -10.9] | +1.8 [-5.1, +8.8] | -217 [-230, -203] | 23 | 0.00 |
| fixed:1x4 | 1x4 | none | 0.602 [0.488, 0.715] | 0.862 [0.815, 0.907] | 82 [78, 85] | -18.2 [-25.5, -11.5] | -2.4 [-9.3, +4.6] | -195 [-206, -183] | 24 | 0.00 |
| fixed:2x4 | 2x4 | none | 0.602 [0.488, 0.715] | 0.862 [0.815, 0.907] | 91 [88, 95] | -18.2 [-25.5, -11.5] | -2.4 [-9.3, +4.6] | -185 [-198, -172] | 25 | 0.00 |
| fixed:4x2 | 4x2 | none | 0.532 [0.421, 0.643] | 0.847 [0.795, 0.898] | 55 [52, 58] | -25.2 [-33.5, -17.2] | -3.9 [-11.5, +3.8] | -222 [-235, -208] | 26 | 0.00 |
| fixed:16x1 | 16x1 | none | 0.490 [0.369, 0.613] | 0.922 [0.876, 0.963] | 58 [48, 70] | -29.4 [-40.9, -18.3] | +3.6 [-4.0, +11.2] | -219 [-234, -204] | 27 | 0.00 |
| fixed:8x1 | 8x1 | none | 0.458 [0.342, 0.576] | 0.901 [0.854, 0.943] | 51 [43, 61] | -32.7 [-43.6, -22.0] | +1.5 [-6.0, +9.0] | -225 [-239, -211] | 28 | 0.00 |
| fixed:1x2 | 1x2 | none | 0.469 [0.368, 0.570] | 0.800 [0.746, 0.854] | 45 [43, 47] | -31.6 [-39.9, -23.3] | -8.5 [-16.4, -0.5] | -231 [-244, -218] | 29 | 0.00 |
| fixed:2x2 | 2x2 | none | 0.469 [0.368, 0.570] | 0.800 [0.746, 0.854] | 50 [48, 52] | -31.6 [-39.9, -23.3] | -8.5 [-16.4, -0.5] | -227 [-240, -213] | 30 | 0.00 |
| fixed:4x1 | 4x1 | none | 0.422 [0.313, 0.533] | 0.872 [0.829, 0.914] | 44 [38, 52] | -36.3 [-46.8, -26.0] | -1.4 [-8.3, +5.9] | -233 [-246, -219] | 31 | 0.00 |
| fixed:1x1 | 1x1 | none | 0.373 [0.280, 0.472] | 0.824 [0.774, 0.874] | 31 [28, 35] | -41.1 [-50.9, -31.4] | -6.2 [-13.7, +1.7] | -246 [-258, -233] | 32 | 0.00 |
| fixed:2x1 | 2x1 | none | 0.373 [0.279, 0.472] | 0.824 [0.774, 0.874] | 37 [32, 42] | -41.2 [-50.9, -31.5] | -6.1 [-13.6, +1.8] | -240 [-252, -227] | 33 | 0.00 |
| oracle/M17 | M17 | none | 0.883 [0.815, 0.943] | 0.907 [0.857, 0.951] | 154 [114, 199] | +9.8 [+4.7, +15.9] | +2.1 [-2.0, +6.3] | -122 [-158, -83] | ref | -- |
| oracle/M15 | M15 | none | 0.829 [0.747, 0.905] | 0.878 [0.821, 0.931] | 128 [100, 159] | +4.4 [+1.2, +8.7] | -0.8 [-4.2, +2.2] | -148 [-174, -120] | ref | -- |
| oracle/M5 | M5 | none | 0.829 [0.747, 0.905] | 0.879 [0.822, 0.932] | 136 [108, 165] | +4.4 [+1.2, +8.7] | -0.7 [-4.1, +2.2] | -141 [-167, -113] | ref | -- |
| fixed:1x25 | 1x25 | none | 0.836 [0.755, 0.910] | 0.894 [0.846, 0.938] | 418 [394, 440] | +5.1 [+1.6, +8.8] | +0.8 [-2.2, +4.1] | +141 [+129, +153] | ref | -- |
| splithalf/M17 | M17 | none | 0.765 [0.680, 0.843] | 0.855 [0.809, 0.900] | 145 [104, 190] | -2.0 [-8.9, +4.6] | -3.1 [-7.0, +1.2] | -132 [-168, -92] | ref | -- |

## C48: statistical and heuristic selectors minus the LLM selectors

A0/A1 against every row; B0/B1 against the rows on their own M15 menu.

| selector | vs | Δacc (pts) | p (acc) | ΔC (pts) | Δlat (s) |
|---|---|---|---|---|---|
| stat:P0/M15 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +69 [+44, +95] |
| stat:P0/M15 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +37 [+12, +62] |
| stat:P0/M15 | LLM:B0 | +11.4 [+6.8, +16.3] | <0.001 | +10.0 [+5.0, +15.4] | +122 [+106, +137] |
| stat:P0/M15 | LLM:B1 | +9.1 [+5.2, +13.1] | <0.001 | +5.9 [+1.6, +10.2] | +74 [+54, +92] |
| stat:P0/M5 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +69 [+44, +95] |
| stat:P0/M5 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +37 [+12, +62] |
| stat:P1/M15 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +113 [+84, +143] |
| stat:P1/M15 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +80 [+54, +107] |
| stat:P1/M15 | LLM:B0 | +11.4 [+6.8, +16.3] | <0.001 | +10.0 [+5.0, +15.4] | +165 [+147, +185] |
| stat:P1/M15 | LLM:B1 | +9.1 [+5.2, +13.1] | <0.001 | +5.9 [+1.6, +10.2] | +118 [+102, +133] |
| stat:P1/M5 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +113 [+84, +143] |
| stat:P1/M5 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +80 [+54, +107] |
| stat:P2/M15 | LLM:A0 | +3.3 [-2.2, +8.4] | 0.264 | +4.1 [-0.4, +8.5] | +59 [+27, +90] |
| stat:P2/M15 | LLM:A1 | +5.0 [+0.6, +10.3] | 0.108 | +1.9 [-2.1, +5.9] | +26 [-1, +54] |
| stat:P2/M15 | LLM:B0 | +9.3 [+4.4, +14.2] | 0.001 | +11.6 [+6.7, +16.8] | +111 [+87, +135] |
| stat:P2/M15 | LLM:B1 | +6.9 [+3.1, +10.6] | 0.002 | +7.5 [+3.1, +12.0] | +63 [+41, +86] |
| stat:P2/M5 | LLM:A0 | +3.0 [-2.7, +8.3] | 0.323 | +4.5 [-0.1, +9.1] | +75 [+39, +113] |
| stat:P2/M5 | LLM:A1 | +4.7 [+0.4, +10.0] | 0.121 | +2.3 [-1.5, +6.2] | +43 [+13, +73] |
| stat:P3/M15 | LLM:A0 | +4.7 [+0.2, +9.4] | 0.073 | +1.6 [-3.3, +6.4] | +72 [+39, +105] |
| stat:P3/M15 | LLM:A1 | +6.4 [+1.9, +11.9] | 0.066 | -0.6 [-4.3, +2.6] | +39 [+9, +70] |
| stat:P3/M15 | LLM:B0 | +10.7 [+6.1, +15.7] | <0.001 | +9.2 [+3.7, +14.8] | +124 [+99, +149] |
| stat:P3/M15 | LLM:B1 | +8.3 [+4.6, +12.3] | 0.001 | +5.0 [+0.5, +9.5] | +77 [+56, +97] |
| stat:P3/M5 | LLM:A0 | +4.7 [+0.2, +9.4] | 0.073 | +1.6 [-3.3, +6.4] | +73 [+40, +106] |
| stat:P3/M5 | LLM:A1 | +6.4 [+1.9, +11.9] | 0.066 | -0.6 [-4.3, +2.6] | +40 [+10, +71] |
| stat:P1-conc/M15 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +115 [+86, +146] |
| stat:P1-conc/M15 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +83 [+57, +108] |
| stat:P1-conc/M15 | LLM:B0 | +11.4 [+6.8, +16.3] | <0.001 | +10.0 [+5.0, +15.4] | +168 [+149, +187] |
| stat:P1-conc/M15 | LLM:B1 | +9.1 [+5.2, +13.1] | <0.001 | +5.9 [+1.6, +10.2] | +120 [+104, +136] |
| stat:P1-conc/M5 | LLM:A0 | +5.4 [+1.9, +9.5] | 0.040 | +2.5 [-1.6, +6.5] | +115 [+86, +146] |
| stat:P1-conc/M5 | LLM:A1 | +7.1 [+2.4, +12.9] | 0.053 | +0.3 [-3.6, +3.8] | +83 [+57, +108] |
| stat:P2-conc/M15 | LLM:A0 | +4.4 [+0.0, +8.9] | 0.077 | +3.4 [-1.2, +7.9] | +71 [+41, +100] |
| stat:P2-conc/M15 | LLM:A1 | +6.1 [+1.5, +11.7] | 0.072 | +1.2 [-2.5, +4.9] | +38 [+12, +64] |
| stat:P2-conc/M15 | LLM:B0 | +10.4 [+5.9, +15.3] | <0.001 | +11.0 [+6.2, +16.0] | +123 [+100, +146] |
| stat:P2-conc/M15 | LLM:B1 | +8.0 [+4.4, +11.9] | 0.001 | +6.8 [+2.7, +10.9] | +76 [+57, +95] |
| stat:P2-conc/M5 | LLM:A0 | +4.4 [+0.0, +8.9] | 0.077 | +3.4 [-1.2, +7.9] | +78 [+45, +110] |
| stat:P2-conc/M5 | LLM:A1 | +6.1 [+1.5, +11.7] | 0.072 | +1.2 [-2.5, +4.9] | +45 [+17, +74] |
| heuristic/M15 | LLM:A0 | +2.1 [-3.9, +7.5] | 0.524 | +1.7 [-2.7, +6.0] | +55 [+20, +90] |
| heuristic/M15 | LLM:A1 | +3.8 [-1.8, +9.9] | 0.245 | -0.5 [-4.1, +2.8] | +22 [-11, +54] |
| heuristic/M15 | LLM:B0 | +8.1 [+1.8, +14.0] | 0.023 | +9.3 [+4.5, +14.4] | +107 [+80, +134] |
| heuristic/M15 | LLM:B1 | +5.7 [+0.4, +10.6] | 0.052 | +5.1 [+1.0, +9.3] | +59 [+37, +82] |
| heuristic/M5 | LLM:A0 | +2.2 [-4.3, +8.0] | 0.514 | +3.4 [-1.0, +7.7] | +65 [+31, +99] |
| heuristic/M5 | LLM:A1 | +3.9 [-1.8, +10.1] | 0.228 | +1.2 [-2.1, +4.3] | +33 [+3, +63] |

## C48 option distributions

E8b rows count decisions (5 per item); the others count items.

| selector | choices |
|---|---|
| fixed:1x1 | 1x1 48 |
| fixed:2x1 | 2x1 48 |
| fixed:4x1 | 4x1 48 |
| fixed:8x1 | 8x1 48 |
| fixed:16x1 | 16x1 48 |
| fixed:1x2 | 1x2 48 |
| fixed:2x2 | 2x2 48 |
| fixed:4x2 | 4x2 48 |
| fixed:8x2 | 8x2 48 |
| fixed:1x4 | 1x4 48 |
| fixed:2x4 | 2x4 48 |
| fixed:4x4 | 4x4 48 |
| fixed:1x8 | 1x8 48 |
| fixed:2x8 | 2x8 48 |
| fixed:1x16 | 1x16 48 |
| fixed:1x25 | 1x25 48 |
| stat:P0/M15 | 1x16 48 |
| stat:P0/M5 | 1x16 48 |
| stat:P1/M15 | 1x16 47, 2x8 1 |
| stat:P1/M5 | 1x16 47, 2x8 1 |
| stat:P2/M15 | 1x16 34, 8x2 4, 1x4 3, 16x1 2, 1x8 2, 4x4 2, +1 more |
| stat:P2/M5 | 1x16 39, 8x2 5, 16x1 2, 4x4 2 |
| stat:P3/M15 | 1x16 37, 8x1 5, 4x2 3, 1x8 1, 2x8 1, 8x2 1 |
| stat:P3/M5 | 1x16 37, 16x1 5, 8x2 4, 2x8 2 |
| stat:P1-conc/M15 | 1x16 48 |
| stat:P1-conc/M5 | 1x16 48 |
| stat:P2-conc/M15 | 1x16 38, 8x2 6, 16x1 2, 1x4 1, 1x8 1 |
| stat:P2-conc/M5 | 1x16 40, 8x2 6, 16x1 2 |
| heuristic/M15 | 1x16 34, 4x1 5, 4x2 5, 1x2 4 |
| heuristic/M5 | 1x16 36, 8x2 11, 16x1 1 |
| LLM:A0 | 1x16 23, 2x8 9, 16x1 7, 4x4 7, 8x2 2 |
| LLM:A1 | 1x16 20, 4x4 10, 2x8 9, 16x1 6, 8x2 3 |
| LLM:B0 | 1x16 60, 1x4 29, 1x8 27, 1x1 22, 2x8 18, 4x4 16, +9 more |
| LLM:B1 | 1x16 68, 1x1 35, 1x8 29, 16x1 17, 4x2 15, 4x4 15, +9 more |
| oracle/M15 | 1x16 13, 1x1 6, 1x8 6, 16x1 5, 8x2 4, 4x1 3, +5 more |
| oracle/M5 | 16x1 15, 1x16 13, 8x2 9, 2x8 6, 4x4 5 |
| oracle/M17 | 1x16 7, 1x25 7, 1x1 6, 1x8 5, 25x1 4, 4x1 3, +7 more |
| splithalf/M17 | split-half 48 |

## T200 (realized outcomes), all selectors, sorted by the macro rank

| selector | menu | overhead | acc mixture [95% CI] | acc macro [95% CI] | latency s (macro) [95% CI] | Δacc mixture vs 1x16 (pts) | Δacc macro vs 1x16 (pts) | Δlat vs 1x16 (s) | rank mixture | rank macro | P(win) mixture | P(win) macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic/M15 | M15 | diag | 0.685 [0.625, 0.745] | 0.779 [0.716, 0.837] | 244 [221, 266] | -2.0 [-4.0, -0.5] | -1.5 [-3.2, -0.2] | -41 [-58, -25] | 17 | 1 | 0.24 | 0.43 |
| heuristic/M5 | M5 | diag | 0.695 [0.635, 0.750] | 0.786 [0.723, 0.843] | 246 [223, 268] | -1.0 [-2.5, +0.0] | -0.8 [-2.1, +0.0] | -39 [-55, -23] | 1 | 2 | 0.23 | 0.19 |
| stat:P3/M15 | M15 | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 267 [246, 288] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -18 [-32, -5] | 2 | 3 | 0.07 | 0.09 |
| sealed:P3/M9 | M9sealed | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 267 [246, 288] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -18 [-32, -4] | 3 | 4 | 0.00 | 0.00 |
| sealed:P3/M5 | M5sealed | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 269 [247, 289] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -16 [-30, -3] | 4 | 5 | 0.00 | 0.00 |
| stat:P3/M5 | M5 | diag | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 269 [247, 289] | -0.5 [-1.5, +0.0] | -0.4 [-1.2, +0.0] | -16 [-30, -3] | 5 | 6 | 0.00 | 0.00 |
| fixed:1x16 | 1x16 | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | -- | -- | -- | 6 | 7 | 0.03 | 0.03 |
| sealed:P0/M5 | M5sealed | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 7 | 8 | 0.00 | 0.00 |
| sealed:P0/M9 | M9sealed | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 8 | 9 | 0.00 | 0.00 |
| stat:P0/M15 | M15 | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 9 | 10 | 0.00 | 0.00 |
| stat:P0/M5 | M5 | none | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 285 [271, 300] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | 10 | 11 | 0.00 | 0.00 |
| stat:P2-conc/M15 | M15 | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 309 [293, 324] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +24 [+18, +29] | 11 | 12 | 0.00 | 0.00 |
| sealed:P2-conc/M9 | M9sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 309 [294, 324] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +24 [+19, +29] | 12 | 13 | 0.00 | 0.00 |
| sealed:P1/M5 | M5sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 18 | 14 | 0.00 | 0.00 |
| sealed:P1/M9 | M9sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 19 | 15 | 0.00 | 0.00 |
| stat:P1/M15 | M15 | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 20 | 16 | 0.00 | 0.00 |
| stat:P1/M5 | M5 | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 313 [298, 328] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +28 [+24, +32] | 21 | 17 | 0.00 | 0.00 |
| stat:P1-conc/M15 | M15 | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [299, 329] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +29 [+25, +32] | 13 | 18 | 0.00 | 0.00 |
| sealed:P1-conc/M9 | M9sealed | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [299, 329] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +29 [+25, +32] | 14 | 19 | 0.00 | 0.00 |
| sealed:P1-conc/M5 | M5sealed | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [300, 330] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +30 [+26, +32] | 15 | 20 | 0.00 | 0.00 |
| stat:P1-conc/M5 | M5 | diag | 0.710 [0.655, 0.765] | 0.798 [0.737, 0.854] | 314 [300, 330] | +0.5 [+0.0, +1.5] | +0.4 [+0.0, +1.2] | +30 [+26, +32] | 16 | 21 | 0.00 | 0.00 |
| sealed:P2/M5 | M5sealed | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [252, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-26, -10] | 26 | 22 | 0.00 | 0.00 |
| stat:P2/M5 | M5 | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [252, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-26, -10] | 27 | 23 | 0.01 | 0.01 |
| sealed:P2/M9 | M9sealed | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [253, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-25, -10] | 28 | 24 | 0.00 | 0.00 |
| stat:P2/M15 | M15 | diag | 0.675 [0.615, 0.735] | 0.776 [0.713, 0.833] | 267 [253, 282] | -3.0 [-6.0, -0.5] | -1.8 [-3.8, -0.1] | -18 [-25, -10] | 29 | 25 | 0.01 | 0.01 |
| sealed:P2-conc/M5 | M5sealed | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 315 [300, 330] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +30 [+26, +33] | 22 | 26 | 0.00 | 0.00 |
| stat:P2-conc/M5 | M5 | diag | 0.705 [0.650, 0.760] | 0.794 [0.733, 0.851] | 315 [300, 330] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +30 [+26, +33] | 23 | 27 | 0.00 | 0.00 |
| LLM:A0 | M5 | llm | 0.680 [0.625, 0.740] | 0.762 [0.694, 0.825] | 202 [184, 222] | -2.5 [-6.0, +1.0] | -3.2 [-7.6, +0.3] | -83 [-103, -62] | 24 | 28 | 0.29 | 0.21 |
| fixed:1x8 | 1x8 | none | 0.650 [0.590, 0.710] | 0.717 [0.642, 0.787] | 150 [142, 158] | -5.5 [-10.0, -0.5] | -7.8 [-13.5, -2.6] | -135 [-152, -119] | 30 | 29 | 0.06 | 0.01 |
| fixed:2x8 | 2x8 | none | 0.650 [0.590, 0.710] | 0.725 [0.653, 0.792] | 178 [166, 192] | -5.5 [-10.0, -1.0] | -6.9 [-12.2, -2.3] | -107 [-124, -89] | 31 | 30 | 0.02 | 0.01 |
| LLM:A1 | M5 | diag+llm | 0.665 [0.605, 0.725] | 0.731 [0.657, 0.799] | 233 [211, 254] | -4.0 [-8.0, +0.0] | -6.3 [-11.8, -1.7] | -52 [-72, -33] | 25 | 31 | 0.04 | 0.01 |
| LLM:B0 | M15 | llm | 0.610 [0.559, 0.659] | 0.664 [0.601, 0.724] | 152 [139, 165] | -9.5 [-13.1, -6.0] | -13.0 [-17.7, -8.6] | -133 [-147, -119] | 32 | 32 | 0.00 | 0.00 |
| LLM:B1 | M15 | diag+llm | 0.593 [0.541, 0.646] | 0.666 [0.605, 0.724] | 184 [167, 200] | -11.2 [-14.8, -7.7] | -12.8 [-16.8, -9.0] | -101 [-116, -87] | 35 | 33 | 0.00 | 0.00 |
| fixed:1x4 | 1x4 | none | 0.547 [0.486, 0.607] | 0.607 [0.524, 0.684] | 79 [74, 84] | -16.1 [-22.1, -10.0] | -18.9 [-27.1, -11.0] | -206 [-221, -192] | 38 | 34 | 0.00 | 0.00 |
| fixed:2x4 | 2x4 | none | 0.577 [0.516, 0.637] | 0.625 [0.543, 0.704] | 91 [84, 98] | -13.1 [-19.1, -7.5] | -17.0 [-24.7, -10.0] | -194 [-208, -180] | 33 | 35 | 0.00 | 0.00 |
| fixed:4x4 | 4x4 | none | 0.580 [0.520, 0.640] | 0.625 [0.543, 0.706] | 100 [93, 108] | -12.5 [-18.0, -7.0] | -16.9 [-24.6, -9.8] | -185 [-199, -170] | 34 | 36 | 0.00 | 0.00 |
| fixed:4x2 | 4x2 | none | 0.530 [0.465, 0.595] | 0.547 [0.463, 0.629] | 56 [53, 60] | -17.5 [-24.0, -11.0] | -24.8 [-33.0, -16.8] | -229 [-244, -214] | 37 | 37 | 0.00 | 0.00 |
| fixed:8x2 | 8x2 | none | 0.555 [0.495, 0.615] | 0.561 [0.479, 0.644] | 66 [61, 71] | -15.0 [-20.5, -9.5] | -23.3 [-31.4, -15.4] | -219 [-235, -203] | 36 | 38 | 0.00 | 0.00 |
| fixed:16x1 | 16x1 | none | 0.490 [0.425, 0.555] | 0.493 [0.412, 0.577] | 43 [39, 46] | -21.5 [-28.0, -15.5] | -30.1 [-38.5, -21.8] | -242 [-257, -228] | 40 | 39 | 0.00 | 0.00 |
| fixed:1x2 | 1x2 | none | 0.490 [0.425, 0.555] | 0.505 [0.424, 0.587] | 43 [41, 46] | -21.5 [-28.0, -15.5] | -28.9 [-37.1, -20.9] | -242 [-256, -228] | 41 | 40 | 0.00 | 0.00 |
| fixed:8x1 | 8x1 | none | 0.480 [0.420, 0.545] | 0.482 [0.402, 0.563] | 38 [35, 41] | -22.5 [-28.5, -16.5] | -31.2 [-39.6, -22.9] | -247 [-262, -233] | 39 | 41 | 0.00 | 0.00 |
| fixed:2x2 | 2x2 | none | 0.495 [0.435, 0.555] | 0.492 [0.417, 0.570] | 48 [45, 52] | -21.0 [-27.5, -14.5] | -30.2 [-38.2, -22.3] | -236 [-251, -222] | 42 | 42 | 0.00 | 0.00 |
| fixed:4x1 | 4x1 | none | 0.431 [0.371, 0.492] | 0.441 [0.361, 0.522] | 34 [31, 38] | -27.6 [-34.2, -21.1] | -35.5 [-43.8, -27.0] | -251 [-265, -237] | 43 | 43 | 0.00 | 0.00 |
| fixed:1x1 | 1x1 | none | 0.386 [0.326, 0.446] | 0.378 [0.306, 0.454] | 27 [25, 30] | -32.2 [-39.2, -25.6] | -41.7 [-50.1, -33.3] | -258 [-272, -244] | 44 | 44 | 0.00 | 0.00 |
| fixed:2x1 | 2x1 | none | 0.376 [0.321, 0.431] | 0.373 [0.303, 0.446] | 31 [28, 35] | -33.2 [-40.2, -26.6] | -42.2 [-50.4, -34.2] | -254 [-268, -240] | 45 | 45 | 0.00 | 0.00 |
| oracle/M15 | M15 | none | 0.785 [0.730, 0.835] | 0.849 [0.792, 0.898] | 55 [43, 69] | +8.0 [+4.5, +12.0] | +5.5 [+2.7, +8.9] | -230 [-247, -213] | ref | ref | -- | -- |
| oracle/M5 | M5 | none | 0.765 [0.710, 0.820] | 0.829 [0.768, 0.883] | 77 [62, 94] | +6.0 [+3.0, +9.5] | +3.5 [+1.7, +5.6] | -208 [-226, -189] | ref | ref | -- | -- |

## T200: statistical and heuristic selectors minus the LLM selectors

A0/A1 against every row; B0/B1 against the rows on their own M15 menu (the post-hoc full-L96 M15 refit and heuristic/M15).

| selector | vs | Δacc mixture (pts) | p mixture | Δacc macro (pts) | p macro | Δlat (s) |
|---|---|---|---|---|---|---|
| stat:P0/M15 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +83 [+62, +103] |
| stat:P0/M15 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +52 [+33, +72] |
| stat:P0/M15 | LLM:B0 | +9.5 [+6.0, +13.1] | <0.001 | +13.0 [+8.6, +17.7] | <0.001 | +133 [+119, +147] |
| stat:P0/M15 | LLM:B1 | +11.2 [+7.7, +14.8] | <0.001 | +12.8 [+9.0, +16.8] | <0.001 | +101 [+87, +116] |
| stat:P0/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +83 [+62, +103] |
| stat:P0/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +52 [+33, +72] |
| sealed:P0/M9 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +83 [+62, +103] |
| sealed:P0/M9 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +52 [+33, +72] |
| sealed:P0/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +83 [+62, +103] |
| sealed:P0/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +52 [+33, +72] |
| stat:P1/M15 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +111 [+90, +131] |
| stat:P1/M15 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +80 [+61, +101] |
| stat:P1/M15 | LLM:B0 | +9.5 [+6.0, +13.1] | <0.001 | +13.0 [+8.6, +17.7] | <0.001 | +161 [+147, +175] |
| stat:P1/M15 | LLM:B1 | +11.2 [+7.7, +14.8] | <0.001 | +12.8 [+9.0, +16.8] | <0.001 | +130 [+116, +144] |
| stat:P1/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +111 [+90, +131] |
| stat:P1/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +80 [+61, +101] |
| sealed:P1/M9 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +111 [+90, +131] |
| sealed:P1/M9 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +80 [+61, +101] |
| sealed:P1/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +111 [+90, +131] |
| sealed:P1/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +80 [+61, +101] |
| stat:P2/M15 | LLM:A0 | -0.5 [-4.0, +3.0] | 0.791 | +1.4 [-2.1, +5.8] | 0.513 | +65 [+44, +86] |
| stat:P2/M15 | LLM:A1 | +1.0 [-3.0, +5.0] | 0.633 | +4.5 [-0.2, +10.1] | 0.122 | +34 [+14, +56] |
| stat:P2/M15 | LLM:B0 | +6.5 [+2.7, +10.2] | <0.001 | +11.2 [+6.6, +16.1] | <0.001 | +115 [+101, +130] |
| stat:P2/M15 | LLM:B1 | +8.2 [+4.7, +11.7] | <0.001 | +11.0 [+7.1, +14.9] | <0.001 | +84 [+69, +99] |
| stat:P2/M5 | LLM:A0 | -0.5 [-4.0, +3.0] | 0.791 | +1.4 [-2.1, +5.8] | 0.513 | +65 [+44, +86] |
| stat:P2/M5 | LLM:A1 | +1.0 [-3.0, +5.0] | 0.633 | +4.5 [-0.2, +10.1] | 0.122 | +34 [+14, +56] |
| sealed:P2/M9 | LLM:A0 | -0.5 [-4.0, +3.0] | 0.791 | +1.4 [-2.1, +5.8] | 0.513 | +65 [+44, +86] |
| sealed:P2/M9 | LLM:A1 | +1.0 [-3.0, +5.0] | 0.633 | +4.5 [-0.2, +10.1] | 0.122 | +34 [+14, +56] |
| sealed:P2/M5 | LLM:A0 | -0.5 [-4.0, +3.0] | 0.791 | +1.4 [-2.1, +5.8] | 0.513 | +65 [+44, +86] |
| sealed:P2/M5 | LLM:A1 | +1.0 [-3.0, +5.0] | 0.633 | +4.5 [-0.2, +10.1] | 0.122 | +34 [+14, +56] |
| stat:P3/M15 | LLM:A0 | +2.0 [-1.5, +6.0] | 0.300 | +2.8 [-0.8, +7.2] | 0.207 | +65 [+42, +88] |
| stat:P3/M15 | LLM:A1 | +3.5 [-0.5, +7.5] | 0.096 | +5.9 [+1.1, +11.5] | 0.059 | +34 [+12, +57] |
| stat:P3/M15 | LLM:B0 | +9.0 [+5.4, +12.6] | <0.001 | +12.6 [+8.1, +17.3] | <0.001 | +115 [+98, +132] |
| stat:P3/M15 | LLM:B1 | +10.7 [+7.3, +14.2] | <0.001 | +12.4 [+8.6, +16.3] | <0.001 | +83 [+68, +99] |
| stat:P3/M5 | LLM:A0 | +2.0 [-1.5, +6.0] | 0.300 | +2.8 [-0.8, +7.2] | 0.207 | +66 [+43, +89] |
| stat:P3/M5 | LLM:A1 | +3.5 [-0.5, +7.5] | 0.096 | +5.9 [+1.1, +11.5] | 0.059 | +36 [+14, +58] |
| sealed:P3/M9 | LLM:A0 | +2.0 [-1.5, +6.0] | 0.300 | +2.8 [-0.8, +7.2] | 0.207 | +65 [+42, +88] |
| sealed:P3/M9 | LLM:A1 | +3.5 [-0.5, +7.5] | 0.096 | +5.9 [+1.1, +11.5] | 0.059 | +34 [+13, +57] |
| sealed:P3/M5 | LLM:A0 | +2.0 [-1.5, +6.0] | 0.300 | +2.8 [-0.8, +7.2] | 0.207 | +66 [+43, +89] |
| sealed:P3/M5 | LLM:A1 | +3.5 [-0.5, +7.5] | 0.096 | +5.9 [+1.1, +11.5] | 0.059 | +36 [+14, +58] |
| stat:P1-conc/M15 | LLM:A0 | +3.0 [-0.5, +7.0] | 0.121 | +3.7 [+0.0, +8.1] | 0.122 | +112 [+91, +132] |
| stat:P1-conc/M15 | LLM:A1 | +4.5 [+1.0, +8.5] | 0.024 | +6.8 [+2.2, +12.2] | 0.040 | +81 [+61, +102] |
| stat:P1-conc/M15 | LLM:B0 | +10.0 [+6.6, +13.5] | <0.001 | +13.4 [+9.0, +18.1] | <0.001 | +162 [+148, +176] |
| stat:P1-conc/M15 | LLM:B1 | +11.7 [+8.3, +15.3] | <0.001 | +13.2 [+9.4, +17.2] | <0.001 | +130 [+116, +145] |
| stat:P1-conc/M5 | LLM:A0 | +3.0 [-0.5, +7.0] | 0.121 | +3.7 [+0.0, +8.1] | 0.122 | +112 [+92, +133] |
| stat:P1-conc/M5 | LLM:A1 | +4.5 [+1.0, +8.5] | 0.024 | +6.8 [+2.2, +12.2] | 0.040 | +82 [+62, +102] |
| sealed:P1-conc/M9 | LLM:A0 | +3.0 [-0.5, +7.0] | 0.121 | +3.7 [+0.0, +8.1] | 0.122 | +112 [+91, +132] |
| sealed:P1-conc/M9 | LLM:A1 | +4.5 [+1.0, +8.5] | 0.024 | +6.8 [+2.2, +12.2] | 0.040 | +81 [+61, +102] |
| sealed:P1-conc/M5 | LLM:A0 | +3.0 [-0.5, +7.0] | 0.121 | +3.7 [+0.0, +8.1] | 0.122 | +112 [+92, +133] |
| sealed:P1-conc/M5 | LLM:A1 | +4.5 [+1.0, +8.5] | 0.024 | +6.8 [+2.2, +12.2] | 0.040 | +82 [+62, +102] |
| stat:P2-conc/M15 | LLM:A0 | +3.0 [-0.5, +7.0] | 0.121 | +3.7 [+0.0, +8.1] | 0.122 | +106 [+86, +127] |
| stat:P2-conc/M15 | LLM:A1 | +4.5 [+1.0, +8.5] | 0.024 | +6.8 [+2.2, +12.2] | 0.040 | +76 [+56, +97] |
| stat:P2-conc/M15 | LLM:B0 | +10.0 [+6.6, +13.5] | <0.001 | +13.4 [+9.0, +18.1] | <0.001 | +156 [+143, +171] |
| stat:P2-conc/M15 | LLM:B1 | +11.7 [+8.3, +15.3] | <0.001 | +13.2 [+9.4, +17.2] | <0.001 | +125 [+110, +140] |
| stat:P2-conc/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +113 [+92, +133] |
| stat:P2-conc/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +82 [+62, +103] |
| sealed:P2-conc/M9 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +107 [+86, +127] |
| sealed:P2-conc/M9 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +76 [+56, +98] |
| sealed:P2-conc/M5 | LLM:A0 | +2.5 [-1.0, +6.0] | 0.170 | +3.2 [-0.3, +7.6] | 0.152 | +113 [+92, +133] |
| sealed:P2-conc/M5 | LLM:A1 | +4.0 [+0.0, +8.0] | 0.049 | +6.3 [+1.7, +11.8] | 0.047 | +82 [+62, +103] |
| heuristic/M15 | LLM:A0 | +0.5 [-3.5, +4.5] | 0.816 | +1.8 [-2.1, +6.3] | 0.439 | +42 [+17, +66] |
| heuristic/M15 | LLM:A1 | +2.0 [-2.5, +6.5] | 0.390 | +4.8 [-0.2, +10.6] | 0.106 | +11 [-12, +35] |
| heuristic/M15 | LLM:B0 | +7.5 [+3.8, +11.2] | <0.001 | +11.5 [+7.0, +16.3] | <0.001 | +92 [+75, +108] |
| heuristic/M15 | LLM:B1 | +9.2 [+6.0, +12.5] | <0.001 | +11.3 [+7.6, +15.2] | <0.001 | +60 [+46, +75] |
| heuristic/M5 | LLM:A0 | +1.5 [-2.0, +5.5] | 0.445 | +2.4 [-1.2, +6.8] | 0.283 | +44 [+20, +68] |
| heuristic/M5 | LLM:A1 | +3.0 [-1.0, +7.5] | 0.166 | +5.5 [+0.6, +11.2] | 0.072 | +13 [-9, +37] |

## T200 option distributions

E8b rows count decisions (5 per item); the others count items.

| selector | choices |
|---|---|
| fixed:1x1 | 1x1 200 |
| fixed:2x1 | 2x1 200 |
| fixed:4x1 | 4x1 200 |
| fixed:8x1 | 8x1 200 |
| fixed:16x1 | 16x1 200 |
| fixed:1x2 | 1x2 200 |
| fixed:2x2 | 2x2 200 |
| fixed:4x2 | 4x2 200 |
| fixed:8x2 | 8x2 200 |
| fixed:1x4 | 1x4 200 |
| fixed:2x4 | 2x4 200 |
| fixed:4x4 | 4x4 200 |
| fixed:1x8 | 1x8 200 |
| fixed:2x8 | 2x8 200 |
| fixed:1x16 | 1x16 200 |
| stat:P0/M15 | 1x16 200 |
| stat:P0/M5 | 1x16 200 |
| sealed:P0/M9 | 1x16 200 |
| sealed:P0/M5 | 1x16 200 |
| stat:P1/M15 | 1x16 189, 2x8 11 |
| stat:P1/M5 | 1x16 189, 2x8 11 |
| sealed:P1/M9 | 1x16 189, 2x8 11 |
| sealed:P1/M5 | 1x16 189, 2x8 11 |
| stat:P2/M15 | 1x16 123, 4x4 43, 8x2 29, 1x8 5 |
| stat:P2/M5 | 1x16 124, 4x4 43, 8x2 29, 2x8 4 |
| sealed:P2/M9 | 1x16 123, 4x4 43, 8x2 29, 1x8 5 |
| sealed:P2/M5 | 1x16 124, 4x4 43, 8x2 29, 2x8 4 |
| stat:P3/M15 | 1x16 151, 4x2 20, 8x1 18, 16x1 5, 8x2 3, 4x1 1, +2 more |
| stat:P3/M5 | 1x16 152, 16x1 24, 8x2 23, 2x8 1 |
| sealed:P3/M9 | 1x16 151, 4x2 20, 8x1 19, 16x1 5, 8x2 3, 1x8 1, +1 more |
| sealed:P3/M5 | 1x16 152, 16x1 24, 8x2 23, 2x8 1 |
| stat:P1-conc/M15 | 1x16 194, 1x8 2, 2x8 2, 4x4 1, 1x1 1 |
| stat:P1-conc/M5 | 1x16 194, 2x8 4, 4x4 1, 8x2 1 |
| sealed:P1-conc/M9 | 1x16 194, 1x8 2, 2x8 2, 4x4 1, 4x2 1 |
| sealed:P1-conc/M5 | 1x16 194, 2x8 4, 4x4 1, 8x2 1 |
| stat:P2-conc/M15 | 1x16 187, 1x8 8, 8x2 3, 1x2 1, 1x4 1 |
| stat:P2-conc/M5 | 1x16 197, 8x2 3 |
| sealed:P2-conc/M9 | 1x16 187, 1x8 10, 8x2 3 |
| sealed:P2-conc/M5 | 1x16 197, 8x2 3 |
| heuristic/M15 | 1x16 133, 4x1 67 |
| heuristic/M5 | 1x16 133, 16x1 67 |
| LLM:A0 | 1x16 84, 4x4 46, 2x8 44, 16x1 16, 8x2 10 |
| LLM:A1 | 1x16 85, 2x8 35, 4x4 32, 16x1 30, 8x2 18 |
| LLM:B0 | 1x16 250, 1x8 126, 1x1 99, 4x4 74, 4x2 65, 1x4 64, +9 more |
| LLM:B1 | 1x16 210, 1x1 190, 1x8 105, 2x8 87, 4x4 73, 16x1 66, +9 more |
| oracle/M15 | 1x1 111, 1x2 23, 4x1 11, 1x4 10, 8x1 8, 1x8 7, +9 more |
| oracle/M5 | 16x1 127, 8x2 29, 4x4 20, 2x8 13, 1x16 11 |

## C48 bank sensitivity: diagnostics removed (primary) vs included (previous report)

Primary C48 scoring bank = the LEX front bank minus each item's five diagnostic draws (240 rows; A-level bank [41] -> [36] draws); only the n x 1 options change. Class labels: the full-bank item classes (as lexfront's split-half check); rebuilding the classes on the reduced bank gives the identical partition on 48 of 48 items. Selector inputs: the statistical rule's predicted-latency table (C48 bank latency per (shard, option), leave-item-out) stays on the front estimates, like the typical times in the E8b prompt; outcomes (scoring, oracles, split-half, the heuristic's C48 training outcomes) use the scoring bank; the split-half reference is re-chosen and re-scored on the no-diagnostic halves (lex/selcmp/front_splithalf_nodiag.parquet).

| A-level option | mean p, full bank | mean p, no-diagnostic bank | max per-item abs Δp |
|---|---|---|---|
| 16x1 | 0.477 | 0.490 | 0.190 |
| 1x1 | 0.366 | 0.373 | 0.046 |
| 25x1 | 0.494 | 0.508 | 0.210 |
| 2x1 | 0.366 | 0.373 | 0.046 |
| 4x1 | 0.413 | 0.422 | 0.048 |
| 8x1 | 0.447 | 0.458 | 0.110 |

Every C48 selector under both banks (full = the previous report's scoring; ranks and P(win) over the deployable selectors including LLM:B0/B1):

| selector | acc no-diag | acc full | Δacc (pts) | C no-diag | C full | lat no-diag | lat full | rank no-diag | rank full | P(win) no-diag | P(win) full | same choices |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed:1x16 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 277 | 277 | 1 | 1 | 0.32 | 0.35 | yes |
| stat:P0/M15 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 277 | 277 | 2 | 2 | 0.00 | 0.00 | yes |
| stat:P0/M5 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 277 | 277 | 3 | 3 | 0.00 | 0.00 | yes |
| stat:P2-conc/M15 | 0.774 | 0.773 | +0.1 | 0.895 | 0.893 | 278 | 278 | 4 | 4 | 0.12 | 0.11 | yes |
| stat:P3/M15 | 0.777 | 0.777 | +0.0 | 0.877 | 0.877 | 279 | 279 | 5 | 5 | 0.08 | 0.08 | yes |
| stat:P3/M5 | 0.777 | 0.777 | +0.0 | 0.877 | 0.877 | 280 | 280 | 6 | 6 | 0.00 | 0.00 | yes |
| stat:P2-conc/M5 | 0.774 | 0.773 | +0.1 | 0.895 | 0.893 | 285 | 285 | 7 | 7 | 0.00 | 0.00 | yes |
| stat:P1/M15 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 320 | 320 | 8 | 8 | 0.00 | 0.00 | yes |
| stat:P1/M5 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 320 | 320 | 9 | 9 | 0.00 | 0.00 | yes |
| stat:P1-conc/M15 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 323 | 323 | 10 | 10 | 0.00 | 0.00 | yes |
| stat:P1-conc/M5 | 0.785 | 0.785 | +0.0 | 0.886 | 0.886 | 323 | 323 | 11 | 11 | 0.00 | 0.00 | yes |
| stat:P2/M15 | 0.763 | 0.762 | +0.1 | 0.902 | 0.900 | 266 | 266 | 12 | 12 | 0.35 | 0.32 | yes |
| heuristic/M5 | 0.753 | 0.753 | -0.0 | 0.895 | 0.895 | 273 | 273 | 13 | 13 | 0.02 | 0.02 | yes |
| stat:P2/M5 | 0.760 | 0.759 | +0.1 | 0.906 | 0.905 | 283 | 283 | 14 | 14 | 0.00 | 0.00 | yes |
| heuristic/M15 | 0.751 | 0.751 | +0.0 | 0.878 | 0.879 | 262 | 262 | 15 | 15 | 0.10 | 0.11 | yes |
| LLM:A1 | 0.713 | 0.712 | +0.2 | 0.883 | 0.881 | 240 | 240 | 16 | 17 | 0.00 | 0.00 | yes |
| LLM:A0 | 0.731 | 0.727 | +0.3 | 0.861 | 0.863 | 207 | 207 | 17 | 16 | 0.01 | 0.01 | yes |
| fixed:1x8 | 0.688 | 0.688 | +0.0 | 0.870 | 0.870 | 151 | 151 | 18 | 18 | 0.00 | 0.00 | yes |
| fixed:2x8 | 0.688 | 0.688 | +0.0 | 0.870 | 0.870 | 166 | 166 | 19 | 19 | 0.00 | 0.00 | yes |
| LLM:B1 | 0.694 | 0.693 | +0.1 | 0.827 | 0.825 | 202 | 203 | 20 | 20 | 0.00 | 0.00 | yes |
| LLM:B0 | 0.670 | 0.669 | +0.1 | 0.785 | 0.784 | 155 | 155 | 21 | 21 | 0.00 | 0.00 | yes |
| fixed:4x4 | 0.645 | 0.645 | +0.0 | 0.919 | 0.919 | 102 | 102 | 22 | 22 | 0.00 | 0.00 | yes |
| fixed:8x2 | 0.597 | 0.597 | +0.0 | 0.904 | 0.904 | 60 | 60 | 23 | 23 | 0.00 | 0.00 | yes |
| fixed:1x4 | 0.602 | 0.602 | +0.0 | 0.862 | 0.862 | 82 | 82 | 24 | 24 | 0.00 | 0.00 | yes |
| fixed:2x4 | 0.602 | 0.602 | +0.0 | 0.862 | 0.862 | 91 | 91 | 25 | 25 | 0.00 | 0.00 | yes |
| fixed:4x2 | 0.532 | 0.532 | +0.0 | 0.847 | 0.847 | 55 | 55 | 26 | 26 | 0.00 | 0.00 | yes |
| fixed:16x1 | 0.490 | 0.477 | +1.4 | 0.922 | 0.915 | 58 | 59 | 27 | 27 | 0.00 | 0.00 | yes |
| fixed:8x1 | 0.458 | 0.447 | +1.1 | 0.901 | 0.899 | 51 | 52 | 28 | 30 | 0.00 | 0.00 | yes |
| fixed:1x2 | 0.469 | 0.469 | +0.0 | 0.800 | 0.800 | 45 | 45 | 29 | 28 | 0.00 | 0.00 | yes |
| fixed:2x2 | 0.469 | 0.469 | +0.0 | 0.800 | 0.800 | 50 | 50 | 30 | 29 | 0.00 | 0.00 | yes |
| fixed:4x1 | 0.422 | 0.413 | +0.9 | 0.872 | 0.870 | 44 | 45 | 31 | 31 | 0.00 | 0.00 | yes |
| fixed:1x1 | 0.373 | 0.366 | +0.7 | 0.824 | 0.824 | 31 | 31 | 32 | 32 | 0.00 | 0.00 | yes |
| fixed:2x1 | 0.373 | 0.366 | +0.7 | 0.824 | 0.825 | 37 | 37 | 33 | 33 | 0.00 | 0.00 | yes |
| fixed:1x25 | 0.836 | 0.836 | +0.0 | 0.894 | 0.894 | 418 | 418 | ref | ref | -- | -- | yes |
| oracle/M15 | 0.829 | 0.825 | +0.4 | 0.878 | 0.876 | 128 | 129 | ref | ref | -- | -- | no |
| oracle/M5 | 0.829 | 0.825 | +0.4 | 0.879 | 0.877 | 136 | 135 | ref | ref | -- | -- | yes |
| oracle/M17 | 0.883 | 0.879 | +0.4 | 0.907 | 0.905 | 154 | 155 | ref | ref | -- | -- | no |
| splithalf/M17 | 0.765 | 0.778 | -1.3 | 0.855 | 0.856 | 145 | 146 | ref | ref | -- | -- | -- |

Heuristic rules (full bank, C48 cross-fit): M15 fold0 tau=0.8 X=4x1, fold1 tau=0.8 X=4x1, fold2 tau=0.8 X=1x2, fold3 tau=0.8 X=4x1, fold4 tau=0.6 X=4x2; M5 fold0 tau=0.8 X=8x2, fold1 tau=0.8 X=16x1, fold2 tau=0.8 X=8x2, fold3 tau=0.8 X=8x2, fold4 tau=0.8 X=8x2.

T200 heuristic fitted on L96 with the full-bank C48 outcomes (the previous report's rows) vs the primary no-diagnostic fit:

| selector | fit | tau | X | acc mixture | acc macro | latency s (macro) | Δacc mixture vs 1x16 (pts) |
|---|---|---|---|---|---|---|---|
| heuristic/M15 | no-diagnostic (primary) | 0.8 | 4x1 | 0.685 [0.625, 0.745] | 0.779 [0.716, 0.837] | 244 [221, 266] | -2.0 [-4.0, -0.5] |
| heuristic/M15 | full bank | 0.8 | 4x1 | 0.685 [0.625, 0.745] | 0.779 [0.716, 0.837] | 244 [221, 266] | -2.0 [-4.0, -0.5] |
| heuristic/M5 | no-diagnostic (primary) | 0.8 | 16x1 | 0.695 [0.635, 0.750] | 0.786 [0.723, 0.843] | 246 [223, 268] | -1.0 [-2.5, +0.0] |
| heuristic/M5 | full bank | 0.8 | 8x2 | 0.700 [0.645, 0.755] | 0.790 [0.728, 0.847] | 252 [231, 273] | -0.5 [-1.5, +0.0] |


## Heuristic rules chosen (primary: no-diagnostic C48 outcomes)

| fit | menu | tau | X | train acc | train C | train latency |
|---|---|---|---|---|---|---|
| fold0 | M15 | 0.8 | 4x1 | 0.771 | 0.894 | 293 |
| fold1 | M15 | 0.8 | 4x1 | 0.756 | 0.887 | 292 |
| fold2 | M15 | 0.8 | 1x2 | 0.752 | 0.872 | 298 |
| fold3 | M15 | 0.8 | 4x1 | 0.776 | 0.888 | 287 |
| fold4 | M15 | 0.6 | 4x2 | 0.766 | 0.864 | 266 |
| fold0 | M5 | 0.8 | 8x2 | 0.778 | 0.896 | 297 |
| fold1 | M5 | 0.8 | 16x1 | 0.754 | 0.894 | 295 |
| fold2 | M5 | 0.8 | 8x2 | 0.756 | 0.887 | 301 |
| fold3 | M5 | 0.8 | 8x2 | 0.784 | 0.893 | 293 |
| fold4 | M5 | 0.8 | 8x2 | 0.780 | 0.875 | 298 |
| all L96 -> T200 | M15 | 0.8 | 4x1 | 0.763 | 0.886 | 293 |
| all L96 -> T200 | M5 | 0.8 | 16x1 | 0.762 | 0.896 | 298 |

## Cross-fit details

| fold | train items | test items | train pools | latency feature | P1 ridge (delivered) | P2 ridge (response/eta) |
|---|---|---|---|---|---|---|
| 0 | 76 | 20 | 942 | False | 100 | 10/10 |
| 1 | 77 | 19 | 953 | False | 100 | 30/100 |
| 2 | 77 | 19 | 964 | False | 100 | 30/100 |
| 3 | 77 | 19 | 950 | False | 100 | 30/100 |
| 4 | 77 | 19 | 959 | False | 100 | 30/30 |

Out-of-fold predictions vs the C48 no-diagnostic bank p (all 15 M15 cells per item):

| predictor | MAE | squared error | mean within-item Spearman |
|---|---|---|---|
| P0 | 0.372 | 0.165 | 0.58 |
| P1 | 0.285 | 0.121 | 0.58 |
| P1-conc | 0.303 | 0.125 | 0.58 |
| P2 | 0.276 | 0.118 | 0.55 |
| P2-conc | 0.316 | 0.137 | 0.56 |
| P3 | 0.271 | 0.124 | 0.59 |
