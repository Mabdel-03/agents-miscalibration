# agent_design_v4_orcd — final report

Generated 2026-09-07 01:15 EDT. Run root `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4`. Frozen at 2026-09-06T01:42:28+00:00 (code `9918df5`, flagship `32B`, judge checkpoint `32B`). Specification: agents_scaling_local_future_state_reversal_experiment_spec_v4_0 with the pre-freeze amendment register below.

## 1. Headline

| family | executed | estimate | 95% CI | p | Holm-adjusted p | decision |
|---|---|---|---|---|---|---|
| A | yes | -0.008 | [-0.029, 0.015] | 0.5099 | 1.0000 | no rejection |
| O | yes | — | — | 0.1829 | 0.9145 | no rejection |
| R | no | — | — | 1.0000 | 1.0000 | no rejection |
| C | yes | -0.000 | [-0.000, -0.000] | 1.0000 | 1.0000 | no rejection |
| G | yes | 0.000 | [0.000, 0.000] | 0.0133 | 0.0798 | no rejection |
| M | no | — | — | 1.0000 | 1.0000 | no rejection |

Family A (VOTE_AWARE effect on VOTE@5, TEAM_FRAME averaged): **prefix** panel, n = {'hle': 200, 'bcb': 200} (the common completed rank prefix held 200 items per domain), estimate -0.008, by domain {'hle': 0.01, 'bcb': -0.025}. smallest common completed rank prefix, 200 items per domain (§9.1)
Family O (native final accuracy at B4, methods ['S_FRESH', 'S_HISTORY', 'IND_VOTE', 'DEC', 'CEN_FLAT']): **prefix** panel, global max-T p = 0.1829; per-method means S_FRESH 0.307, S_HISTORY 0.312, IND_VOTE 0.328, DEC 0.302, CEN_FLAT 0.287. smallest common completed rank prefix, 200 items per domain (§9.1)

Analysis manifest: seed 2562178283353427662, 20,000 primary resamples, code `4a9f97381326d4abe62b33ccb2d7254abe100902`, common prefix by module (aggregate) {'A': 200, 'B': 50, 'D': 0, 'E': 0, 'F': 200, 'M': 75, 'MX': 50, 'N': 50}.

## 2. Scope achieved (reconcile)

| manifest | module/lane | planned cells | cells done | planned items | items done | incomplete | suspended | generated req | aliased req | PFLOP |
|---|---|---|---|---|---|---|---|---|---|---|
| cells_1-eval_32B.7eb745a5.w10.json | eval/32B | 1 | 1 | 16 | 16 | 0 | 0 | 108 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w11.json | eval/32B | 1 | 1 | 15 | 15 | 0 | 0 | 409 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w12.json | eval/32B | 1 | 1 | 14 | 14 | 0 | 0 | 226 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w13.json | eval/32B | 1 | 1 | 16 | 16 | 0 | 0 | 581 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w14.json | eval/32B | 1 | 1 | 15 | 15 | 0 | 0 | 83 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w15.json | eval/32B | 1 | 1 | 6 | 6 | 0 | 0 | 71 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w16.json | eval/32B | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w17.json | eval/32B | 1 | 1 | 8 | 8 | 0 | 0 | 6 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w3.json | eval/32B | 1 | 1 | 11 | 11 | 0 | 0 | 715 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w4.json | eval/32B | 1 | 1 | 9 | 9 | 0 | 0 | 68 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w5.json | eval/32B | 1 | 1 | 13 | 13 | 0 | 0 | 278 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w6.json | eval/32B | 1 | 1 | 13 | 13 | 0 | 0 | 433 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w7.json | eval/32B | 1 | 1 | 14 | 14 | 0 | 0 | 187 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w8.json | eval/32B | 1 | 1 | 19 | 19 | 0 | 0 | 250 | 0 | 0 |
| cells_1-eval_32B.7eb745a5.w9.json | eval/32B | 1 | 1 | 11 | 11 | 0 | 0 | 190 | 0 | 0 |
| cells_1-eval_32B.w1.json | eval/32B | 1 | 1 | 12 | 12 | 0 | 0 | 745 | 0 | 0 |
| cells_1-eval_32B.w2.json | eval/32B | 1 | 1 | 7 | 7 | 0 | 0 | 142 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w10.json | eval/eval | 3 | 3 | 21 | 21 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w11.json | eval/eval | 2 | 2 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w12.json | eval/eval | 2 | 2 | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w13.json | eval/eval | 2 | 2 | 19 | 19 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w14.json | eval/eval | 2 | 2 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w15.json | eval/eval | 1 | 1 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w17.json | eval/eval | 1 | 1 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w3.json | eval/eval | 2 | 2 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w4.json | eval/eval | 2 | 2 | 11 | 11 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w5.json | eval/eval | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w6.json | eval/eval | 2 | 2 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w7.json | eval/eval | 2 | 2 | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w8.json | eval/eval | 1 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.7eb745a5.w9.json | eval/eval | 2 | 2 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.w1.json | eval/eval | 1 | 1 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| cells_1-eval_eval.w2.json | eval/eval | 1 | 1 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| cells_1-select_32B.7eb745a5.w10.json | eval/32B | 2 | 2 | 37 | 37 | 0 | 0 | 5438 | 0 | 296 |
| cells_1-select_32B.7eb745a5.w11.json | eval/32B | 2 | 2 | 31 | 31 | 0 | 0 | 4492 | 428 | 227 |
| cells_1-select_32B.7eb745a5.w12.json | eval/32B | 2 | 2 | 29 | 29 | 0 | 0 | 3954 | 0 | 189 |
| cells_1-select_32B.7eb745a5.w13.json | eval/32B | 2 | 2 | 35 | 35 | 0 | 0 | 2678 | 0 | 121 |
| cells_1-select_32B.7eb745a5.w14.json | eval/32B | 2 | 2 | 31 | 31 | 0 | 0 | 3966 | 0 | 222 |
| cells_1-select_32B.7eb745a5.w15.json | eval/32B | 1 | 1 | 15 | 15 | 0 | 0 | 2142 | 0 | 109 |
| cells_1-select_32B.7eb745a5.w16.json | eval/32B | 1 | 1 | 1 | 1 | 0 | 0 | 23 | 72 | 4 |
| cells_1-select_32B.7eb745a5.w17.json | eval/32B | 1 | 1 | 17 | 17 | 0 | 0 | 297 | 2284 | 123 |
| cells_1-select_32B.7eb745a5.w3.json | eval/32B | 1 | 1 | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| cells_1-select_32B.7eb745a5.w4.json | eval/32B | 1 | 1 | 20 | 20 | 0 | 0 | 3109 | 0 | 145 |
| cells_1-select_32B.7eb745a5.w5.json | eval/32B | 1 | 1 | 21 | 21 | 0 | 0 | 0 | 0 | 0 |
| cells_1-select_32B.7eb745a5.w6.json | eval/32B | 2 | 2 | 27 | 27 | 0 | 0 | 4408 | 0 | 211 |
| cells_1-select_32B.7eb745a5.w7.json | eval/32B | 2 | 2 | 29 | 29 | 0 | 0 | 4041 | 0 | 203 |
| cells_1-select_32B.7eb745a5.w8.json | eval/32B | 2 | 2 | 29 | 29 | 0 | 0 | 4010 | 0 | 193 |
| cells_1-select_32B.7eb745a5.w9.json | eval/32B | 1 | 1 | 25 | 25 | 0 | 0 | 3546 | 0 | 171 |
| cells_1-select_32B.w1.json | eval/32B | 1 | 1 | 18 | 18 | 0 | 0 | 1775 | 129 | 83 |
| cells_1-select_32B.w2.json | eval/32B | 1 | 1 | 13 | 13 | 0 | 0 | 0 | 0 | 0 |
| cells_1_32B.json | A/32B | 304 | 304 | 2000 | 2000 | 0 | 0 | 62374 | 17301 | 24209 |
| cells_1_32B.json | F/32B | 96 | 96 | 1600 | 1600 | 0 | 0 | 13590 | 500 | 4659 |
| cells_2-eval_32B.565b8afc.w1.json | eval/32B | 3 | 3 | 75 | 75 | 0 | 0 | 1024 | 0 | 0 |
| cells_2-eval_32B.7c09e7eb.w1.json | eval/32B | 3 | 3 | 75 | 75 | 0 | 0 | 825 | 0 | 0 |
| cells_2-eval_32B.a51f58bc.w1.json | eval/32B | 3 | 3 | 75 | 75 | 0 | 0 | 826 | 0 | 0 |
| cells_2-eval_32B.ae44133a.w1.json | eval/32B | 2 | 2 | 42 | 42 | 0 | 0 | 1101 | 0 | 0 |
| cells_2-eval_32B.ae44133a.w2.json | eval/32B | 1 | 1 | 8 | 8 | 0 | 0 | 118 | 0 | 0 |
| cells_2-eval_eval.565b8afc.w1.json | eval/eval | 8 | 8 | 75 | 75 | 0 | 0 | 0 | 0 | 0 |
| cells_2-eval_eval.7c09e7eb.w1.json | eval/eval | 8 | 8 | 75 | 75 | 0 | 0 | 0 | 0 | 0 |
| cells_2-eval_eval.a51f58bc.w1.json | eval/eval | 8 | 8 | 75 | 75 | 0 | 0 | 0 | 0 | 0 |
| cells_2-eval_eval.ae44133a.w1.json | eval/eval | 5 | 5 | 41 | 41 | 0 | 0 | 0 | 0 | 0 |
| cells_2-eval_eval.ae44133a.w2.json | eval/eval | 1 | 1 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| cells_2-select_32B.565b8afc.w1.json | eval/32B | 6 | 6 | 150 | 150 | 0 | 0 | 9556 | 0 | 458 |
| cells_2-select_32B.7c09e7eb.w1.json | eval/32B | 6 | 6 | 150 | 150 | 0 | 0 | 7579 | 0 | 385 |
| cells_2-select_32B.a51f58bc.w1.json | eval/32B | 6 | 6 | 150 | 150 | 0 | 0 | 11606 | 0 | 593 |
| cells_2-select_32B.ae44133a.w1.json | eval/32B | 4 | 4 | 83 | 83 | 0 | 0 | 10247 | 7170 | 861 |
| cells_2-select_32B.ae44133a.w2.json | eval/32B | 1 | 1 | 17 | 17 | 0 | 0 | 11507 | 1711 | 605 |
| cells_2-select_32B.w1.json | eval/32B | 6 | 6 | 150 | 150 | 0 | 0 | 9556 | 0 | 458 |
| cells_2_14B.json | M/14B | 60 | 60 | 450 | 450 | 0 | 0 | 14390 | 348 | 1991 |
| cells_2_32B.json | D/32B | 50 | 50 | 500 | 500 | 0 | 0 | 2500 | 4500 | 2011 |
| cells_2_32B.json | E/32B | 75 | 75 | 600 | 600 | 0 | 0 | 22799 | 415 | 7109 |
| cells_2_32B.json | N/32B | 220 | 220 | 1700 | 1700 | 0 | 0 | 21415 | 29538 | 16064 |
| cells_2_4B.json | M/4B | 60 | 60 | 450 | 450 | 0 | 0 | 14984 | 300 | 654 |
| cells_2_8B.json | M/8B | 60 | 60 | 450 | 450 | 0 | 0 | 8879 | 866 | 844 |
| cells_3-eval_32B.411daa59.w1.json | eval/32B | 2 | 2 | 48 | 48 | 0 | 0 | 56 | 0 | 0 |
| cells_3-eval_32B.411daa59.w3.json | eval/32B | 1 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| cells_3-eval_32B.f99e3d23.w1.json | eval/32B | 2 | 2 | 50 | 50 | 0 | 0 | 557 | 0 | 0 |
| cells_3-eval_eval.411daa59.w1.json | eval/eval | 5 | 5 | 46 | 46 | 0 | 0 | 0 | 0 | 0 |
| cells_3-eval_eval.411daa59.w3.json | eval/eval | 1 | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| cells_3-eval_eval.f99e3d23.w1.json | eval/eval | 5 | 5 | 50 | 50 | 0 | 0 | 0 | 0 | 0 |
| cells_3-select_32B.411daa59.w1.json | eval/32B | 4 | 4 | 94 | 94 | 0 | 0 | 3337 | 9637 | 641 |
| cells_3-select_32B.411daa59.w3.json | eval/32B | 1 | 1 | 6 | 6 | 0 | 0 | 1064 | 0 | 53 |
| cells_3-select_32B.f99e3d23.w1.json | eval/32B | 4 | 4 | 100 | 100 | 0 | 0 | 10006 | 0 | 513 |
| cells_3_32B.json | B/32B | 152 | 152 | 1000 | 1000 | 0 | 0 | 3569 | 34255 | 9394 |
| cells_3_32B.json | MX/32B | 120 | 120 | 900 | 900 | 0 | 0 | 3964 | 23099 | 8435 |
| cells_3_8B.json | MX/8B | 120 | 120 | 900 | 900 | 0 | 0 | 14667 | 16787 | 2694 |
| cells_pilot-eval_32B.9b110309.w2.json | eval/32B | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-eval_32B.9b110309.w3.json | eval/32B | 1 | 1 | 10 | 10 | 0 | 0 | 46 | 0 | 0 |
| cells_pilot-eval_32B.9b110309.w4.json | eval/32B | 1 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-eval_32B.json | eval/32B | 1 | 1 | 8 | 8 | 0 | 0 | 63 | 0 | 0 |
| cells_pilot-eval_eval.9b110309.w2.json | eval/eval | 1 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-eval_eval.9b110309.w3.json | eval/eval | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-eval_eval.9b110309.w4.json | eval/eval | 1 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-eval_eval.json | eval/eval | 1 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| cells_pilot-select_32B.9b110309.w3.json | eval/32B | 1 | 1 | 18 | 18 | 0 | 0 | 1960 | 508 | 143 |
| cells_pilot-select_32B.json | eval/32B | 1 | 1 | 18 | 18 | 0 | 0 | 512 | 16 | 24 |
| cells_pilot_32B.json | A/32B | 11 | 11 | 84 | 84 | 0 | 0 | 82 | 2222 | 716 |
| cells_pilot_32B.json | F/32B | 4 | 4 | 80 | 80 | 0 | 0 | 0 | 0 | 0 |

Run totals over 1,486 distinct cells: generated requests 303,492; aliased (content-addressed reuse) 149,864; logical work 84.6 EFLOP.

HLE judge: 9114 judged answers, ambiguous rate 0.0000. VOTE selections: 21631; no-valid-candidate rate 0.089; all-singleton (tie-broken by HMAC) rate 0.441.

## 3. Statistics (tables/stats.md, verbatim)

### study_v4 confirmatory statistics

analysis seed `2562178283353427662` from study seed; 20,000 primary resamples; code `4a9f97381326d4abe62b33ccb2d7254abe100902`; tables: {'selections': 42679, 'banks': 1680, 'episodes': 8760}

## Six primary families (Holm, α = 0.05)

| family | executed | estimate | 95% CI | n (hle/bcb) | p | Holm p | reject | note |
|---|---|---|---|---|---|---|---|---|
| A | yes | -0.008 | [-0.029, 0.015] | 200/200 | 0.5099 | 1.0000 | no | panel: prefix (smallest common completed rank prefix, 200 items per domain (§9.1)). 0.5*[(Y01-Y00)+(Y11-Y10)] on VOTE@5 of the four framing banks; framing = <TEAM_FRAME><VOTE_AWARE> |
| O | yes | — | — | 200/200 | 0.1829 | 0.9145 | no | panel: prefix (smallest common completed rank prefix, 200 items per domain (§9.1)). omnibus equality of native final accuracy across ['S_FRESH', 'S_HISTORY', 'IND_VOTE', 'DEC', 'CEN_FLAT'] at B4 on N_main; max-T over all pairs |
| R | no | — | — | —/— | 1.0000 | 1.0000 | no | family R (recursive scaling, L32 D2 vs D1) not executed: amendment register (no RLM frontier) |
| C | yes | -0.000 | [-0.000, -0.000] | —/— | 1.0000 | 1.0000 | no | family C: one-sided (improvement > 0) from C_summary.json;  |
| G | yes | 0.000 | [0.000, 0.000] | —/— | 0.0133 | 0.0798 | no | family G: one-sided (improvement > 0) from G_confirm_summary.json;  |
| M | no | — | — | —/— | 1.0000 | 1.0000 | no | family M (content use / activation edits) not executed: amendment register |

### A — awareness (primary panel: prefix — smallest common completed rank prefix, 200 items per domain (§9.1); prefix n = 200 per domain; complete-case n = {'hle': 200, 'bcb': 200})

| panel | estimate | by domain | SE | t | p (two-sided) | 95% CI (studentized) | conservative primary CI (alpha .05/6) |
|---|---|---|---|---|---|---|---|
| prefix | -0.008 | {'hle': 0.01, 'bcb': -0.025} | 0.011 | -0.663 | 0.5099 | [-0.029, 0.015] | [-0.037, 0.023] |
| complete | -0.008 | {'hle': 0.01, 'bcb': -0.025} | 0.011 | -0.663 | 0.5070 | [-0.029, 0.015] | [-0.037, 0.023] |

S1 (secondary, prefix panel): TEAM_FRAME effect 0.003 [-0.020, 0.025]; interaction 0.005 [-0.039, 0.046]; cell means 00=0.302, 01=0.292, 10=0.302, 11=0.297

### O — orchestration (primary panel: prefix — smallest common completed rank prefix, 200 items per domain (§9.1); prefix n = 200 per domain; prefix max-T p = 0.1829; complete-case p = 0.1846)

| method | mean | by domain | 95% CI |
|---|---|---|---|
| S_FRESH | 0.307 | {'hle': 0.185, 'bcb': 0.43} | [0.265, 0.352] |
| S_HISTORY | 0.312 | {'hle': 0.165, 'bcb': 0.46} | [0.270, 0.357] |
| IND_VOTE | 0.328 | {'hle': 0.175, 'bcb': 0.48} | [0.283, 0.370] |
| DEC | 0.302 | {'hle': 0.16, 'bcb': 0.445} | [0.260, 0.345] |
| CEN_FLAT | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.245, 0.330] |

| contrast | estimate | SE | t | max-T adjusted p | within-O simultaneous 95% CI (secondary) | conservative primary CI (alpha .05/6) |
|---|---|---|---|---|---|---|
| S_FRESH - S_HISTORY | -0.005 | 0.017 | -0.295 | 0.9984 | [-0.051, 0.041] | [-0.061, 0.051] |
| S_FRESH - IND_VOTE | -0.020 | 0.014 | -1.470 | 0.5831 | [-0.057, 0.017] | [-0.065, 0.025] |
| S_FRESH - DEC | 0.005 | 0.016 | 0.308 | 0.9976 | [-0.039, 0.049] | [-0.048, 0.058] |
| S_FRESH - CEN_FLAT | 0.020 | 0.018 | 1.131 | 0.7870 | [-0.028, 0.068] | [-0.038, 0.078] |
| S_HISTORY - IND_VOTE | -0.015 | 0.018 | -0.847 | 0.9127 | [-0.063, 0.033] | [-0.073, 0.043] |
| S_HISTORY - DEC | 0.010 | 0.015 | 0.648 | 0.9670 | [-0.032, 0.052] | [-0.041, 0.061] |
| S_HISTORY - CEN_FLAT | 0.025 | 0.019 | 1.337 | 0.6689 | [-0.026, 0.076] | [-0.036, 0.086] |
| IND_VOTE - DEC | 0.025 | 0.017 | 1.509 | 0.5553 | [-0.020, 0.070] | [-0.029, 0.079] |
| IND_VOTE - CEN_FLAT | 0.040 | 0.018 | 2.188 | 0.1829 | [-0.010, 0.090] | [-0.020, 0.100] |
| DEC - CEN_FLAT | 0.015 | 0.019 | 0.773 | 0.9366 | [-0.038, 0.068] | [-0.049, 0.079] |

The within-O interval controls only inside family O and is labelled secondary (spec §9.3); the conservative primary interval uses the same all-pair max-T distribution at family alpha .05/6.

## Secondary tables (estimates with percentile intervals; not Holm-adjusted)

### S4_pass_at_k_by_framing.pass_at_1

| checkpoint | framing | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|
| 32B | 00 | 0.269 | {'hle': 0.127, 'bcb': 0.4115} | [0.236, 0.303] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 0.272 | {'hle': 0.1275, 'bcb': 0.41650000000000004} | [0.238, 0.306] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 0.266 | {'hle': 0.1295, 'bcb': 0.402} | [0.232, 0.301] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 0.276 | {'hle': 0.133, 'bcb': 0.41950000000000004} | [0.240, 0.313] | {'hle': 200, 'bcb': 200} |

### S4_pass_at_k_by_framing.pass_at_2

| checkpoint | framing | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|
| 32B | 00 | 0.329 | {'hle': 0.17522222222222222, 'bcb': 0.483} | [0.290, 0.367] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 0.331 | {'hle': 0.1738888888888889, 'bcb': 0.4877777777777778} | [0.291, 0.370] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 0.325 | {'hle': 0.1781111111111111, 'bcb': 0.4723333333333333} | [0.286, 0.362] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 0.331 | {'hle': 0.17855555555555555, 'bcb': 0.4834444444444444} | [0.293, 0.372] | {'hle': 200, 'bcb': 200} |

### S4_pass_at_k_by_framing.pass_at_3

| checkpoint | framing | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|
| 32B | 00 | 0.362 | {'hle': 0.205375, 'bcb': 0.5185833333333334} | [0.321, 0.402] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 0.362 | {'hle': 0.2021666666666667, 'bcb': 0.5215416666666667} | [0.322, 0.402] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 0.357 | {'hle': 0.20654166666666665, 'bcb': 0.506625} | [0.318, 0.398] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 0.359 | {'hle': 0.20404166666666668, 'bcb': 0.51325} | [0.318, 0.400] | {'hle': 200, 'bcb': 200} |

### S4_pass_at_k_by_framing.pass_at_5

| checkpoint | framing | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|
| 32B | 00 | 0.400 | {'hle': 0.24353174603174602, 'bcb': 0.5566666666666668} | [0.358, 0.442] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 0.398 | {'hle': 0.23952380952380953, 'bcb': 0.556686507936508} | [0.356, 0.439] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 0.393 | {'hle': 0.24107142857142858, 'bcb': 0.544047619047619} | [0.350, 0.436] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 0.390 | {'hle': 0.23498015873015873, 'bcb': 0.5444642857142857} | [0.347, 0.432] | {'hle': 200, 'bcb': 200} |

### S4_pass_at_k_by_framing.pass_at_10

| checkpoint | framing | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|
| 32B | 00 | 0.448 | {'hle': 0.295, 'bcb': 0.6} | [0.400, 0.495] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 0.445 | {'hle': 0.3, 'bcb': 0.59} | [0.395, 0.492] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 0.440 | {'hle': 0.29, 'bcb': 0.59} | [0.393, 0.485] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 0.425 | {'hle': 0.28, 'bcb': 0.57} | [0.375, 0.470] | {'hle': 200, 'bcb': 200} |

### S4_bank_prefix_selection_accuracy

| checkpoint | framing | prefix_k | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|
| 32B | 00 | 1 | JUDGE_BEST | 0.277 | {'hle': 0.12, 'bcb': 0.435} | [0.235, 0.318] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 1 | VOTE | 0.277 | {'hle': 0.12, 'bcb': 0.435} | [0.237, 0.318] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 10 | JUDGE_BEST | 0.275 | {'hle': 0.15, 'bcb': 0.4} | [0.235, 0.318] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 10 | VOTE | 0.287 | {'hle': 0.15, 'bcb': 0.425} | [0.247, 0.328] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 2 | JUDGE_BEST | 0.278 | {'hle': 0.115, 'bcb': 0.44} | [0.235, 0.320] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 2 | VOTE | 0.302 | {'hle': 0.145, 'bcb': 0.46} | [0.260, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 3 | JUDGE_BEST | 0.287 | {'hle': 0.135, 'bcb': 0.44} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 3 | VOTE | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.258, 0.343] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 5 | JUDGE_BEST | 0.290 | {'hle': 0.14, 'bcb': 0.44} | [0.250, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 5 | VOTE | 0.302 | {'hle': 0.16, 'bcb': 0.445} | [0.263, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 9 | JUDGE_BEST | 0.280 | {'hle': 0.15, 'bcb': 0.41} | [0.240, 0.323] | {'hle': 200, 'bcb': 200} |
| 32B | 00 | 9 | VOTE | 0.290 | {'hle': 0.15, 'bcb': 0.43} | [0.247, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 1 | JUDGE_BEST | 0.285 | {'hle': 0.15, 'bcb': 0.42} | [0.245, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 1 | VOTE | 0.285 | {'hle': 0.15, 'bcb': 0.42} | [0.242, 0.328] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 10 | JUDGE_BEST | 0.307 | {'hle': 0.165, 'bcb': 0.45} | [0.265, 0.353] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 10 | VOTE | 0.295 | {'hle': 0.16, 'bcb': 0.43} | [0.253, 0.338] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 2 | JUDGE_BEST | 0.273 | {'hle': 0.155, 'bcb': 0.39} | [0.230, 0.315] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 2 | VOTE | 0.307 | {'hle': 0.175, 'bcb': 0.44} | [0.263, 0.350] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 3 | JUDGE_BEST | 0.270 | {'hle': 0.15, 'bcb': 0.39} | [0.230, 0.315] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 3 | VOTE | 0.295 | {'hle': 0.16, 'bcb': 0.43} | [0.253, 0.340] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 5 | JUDGE_BEST | 0.288 | {'hle': 0.17, 'bcb': 0.405} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 5 | VOTE | 0.292 | {'hle': 0.175, 'bcb': 0.41} | [0.250, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 9 | JUDGE_BEST | 0.302 | {'hle': 0.17, 'bcb': 0.435} | [0.260, 0.348] | {'hle': 200, 'bcb': 200} |
| 32B | 01 | 9 | VOTE | 0.297 | {'hle': 0.165, 'bcb': 0.43} | [0.253, 0.342] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 1 | JUDGE_BEST | 0.270 | {'hle': 0.135, 'bcb': 0.405} | [0.230, 0.313] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 1 | VOTE | 0.270 | {'hle': 0.135, 'bcb': 0.405} | [0.233, 0.312] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 10 | JUDGE_BEST | 0.295 | {'hle': 0.135, 'bcb': 0.455} | [0.255, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 10 | VOTE | 0.297 | {'hle': 0.18, 'bcb': 0.415} | [0.258, 0.343] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 2 | JUDGE_BEST | 0.273 | {'hle': 0.14, 'bcb': 0.405} | [0.232, 0.315] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 2 | VOTE | 0.282 | {'hle': 0.15, 'bcb': 0.415} | [0.242, 0.323] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 3 | JUDGE_BEST | 0.267 | {'hle': 0.125, 'bcb': 0.41} | [0.225, 0.310] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 3 | VOTE | 0.292 | {'hle': 0.145, 'bcb': 0.44} | [0.253, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 5 | JUDGE_BEST | 0.282 | {'hle': 0.125, 'bcb': 0.44} | [0.242, 0.323] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 5 | VOTE | 0.302 | {'hle': 0.175, 'bcb': 0.43} | [0.260, 0.347] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 9 | JUDGE_BEST | 0.290 | {'hle': 0.13, 'bcb': 0.45} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | 10 | 9 | VOTE | 0.292 | {'hle': 0.18, 'bcb': 0.405} | [0.250, 0.333] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 1 | JUDGE_BEST | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.250, 0.333] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 1 | VOTE | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.250, 0.333] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 10 | JUDGE_BEST | 0.305 | {'hle': 0.165, 'bcb': 0.445} | [0.263, 0.347] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 10 | VOTE | 0.297 | {'hle': 0.175, 'bcb': 0.42} | [0.255, 0.340] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 2 | JUDGE_BEST | 0.302 | {'hle': 0.17, 'bcb': 0.435} | [0.263, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 2 | VOTE | 0.315 | {'hle': 0.185, 'bcb': 0.445} | [0.273, 0.357] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 3 | JUDGE_BEST | 0.292 | {'hle': 0.15, 'bcb': 0.435} | [0.250, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 3 | VOTE | 0.307 | {'hle': 0.18, 'bcb': 0.435} | [0.263, 0.352] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 5 | JUDGE_BEST | 0.292 | {'hle': 0.16, 'bcb': 0.425} | [0.250, 0.335] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 5 | VOTE | 0.297 | {'hle': 0.18, 'bcb': 0.415} | [0.255, 0.340] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 9 | JUDGE_BEST | 0.305 | {'hle': 0.165, 'bcb': 0.445} | [0.263, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | 11 | 9 | VOTE | 0.307 | {'hle': 0.185, 'bcb': 0.43} | [0.265, 0.352] | {'hle': 200, 'bcb': 200} |

### S4_architecture_selection.selected_accuracy

| checkpoint | method | N | B | pool | selector | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.245, 0.333] | {'hle': 200, 'bcb': 200} |
| 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.292 | {'hle': 0.145, 'bcb': 0.44} | [0.253, 0.333] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | VOTE | 0.295 | {'hle': 0.155, 'bcb': 0.435} | [0.253, 0.338] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.155, 'bcb': 0.445} | [0.258, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.302 | {'hle': 0.16, 'bcb': 0.445} | [0.260, 0.345] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.307 | {'hle': 0.175, 'bcb': 0.44} | [0.265, 0.353] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.328 | {'hle': 0.175, 'bcb': 0.48} | [0.285, 0.370] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | JUDGE_BEST | 0.305 | {'hle': 0.16, 'bcb': 0.45} | [0.263, 0.348] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | VOTE | 0.307 | {'hle': 0.185, 'bcb': 0.43} | [0.265, 0.353] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | JUDGE_BEST | 0.307 | {'hle': 0.16, 'bcb': 0.455} | [0.265, 0.352] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | VOTE | 0.312 | {'hle': 0.165, 'bcb': 0.46} | [0.268, 0.355] | {'hle': 200, 'bcb': 200} |

### S4_architecture_selection.candidate_mean

| checkpoint | method | N | B | pool | selector | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.245, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.287 | {'hle': 0.13644444444444445, 'bcb': 0.43766666666666665} | [0.249, 0.325] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | VOTE | 0.287 | {'hle': 0.13644444444444445, 'bcb': 0.43766666666666665} | [0.246, 0.326] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.296 | {'hle': 0.145, 'bcb': 0.447} | [0.255, 0.337] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.296 | {'hle': 0.145, 'bcb': 0.447} | [0.256, 0.338] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.277 | {'hle': 0.133359375, 'bcb': 0.4209375} | [0.244, 0.312] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.277 | {'hle': 0.133359375, 'bcb': 0.4209375} | [0.243, 0.312] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | JUDGE_BEST | 0.271 | {'hle': 0.129765625, 'bcb': 0.413046875} | [0.237, 0.306] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | VOTE | 0.271 | {'hle': 0.129765625, 'bcb': 0.413046875} | [0.235, 0.305] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | JUDGE_BEST | 0.303 | {'hle': 0.151328125, 'bcb': 0.454296875} | [0.265, 0.344] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | VOTE | 0.303 | {'hle': 0.151328125, 'bcb': 0.454296875} | [0.265, 0.345] | {'hle': 200, 'bcb': 200} |

### S4_architecture_selection.oracle_coverage

| checkpoint | method | N | B | pool | selector | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.245, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.245, 0.330] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.422 | {'hle': 0.28, 'bcb': 0.565} | [0.375, 0.468] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | VOTE | 0.422 | {'hle': 0.28, 'bcb': 0.565} | [0.378, 0.470] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.318 | {'hle': 0.175, 'bcb': 0.46} | [0.275, 0.363] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.318 | {'hle': 0.175, 'bcb': 0.46} | [0.275, 0.358] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.542 | {'hle': 0.43, 'bcb': 0.655} | [0.495, 0.590] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.542 | {'hle': 0.43, 'bcb': 0.655} | [0.492, 0.588] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | JUDGE_BEST | 0.532 | {'hle': 0.42, 'bcb': 0.645} | [0.485, 0.578] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | VOTE | 0.532 | {'hle': 0.42, 'bcb': 0.645} | [0.485, 0.583] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | JUDGE_BEST | 0.365 | {'hle': 0.25, 'bcb': 0.48} | [0.320, 0.412] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | VOTE | 0.365 | {'hle': 0.25, 'bcb': 0.48} | [0.320, 0.412] | {'hle': 200, 'bcb': 200} |

### S4_architecture_selection.selection_gap

| checkpoint | method | N | B | pool | selector | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.000 | {'hle': 0.0, 'bcb': 0.0} | [0.000, 0.000] | {'hle': 200, 'bcb': 200} |
| 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.000 | {'hle': 0.0, 'bcb': 0.0} | [0.000, 0.000] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.130 | {'hle': 0.135, 'bcb': 0.125} | [0.098, 0.163] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | archive | VOTE | 0.128 | {'hle': 0.125, 'bcb': 0.13} | [0.095, 0.160] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.018 | {'hle': 0.02, 'bcb': 0.015} | [0.005, 0.033] | {'hle': 200, 'bcb': 200} |
| 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.015 | {'hle': 0.015, 'bcb': 0.015} | [0.005, 0.028] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.235 | {'hle': 0.255, 'bcb': 0.215} | [0.195, 0.275] | {'hle': 200, 'bcb': 200} |
| 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.215 | {'hle': 0.255, 'bcb': 0.175} | [0.175, 0.255] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | JUDGE_BEST | 0.228 | {'hle': 0.26, 'bcb': 0.195} | [0.185, 0.270] | {'hle': 200, 'bcb': 200} |
| 32B | S_FRESH | 5 | 4 | archive | VOTE | 0.225 | {'hle': 0.235, 'bcb': 0.215} | [0.190, 0.265] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | JUDGE_BEST | 0.057 | {'hle': 0.09, 'bcb': 0.025} | [0.035, 0.080] | {'hle': 200, 'bcb': 200} |
| 32B | S_HISTORY | 5 | 4 | archive | VOTE | 0.053 | {'hle': 0.085, 'bcb': 0.02} | [0.030, 0.075] | {'hle': 200, 'bcb': 200} |

### S5_budget_response_native_final

| module | checkpoint | method | N | B | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|
| A | 32B | CEN_FLAT | 5 | 4 | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| A | 32B | DEC | 5 | 4 | 0.302 | {'hle': 0.16, 'bcb': 0.445} | [0.263, 0.347] | {'hle': 200, 'bcb': 200} |
| A | 32B | IND_VOTE | 5 | 4 | 0.328 | {'hle': 0.175, 'bcb': 0.48} | [0.283, 0.373] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_FRESH | 5 | 4 | 0.307 | {'hle': 0.185, 'bcb': 0.43} | [0.263, 0.350] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_HISTORY | 5 | 4 | 0.312 | {'hle': 0.165, 'bcb': 0.46} | [0.270, 0.355] | {'hle': 200, 'bcb': 200} |
| B | 32B | CEN_FLAT | 5 | 1 | 0.260 | {'hle': 0.1, 'bcb': 0.42} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| B | 32B | CEN_FLAT | 5 | 2 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 1 | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 2 | 0.340 | {'hle': 0.22, 'bcb': 0.46} | [0.260, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 1 | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 2 | 0.350 | {'hle': 0.22, 'bcb': 0.48} | [0.260, 0.440] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 1 | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 2 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 1 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 2 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 1 | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 4 | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.320] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 9 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 1 | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 4 | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 9 | 4 | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 1 | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 9 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 1 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 4 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 9 | 4 | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 1 | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.350] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 4 | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 9 | 4 | 0.330 | {'hle': 0.26, 'bcb': 0.4} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 1 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 4 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 9 | 4 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |

### S5_budget_response_selected

| module | checkpoint | method | N | B | pool_kind | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.242, 0.330] | {'hle': 200, 'bcb': 200} |
| A | 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.287 | {'hle': 0.155, 'bcb': 0.42} | [0.247, 0.330] | {'hle': 200, 'bcb': 200} |
| A | 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.292 | {'hle': 0.145, 'bcb': 0.44} | [0.250, 0.335] | {'hle': 200, 'bcb': 200} |
| A | 32B | DEC | 5 | 4 | archive | VOTE | 0.295 | {'hle': 0.155, 'bcb': 0.435} | [0.253, 0.338] | {'hle': 200, 'bcb': 200} |
| A | 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.155, 'bcb': 0.445} | [0.258, 0.345] | {'hle': 200, 'bcb': 200} |
| A | 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.302 | {'hle': 0.16, 'bcb': 0.445} | [0.260, 0.345] | {'hle': 200, 'bcb': 200} |
| A | 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.307 | {'hle': 0.175, 'bcb': 0.44} | [0.268, 0.352] | {'hle': 200, 'bcb': 200} |
| A | 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.328 | {'hle': 0.175, 'bcb': 0.48} | [0.283, 0.370] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_FRESH | 5 | 4 | archive | JUDGE_BEST | 0.305 | {'hle': 0.16, 'bcb': 0.45} | [0.260, 0.345] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_FRESH | 5 | 4 | archive | VOTE | 0.307 | {'hle': 0.185, 'bcb': 0.43} | [0.263, 0.350] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_HISTORY | 5 | 4 | archive | JUDGE_BEST | 0.307 | {'hle': 0.16, 'bcb': 0.455} | [0.265, 0.350] | {'hle': 200, 'bcb': 200} |
| A | 32B | S_HISTORY | 5 | 4 | archive | VOTE | 0.312 | {'hle': 0.165, 'bcb': 0.46} | [0.272, 0.358] | {'hle': 200, 'bcb': 200} |
| B | 32B | CEN_FLAT | 5 | 1 | native | JUDGE_BEST | 0.260 | {'hle': 0.1, 'bcb': 0.42} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| B | 32B | CEN_FLAT | 5 | 1 | native | VOTE | 0.260 | {'hle': 0.1, 'bcb': 0.42} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| B | 32B | CEN_FLAT | 5 | 2 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | CEN_FLAT | 5 | 2 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 1 | archive | JUDGE_BEST | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.250, 0.420] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 1 | archive | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 1 | latest_slots | JUDGE_BEST | 0.340 | {'hle': 0.22, 'bcb': 0.46} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 1 | latest_slots | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 2 | archive | JUDGE_BEST | 0.320 | {'hle': 0.18, 'bcb': 0.46} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 2 | archive | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 2 | latest_slots | JUDGE_BEST | 0.340 | {'hle': 0.22, 'bcb': 0.46} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | DEC | 5 | 2 | latest_slots | VOTE | 0.340 | {'hle': 0.22, 'bcb': 0.46} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 1 | archive | JUDGE_BEST | 0.330 | {'hle': 0.24, 'bcb': 0.42} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 1 | archive | VOTE | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 2 | archive | JUDGE_BEST | 0.310 | {'hle': 0.22, 'bcb': 0.4} | [0.230, 0.400] | {'hle': 50, 'bcb': 50} |
| B | 32B | IND_VOTE | 5 | 2 | archive | VOTE | 0.350 | {'hle': 0.22, 'bcb': 0.48} | [0.260, 0.440] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 1 | archive | JUDGE_BEST | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 1 | archive | VOTE | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 2 | archive | JUDGE_BEST | 0.310 | {'hle': 0.22, 'bcb': 0.4} | [0.230, 0.400] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_FRESH | 5 | 2 | archive | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 1 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 1 | archive | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 2 | archive | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| B | 32B | S_HISTORY | 5 | 2 | archive | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 1 | native | JUDGE_BEST | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 1 | native | VOTE | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 4 | native | JUDGE_BEST | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 3 | 4 | native | VOTE | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 9 | 4 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| MX | 32B | CEN_FLAT | 9 | 4 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 1 | archive | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 1 | archive | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 1 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.230, 0.400] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 1 | latest_slots | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 3 | 4 | latest_slots | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 9 | 4 | archive | VOTE | 0.330 | {'hle': 0.2, 'bcb': 0.46} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 9 | 4 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.220, 0.400] | {'hle': 50, 'bcb': 50} |
| MX | 32B | DEC | 9 | 4 | latest_slots | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 1 | archive | JUDGE_BEST | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 1 | archive | VOTE | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 32B | IND_VOTE | 9 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 1 | native | JUDGE_BEST | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 1 | native | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 4 | native | JUDGE_BEST | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 3 | 4 | native | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 9 | 4 | native | JUDGE_BEST | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| MX | 8B | CEN_FLAT | 9 | 4 | native | VOTE | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 1 | archive | JUDGE_BEST | 0.270 | {'hle': 0.18, 'bcb': 0.36} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 1 | archive | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 1 | latest_slots | JUDGE_BEST | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.170, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 1 | latest_slots | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.350] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 4 | archive | JUDGE_BEST | 0.270 | {'hle': 0.18, 'bcb': 0.36} | [0.190, 0.350] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 4 | archive | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 4 | latest_slots | JUDGE_BEST | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 3 | 4 | latest_slots | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 9 | 4 | archive | JUDGE_BEST | 0.320 | {'hle': 0.24, 'bcb': 0.4} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 9 | 4 | archive | VOTE | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 9 | 4 | latest_slots | JUDGE_BEST | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| MX | 8B | DEC | 9 | 4 | latest_slots | VOTE | 0.330 | {'hle': 0.26, 'bcb': 0.4} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 1 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 1 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 4 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 9 | 4 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| MX | 8B | IND_VOTE | 9 | 4 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |

### S3_m_cross_native_final

| checkpoint | method | N | B | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 3 | 1 | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.320] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 1 | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 1 | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 1 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 4 | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 9 | 4 | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 1 | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 4 | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 9 | 4 | 0.330 | {'hle': 0.26, 'bcb': 0.4} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 1 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 4 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 9 | 4 | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |

### S3_m_cross_selected

| checkpoint | method | N | B | pool_kind | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 3 | 1 | native | JUDGE_BEST | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 1 | native | VOTE | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | native | JUDGE_BEST | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | native | VOTE | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 1 | archive | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 1 | archive | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 1 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.230, 0.400] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 1 | latest_slots | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | latest_slots | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | archive | VOTE | 0.330 | {'hle': 0.2, 'bcb': 0.46} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.220, 0.400] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | latest_slots | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 1 | archive | JUDGE_BEST | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 1 | archive | VOTE | 0.280 | {'hle': 0.18, 'bcb': 0.38} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 1 | native | JUDGE_BEST | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 1 | native | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 4 | native | JUDGE_BEST | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 3 | 4 | native | VOTE | 0.290 | {'hle': 0.2, 'bcb': 0.38} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 9 | 4 | native | JUDGE_BEST | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | CEN_FLAT | 9 | 4 | native | VOTE | 0.290 | {'hle': 0.22, 'bcb': 0.36} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 1 | archive | JUDGE_BEST | 0.270 | {'hle': 0.18, 'bcb': 0.36} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 1 | archive | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.350] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 1 | latest_slots | JUDGE_BEST | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.350] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 1 | latest_slots | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 4 | archive | JUDGE_BEST | 0.270 | {'hle': 0.18, 'bcb': 0.36} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 4 | archive | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.350] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 4 | latest_slots | JUDGE_BEST | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 3 | 4 | latest_slots | VOTE | 0.260 | {'hle': 0.16, 'bcb': 0.36} | [0.180, 0.340] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 9 | 4 | archive | JUDGE_BEST | 0.320 | {'hle': 0.24, 'bcb': 0.4} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 9 | 4 | archive | VOTE | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 9 | 4 | latest_slots | JUDGE_BEST | 0.340 | {'hle': 0.26, 'bcb': 0.42} | [0.250, 0.430] | {'hle': 50, 'bcb': 50} |
| 8B | DEC | 9 | 4 | latest_slots | VOTE | 0.330 | {'hle': 0.26, 'bcb': 0.4} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 1 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 1 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 4 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 9 | 4 | archive | JUDGE_BEST | 0.280 | {'hle': 0.2, 'bcb': 0.36} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 8B | IND_VOTE | 9 | 4 | archive | VOTE | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |

### S3_checkpoint_panel_native_final

| checkpoint | method | N | B | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|
| 14B | CEN_FLAT | 5 | 4 | 0.273 | {'hle': 0.13333333333333333, 'bcb': 0.41333333333333333} | [0.207, 0.340] | {'hle': 75, 'bcb': 75} |
| 14B | DEC | 5 | 4 | 0.293 | {'hle': 0.14666666666666667, 'bcb': 0.44} | [0.227, 0.367] | {'hle': 75, 'bcb': 75} |
| 14B | IND_VOTE | 5 | 4 | 0.307 | {'hle': 0.16, 'bcb': 0.4533333333333333} | [0.240, 0.380] | {'hle': 75, 'bcb': 75} |
| 4B | CEN_FLAT | 5 | 4 | 0.267 | {'hle': 0.17333333333333334, 'bcb': 0.36} | [0.200, 0.340] | {'hle': 75, 'bcb': 75} |
| 4B | DEC | 5 | 4 | 0.300 | {'hle': 0.21333333333333335, 'bcb': 0.38666666666666666} | [0.233, 0.373] | {'hle': 75, 'bcb': 75} |
| 4B | IND_VOTE | 5 | 4 | 0.260 | {'hle': 0.18666666666666668, 'bcb': 0.3333333333333333} | [0.193, 0.333] | {'hle': 75, 'bcb': 75} |
| 8B | CEN_FLAT | 5 | 4 | 0.320 | {'hle': 0.21333333333333335, 'bcb': 0.4266666666666667} | [0.247, 0.393] | {'hle': 75, 'bcb': 75} |
| 8B | DEC | 5 | 4 | 0.327 | {'hle': 0.2, 'bcb': 0.4533333333333333} | [0.260, 0.400] | {'hle': 75, 'bcb': 75} |
| 8B | IND_VOTE | 5 | 4 | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.253, 0.393] | {'hle': 75, 'bcb': 75} |

### S3_checkpoint_panel_selected

| checkpoint | method | N | B | pool_kind | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 14B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.273 | {'hle': 0.13333333333333333, 'bcb': 0.41333333333333333} | [0.207, 0.340] | {'hle': 75, 'bcb': 75} |
| 14B | CEN_FLAT | 5 | 4 | native | VOTE | 0.273 | {'hle': 0.13333333333333333, 'bcb': 0.41333333333333333} | [0.207, 0.347] | {'hle': 75, 'bcb': 75} |
| 14B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.320 | {'hle': 0.16, 'bcb': 0.48} | [0.253, 0.387] | {'hle': 75, 'bcb': 75} |
| 14B | DEC | 5 | 4 | archive | VOTE | 0.307 | {'hle': 0.16, 'bcb': 0.4533333333333333} | [0.240, 0.380] | {'hle': 75, 'bcb': 75} |
| 14B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.293 | {'hle': 0.14666666666666667, 'bcb': 0.44} | [0.227, 0.360] | {'hle': 75, 'bcb': 75} |
| 14B | DEC | 5 | 4 | latest_slots | VOTE | 0.293 | {'hle': 0.14666666666666667, 'bcb': 0.44} | [0.227, 0.367] | {'hle': 75, 'bcb': 75} |
| 14B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.313 | {'hle': 0.17333333333333334, 'bcb': 0.4533333333333333} | [0.247, 0.387] | {'hle': 75, 'bcb': 75} |
| 14B | IND_VOTE | 5 | 4 | archive | VOTE | 0.307 | {'hle': 0.16, 'bcb': 0.4533333333333333} | [0.240, 0.380] | {'hle': 75, 'bcb': 75} |
| 4B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.267 | {'hle': 0.17333333333333334, 'bcb': 0.36} | [0.200, 0.340] | {'hle': 75, 'bcb': 75} |
| 4B | CEN_FLAT | 5 | 4 | native | VOTE | 0.267 | {'hle': 0.17333333333333334, 'bcb': 0.36} | [0.200, 0.340] | {'hle': 75, 'bcb': 75} |
| 4B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.21333333333333335, 'bcb': 0.38666666666666666} | [0.233, 0.367] | {'hle': 75, 'bcb': 75} |
| 4B | DEC | 5 | 4 | archive | VOTE | 0.300 | {'hle': 0.21333333333333335, 'bcb': 0.38666666666666666} | [0.233, 0.373] | {'hle': 75, 'bcb': 75} |
| 4B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.21333333333333335, 'bcb': 0.38666666666666666} | [0.233, 0.373] | {'hle': 75, 'bcb': 75} |
| 4B | DEC | 5 | 4 | latest_slots | VOTE | 0.300 | {'hle': 0.21333333333333335, 'bcb': 0.38666666666666666} | [0.227, 0.367] | {'hle': 75, 'bcb': 75} |
| 4B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.273 | {'hle': 0.17333333333333334, 'bcb': 0.37333333333333335} | [0.207, 0.340] | {'hle': 75, 'bcb': 75} |
| 4B | IND_VOTE | 5 | 4 | archive | VOTE | 0.260 | {'hle': 0.18666666666666668, 'bcb': 0.3333333333333333} | [0.193, 0.327] | {'hle': 75, 'bcb': 75} |
| 8B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.320 | {'hle': 0.21333333333333335, 'bcb': 0.4266666666666667} | [0.247, 0.393] | {'hle': 75, 'bcb': 75} |
| 8B | CEN_FLAT | 5 | 4 | native | VOTE | 0.320 | {'hle': 0.21333333333333335, 'bcb': 0.4266666666666667} | [0.247, 0.393] | {'hle': 75, 'bcb': 75} |
| 8B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.333 | {'hle': 0.2, 'bcb': 0.4666666666666667} | [0.260, 0.407] | {'hle': 75, 'bcb': 75} |
| 8B | DEC | 5 | 4 | archive | VOTE | 0.327 | {'hle': 0.2, 'bcb': 0.4533333333333333} | [0.253, 0.400] | {'hle': 75, 'bcb': 75} |
| 8B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.327 | {'hle': 0.2, 'bcb': 0.4533333333333333} | [0.253, 0.400] | {'hle': 75, 'bcb': 75} |
| 8B | DEC | 5 | 4 | latest_slots | VOTE | 0.327 | {'hle': 0.2, 'bcb': 0.4533333333333333} | [0.253, 0.400] | {'hle': 75, 'bcb': 75} |
| 8B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.313 | {'hle': 0.21333333333333335, 'bcb': 0.41333333333333333} | [0.240, 0.387] | {'hle': 75, 'bcb': 75} |
| 8B | IND_VOTE | 5 | 4 | archive | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.253, 0.387] | {'hle': 75, 'bcb': 75} |

### S2_membership_panel_native_final

| checkpoint | method | N | B | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 1 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 2 | 4 | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 5 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 1 | 4 | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 2 | 4 | 0.320 | {'hle': 0.18, 'bcb': 0.46} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 5 | 4 | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | DEC_ONE_ROUND | 5 | 4 | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_PRIVATE_REVISION | 5 | 4 | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.210, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 1 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 2 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 5 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |

### S2_membership_panel_selected

| checkpoint | method | N | B | pool_kind | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 1 | 4 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 1 | 4 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 2 | 4 | native | JUDGE_BEST | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 2 | 4 | native | VOTE | 0.270 | {'hle': 0.16, 'bcb': 0.38} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | native | JUDGE_BEST | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 3 | 4 | native | VOTE | 0.240 | {'hle': 0.18, 'bcb': 0.3} | [0.160, 0.330] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 5 | 4 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 5 | 4 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.190, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | native | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | CEN_FLAT | 9 | 4 | native | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 1 | 4 | archive | JUDGE_BEST | 0.290 | {'hle': 0.18, 'bcb': 0.4} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 1 | 4 | archive | VOTE | 0.290 | {'hle': 0.18, 'bcb': 0.4} | [0.200, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 1 | 4 | latest_slots | JUDGE_BEST | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.370] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 1 | 4 | latest_slots | VOTE | 0.280 | {'hle': 0.16, 'bcb': 0.4} | [0.200, 0.360] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 2 | 4 | archive | JUDGE_BEST | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 2 | 4 | archive | VOTE | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 2 | 4 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.230, 0.400] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 2 | 4 | latest_slots | VOTE | 0.320 | {'hle': 0.18, 'bcb': 0.46} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 3 | 4 | latest_slots | VOTE | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.220, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 5 | 4 | archive | JUDGE_BEST | 0.330 | {'hle': 0.2, 'bcb': 0.46} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 5 | 4 | archive | VOTE | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 5 | 4 | latest_slots | JUDGE_BEST | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 5 | 4 | latest_slots | VOTE | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.250, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.18, 'bcb': 0.42} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | archive | VOTE | 0.330 | {'hle': 0.2, 'bcb': 0.46} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | latest_slots | JUDGE_BEST | 0.310 | {'hle': 0.18, 'bcb': 0.44} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC | 9 | 4 | latest_slots | VOTE | 0.320 | {'hle': 0.2, 'bcb': 0.44} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | DEC_ONE_ROUND | 5 | 4 | archive | JUDGE_BEST | 0.330 | {'hle': 0.18, 'bcb': 0.48} | [0.240, 0.420] | {'hle': 50, 'bcb': 50} |
| 32B | DEC_ONE_ROUND | 5 | 4 | archive | VOTE | 0.310 | {'hle': 0.16, 'bcb': 0.46} | [0.230, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | DEC_ONE_ROUND | 5 | 4 | latest_slots | JUDGE_BEST | 0.320 | {'hle': 0.18, 'bcb': 0.46} | [0.230, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | DEC_ONE_ROUND | 5 | 4 | latest_slots | VOTE | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_PRIVATE_REVISION | 5 | 4 | archive | JUDGE_BEST | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_PRIVATE_REVISION | 5 | 4 | archive | VOTE | 0.320 | {'hle': 0.18, 'bcb': 0.46} | [0.240, 0.410] | {'hle': 50, 'bcb': 50} |
| 32B | IND_PRIVATE_REVISION | 5 | 4 | latest_slots | JUDGE_BEST | 0.300 | {'hle': 0.16, 'bcb': 0.44} | [0.220, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_PRIVATE_REVISION | 5 | 4 | latest_slots | VOTE | 0.290 | {'hle': 0.16, 'bcb': 0.42} | [0.210, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 1 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 1 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 2 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 2 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 3 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.220, 0.380] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 5 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 5 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | archive | JUDGE_BEST | 0.300 | {'hle': 0.2, 'bcb': 0.4} | [0.210, 0.390] | {'hle': 50, 'bcb': 50} |
| 32B | IND_VOTE | 9 | 4 | archive | VOTE | 0.300 | {'hle': 0.22, 'bcb': 0.38} | [0.220, 0.390] | {'hle': 50, 'bcb': 50} |

### S2_degree_native_final

| checkpoint | method | degree | B | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|
| 32B | DEGREE | 0 | 4 | 0.174 | {'hle': 0.17391304347826086} | [0.065, 0.283] | {'hle': 46} |
| 32B | DEGREE | 1 | 4 | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 2 | 4 | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 4 | 4 | 0.200 | {'hle': 0.2} | [0.089, 0.333] | {'hle': 45} |
| 32B | DEGREE | 8 | 4 | 0.239 | {'hle': 0.2391304347826087} | [0.130, 0.370] | {'hle': 46} |

### S2_degree_selected

| checkpoint | method | degree | B | pool_kind | selector_id | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|---|---|---|
| 32B | DEGREE | 0 | 4 | native | JUDGE_BEST | 0.174 | {'hle': 0.17391304347826086} | [0.065, 0.283] | {'hle': 46} |
| 32B | DEGREE | 0 | 4 | native | VOTE | 0.174 | {'hle': 0.17391304347826086} | [0.065, 0.283] | {'hle': 46} |
| 32B | DEGREE | 1 | 4 | native | JUDGE_BEST | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 1 | 4 | native | VOTE | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 2 | 4 | native | JUDGE_BEST | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 2 | 4 | native | VOTE | 0.217 | {'hle': 0.21739130434782608} | [0.109, 0.348] | {'hle': 46} |
| 32B | DEGREE | 4 | 4 | native | JUDGE_BEST | 0.200 | {'hle': 0.2} | [0.089, 0.312] | {'hle': 45} |
| 32B | DEGREE | 4 | 4 | native | VOTE | 0.200 | {'hle': 0.2} | [0.089, 0.311] | {'hle': 45} |
| 32B | DEGREE | 8 | 4 | native | JUDGE_BEST | 0.239 | {'hle': 0.2391304347826087} | [0.130, 0.370] | {'hle': 46} |
| 32B | DEGREE | 8 | 4 | native | VOTE | 0.239 | {'hle': 0.2391304347826087} | [0.130, 0.370] | {'hle': 46} |

### E_repeated_episodes_native_final

| checkpoint | method | episode_rep | mean | by domain | 95% CI | n |
|---|---|---|---|---|---|---|
| 32B | CEN_FLAT | 1 | 0.233 | {'hle': 0.06666666666666667, 'bcb': 0.4} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | CEN_FLAT | 2 | 0.100 | {'hle': 0.13333333333333333, 'bcb': 0.06666666666666667} | [0.000, 0.233] | {'hle': 15, 'bcb': 15} |
| 32B | CEN_FLAT | 3 | 0.167 | {'hle': 0.13333333333333333, 'bcb': 0.2} | [0.033, 0.300] | {'hle': 15, 'bcb': 15} |
| 32B | CEN_FLAT | 4 | 0.167 | {'hle': 0.06666666666666667, 'bcb': 0.26666666666666666} | [0.033, 0.300] | {'hle': 15, 'bcb': 15} |
| 32B | CEN_FLAT | 5 | 0.133 | {'hle': 0.06666666666666667, 'bcb': 0.2} | [0.033, 0.267] | {'hle': 15, 'bcb': 15} |
| 32B | DEC | 1 | 0.233 | {'hle': 0.06666666666666667, 'bcb': 0.4} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | DEC | 2 | 0.267 | {'hle': 0.13333333333333333, 'bcb': 0.4} | [0.133, 0.433] | {'hle': 15, 'bcb': 15} |
| 32B | DEC | 3 | 0.233 | {'hle': 0.06666666666666667, 'bcb': 0.4} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | DEC | 4 | 0.233 | {'hle': 0.13333333333333333, 'bcb': 0.3333333333333333} | [0.100, 0.400] | {'hle': 15, 'bcb': 15} |
| 32B | DEC | 5 | 0.267 | {'hle': 0.13333333333333333, 'bcb': 0.4} | [0.133, 0.433] | {'hle': 15, 'bcb': 15} |
| 32B | IND_VOTE | 1 | 0.333 | {'hle': 0.13333333333333333, 'bcb': 0.5333333333333333} | [0.167, 0.500] | {'hle': 15, 'bcb': 15} |
| 32B | IND_VOTE | 2 | 0.200 | {'hle': 0.06666666666666667, 'bcb': 0.3333333333333333} | [0.067, 0.333] | {'hle': 15, 'bcb': 15} |
| 32B | IND_VOTE | 3 | 0.267 | {'hle': 0.06666666666666667, 'bcb': 0.4666666666666667} | [0.133, 0.400] | {'hle': 15, 'bcb': 15} |
| 32B | IND_VOTE | 4 | 0.267 | {'hle': 0.06666666666666667, 'bcb': 0.4666666666666667} | [0.133, 0.400] | {'hle': 15, 'bcb': 15} |
| 32B | IND_VOTE | 5 | 0.167 | {'hle': 0.06666666666666667, 'bcb': 0.26666666666666666} | [0.033, 0.300] | {'hle': 15, 'bcb': 15} |
| 32B | S_FRESH | 1 | 0.233 | {'hle': 0.06666666666666667, 'bcb': 0.4} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | S_FRESH | 2 | 0.200 | {'hle': 0.06666666666666667, 'bcb': 0.3333333333333333} | [0.067, 0.333] | {'hle': 15, 'bcb': 15} |
| 32B | S_FRESH | 3 | 0.233 | {'hle': 0.06666666666666667, 'bcb': 0.4} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | S_FRESH | 4 | 0.233 | {'hle': 0.13333333333333333, 'bcb': 0.3333333333333333} | [0.100, 0.367] | {'hle': 15, 'bcb': 15} |
| 32B | S_FRESH | 5 | 0.200 | {'hle': 0.06666666666666667, 'bcb': 0.3333333333333333} | [0.067, 0.333] | {'hle': 15, 'bcb': 15} |

### E_within_item_variance

- IND_VOTE: 0.044 [0.010, 0.080] n={'hle': 14, 'bcb': 15}
- S_FRESH: 0.033 [0.013, 0.060] n={'hle': 15, 'bcb': 15}
- DEC: 0.030 [0.000, 0.070] n={'hle': 15, 'bcb': 15}
- CEN_FLAT: 0.080 [0.043, 0.120] n={'hle': 15, 'bcb': 15}

### native_final_missingness (missing/invalid finals are scored incorrect; counts shown for transparency)

- A.CEN_FLAT.32B.B4: episodes=400 missing_final=0 invalid_final=26
- A.DEC.32B.B4: episodes=400 missing_final=28 invalid_final=0
- A.IND_VOTE.32B.B4: episodes=400 missing_final=18 invalid_final=0
- A.S_FRESH.32B.B4: episodes=400 missing_final=17 invalid_final=0
- A.S_HISTORY.32B.B4: episodes=400 missing_final=24 invalid_final=0
- B.CEN_FLAT.32B.B1: episodes=100 missing_final=0 invalid_final=14
- B.CEN_FLAT.32B.B2: episodes=100 missing_final=0 invalid_final=6
- B.DEC.32B.B1: episodes=100 missing_final=7 invalid_final=0
- B.DEC.32B.B2: episodes=100 missing_final=8 invalid_final=0
- B.IND_VOTE.32B.B1: episodes=100 missing_final=5 invalid_final=0
- B.IND_VOTE.32B.B2: episodes=100 missing_final=5 invalid_final=0
- B.S_FRESH.32B.B1: episodes=100 missing_final=6 invalid_final=0
- B.S_FRESH.32B.B2: episodes=100 missing_final=4 invalid_final=0
- B.S_HISTORY.32B.B1: episodes=100 missing_final=7 invalid_final=0
- B.S_HISTORY.32B.B2: episodes=100 missing_final=7 invalid_final=0
- D.DEGREE.32B.B4: episodes=229 missing_final=0 invalid_final=58
- E.CEN_FLAT.32B.B4: episodes=150 missing_final=0 invalid_final=10
- E.DEC.32B.B4: episodes=150 missing_final=5 invalid_final=0
- E.IND_VOTE.32B.B4: episodes=150 missing_final=6 invalid_final=0
- E.S_FRESH.32B.B4: episodes=150 missing_final=2 invalid_final=0
- M.CEN_FLAT.14B.B4: episodes=150 missing_final=0 invalid_final=6
- M.CEN_FLAT.4B.B4: episodes=150 missing_final=0 invalid_final=8
- M.CEN_FLAT.8B.B4: episodes=150 missing_final=0 invalid_final=11
- M.DEC.14B.B4: episodes=150 missing_final=9 invalid_final=0
- M.DEC.4B.B4: episodes=150 missing_final=5 invalid_final=0
- M.DEC.8B.B4: episodes=150 missing_final=11 invalid_final=0
- M.IND_VOTE.14B.B4: episodes=150 missing_final=6 invalid_final=0
- M.IND_VOTE.4B.B4: episodes=150 missing_final=2 invalid_final=0
- M.IND_VOTE.8B.B4: episodes=150 missing_final=6 invalid_final=0
- MX.CEN_FLAT.32B.B1: episodes=100 missing_final=0 invalid_final=8
- MX.CEN_FLAT.32B.B4: episodes=200 missing_final=0 invalid_final=13
- MX.CEN_FLAT.8B.B1: episodes=100 missing_final=0 invalid_final=6
- MX.CEN_FLAT.8B.B4: episodes=200 missing_final=0 invalid_final=10
- MX.DEC.32B.B1: episodes=100 missing_final=6 invalid_final=0
- MX.DEC.32B.B4: episodes=200 missing_final=12 invalid_final=0
- MX.DEC.8B.B1: episodes=100 missing_final=9 invalid_final=0
- MX.DEC.8B.B4: episodes=200 missing_final=14 invalid_final=0
- MX.IND_VOTE.32B.B1: episodes=100 missing_final=6 invalid_final=0
- MX.IND_VOTE.32B.B4: episodes=200 missing_final=8 invalid_final=0
- MX.IND_VOTE.8B.B1: episodes=100 missing_final=2 invalid_final=0
- MX.IND_VOTE.8B.B4: episodes=200 missing_final=4 invalid_final=0
- N.CEN_FLAT.32B.B4: episodes=500 missing_final=0 invalid_final=38
- N.DEC.32B.B4: episodes=500 missing_final=40 invalid_final=0
- N.DEC_ONE_ROUND.32B.B4: episodes=100 missing_final=6 invalid_final=0
- N.IND_PRIVATE_REVISION.32B.B4: episodes=100 missing_final=8 invalid_final=0
- N.IND_VOTE.32B.B4: episodes=500 missing_final=20 invalid_final=0

### stop_reasons

- A.CEN_FLAT.32B.B4: {'CYCLE_CAP': 31, 'VOLUNTARY_FINISH': 369}
- A.DEC.32B.B4: {'ROUND_CAP': 400}
- A.IND_VOTE.32B.B4: {'CALL_CAP': 400}
- A.S_FRESH.32B.B4: {'CALL_CAP': 400}
- A.S_HISTORY.32B.B4: {'CALL_CAP': 400}
- B.CEN_FLAT.32B.B1: {'VOLUNTARY_FINISH': 71, 'BUDGET': 29}
- B.CEN_FLAT.32B.B2: {'VOLUNTARY_FINISH': 91, 'CYCLE_CAP': 9}
- B.DEC.32B.B1: {'ROUND_CAP': 20, 'BUDGET': 80}
- B.DEC.32B.B2: {'BUDGET': 12, 'ROUND_CAP': 88}
- B.IND_VOTE.32B.B1: {'BUDGET': 77, 'CALL_CAP': 23}
- B.IND_VOTE.32B.B2: {'BUDGET': 34, 'CALL_CAP': 66}
- B.S_FRESH.32B.B1: {'BUDGET': 71, 'CALL_CAP': 29}
- B.S_FRESH.32B.B2: {'CALL_CAP': 70, 'BUDGET': 30}
- B.S_HISTORY.32B.B1: {'CALL_CAP': 68, 'BUDGET': 32}
- B.S_HISTORY.32B.B2: {'CALL_CAP': 83, 'BUDGET': 17}
- D.DEGREE.32B.B4: {'COMPLETED': 229}
- E.CEN_FLAT.32B.B4: {'VOLUNTARY_FINISH': 139, 'CYCLE_CAP': 11}
- E.DEC.32B.B4: {'ROUND_CAP': 150}
- E.IND_VOTE.32B.B4: {'CALL_CAP': 150}
- E.S_FRESH.32B.B4: {'CALL_CAP': 150}
- M.CEN_FLAT.14B.B4: {'VOLUNTARY_FINISH': 141, 'CYCLE_CAP': 9}
- M.CEN_FLAT.4B.B4: {'VOLUNTARY_FINISH': 142, 'CYCLE_CAP': 8}
- M.CEN_FLAT.8B.B4: {'VOLUNTARY_FINISH': 138, 'CYCLE_CAP': 12}
- M.DEC.14B.B4: {'ROUND_CAP': 150}
- M.DEC.4B.B4: {'ROUND_CAP': 150}
- M.DEC.8B.B4: {'ROUND_CAP': 150}
- M.IND_VOTE.14B.B4: {'CALL_CAP': 150}
- M.IND_VOTE.4B.B4: {'CALL_CAP': 150}
- M.IND_VOTE.8B.B4: {'CALL_CAP': 150}
- MX.CEN_FLAT.32B.B1: {'VOLUNTARY_FINISH': 92, 'CYCLE_CAP': 5, 'BUDGET': 3}
- MX.CEN_FLAT.32B.B4: {'VOLUNTARY_FINISH': 185, 'CYCLE_CAP': 15}
- MX.CEN_FLAT.8B.B1: {'CYCLE_CAP': 6, 'VOLUNTARY_FINISH': 94}
- MX.CEN_FLAT.8B.B4: {'VOLUNTARY_FINISH': 189, 'CYCLE_CAP': 11}
- MX.DEC.32B.B1: {'ROUND_CAP': 84, 'BUDGET': 16}
- MX.DEC.32B.B4: {'CALL_CAP': 100, 'ROUND_CAP': 100}
- MX.DEC.8B.B1: {'ROUND_CAP': 100}
- MX.DEC.8B.B4: {'CALL_CAP': 100, 'ROUND_CAP': 100}
- MX.IND_VOTE.32B.B1: {'BUDGET': 71, 'CALL_CAP': 29}
- MX.IND_VOTE.32B.B4: {'CALL_CAP': 200}
- MX.IND_VOTE.8B.B1: {'CALL_CAP': 100}
- MX.IND_VOTE.8B.B4: {'CALL_CAP': 200}
- N.CEN_FLAT.32B.B4: {'VOLUNTARY_FINISH': 454, 'CYCLE_CAP': 46}
- N.DEC.32B.B4: {'ROUND_CAP': 400, 'CALL_CAP': 100}
- N.DEC_ONE_ROUND.32B.B4: {'ROUND_CAP': 100}
- N.IND_PRIVATE_REVISION.32B.B4: {'ROUND_CAP': 100}
- N.IND_VOTE.32B.B4: {'CALL_CAP': 500}


## 4. Neural readouts (N1 capture, N3 analysis)

- **R3 geometry / functional diversity**: present; summary keys ['anchors', 'association', 'blocks', 'by_config', 'display_transforms', 'n_dev_items', 'n_groups', 'n_rows_loaded', 'note', 'outcomes_joined'].
- **G-dev (predictor selection, dev only)**: present; summary keys ['assembly', 'blocks', 'caveat', 'dev_contrast_descriptive', 'frozen', 'losses_table', 'outer', 'penalties', 'ranks', 'run_root'].
- **G-confirm (frozen predictor on confirmation)**: contrast 0.0003 (95% CI [0.0000, 0.0005]), one-sided p = 0.0133; n = None; note: 
- **C-dev (recalibrators, dev only)**: present; summary keys ['caveat', 'dev_assembly', 'dev_descriptive', 'dev_losses_table', 'frozen', 'refrozen', 'run_root', 'scopes', 'seal', 'stage'].
- **C (confirmation)**: contrast -0.0004 (95% CI [-0.0005, -0.0003]), one-sided p = 1.0000; n = None; note: 

Capture chunks: native 630 jsonl/npz pairs, report 61. Fidelity files: ['fidelity.native.s000of001.json', 'fidelity.native.s000of004.json'].

## 5. Amendment register (freeze/amendments.json, verbatim ids)

| id | deviation | spec clause | when |
|---|---|---|---|
| F1 | Flagship := `Qwen/Qwen3-32B` (dense) instead of `Qwen/Qwen3.8-27B` | §6.4, §7.3, §10.1 |  |
| F2 | Sampling recipe: temp 0.6, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0, thinking on, `max_tokens=8192`, for **all** solver/hub/worker/revision/consumer calls on every checkpoint | §10.1 ("dense Qwen3 ... .6/.95/20/0"), §6.4 |  |
| S1 | Primary code VOTE = Python AST canonicalization (strip positions/comments/docstrings? no: strip positions and comments only; preserve identifiers, literals, order) then exact-source identity; no public probe suite | §3.2, §4.4, §5.3 |  |
| S1b | The literal `CODE_SELECTOR` clause in `prompts/framing_clauses.json` describes a probe suite that does not exist; §5.5 requires the aware clause to be truthful | §5.5 ("accurately describes the frozen ... grouping, fallback and tie procedure") |  |
| N1 | N_main=400 (200 HLE, 200 BCB); dev=60 (30/30) | §3.1, §6.2, §9.4 (allowed N_main ∈ {1100,1600}; dev=300) |  |
| N1b | HLE 200 = 100 Gold + 100 Revision (brief) rather than SRS across text-only Gold+Revision | §3.1 ("simple random sampling by a frozen salted hash within each superdomain") |  |
| R1 | Families R (RLM L32 frontier + budgets, D0/D1/D2 controls, long-context OOLONG/LongBench), G (activation readouts), M (content units, annotators, edits) are not executed; p=1 for each in the six-entry Holm list | §1.4, §6.2, §7.2, §8, §9.2 |  |
| A1 | CEN_RLM D2/L5 absent from A, N, M, C unless the `rlms` adapter passes the §7.10 fixtures by Sun 12:00 | §4.2, §6.2, §7.2 |  |
| A2 | Main-tier DEC roots carry the native truthful clause (`root_framing: native_truthful` in experiment_matrix.yaml; §4.1) and therefore do **not** alias F banks; CEN_FLAT never aliases | §4.1, §4.2, §6.2 ("Alias ... only when prompts ... match exactly") |  |
| A3 | The DEC truthful root/revision clause is not in the handoff; must be authored | §4.1 ("truthfully told that their terminal complete answers enter the stated vote and that they can exchange messages") |  |
| B1 | B0 is resolved analytically from frozen per-role maximum rendered envelopes and full 8,192 decode reservations (§6.5), not from pilot-observed token counts; whether B4 or the 64-call cap binds is a per-episode logged fac | §6.5 ("Resolve its cost from maximum permitted rendered lengths and full decode reservations"), §6.7 |  |
| B2 | Frozen prompt envelopes for hub/worker/subtask-result rendering (not in handoff) so that 32,768 input is enforceable and B0 computable | §4.3 ("Wrappers, schema and role instructions must also fit"), §6.4 |  |
| E1 | Evaluator: HLE exact-answer correctness judged by Qwen3-32B (thinking off, temp 0) with the official centerforaisafety/hle judge prompt; MC by exact letter; no human audit | §3.3 ("frozen isolated judge ... plus a blinded human audit") |  |
| E2 | Guided JSON decoding (vLLM `guided_json`) used for HLE judge and shadow forecast; optional for JUDGE_BEST; never for solver/hub/worker/revision/consumer calls | §3.5 ("No repair model"), §4.4 ("Invalid scores use a frozen low score") |  |
| E3 | BCB evaluation via apptainer `docker://bigcodebench/bigcodebench-evaluate:v0.2.4`, per-candidate fresh process, official `sanitize()` applied identically before both public grouping and protected testing | §3.2, §3.3 |  |
| J1 | JUDGE_BEST judge = Qwen3-32B thinking off, 1,024-token cap, `prompts/judge_best.txt`, guided JSON optional; run on F banks (10), IND/S_FRESH admitted pools (≤64), S_HISTORY archive, DEC latest-N; not on CEN_FLAT | §4.4, §5.3 |  |
| C1 | C on IND_VOTE/DEC/CEN_FLAT only; report reader = Qwen3-32B thinking off, temp 0, 256 cap; scope mask: q_personal and q_team_now required, q_child_contract/q_recover/q_preserve null; operation_id=NONE; panel = first 200 i | §8.7, §9.2 |  |
| C2 | Dev recalibration requires native episodes + forecasts on all 60 dev items for the three architectures (240 episodes, 180 forecasts), beyond the 20-item pilot | §8.7, §9.2 ("development-calibrated") |  |
| D1 | Degree module on 100 items; 9 roots alias F00; no unit panel/annotation/decoder; outcomes = focal revised correctness and the 4-child continuation pool (mean, any-correct) | §4.6, §6.3, §8.10 |  |
| N2 | N panel on 100 items; IND at every N is derived from the F00 bank with the N-specific admission schedule (0 new generation); DEC-00 and CEN_FLAT at N ∈ {1,2,3,5,9} | §4.6, §6.2, §6.3 |  |
| E4 | E panel: 30 items × 6 episodes × {S_FRESH, IND_VOTE, DEC, CEN_FLAT}; episode 0 aliases the main cell; K ∈ {1,2,3,5} from n=6; K=10 not estimable | §5.6, §6.2, §9.3 (30-cluster minimum) |  |
| M1 | Dense panel 4B/8B/14B on 150 items, N=5, B4 (same numerical FLOP allowance as 32B), IND/DEC/CEN_FLAT; 32B row aliases main A; no CEN_RLM | §6.2, §6.4, §6.5 |  |
| M2 | M-cross (8B, 32B; N ∈ {3,9}; B1, B4) on 100 items, stretch | §6.2 |  |
| AB1 | A-budget: B1, B2 on 100 items for the five methods; B8 by the per-episode alias rule (B1), otherwise not run | §6.2, §6.5 (validate every mandatory cell at its lowest budget) |  |
| P1 | Sampled-token logprobs are not stored (throughput); token IDs, rendered bytes, reasoning/final channels, usage, finish_reason are | §10.3 ("where supported") |  |
| P2 | Public export strips the HLE `json` column (contains the answer and rationale), `answer`, and BCB `test`/`canonical_solution`; only the evaluator identity mounts them | §3.2, §10.6 |  |
| P3 | `prior_exposure_manifest.json` must inventory the old harness's datasets; if HLE/BCB never entered prior runs, say so | §3.1 |  |
| P4 | Model/tokenizer revisions: `models.py` carries no revision; freeze the HF snapshot hashes for 4B/8B/14B/32B, vLLM 0.21.0, chat template hash | §6.4, §10.1 |  |
| L1 | Output cap 8,192 total (thinking + final): truncated thinking yields no JSON → invalid candidate (a real outcome, §3.5). The cap is not changed after any main outcome is seen | §3.5, §6.4, §6.6 |  |
| X1 | Package layout `src/agents_scaling/study/` mirrors §11.2 subpackage names; only a subset of the §11.2 CLI verbs is implemented (`cells build`, `run_one`, `seal`, `evaluate`, `analyze`) | §11.2 |  |
| E3' | Code string rule for BCB is one frozen rule used identically for public grouping and protected testing: program := strip_one_enclosing_fence(final_answer) (the §3.5 salvage rule), not the official bigcodebench.sanitize() | §3.2, §3.3, §3.5 |  |
| N9B1 | N=9 at B1 is infeasible under the analytic B0 (9 full-cap roots ≈ 10.2 PFLOP > B1 ≈ 5.66 PFLOP). N=9 appears only at B4; M-cross is 8B/32B at N∈{3,9}×B4, with B1 only for N=3. | §6.2, §6.5 (raise B0 pre-freeze or consistently change an envelope; validate every mandatory cell at its lowest budget) |  |
| N1-SRS | HLE sampling is plain salted-hash SRS within superdomain over the 1,600 eligible text-only Gold+Revision ids (spec-literal); Gold/Revision are reported as strata, not balanced. dev = first 30 HLE + first 30 BCB in rank o | §3.1, §3.3, §6.2 |  |
| E4b | Repeated-episode panel = 30 items × 6 episodes × {S_FRESH, IND_VOTE, DEC, CEN_FLAT}; S_HISTORY excluded because of its 64-call sequential depth; estimation-only. | §5.6, §6.2, §9.3 |  |
| M1b | Dense checkpoint panel committed at 100 items with 150 as a nested hash-prefix extension. | §6.2, §6.4 |  |
| C1b | Family C is tier 2b, text-only shadow forecasts, and may be unexecuted → p=1 in the Holm list. | §8.7, §9.2 |  |
| B2b | Table E is extended: a CEN_FLAT hub prompt's returned-results block is bounded at 16,384 recipient tokens in total (each returned result is clipped to min(4,096, 16,384 // k) for a k-assignment cycle) and a 512-token all | §4.2, §4.3, §6.5 |  |
| C3 | Forecast output-contract line (harness-authored, appended after the trusted manifest) rewritten to state the flat five-key shape explicitly (each value a single number in [0,1] or null); the handoff template templates/fo | §8.7 (forecast schema), §3.5 (strict parse), §10 (technical reruns) | Sun 2026-09-06 11:20 EDT, after the dev pilot forecasts (sea |
| E5 | Development-split outcomes come from the pilot manifest (cells_pilot_32B, seal 9b110309…): its selections were re-sealed in waves (11:10 and 11:31 EDT) once the pilot cells completed, and every pilot item was re-evaluate | §3.6/§10.3 seal order (P0-3), §8.7 development recalibrators, §9.1 analysis freeze | Sun 2026-09-06 11:07–11:55 EDT, before C-dev/G-dev fitting |
| P5 | Primary analysis panel rule made explicit: the smallest common completed rank prefix (§9.1) is primary when it holds at least one 25-item block per domain; otherwise the primary panel is the MATCHED complete-case panel ( | §9.1 (analysis freeze, smallest common completed prefix, intention-to-run), §9.2 | Sun 2026-09-06 13:05 EDT, before any confirmatory statistics |

## 6. Scientific caveats

- Qwen3-32B (the amended flagship, F1) is near floor on HLE-Verified; framing and orchestration effects there are imprecise and BigCodeBench carries most of the signal. Domain-specific estimates are reported beside every pooled effect.
- N_main = 400 (200 HLE + 200 BCB, salted-hash rank prefixes) makes families A and O estimation-oriented (planned power ≈ 0.5–0.65 at Δ = 0.06); intervals matter more than the Holm decisions.
- The HLE judge is the solver checkpoint itself (amendment E1): the multiple-choice audit substitute and the exported 100-candidate human-audit sample bound the judge error; a differential judge error across architectures is not excluded.
- Families R (recursive scaling), G-as-planned (activation edits) and M (content use) were not executed (amendment register); they enter the six-family Holm list with p = 1, which cannot enlarge the other families' error budget.
- Family C uses a development recalibrator fitted on the pilot's dev-split items only (≈18–20 items instead of the planned 60) after the C3 output-contract clarification; the C3 example number (0.25) is a possible mild anchor.
- Family G is estimated on the frozen report-anchor readouts, whose block, rank and penalty were selected on the pilot development split alone (20 items, 58 rows; amendment E5) rather than the planned 60 items, so it is estimation-oriented; cross-checkpoint coordinate comparisons were not attempted.
- Repeated episodes (E) are estimation-only (30 items × 6 episodes) and S_HISTORY is excluded from E (64 sequential calls).
- Episodes without a native final answer (invalid/truncated/no valid candidate) are scored incorrect under the frozen common failure rule and counted in the missingness table; infrastructure-incomplete items are listed by reconcile and excluded from the rank-prefix panels only through the prefix rule.

## 7. Artifacts

- Tables: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/tables/` (selections/banks/episodes parquet, summary.json, stats.json, stats.md)
- Seals: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/seals/<manifest sha>/{POOLS,SELECTIONS}.json`; freeze: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/FROZEN.yaml`, `freeze/amendments.json`, `freeze/prior_exposure_manifest.json`; prompt hashes: `src/agents_scaling/study/prompts/PROMPT_HASHES.json`
- Requests (content-addressed): `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/requests/`; cells: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/cells/<cell_id>/`; evaluator outputs: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/eval/{hle,bcb}/`; HLE audit sample: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/eval/hle_audit_sample.jsonl` (0600) when exported
- Neural: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/neural/` (native/report captures, worklists, tables, summaries); forecasts: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/forecast/`
- Reconciliation: `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/reconcile.json`
