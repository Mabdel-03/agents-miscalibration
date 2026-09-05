# Measured throughput (2026-09-05 19:05 EDT, outcome-blind, MATH-500 prompts through the study root template, thinking on, max_tokens 8192, temp 0.6 / top_p 0.95 / top_k 20)

| endpoint | concurrency | items | wall s | aggregate gen tok/s | completion tokens mean / p50 / p90 / max | latency p50 / max s | finish=length | strict-JSON valid |
|---|---|---|---|---|---|---|---|---|
| 32B-long TP2 (node3807:8829) | 48 | 96 | 320 | **577** | 1,921 / 1,011 / 6,064 / 8,192 | 38.8 / 273 | 5/96 | 87.5% |
| 14B-long TP1 (node3904:8813) | 48 | 96 | 240 | **714** | 1,781 / 1,037 / 4,675 / 8,192 | 35.6 / 233 | 4/96 | 93.8% |

Notes: wall includes the low-concurrency tail of the second wave, so steady state is somewhat higher (~650–750 tok/s per 32B server). Single-stream 32B = 38 tok/s; 14B = 47 tok/s. Planning revision: 32B-long server ≈ 650 tok/s (not 900). Six core servers ≈ 3.9k tok/s → the opportunistic ou_bcs_normal/low replicas (up to +8 servers) are required for tier 1 at N_main=400 within the window; keepalive switched to the full spec at 19:10 EDT. Invalid-JSON causes were not inspected here (outcome-blind); the dev pilot reports failure codes.
Raw per-request records: <run_root>/logs/calib_32B-long_c48.json, calib_14B-long_c48.json.
