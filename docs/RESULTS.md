# Results & run status

Living document: validation findings, pilot numbers, and the status of the full sweep.
**All numbers below are validation/sanity signals (tiny n, single seed), not the study's
findings.** They confirm the pipeline produces scientifically sensible, non-degenerate
measurements across every axis.

_Last updated: 2026-05-27._

## Validation milestones (all on real A100s)

- ✅ **Serving + logprob read-off (Risk 1 retired).** `score_options` returns a normalized
  option distribution (sums to 1.0); `chat()` captures token logprobs. The smallest models
  already show realistic over-confidence-when-wrong — exactly the miscalibration the study
  targets.
- ✅ **Topology equivalence.** With n=1, `independent`/`decentralized`/`centralized` reduce
  to `single_agent` (unit-tested).
- ✅ **Reasoning axis (two-call ECE).** On a GPQA item, Qwen3-8B at `off` answered wrong
  (0 thinking tokens); at `b2048` it thought for 1299 tokens and flipped to the **correct**
  answer — with the option-logprob distribution still cleanly recovered from the
  forced-answer probe. Reasoning tokens scale monotonically off→512→2048→unlimited
  (0 → 185 → 407 → 423 on a simple item; ~7,500/question on GPQA debate cells).
- ✅ **End-to-end aggregation.** `analyze.py` + `aggregate_results.py` produce a tidy table;
  per-agent and system ECE, Kim efficiency metrics, and reasoning tokens all populate.

## Original 3-axis pilot (Qwen2.5, TruthfulQA, 100 q/cell — methodology checkpoint)

Run before switching to Qwen3; kept as a methodology checkpoint, **not part of the final
study**.

| cell | acc | per-agent ECE | system ECE | ΔECE | O% |
|---|---|---|---|---|---|
| 1.5B single-agent | 0.31 | 0.245 | 0.690 | +0.445 | 0 |
| 1.5B debate (artifact) | 0.35 | 0.243 | 0.367 | +0.123 | 500 |
| 1.5B debate (+CoT) | 0.37 | 0.244 | 0.450 | +0.206 | 500 |
| 7B single-agent | 0.56 | 0.183 | 0.440 | +0.257 | 0 |
| 7B debate (artifact) | 0.62 | 0.145 | 0.287 | +0.142 | 500 |
| 7B debate (+CoT) | 0.61 | 0.156 | 0.323 | +0.168 | 500 |

Early directional signals (sanity only): capacity helps accuracy *and* calibration (7B
better than 1.5B on both); **system ECE > per-agent ECE everywhere** (ΔECE>0) — i.e. the
aggregated answer's confidence is more miscalibrated than the individuals', exactly the
effect the headline question probes.

## Reasoning pilot (Qwen3, GPQA, 50 q/cell — partial)

Validated the reasoning path; one cell completed before the pilot servers were repurposed
for the full sweep:

| cell | acc | mean reasoning tok/q | per-agent ECE |
|---|---|---|---|
| Qwen3-1.7B decentralized, `b2048` | 0.34 | ~7,500 | 0.295 |

This single cell confirms the full reasoning data path (reasoning tokens captured per agent
and aggregated; two-call ECE intact). Cross-rung comparison (off / 2048 / unlimited) is part
of the full sweep.

## Full sweep — `full_sweep_v1`

- **Config:** `configs/full_sweep.yaml` — 6 sizes × 4 topologies × 3 context × 4 prompts ×
  5 reasoning rungs × 4 benchmarks × 3 seeds ⇒ **~11,520 cells** after collapsing.
- **Servers:** one vLLM per size, spread across `pi_tpoggio` (0.6/1.7/14/32B) and
  `ou_bcs_normal` (4B/8B) — the multi-endpoint registry round-robins cells across them.
- **Cells:** chained array chunks of 480 on `mit_preemptable` (resumable; preemption-safe).
- **Projected scale:** ~tens of thousands of A100-hours / weeks–months (32B + unlimited
  thinking dominate). Deliberately large; survivable via resume + chunking.

**Status (launched):** all 6 servers healthy — `0.6B/1.7B/14B` on `pi_tpoggio`,
`4B/8B/32B` on `ou_bcs_normal` (the multi-endpoint registry round-robins cells across
them). The 29 cell chunks (400 each) are dispatched by the **drive loop**
(`launch_chunked.py --drive`, running under nohup) which submits one chunk at a time under
the `mit_preemptable` QOS submit cap (448). Chunk 0 (`14571830`, tasks 0–399) is queued.
Track with the commands in [OPERATIONS.md](OPERATIONS.md#monitoring); re-run
`aggregate_results.py --run-id full_sweep_v1` any time for partial results. The driver and
runner are resumable, so kills/preemption are safe to recover from by re-running.

**Two background loops sustain the run** (both under nohup): the **chunk driver**
(`launch_chunked.py`) dispatches cells under the QOS submit cap, and the **keepalive**
(`keepalive.py`, 10-min passes) maintains the desired server replica count so cells never
block on a dead endpoint. Together with the resumable runner, the months-long run survives
server churn and cell preemption without manual intervention.

**Fleet scaled 6 → ~21 GPUs (2026-05-27).** At 6 servers (1/size) the run did ~15 cells/hr
(~32 days). Added replicas weighted to the slow models — 32B×6, 14B×5, 8B×4, 4B×3, 1.7B×2,
0.6B×1 = ~21 A100 endpoints, the extras on `ou_bcs_low` (32-GPU QOS cap). Cells round-robin
across replicas via the multi-endpoint registry; cell throttle raised 36 → 240 (~11/endpoint;
the 600s client timeout absorbs the higher concurrency). `scale_up.py` does the one-shot
burst; `keepalive.py` (new `--spec` format) maintains the count as ou_bcs's 1-day servers
churn. Expected throughput improvement roughly proportional to GPUs, weighted toward clearing
the 32B long pole soonest.

## How to read the eventual results

- **Performance vs each axis** — accuracy curves, faceted by topology.
- **Efficiency vs each axis** — `Ec/Ae/O%/c/R` vs the matched single-agent baseline.
- **Headline (calibration)** — two complementary ΔECE measures vs each axis:
  - `system ECE − mean per-agent ECE` — does coordination amplify (>0) or correct (<0)
    miscalibration overall;
  - `final_producer ECE − per-agent ECE` — is the *model that emits the final answer*
    (orchestrator / vote-winner / sole agent) better- or worse-calibrated than a lone
    agent, as you scale capacity, context-sharing, prompt complexity, and reasoning depth.
- Scaling-law fits and the Kim-style regression are in `analysis/fitting.py`.

## Operational notes / incidents

- **Timeout storm at throttle 400 (fixed).** The first full-sweep launch ran the cell
  driver at `--throttle 400`, ~1000+ concurrent requests against 6 servers. Requests queued
  behind unlimited-thinking generations and exceeded the 120s client timeout →
  `APITimeoutError`/`ReadTimeout` on ~197 cells (0 rows written). Fix: client timeout
  120→600s + 8 retries, and **throttle 400→36** (~6 cells/server). The resumable runner
  meant nothing was permanently lost — relaunching at throttle 36 resumed the partial cells
  and rows accumulate cleanly with zero timeouts. See OPERATIONS throttle note.
- **Qwen3-32B KV-cache OOM (fixed).** At `max_model_len=32768`, 32B weights (~61 GiB bf16)
  leave too little KV-cache room on a single A100-80GB (vLLM needs 8.0 GiB KV vs ~7.8 free)
  and the server fails at engine init. Fixed by setting 32B `max_model_len=16384` in
  `models.py` (ample for short QA prompts + the 8192 thinking budget). Other sizes are
  unaffected. Relaunched successfully.
