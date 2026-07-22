# Data schema

What the harness writes to disk and what the tidy analysis table contains. All outputs
live under `$ASYS_RESULTS_ROOT/<run_id>/`.

## On-disk layout of a run

```
<run_id>/
  cells.json                          # immutable ordered ExperimentCell manifest
  cells.sha256                        # checksum pin for cells.json
  benchmark_contracts.v1.json         # normalized Question/source identities
  benchmark_contracts.v1.sha256       # checksum pin for the contract sidecar
  chunk_jobs.json                     # SLURM job ids of the submitted chunk arrays
  chunk_*.sbatch, run_cells.sbatch    # rendered SLURM scripts (provenance)
  servers/<size>/<host>_<port>.json   # one file per live server endpoint (multi-endpoint registry)
  servers/serve_<size>.sbatch         # rendered server scripts
  logs/                               # SLURM stdout (serve_*, cell_*)
  incidents/discarded_server_response_protocol_v4/<cell_id>/
      incident.json                   # immutable evidence index for a reset cell
      reset_complete.json             # archive-before-reset transaction marker
  qid_checkpoint_schema_1_to_2_incident_v1.json
                                      # exact pre-migration checkpoint evidence
  cells/<cell_id>/
      results.jsonl                   # canonical one-valid-row-per-expected-QID records
      meta.json                       # one CellMeta (config + measured attributes)
      failure.json                    # atomic classified failure/backoff state, when present
      .qid_checkpoints/<qid-sha>.json # transient integrity-bound stochastic journal
      .cell.lock                      # non-blocking advisory worker lock
```

A cell is **complete** only when the shared completion validator accepts `meta.json`, its
configuration hash and protocol provenance, and exactly one valid row for every
benchmark-derived expected QID. Each row's `answer_key` must equal that Question's gold
key, and `correct` must equal a fresh benchmark grade of `final_answer`; rows cannot
self-certify their scientific outcome. On resume, lock-protected canonicalization retains
the first truth-valid row per QID, discards malformed/duplicate/unexpected/forged rows,
and atomically rewrites `results.jsonl`; the runner executes only missing QIDs.
`meta.json` is written atomically only after the complete canonical payload passes
semantic validation.

For a QID that is still in flight, every observed stochastic topology coordinate and
self-consistency sample is written immediately to an atomic, integrity-hashed checkpoint.
The checkpoint identity binds the complete cell configuration/hash, benchmark Question,
source-code identity, serving profile, and all generation/context protocol versions and
hashes. An endpoint fallback or worker restart replays a retained output, exact length
censor, or exact response-shape protocol censor and contacts a server only for coordinates
that were never durably observed.
Malformed, tampered, or identity-mismatched checkpoints fail closed instead of drawing a
replacement sample. The terminal assembled topology is checkpointed as well. A journal is
deleted only after the final QID row has been appended and fsynced; if a kill lands between
those operations, canonical resume recognizes the durable result row and removes the stale
journal without inference.

The current checkpoint format is schema 2. Schema-1 journals from the paused rollout are
not loaded or rewritten opportunistically by workers. They must first pass the explicit,
offline schema-1-to-2 migration described in `SWEEP_OPERATIONS.md`; the migration preserves
their exact observed coordinates and records the source bytes and old/new identities before
atomic replacement.

## `results.jsonl` — one row per question (`QuestionResult`)

```jsonc
{
  "cell_id": "8B_decentralized_artifact_only_p1_rb2048_gpqa_s0",
  "qid": "gpqa-12",
  "benchmark": "gpqa",
  // --- the four axes (denormalized for easy grouping) ---
  "model_size": "8B",
  "topology": "decentralized",
  "context_share_level": "artifact_only",
  "prompt_complexity_level": 1,
  "reasoning_level": "b2048",
  // --- outcome ---
  "final_answer": "B",
  "answer_key": "B",
  "correct": true,
  // --- immutable normalized-benchmark provenance ---
  "question_sha256": "...",             // prompt/options/gold/type for this exact QID
  "benchmark_contract_sha256": "...",   // ordered selected-Question contract
  // --- per-agent detail (for per-agent calibration) ---
  "per_agent": [
    {
      "agent_id": "agent0", "round": 0,
      "answer": "B",
      "raw_text": "...",                                         // complete answer text
      "option_logprobs": {"A":0.01,"B":0.55,"C":0.38,"D":0.06},  // from the forced-answer probe
      "verbalized_conf": 0.70,                                    // parsed "Confidence: X%"
      "cot_text": "...",                                          // reasoning trace / CoT
      "intermediate_results": "...",
      "prompt_tokens": 412, "completion_tokens": 1500,
      "reasoning_tokens": 1394, "reasoning_text": "...",          // exact Axis 4 span
      "finish_reason": "stop",
      "reasoning_token_source": "vllm_native_token_ids",
      "reasoning_word_count_legacy": 156,
      "thinking_budget_protocol_version": 4,
      "thinking_budget_protocol_hash": "...",
      "thinking_budget_requested": 2048,
      "thinking_budget_saturated": false,
      "thinking_budget_consumed_tokens": 1394,
      "generation_phase_count": 1,
      "generation_phase_seeds": [0],
      "generation_phase_finish_reasons": ["stop"],
      "generation_phase_prompt_tokens": [412],
      "generation_phase_completion_tokens": [1500],
      "output_capacity_floor_tokens": 6144,
      "generation_phase_requested_max_tokens": [32228],          // 32768 - 412 - 128
      "generation_phase_prompt_token_id_hashes": ["..."],
      "generation_phase_completion_token_id_hashes": ["..."],
      "endpoint_generation": "18290000@gpu001:8000#1779860000.000000",
      "reasoning_start_token_index": 412,
      "reasoning_end_token_index": 1807,                          // first generated end
      "reasoning_start_token_count": 1,
      "reasoning_end_token_count": 2,                            // includes literal answer end IDs
      "injected_transition_tokens": 0,
      "peer_context_tokens": 1200,
      "peer_context_sha256": "...",
      "peer_context_block_token_counts": [600, 600],
      "peer_context_truncation_marker_count": 0
    }
    // ... one per (agent, round)
  ],
  // --- system-level confidence under MULTIPLE definitions (each gets its own ECE) ---
  "system_conf": {
    // pooled/aggregate views
    "vote_fraction": 0.67,
    "mean_agreeing_logprob": 0.55,
    "mean_agreeing_verbal": 0.72,
    "mean_all_logprob": 0.40,
    // FINAL-PRODUCER views — calibration of the model output that DETERMINED the system
    // answer, uniform across topologies (orchestrator for centralized; winning-side agent
    // for vote/debate; sole agent for single-agent):
    "final_producer_logprob": 0.55,   // decisive producer's option-logprob in the final answer
    "final_producer_verbal": 0.70,    // decisive producer's verbalized confidence
    "mean_producer_logprob": 0.52,    // mean over all producers (winning side)
    "orchestrator_logprob": 0.55,     // centralized only (coincides with final_producer_*)
    "orchestrator_verbal": 0.70       // centralized only
  },
  // --- self-consistency / semantic entropy (single_agent cells with n_samples>1) ---
  "self_consistency": {
    "protocol_version": 2,
    "protocol_hash": "...",
    "sample_count": 3,
    "completed_sample_count": 3,
    "length_censored_sample_count": 0,
    "protocol_censored_sample_count": 0,
    "samples": [
      {"sample_index": 0, "seed": 1000, "termination_status": "completed",
       "agent_output": {/* full AgentOutput provenance */},
       "censored_generation": null}
      // ... exactly one outcome for every scheduled sample index/seed
    ],
    "majority": "B",
    "self_consistency_conf": 0.67,
    "semantic_entropy_conf": 0.67,
    "semantic_entropy": 0.92,
    "auxiliary_efficiency_raw": {
      "n_samples": 3, "completed_samples": 3, "length_censored_samples": 0,
      "protocol_censored_samples": 0,
      "total_prompt_tokens": 1236, "total_completion_tokens": 4500,
      "total_reasoning_tokens": 3900, "wall_ms": 16200.0
    }
  },
  // --- raw efficiency counters (turn-based primary; tokens logged alongside) ---
  "efficiency_raw": {
    "n_turns": 6, "n_messages": 6, "n_rounds": 2, "n_agents": 3,
    "total_prompt_tokens": 2474, "total_completion_tokens": 9000,
    "total_reasoning_tokens": 7933, "wall_ms": 31840.2
  },
  "timestamp": 1779861234.5,
  "serving_profile": "8B",
  "effective_context_limit": 32768,
  "tensor_parallel_size": 1,
  "termination_status": "completed",
  "censored_generation": null,
  "observed_topology_coordinates": [],
  "schema_version": 5
}
```

Artifact schema v5 keeps the frozen generation/thinking protocol v4: one native chat
request per sampled trajectory. Schema 5 changes retention and provenance, not request
construction, sampling parameters, or seeds. After exact chat-template tokenization, the
requested generation envelope is

```text
max_tokens = effective_context_limit - exact_prompt_tokens - 128
```

The request is admitted only when that envelope reaches the manifest treatment's minimum
output-capacity floor:

| reasoning level | output-capacity floor |
| --- | ---: |
| `off` | 4,096 |
| `b512` | 4,608 |
| `b2048` | 6,144 |
| `b8192` | 12,288 |
| `unlimited` | 12,288 |

Thus `output_capacity_floor_tokens` identifies the treatment gate, while
`generation_phase_requested_max_tokens[0]` records the usually larger, prompt-specific
`max_tokens` sent to vLLM. The validator requires
`prompt_tokens + requested_max_tokens + 128 == effective_context_limit` and requires a
non-empty `endpoint_generation` on every completed agent output. The endpoint generation
has the stable form `<Slurm job or unmanaged>@<host>:<port>#<server start time>` and
identifies the exact serving process, not just a reusable host and port.

Before any topology participant issues a stochastic chat, every participant completes
and caches its independent question-only forced-option calibration probe. Consequently,
a probe failure for a later agent cannot cause an earlier observed answer to be sampled
again. For single-agent `n_samples>1` cells, every auxiliary schedule coordinate is then
retained as a full stopped `AgentOutput`, a full `length_censored` record, or a full
`protocol_censored` record. Auxiliary censors are never replaced and do not relabel the
already completed primary answer. If any auxiliary censors, all
self-consistency/semantic-entropy aggregate fields are `null`; their generation costs
remain explicit under `auxiliary_efficiency_raw` and are never folded into primary
topology efficiency.

Schema 4 remains an explicitly supported read contract. A schema-4 row may be
`completed` or `length_censored`, uses self-consistency protocol v1, and has no
`observed_topology_coordinates` or generation-censor provenance. A schema-4 metadata
record likewise has no schema-5-only protocol-censor counts or schema-count map. Readers
do not reinterpret or silently upgrade it. Schema-5 metadata can truthfully summarize a
canonical mixed set of retained schema-4 and newly produced schema-5 rows through
`artifact_schema_counts`; only schema 5 may certify a `protocol_censored` row.

For current rows, each MCQ probe must contain exactly the benchmark option letters and
normalize to one. The retained agent answer is re-extracted from `raw_text`, using the
same option-probe argmax fallback as inference. The validator then replays the registered
topology aggregation (single producer, independent/last-round vote, or centralized
orchestrator) and requires exact agreement from `final_answer` and every `system_conf`
key/value. Calibration fields therefore cannot be deleted or forged independently of
the retained producer outputs.

### Exact-envelope length censoring

A finite context cannot guarantee that a sampled trajectory emits its terminal token.
When the sole request returns `finish_reason="length"` after consuming exactly its full
requested envelope, schema v5 writes one valid QID row instead of resampling or accepting
a parseable prefix:

```jsonc
{
  "qid": "gpqa-12",
  "final_answer": null,
  "correct": false,
  "per_agent": [],
  "system_conf": {},
  "self_consistency": {},
  "termination_status": "length_censored",
  "censored_generation": {
    "reason": "length",
    "finish_reason": "length",
    "sampling_attempt_count": 1,
    "qid": "gpqa-12",
    "agent_id": "agent2",
    "round": 0,
    "generation_role": "topology",
    "sample_index": null,
    "seed": 2,
    "serving_profile": "8B",
    "effective_context_limit": 32768,
    "context_reserve_tokens": 128,
    "prompt_tokens": 412,
    "requested_output_tokens": 32228,
    "output_capacity_floor_tokens": 6144,
    "completion_tokens": 32228,
    "prompt_token_ids": [151644 /* ... all integer IDs retained ... */],
    "completion_token_ids": [151667 /* ... all integer IDs retained ... */],
    "prompt_token_id_sha256": "...",
    "completion_token_id_sha256": "...",
    "decoded_completion": "...",
    "server_content": null,
    "server_reasoning": "...",
    "prompt_think_start_positions": [],
    "prompt_think_end_positions": [],
    "completion_think_start_positions": [0],
    "completion_think_end_positions": [],
    "endpoint_generation": "18290000@gpu001:8000#1779860000.000000",
    "thinking_budget_protocol_version": 4,
    "thinking_budget_protocol_hash": "...",
    "created_at": 1779861234.5
  },
  "observed_topology_coordinates": [
    {
      "coordinate_key": "topology:agent2:0",
      "request": {/* exact QID/agent/round/seed/peer-context request identity */},
      "outcome": {
        "termination_status": "length_censored",
        "agent_output": null,
        "censored_generation": {/* exact same censor payload as above */}
      },
      "observed_at": 1779861234.5,
      "producer_wall_ms": 31840.2
    }
  ],
  "schema_version": 5
}
```

The elisions above are documentation-only: an artifact retains every prompt and
completion token ID, their hashes, decoded completion, exact nullable server parser fields, delimiter
positions, request coordinates, seed, profile/context, endpoint generation, and creation
time. The completion validator accepts length censorship only for one attempt whose exact
token count exhausts the full registered envelope. Schema 5 additionally requires the
non-empty checkpoint snapshot containing at least one censor. Concurrent siblings can
independently censor in the same terminal wave; the top-level terminal must match exactly
one of them. Any completed siblings and all sibling costs remain visible instead of being
discarded. `n_messages` is derived from the peer outputs actually consumed by those
coordinates, not from the number of turns.

### Exact response-shape protocol censoring

After prompt IDs, completion IDs, usage counts, and the full request envelope have been
verified, an observed response can still violate the frozen v4 answer-shape contract. For
example, it may end on Qwen's alternate EOS token, omit or introduce a reasoning delimiter,
exceed a finite reasoning budget, or return an unexpected finish reason. Schema 5 retains
that one sampled response as data rather than conditioning the dataset on a different
retry:

```jsonc
{
  "termination_status": "protocol_censored",
  "final_answer": null,
  "correct": false,
  "per_agent": [],
  "system_conf": {},
  "self_consistency": {},
  "censored_generation": {
    "reason": "protocol",
    "sampling_attempt_count": 1,
    "finish_reason": "stop",
    "protocol_violation_codes": ["alternate_eos_under_v4"],
    "actual_terminal_token_id": 151643,
    "prompt_token_ids": [151644 /* ... exact IDs ... */],
    "completion_token_ids": [151667 /* ... */, 151643],
    "prompt_token_id_sha256": "...",
    "completion_token_id_sha256": "...",
    "seed": 2,
    "serving_profile": "8B",
    "effective_context_limit": 32768,
    "endpoint_generation": "18290000@gpu001:8000#1779860000.000000",
    "thinking_budget_protocol_version": 4,
    "thinking_budget_protocol_hash": "...",
    "generation_censor_protocol_version": 1,
    "generation_censor_protocol_hash": "..."
  },
  "observed_topology_coordinates": [/* checkpoint-preserved sibling(s) and censor */],
  "schema_version": 5
}
```

The validator recomputes the sorted violation-code set and terminal token from the exact
retained IDs, requires the top-level censor to equal exactly one checkpoint censor, proves
that the coordinates form a legal complete terminal wave, and derives token/turn/message
costs from the complete coordinate snapshot. Server parser fields preserve `null` distinctly
from an empty string. The response is retained
once under its original seed; no endpoint fallback, replacement draw, or parseable-prefix
acceptance is allowed.

The registered codes are `alternate_eos_under_v4`, `premature_reasoning_eos`,
`reasoning_delimiter_contract`, `reasoning_budget_exceeded`,
`unexpected_finish_reason`, `unexpected_reasoning_delimiter`, and
`unknown_terminal_under_v4`. The abbreviated example still requires every common envelope,
decoded/server-text, delimiter-position, coordinate, timestamp, and endpoint field shown in
the length-censor contract.

`protocol_censored` is only for a response whose envelope is trustworthy enough to retain
as scientific data. Missing token IDs, prompt/usage disagreement, parser-content mismatch,
or another untrusted envelope remains `ServerResponseProtocolError`: it is a blocked
configuration incident, not an automatic retry and not a fabricated censor. Historical
cells in which such a response was discarded and then retried require an immutable incident
archive plus a clean whole-cell reset before schema-5 rerun.

## `meta.json` — one per cell (`CellMeta`)

```jsonc
{
  "cell_id": "...",
  "config": { /* full ExperimentCell.to_dict() */ },
  "config_hash": "ab12cd34ef56",
  "model_hf_id": "Qwen/Qwen3-8B",
  "served_model_name": "8B",
  "benchmark_contract_sha256": "...",   // exact ordered Question contract for this cell
  "serving_profile": "8B",
  "effective_context_limit": 32768,
  "tensor_parallel_size": 1,
  "serving_profile_counts": {"8B": 50},
  "serving_profile_inferred_counts": {},
  "prompt_token_count": 41,                 // Axis 3 measured attribute
  "prompt_quality": {"heuristic": 62.5, "llm_judge": null, "features": {...}},
  "mean_reasoning_tokens": null,            // null unless every row has exact token IDs
  "mean_reasoning_tokens_exact": 7377.0,    // exact subset; null if none
  "exact_reasoning_question_count": 48,
  "nonexact_reasoning_question_count": 2,
  "mean_reasoning_word_count_legacy": 812.4,
  "peer_context_protocol_version": 2,
  "peer_context_protocol_hash": "...",
  "peer_cot_char_limit": 4000,
  "peer_rendered_block_token_limit": 4000,
  "thinking_budget_protocol_version": 4,
  "thinking_budget_protocol_hash": "...",
  "generation_censor_protocol_version": 1,
  "generation_censor_protocol_hash": "...",
  "git_commit": "6dd79aa...",
  "n_questions": 50,
  "completed_question_count": 48,
  "length_censored_question_count": 1,
  "protocol_censored_question_count": 1,
  "artifact_schema_counts": {"5": 50},
  "started_at": 1779860000.0, "finished_at": 1779861800.0,
  "schema_version": 5
}
```

## `failure.json` — retry and permanent-failure state

```jsonc
{
  "schema_version": 2,
  "cell_id": "...",
  "config_hash": "ab12cd34ef56",
  "classification": "connection", // connection | context_capacity | configuration | runtime
  "disposition": "retryable",      // retryable | permanent
  "attempts": 3,
  "first_failed_at": 1779860000.0,
  "last_failed_at": 1779860600.0,
  "last_error": {"type": "APIConnectionError", "message": "..."},
  "next_eligible_at": 1779861200.0,
  "serving_profile": "32B-long",
  "code_version": "6dd79aa...+source.0123456789abcdef",
  "server_pool_generation": "spg-v1:...",
  "dormant": false
}
```

`server_pool_generation` is a canonical hash of every live serving process and its
layout for one profile. It changes when membership changes or a server restarts at the
same host and port, and is used only to revive dormant connection/runtime failures after
a genuinely new fleet appears. It is deliberately distinct from the human-readable
per-response `endpoint_generation`, which identifies the exact process that produced an
agent output or censor. Schema-v1 ledgers remain readable; their historical
`endpoint_generation` field is treated once as the old pool-generation value and is
rewritten as schema v2 on the next failure update.
The schema also enforces state-machine invariants: context/configuration failures are
permanent and unscheduled, while connection/runtime failures always remain retryable,
using either a future eligibility time or a dormant state (never both).
An untrusted `ServerResponseProtocolError` is configuration-blocked because retrying after
a sampled-but-unrecordable response would condition admission on parser success. A verified
response-shape anomaly does not create `failure.json`; it is instead retained exactly once
as the schema-5 `protocol_censored` outcome described above.

## Tidy table (`analyze.aggregate_run` → parquet)

One row per cell. Nested dicts (`calibration`, `efficiency`, `prompt_quality`) are kept as
JSON strings by `scripts/aggregate_results.py` for a flat parquet.

| column | meaning |
|---|---|
| `cell_id`, `seed`, `n_questions` | identity |
| `model_size`, `param_count` | Axis 1 (param_count = regressor) |
| `context_share_level` | Axis 2 |
| `prompt_complexity_level`, `prompt_token_count`, `prompt_quality` | Axis 3 (knob + measured) |
| `reasoning_level`, `mean_reasoning_tokens`, `mean_reasoning_tokens_completed_only`, exactness/source fields | Axis 4 (knob + measured); the primary topology mean requires uncensored top-level QIDs, the explicitly conditional mean contains terminating QIDs only, and legacy whitespace counts are never pooled with exact token-ID counts |
| `topology`, `benchmark` | setup |
| `n_completed_questions`, `n_length_censored_questions`, `n_protocol_censored_questions` | schema-5 outcome accounting |
| `length_censor_rate`, `protocol_censor_rate`, `any_censor_rate` | separate and combined top-level QID censor rates; these compatibility names never include auxiliary draws |
| `n_auxiliary_samples`, `n_auxiliary_completed_samples`, `n_auxiliary_length_censors`, `n_auxiliary_protocol_censors` | self-consistency outcome accounting under its own scheduled-draw denominator |
| `auxiliary_length_censor_rate`, `auxiliary_protocol_censor_rate`, `auxiliary_any_censor_rate` | auxiliary self-consistency censor rates (`null` when no auxiliary draws were scheduled) |
| `all_generation_length_censor_rate`, `all_generation_protocol_censor_rate`, `top_level_uncensored`, `whole_cell_uncensored` | health summaries distinguishing the primary-QID contract from the union of top-level and auxiliary generation outcomes |
| `accuracy`, `error_rate` | performance |
| `mean_turns`, `mean_messages`, `mean_total_tokens` | raw efficiency inputs |
| `efficiency` | Kim metrics dict: `coordination_efficiency` (Ec), `error_amplification` (Ae), `overhead_pct` (O%), `message_density` (c), `redundancy` (R) — vs the matched SAS baseline |
| `calibration.per_agent` | ECE/MCE/Brier over individual agent answers (option-logprob confidence) |
| `calibration.system.<conf_def>` | ECE/MCE/Brier per system-confidence definition |

### System-confidence definitions (each yields its own ECE)

"Confidence of an aggregated answer" is genuinely ambiguous, so it is logged under several
definitions and ECE is computed for each (`calibration.system.<def>`):

- **Pooled views** — `vote_fraction`, `mean_agreeing_logprob`, `mean_agreeing_verbal`,
  `mean_all_logprob`: aggregate the agent population.
- **Final-producer views** — `final_producer_logprob`, `final_producer_verbal`,
  `mean_producer_logprob`: the calibration of the **model output that actually produced the
  system answer**, defined uniformly across topologies:
  - `single_agent` → the sole agent;
  - `independent` / `decentralized` → the winning-side agent(s) (decisive = the agreeing
    agent most confident in the final answer; `mean_producer_*` averages the winning side);
  - `centralized` → the orchestrator (also exposed as `orchestrator_*` for continuity).

This answers "is the model that emits the final answer well-calibrated?" — distinct from
the pooled vote confidence.

**Headline quantities** (computed downstream from the `calibration` columns):
- `system ECE − mean per-agent ECE` per axis — does coordination amplify/correct
  miscalibration overall;
- `final_producer ECE − per-agent ECE` — does the *decision-making* model become better/
  worse calibrated than a lone agent as you scale each axis.

Efficiency is keyed to the matched `(model_size, benchmark, seed, reasoning_level)`
single-agent baseline. A length- or protocol-censored QID remains in the accuracy
denominator as incorrect, but contributes no fabricated system answer, confidence, or
complete-topology trajectory. If either a target cell or its matched baseline contains
either censor type, `efficiency` is withheld (`null`) because the
coordination-efficiency comparison is not like-for-like. Primary calibration/ECE and
calibration deltas are also undefined for the entire affected cell: confidence is
structurally absent on a censored trajectory, so an ECE computed only on terminating rows
would be an outcome-selected estimand. Analysis may expose that completed-only calibration
as explicitly supplementary, alongside both censor rates, but never as the headline
metric. Missing baselines are likewise reported without an efficiency value.

Reasoning-token analysis follows the non-conditioning rule for its primary topology
estimand. `mean_reasoning_tokens` summarizes primary topology rows only and is `null` if
any top-level QID is length- or protocol-censored. Auxiliary self-consistency token costs
and censor outcomes are explicitly separate and therefore do not null an otherwise exact
primary topology mean.
The observed mean over terminating top-level QIDs is retained only as
`mean_reasoning_tokens_completed_only`, with
`reasoning_tokens_completed_only_exact` recording token-ID provenance. The legacy/mixed
counterpart follows the same naming rule. `reasoning_metric_primary_defined` and
`reasoning_tokens_exact` are true only when every top-level QID terminates and every
top-level reasoning count has exact native-token provenance. `whole_cell_uncensored`
remains the stricter all-generation health flag and is false for an auxiliary-only censor.
This distinction prevents downstream scaling fits from silently selecting on primary
termination without conflating the separately reported auxiliary estimand.
