# Data schema

What the harness writes to disk and what the tidy analysis table contains. All outputs
live under `$ASYS_RESULTS_ROOT/<run_id>/`.

## On-disk layout of a run

```
<run_id>/
  cells.json                          # the canonical list of ExperimentCell dicts (array indexes into this)
  chunk_jobs.json                     # SLURM job ids of the submitted chunk arrays
  chunk_*.sbatch, run_cells.sbatch    # rendered SLURM scripts (provenance)
  servers/<size>/<host>_<port>.json   # one file per live server endpoint (multi-endpoint registry)
  servers/serve_<size>.sbatch         # rendered server scripts
  logs/                               # SLURM stdout (serve_*, cell_*)
  cells/<cell_id>/
      results.jsonl                   # one QuestionResult per question (append-only)
      meta.json                       # one CellMeta (config + measured attributes)
```

A cell is **complete** iff `meta.json` exists. The runner skips complete cells and resumes
partial `results.jsonl` by `qid`.

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
  // --- per-agent detail (for per-agent calibration) ---
  "per_agent": [
    {
      "agent_id": "agent0", "round": 0,
      "answer": "B",
      "option_logprobs": {"A":0.01,"B":0.55,"C":0.38,"D":0.06},  // from the forced-answer probe
      "verbalized_conf": 0.70,                                    // parsed "Confidence: X%"
      "cot_text": "...",                                          // reasoning trace / CoT
      "intermediate_results": "...",
      "prompt_tokens": 412, "completion_tokens": 88,
      "reasoning_tokens": 1394, "reasoning_text": "..."           // Axis 4
    }
    // ... one per (agent, round)
  ],
  // --- system-level confidence under MULTIPLE definitions (each gets its own ECE) ---
  "system_conf": {
    "vote_fraction": 0.67,
    "mean_agreeing_logprob": 0.55,
    "mean_agreeing_verbal": 0.72,
    "mean_all_logprob": 0.40,
    "orchestrator_logprob": 0.55,   // centralized only
    "orchestrator_verbal": 0.70     // centralized only
  },
  // --- self-consistency / semantic entropy (single_agent cells with n_samples>1) ---
  "self_consistency": {"samples": ["B","B","C"], "majority": "B",
                       "self_consistency_conf": 0.67, "semantic_entropy_conf": 0.67,
                       "semantic_entropy": 0.92},
  // --- raw efficiency counters (turn-based primary; tokens logged alongside) ---
  "efficiency_raw": {
    "n_turns": 6, "n_messages": 6, "n_rounds": 2, "n_agents": 3,
    "total_prompt_tokens": 2474, "total_completion_tokens": 531,
    "total_reasoning_tokens": 7933, "wall_ms": 31840.2
  },
  "timestamp": 1779861234.5
}
```

## `meta.json` — one per cell (`CellMeta`)

```jsonc
{
  "cell_id": "...",
  "config": { /* full ExperimentCell.to_dict() */ },
  "config_hash": "ab12cd34ef56",
  "model_hf_id": "Qwen/Qwen3-8B",
  "served_model_name": "8B",
  "prompt_token_count": 41,                 // Axis 3 measured attribute
  "prompt_quality": {"heuristic": 62.5, "llm_judge": null, "features": {...}},
  "mean_reasoning_tokens": 7377.0,          // Axis 4 measured attribute
  "git_commit": "6dd79aa...",
  "n_questions": 50,
  "started_at": 1779860000.0, "finished_at": 1779861800.0
}
```

## Tidy table (`analyze.aggregate_run` → parquet)

One row per cell. Nested dicts (`calibration`, `efficiency`, `prompt_quality`) are kept as
JSON strings by `scripts/aggregate_results.py` for a flat parquet.

| column | meaning |
|---|---|
| `cell_id`, `seed`, `n_questions` | identity |
| `model_size`, `param_count` | Axis 1 (param_count = regressor) |
| `context_share_level` | Axis 2 |
| `prompt_complexity_level`, `prompt_token_count`, `prompt_quality` | Axis 3 (knob + measured) |
| `reasoning_level`, `mean_reasoning_tokens` | Axis 4 (knob + measured) |
| `topology`, `benchmark` | setup |
| `accuracy`, `error_rate` | performance |
| `mean_turns`, `mean_messages`, `mean_total_tokens` | raw efficiency inputs |
| `efficiency` | Kim metrics dict: `coordination_efficiency` (Ec), `error_amplification` (Ae), `overhead_pct` (O%), `message_density` (c), `redundancy` (R) — vs the matched SAS baseline |
| `calibration.per_agent` | ECE/MCE/Brier over individual agent answers (option-logprob confidence) |
| `calibration.system.<conf_def>` | ECE/MCE/Brier per system-confidence definition |

**Headline quantity:** `system ECE − mean per-agent ECE` per axis (computed downstream from
the `calibration` columns). Efficiency is keyed to the matched
`(model_size, benchmark, seed, reasoning_level)` single-agent baseline; aggregation fails
loudly if a baseline is missing.
