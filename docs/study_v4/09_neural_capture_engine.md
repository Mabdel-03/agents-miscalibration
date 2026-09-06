# N1 — HF-transformers capture engine: conventions, interfaces, verification (Sat 2026-09-05)

Implements the engine half of `09_neural_readouts_brief.md` (R1 report readouts, R2 native
temporal geometry) as `src/agents_scaling/study/neural/`.  The frozen generation pipeline is
untouched: the engine only *reads* `<run_root>/requests/` (WP2 `RequestStore`) and the N2
report renders, and writes only under `<run_root>/neural/`.

## Hook convention (spec §8.4 "store … hook placement")

* Forward hook on `model.model.layers[b]` (`Qwen3DecoderLayer`), capturing the block's
  **output** hidden state = the residual stream **after** block b = the tensor entering
  block b+1.  This is the spec's "residual-stream outputs after the unique blocks
  `floor(f·(L−1))`"; the block output already contains both sublayer updates
  (`h + attn(rmsnorm(h)) + mlp(rmsnorm(·))`).
* Not the block *input* (that would be block b−1's output / the embeddings at b=0) and not
  the pre-norm ("input_layernorm") activation, which is a per-block RMS-normalised
  projection that is not on the residual stream and is not what later blocks read.
* The final `model.model.norm` is **not** applied to captured residuals (it is applied
  only when logits are computed); b = L−1 therefore yields the pre-final-norm residual —
  note that `output_hidden_states` returns the post-norm tensor as its last entry.
* GPU verification (Qwen3-8B, transformers 5.9): hook capture == `hidden_states[b+1]` with
  max |Δ| = 0.0 at blocks (8, 17, 26).
* Sites: `blocks_for(L) = floor(f·(L−1))`, f ∈ {0.25, 0.5, 0.75}: 32B (L=64) → 15, 31, 47;
  14B (L=40) → 9, 19, 29; 8B (L=36) → 8, 17, 26.
* Positions/masking: **right padding** with an explicit `attention_mask`, `use_cache=False`,
  `position_ids` = 0..n−1 for the real tokens (padding is causal-inert; batched vs single
  capture agrees to bf16 noise).  Batches are packed by descending length so that
  `rows · max_len ≤ max_batch_tokens` (default 16,384); a sequence longer than 40,960 is
  refused.  Residuals are sliced at the requested positions inside the hook and moved to
  CPU as fp16; nothing else is kept.  Logits are computed only at requested positions by
  applying `lm_head` to the post-norm hidden state (never `[B, T, vocab]`).  GPU memory is
  released between batches.  `dtype` bf16, stored fp16 (`tensor_hash` = sha256 of the fp16
  little-endian bytes).

## Anchors (`neural/anchors.py`)

Teacher-forced sequence = `prompt_token_ids + token_ids[:k_max]` (spec §8.4 "measurement-only
exact-prefix teacher-forced replay"); positions index that sequence.

| anchor | position | missingness |
|---|---|---|
| `NATIVE_PREFILL` | `prompt_len − 1` (last prompt token, after it was processed) | — |
| `GENERATED_k` (k ∈ 32, 128, 512) | `prompt_len + k − 1` — the k-th generated token counted over **all** native channels (the `<think>` span is part of the stored ids) | `NOT_REACHED` when `k > completion_len` |
| `FINAL_OBJECT_CLOSE` | the completion token whose bytes contain the closing brace of the parsed final object | `JSON_CLOSE_MISSING` (invalid/truncated/empty), `FINAL_OBJECT_NOT_APPLICABLE` (worker = subtask-result role), `CHANNEL_UNAVAILABLE` (content channel cannot be aligned) |
| `TASK_ONLY_ANCHOR` (reports) | last token of the task span | `ANCHOR_UNRESOLVED` when the report gives no offset |
| `STATE_ANCHOR` (reports) | last token of the compiled report span (defaults to the last prefill token) | — |
| `LAST_PREFILL` (reports) | `len(prompt_token_ids) − 1`, recorded separately (§8.4 "record both locations separately") | — |

Byte → token mapping is exact: Qwen's byte-level BPE lets every regular token map back
to raw bytes through the fixed GPT-2 alphabet, added/special tokens contribute their
literal text, and the concatenation is asserted equal to `tokenizer.decode(ids)`.

`FINAL_OBJECT_CLOSE` resolution, per record: (1) locate the content channel structurally
= bytes after the `</think>` token (whole completion when thinking was off), minus trailing
special tokens (`<|im_end|>`), and assert exact identity with the stored `content` (vLLM
keeps the leading `\n\n`; verified on a real 32B record: `content_channel="exact"`);
(2) validity by the same rule the study uses — `parse_candidate` for root/revise
(candidate roles), a strict one-fence `{"action": final|delegate}` object for the hub
(its declared structured final object; the candidate inside is validated for `final`),
`FINAL_OBJECT_NOT_APPLICABLE` for workers; (3) the object span is derived by mirroring
`strip_one_fence` step by step (asserted equal to its output), the brace's UTF-8 byte
offset is mapped to the token containing it.  A fenced object resolves to the brace
token, never the closing fence.  `channel` records whether a GENERATED_k token sits in
the thinking or content channel; `generated_token_count` is the number of generated
tokens processed at the anchor.

Report anchors: the N2 render (`forecast.manifest.report_render`) supplies exclusive-end
UTF-8 byte offsets into the single user message as `byte_anchors.{task_only_anchor,
state_anchor}` and their token indices as `anchor_tokens.<name>.index`, mirrored flat as
`task_only_anchor_byte` / `state_anchor_byte` / `task_only_anchor_token` /
`state_anchor_token` (`anchors.report_anchor_fields` reads both shapes; a flat key wins).
The content's byte offset inside the chat-template render is found structurally (two probe
renders differing in their first character) — never by sentinel search — and the anchor
token is the one containing the span's last byte; a token index present alongside the byte
form must agree (this is the N1↔N2 agreement check, and it runs on every real render since
N2 writes both).  `capture.check_report_anchors` refuses a render whose task-only anchor is
unresolved or whose state anchor would fall back to the last prefill token (P0-1), and the
render's `checkpoint.model_revision` must equal the engine snapshot unless
`--allow-model-mismatch` (P1-3).

## Selection rule for R2 (`capture.select_native_calls`)

Per (episode, declared role, phase), among calls with a committed request record (no
context failure), order by `HMAC-SHA256(study_seed, JCS(["NEURAL_R2", episode_id,
declared_role, phase, request_id]))` and keep the argmin (`group_rank` 0) — except the
**geometry groups** `GEOMETRY_GROUPS` = {INDEPENDENT_SOLVER/ROOT, DECENTRALIZED_MEMBER/TERMINAL,
CENTRAL_HUB/ROOT, CENTRAL_WORKER/ROOT}, where every eligible call is captured with its HMAC
`group_rank`, so R3 has its s = 5 comparable IND roots / DEC terminal members (brief R3;
P1-1: ≤4 extra sequences per IND/DEC item, inclusion still outcome-blind).  Phases from the
episode's call ledger only:
IND_VOTE `root`→ROOT; DEC `root`→ROOT, `revise` step<max→COORDINATION, step==max→TERMINAL;
CEN_FLAT `hub` step 0→ROOT, the last hub call (if not step 0)→TERMINAL, other hub
calls→COORDINATION, `worker` step 0→ROOT, later workers→COORDINATION.  Declared roles:
INDEPENDENT_SOLVER, DECENTRALIZED_MEMBER, CENTRAL_HUB, CENTRAL_WORKER (CEN_RLM: none yet).
Independent of call length, validity, correctness and activations; every group's eligible
count and context-failure count (→ inclusion probability) is written to
`<run_root>/neural/native/selection.<shard>.json` together with the rule text.
Panel: the flagship N=5/B=4, split `main`, episode_rep 0 generate cells of modules A then
N (dedupe per (method, item)), restricted to the panel of `neural/panel.py`:
`PublicTask.rank < 150` per superdomain on `main` (the same rule N2 `--panel-per-domain`
and N3 `--panel` apply; P1-4).  The resolved item list and the rule are written to
`selection.<shard>.json` (`panel_rule`, `panel`, `panel_resolved`); the cell-order fallback
is used only when `data/public/tasks.jsonl` is absent and is recorded as such.

## Storage (`neural/storage.py`) and the ActivationRow ledger

`<run_root>/neural/<stage>/<shard>.cNNNNN.npz` (`vectors` fp16 `[m, hidden]`, `keys`) +
`.jsonl` (one `ActivationRow` per (StateSnapshot_id, block, anchor_kind)); missing anchors
are rows with `vector_index=-1` and **no vector**.  Atomic writes (temp + `os.replace` +
dir fsync, `.npz` before `.jsonl`), resume by `existing_keys()`.  Ledger fields (spec
§8.13): `StateSnapshot_id` (= request_id | report_id), `consumer_revision` (engine
snapshot revision), `condition` (`native:<METHOD>` | `report:<METHOD>`), `child_slot`
(null), `nonce_hash` (= `input_hash` of the prefilled prompt), `block`, `anchor_kind`,
`structural_span`, `token_offset`, `channel`, `generated_token_count`, `missingness`,
`tensor_hash`, `measurement_cost` (`sequence_tokens`, batch rows/tokens/seconds and the
sequence's share), `operationally_available_at_checkpoint` (true: every anchor is a state
that existed when its token was processed).  Join keys kept alongside: `stage`,
`sequence_id`, `source_id`, `method`, `role` (declared), `phase`, `cell_id`, `episode_id`,
`checkpoint`, `hidden_size`, `sequence_tokens`, `hook_convention`, `extra` (native role,
slot/step/owner, group size, selection key, k_max, anchor detail).  `load_stage(run_root,
stage, blocks=…, anchor_kinds=…)` → `(fp16 matrix, pandas frame)` with ledger fields first.

## N2 → N1 report interface (`<run_root>/forecast/reports/<item>.<method>.json`)

Written by `forecast.manifest.report_render`.  Required: `report_id`, `messages` (exactly
one user message = the shadow-forecast prompt bytes), `prompt_token_ids` (thinking off,
exact render; asserted), `checkpoint.model_revision` (must equal the engine snapshot), and
the anchors — `byte_anchors: {task_only_anchor, state_anchor}` (exclusive-end UTF-8
offsets into the content) + `anchor_tokens: {task_only_anchor: {index, …}, state_anchor:
{index, …}}`, mirrored flat as `task_only_anchor_byte`, `state_anchor_byte`,
`task_only_anchor_token`, `state_anchor_token` (either shape suffices; both must agree).
Copied into the ledger `extra`: `report_sha256` (= `report.text_sha256`),
`rendered_text_sha256`, `request_id` (→ `forecast_request_id`), `seal`, `pool_id`,
`manifest`, `selection_id`, `render_model_revision`; join keys `source_id`, `method`,
`cell_id`, `episode_id` (from `report.item` when not top-level).  Verified end to end by
`tests/study_neural/test_n1n2_interface.py` (real N2 render → N1 resolver, pinned tokenizer).

## CLI and Slurm

```
python -m agents_scaling.study.neural.capture --run-id study_v4 --stage native|report \
    --checkpoint 32B [--blocks 15,31,47] --shard K --num-shards N [--items-file F] \
    [--max-batch-tokens 16384] [--fidelity 20] [--fidelity-tokens 64] [--cells-file cells_1_32B.json ...] \
    [--methods IND_VOTE,DEC,CEN_FLAT] [--modules A,N] [--panel-items 300] [--generated-ks 32,128,512] \
    [--device-map auto] [--dtype bf16] [--group-size 32] [--chunk-rows 256] [--dry-run] [--allow-model-mismatch]
```
GPU python: `/orcd/data/tpoggio/001/mabdel03/envs/neural_env/bin/python` (the module puts
`<repo>/src` on `sys.path`; `PYTHONPATH=src` for the `-m` form).  Outputs per shard:
`selection.<shard>.json`, `stats.<shard>.json`, `fidelity.<shard>.json` (§10.5 numbers over
the first `--fidelity` records: argmax agreement with the stored sampled tokens, mean/min
log-prob, two-pass max/mean |Δlogit|, identity checks).  `slurm/study_neural.sbatch.tmpl`
placeholders: `RUN_ID STAGE SHARD NUM_SHARDS PARTITION GPUS LOG_DIR REPO HF_HOME PYTHON
CHECKPOINT ARGS` (`GPUS`=2 for 32B, 1 for 8B/14B; 8 CPUs, 120G, 12 h; node-local caches
under `/tmp/asys_neural_$SLURM_JOB_ID` removed by an EXIT trap; HF offline).

## Verification (Qwen3-8B, 1×A100-80GB, pi_manoli node3904, job `asys-neural-test`)

* Load 54.7 s cold / 5.9 s warm; `device_map=auto`; sdpa attention; jobs 22075503 and
  22077321 (the second is the clean `ok=True` run; the first hit a chunk-naming bug in
  storage that is fixed and covered by `tests/study_neural/test_n1_storage.py`).
* Hook == `hidden_states[b+1]`: max |Δ| 0.0 at blocks 8/17/26; batched (padded) vs single
  capture: bf16 noise only (max |Δ| 1.5 absolute = 1.2e-2 relative to the largest entry).
* Fidelity, greedy HF completions replayed (3 × 20 tokens): argmax agreement 1.000,
  mean log-prob −0.082, two-pass max |Δlogit| 0.0; identity checks pass.  Three real 32B
  records from `<run_root>/requests` (prompts 283–1,613 tokens, thinking on) pass both
  identity checks and resolve all five anchors (GENERATED_32/128/512 inside the thinking
  channel, FINAL_OBJECT_CLOSE in the content channel, `content_channel="exact"`); replaying
  them through the *8B* engine is only a cross-checkpoint sanity number (argmax agreement
  0.87, mean log-prob −0.38 over 60 sampled tokens) — the 32B parity run still has to be
  done with the 32B engine on 2 GPUs.
* Peak GPU memory 17.0 GiB for 8B at 16,384 batch tokens (15.3 GiB weights): the 32B run
  on 2×80 GB has ample headroom for `--max-batch-tokens 32768`.
* Throughput at `max_batch_tokens=16384`: 46,080 tokens in 5.97 s forward = **7.7k tok/s**
  (batches: 1×16384 tok 1.82 s; 2×6144 2.53 s; 8×2048 1.51 s).  The brief's 3–5k tok/s
  estimate for 32B on 2 GPUs (pipeline-sequential `device_map=auto`) remains the planning
  number; R1+R2 ≈ 50M tokens → ~3–5 h.
* Smoke reports: `<results>/study_v4/neural/_smoke/smoke_<job>.json`.
