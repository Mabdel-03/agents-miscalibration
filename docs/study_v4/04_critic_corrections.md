The Write tool is not available in this read-only agent, so the plan is returned directly below (the caller reads this text).

# Completeness critique and final plan for `agent_design_v4` on ORCD

Authority: brief (cluster facts) > spec v4.0 > handoff > the three designs. Everything marked **[verified]** was checked by reading the file or running a read-only command in this session.

## 0. Facts verified now that override parts of all three designs

| Fact [verified] | Consequence |
|---|---|
| `pi_tpoggio` QOS `GrpTRES=cpu=144,gres/gpu:a100=6,mem=750G` (`sacctmgr`) | Max 3 × `32B-long` TP2 there; the brief's and the architecture design's `14B-long:1:pi_tpoggio` will pend forever (`QOSGrpGRES`). Ops design is right. |
| Right now node3904 (pi_manoli) has 4 A100 held by `imgriff` on `ou_bcs_low` (tier 10, `PreemptMode=REQUEUE`); node3807 (pi_tpoggio) has 6 A100 held by `mit_preemptable`/`ou_bcs_low` jobs | Neither pi node is idle (ops design said node3904 idle). Our tier-100 jobs preempt those, so 8 + 6 GPUs are still obtainable, but expect a 5–15 min preemption delay and do not assume "idle". |
| ~60 idle A100s across `ou_bcs_low` A100 nodes (3810/3906/3911 fully idle; 3907/3910/3905/3909/4001/3811 partially); `ou_bcs_low` per-user cap 32 A100, `ou_bcs_normal` cap 8 | The ops "mid/max" fleet is attainable tonight if launched early; preemption by tier ≥ 25 jobs remains the risk. |
| Association `mit_general` MaxSubmit=500; `mit_normal` QOS `cpu=96`; `mit_preemptable` `cpu=1024`, 2-day, REQUEUE | Cells: `mit_preemptable` (template has `--no-requeue`, driver re-queues by missing `meta.json`) unless patch A frees pi_manoli RAM. |
| Login node has network: GitHub raw 200, HF API 200, Docker Hub reachable (401 auth challenge) | The cais/hle judge prompt, BCB parquet and the apptainer image can all be fetched on the login node at T0. |
| `asys_env`: no `jsonschema`, no `rlms`, no `bigcodebench`; `openai 2.38.0`, `transformers 5.9.0`, `datasets 4.8.5`; `serve_env` vllm 0.21.0; `apptainer 1.4.5`; `/orcd/data/tpoggio/001/mabdel03/containers` absent | Hand-written strict validators; `bigcodebench.sanitize` is only available inside the container (see correction C7). |
| HLE shards: eligible text-only Gold+Revision = **1,600** (Gold 575, Revision 1,025); MC 378 / exact 1,222; 5 MC rows have non-letter answers; ids unique; top-level `question`/`answer` present; `json` is a string-encoded dict with `image`, `image_preview`, `answer_type`, `rationale` | Brief's 371 MC is slightly off (378 incl. 5 non-letter). MC scorer must fall back to exact normalization for those 5. |
| HOME 852k/1000k files; data FS 15 T free | Patch C (cache exports in `serve_qwen.sbatch.tmpl`) before the first server launch. |
| Repo APIs: `registry.list_live_servers(run_root, model_size, *, probe_timeout=3.0, probe_attempts=3, require_current_provenance=False)`; `lookup_server(run_root, model_size, shard=0, *, live_only=False, …)`; `wait_for_server(run_root, model_size, shard=0, timeout_s=300.0, poll_s=5.0, …)` — the positional "model_size" is the registry key and accepts `"32B-long"` (`entry_matches_profile` handles profile names) | Architecture's `EndpointPool` calls are correct as written. |
| `serving/context.py::rendered_chat_token_ids(tokenizer, messages, *, enable_thinking)` and `rendered_chat_token_count` exist (transformers-5 safe) | Reuse; do not re-implement `render_chat` in `inference/tokens.py`. |
| `launch_chunked._chunk_complete` → `ExperimentCell.from_dict(cells[idx]).cell_id`; `run_cell_array.sbatch.tmpl` hardcodes `--mem=4G --time=12:00:00` and `agents_scaling.experiment.run_one`; `_render` substitutes `MEM`/`TIME` that the template never uses | Fork both (all three designs agree). |
| `launch_server.render_sbatch` legacy path works with no release id (`_legacy_runtime_pins`, `PRODUCTION_RUNTIME=0`); `cpus=max(8,tp*8)`, `mem=f"{tp*120}G"` hardcoded at lines 518-519; keepalive imports `submit as submit_server` and counts in-flight per (profile, partition) only (`_serve_jobs_in_flight`, keepalive.py:206) | Legacy launch "verbatim" is valid; ops patches A/B are real but optional. |
| Handoff `framing_clauses.json` `CODE_SELECTOR` describes a "frozen public task-derived probe suite"; no DEC truthful clause, no HLE judge prompt, no `dec_root.txt` in the handoff | Two prompt artifacts must be authored and one re-worded **before any BCB F-bank request** (§5.5 truthfulness). |

## 1. Prioritized corrections (P0 = will break the run or violate a spec MUST)

**P0-1. Alias table: the architecture design is wrong for the N panel; adopt the fidelity table.**
`experiment_matrix.yaml` `architecture_main.DEC.root_framing: native_truthful` but `static_membership.root_framing_cell: '00'` and `one_round_private_revision_control.initialization: identical_DEC_root_bank`; spec §6.3 "neutral 00 root-generation wording at every N … Reuse only exact neutral resource requests", §4.6, §6.2 row N. Therefore:
- Main-tier DEC (module A) roots = `render_root(task, NATIVE, DEC_ROOT_TRUTHFUL)` → **never alias F**; CEN_FLAT never aliases.
- N-panel DEC-00 (all N), IND_PRIVATE_REVISION, DEC_ONE_ROUND roots = `render_root(task, F00)` with stateless keys → alias **F00 draws 0..N−1**; the truthful communication disclosure lives in the round-1 `role_contract` of `focal_revision.txt`, not in the root.
- IND_VOTE-11 draws 0..9 ← F11 (both designs agree); S_FRESH-00 ← F00; S_HISTORY draw 0 ← F00[0]; DEGREE roots ← F00[0..8]; N-panel IND-00 ← F00 then the S_FRESH-00 sequence (content-addressed, automatic).
Fix in `src/agents_scaling/study/policies/dec.py`: `framing = Framing.NATIVE if cell.module == "A" else Framing.F00`. Add fidelity fixture T17 (`test_alias_table`) with a third assertion: N-panel DEC N=5 root ids == F00[0..4] ids.

**P0-2. Author and freeze two prompt artifacts before any F-bank/DEC request; reword one.**
- `prompts/dec_root_truthful.txt` (amendment A3 text from the fidelity design) — §4.1 requires DEC members be told their terminal answer enters the vote and that they can exchange messages.
- `prompts/hle_judge.txt` — fetch `hle_eval/run_judge_results.py` from `centerforaisafety/hle` on the login node at T0 (network verified) and copy the `JUDGE_PROMPT` literal; record sha256.
- `framing_clauses.json::CODE_SELECTOR` → re-word to describe AST-canonical/exact-source grouping (amendment S1b). The literal handoff clause is untruthful under amendment S1 and §5.5 says the aware clause must "accurately describe the frozen … grouping, fallback and tie procedure". Store the amended file as `prompts/framing_clauses.v4_orcd.json` with `PROMPT_HASHES.json`.
Neither the architecture nor the ops design does any of this; the architecture's `render_root(..., NATIVE, dec_clause(N))` references a clause that does not exist.

**P0-3. Seal order is wrong in the architecture's tier "1-eval".** It lumps `JUDGE_HLE`, `EVAL_BCB`, `JUDGE_BEST` into one post-seal wave. JUDGE_BEST is a *selector* (§4.4/§5.3); its scores and selection must be sealed **before** any correctness join (§3.6 "seal pool definitions and selected IDs before correctness joins", §10.3). Required order: `seal --kind candidates,pools` → JUDGE_BEST cells (public inputs only) → `seal --kind selections` (VOTE + JUDGE_BEST) → `JUDGE_HLE`/`EVAL_BCB` cells → `aggregate` refuses to join unless `seals/<sha>/SELECTIONS.json` exists and its `sealed_at` precedes every evaluation record's `started_at`. Physically JUDGE_BEST and HLE-judge cells may run concurrently on the fleet only if `aggregate` enforces that timestamp rule; simplest is two dispatch waves (`--tier 1-select`, `--tier 1-eval`).

**P0-4. Wall-clock break: sequential items × sequential S_HISTORY.** Architecture: 10 items/cell, items sequential, `max_inflight=N=1` for S_HISTORY → 10 × 64 calls × ~2.5–3 min ≈ 27–32 h per cell, over the 12 h (or 24 h) walltime, and the cell never writes `meta.json`. CEN_FLAT (≤17 sequential steps ≈ 50 min/item × 10 = 8.5 h) is also marginal. Fix in `cells.py`/`runner.py`: add `parallel_items: int` to `CellSpec` (outer `ThreadPoolExecutor(parallel_items)` over items; inner per-item group pool), and size cells: S_HISTORY 4 items/`parallel_items=4`; CEN_FLAT 5 items/2; DEC 10/2; S_FRESH/IND 10/1 with per-item concurrency 8; F 20/2 with 10-wide draws. Use `--cell-time 1-00:00:00` on `mit_preemptable` (2-day max). Also dispatch S_HISTORY cells first within each block (ops ordering) so the long pole does not become the tail.

**P0-5. OpenAI SDK auto-retry must be disabled.** `openai` 2.38 retries timeouts/5xx twice by default; that silently violates §10.4's "fixed maximum of two retries per canonical request" accounting and duplicates generation submissions on a loaded server. In `inference/client.py`: `openai.OpenAI(base_url=…, api_key="EMPTY", max_retries=0, timeout=3600)`; the harness owns the ≤ 2 exogenous retries and records `attempts` and `endpoint` per attempt. Architecture's 1,800 s per-request timeout is too short under queueing (8,192 tokens at ~19 tok/s per sequence ≈ 7 min plus queue); use 3,600 s.

**P0-6. Per-endpoint concurrency must respect KV capacity, not `max_num_seqs=256`.** 32B TP2 at 0.90 utilization leaves ≈ 36 GiB/GPU KV ≈ 294k tokens ≈ 49 sequences at 6k tokens (ops table, KV = 2·64·8·128·2 B = 256 KiB/token). Architecture's "≤ 800 concurrent sequences fleet-wide over 8 servers" (100/endpoint) will queue and thrash. Throttle ≈ 6 cells × 8 in-flight per live 32B endpoint (≈ 48); dense: 14B ≈ 40, 8B ≈ 60, 4B ≈ 70. Set `--throttle = 6 × n_live_endpoints` per lane (recomputed manually at gates; live `scontrol update ArrayTaskThrottle` is a nice-to-have, not required).

**P0-7. Per-lane cells files and job names.** One global array with one throttle cannot serve the 32B lane and three dense lanes with different capacities. `cells.py --tier … --lane 32B|14B|8B|4B|eval` writes `cells_<tier>_<lane>.json`; `study_launch_chunked.py` names arrays `asys-study-<run_id>-<lane>` and counts headroom by that exact name (`squeue -r -h -o %j`). `EVAL_BCB` cells: `--cpus 2 --mem 8G`.

**P0-8. Fleet spec correction.** Replace the architecture's command with the ops spec minus the optional entries:
`"32B-long:3:pi_manoli:7-00:00:00,14B-long:1:pi_manoli:7-00:00:00,8B-long:1:pi_manoli:7-00:00:00,4B-long:1:pi_manoli:7-00:00:00,32B-long:3:pi_tpoggio:7-00:00:00,32B-long:2:ou_bcs_normal:1-00:00:00,14B-long:2:ou_bcs_normal:1-00:00:00,8B-long:1:ou_bcs_normal:1-00:00:00,4B-long:1:ou_bcs_normal:1-00:00:00,32B-long:6:ou_bcs_low:1-00:00:00"`
(pi_manoli 6+1+1+1 = 9 > 8 GPUs: choose either the 14B or the 8B+4B on pi_manoli; without the pair template, put `14B-long:1` on pi_manoli and `8B/4B` on `ou_bcs_normal`/`ou_bcs_low`.) Launch pi_* servers first (node3904/3807 are also `ou_bcs_low` members). Drop H100 entries (keepalive counts per partition only; patch B is optional).

**P1-1. B0 must be computed analytically, not "set in the pilot".** All three now agree on the formula; freeze `resources/profile.py` output (B0 = max over {IND5, DEC5, CEN_FLAT5, S_FRESH, S_HISTORY} of minimal full-cap path with root prompt envelope `L_root_max`) before tier-1 dispatch, and use pilot telemetry only for scheduling (§6.5 "resolve its cost from maximum permitted rendered lengths and full decode reservations"). Fix the architecture's hand-picked `root_prompt_max_tokens=6144`: derive it as `4096 (task) + measured wrapper tokens of the longest framing cell + schema line` from the real tokenizer (fidelity Table E), and freeze Table E caps for hub/worker/subtask-result rendering (subtask `result` ≤ 4,096 recipient tokens via the packet clipper; hub prior plan ≤ 8,192; forwarded results ≤ 8,192 total) so that a CEN_FLAT prompt can never exceed 32,768 and B0 is computable.

**P1-2. N=9 at B1 is infeasible under this B0 (9 roots ≈ 10.2 PFLOP > 5.66).** Spec §6.5 says raise B0 pre-freeze or consistently change an envelope; do not plan M-cross/A-budget N=9 B1 cells unless B0 is raised (which would move B4 for everything). Decision to record in `freeze/amendments.json` (fidelity AB1/M2): N=9 appears only at B4; M-cross is 8B/32B at N∈{3,9} × B4 only (and B1 only for N=3).

**P1-3. Truthful N=1 and hub prompts.** `render_dec_revision` at N=1 must not mention peers (empty packet list and a role_contract variant without "other members"); `central_hub.txt` at N=1 gets `N_WORKERS=0`. Spec §4.6 prohibits a false "other team members" claim. Add fixture.

**P1-4. Firewall grep test must not use `\btest\b`** (fidelity T14 greps `"test"` which matches `pytest` and `failure_checks` text). Use import-graph assertion: no module outside `study/evaluation/` imports `study.data.protected` and no non-evaluation module opens a path containing `/protected/`; plus a runtime check that `load_public_tasks` rows lack `answer|json|test|canonical_solution`.

**P1-5. Code string rule for BCB must be one frozen rule used for both grouping and testing.** Fidelity says official `bigcodebench.sanitize()`; architecture says strip-one-fence. `bigcodebench` is not importable in `asys_env` (only in the container) so the grouping side cannot call it without a second container invocation. Freeze: `program = strip_one_enclosing_fence(final_answer)` (same salvage rule as §3.5), applied identically in `selection/normalize.py::code_key` and `evaluation/bcb.py`; record as amendment E3'.

**P1-6. Item-file resume: drop the `.partial.jsonl` journaling (ops/fidelity T9).** The content-addressed request store already gives exact replay (prompts are pure functions of committed records; ledger debits are recomputed from stored `prompt_tokens`/`completion_tokens`). One store, one write path. Keep fidelity T9 but implement it as "kill mid-item, rerun, assert zero new HTTP calls".

**P1-7. Request store layout: keep per-request files** (`<run_root>/requests/<id[:2]>/<id>.json`, `O_EXCL`), not the ops design's per-cell JSONL + index. Inodes on data are not a constraint (verified), and per-cell JSONL breaks first-writer-wins aliasing across cells. Ops' storage estimate (≈ 50–80 GB) still holds.

**P1-8. Balanced Gold/Revision (brief) vs SRS (spec §3.1).** Pick plain salted-hash SRS within each superdomain over the 1,600 eligible HLE ids (spec-literal, one rank order per domain, fewer code paths); report Gold/Revision strata as covariates. If the user insists on 100/100, record amendment N1b. Either way dev = first 30 HLE + 30 BCB in rank order, main = next 200 + 200, reserve = rest (10 %+ retained, §3.3).

**P1-9. Amendment register and prior-exposure manifest are missing from the architecture.** Add `freeze/amendments.json` (the fidelity table verbatim, items F1…X1 plus E3' and the N=9 decision) and `freeze/prior_exposure_manifest.json` (HLE-Verified/BigCodeBench first downloaded 2026-09-05; never used by the old harness, whose runs under `agents_scaling_results/` are GPQA/MMLU-Pro/MATH/TruthfulQA — verified from the HF cache) written by `config.py::freeze()` together with `FROZEN.yaml`.

**P1-10. Judge audit substitute (fidelity E1).** Because Qwen3-32B judges its own exact-answer outputs (§3.3 requires a human audit), run the judge also on all MC candidates (378 MC items → letter scoring vs judge), report FP/FN, compute the 2-pp differential-error bound, and export a stratified 100-candidate sample for later human audit. Cheap (MC judge calls are short) and it is the only thing that makes the judged HLE numbers defensible.

**P1-11. Patch C before the first server launch** (`slurm/serve_qwen.sbatch.tmpl` after `export HF_HOME=…`): `export XDG_CACHE_HOME=/orcd/data/tpoggio/001/mabdel03/.cache VLLM_CACHE_ROOT=/orcd/data/tpoggio/001/mabdel03/.cache/vllm TORCHINDUCTOR_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/torchinductor TRITON_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/triton`. The template uses `--export=NONE` and unsets `VLLM_*`, so environment-side redirects (architecture design) do nothing; the exports must be inside the template. Optional: `--max-num-seqs 64`.

**P2-1. Guided JSON for judges is a pilot decision, not a design assumption.** Try `extra_body={"guided_json": schema}` for `HLE_JUDGE`/`JUDGE_BEST` on one live endpoint during the smoke test; if the server rejects or is slow (xgrammar), fall back to strict regex/JSON parse and record. Never for solver roles (§3.5).

**P2-2. MC key extraction must handle the 5 non-letter MC answers** (fall back to `hle_exact_key`), and `hle_mc_key` must only be used when `answer_format == "multipleChoice"`.

**P2-3. Pair-serving template (ops E), keepalive H100 patch (ops B), live `ArrayTaskThrottle` updates, `study.dispatch` rewrite:** all optional; cut first (see §5).

## 2. Cross-design agreement check

| Topic | Architecture | Fidelity | Ops | Decision |
|---|---|---|---|---|
| Package root | `src/agents_scaling/study/` flat | `study/` with §11.2 subpackages (`orchestration/`, `artifacts/`, `statistics/`) | `study/dispatch.py`, `study/eval_bcb.py`, `study/judge_hle.py` | Architecture's flat layout **plus** `freeze/` (amendments, prior exposure) and `evaluation/hle_judge.py`, `evaluation/bcb.py`. Rename nothing else; §11.2 is "a guide". |
| Request store | per-request file, `O_EXCL`, 256-way fan-out | "artifacts/store.py wraps io.py atomic writes" | per-cell JSONL + index | Per-request file (P1-7). |
| Cell dir | `cells/<cell_id>/{items/<sid>.json, incomplete/, events.jsonl, meta.json}` | `items/<sid>.json`, `meta.json` last, `items/<id>.partial.json` | `items/<sid>.partial.jsonl`, `meta.json` | Architecture's, no partial journal (P1-6). |
| Seals | `seals/<manifest_sha>/{candidates,pools,selections}.jsonl + SEAL.json` | `candidates.json` → `selection.json` per item/method, evaluate refuses without it | `seals/<block>.json` per 50-item block | Architecture's directory, but **two seal kinds** (`POOLS.json`, `SELECTIONS.json`) and per-block sealing so evaluation waves can start Sun 08:00 (P0-3). |
| cell_id | `F.BANK.32B.N1.B0.F00.e0.s003` | `{cell_id, module, method, checkpoint, N, B_mult, framing, episode_rep, item_ids, endpoint_profile, shard}` | + `lane` | Architecture format + `lane`, `parallel_items`, `max_inflight` fields. |
| Semantic-seed purposes | `root/revise/hub/worker/consumer/hle_judge` with namespaces `stateless_bank/S_HISTORY/DEC/CEN_FLAT/DEGREE/judge` | `ROOT/DEC_REV/HUB/WORKER/CONSUMER`, namespace `MAIN` | — | Architecture's table (already frozen as a table); the strings are arbitrary but must be frozen once in `types.py`. |
| Alias rules | N-panel DEC never aliases F (wrong) | fidelity table | "A aliases F" (vague) | Fidelity table (P0-1). |
| Driver | fork `study_launch_chunked.py` + `study_cell_array.sbatch.tmpl` + `study_loops.py` | new `run_study_cell_array.sbatch.tmpl` | `study/dispatch.py` + per-lane arrays + live throttle | Architecture's forks + per-lane cells files/job names (P0-7). |
| Cell partition | `mit_preemptable`, 12 h, 4G | — | `pi_manoli` with patch A, 1-day | `mit_preemptable --cell-time 1-00:00:00 --cell-mem 4G` (eval 8G/2 CPU). Patch A only if pi_manoli CPU slots are wanted later. |
| Evaluator | one `apptainer exec` per cell running an in-container loop | per-candidate fresh process, `sanitize()` | per-shard `apptainer exec`, 96-task arrays | Architecture's one-exec-per-cell loop with fresh subprocess per candidate, rlimits, `--network none --containall --no-home`, strip-one-fence rule (P1-5). |
| B0 | analytic from `L_root_max=6144` | analytic from frozen Table E | from pilot ("outcome-blind profile") | Analytic from Table E (P1-1). |

## 3. Compute model vs the 64-call cap

Using the verified config dims, the oracle constants in the architecture design are correct (32B: `C_lin` 63.97 GFLOP/token, `C_attn` 2,097,152 FLOP/token/key; 4B 8.04 G / 589,824; 8B 15.14 G / 589,824; 14B 27.98 G / 819,200). With `L_root_max ≈ 6,144`, `B0 = 5 × F(6144, 8192) ≈ 5.66 PFLOP`, `B4 ≈ 22.6 PFLOP`.

- Stateless root (S_FRESH/IND) realized at ~1.5k prompt / 3.5k output ≈ 0.35 PFLOP; the reservation before each call is `F(1.5k, 8192) ≈ 0.72 PFLOP`, so admitted calls ≈ `(22.6 − 0.72)/0.35 ≈ 62` → on BCB-like outputs the 64-call cap and the budget bind at about the same point; at HLE-like 4.5–5k outputs (≈ 0.45–0.5 PFLOP) the budget binds at ≈ 45–50 calls. So the brief's "B4 typically binds at the 64-call cap" is true for code and false for HLE; the ops table (64 calls for S_FRESH/IND/S_HISTORY) is therefore an **upper bound**, which is the right direction for scheduling.
- DEC N=5: roots ≈ 1.7 PFLOP; each round reserves `5 × F(~6.5k, 8192) ≈ 5.9 PFLOP` and realizes ≈ 3.7 → ≈ 5 rounds (≈ 30 calls), `stop_reason=BUDGET`. Ops' 45 is again an upper bound.
- Dense panel at the same numerical B4: 4B realized call ≈ 0.05 PFLOP → the 64-call/8-round/8-cycle caps bind everywhere (ops' 150 calls/item is exact, not an upper bound).
- Consequence for the ops timeline: tier 1 on the 8-server core is ≤ 19 h (likely 14–16 h on HLE because the budget binds earlier); the ≈ 30 h window (Sun 00:30 → Mon 06:00) holds tier 1 + tier 2 only with ≥ 4 opportunistic `ou_bcs_low` 32B replicas, which are available tonight. The per-episode `stop_reason` and `slack` must be recorded (§6.5) and the pilot must print the realized admitted-call schedule per method before freeze.
- "B8 aliases B4" is a per-episode fact (fidelity B1): alias only when the B4 episode stopped on `CALL_CAP/ROUND_CAP/CYCLE_CAP` with no remaining-budget input in any prompt (true for all five policies: no prompt shows remaining budget). Implement as: a B8 cell first checks the B4 item file's `stop_reason` and copies it when the rule holds.

## 4. Spec MUSTs the architecture design forgot or under-specified

1. Token-envelope preflight with the real tokenizer on all 460 items × 4 checkpoints before freeze (§4.3 "Preflight checks actual token IDs for every checkpoint") — add `data/export.py --preflight` step; the packet-compiler bound test (fidelity T13) must run once with the real 32B tokenizer, not the whitespace stub.
2. Reserved final hub call: present in the architecture; add the "typed error counts as a used hub call and consumes a cycle" fixture (fidelity T7) and the explicit "no-more-work" instruction text frozen in `prompts/cen_final_instruction.txt`.
3. DEC barrier + previous-round-only packets: present; add the prompt-bytes assertion test.
4. Selector cost charged to B: VOTE = 0; JUDGE_BEST companion outside B with `companion_cost` recorded (§4.4) — present; but the hypothetical "same-B judged deployment" is not run (record in amendments).
5. Protected-field firewall: present but the grep rule needs P1-4.
6. Seal order: P0-3.
7. Context-failure record (§4.3) when a rendered prompt exceeds 32,768: present as `ContextFailure`; make sure it counts as a **used opportunity** (`Y=0`) and, for DEC, that the member's slot gets the sentinel next round.
8. §3.4 counters (roster, unique actors, resets, peak live contexts, calls by role): present in `EpisodeResult.counters`.
9. `study_seed` provenance: generated once by `secrets.token_hex(32)` and written into `FROZEN.yaml`; §3.6 collision check = engine-seed uniqueness within an episode (not across the manifest; cross-cell collisions are harmless).
10. The C tier needs the §8.7 report compiler (8,192-token evidence cap, HMAC-ordered ≤ 4 nonselected packets, trusted manifest with scope mask), a 256-token thinking-off temperature-0 forecast, and dev recalibrators fitted on all 60 dev items — none of which the architecture designs. Treat C as tier 2b; cut first (§5).
11. Missing from all three: the `hle_judge` cell must deduplicate by `sha256(final_answer)` per item *within the sealed pool only* and store judge outputs under `<run_root>/eval/` (evaluator identity), never in `cells/`.
12. Intention-to-run: `INFRA_INCOMPLETE` items (after 2 retries) are excluded from paired contrasts only and reported (§10.4); the `incomplete/` directory blocks `meta.json` so the driver re-queues; a final `study.reconcile` (10 lines) lists remaining `incomplete/` files before the Mon 06:00 seal.

## 5. Cut order if implementation slips 3 hours (highest → lowest priority to cut)

1. Pair-serving template, keepalive H100/gpu-type patch, live `ArrayTaskThrottle` updates, patch A (pi_manoli cells).
2. C tier (report compiler, forecasts, recalibrators) → family C `p=1`.
3. E repeated episodes; A-budget B1/B2; M-cross; CEN_RLM adapter (already stretch).
4. DEGREE module and IND_PRIVATE_REVISION/DEC_ONE_ROUND controls (tier 2 → tier 3).
5. Dense M panel from 150 → 100 items; if `dec.py`/`cen_flat.py` slip, run dense IND_VOTE only first (stateless, needs only WP1–WP3).
6. N panel: keep IND (free via aliases) and DEC/CEN at N ∈ {1,3,9} only.
7. `aggregate.py` bootstrap/Holm machinery → post-deadline; ship per-item parquet tables only.
8. JUDGE_BEST on A-pools (keep on F banks; §5.5 requires both selectors on F).
Never cut: F (4 × 10), S_FRESH, S_HISTORY, IND_VOTE, DEC, CEN_FLAT at N=5/B4, VOTE, seals, HLE judge (+ MC audit), BCB evaluator, FROZEN.yaml + amendments.

## 6. First 90 minutes (T0 → T0+1:30): what must exist so the fleet and pilot are not blocked

Ops (lead, 20 min): patch C in `slurm/serve_qwen.sbatch.tmpl`; `mkdir` run root `study_v4` + `/orcd/data/tpoggio/001/mabdel03/.cache/{vllm,torchinductor,triton,apptainer/tmp}` + `containers/`; launch 3 × `32B-long` + `14B-long` on `pi_manoli`, then 3 × `32B-long` on `pi_tpoggio` (expect preemption of the current occupants), then `2 × 32B-long` on `ou_bcs_normal` and `6 × 32B-long` on `ou_bcs_low` by hand (`launch_server … --replica 8..15`); start `apptainer pull` in the background with `APPTAINER_CACHEDIR/TMPDIR` on data; `curl` the cais/hle judge prompt; `datasets.load_dataset("bigcode/bigcodebench", split="v0.1.4", revision="b74c0d0…")` → parquet in `<run_root>/data/raw/`; copy `hle_shards/` out of the login-node `/tmp` scratchpad into `<run_root>/data/raw/` (compute nodes cannot see `/tmp/claude-…`).

WP0 (lead, 45 min, blocking): `study/types.py`, `study/identity.py`, `study/config.py` + `configs/study_v4.yaml` (with `study_seed_hex`, `split_salt_hex`, checkpoint snapshot hashes `1cfa9a72…/b968826d…/40c06982…/9216db57…`, caps, decoding), vendored `metrics_reference.py`, `prompts/templates/*` byte-copied + `dec_root_truthful.txt` + `hle_judge.txt` + amended `framing_clauses.json` + `PROMPT_HASHES.json`, `tests/study/conftest.py` (tmp run_root, registry-writing helper, real-tokenizer fixture that skips when HF cache is absent), and the `RequestRecord`/`EpisodeResult` JSON examples as fixtures.

WP2 first deliverable (60 min): `tests/study/fake_vllm.py` + `register_fake()` so every other package can test end-to-end from hour 1. One **real** smoke request against the first live `32B-long` endpoint (T0+35–60 min) to pin `reasoning` vs `reasoning_content`, `prompt_token_ids`, `choice.token_ids`, `usage`, `finish_reason`, and whether `guided_json` works; encode the observed shape into the fake server.

WP1 first deliverable (90 min): `data/export.py` producing `<run_root>/data/public/tasks.jsonl`, `protected/*.jsonl` (0600/0700), `splits.json`, `exclusions.jsonl`, plus the envelope preflight report — the pilot cannot start without it.

## 7. Final module / work-package list (5 agents + lead, ~6 h)

```
src/agents_scaling/study/
  types.py identity.py config.py metrics_reference.py            WP0 (lead)
  freeze/{amendments.json, prior_exposure_manifest.json}          WP0 (lead; text from fidelity §1 + E3' + N=9 decision)
  data/{hle.py, bcb.py, splits.py, export.py, public.py}          WP1
  prompts/{templates/*.txt, framing_clauses.v4_orcd.json, render.py, PROMPT_HASHES.json}   WP1
  inference/{tokens.py (wraps serving/context.rendered_chat_token_ids), client.py (max_retries=0, timeout 3600), store.py}   WP2
  parse/{candidate.py, coordinator.py}  packets.py                WP3
  selection/{normalize.py, vote.py, judge_best.py, seal.py}       WP3 (+ seal.py by WP5)
  resources/{oracle.py, broker.py, profile.py}                    WP4
  policies/{base.py, s_fresh.py, s_history.py, ind_vote.py, dec.py, cen_flat.py, degree.py}   WP4 (S_FRESH/IND first, DEC/CEN second, DEGREE/one-round controls last)
  evaluation/{hle_judge.py, bcb.py, bcb_container_driver.py, audit.py}   WP5 (only importer of data.protected)
  cells.py runner.py run_one.py aggregate.py reconcile.py          WP5
configs/study_v4.yaml                                              WP0
slurm/{study_cell_array.sbatch.tmpl, study_launch_chunked.py, study_loops.py, study_loop_driver.sbatch.tmpl}   WP5
slurm/serve_qwen.sbatch.tmpl (patch C only)                         lead, T0
tests/study/{conftest.py, fake_vllm.py, test_identity.py, test_data.py, test_render.py, test_client_store.py,
             test_parse_packets.py, test_selection.py, test_oracle_broker.py, test_policies.py,
             test_cells_runner.py, test_evaluation.py, test_e2e.py}
```
Acceptance before tier-1 dispatch (§10.6): fidelity fixtures T1–T20 with T13/T15 on the real tokenizer, T17 extended per P0-1, T18 per P0-3, plus the 20-item dev pilot on the live fleet producing `FROZEN.yaml` (B0, Table E, prompt hashes, amendments, alias table, realized admitted-call schedule per method).

Exact pilot/tier-1 commands: as in the architecture design §5, with these substitutions: fleet spec from P0-8; `--cell-time 1-00:00:00`; per-lane `--cells-file cells_tier1_32B.json` / `cells_M_14B.json` …; `--throttle 6×n_live`; `--qos-limit 460 --submit-cap 380 --chunk-size 100`; waves `--tier 1` → `seal --kind pools` → `--tier 1-select` → `seal --kind selections` → `--tier 1-eval` → `aggregate`.

## Critical Files for Implementation
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/types.py` + `identity.py` + `config.py` (new; the frozen contract, seed-key table, alias table per P0-1, amendments register)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/policies/dec.py` + `resources/broker.py` (new; N-panel vs main framing, full-round reservation, stop reasons)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/runner.py` + `inference/client.py` + `inference/store.py` (new; `parallel_items`, `max_retries=0`, per-request O_EXCL store, seal-gated evaluation)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/serving/registry.py` and `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/serving/context.py` (reused verbatim: `list_live_servers`/`lookup_server`/`wait_for_server`, `rendered_chat_token_ids`)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/launch_chunked.py`, `/orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/run_cell_array.sbatch.tmpl`, `/orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/serve_qwen.sbatch.tmpl` (forked/patched: per-lane arrays, `{MEM}/{TIME}/{CPUS}`, cache exports)