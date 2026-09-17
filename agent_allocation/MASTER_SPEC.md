# OpenRouter Agent Allocation Study: Parallel Implementation Specification

**Project:** PoggioLab - Agents Scaling Project  
**Study ID:** `openrouter_agent_allocation_v1`  
**Specification version:** 1.0  
**Prepared:** September 17, 2026  
**Default budget:** $150 cash target; at most $135 in inference credits, including pilot and retries  
**Lower-cost preset:** $100 cash target; at most $90 in inference credits  
**Primary model:** `google/gemini-2.5-flash-lite`  
**Primary requested reasoning allowance:** A = 1,024 tokens per generation  
**Higher-allowance sensitivity:** A = 4,096  
**Status:** Implementation handoff, not a completed experiment. The included configuration files, reference contracts, and offline cost calculator are supplied artifacts. The experiment runner, dataset integrations, graders, and live API pilot must be implemented and tested by the assigned agents. No paid inference was launched while preparing this package.

## 0. Objective and essential distinctions

Build and run a reproducible, budget-aware API experiment comparing independent sampling, iterative private revision, longer uninterrupted reasoning, and communicating teams. The primary scientific endpoint is **the correctness of the answer the policy actually delivers**, not agreement or oracle availability.

For the same questions, model, and per-call requested reasoning allowance, collect:

1. Independent samples through 25; report every k from 1 through 25, prominently including the requested 1–16 sweep.
2. One private-revision trajectory through 25 total generations; report every prefix.
3. Teams of 2, 3, 4, and 5 with 1–4 synchronous peer-update rounds after independent initialization.
4. Matched teams that revise privately without seeing peers.
5. Longer single calls with larger thinking allowances.
6. A four-agent communicate-once-then-private continuation and a single pooled synthesizer on a smaller, fixed panel.

The first five are the core experiment. The sixth connects the sweep to the question of whether communication changes the return on subsequent computation, and supplies a cheap information-pooling control.

Do not conflate these:

- **Oracle pass@k / coverage:** at least one candidate passes the private evaluator. This is not an available selector.
- **Delivered accuracy:** a frozen, implementable readout picks or produces an answer without using gold labels or private tests.
- **Requested reasoning allowance A:** an API setting, not a guarantee that exactly A thinking tokens were used.
- **Resource equivalence:** the same nominal generation count is not the same input processing, realized token usage, API cost, or hardware FLOPs.
- **Experimental reuse:** shared initial responses save collection cost but are still charged to each hypothetical deployed policy that needs them.
- **API statelessness:** private revision continues from a visible written solution. It does not preserve an uninterrupted hidden reasoning state. The long-single-call arm addresses that different intervention.

**This is a token- and API-cost-accounted study. Do not label it an exact FLOP-matched experiment.** Model-family transfer and broad statistical power are not established by the small optional replication panel.

### Source provenance

The implementation choices below are proposed design decisions. External facts about APIs and benchmark interfaces are supported by the numbered references in `SOURCES.md`; they are not claimed as observations from a live pilot.

Two user-supplied reports motivate specific regression tests: the repaired **ARD36** report, pages 2–3 and 6, documents root/control key collisions, tri-state grading mistakes, and inconsistent pass@k populations; **A Scaffolding Atlas**, pages 2–3 and 6–10, documents response aliasing, unrelated stopping caps, truncation, and syntactic code-voting failures. Treat these reports as prior evidence and test-design input, not as an executable repository specification and not as results of this new experiment.

## 1. Dispatch instructions: one lead plus eight parallel coding agents

Read `agents/00_LEAD.md` first. The lead owns repository integration, the frozen contracts/configuration, authorization of live runs, and the single paid dispatcher. Start agents 01–08 concurrently after the contract skeleton is committed. Each agent works in a separate Git worktree and edits only its assigned files. Agents may propose interface changes, but only the lead merges changes to shared contracts.

| Agent | Primary ownership | Can start against mocks? |
|---|---|---|
| 01 | OpenRouter adapter, endpoint discovery, transport and usage normalization | Yes |
| 02 | DAG scheduler, persistence, global reservations, crash recovery | Yes |
| 03 | Protocol DAGs, prompts, shared-root references, policy accounting | Yes |
| 04 | MATH-500, MMLU-Pro, BBH subset loaders and text graders | Yes |
| 05 | MBPP+ adapter, public checks, private EvalPlus grading, sandbox | Yes |
| 06 | Extraction, deployable selection, coverage and transition metrics | Yes |
| 07 | Statistical analysis, cost curves, completeness report and exports | Yes |
| 08 | Independent QA, synthetic fixtures, invariant and failure-injection tests | Yes |

Do not wait for all eight modules before integrating. Once shared contracts, a minimal adapter, scheduler/ledger, one benchmark adapter, and the smoke-test suite work, run the small **development** pilot. Other benchmark adapters can finish in parallel. Main-generation jobs for a benchmark may begin only after its manifest, prompt, extraction, and evaluation smoke tests pass. Do not bypass the MBPP sandbox gate to obtain faster progress.

All paid generation goes through one dispatcher and one study ledger. The eight coding agents do not each start an independent paid experiment or interpret $135 as their own allowance. Mock tests must work with no API key. No automatic credit purchases or limit increases.

## 2. Four benchmark families

### 2.1 MATH-500: mathematical problem solving

- Upstream: `HuggingFaceH4/MATH-500`, pinned dataset revision, test inventory [S08].
- Input allowlist: the problem text only, plus the common response-format instruction. Never expose reference solution, answer, or gold-dependent metadata.
- Main panel: 100 items; sample by a frozen, seeded procedure, approximately preserving level/type proportions where metadata permits.
- Gold grading: pinned Math-Verify, with its documented gold/prediction argument ordering [S09]. Run symbolic parsing/comparison with resource limits.
- Selector: answer-only plurality using a gold-blind equivalence policy, below.
- Output: a concise rationale and one mathematical expression in `<final>...</final>`; a boxed expression inside that element is permitted.
- Parser/equivalence errors are logged explicitly. A grader exception must not silently become semantic incorrectness.

### 2.2 MMLU-Pro: broad knowledge and multi-step multiple-choice reasoning

- Upstream: `TIGER-Lab/MMLU-Pro`, pinned revision and test inventory [S10].
- Input allowlist: question and options. Never include `answer`, `answer_index`, or reference rationale fields.
- Main panel: 100 items, proportionally stratified by subject with deterministic largest-remainder allocation.
- Preserve option ordering across all conditions for an item. Trim documented accidental outer whitespace consistently; record preprocessing version.
- Gold grading: exact option identity. Accept `A` and `(A)` only through a frozen parser and only when A is a legal option for that item. Do not search arbitrary rationale text for a letter.
- Selector: plurality of normalized option identities.
- Output `<final>A</final>` or another legal option.

The upstream card documents an option-whitespace repair in January 2026. Use the resolved pinned data, not an unidentified old local copy [S10]. Subject questions are benchmark material; no live web search is granted to solvers.

### 2.3 BBH-logic4: symbolic, ordering, and state-tracking reasoning

- Upstream: `suzgunmirac/BIG-Bench-Hard`, pinned Git commit [S11].
- Use exactly these four task files under `bbh/`:
  - `logical_deduction_seven_objects.json`
  - `tracking_shuffled_objects_seven_objects.json`
  - `boolean_expressions.json`
  - `temporal_sequences.json`
- Main panel: 25 items per subtask, 100 total.
- Input: the provided `input` string plus the common output contract. The `target` is evaluation-only.
- Use zero-shot prompts. Do not quietly incorporate the repository's few-shot CoT prompt files.
- Choice tasks: normalize the explicit final answer to the original option identity. Boolean task: normalize only an explicit final `True`/`False` value, with case/outer-punctuation normalization fixed in advance.
- Primary metric: average task success; also report all four subtasks separately, with equal subtask weights in the BBH aggregate.
- Report name: **BBH four-subtask zero-shot subset**, not the official full BBH score.

This benchmark adds controlled state tracking and symbolic inference without an LLM judge, external retrieval, or code-generation ambiguity.

### 2.4 MBPP+: executable Python synthesis

- Upstream: the pinned `evalplus/evalplus` MBPP+ inventory and its associated original/sanitized public problem metadata [S12–S14]. Use the installed pinned package's inventory; do not hardcode a remembered task count.
- Main panel: 100 items drawn from that inventory, excluding only development items and documented preflight incompatibilities, never based on model success.
- Input: the official task prompt/interface and examples already intended to be shown to the model. Do not append reference code, `plus_input`, private expected outputs, or additional base tests merely because they are locally accessible.
- Gold evaluation: official full EvalPlus base+plus correctness under the pinned release's comparison and timeout semantics.
- **Primary deployable selector: public-example test selection**, not AST or source-string voting. Rank complete candidates by the fraction of official prompt-visible examples they pass. Tie-break with a frozen, correctness-blind random priority. Give every candidate the same public checks. Never reward faster execution or shorter code through an undeclared tie-break.
- Public checks happen at terminal readout. Their pass/fail results are not fed back into any solver in the main study. This keeps generation tool access matched.
- Every emitted candidate can be privately graded offline for diagnostic coverage; those results cannot affect selection, stopping, prompts, or scheduling priority.
- Report sampling as `public_select@k` plus oracle `pass@k`. **Do not label public-test selection vote@k.** Code behavioral/syntactic grouping is not needed for the primary experiment.
- Output: complete Python solution, including required imports and entry point, inside `<final>`. Allow one removable outer Python fence; do not repair or execute prose as code.

For code, a valid candidate that fails private tests is known wrong. A missing final/program is an operational failure. A broken grading service is unknown, not evidence the program is incorrect.

**Benchmark limitation:** these are established public benchmarks, not a contamination-free or temporally held-out evaluation. Four task families improve diversity but do not establish generalization to all knowledge work, all coding, or agentic tool use.

## 3. Panels, subset nesting, and study stages

### Default `balanced150`

| Panel | Total items | Per benchmark | Conditions |
|---|---:|---:|---|
| Development pilot | 32, disjoint from main | 8 | Budget/control/format checks only |
| Main | 400 | 100 | Full core grid, A=1,024 |
| Higher allowance | 96, nested in main | 24 | Full core grid, A=4,096 |
| Continuation and pooled controls | 192, nested in main | 48 | CP and POOLED at A=1,024 |
| Optional model check | 48, nested in main | 12 | Reduced grid, Flash at A=1,024 |

BBH subdivisions are equal in every panel: main 25/subtask; high 6/subtask; extension 12/subtask; replication 3/subtask; pilot 2/subtask. The panel order is frozen before main inference. Keep the same items for all comparisons within a panel.

### `lean100`

320 main items (80/family), 64 high-allowance items (16/family), 128 extension items (32/family), no second model, and the same disjoint 32-item development pilot. All BBH subtask allocations remain equal.

### Manifest construction

Resolve source versions, inspect schema, validate reference grading, and write:

- A public inference manifest containing only allowed fields.
- A separate private evaluation manifest.
- Source revision IDs, hashes, license metadata, preprocessing version, and selection seed.
- An ordered balanced-wave manifest. One wave contains 16 items: four from each benchmark, and exactly one from each BBH subtask.
- Predeclared nested-panel membership and a reserve list of items.

Do not choose main items by per-item pilot performance, disagreement, correct-minority status, or whichever examples make debate look interesting. Pilot findings can trigger a documented overall model/budget/prompt amendment before main collection. They cannot trigger silent cherry-picking. Keep administrative exclusions in an audit file and freeze them before seeing generated main answers.

No inferential test-set tuning. The grid frontier on the main set is descriptive. Any development-selected fixed policy must be frozen before its main evaluation. No learned router is required for this run.

## 4. Model and request contract

### 4.1 Model locking

Use `google/gemini-2.5-flash-lite` as the primary. Optional replication uses `google/gemini-2.5-flash` on its reduced panel. Current standard-rate planning snapshots are $0.10/$0.40 and $0.30/$2.50 per million input/output tokens respectively [S01–S02]. Fetch live endpoint information before any paid run and lock the exact chosen standard-tier endpoint tag/region when exposed [S06]. Do not use `:free`, preview, automatic model routing, flex/batch, or priority variants by accident.

Prefer a standard Google AI Studio endpoint when its numeric-budget pilot passes, or one explicitly pinned standard Vertex endpoint if it does not. Resolve the actual endpoint slug from current metadata. Never invent a provider string. Save the endpoint response and the realized provider on every generation. If the model or pinned provider disappears, pause the affected study partition; do not silently replace it or turn fallbacks on.

### 4.2 Budget and sampling parameters

For a generation with requested allowance `a`:

- `reasoning.max_tokens = a`
- `max_tokens = a + headroom`
- `headroom = 1024` for MATH/MMLU/BBH; `2048` for MBPP+.
- `temperature = 0.7`, `top_p = 0.95`, one completion per request.
- Omit `reasoning.effort`, tools, search/grounding plugins, automatic message compression, and unsupported parameters.
- Do not send an API seed unless the endpoint is documented and pilot-tested to support it. Study seeds always control manifests and ordering, but do not imply deterministic model sampling.
- Non-streaming is the simplest default. A streaming adapter is optional only with tests for final usage collection and partial-response errors.

Illustrative request body, after replacing the endpoint from the lock file:

```json
{
  "model": "google/gemini-2.5-flash-lite",
  "messages": [
    {"role": "system", "content": "<frozen common instructions>"},
    {"role": "user", "content": "<frozen task and protocol input>"}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 2048,
  "reasoning": {"max_tokens": 1024, "exclude": false},
  "provider": {
    "only": ["<resolved exact standard endpoint tag>"],
    "allow_fallbacks": false,
    "require_parameters": true
  },
  "stream": false
}
```

Use `X-OpenRouter-Cache: false`; record cache-status headers. Independent draws must not be OpenRouter response-cache hits [S05]. Ordinary prompt-prefix caching is a different mechanism; log its usage rather than confusing it with repeated-output caching. Do not add random text to the problem solely to defeat caching.

Google documents numeric Gemini 2.5 thinking budgets as soft allowances, with Flash-Lite's documented range 512–24,576 [S03]. OpenRouter warns that thinking and visible output generally share the total completion cap [S04]. The headroom is not a guaranteed partition. Verify the actual endpoint behavior in the pilot. Do not demand exact equality between requested and realized reasoning tokens.

### 4.3 What is and is not communicated

Store returned reasoning fields, if any, as raw API metadata. Do not send them to peers, parse them as final answers, or assume they are full hidden reasoning traces. All main-study continuation uses the latest **visible written solution**. All earlier responses remain in the external archive, not in the active prompt.

## 5. Protocol DAG and exact reuse rules

Define a generation node as one fresh API sampling opportunity. Define a policy checkpoint separately as a reference to a set of generation nodes and a selector. A request's semantic node ID is not its prompt hash.

Let `root[j]`, j=0..24, be independent samples in a fixed, prespecified order. Never order them by completion time or correctness.

### 5.1 SAMPLE

Generate 25 roots. The primary realized prefix at k uses `root[0:k]` and is charged those k requests. Calculate deployed plurality or public-test selection for every k=1..25.

For the primary realized prefix, calculate oracle coverage from that exact prefix and compute its selection gap against the selector using that same pool. Separately, report an oracle finite-bank pass@k estimator using `1 - C(25-c,k)/C(25,k)`. Do not subtract one fixed-prefix vote result from the subset-averaged estimator and call the difference a nonnegative same-pool selection gap. Use stable integer/combinatorial arithmetic and defined k=1/k=25 edge cases. Do not apply this independent-sampling estimator to correlated revision or debate outputs. Optional subset-averaged vote estimates are secondary and must retain item-level clustering.

### 5.2 PRIVATE_SINGLE

The first cycle is `root[0]`. Generate 24 further private updates, each depending only on the preceding visible candidate plus the task. At total cycle t=1..25, deliver that cycle's candidate. No early stopping because answers repeat or agree. No gold feedback.

Do not include the ultimate stopping horizon in the prompt; this is what permits one long trajectory's prefixes to represent shorter fixed-horizon policies.

### 5.3 PRIVATE_TEAM

For root slots 1..4, generate four private updates per slot. Root slot 0's first four updates are already present in PRIVATE_SINGLE. Thus the extra cost is 16 calls.

For each n in {2,3,4,5} and r in {0,1,2,3,4}, define a checkpoint containing the latest private candidate from each of the first n roots. Select among those n current candidates. Charge n(1+r) generation opportunities, with actual ancestor costs.

These trajectories do not depend on team size. Reusing them across n is legitimate. Their records must not be duplicated and passed off as independent runs.

### 5.4 PEER

For each n in {2,3,4,5}, initialize from the first n roots. For r=1..4, build an immutable snapshot of all n previous-round packets, then issue n fresh revision calls. Each receiver sees its own latest solution and all n-1 peer solutions.

Use a deterministic, correctness-blind permutation of peer order per item/n/r/receiver. Do not label peers as experts, correct, checked, a majority, or a minority. Do not create different roles or prompts for different agents beyond the needed own/peer distinction.

All next-round updates wait for the preceding team round's terminal responses; they never read another agent's same-round result. No global barrier across questions or team sizes. Total additional PEER calls: 4(2+3+4+5)=56.

A team with n agents and R peer rounds uses n(1+R) generations. The five-agent/four-round endpoint is 25, not 16 or 20.

### 5.5 DEEP_SINGLE

Generate fresh independent single calls at A times {2,4,8,16} in the main panel. The A-level is root[0]. At A=4,096, generate only {2A,4A}, so the maximum remains 16,384. Validate all requested settings against the endpoint limit; do not silently clip unsupported budgets.

Keep long-single samples independent of other generation nodes in the first implementation. Cross-panel reuse between, for example, a 4,096-token deep sample and a high-budget bank root is optional only with an explicit alias manifest; it is unnecessary for the stated budget.

### 5.6 CP: communicate once, then privately continue

On the fixed 192-item extension panel, take the already-generated n=4 PEER round 1. From each of those four candidates, generate three private updates using exactly the PRIVATE_SINGLE update template. Deliver a selected answer after each continuation depth h=0,1,2,3. Additional cost: 12 generations/item.

Compare CP at h with PRIVATE_TEAM n=4,r=1+h and PEER n=4,r=1+h. Their nominal generation counts match: 4(2+h). Analyze final delivered accuracy and its change with h. The CP comparison is unconditional on post-treatment correctness or unanimity.

For the fixed panel, define:

`L = mean(Y_CP,h=0 - Y_PRIVATE_TEAM,r=1)`

`H(h) = mean[(Y_CP,h - Y_CP,0) - (Y_PRIVATE_TEAM,1+h - Y_PRIVATE_TEAM,1)]`

This is a descriptive decomposition of a paired controlled-exposure comparison, not a new theorem. Interleave condition requests using a prespecified, correctness-blind schedule and record serving times; interpretation assumes a sufficiently stable pinned service. API dollar costs differ because peer prompts are longer; report both nominal-call and realized-resource views. Do not call H alone a fully cost-optimized causal frontier.

### 5.7 POOLED

On the same extension panel, one new A-budget call reads the four initial visible solutions and the original task, then produces a final solution. It is a centralized synthesis baseline, not a candidate-only oracle selector and not an iterative debate. Charge four root calls plus the synthesis call. Include it at its actual cost; do not pretend it has the cost of four rounds of debate.

### 5.8 Archives

Retain every raw response and every parsed candidate. A checkpoint exposes only its declared current pool to its primary selector. Compute archive coverage separately using only ancestors that its policy really generated. A SAMPLE-25 bank is not a free archive available to a five-agent policy.

On the extension panel, also calculate a secondary archive readout: gold-blind plurality over generated candidate occurrences for text, and the same public-example selector over the code archive. Label these archive policies separately, account for local selection work, and do not interpret repeated serial answers as independent evidence.

## 6. Call counts and resource planning

Full grid, per question/model/base allowance:

| Component | New generations |
|---|---:|
| Independent bank | 25 |
| Additional PRIVATE_SINGLE updates | 24 |
| Additional PRIVATE_TEAM updates | 16 |
| PEER updates | 56 |
| Subtotal | 121 |
| DEEP_SINGLE calls, main A | 4 |
| Total main | 125 |
| DEEP_SINGLE calls, high A | 2 |
| Total high | 123 |

Optional second-model reduced grid: 16 roots + 15 private-single updates + 16 peer updates (n=3,5; R=2) + 8 extra private-team updates + 2 deep calls (4A and 16A) = **57 generations/item**.

Default total, excluding pilot/retries: **67,040 generations**. Of these, 50,000 belong to the main panel, 11,808 to high allowance, 2,304 to CP, 192 to POOLED, and 2,736 to optional replication. A smaller actual count due to explicit successful reuse must be explainable by the alias manifest, never by overwriting independent draws.

The supplied offline calculator assumes mean input length 2,500 tokens/call, saturated completion caps, no cache discounts, and code on one quarter of items. It estimates:

- Main full: about $62.84.
- High full: about $28.97.
- CP: about $2.70.
- POOLED: about $0.22.
- Optional Flash check: about $20.02.
- Plus $5 pilot allowance and $10 contingency: about **$129.76 inference credits**.

The lower-cost preset estimates about **$86.54 inference credits**, including its $5 pilot and $10 contingency. Run `python tools/estimate_budget.py --preset lean100` to recompute it. Both presets leave cash headroom for applicable credit-purchase fees; verify the actual checkout rather than assuming a fixed fee rate.

These are planning estimates, not a bound on unknown input lengths. Live pricing, prompt sizes, token usage, local grading, payment fees, and actual provider capacity must be checked. Local CPU time is reported separately and is not silently converted into zero compute. No external paid grading service is authorized by this spec.

Run `tools/estimate_budget.py` with 2,500 and 5,000 average input tokens to see sensitivity. Replace the average with measured per-benchmark/per-protocol pilot estimates before admitting the main grid.

## 7. Two accounting ledgers and hard-budget semantics

### 7.1 Collection ledger: protects the user's money

For every outbound attempt, maintain exact-decimal monetary fields. Enforce a single atomic invariant:

`settled_spend + in_flight_reservations + unresolved_attempt_reservations + new_reservation <= active_credit_ceiling`

Reserve before sending. Settle only when confirmed usage/cost arrives. The cost estimate must use the pinned price snapshot, bounded completion cap, a conservative input estimate, and a documented safety multiplier. Reconcile with actual usage and `GET /api/v1/key`; use generation metadata when needed [S07].

Initially allow only the $5 development pilot. After its gates pass, the lead explicitly enables the frozen main run. The default soft admission ceiling is $125; $135 is the absolute study ceiling, not an invitation to spend the reserve. The remaining $10 is for known retries/reconciliation, not new optional sweeps. The optional model panel launches only when the completed core plus its forecast and reserve fit.

A robust reservation estimate is an operational safeguard, not a mathematical guarantee about undocumented server billing. Request the owner to use an appropriately capped study key as an additional provider-side guard. Never increase the key cap or buy credits automatically. A shared account's unrelated spending must not be mistaken for this experiment's generation charges.

For ambiguous network timeouts, retain the reservation until reconciliation or conservatively settle the maximum. Never assume a timed-out request was free. Pause when unresolved exposure exceeds $2. On 402/401/403, a cache hit on an intended independent draw, missing required budget support, or unapproved provider drift, open the circuit rather than continuing blindly.

### 7.2 Policy ledger: supports scientific comparison

Each policy checkpoint stores the unique set of generation nodes required to run it from scratch, including all necessary ancestors, plus terminal selection/check costs. Report:

- Nominal generation opportunities and requested thinking allowances.
- Actual prompt/completion/reasoning token counts.
- Charged API cost.
- Standardized uncached input/output-price cost as a sensitivity view.
- Actual retry charges and successful-generation-only charges separately.
- Local public-check/selection CPU time; private research grading cost separately.
- Latency and critical-path length separately from monetary cost.

Sharing root[0] between SAMPLE and PRIVATE_SINGLE saves one collection request; it does not remove root[0] from either policy's deployment cost. Do not double-charge one root inside a single policy when several descendants depend on it.

### 7.3 Budget-controlled comparison

The full grid first produces prespecified accuracy-versus-realized-cost curves. These are not, by themselves, hard per-problem budget results.

Implement a causal budget-admission replay over stored policy prefixes: starting from no responses, admit the next call or entire synchronous round only if its ex ante reservation fits the remaining per-item budget. The admission decision can use already observed usage and rendered prompt size, but not the future request's realized length, hidden correctness, or unseen candidates. Select from the last fully available checkpoint. If initialization does not fit, record an abstention/failure under that budget; do not drop the item.

Allow prefix replay only for horizon-independent prompts and stopping decisions that do not alter generated input. Label it budgeted policy replay. Validate at least a small development sample by live execution with the same admission logic. Report reservation slack and binding-cap frequency. Do not use interpolation through different actual costs to claim an exact feasible policy.

## 8. Prompt and packet contract

The files in `prompts/` are the starting frozen templates. All methods use the same task-specific final-answer format and the same common system instruction.

A normal output is:

```text
<rationale>
A concise explanation of the decisive steps.
</rationale>
<final>
Exactly one final answer or the complete Python solution.
</final>
```

Ask for a rationale of at most roughly 160 words; do not pay a second model to summarize it. Preserve raw output regardless of compliance. Parse only the explicit final element. Apply deterministic, gold-blind extraction rules. One surrounding code fence may be stripped from a code final. No answer-repair call in the main design.

Packets contain the latest visible solution, not confidence claims, private evaluation scores, hidden thinking, or the full chat history. A malformed candidate can be relayed as a bounded raw visible response with an explicit format-neutral availability marker. A truly absent response is rendered as `[No prior candidate response is available.]`; it is not described as wrong. All arms follow the same invalid-prior rule.

No silent final-answer or code truncation in peer packets. When a packet exceeds the pilot-approved input envelope, preserve the final answer/code, trim only the rationale using a documented symmetric rule, and record bytes removed. If the final alone cannot fit, record a context-limit event and invoke the prespecified failure policy, not an ad hoc summarizer. Freeze thresholds before main dispatch. Avoid leaving the prompt token length unbounded through history accumulation.

Stored peer order is determined by a stable hash of seed/item/n/round/receiver, never by return order, confidence, gold, or response quality.

## 9. Storage, contracts, and identity

Suggested new package namespace: `agent_allocation`. This is not an assertion about the user's existing repository. The lead may adapt the namespace after inspecting the actual repository, but preserve the contracts and isolated study ID.

Recommended minimal stack: Python 3.11+, `asyncio`, `httpx`, typed dataclasses or Pydantic, SQLite WAL on local disk, immutable compressed JSON responses, and Parquet exports. One event-loop coordinator with one database writer is sufficient at this scale. Avoid adding a distributed orchestration framework to the critical path.

The included `shared/contracts.py` is the proposed interface seed. Required records:

- `PublicTask`: stable task key, source revision, prompt, output kind, public examples only.
- `PrivateEvalSpec`: gold or private tests, accessible only to grader processes.
- `GenerationSpec`: immutable semantic identity, model/endpoint lock, A/headroom, protocol, item, root slot, round, repetition/draw ID, prompt version, parent references.
- `Attempt`: attempt ID, generation node, timestamps, transport status, request ID, cost reservation and settlement.
- `GenerationResult`: raw response pointer/hash, endpoint observed, all usage fields, completion and parsing status.
- `PolicyCheckpoint`: family/n/round/budget, candidate pool IDs, required ancestor IDs, selector, pool hash.
- `SelectionResult`: chosen ID or synthesized candidate, declared public-check inputs, selector version, CPU time, sealed pool hash. No correctness field supplied to selector.
- `GradeResult`: correctness `true/false/null`, grader status, operational success, evaluation version and private-test hash.

Never key a decision or policy row on `episode_id` alone. Uniqueness includes run/config/model/A/benchmark/item/protocol/n/round/slot/repetition and explicit record kind. Roots are immutable generation records with their own namespace; sharing is through foreign-key references, not a default protocol label.

A prompt hash indexes audit information, not a response cache. Identical prompts with different independent draw IDs must yield distinct API attempts. The only accepted sharing is explicit reference reuse. Export an alias/reuse report and check that all intended independent draws were actually dispatched.

Write raw responses atomically, then commit the corresponding terminal record. Preserve raw attempt metadata and error payloads with secrets redacted. Exclude keys, `.env`, authorization headers, and credentials from logs, reports, Git, and code sandboxes.

## 10. Scheduler design for maximum useful parallelism

### 10.1 DAG scheduling, not nested loops

Do not implement `for benchmark: for item: for protocol: await run_entire_episode()`.

Generate a DAG and enqueue every ready node across all admitted items, benchmark families, team sizes, and model/budget partitions. On completion, immediately unlock dependents. Start root[0] and its long private chain early; that 25-generation path is the longest serial dependency. Prioritize the first five roots over unused extra roots because they unlock all team arms.

Only these barriers are allowed:

- A private update waits for its own previous candidate.
- A PEER round waits for its own team's previous complete round.
- A CP private update waits for its own previous CP candidate.
- A pooled readout waits for its four roots.
- Selection waits for its declared pool, not private gold grades.

There is no barrier across items, benchmarks, unrelated team sizes, or all roots in the full study.

### 10.2 Fair admission and bounded work-in-progress

Admit items in frozen balanced waves of 16, up to four active waves. Prioritize completion of older waves; open another wave when needed to keep the dispatcher busy. This creates hundreds of ready nodes while limiting the number of partly finished items at a budget stop.

Do not run every sampling bank for the entire study before permitting any team continuation. That can consume the budget before producing paired comparisons.

Use a fair queue across benchmark and protocol families, plus critical-path priority. No outcome-aware scheduling. Interleave PEER and PRIVATE conditions in time to reduce systematic service-period confounding.

If spending forecast rises, reduce **future balanced waves** or skip the optional model block according to the predetermined order. Do not abandon expensive questions halfway and then report only fast successes. Report a coverage matrix for all planned/started/completed items and all conditions.

### 10.3 Concurrency and rate limits

Start at 16 in-flight requests and 2 admissions/second. After a 60-second stable window, increase concurrency by 8 and admission rate by 1/second, up to 128 in-flight and 16 admissions/second, unless actual limits require lower values. Treat these as tunable ceilings, not promised provider capacity.

On 429/capacity errors, honor `Retry-After`, halve the relevant endpoint's rate/concurrency allowance, and use randomized backoff. Do not use extra keys/accounts to evade limits. Official documentation says capacity is governed globally and 429s can originate from upstream providers [S07]. Preserve provider pinning even if this reduces speed.

Use persistent HTTP connections; do not create a new client per generation. Set client automatic retries to zero and centralize retry decisions. Default transport attempts: at most 3, including the original. Logical failures such as a completed malformed answer, refusal, or empty length-limited response are not reasons to redraw.

A semaphore should cover only active HTTP work, not dependency waiting, sleep/backoff, grading, or database queue waiting. Otherwise nominal concurrency will not be realized.

### 10.4 Parallel grading

Send completed code candidates to a separate bounded CPU/sandbox queue. Text grading also runs outside the networking event loop. Start with at most `min(8, max(1, cpu_count-2))` grading workers and monitor memory. Grading backlog does not normally block raw generation, but a broken evaluator or disk-pressure threshold does.

Cache grading by benchmark revision + item + source code hash + evaluator/test kind + environment digest. Identical generated code may share a grade, but remains multiple generated candidates and paid sampling opportunities.

### 10.5 Crash recovery and ambiguous requests

Keep a durable state machine: `pending -> leased -> sent -> response_persisted -> terminal`, with retries as separate attempts. Heartbeat active leases. A stale lease is not proof a request never reached the provider.

On restart, reuse terminal artifacts, reconcile sent-but-unknown attempts, and never launch the same leased node concurrently. Use generation IDs to retrieve available metadata, but do not assume the API can recover the full lost response or supports idempotent sampling unless documented. Reconcile unresolved cost conservatively and allow the prespecified retry as a new attempt. Record all charges; one successful result is accepted per semantic node.

A global stop must stop new admissions, then drain or safely account for in-flight requests. Killing sockets is not evidence billing stopped. Do not let local cancellation release unknown-cost reservations.

## 11. Deployable selectors and evaluator isolation

### 11.1 Text plurality

For MMLU/BBH, equality is normalized final-answer identity. For MATH, construct gold-blind classes in deterministic candidate order. Use exact normalized equality first; optionally merge when a restricted, symmetric mathematical equivalence check succeeds. Math-Verify's documented comparison can be asymmetric [S09]; do not blindly use one-way `verify(a,b)` as an equivalence relation.

A safe first implementation uses mutual verification and complete-link class admission: a candidate joins a class only when equivalence succeeds both ways against every existing class member. Timeout/uncertainty means no merge, not a model redraw. Log exact-string versus mathematical-class results as a sensitivity analysis. Selected class representatives are genuine generated candidates.

Choose the class with greatest support among valid candidates. Resolve tied classes using a stable, correctness-blind pseudo-random key derived from item/selection seed/canonical class ID, independent of completion order. Within a winning class use a fixed candidate priority. When all candidates are invalid, abstain; operational delivered accuracy is zero. Store valid-candidate counts and invalid rates.

### 11.2 Code public-example selection

Public checks are the exact official examples included in the prompt, with known expected outputs. They are not EvalPlus private test cases selected after observing results. The input/prompt hash of this public checker is recorded.

Score each candidate on all public cases, including failures/timeouts. Select by pass fraction, then frozen random candidate priority. A public-check infrastructure error pauses this selector, rather than favoring unchecked code. All arms use the same selector; hidden tests are joined only afterward.

This is a weak but legitimate verifier. Report its limitations and the oracle selection gap. Do not silently strengthen it with hidden tests to make code results look better. No LLM judge is required for any primary benchmark.

### 11.3 Sandbox safety and grading fidelity

Generated code is untrusted. Run it in an OS-isolated environment with no network, no API keys, no home directory or repository mount, no Docker socket, a read-only base filesystem, disposable writable scratch, a non-root identity, bounded processes/memory/CPU/output, and a hard termination mechanism. Preserve the official pinned EvalPlus verdict semantics and establish compatibility using reference implementations and deliberately broken fixtures.

Separate public selection from private evaluation jobs. Private reference implementations, full test files, and expected outputs remain in the trusted grader. Candidate execution receives only what is required for the individual invocation, not a directory of gold solutions or an oracle callable. A stock helper that merely removes a few Python builtins is not a sufficient security boundary. If the existing evaluator cannot provide the required separation, agent 05 must adapt the execution boundary and validate verdict parity before code inference scales.

Use the official package's comparison rules for floating values, unordered outputs, exceptions, and task-specific special cases. Do not replace them with generic `repr` equality. Preserve official per-case timeout conventions; the configured 60-second suite watchdog is an outer safeguard and must be checked against reference runtimes on the actual machine. Adjust once during pilot if reference programs require it, then freeze and report it. A worker outage is not a candidate timeout.

Do not force remote code execution or external paid services. Local sandbox access is a preflight requirement for the fourth benchmark. Other benchmarks can run while this requirement is addressed, but the study cannot claim completed four-benchmark results without it.

## 12. Required metrics and statistical analysis

Primary endpoints by benchmark, model, A, policy, and checkpoint:

- Delivered accuracy: last answer for PRIVATE_SINGLE/DEEP/POOLED; plurality for text sample/team arms; public-example selection for code sample/team arms.
- Oracle pass@k for the independent bank, with fixed populations.
- Current coverage and archive coverage for each policy's actual candidate sets.
- Selection gap: coverage minus delivered accuracy for candidate-only selectors. Do not apply that identity to POOLED synthesis against its input bank, because synthesis may produce a new correct answer.
- Actual API spend, standardized token-price cost, prompt/reasoning/completion counts, local selection CPU time, failure and truncation rates.

Dynamics where class identities and correctness are known:

- Per-agent correctness and changes in correct-agent count.
- Wrong-to-correct and correct-to-wrong transitions.
- Last-correct-candidate loss and recovery, separate from delivered-answer changes.
- Agreement, unanimity, answer support and repetition; for code, do not substitute syntactic identity for semantic agreement.
- Root-correct-minority diagnostics defined using pre-communication roots. Report how many independent items enter each stratum.
- For difficulty diagnostics, roots 5..24 can supply an offline independent-from-team-initialization reference bank. Those correctness labels are not router features at inference time.

For code, current oracle correct count is defined by private tests, but a correct-solution class need not have a single textual form. Report correctness counts and public-test behavior; omit unsupported semantic-consensus metrics rather than inventing them.

Use paired bootstrap intervals, clustering all seeds, rounds, team sizes, and methods for a question together. Bootstrap source items within each benchmark; preserve equal BBH subtask weights. Report all four benchmarks separately, plus an equal-benchmark macro average. Do not pool tens of thousands of agent decisions as independent observations.

Prespecify two main contrasts at A=1,024: (i) PEER n=4,r=4 versus PRIVATE_TEAM n=4,r=4 on delivered accuracy; (ii) the CP-versus-private difference in improvement from h=0 to h=3 on the extension panel. Report effect sizes and item-clustered intervals; apply Holm to these two primary tests if reporting p-values. All remaining grid searches are exploratory/descriptive. These are nominal-resource contrasts with actual cost shown, not proof of equal dollar expenditure.

Unknown grades: maintain true/false/unknown, plus parse and infrastructure status. Missing/unparseable model output is operationally unsuccessful, but not a semantic answer-class switch. An evaluator failure is unknown. Report fixed-denominator lower/upper bounds until it is repaired. Administrative jobs never admitted are not model errors and must appear in the completeness report, not be coerced to false.

Required algebraic checks on fully known panels: finite-bank estimated pass@1 equals mean draw correctness; pass@25 equals full-bank coverage; archive coverage cannot decrease as candidates are added; `C_next-C_prev = W_to_C-C_to_W`; delivered accuracy cannot exceed coverage for a selector that returns one member of that same pool. Current coverage and vote accuracy are not required to be monotone.

Do not declare the best test-set configuration a validated optimal policy. Mark its frontier as the best among evaluated configurations on that sample. A generalized controller or mechanism-prediction claim is out of scope unless separately implemented, frozen, and held out.

## 13. Pilot and release gates

### Gate 0: offline, no spending

Contracts compile; immutable IDs and DAG-count tests pass; mocks cover API success/failure/empty response; all four loaders resolve pinned versions; source licenses and public/private separation are recorded; reference graders pass; root-bank aliasing and cost tests pass; sandbox negative tests pass for code.

### Gate 1: live capability canary

Only the lead authorizes the dispatcher. Use a handful of development requests at A=512/1,024/4,096 and at least one long setting. Check endpoint identity, budget support, charged reasoning usage, completion headroom, cache status, empty responses, and raw usage persistence.

Do not require every request to spend its allowance. Inspect a distribution on nontrivial development tasks. If numeric budgets are silently ignored or reasoning usage cannot be audited, pause the affected model-budget claim. Do not automatically relabel an effort bucket as exactly A tokens.

### Gate 2: small paired pilot

Use disjoint development questions spanning all four benchmarks. Exercise sampling, private updates, n=3 peer updates, private-team selection, a long call, code public selection and private grading, resume, and a budget stop. Aim to stay below 500 generation requests and $5 total, with the monetary cap controlling admission.

Detect systematic truncation, malformed outputs, incompatible graders, pathological input growth, missing selector rows, and benchmark floor/ceiling effects. Correct engineering faults and freeze versions before main collection. Do not tune against main outcomes.

### Gate 3: main wave launch

Produce `PILOT_REPORT.md`, endpoint/data/prompt locks, the exact job-count plan, and a revised per-benchmark cost forecast. Then switch the active spend gate from pilot to the approved main ceiling. Run balanced waves with one shared ledger. Large-run authorization is an explicit state transition, never the accidental default of a unit test.

### Gate 4: sensitivity and optional replication

High A and CP/POOLED must not jeopardize completion of the primary paired grid. Their panels are prespecified. Schedule them once the main forecast is trustworthy, using spare capacity under the same ledger. The optional second-model panel starts only after core completion and a remaining-credit check with reserve.

### Gate 5: sealed results

Finish pending grading, seal selection records before private-label joins, reconcile cost, validate invariants, and export the complete report. Do not call the run complete because generation finished while code grading or selectors remain absent.

## 14. Expected output tree and command contract

The lead must implement a CLI with the following behavior. These commands are **required interfaces to build**, not executables already present in this handoff package.

```bash
uv run allocation validate-config --config configs/balanced150.json
uv run allocation prepare-data --config configs/balanced150.json
uv run allocation doctor --config configs/balanced150.json
uv run allocation plan --config configs/balanced150.json --dry-run
uv run allocation pilot --config configs/balanced150.json --allow-paid --max-usd 5
uv run allocation freeze --config configs/balanced150.json
uv run allocation run --config configs/balanced150.json --allow-paid --max-usd 135 --resume
uv run allocation status --run-dir artifacts/RUN_ID
uv run allocation stop --run-dir artifacts/RUN_ID --drain
uv run allocation grade --run-dir artifacts/RUN_ID --resume
uv run allocation analyze --run-dir artifacts/RUN_ID
uv run allocation audit --run-dir artifacts/RUN_ID
```

`--max-usd` is cumulative study spend from the same ledger, not another allowance after pilot. Code must reject a requested maximum above the preset unless the owner explicitly provides a new approved configuration; agents may not autonomously amend it.

```text
artifacts/RUN_ID/
  locks/                 # sources, endpoints, environment, prompt hashes
  manifests/             # public manifest, private pointer, job/policy graph
  state.sqlite           # coordinator-owned, local disk
  raw/                   # immutable per-attempt response objects
  parsed/                # extraction results
  selections/            # sealed before gold joins
  grades/                # private offline verdicts, access restricted
  tables/                # candidates, policies, transitions, costs, completeness
  figures/               # accuracy-cost, pass/vote, dynamics, resource use
  reports/
    PILOT_REPORT.md
    RUN_REPORT.md
    COST_RECONCILIATION.md
    COMPLETENESS.md
    AUDIT.md
```

Static HTML report and CSV/Parquet exports are sufficient. Do not make a dashboard frontend or hosted service a prerequisite for inference. Reports must identify the subset nature of BBH, the public-test code selector, realized versus requested budgets, and all unfinished partitions.

## 15. Acceptance tests that must block dispatch or claims

1. Same-prompt different-draw requests are distinct nodes and produce distinct network opportunities; explicit shared roots are not re-generated.
2. A roots record cannot overwrite a private or peer decision even if an episode label matches.
3. Full main DAG has 125 generations/item; high has 123; reduced replication has 57; CP adds 12 and POOLED adds 1.
4. PRIVATE_SINGLE t=1 and all team r=0 reference the expected immutable roots. Twenty-five-draw sampling does not replace a five-root team baseline.
5. PEER n=5,R=4 has 25 required generation ancestors; PRIVATE_TEAM has the matching nominal count. No future or same-round peer messages are visible.
6. PRIVATE_TEAM independent trajectories legitimately reuse across n; PEER trajectories at different n do not.
7. One returned usage object is charged once in collection and once per logical occurrence in a policy ancestor set; no reasoning-token double counting.
8. Concurrent reservations cannot exceed the study ceiling; simulated timeout, restart, delayed billing, and 429 retain correct reservations.
9. 400-question main results keep the same fixed population across checkpoints; unknown grades and unrun administrative jobs are not silently dropped.
10. A malformed response is not retried as a fresh solution; actual infrastructure retries are bounded and metered.
11. The selector API cannot access private evaluation fields. Secret canary answers/private tests injected into a fixture never appear in request logs.
12. MBPP public selection never uses private base/plus outcomes; hidden reference code is inaccessible to generated code. Program attempts to read secrets, contact a network, fork excessively, or run indefinitely are contained.
13. Text answer-class grouping does not use gold. MATH asymmetry cannot create a one-direction-only equivalence merge.
14. Unknown grader status stays unknown; parser failure and valid wrong answer remain distinguishable in transitions.
15. Missing reasoning usage is null, not zero. A response-cache HIT on a supposedly fresh draw blocks that partition.
16. A larger nominal budget is not automatically reported as more realized reasoning; resource plots show early finishing and binding caps.
17. Terminal selectors have real populated rows. No all-null selector gets plotted as zero accuracy.
18. Synthesis may exceed input-bank coverage; candidate-only selection may not. The corresponding invariant is scoped correctly.
19. Actual peer-order permutation and root identities do not depend on request completion time.
20. Resume is idempotent for stored successful nodes. It does not regenerate an entire item, lose raw artifacts, or silently erase failed attempts.

## 16. Scope and stop rules

Do not add trained routing, RL, post-training, confidence-based selection, neural probes, external search, elaborate roles, new debate topologies, or an LLM judge to the critical path. Finish the paired core first.

Stop or quarantine an affected partition for provider drift, unsupported reasoning settings, cache contamination, widespread extraction failure, faulty grading, ledger discrepancy, leaked gold, or source-version change. Preserve existing raw data. Repair grading from stored outputs when valid; do not regenerate answers merely because repaired metrics look worse.

Success is a complete, auditable, cost-accounted dataset and a clear answer about the evaluated allocations, even when communication has no advantage. A favorable debate result is not a launch gate or completion criterion.
