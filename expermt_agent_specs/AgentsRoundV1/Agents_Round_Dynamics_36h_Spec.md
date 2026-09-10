# Agents Round Dynamics: a 36-hour follow-up specification

**Version:** 1.0, 9 September 2026  
**Purpose:** Implementation handoff for coding agents; a prospective follow-up to `ScalingAgents_Atlas-1.pdf`.  
**Hardware envelope:** 10–20 allocated NVIDIA A100/H100 GPUs; verify actual memory and interconnect before scheduling.  
**Status:** Proposed experiments, not executed. Runtime is conditional on the throughput gates below.  
**Primary target:** Explain and predict how communication changes individual correctness and later team performance, including—but not limited to—initially minority-correct teams.

## 0. Executive decision

Run a paired, round-resolved intervention study, not another unrestricted factorial sweep. Generate neutral independent roots once, branch the same roots into explicitly specified protocols, record every agent at every round, and test whether a small controller can choose who should continue privately rather than read peers. Keep task ground truth outside every deployed controller and selector.

The contribution to seek is **prospective, task-conditional control of communication**, with evidence connecting (1) a measured pre-update signal, (2) the effect of a randomized or paired communication intervention, and (3) held-out downstream utility. Minority preservation alone, confidence weighting alone, a hidden-state correctness probe alone, and shared-prefix branching alone all have close precedents.

The 36-hour run is intended to establish or falsify a focused thesis. It cannot guarantee a publishable result, small-error-bar subgroup estimates, generality to every task or model family, or convergence to truth on every problem. It can yield a clean main experiment, an operational controller test, a controlled task-structure assay, and a limited neural analysis.

**Working thesis, not a conclusion:** The useful quantity for harness control is not simply an agent's confidence or the number of agents endorsing its answer. It is the expected effect of exposing that agent to the current peer information. Predicting this effect may allow a system to retain useful alternatives while still correcting wrong agents.

**Working title:** *Who Should Listen? Predicting and Controlling Correctness Dynamics in Multi-Agent Reasoning*.

---

## 1. What the existing experiment does and does not establish

### 1.1 Source-derived observations

The supplied atlas, dated 7 September 2026, reports the following. These are document-reported exploratory results; the raw run store was not available for this review.

| Atlas location | Reported observation | Role in this follow-up |
|---|---|---|
| §7.4, pp. 7–8 | Peer-informed revision changes 39.2% of answers; matched private revision changes 8.5%. | Motivation for the peer/private intervention. |
| §7.4 | Mean confidence increases 0.853→0.930 over eight rounds, while member correctness changes 0.314→0.320. | Separate consensus, certainty, and correctness trajectories. |
| §7.4 | Current-round oracle coverage decreases 0.438→0.352. | Measure disappearance and later recovery separately from archived availability. |
| §7.4 | In 32 initially lone-correct episodes, 75% lose the correct answer by the final round. | Motivation, not an adequately powered estimate across tasks/scales. |
| §§7.8–7.9 | Independent sampling has a large generation–selection gap; within-item confidence AUC is approximately 0.495. | Require within-question discrimination and strong simple selector baselines. |
| §§8–9 | Effective-rank changes do not imply contraction; available geometry does not reliably distinguish correctness. | Neural features are candidate signals, not assumed truth detectors. |
| §§3, 4.1, 7.6, Appendix B | Missing activation roots, nominal team-size aliasing, binding structural caps, and unaudited HLE labels. | Fix the measurement design rather than extend those defects. |

Figure 4a on p. 8 and the textual agreement statistic in §7.4 have different reported magnitudes. Recompute agreement from raw current-round candidates before using either. Do not silently reconcile them. The supplied audit files are secondary reviews, not independent experimental evidence.

### 1.2 Reuse rules

Reuse existing graders, serving infrastructure, canonicalization code, raw event parsers, activation replay, and immutable storage **after smoke tests**. Reuse legacy samples only within clearly marked legacy analyses or when the checkpoint revision, prompt bytes, sampler, reasoning budget, output contract, and seed policy exactly match the new condition.

Do not relabel old 8,192-total-token runs as new split-budget runs. Do not use the previous 20-item development set as the sole training set for a new controller. Do not put old selected-on-results items into a supposedly untouched confirmatory cohort.

The repository path, checkpoint cache paths, Slurm account, partitions, node topology, and existing CLI were not supplied. They are configuration fields to resolve locally in hour 0, not invented paths in this specification.

---

## 2. Closest literature and the novelty boundary

The following papers were checked against primary online sources on 9 September 2026. Some are recent preprints; their claims should be treated as reported evidence, not field-wide settled conclusions. Bibliographic links are in §23.

| Precedent | What it already covers | Required distinction/comparison |
|---|---|---|
| **Debate or Vote** [R1] | Voting versus interaction; a stochastic belief-update model and interventions favoring correction. | Do not claim that consensus and correctness differ, or that positive correction bias matters, as new discoveries. |
| **Minority Sentinel** [R2] | A meta-classifier over debate logs decides when to replace the majority answer. | Compare static/post-debate selection to intervention **before** the next update. A classifier alone is insufficient novelty. |
| **Demystifying Multi-Agent Debate** [R3] | Diversity-aware initialization and confidence-modulated updates, including training confidence expression and use. | A calibrated-confidence prompt is a baseline, not a reproduction of its trained models. |
| **Hear Both Sides / DAR** [R4] | Selective retention of disagreeing messages during debate. | Compare to disagreement-preserving communication. Our simple deduplication arm must not be called an exact DAR reproduction. |
| **CASE** [R5] | Hidden-state candidate selection and within-question decodability as a predictor of selection gains. | Test incremental prospective signal about communication effects, not just static correctness classification. |
| **Resistance–Receptivity Frontier** [R6] | Preserving a correct answer can reduce uptake of correct peer information; task and reasoning conditions matter. | Measure both correct retention and wrong-to-correct uptake. Preserving every minority is not success. |
| **TreeCredit** [R7] | Shared-prefix operator comparisons and adaptive routing using downstream credit. | Shared prefixes and routing are controls/methods, not novelty. Our question is agent-specific communication susceptibility, measured under fixed continuations. |
| **The Interaction Tax** [R8]; **ArcticSwarm** [R9] | Content/timing of communication, solution convergence, delayed integration. | Delaying consensus or sharing less content alone is not a sufficient claim. |
| **CISC** [R10]; **The Cost of Consensus** [R11] | Within-question confidence evaluation; private/noise controls and debate failures. | Include their core distinctions and controls. |

The most valuable possible finding is a reproducible boundary: **the same pre-update uncertainty or representation signal predicts when peer exposure helps versus harms, and using it improves downstream utility relative to simple confidence, disagreement preservation, random protection, and static selection.** This remains a hypothesis. Establishing a measurement limitation or a robust negative boundary can also be useful, but should not be rewritten as a successful controller.

A further distinction worth testing: a signal can predict which answer is currently correct without predicting who should listen; conversely, it can predict susceptibility to harmful updates without being a strong static correctness probe.

---

## 3. Definitions: pass@k, correct counts, and votes

For task i, checkpoint M, frozen generation recipe b, and independent draw j, let Y_ij∈{0,1} denote correctness. Invalid outputs count as 0 for end-to-end accuracy, with their failure type retained separately.

For a team of N agents at round t:

- `C_it = sum_j Y_ijt`: number correct.
- `m_it = C_it/N`: correct fraction.
- `coverage_it = 1[C_it > 0]`: at least one current correct answer.
- `archive_coverage_it = 1[any answer generated through t is correct]`.
- `selected_correct_it`: correctness of the actual selected final candidate.
- `answer_class_ijt`: observable, gold-independent canonical answer identity.

**pass@5=1 for one realized set means at least one of its five answers is correct. It does not mean exactly one is correct.** A correctness fraction of 1/5 is different. For a binary yes/no question with one correct “yes” and four wrong “no” answers, majority voting fails. In a multiple-choice/open-answer task, wrong answers can split across classes, so correctness count alone does not determine the plurality result.

Under an ideal homogeneous independent-draw model with task-specific success probability p_i:

`C_i0 ~ Binomial(N,p_i)`

`P(C_i0 >= 1) = 1 - (1-p_i)^N`

`E[C_i0] = N*p_i`.

For N=5 and p_i=0.2, expected correct count is 1; coverage is 0.67232; exactly-one-correct probability is 0.4096; binary majority success is 0.05792. These are mathematical examples, not empirical estimates of the user's models.

For a bank with n samples and c correct, use the standard finite-bank estimator [R12]:

`passhat@k = 1 - choose(n-c,k)/choose(n,k)`, for 1≤k≤n.

For random size-N subsets, the finite-bank correct-count distribution is hypergeometric:

`P(C_N=j | bank) = choose(c,j)*choose(n-c,N-j)/choose(n,N)`.

For independent heterogeneous agents with probabilities p_ij, use a Poisson-binomial count model and coverage `1-prod_j(1-p_ij)`. Do not infer it from one homogeneous pass@k curve.

**Important:** Similar learned mistakes do not, by themselves, make independently seeded same-prompt samples statistically dependent conditional on a fixed question. Shared biases can concentrate the wrong answer distribution. Pooling different question difficulties also induces apparent correlation. Communication adds a different source of dependence. Keep these mechanisms distinct.

The root-matching result under identical isolated prompts is largely a construction/sanity check, not a paper claim. Empirical departures become interesting when roles, model composition, shared context, or interaction alter the sampling distribution.

---

## 4. Scientific questions and estimands

### Q1. What happens to the correct-count distribution after communication?

Estimate `P(C_t=c | C_0, protocol, task_family, checkpoint)` and the corresponding selected-answer accuracy. Include C_0=0 and C_0=N, not only minority-correct starts. Label this a descriptive transition summary; C_t alone is not assumed to be a Markov state.

### Q2. What causes changes at each round?

For each agent transition record correct→correct, correct→wrong, wrong→correct, wrong→wrong, plus invalid outcomes and answer identity changes. Within a fixed team the identity is:

`C_(t+1)-C_t = #wrong_to_correct - #correct_to_wrong`.

For a state with fraction correct m, define conditional correction probability a and destruction probability d. Then expected one-step correct-fraction drift is:

`E[Δm | state] = (1-m)*a - m*d`.

This is an accounting identity, not a novel dynamical theorem. It prevents high agreement or many answer changes from being confused with progress.

### Q3. Can pre-update signals predict correctness and intervention value?

Keep two targets separate:

1. `s_j(S_t) = P(Y_jt=1 | observable state)`.
2. `u_j(S_t) = E[Y_j,t+1(private) - Y_j,t+1(peer) | observable state]`.

The second asks who benefits from not reading peers. It is the primary controller target. A positive score favors private revision for that agent; it does not automatically imply that the agent is correct or should dominate everyone else.

### Q4. Does acting on the signal improve later team utility?

Compare full peer exchange, a controller privately revising one chosen agent, and a random one-agent private-revision control. All perform five answer-generating calls per round. Primary endpoint: end-to-end selected correctness at round 3, not just survival of a chosen agent.

### Q5. Does an immediate gain reduce the value of later inference?

From the states produced after one full-peer or private round, apply an identical no-new-peer continuation policy with independent suffix seeds. Compare absolute final utility, not just improvement relative to an easier/harder starting point. Current-pool loss, archive loss, inability to regenerate, and selector failure are separate outcomes.

### Q6. What changes with task structure, checkpoint, and reasoning allowance?

Natural task families test external differences; a separate controlled program-trace assay manipulates checkable evidence while holding the underlying task/claims fixed. A frozen-packet receiver comparison separates receiver behavior from differences in what teams generate. These are deliberately different estimands.

---

## 5. Cohorts and split discipline

### 5.1 Natural cohort: 384 tasks

Use 192 MATH-500 items [R13] and 192 MMLU-Pro items [R14]. These provide exact/verified mathematical answers and fixed-key multi-choice grading without the unaudited self-judge used for HLE.

For each family assign, **before new generation**:

- 48 training items;
- 24 validation items;
- 120 held-out test items.

Totals: 96 train, 48 validation, 240 test. Freeze dataset revision, item IDs, split seed, prompt/answer extraction policy, and grading code hashes. Use stratified sampling over available subject/level metadata. Do not call the selected subset a full benchmark evaluation.

MMLU-Pro is not a pure recall dataset; it includes reasoning-focused questions. Do not interpret MATH versus MMLU-Pro as an isolated verifiability intervention. Report available subject categories; additional knowledge/derivation annotations must be assigned without seeing experimental effects.

MATH grouping: canonicalize predicted answers using a frozen parse/normalization procedure independent of the reference. Grade separately against the reference with the existing validated symbolic evaluator or a pinned equivalent. Ambiguous parses/timeouts remain explicit; no solver-model judgment silently substitutes for the evaluator.

MMLU grouping: exact option ID after a fixed per-item option permutation. All agents/protocols see the same permutation for that item. No option key is exposed. Diagnostic alternative permutations remain in the same split group.

Select the cohort without looking at topology outcomes. Preserve all root-correctness strata in the natural panel. A floor/ceiling outcome is still an outcome.

### 5.2 Root banks

For Qwen3-8B, generate 16 independent neutral roots per natural item. The first five form the natural team; the remaining eleven provide an independent auxiliary bank for difficulty/coverage checks. All sixteen can be used for descriptive bank statistics, explicitly acknowledging overlap with the team.

For Qwen3-32B, use a frozen 96-item subset, 48 per family: 24 train, 12 validation, 60 test overall. Generate sixteen native roots per item. Select this subset from the original manifest, not on measured cross-scale gains.

A root uses the same prompt regardless of its future protocol. Model identity and team framing are not included in the prompt. Independent seeds are part of the request identity. Never put an agent slot name into an otherwise neutral root prompt just to distinguish requests.

### 5.3 Composition-controlled challenge cohort

Target at most 96 distinct test items from the natural cohort with at least four correct and four incorrect valid roots in the sixteen-draw 8B bank. Per eligible item, form two five-member teams **without replacement within a team**:

- `C0=1`: one correct and four incorrect roots;
- `C0=4`: four correct and one incorrect root.

Reuse the exact roots across the protocols tested on that team. Candidate overlap between the two challenge teams is allowed and logged; neither is an independent item replicate. No fabricated solution is inserted. No root is duplicated to pretend there are five independently generated agents.

Eligibility and team composition use offline labels, so this is an explicitly **label-conditioned intervention assay**, not a deployable initialization method or natural-distribution accuracy estimate. Never merge its accuracies into the natural cohort average. All of its labels remain inaccessible to the runtime.

Use a fixed hash order among eligible items and candidates. If fewer than 96 items qualify, run all eligible items and report the actual number. Do not keep sampling indefinitely to manufacture the desired stratum. A larger bank or additional cohort is a separately budgeted extension.

Classify wrong-answer concentration separately: concentrated wrong plurality, fragmented wrong classes, and ties. `C0=1` is not synonymous with “four agents endorse the same wrong answer.” Report that subset explicitly.

### 5.4 Controlled task-structure assay

Generate 64 fresh, deterministic small program-trace tasks, balanced over dependency depths 8 and 16 and over two operator mixes. The task is to evaluate a straight-line integer program modulo 97. All operations and inputs are visible. A CPU interpreter computes the reference privately.

Each program must depend on prior state rather than independent unused calculations. Example permitted updates: `x=(a*x+b)%97`, `x=(x+y)%97`, `y=(x+2*y+b)%97`; track sequential assignment semantics exactly. Use coefficients 2–11 and seeds 0–96. Disallow division, imports, loops, randomness, file access, and undefined behavior. A second independent evaluator validates the gold output.

Create two candidate traces using the generator, not the model: one correct trace and one trace with one arithmetic error at an internal step, followed by internally consistent propagation. Verify that the final answers differ. Randomize which trace is A/B. This is a **controlled supplied-candidate assay**, not a measurement of natural pass@k or actual five-agent generation.

Before an update, present a fixed binary provisional answer (correct for half the tasks, wrong for half), labeled as a provisional candidate rather than falsely attributed to that model's own earlier generation. Crossed packet conditions and limitations are in E3. This deliberately isolates receiver response; do not describe its provisional correctness as native model accuracy.

### 5.5 Optional native-code bridge

If the core launch gate is green, add 64 BigCodeBench tasks using the existing containerized grader [R15], preferably excluding the previous 200 items. Run private/full-peer/balanced protocols at 8B. Keep this extension separate because free-form code adds answer-equivalence and tooling problems.

Never group programs by hidden-test success for selection. AST voting is a labeled weak diagnostic only. Behavioral fingerprints, if used, must use public or task-generated non-evaluation inputs. Agreement on finite inputs is not proof of semantic equivalence. Hidden tests are final grading only. Runtime tools receive no reference implementation or hidden test outputs.

---

## 6. Generation contract and compute manipulation

### 6.1 Checkpoints

Core: original dense `Qwen/Qwen3-8B` and `Qwen/Qwen3-32B`, using pinned revisions compatible with the atlas [R16–R17]. Verify local hashes; do not silently substitute an updated Instruct/Thinking model. These comparisons vary released checkpoint/parameter scale, not isolated pretraining compute.

Use one precision throughout the primary comparisons, preferably BF16. Do not quantize only the larger model and then interpret the difference as size. Do not compare raw residual coordinates across different models.

### 6.2 Sampling

Keep thinking mode on, temperature 0.6, top-p 0.95, top-k 20 for answer-generating calls, as in the atlas and Qwen's documented thinking recipe [R16]. Independent stochastic request seeds must be explicit and logged.

Core reasoning allowance: **1,024 thinking tokens**, followed by an independent **256-token final-output allowance**. Receiver assay: 1,024 versus 4,096 thinking tokens, always with the same 256-token final allowance.

Implement split budgets using the existing tested backend mechanism. If reasoning reaches its limit, end the reasoning span with the model's native boundary and resume answer generation from that token prefix. Record whether this boundary was generated naturally or injected. Do not merely shorten a shared thinking-plus-answer cap. Do not claim the injected boundary is an unmodified natural sample. Preserve exact token IDs across the continuation.

The two-phase implementation can entail two backend requests; ledger tables count one **logical solver decision**. Count both physical requests and any repeated prefill in runtime cost. No second fresh solution prompt, hidden retry, or best-answer extraction is allowed.

This is a budgeted inference study, not an estimate of unrestricted checkpoint ability. The Qwen model card recommends substantially larger output budgets for broad capability evaluation; our cap is an intentional resource intervention. Show realized thinking-length distributions and cap-hit rates, not just nominal labels. Do not force models to continue if they end reasoning early.

### 6.3 Final report

Use the same compact schema everywhere:

```
<final>canonical answer or option ID</final>
<evidence>one or two concise, checkable reasons, at most 120 tokens</evidence>
<confidence>probability from 0 to 1</confidence>
```

Answer precedes confidence. The total final report has at most 256 tokens. For code extension only, use a separately frozen larger final budget and include it in the runtime estimate.

Avoid a deeply nested JSON action schema. Missing/malformed answer is an incorrect delivered answer; missing confidence is a missing feature, not automatically 0.5. Parsing must not search the private reasoning span for a favorable answer. Diagnostic validity-conditional results are secondary.

At the smoke test, if valid final answers are below 95% due primarily to format or final-budget failures, repair the schema/cap **before freezing the run**. Regenerate affected pilot data; update token projections for every condition. Do not change the cap selectively on difficult test items.

### 6.4 Context and memory

All iterative protocols use: task, the agent's original root report, its latest report, and the specified current peer packet. If root and latest report are identical, include one copy. Do not include private thinking traces. This is a finite report-memory harness, not full-transcript conversation.

Preserve exact root/latest reports and all messages in the immutable store. Every protocol has the same external archive for retrospective analyses; the deployed agent sees only the memory allowed above. The archive is not silently shown only to the proposed controller.

Peer confidence numbers are stripped in the base protocols. They remain measurable in logs. A calibrated-confidence communication ablation is optional and explicitly different from the base packet. Do not erase confidence wording embedded in the evidence field after generation unless a frozen sanitizer does so for all protocols.

---

## 7. Protocols: what happens at every round

N=5 and rounds t=0,1,2,3. Round 0 is the common neutral root team. Three update rounds are mandatory. There is **no early stopping on unanimity** in the measurement run; unanimous error is not an absorbing state by assumption.

All main protocols make five logical answer-generating calls per update round. Their input tokens, communication, critical-path length, and realized GPU cost can differ. Matched calls are not matched FLOPs.

| ID | Update rule | What it tests |
|---|---|---|
| `FRESH` | Five new independent task-only solutions each round, no previous answers or peer messages. | More sampling without interaction. |
| `PRIVATE` | Each agent revises using task + its root + its latest report; peer packet is empty. | Repeated private reasoning with the same number of generation opportunities. |
| `PEER` | Each agent sees the other four agents' previous-round reports and revises synchronously. | All-to-all full report exchange. |
| `BALANCED` | From the other four reports, retain one report per canonical answer class, hide class multiplicity, and rotate the representative deterministically when a class has several reports. | Simple disagreement-preserving communication. Inspired comparator, not exact DAR. |
| `HUB5` | A preassigned rotating/randomized leader reads the other four previous reports and emits its answer/evidence. Then each of the other four revises after seeing this current leader report and its own root/latest report. | A functioning central broadcast, exactly five calls per round. |

`HUB5` is explicitly a sequential 1+4 update, not an endogenous delegation benchmark. Randomize the leader at episode initialization and hold its identity fixed for that episode. Record the order. A round's leader output is earlier than the four worker outputs. Do not disguise this asynchronous information structure.

`BALANCED` removes multiplicity and some potentially distinct reasoning. Its comparison with PEER is a policy-bundle effect, not a pure social-pressure effect. E3 isolates vote-count metadata while preserving candidate content.

For every peer packet use stable pseudonymous agent IDs. Randomize packet order with a recorded schedule independent of correctness, shared across paired compatible arms. Never sort messages by gold correctness or fit a test-outcome-dependent order.

**End-of-round selection:** primary selector is plurality over the five current canonical answer classes. Ties use a precommitted uniform class-level tie-break, not first-in-list. Also report expected correctness under all tied winning classes analytically where possible. Invalid answers cannot form a winning content class; if all outputs are invalid, selection fails.

Report root, current-round, and full-archive selectors separately. **Mandatory practical baseline:** FRESH also votes over all independent candidates accumulated so far: 10, 15, and 20 candidates after rounds 1, 2, and 3. Its five-current-candidate result is only the matched-pool dynamics diagnostic; do not claim to beat independent sampling by comparing against a baseline that discards its earlier independent samples. Include this cumulative FRESH vote and its confidence/CURRENT selection variants in the cost–accuracy comparison. For interacting protocols, a full-archive vote includes repeated descendants and must not be interpreted as independent voting.

---

## 8. E0 — Reanalyze the existing atlas (CPU first)

Before expensive runs, extract from the legacy raw item/round files:

1. Exact root and per-round member answers, validation flags, confidence, correctness, and canonical classes.
2. Current-round and cumulative-archive coverage, separately.
3. Trajectories conditional on C0=0,1,2,3,4,5 where N=5.
4. Correct-minority and wrong-minority survival; transitions into and out of unanimity.
5. Same-question confidence/geometry performance with question-grouped splits.
6. Actual request hashes, sampler seeds, model hashes, call counts, token counts, and stop reasons.

Do not reuse the shipped geometry-table agreement field. Flag mismatched pool definitions and orphaned episodes. Do not discard bad transitions to make the counts consistent. Produce an audit table listing the reason each analysis row is included/excluded.

Use legacy HLE labels only as legacy judge-agreement labels until independently audited. Existing code execution labels are usable subject to grader and environment checks. The purpose is to recover inexpensive dynamics and choose frozen feature definitions; it is not independent confirmation.

Deliver: `legacy_audit.json`, `legacy_transitions.parquet`, `legacy_root_count_distributions.json`, and a brief discrepancy report.

---

## 9. E1 — Root supply, vote structure, and baseline equivalence

Generate banks in §5.2. Compute pass@k for k=1,2,3,5,9,10,16 where supported, plus empirical/subset plurality and confidence-weighted plurality on **identical candidate subsets**.

At k=1, oracle coverage, plurality correctness, and any non-abstaining selector must coincide on the same candidate under the same validity convention. Make this an automated assertion.

Record the full class histogram, not just number correct. For each task show how often a correct candidate is a strict majority, a unique plurality, a tied plurality, or below the plurality. For code bridge, do not equate correctness class with semantic class.

Validate independent-team construction against the auxiliary bank. Same model + same prompt + same sampler + independent seeds should give the same root sampling law. Use uncertainty-aware predictive checks; do not fit a binomial with an unregularized p estimated from only five observations and declare correlation when it misses.

Recommended descriptive estimator: Beta(1/2,1/2) smoothing from the auxiliary bank and a beta-binomial predictive distribution for the next five draws, clearly interpreted as parameter uncertainty, not proof of exchangeable correlated generation. Also compare empirical finite-bank subset results.

Difficulty is for analysis and stratification only. Ground-truth-derived pass rates are forbidden runtime controller features.

---

## 10. E2 — Natural and controlled-composition round dynamics

### E2a: 8B natural panel

Run all five protocols on all 384 common-root teams for three rounds. Schedule training and validation items first so the controller can be frozen while test trajectories are being generated. This scheduling must not change seeds or batch-dependent sampling semantics.

### E2b: 32B replication

Run PRIVATE, PEER, and BALANCED on the frozen 96-item subset, three rounds, using native 32B roots. Compare their paired differences with 8B differences on the exact same item subset. This is a model-by-protocol comparison, not a claim that initial team quality was held constant.

### E2c: challenge starts

On up to 96 eligible 8B test items, run PRIVATE, PEER, BALANCED, GATE, and RANDOM_PROTECT from both C0=1 and C0=4 constructed teams. Freeze team selection before any challenge continuation. Dispatch the two controller-based arms only after the controller is frozen; the three fixed arms can run earlier. These two additional arms are part of the core because they directly test correct-minority benefit and correct-majority harm under the same controlled starts.

### E2d: matched-root reasoning-budget replication

On a frozen 32-item test subset of the shared 8B/32B panel (16 tasks per family), additionally run PRIVATE and PEER for three update rounds with a 4,096-thinking-token allowance and the same 256-token final allowance. Reuse the checkpoint's exact low-budget roots. The corresponding 1,024-token runs already exist in E2a/E2b.

This adds 32×2 protocols×3 rounds×5 agents = 960 decisions per checkpoint, or 1,920 total. It isolates the effect of spending more reasoning allowance **during updates**, rather than changing initial candidate supply. It does not estimate high-budget root pass@k. With only 32 items, treat detailed task-by-size-by-budget interactions as secondary and show wide intervals; do not infer a phase transition.

### Required round outputs

- Correct-count distribution over 0…5; selected accuracy; current and archive coverage.
- Agreement fraction, class entropy, valid count, unique class count, unanimity-correct and unanimity-wrong.
- Correction and destruction probabilities with denominators of initially wrong and correct agents, respectively.
- First correct-answer loss time, recovery time, and final survival; handle right censoring at round 3.
- Persistence and dissemination of the initially correct hypothesis, distinguished from preservation by the same agent.
- Novel correct answers on C0=0 teams and new correct classes after a previously covered team loses coverage.
- Self-reported confidence, Brier/NLL, within-item ranking, answer logprob, and realized thinking length by round.

Matching an old answer does not prove copying. Log declared cited source IDs if available, but call this reported attribution. Causal influence requires randomized packet manipulations or matched exposure interventions.

Do not interpret attrition of the current correct hypothesis as permanent erasure of model knowledge. At inference, weights are fixed; information can remain in the model, archive, or future sampling distribution.

---

## 11. E3 — Frozen-content receiver assay: vote multiplicity, checkable evidence, size, and compute

Use the 64 controlled program tasks from §5.4. Each has the same two candidate traces, one correct and one incorrect, a fixed provisional answer, and a known incorrect internal step. Neither trace is labeled correct to the model.

Five presentations:

1. `PRIVATE_TRACE`: task and provisional trace only.
2. `MINORITY_UNCHECKED`: both candidate traces once; metadata says one of five endorsements favors the correct candidate, four favor the wrong candidate.
3. `MAJORITY_UNCHECKED`: identical traces; endorsement metadata reversed.
4. `MINORITY_CHECKED`: presentation 2 plus two local arithmetic-check results.
5. `MAJORITY_CHECKED`: presentation 3 plus the exact same two local checks.

The endorsement counts are synthetic randomized experimental metadata, not a claim that five agents independently generated these traces. Keep packet content and length identical between the two count assignments apart from the swapped counts. Cross counts with trace order and provisional correctness.

For checked presentations, reveal verification of two preregistered internal transitions, including the corrupted transition and one matched valid transition; exclude the final transition and final output. The unchecked counterpart contains the same trace locations with status `NOT CHECKED`, not an additional explanation. This supplies useful partial evidence by construction, so it estimates **ability to use supplied diagnostic evidence**, not ability to find a useful test. Report this distinction prominently.

Use both checkpoints, both reasoning allowances (1,024/4,096), and two independent completion seeds. Total ceiling: 64×5×2×2×2 = 2,560 logical decisions. Final-output budget stays 256. Model weights remain fixed.

Primary assay outcomes: switching to the correct candidate, retention when the provisional answer was correct, correction when it was wrong, final confidence, answer-logit margin, and their interactions with endorsement counts, evidence, depth, model, and reasoning allowance. The model can also reject both supplied choices; grade its produced answer rather than forcing agreement with the packet.

All comparisons use the same task and candidate content. This separates social-count sensitivity from the independent generation of better roots at larger scale. It does not establish multi-round performance by itself; E2/E4/E5 supply that link.

A positive result only with the engineered diagnostic check should be written as an evidence-availability/use result, not general minority recognition.

---

## 12. E4 — Prospective signal learning and a closed-loop communication controller

### 12.1 Targets and training data

Fit two distinct small prediction systems:

- `CURRENT`: candidate correctness using only information available after that candidate was generated.
- `EXPOSURE_VALUE`: private-minus-peer next-step correctness using information available **before** the next update.

For every 8B training/validation PEER state at t=1 and t=2, fork each of the five agents into one private next-step revision. The peer next-step outcome already exists in the main trajectory. Other agents do not need regeneration for the one-step individual target. This adds 144×2×5 = 1,440 logical decisions. Root-state private counterparts at t=0 already exist in PRIVATE and can be reused if byte-exact construction matches.

Use one stochastic outcome per branch as noisy supervision, not an exact individual causal effect. Aggregate estimates over items; never claim the realized paired difference reveals a deterministic counterfactual truth for that candidate. A repeated-seed subset is a priority extension.

Train two regularized logistic predictors, `p_private` and `p_peer`, on branch correctness. Their difference estimates `u`. Compare to direct ridge regression on the paired {-1,0,1} difference as a validation-only model choice. Maximum feature/hyperparameter choices are fixed below; do not search a large controller family.

Training sets include generated states but are grouped by original item. No branch, round, candidate, alternative option permutation, or challenge team from a test item can enter training or calibration.

### 12.2 Cheap feature set

Predefine a maximum of approximately twenty behavioral features:

- Own current confidence and missing-confidence flag.
- Own answer mean token logprob and answer length.
- Own previous confidence and confidence change; unavailable-history flags.
- Whether own answer changed since previous round and since root.
- Own answer's current class count and prior class count.
- Number of unique peer answer classes and peer-class entropy.
- Largest competing class count and count margin relative to own class.
- Mean/max observable peer confidence and own-minus-peer confidence, even when confidence is not transmitted.
- Mean peer answer logprob, recorded only if runtime access is available to the controller.
- Reasoning length, cap-hit flag, output-validity flag.
- Round index and model identity when applicable.

No item ID, reference answer, judged correctness, root pass rate, future stability, or final consensus feature is allowed. Task-family indicators can be included for within-family prediction, but also report a model without them and leave-one-family-out transfer.

The controller can have more observational access than an individual model, such as logged confidence, but its interface must be frozen and included in the method description. Feature extraction calls or replay costs are counted.

For CURRENT use a matching regularized correctness predictor. Baselines: answer frequency, raw confidence, validation-recalibrated confidence, answer logprob, and frequency+confidence. A shallow gradient-boosted tree is an optional dynamics-only Minority-Sentinel-inspired comparator, **not** an exact reproduction of its semantic-audit system.

### 12.3 Controller

At each round, before anyone updates:

```
features = features_from_observable_history(state)
u[j] = p_private(features[j]) - p_peer(features[j])
protected = argmax_j u[j]        # tie broken by frozen random schedule
for j in agents:
    action[j] = PRIVATE if j == protected else PEER
execute the five updates synchronously from the previous-round state
```

Exactly one agent privately revises; it is not forced to keep its answer. The other four receive normal previous-round peer reports, including the protected agent's previous report. Thus the intervention changes the selected agent's incoming information, not its ability to transmit its existing evidence.

This always-protect-one design is chosen for a clean action-budget control. If all predicted values are negative, it may be inferior to PEER; that is a meaningful failure mode. Do not add a test-tuned threshold after seeing results.

Run `GATE` and `RANDOM_PROTECT` on all 240 natural test items for three rounds after training/validation decisions are frozen. Random protection always privately revises exactly one uniformly selected agent per round using a recorded schedule. Both spend five logical calls per round and use the same selection rule as PEER.

Do not describe the controller as choosing “the correct agent.” It selects the agent predicted to benefit most from private continuation. Its success criterion is improved final utility, not just protection accuracy.

### 12.4 Static selection competitors

Using frozen generated pools, evaluate root/current/final candidate selection via plurality, confidence-weighted plurality, CURRENT, and a dynamics-only static selector. All are trained/calibrated only on training/validation items. Archived candidates are available to every archive selector.

Budget a small set-wise LLM selector audit on held-out root, PEER-final, and GATE-final pools: 240×3 = 720 additional logical 8B calls, with the same 1,024+256 cap. Return a candidate ID only; do not allow a newly synthesized answer. Candidate order is randomized. This is a secondary diagnostic, not the common primary selector.

The decisive question is whether changing trajectories adds value beyond selecting more intelligently from already generated candidates. A selector-only gain is valuable but belongs to a different claim.

### 12.5 Success and failure criteria

Success requires improvement on the natural test population and an informative resistance/receptivity trade-off, not only a gain on C0=1. Report harm when the initial plurality/majority is correct and inability to find an answer when C0=0.

CURRENT may work while EXPOSURE_VALUE fails. EXPOSURE_VALUE may predict one-step effects while closed-loop utility does not improve. GATE may perform no better than random protection. These outcomes falsify different parts of the proposed mechanism; do not merge them into a generic “promising trend.”

---

## 13. E5 — Common continuation: separating immediate selection from future utility

Select 64 test items at 8B and the 60 available 32B test items plus, for runtime upper bounding only, capacity for 64. For strict held-out reporting the 32B analysis contains **60** items; do not substitute validation items just to reach 64.

Take the states after round 1 in PRIVATE and PEER. For each parent state, run two independent suffixes. Each suffix is one further private revision by all five agents, using the exact same continuation procedure: task + own original root + own current branch report, no new peer messages. Apply the same plurality rule afterward.

This is a common **report-state continuation**, not preservation of a live hidden state between calls. The conditioning reports differ because the parent intervention differs. Its effect includes information retained in those reports; model weights do not change.

Let J_0(S) be immediate selected correctness and J_5(S) expected selected correctness after the additional five decisions. Seek or falsify an ordering reversal:

`E[J_0(S_peer)-J_0(S_private)] > 0`

but

`E[J_5(S_peer)-J_5(S_private)] < 0`.

Do not infer a reversal merely because the initially better state has less remaining improvement. Also examine current coverage, regeneration after loss, and archive selection. Average the two suffix seeds within item; they do not double the number of independent tasks.

Default conservative budget reserves 1,280 decisions per checkpoint (64×2 parents×2 suffixes×5 agents). Actual 32B held-out count is 1,200. A second common private round is an extension, not silently part of the core.

A root-reset control can reuse a fresh task-only bank: it establishes a reset baseline, not state-conditioned continuation. It should not be pooled with the primary suffix estimate.

---

## 14. E6 — Neural/uncertainty instrumentation

### 14.1 What changes at inference

There is no weight update in the proposed experiment. Therefore the evolving quantities are activations, conditional token distributions, reports, and interaction history—not parameter-space geometry. Cross-agent residual-space geometry is meaningful for copies of the same checkpoint. Raw activation coordinates from 8B and 32B are not directly comparable.

The current atlas does not show a reliable geometric correctness signal. Treat neural features as a tested extension with a precommitted compute ceiling, not as the controller's assumed foundation.

### 14.2 Capture cohort

At 8B, select 144 natural items by frozen hash: 64 train, 16 validation, 64 test, balanced between families. Capture all five agents at root and after each of the three rounds for PRIVATE, PEER, and BALANCED. Reuse root captures across protocols. Maximum unique replayed decisions: 144×5×(1+3×3)=7,200.

Store three relative-depth layers: floor(L/4)-1, floor(L/2)-1, floor(3L/4)-1, clamped to legal zero-based indices. The middle layer is primary; the others are exploratory. Read native layer outputs consistently; do not confuse them with embedding or final normalized states.

Structural anchors:

1. Last prompt token, after the packet but before generated reasoning.
2. Last token immediately before `<final>` answer content.
3. Last answer-content token, before confidence content.
4. End of the final report.

Save exact token offsets and marker parse validity. No anchor is guessed from a fixed 128-token offset. Record actual prompt/output lengths. At a byte-identical neutral root prompt the initial hidden state should be essentially identical across agents; meaningful diversity arises after sampled generation. This is a sanity check, not evidence of collapse.

### 14.3 Features

At fixed N=5 report both scale-free and scale-carrying features:

- Participation ratio and eigenvalue-entropy effective rank; ceiling at most N-1=4 after centering.
- Leading eigenvalue share, total variance, average/min/max pairwise distance.
- State norms and cosine distances after normalization.
- Per-agent distance to the peer centroid/nearest peer and change from its own previous-round state.
- Alignment of an agent's displacement with a pre-update peer-versus-self direction, recording near-zero denominators.

A team-level rank statistic is permutation invariant and cannot, by itself, tell which agent is correct. It may predict a team's susceptibility to convergence; choosing an individual requires candidate-specific features.

Raw and normalized metrics must be shown together. Five-sample rank is ensemble geometry, not intrinsic dimensionality of reasoning. Direction signs and scales fitted per item are not a shared global truth/consensus axis.

### 14.4 Prospective evaluation

Two timing claims are allowed:

- Pre-answer features predicting the upcoming answer are genuinely prospective for that answer.
- Post-answer features can predict that answer's correctness and the **next** round, but cannot establish the cause of an answer already emitted.

Test whether neural features improve CURRENT or EXPOSURE_VALUE beyond the identical behavioral/text baseline on held-out items. Report within-item AUC, paired ranking accuracy, Brier, and policy relevance. Do not call high pooled AUC minority recognition.

Use regularized linear probes; all standardization, feature selection, PCA/random projection settings, layer/anchor choice, and calibration are trained or selected without test items. A train-only fixed random projection to 256 dimensions may reduce cost; compare a behavior-only baseline and shuffled-within-item neural features. No layer sweep on test data.

For answer-identity confounding, include option/answer class baselines and check fixed alternative option permutations within the same split. Preserve phase and answer-format controls.

The default core does **not** run activation patching or a neural online controller. With spare GPUs, evaluate a frozen neural-augmented controller on the same 64 test items as a matched behavioral controller, counting every replay/feature-extraction call. Do not substitute an untested neural claim into the main paper.

### 14.5 Replay engineering

Use the existing exact-token replay path when possible. Store token IDs at generation, not only decoded strings. Verify model/tokenizer revision, boundary insertions, and teacher-forced token positions. Check replay logits on a small parity panel; require no unexplained differences large enough to change the analysis, but do not demand unrealistic bit identity across different numerical kernels.

Use forward hooks on selected layers and gather only selected token states. Disable autograd and avoid materializing all layers/all tokens with `output_hidden_states=True`. Store BF16/FP16 features, compressed metadata, and extraction hashes. Do not persist full vocabulary logits for every thinking token. Full-vocabulary entropy, if desired, is computed only at a few designated anchors; top-k logprobs are not full entropy.

Estimated selected-vector storage is modest (order of 1 GB before indexing/metadata); replay time, not storage, is the likely constraint. Benchmark it. Put scratch shards on local storage and consolidate atomically.

---

## 15. Statistics and interpretation

### 15.1 Units and uncertainty

The independent generalization unit is the original task, not an agent, round, candidate pair, layer, fork, or suffix seed. Bootstrap tasks, preserving their complete protocol/seed tree. Use paired item differences wherever roots/items are shared. Report family-specific estimates and a prespecified equal-family macro average.

All variants of one task—including option permutations and challenge teams—stay in one split. For generated programs, split/group by program/template seed as appropriate. Do not claim 14,000 transitions constitute 14,000 independent reasoning problems.

For within-question discrimination, average pairwise correct-versus-wrong ranking performance within mixed-correctness items, then average items. Report how many items are eligible. An AUC conditional on mixed items does not summarize all-zero/all-one pools.

### 15.2 Primary hypotheses (freeze before test-label joins)

1. `GATE - PEER`: selected round-3 accuracy on 240 natural test tasks.
2. `GATE - RANDOM_PROTECT`: same endpoint and tasks.
3. `PEER - PRIVATE`: current-round coverage change from root to round 3 on the natural test panel.

Report two-sided paired intervals and tests; apply Holm across these three confirmatory contrasts. Do not redefine a failed contrast as success after looking at its sign. The mechanism narratives require their additional descriptive/secondary evidence; significance on one endpoint is not enough.

Secondary: challenge C0=1/C0=4 outcomes, receiver count×checkability×budget interactions, 8B/32B protocol differences, common-suffix utility, confidence calibration, and neural incremental prediction. Use clear multiplicity families or exploratory labels for these, and show intervals rather than binary claims.

### 15.3 Power expectations

A rough paired-binary calculation with discordance rate 0.20 gives standard error sqrt(0.20/240)≈0.0289. The unadjusted 95% half-width is approximately 5.7 percentage points, and a conventional 80%-power effect is about 8.1 points before multiplicity correction. This is a planning approximation, not a promised power level.

A 96-item minority-conditioned panel is mainly suited to large effects. Repeated agents/rounds do not substitute for more items. Do not market a 1–2 point difference or absence of observed harm as a definitive improvement or noninferiority result.

Report both average net benefit and correct-majority destruction. A safety claim such as “does not hurt correct majorities” requires an explicit noninferiority margin and adequate intervals; a nonsignificant harm test does not establish safety.

### 15.4 Important nonclaims

- Identical isolated roots matching pass@k is not novel.
- Positive expected drift over a few rounds does not prove eventual convergence to truth.
- A low-rank state or an outlier is not inherently correct.
- High confidence may reflect copied certainty; calibration is not within-item selection ability.
- A useful predictor is not a causal neural mechanism.
- Native checkpoint comparisons do not isolate pretraining scale.
- Conditional correct-count transition tables are not a validated Markov law unless history-sufficiency is tested.
- More rounds are not independent samples; their cumulative coverage is not iid pass@k.
- No universal minority detector exists from uninformative observations: indistinguishable visible evidence can be compatible with different correct answers. The operational objective is better decision utility under stated information and compute constraints.

---

## 16. Compute ledger and 36-hour gate

### 16.1 Conservative logical-decision ceiling

This ledger intentionally ignores possible cache savings and reserves 64 common-suffix items at both checkpoints. Actual counts can be lower because challenge eligibility is limited and the 32B held-out suffix has 60 items.

| Module | 8B decisions | 32B decisions | Budget per decision |
|---|---:|---:|---|
| E1 sixteen-root banks | 6,144 | 1,536 | 1,024 thinking + 256 final |
| E2 natural protocols | 28,800 | 4,320 | 1,024 + 256 |
| E2 matched-root high-budget dynamics, 32 test items | 960 | 960 | 4,096 + 256 |
| E2 challenge, at most 96 items × two compositions × five protocols | 14,400 | 0 | 1,024 + 256 |
| E4 private counterfactuals on train/validation PEER states | 1,440 | 0 | 1,024 + 256 |
| E4 GATE and random protection on 240 test items | 7,200 | 0 | 1,024 + 256 |
| E4 selector diagnostic | 720 | 0 | 1,024 + 256 |
| E5 common private suffixes, capacity for 64 items | 1,280 | 1,280 | 1,024 + 256 |
| E3 controlled receiver assay | 1,280 | 1,280 | half 1,024 + 256; half 4,096 + 256 |
| **Total ceiling** | **62,224** | **9,376** | **71,600 total logical decisions** |

Conservative generated-token ceilings: approximately **84.56 million at 8B** and **16.92 million at 32B**, excluding optional modules and forced-boundary overhead. These are output ceilings, not wall-time predictions. Prefill, repeated prefill from split generation, serialization, model loading, grading, and replay must be included in timing estimates.

The included `budget_check.py` computes exact ceilings and checks measured throughput. It does not launch the experiment.

### 16.2 Suggested 10-GPU allocation

- Five single-GPU 8B replicas.
- Two 32B replicas, each tensor-parallel over two GPUs on a suitable same-node interconnect.
- One 8B replay/instrumentation GPU, dynamically reassigned while capture is idle.

This is a starting plan, not a guarantee of fit. Check actual GPU memory. 40-GB A100s may require lower batch/concurrency or a different 32B parallelism plan. Never assume a 32B model plus long-context KV cache fits comfortably in one 80-GB GPU solely from weight size. Prefer independent serving replicas over a cross-node tensor-parallel group when the model fits per-node [R18].

With 20 GPUs, start with ten 8B replicas, four two-GPU 32B replicas, and two GPUs for capture/overflow. Avoid mixing A100 and H100 within one tensor-parallel group. Keep fast checkpoints resident; do not reload on every Slurm task.

### 16.3 The launch gate

By hour 2, measure representative batches for each actual checkpoint/GPU/parallelism combination using realistic peer-context lengths and the split-budget path. Include both short/long reasoning and hub serialization. Also measure CPU grading throughput and activation replay speed.

Required measured fields:

- sustained aggregate generated tokens/second per serving replica, **including prefill and split-generation overhead**;
- average and high-quantile prompt length, generated length, cap hits, latency;
- GPU memory/KV cache behavior and error rates;
- replay sequences/second for representative long contexts;
- grader throughput and failure rates.

Let T_g be projected serving time from actual logical counts and measured effective throughput, stratified by prompt/call type where possible. Add model loading/setup, controller fitting/freeze, queue/DAG dependencies, capture, CPU grading backlog, and analysis. Use an explicit 20% uncertainty reserve.

Approve the selected tier only if projected completion including this reserve is within 36 hours. The nominal 10-GPU total is 360 GPU-hours, but a sum of GPU-hours alone does not prove the critical path fits. Controller inference waits for its training/validation freeze; hub rounds serialize; the longest resource lane matters.

As a rough feasibility threshold, the default output ceiling over 28 serving hours requires approximately 168 generated tokens/s per each of five 8B replicas and 84 tokens/s per each two-GPU 32B replica. These are **requirements to measure**, not asserted hardware performance. The executable helper adds a further throughput margin.

### 16.4 Downscope policy, in order

Freeze the chosen tier before seeing new test outcomes. Never stop an arm because its results look poor.

1. Disable all optional extensions and reduce the secondary LLM selector diagnostic.
2. Drop HUB5 if it jeopardizes the causal/control core; retain FRESH, PRIVATE, PEER, BALANCED. Reduce the high-budget native subset from 32 to 16 paired test items before removing that comparison entirely.
3. Reduce challenge cap from 96 to 48 and receiver assay from 64 to 32 tasks, preserving all factor levels.
4. Reduce 32B natural panel from 96 to 48 with frozen balanced split, and receiver seeds from two to one; report weaker replication explicitly.
5. Reduce natural cohort to 256 with a new frozen 64/32/160 split before dispatch; automatically recompute every dependent module count.

Do not drop PRIVATE, the random-protection control, train/test separation, all five root captures in the neural subset, or separate final-output budgets. If even the reduced plan fails, finish the paired behavioral experiment and postpone neural/scale claims rather than run an incomplete factorial with missing controls.

### 16.5 Schedule

| Hours after allocated start | Work |
|---|---|
| 0–2 | Pin artifacts, inspect hardware, audit legacy store, smoke-test graders/schema/boundaries, measure throughput, freeze tier. |
| 2–8 | Generate roots and training/validation protocol trajectories first; CPU graders stream continuously. Start 32B subset in parallel. |
| 6–12 | Generate private counterfactuals; fit predictors; validate only on validation tasks; freeze controller hashes. Test labels remain inaccessible to fitting code. |
| 8–24 | Complete main natural/challenge trajectories; run held-out GATE/random controls; execute receiver assay. Replay completed sequences concurrently. |
| 20–30 | Common suffixes, selector diagnostics, completion/coverage audits, any approved extensions. Stop admitting new optional work at hour 28. |
| 30–34 | Close grading backlog, immutable data join, grouped bootstrap, figures, failure analysis. |
| 34–36 | Validation/rebuild buffer and export. No new experiment family. |

This timetable assumes GPUs are already allocated and weights/datasets are available. Queue delays and installation/download failures consume the same deadline unless the user explicitly starts the clock after allocation.

---

## 17. Optional extensions, strictly gated

Prioritize these only when the core completion projection remains green:

1. **Additional challenge seeds:** replicate the already-core GATE/random and fixed protocols on the same constructed starts, with no retraining. Seeds reduce within-item Monte Carlo noise but do not add independent tasks.
2. **Confidence protection:** choose the privately revised agent by raw or calibrated confidence, using the same one-agent action budget. More informative than adding an unrelated architecture.
3. **Additional suffix depth:** one further common private round, preserving identical suffix policy.
4. **Team size:** N=3 and N=9 on 32 test items, PEER/BALANCED, three rounds; roots drawn from the same bank, but composition changes are explicitly modeled. Root pass@k curves alone do not answer post-interaction N effects.
5. **Heterogeneous teams:** on the common model subset, 4×8B+1×32B and 2×8B+3×32B, PEER/BALANCED. Predict root coverage from each model's native banks; do not compare raw cross-model hidden states.
6. **Native code bridge** from §5.5, contingent on a validated non-gold selector.
7. **Neural-augmented GATE** on the fixed 64 neural test items with a matched behavioral GATE control.
8. **Second model family** only if an already cached, operational checkpoint can run the identical contract. Do not spend the 36-hour core window debugging a new serving stack.
9. **Long single-agent baseline** on a subset at comparable measured cost, for any claim of superiority over optimized single-agent inference.

None is automatically part of the default 71,600-decision ledger. Adding a module requires a new manifest and refreshed runtime gate.

---

## 18. Data schema and deterministic identities

Minimum tables:

### `items.parquet`
`item_id, family, dataset_revision, source_id, split, group_id, prompt_hash, option_permutation, gold_location, grader_hash, subject, declared_difficulty_metadata`.

Gold/reference fields live in the evaluator namespace and must not be loaded by serving/controller workers.

### `episodes.parquet`
`episode_id, item_id, cohort, protocol, checkpoint_revision, root_candidate_ids, root_seed_ids, N, horizon, reasoning_allowance, final_allowance, packet_policy, memory_policy, leader_id, condition_manifest_hash`.

Challenge-only offline fields `constructed_C0` and eligibility counts live in an analysis-side table, not episode prompts/features.

### `decisions.parquet`
`decision_id, episode_id, parent_state_id, agent_id, round, phase, action, checkpoint, seed, request_hash, prompt_token_hash, completion_token_hash, root_report_id, prior_report_id, seen_message_ids, message_order, answer_raw, answer_class, evidence_text, confidence, parse_status, thinking_tokens, final_tokens, prompt_tokens, cap_hit, boundary_injected, logical_decision_id, physical_request_ids, latency, endpoint_id`.

### `labels.parquet` (analysis-only)
`decision_id, is_correct, evaluator_status, grader_version, hidden_test_or_reference_hash, grading_duration`.

### `features.parquet`
`decision_or_state_id, feature_time, last_allowed_round, feature_schema_version, behavior_features, neural_feature_file, model_revision, training_eligibility`.

### `forks.parquet`
`fork_id, shared_prefix_hash, focal_agent, action, branch_seed, common_suffix_policy, suffix_seed, parent/child_decision_ids`.

### `resources.parquet`
`timestamp, job_id, node, gpu_uuid, gpu_type, memory_total, allocated_gpu_seconds, utilization, memory_used, power_if_available, server_batch_id, physical_request_ids, prefill_tokens, decode_tokens, retries, errors`.

Do not claim exact per-request FLOPs from token count alone. Report logical request costs separately from physical cached execution. For batched GPU time attribution use a documented approximation and retain server-level totals.

**Cache identity:** include checkpoint revision, tokenizer/template hash, exact prompt token IDs, sampling configuration, all budget/stop settings, stochastic seed, and experiment request purpose where needed to prevent unintended reuse. Agent slot identity belongs in metadata unless intentionally part of the prompt. Two protocol names with identical requests should share only explicitly intended roots/no-op branches; assert actual graph differences for intended interventions.

---

## 19. Implementation contracts and pseudocode

Reuse existing runner names if available. The following are interface contracts, not claims that these commands already exist in the user's repository.

```
prepare_manifest(config) -> immutable item/condition manifests
validate_generation_contract(manifest, pilot_items) -> smoke_report
build_roots(manifest) -> candidate bank + exact tokens
grade_pending(evaluator_queue) -> analysis-only labels
build_episode(protocol, root_ids) -> request DAG
run_round(previous_state, protocol, seed_schedule) -> five reports + events
fork_private(previous_peer_state, focal_agent) -> counterfactual report
fit_current(train, validation) -> frozen correctness scorer
fit_exposure_value(train_forks, validation_forks) -> frozen p_private/p_peer
run_gate(frozen_controller, heldout_roots) -> closed-loop trajectories
run_common_suffix(parent_states, fixed_policy, suffix_seeds) -> suffix reports
replay_capture(token_store, capture_manifest) -> selected native states
analyze(frozen_predictions, analysis_labels) -> metrics + grouped intervals
```

Episode loop:

```
state = common_root_state
for t in 1..3:
    packet_plan = build_packets(state, protocol)    # prior round only
    snapshot = immutable_copy(state)
    reports = execute_updates(snapshot, packet_plan)
    assert no_request_read_same_round_peers_unless_HUB5
    append_archive(reports)
    emit_metrics_inputs_without_reading_gold()
    state = advance(snapshot, reports)
```

Counterfactual feature boundary:

```
x = features(shared_prefix)   # freeze before intervention
peer = stored_peer_child(shared_prefix, agent)
private = run_private_child(shared_prefix, agent, independent_seed)
# later, in a separate evaluator process:
y_peer, y_private = grade(peer), grade(private)
train_record = (x, y_peer, y_private, item_group)
```

No test feature may use `y_peer`, `y_private`, C0, true-majority membership, or a future agreement state. The test controller sees only x and frozen model parameters.

---

## 20. Mandatory automated tests

Before dispatch:

- k=1 oracle/selector equality; pass@k monotonicity for a fixed bank; exact finite-bank count probabilities sum to 1.
- Root request hashes identical across intended paired protocols; independent draw seeds genuinely distinct.
- N changes create the intended number of samples/calls, not only roster metadata.
- PEER at t reads only t-1, and HUB5 has the explicit 1+4 dependence.
- Current coverage may fall; cumulative archive coverage cannot fall without recorded pruning.
- Confidence is not accidentally reprinted by an answer parser as a forecast.
- Provisional/gold correctness and hidden tests never enter runtime features/messages.
- Split-budget final-answer capacity remains available after a thinking-cap hit.
- All five root states and all selected round states appear in the capture manifest; missingness is reported, not silently dropped.
- Model-specific layer indices and anchor spans are correct; generated-confidence tokens are excluded from pre-confidence features.
- Correct/incorrect agent transition counts reconcile to C_(t+1)-C_t, including invalidity transitions.
- Vote tie handling is unbiased/frozen and independent of correctness.
- Constant-rate calibration baselines use training/validation data only and the identical evaluated population.
- Within-item AUC does not become pooled AUC when an item has only one outcome class.
- All paired tables use the same joined item set, weighting, cohort, and final-validity convention.
- At hour 12/24, assert actual completion coverage by protocol/model rather than merely total request count.

---

## 21. Figures and deliverables

### Main figures

1. **Where correct answers go:** per-round C_t distribution and selected accuracy conditional on C0, with PEER/PRIVATE/BALANCED and selected additional protocols.
2. **Correction versus destruction:** one-step correction probability versus correct-answer destruction, stratified by task and checkpoint; mark private and communication controls.
3. **Prediction before intervention:** CURRENT versus EXPOSURE_VALUE performance, behavior-only versus neural increment where available, with grouped holdout evaluation.
4. **Closed-loop utility:** GATE versus PEER, random protection, BALANCED, and static selection at final accuracy and measured cost; minority benefit and majority harm shown separately.
5. **Controlled boundary:** endorsement-count sensitivity with/without diagnostic evidence at low/high reasoning and small/large checkpoint; common-suffix results alongside immediate utility.

Do not fabricate a Sankey “influence” edge from answer matches alone. A trajectory plot may show answer identity transitions; causal influence edges require actual intervention evidence.

### Required export

- Frozen configuration, manifests, model/dataset/code hashes, and actual hardware report.
- Complete event, candidate, label, feature, fork, and resource tables.
- Source-grounded legacy discrepancy report.
- Held-out predictions written before test labels are joined.
- Per-family/current/archived/selected metrics and task-grouped uncertainty.
- Missingness, parser failures, cap hits, and grading audit.
- `results_summary.md` separating confirmed, exploratory, and unexecuted claims.
- An explicit statement of which optional modules were not run.

---

## 22. Paper decision after the run

**Strongest positive outcome:** pre-update features predict the effect of peer exposure, a frozen controller improves later team accuracy beyond random protection and static selection, the gain survives correct-majority cases, and the controlled assay explains where the signal is useful. Neural features are an additional contribution only if they provide held-out incremental value.

**Useful alternative outcome:** current correctness remains detectable but peer exposure systematically weakens its decodability or usefulness for future decisions; a carefully controlled intervention restores it. This requires temporal comparisons and functional validation, not merely changing rank.

**Boundary/negative outcome:** minority survival improves but wrong-minority persistence increases enough to cancel it; the controller does not beat random protection; or metrics predict consensus but not intervention value. This identifies a limitation but should not be sold as a successful steering method.

**Insufficient outcome:** only reproducing conformity, only showing pass@k exceeds vote@k, only fitting a static minority classifier, or only reporting a correlational geometric change. Those overlap too strongly with the closest literature.

The 36-hour sprint should end with a decision about which of these happened, not a commitment to a preferred story.

---

## 23. References and official resources

Atlas source: user-supplied *A Scaffolding Atlas*, 7 September 2026, `ScalingAgents_Atlas-1.pdf`, especially §§3–4, 7.4–7.9, 8–10, and Appendix B. The attached audit documents are interpretive reviews of this source.

[R1] Choi, Zhu, and Li. *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?* 2025. https://arxiv.org/abs/2508.17536

[R2] He et al. *Minority Sentinel: When to Overturn Majority Voting in Multi-Agent LLM Debates.* 2026. https://arxiv.org/abs/2606.29270

[R3] Zhu et al. *Demystifying Multi-Agent Debate: The Role of Confidence and Diversity.* 2026, v3. https://arxiv.org/abs/2601.19921

[R4] Nguyen et al. *Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention.* 2026. https://arxiv.org/abs/2603.20640

[R5] Wang, Hong, and Bagci. *A decodability criterion predicts when hidden-state selection beats majority voting in large language models.* 2026. https://arxiv.org/abs/2608.17124

[R6] Hussain and Nielbo. *Conformity Mitigations in Large Language Models Lie on a Single Resistance-Receptivity Frontier.* 2026. https://arxiv.org/abs/2608.11247

[R7] Liu et al. *Shared Prefixes, Better Credit: Adaptive Routing for Multi-Agent Reasoning.* 2026. https://arxiv.org/abs/2608.02291

[R8] Ann, Liu, and Tan. *The Interaction Tax: When Communication Erases Diversity in Multi-Agent Teams.* 2026. https://arxiv.org/abs/2608.23541

[R9] Yoon et al. *ArcticSwarm: Deferring Early Consensus in Long-Horizon Multi-Agent Research.* 2026. https://arxiv.org/abs/2609.01870

[R10] Taubenfeld et al. *Confidence Improves Self-Consistency in LLMs.* Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1030/

[R11] Bertalanič and Fortuna. *The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate.* 2026. https://arxiv.org/abs/2605.00914

[R12] Chen et al. *Evaluating Large Language Models Trained on Code.* 2021. https://arxiv.org/abs/2107.03374

[R13] MATH-500 official dataset card. https://huggingface.co/datasets/HuggingFaceH4/MATH-500

[R14] Wang et al. *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* 2024. https://arxiv.org/abs/2406.01574 ; dataset https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

[R15] BigCodeBench official repository. https://github.com/bigcode-project/bigcodebench

[R16] Qwen3-8B official model card. https://huggingface.co/Qwen/Qwen3-8B

[R17] Qwen3-32B official model card. https://huggingface.co/Qwen/Qwen3-32B

[R18] vLLM official parallelism/scaling documentation. https://docs.vllm.ai/en/latest/serving/parallelism_scaling/

[R19] Jiang, Yi, and Zhu. *Semantic Voting: Execution-Grounded Consensus for LLM Code Generation.* 2026. https://arxiv.org/abs/2605.08680

[R20] Hao et al. *Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate.* 2026. https://arxiv.org/abs/2606.00820

[R21] Zhang et al. *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification.* 2025. https://arxiv.org/abs/2504.05419

[R22] Qwen Team. *Qwen3 Technical Report.* 2025. https://arxiv.org/abs/2505.09388
