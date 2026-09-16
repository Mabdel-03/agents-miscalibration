expermt_agent_specs/taskCondAgentDynV1/task_conditioned_agent_dynamics_experiment_spec_v1_0.md# Task-Conditioned Multi-Agent Dynamics: Next-Experiment Specification

**Project:** PoggioLab - Agents Scaling Project  
**Study ID:** `task_conditioned_dynamics_v1`  
**Specification version:** 1.0  
**Date:** 2026-09-15  
**Status:** Proposed experiments and implementation contract. No new experiments are represented as completed.  
**Intended readers:** Research leads, coding agents, experiment operators, and analysts.

## High-level objective and connection to existing experiments

The next experiments ask **why communication helps some agent teams and harms others, especially when the correct answer starts in the minority**. We will separate three stages: the alternatives agents initially generate, how communication changes those alternatives over successive rounds, and whether the final selector returns a correct answer. The objective is not to establish that debate is universally beneficial or harmful. It is to identify measurable task, model, and communication conditions that predict whether correctness survives, spreads, wins selection, or disappears, and then intervene on those conditions to test the explanation.

This study builds directly on three existing reports. **Run0** suggests task- and capability-dependent benefits from sharing rationales, alongside distinct changes in agreement-based calibration. **The Scaffolding Atlas** shows why difficulty, answer grouping, output validity, and realized compute must be separated before interpreting benchmark differences. **Repaired ARD36** supplies the shared-root comparison in which peer communication increases mean agent correctness while reducing active truth-coverage relative to private revision, but leaves the constructed correct-minority challenge largely untested. The next run therefore prioritizes **matched initial disagreements, evidence-versus-endorsement receiver assays, and directed communication interventions**, rather than another full factorial sweep. Confidence and neural geometry are candidate measurements; a predictive controller and a recovery-cost study are optional extensions, not prerequisites. [RUN0, §§6.4, 7.3, 13.7; ATLAS, §§3, 7–10; ARD, §§2–3, 9–12]

**Desired paper-level advance:** move from “communication behaves differently across benchmarks” to **“the difference is attributable to specified initial-state and receiver/communication properties, and changing those properties changes the dynamics in the predicted direction.”** All directional predictions below are hypotheses to test, not required outcomes.

---

## 1. Reading guide, scope, and authority

**Navigation:** scientific framing and shared contracts (§§2–6); exact core experiments (§§7–10); transfer and optional prediction (§§11–12); measurements, statistics, and compute (§§13–16); implementation and run gates (§§17–19); figures, extensions, completion, and sources (§§20–23).

### 1.1 Source versus proposal

Statements attributed to **[ARD]**, **[ATLAS]**, and **[RUN0]** describe the supplied reports, not an independent audit of their raw runs. Section 23 identifies the files and page references. **[LEGACY]** identifies an earlier project specification used only for interface continuity. **[L1–L6]** and **[D1–D8]** identify independently checked publications or official implementation/data resources. Every new sample allocation, threshold, prompt, model contrast, and experimental choice in this document is a **proposed design decision**.

The PDFs do not establish the current filesystem layout, accessible GPUs, repository revision, model cache, or availability of all underlying logs. Missing implementation inputs are explicit preflight blockers; they must not be inferred from a report filename. The commands below specify an interface to implement or map onto existing code, not commands verified to exist today.

### 1.2 Priority tiers

| Tier | Required work | Scientific purpose |
|---|---|---|
| **Core** | E0 reanalysis/audit; E1 natural and constructed disagreements; E2 receiver mechanism assays; E3a directed channels; read-only measurements | Explain and causally characterize conditional communication dynamics. |
| **Transfer** | E3b restricted topology/size/budget blocks; E3c model replay and native-initialization comparisons | Determine the scope of the explanation without a full Cartesian sweep. |
| **Optional** | E4 action prediction/control; X1 recovery; X2 answer-cardinality manipulation; additional code dynamics | Demonstrate practical control or extend task-structure analysis after the core is sound. |

Complete the core before expanding breadth. An informative null or reversal remains reportable. Do not gate the core on discovering a favorable sign.

### 1.3 Changes from earlier planning, made explicit

1. **Round convention:** the new main experiment uses root `r=0` plus **four revision rounds**, `r=1..4`. ARD36 used states `r=0..3`, meaning three revisions. The new `r=3` checkpoint is the horizon bridge; `r=4` is the new primary endpoint.
2. **Public state:** the new primary protocol transmits a compact public rationale and claims, not the entire native thinking trace. Confidence is logged but hidden from subsequent agents in the primary communication arms. This is a deliberate controlled setting, not a byte-for-byte ARD36 or Run0 replication.
3. **Receiver checking:** “verification” means an actually executable local-step check, not merely a message claiming to have been checked. ARD36's checked framing was the latter. [ARD, §9, p. 9]
4. **Sample targets are provisional:** 300 constructed-cohort items and 200 natural-cohort items per task are planning targets, subject to power and actual eligibility. They are not guaranteed feasible on every fixed benchmark.
5. **Earlier mandatory neural-intervention and benefit-then-harm gates are not inherited.** This is a task-conditioned characterization study, not the older continuation-value study. Retain useful root, lineage, visibility, and audit conventions without rewriting or relabeling older runs. [LEGACY]
6. **Synthetic depth control:** the receiver task uses 24 displayed operations at every depth, with a controlled dependency chain of length 4, 12, or 24. This refines the prior sketch so that depth is not simply the number of displayed rows.

### 1.4 Scope of causal claims

Randomized protocol, support-cue, checker-access, and directed-edge contrasts are experimental effects within the declared task population. Benchmark identity, model parameter count, and measured difficulty are not randomized causes in the same sense. A cross-benchmark interaction establishes conditional heterogeneity, not that a benchmark name causes it. Matching observed initial summaries does not make all arguments, latent difficulty, or priors identical.

An exact accounting identity, a correlational predictor, a randomized behavioral effect, and a neural mechanism must remain distinct in the paper.

---

## 2. Existing evidence and the gaps this run addresses

| Existing evidence | What the report supports | Unresolved question addressed here |
|---|---|---|
| ARD36 repaired coverage contrast | On the 240-item shared-root panel, terminal PEER coverage is 77.92% versus PRIVATE 82.50%; the reported primary contrast is −4.58 percentage points, Holm-adjusted `p=0.024`. PEER mean correct-count rises from 3.508 to 3.746 across its rounds. [ARD, §§2–3, pp. 2–4] | Does this persist within task and comparable initial disagreement, and does the same structure also protect incorrect dissent? |
| ARD36 minority challenge | The constructed `C0 ∈ {1,4}` challenge was not run because only 27 items met the repaired eligibility rule. [ARD, §12, p. 11] | Obtain enough independently eligible questions to estimate minority survival, propagation, and selection directly. |
| ARD36 confidence and geometry | Repaired within-item confidence AUC is 0.5723; geometry is explicitly descriptive, without a fitted prospective correctness probe. [ARD, §2.2, p. 3; §6, p. 8] | Do prospective measurements add predictive value beyond answer counts, text, and independently estimated difficulty? |
| ARD36 receiver assay | Reasoning allowance changes performance substantially; the “checked” manipulation asserted verification rather than providing a checker. [ARD, §9, pp. 9–10] | Distinguish endorsement, receiver budget, and actual ability to check evidence. |
| Atlas difficulty-conditioned effects | Topology spread is 0.156 in the contested band versus 0.040 pooled; the report narrows the claim because effects also occur on easy items. [ATLAS, §7.1, Fig. 1/Table 4, p. 5] | Separate initial answer composition from conditional communication dynamics. |
| Atlas selection differences | Reported failure decompositions differ between exam and code pools; code AST grouping splits behaviorally equivalent programs. [ATLAS, §7.8, p. 9] | Avoid attributing a selector's grouping defect to communication or minority suppression. |
| Atlas geometry | Lower dimensionality coexists with essentially unchanged mean distance in its answer-state comparison, unlike ARD36's contraction setting. [ATLAS, §8.1, pp. 12–13] | Measure scale-carrying and scale-free geometry separately at matched anchors and complete roots. |
| Run0 rationale sharing | Matched rationale-sharing gains differ by benchmark and capability/reasoning regime. [RUN0, §6.4, p. 41; §7.3, pp. 48–49] | Does receiver competence explain the variation after fixing the actual input evidence? |
| Run0 measurement boundaries | The option probe precedes generation; some model/budget comparisons have context and sampling confounds; the dataset is supplementary mixed-protocol evidence. [RUN0, §§3.4–3.6, pp. 16–19; §13.7, p. 75] | Use post-reasoning measurements, canonical answer identities, and a homogeneous new protocol. |

Do not repeat as established facts that consensus is literally absorbing, that residual contraction causes error, that confidence is universally useless, or that oracle pass@k is a deployable compute-efficient selector. Those conclusions exceed the supporting comparisons. This is an interpretation boundary for the new study, not an alteration of the reports.

---

## 3. Target results and falsifiable predictions

### R1. Initial composition and conditional dynamics explain different portions of task heterogeneity

**Question:** after standardizing initial correct-count, wrong-answer concentration, and independent difficulty estimates, do task differences in PEER versus PRIVATE remain?

**Supporting patterns:** substantial attenuation after standardization indicates a composition contribution; reproducible residual interactions indicate that these initial summaries are insufficient and that comparable disagreements evolve differently.

**Disconfirming pattern for a task-specific-dynamics claim:** the residual effects are practically equivalent under a prespecified equivalence margin, with adequate precision. Nonsignificance alone does not establish equivalence.

### R2. Evidence evaluation and endorsement are distinguishable channels

**Question:** with solution text fixed, do support counts change revisions, and do actual checking or more reasoning change that susceptibility?

**Prediction:** usable verification and/or additional receiver reasoning may reduce the correct-majority versus correct-minority performance gap. They may instead improve both equally, worsen reliance on a plausible wrong trace, or have no measurable effect. These outcomes distinguish mechanisms.

**Disconfirming pattern for the proposed interaction:** verification or budget raises accuracy but does not reduce endorsement sensitivity. That supports useful computation, not the specific evidence-versus-endorsement explanation.

### R3. Outgoing minority influence and incoming minority vulnerability are separable

**Question:** can a focal agent transmit its argument without being revised by the bloc that opposes it, and what happens when the focal answer is wrong?

**Prediction:** blocking incoming peer messages may improve focal retention; maintaining outgoing messages may affect others' adoption. The same operation may protect a wrong minority, so a preservation effect is not automatically a better system.

**Disconfirming patterns:** all effects are explained by generic context load; outbound exposure does not affect others; or correct-minority gains are offset by incorrect-minority harm in the natural population.

### R4. Dynamics predict worthwhile next operations, not merely failure

**Question:** do observable state features predict the incremental value of PEER, PRIVATE, FRESH, or STOP on held-out states?

**Supporting pattern:** a frozen policy improves delivered accuracy at a common budget, or attains the same accuracy with lower measured cost, against strong fixed policies.

**Disconfirming pattern:** a predictor forecasts error but cannot choose a better action. This is a useful forecasting result, not successful control.

**Novelty boundary:** repeated sampling, conditional debate effectiveness, stochastic debate models, communication-induced coverage loss, topology-dependent propagation, and confidence-weighted selection have precedents. The target here is their experimentally separated, task-conditioned explanation and held-out validation, not a claim that these individual phenomena are new. [L1–L6]

---

## 4. Formal objects, outcomes, and state conventions

### 4.1 Units and notation

- `source_item_id`: canonical original question, shared by all renderings and experimental variants.
- `item_variant_id`: option permutation or synthetic variant of a source item.
- `root_team_id`: ordered set of independent root-generation records assigned to agent slots.
- `episode_id`: one root team, protocol, model/budget condition, and continuation seed schedule.
- `decision_id`: one agent update; uniquely nested inside an episode.
- `r=0`: root state; `r=1..R`: successive completed synchronous updates.
- `N`: agent slots; fixed at 5 in the main experiment.
- `S_r`: the complete externally stored state at round r, including public drafts, actual communication edges, and lineage.
- `c_ir ∈ {C,W,U}`: semantically correct, semantically incorrect, or unknown/ungradeable.
- `C_r`, `W_r`, `U_r`: corresponding counts, summing to N.
- `a_ir`: canonical answer class, which can be defined even when semantic grading is unknown.

No model sees `c_ir`, correct-counts, gold answers, evaluator outputs, or bank correctness labels, except for the explicitly authorized public local-step checker in E2. A grading-aware offline team constructor is not a deployable policy.

### 4.2 Keep four outcomes separate

On fully gradeable teams:

\[
M_r=\mathbb{E}[C_r/N],\qquad
Q_r=\Pr(C_r>0),\qquad
H_r=\Pr(\exists u\le r:C_u>0),\qquad
A_r=\Pr(\hat y_r\text{ is correct}).
\]

`M` is individual accuracy, `Q` active truth-coverage, `H` historical truth-coverage, and `A` delivered system accuracy under a named selector and candidate pool.

For a selector restricted to current candidates, and complete grading:

\[
1-A_R=(1-H_R)+(H_R-Q_R)+(Q_R-A_R).
\]

These are respectively never-generated, generated-but-no-longer-active, and active-but-unselected failures. This is an arithmetic decomposition, not causal mediation. It is invalid for a synthesizer that can generate an unenumerated new answer unless that generation is explicitly added to the candidate process.

### 4.3 Transition outcomes

Define average wrong-to-correct and correct-to-wrong transition rates conditional on the recorded state:

\[
\alpha_r(S)=\Pr(W\to C\mid S),\quad
\beta_r(S)=\Pr(C\to W\mid S).
\]

On fully gradeable consecutive states with `C_r=k`:

\[
\mathbb{E}[C_{r+1}-C_r\mid S_r]
=(N-k)\alpha_r(S_r)-k\beta_r(S_r).
\]

Measure last-correct-answer loss separately:

\[
h_r(S)=\Pr(C_{r+1}=0\mid C_r>0,S_r).
\]

Also measure discovery/recovery `Pr(C_{r+1}>0 | C_r=0)`, continuous preservation `Pr(C_u>0 for every u≤R | C_0>0)`, and terminal preservation `Pr(C_R>0 | C_0>0)`. Terminal preservation allows intervening extinction and rediscovery; continuous preservation does not.

The individual rates alone do **not** determine team extinction because updates can be dependent. Fit or measure the joint/team outcome rather than assuming independent Bernoulli updates.

### 4.4 Minority definitions

Record both:

- **Arithmetic correct minority:** `0 < C_r < N/2`.
- **Outnumbered correct answer:** the correct answer class has strictly fewer members than the largest individual wrong-answer class.

For the main five-agent challenge, `C0=1` must face a wrong class of size at least 2. The mirrored `C0=4` team contains one specific wrong answer `w*` opposed by four correct members. All-singleton teams and tied pluralities are separate strata, not silently classified as wrong majorities.

Where multiple correct answer forms exist, use the task's gold-independent canonical equivalence procedure for vote grouping. Do not merge candidates merely because the evaluator labels both correct. For code, correctness and behavioral equivalence are especially distinct.

### 4.5 Additional registered measurements

| Measurement | Exact definition or reporting requirement |
|---|---|
| Focal retention | Focal agent still has its initial canonical class; report its semantic correctness separately. |
| Correct propagation | Number/fraction of initially wrong nonfocal agents that become correct. A causal transmission interpretation requires the outbound-exposure contrast. |
| Wrong-minority propagation | Number/fraction of initially correct nonfocal agents adopting the focal root's wrong class `w*`; also report all `C→W` transitions. |
| Winning | Correct plurality under the frozen tie rule; separately strict majority and unanimous correctness. |
| Wrong unanimity | All N valid answers share one canonical wrong class. All wrong with different answers is not unanimity. |
| Answer concentration | Maximum class share; normalized class entropy; wrong-class maximum and collision probability for offline analysis. |
| Switching | Canonical answer changes between two valid answers. Validity changes are a separate event. |
| Copying | Exact public-rationale reuse and n-gram overlap, with same-item/different-episode placebo; not a standalone correctness mechanism. |
| Recovery | A previously all-wrong fully gradeable team later has a correct active answer; report selector success separately. |
| Cost | Generated tokens by phase, prompt tokens, calls, model compute estimate, attributed GPU seconds, latency, verifier cost, and unused budget. |

### 4.6 Missingness and validity are not semantic error

Maintain separate fields for `parse_status`, `grading_status`, `semantic_correct`, `delivery_success`, and `infrastructure_status`.

- A valid, deterministically incorrect answer is `W`.
- No extractable answer, an unresolved evaluator result, or unresolved infrastructure failure is `U` for semantic transition analysis, with a typed reason.
- A refusal or malformed output counts as zero **operational delivery success**, without manufacturing a `C→W` semantic transition.
- No extra model calls may repair malformed answers in the primary arms. Infrastructure retries are allowed only under the fixed retry policy.

For each team, report coverage bounds:

\[
Q_r^{lo}=1[C_r>0],\qquad Q_r^{hi}=1[C_r+U_r>0].
\]

For individual accuracy, use `[C_r/N,(C_r+U_r)/N]`. For a treatment difference, conservative bounds are `[mean(lo_PEER)-mean(hi_PRIVATE), mean(hi_PEER)-mean(lo_PRIVATE)]`. The primary all-item operational estimate and these semantic bounds must accompany any complete-case estimate. A fixed complete-case subset is a sensitivity population, not a substitute for the intended population.

For the full three-state transition table, the exact identity is:

\[
C_{r+1}-C_r=N_{WC}+N_{UC}-N_{CW}-N_{CU}.
\]

Test it directly. Use the simpler identity only when neither endpoint includes unknowns. Do not treat uncertainty intervals from bootstrap as substitutes for missing-outcome bounds.

---

## 5. Dataset, split, model, and serving contracts

### 5.1 Primary task panel

| Panel | Source and use | Contract |
|---|---|---|
| **MATH** | MATH-500 for a continuity panel; disjoint broader MATH test items for additional independent questions when accessible. [D1–D2] | Preserve original problem, subject/level metadata, source identity, and gold answer privately. Name the expanded panel separately from MATH-500. |
| **MMLU-Pro** | Primary multiple-choice task family; preserve subject metadata. [D3] | Preserve canonical option IDs and original option text; render one shared permutation across a paired team block. |
| **TruthfulQA legacy MC1** | Continuity with Run0; use a pinned legacy MC1 option set. [RUN0, §3.6; D4] | Explicitly name the new experiment a **generated-choice adaptation of the legacy MC1 question set**, not the original isolated-option log-likelihood evaluation. |
| **BigCodeBench** | E0 selector repair and optional later transfer. [D5] | Same frozen tasks/candidates as Atlas where recoverable; pinned sandbox and evaluator. No hidden test in generation or selection. |
| **Synthetic mod-97** | E2 controlled mechanism experiment. | Fully specified generator, independent evaluator, and local checker below; new data, not an existing benchmark score. |

**TruthfulQA version warning.** The official repository now also provides a newer binary multiple-choice version. Do not silently substitute it for the legacy MC1 option set, because that changes the answer space central to this study. Archive the exact questions/options actually used. Any newer binary panel is a separately named extension. [D4]

**Task labels are not mechanisms.** MMLU-Pro is not pure fact recall; MATH is not the only task involving reasoning. Do not infer verifiability, dependence depth, or knowledge requirements solely from a benchmark's name. [D3]

### 5.2 Item inventory and split policy

Create an inventory of every source question already used in Run0, Atlas, ARD36, earlier development, and prior project experiments for which manifests are available. Identify duplicates by source ID plus normalized text hash, with an audited near-duplicate check. If a prior manifest is unavailable, mark overlap status `unknown`; do not claim untouched items.

Assign `split_role` before new condition outcomes are inspected:

| Role | Default allocation | Permitted use |
|---|---:|---|
| Technical/scientific pilot | 60 original questions per primary family; 30 synthetic source problems | Prompt, evaluator, eligibility, variance, throughput, and adapter diagnostics; excluded from confirmation. |
| Natural confirmation | Target 200 unselected questions per primary family | Estimate effects on a declared source distribution; selection precedes root outcomes. |
| Constructed confirmation | Target 300 eligible original questions per family for each primary composition | Conditional minority experiments; may overlap natural confirmation, but not pilot/development. |
| Transfer | Prespecified subsets or fresh held-out items, as stated for each block | Test transfer of the protocol contrast or a frozen predictor. Same-item model replication is not new-item replication. |

The natural and constructed cohorts may share confirmation source questions. They remain distinct estimands; do not add their counts as independent questions or average their different root-selection distributions indiscriminately. All variants and seeds for a source item share its analysis fold and cluster ID.

Sample the natural cohort uniformly from the declared untouched source frame. If subject-stratified allocation departs from source proportions, record sampling probabilities and use prespecified population weights. Equal-task weighting is for the cross-task summary, not a claim about real-world task frequencies.

For constructed cohorts, scan a frozen random ordering of the eligible source frame. Stop at the preregistered quota or exhausted frame, not at a desired treatment result. Publish the recruitment flow and eligibility rates. If a finite benchmark cannot supply 300 eligible items, retain the actual sample, reduce the scope/precision claim, or amend the panel **before continuation outcomes are opened**. Do not fabricate independent questions by paraphrasing existing ones.

### 5.3 Separate sampling banks

For each source item and donor model/budget condition:

1. **Difficulty bank:** 16 independent draws, fixed count, never used as challenge donors or root answers.
2. **Donor bank:** 32 independent draws for constructed teams, extendable to 64 under a frozen eligibility-only rule.
3. **Natural root bank:** 5 independent draws on natural-cohort questions, independent of both banks above.
4. **Fresh continuations:** independent new draws under the root prompt; never reused from difficulty or donor banks.

Use disjoint seed namespaces. A bank has a typed `cell_kind`, not an implicit PRIVATE protocol default. A bank record can never overwrite an episode root. [ARD, §2]

Estimate difficulty using the 16-draw bank only: report raw correct fraction among resolved draws, resolved draw count, operational success, and uncertainty/bounds. A smoothed exploratory estimate `(c+0.5)/(g+1)` may be used for modeling, where g is the resolved draw count. Report sensitivity to this finite-bank measurement error. “No success in 16 draws” does not mean the model cannot solve the item.

Do not estimate unconditional pass@k from an adaptively extended or correctness-selected donor pool. Use a fixed unselected sampling population for that purpose.

### 5.4 Model cells

| Cell | Intended role | Interpretation |
|---|---|---|
| `Qwen/Qwen3-8B`, native thinking | Main E1/E2/E3 behavioral model | Continuity with ARD36, with the new explicit protocol. |
| `Qwen/Qwen3-32B`, native thinking | Receiver and restricted end-to-end scale comparison | Same-family scale contrast; not isolated pretraining-compute causality. |
| `meta-llama/Llama-3.1-8B-Instruct` | Restricted family transfer | Requires authorized access. Compare within-model protocol effects; do not equate its output tokens to Qwen native thinking tokens. |

These fixed models are proposed for continuity and controlled contrasts, not as claims about the latest or strongest available model. Official identities/mode documentation were checked for the specification. [D6–D8]

Pin full model/tokenizer revisions, chat template bytes, model config, weight checksums, dtype, quantization, backend, CUDA environment, and parser. Resolve mutable upstream references once into `model_lock.json`; confirmation may not use `main`, `latest`, or unrecorded defaults.

### 5.5 Main sampling and token contract

For all native-thinking Qwen root and revision calls in the core:

```yaml
sampling:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
reasoning:
  enabled: true
  token_cap: 2048
  enforcement: engine_token_boundary
output:
  final_phase_token_cap: 1024
  max_total_new_tokens: 3072
context:
  served_window_tokens: 32768
  safety_reserve_tokens: 256
  accumulated_transcript_visible: false
```

The requested final allowance must remain available even when reasoning reaches its cap. Prevent unused thinking tokens from becoming unlimited public-output space: impose the final-phase cap separately. Freeze the forced reasoning-end sequence; do not add treatment-specific text at the cap. Record whether the model naturally stopped thinking or was forced to stop.

The sampling values match the official Qwen thinking guidance, but the short reasoning/output caps are **experimental choices**, not maximum-capability evaluation settings. [D6]

Use the existing verified vLLM adapter where possible. Current official vLLM documentation describes `thinking_token_budget` and warns of reasoning-output field-name changes; validate the exact pinned version rather than assume a particular response field. Parse native reasoning and the public JSON `reasoning` field into different storage fields. [D8]

**Hard preflight:** worst-case prompt plus reasoning, final phase, and reserve must fit every compared model's served window. Use enough hardware/parallelism for 32B, or reduce the entire paired design's visible packet envelope before freeze. Never quietly give only 32B a smaller window. Do not automatically enable context-extension scaling.

### 5.6 Family-transfer adapter

Llama family transfer uses its ordinary instruction-following generation and the same public output contract, with a total generated-token cap of 3,072 and the same sampling settings where supported. It does not receive a fictitious Qwen native thinking flag.

Run a Qwen non-thinking companion at the same **total output envelope** for the family block, with identical public instructions. Report this separately from native-thinking Qwen size contrasts. The effect being transferred is PEER versus PRIVATE within a declared model/adapter, not equality of reasoning computation between families.

No inaccessible model may be silently replaced. A replacement requires a named amendment before that block's outcomes are examined.

---

## 6. Public agent state, prompts, and selection

### 6.1 Output schema

After any native thinking block, request one JSON object:

```json
{
  "reasoning": "A concise public rationale for the answer.",
  "shareable_claims": ["An explicit premise, calculation, or check."],
  "final_answer": "One option label or one mathematical answer.",
  "confidence": 0.75
}
```

Request the keys in the displayed order: public reasoning, claims, final answer, then confidence. Record field-order violations without discarding an otherwise gradeable answer. A capture labeled `ANSWER_END` is eligible for pre-confidence analysis only when confidence has not already been emitted. Freeze whether final-phase constrained JSON decoding is used; the proposed default is prompt-only formatting, with no unrecorded decoder fallback.

`reasoning` is a **public rationale**, not a claim to faithfully expose every internal computation. Native thinking tokens are stored separately as `native_reasoning_text`. The public `confidence` is a probability in `[0,1]`; malformed or absent confidence is null, never 0.5 by default.

Primary QA public-rationale limit: 256 tokens; at most four claims of 64 tokens each. Enforce a deterministic **transmitted** packet cap of 512 tokens per agent using token-aware field truncation. Preserve the final answer first, then the public rationale, then claims in order. Mark exactly which fields were truncated, retaining original bytes and hashes externally. Do not silently invoke a summarizer.

Use the same frozen rendering for an agent's own previous public draft and its peers, except for source-slot labels. Hide confidence in both. The original model response remains available for grading, audit, and optional measurement, even if its transmitted rationale is shorter.

### 6.2 Root template

```text
SYSTEM
Solve the user's question as accurately as you can. After reasoning, output one
JSON object with fields reasoning, shareable_claims, final_answer, confidence.
The reasoning field is a concise public rationale. Confidence is your estimated
probability that the final answer is correct. Use the stated answer format.

USER
{question_with_options_if_any}

Return exactly one answer. {task_specific_answer_instruction}
```

No role diversity, expertise persona, social-count hint, or agent-slot identifier appears in the root prompt. Independent roots differ through their sampling stream, not covert prompt treatments.

### 6.3 Common revision template

```text
SYSTEM
Solve the question as accurately as you can. Reconsider the working draft using
the information provided. Keep it or change it according to your assessment of
its correctness. Output the same JSON fields as specified for the initial answer.

USER
QUESTION
{question_with_options_if_any}

WORKING DRAFT
{own_previous_public_packet}

OTHER DRAFTS
{visible_peer_packets_or_literal_NONE}

Give your current best answer and a concise public rationale.
```

PRIMARY PRIVATE and PEER use this exact template; only OTHER DRAFTS changes. FRESH uses the root template and sees no own or peer history. Read-only external archives are never inadvertently inserted in a model request.

Do not give generic advice to resist majorities only in one arm. Such prompt advice is another intervention and would obscure the channel manipulation.

### 6.4 Memory and scheduling

The active state is the immediately preceding public draft, not an ever-growing transcript. Rebuild each round's model input from the declared state. Do not rely on backend conversational memory or carry undisclosed KV state between rounds.

Updates are synchronous: every agent at `r+1` sees only packets from `r`. Freeze the entire input set before dispatch. A faster agent's new output cannot enter another agent's same-round input.

Randomize peer display order by a seed independent of correctness. Share the order schedule across paired arms for packets they have in common. Slot IDs are randomly permuted and hidden from evaluators; visible IDs never encode correctness, source model size, or treatment.

### 6.5 Selectors and archives

**Primary selector: current-round plurality**, using a gold-independent canonicalizer and all valid current answers. Each valid agent contributes exactly one vote. If all outputs are invalid, return no answer. Break tied classes with a stable hash of `(source_item_id, canonical_class, selection_seed)`, not confidence, agent order, or correctness. The class key must not include the protocol ID.

Report unweighted vote as the primary. On the same frozen candidate sets, run these secondary selectors:

| Selector | Candidate pool | Requirements |
|---|---|---|
| Random valid candidate | Current round | Common seeded tie/random stream; no correctness labels. |
| Confidence-weighted vote | Current round | Confidence mapping fitted on development only; missing values use an explicitly specified weight, with complete-confidence sensitivity. |
| Pointwise audit | Current round or full archive, separately | Fixed verifier model/prompt; returns a score per candidate, then selects an existing candidate ID; no new synthesis. |
| Oracle | Same named pool | Evaluation-only ceiling, never a deployed policy. |

The pointwise audit default is Qwen3-32B, thinking enabled with 1,024 reasoning and 256 public-output tokens, and one scoring draw per unique visible candidate packet. Record that it is a same-family verifier. Candidate confidence, endorsement counts, source slot, and correctness are hidden. This is a baseline, not a trusted judge.

A verifier score schema is `{"p_correct": float}`. A failed score uses the preregistered fallback current-round vote; a failed individual score is not a semantic error label. Freeze the verifier before scoring confirmation candidates. Budget verification explicitly.

Archive-aware selection is optional but uses the same verifier and scoring policy on the whole permitted pool. Deduplicate only byte-identical public candidate packets for compute reuse; log every original occurrence. Preserve old differing rationales. Never use a correctness-aware deduplication rule.

Maintain archive scope labels `terminal_only`, `stage_and_terminal`, and `full_archive`. The new primary selection is `terminal_only`; it does not inherit the older specification's default archive. Exact aliases to legacy action names are recorded in a compatibility map, with different protocol hashes where content differs.

---
## 7. E0: Existing-run reanalysis and measurement audit

**Priority:** first. **New solver generations:** none by default. Generated code-test inputs and independent grading audits, if needed, are separately metered.

### 7.1 Inputs and migration

Read ARD36 repaired `tables_v2` and underlying decision blobs; Atlas sealed item/candidate tables and activation blobs; and Run0's deduplicated source tables. Locate them through explicit repository/run-root configuration, not guessed absolute paths. Preserve each study's native semantics in a `legacy_*` view and create a separately versioned harmonized view.

Required outputs before interpreting trends:

- `legacy_inventory.parquet`: study, original path, source hash, schema, completeness, available identifiers, and missing artifacts.
- `legacy_protocol_map.json`: actual root, memory, sampling, truncation, output, selection, and grading conventions by study/cell.
- `legacy_claim_reproduction.csv`: published estimate, recomputed estimate, population/denominator, and pass/fail/unavailable status.
- `overlap_inventory.parquet`: source-item overlap among legacy studies and new candidate frames.

A PDF alone is sufficient to document a reported result, not to re-estimate it. If raw inputs are unavailable, mark E0 subanalyses `BLOCKED_MISSING_SOURCE`, retain their source-reported status, and do not invent reconstructed observations.

### 7.2 Required source-specific checks

**ARD36:** reproduce the 240-item PEER/PRIVATE coverage comparison, the 12-versus-1 discordant endpoint cases where the same exact pairing applies, the repaired confidence AUC, and the distinct 171-item sampling comparison. Audit every discordant item manually or through an independent blinded grading route. Do not merge denominators. Do not classify missing/unfitted selectors as failed algorithms. [ARD, §§2, 4, 10–12]

**Atlas:** recompute realized rather than nominal configuration differences; identify aliased cells; reproduce domain/difficulty plots and current/archived selection gaps; recompute answer agreement from item records rather than the defective geometry-table column. Audit HLE's self-judged labels before using absolute accuracy as ground truth. [ATLAS, §§3–4, 7; Appendix B]

**Run0:** isolate homogeneous serving and prompt contracts; recover canonical option contents through each recorded shuffle where possible; stratify by budget enforcement, context cap, and execution regime. A missing option-permutation record makes a cross-cell collision comparison unavailable. Do not infer it from letters. Preserve the designation “supplementary mixed-protocol evidence.” [RUN0, §§3–5, 13]

### 7.3 Task-resolved contrast table

For each estimable study/task/model/budget/protocol stratum, produce root distribution, round outcomes, transitions, validity, calibration, selected-answer accuracy, and realized resource use. Report PEER versus PRIVATE only where both exist under a comparable contract. A Run0 debate-versus-one-pass contrast is not relabeled PEER-versus-private-revision.

Estimate natural and standardized effects using **only pre-communication variables**. Primary standardization variables are initial correct-count, largest wrong-class count, independent-bank difficulty bin, and task-format metadata where comparable.

For a common supported set of strata z:

\[
\Delta_d^{std}=\sum_z w(z)\{E[Y\mid PEER,z,d]-E[Y\mid PRIVATE,z,d]\}.
\]

Freeze w from the pre-outcome recruitment frame. Use equal weights over supported task distributions for a cross-task reference; report overlap retained and effective sample size. Empty or sparse strata are not filled by unacknowledged extrapolation. Provide a prespecified regression-standardization sensitivity when coarse stratification loses too much support.

Do not call attenuation a uniquely identified mediation proportion. Measured difficulty is noisy, arguments remain heterogeneous, and standardization depends on the chosen reference distribution. Do not adjust the total effect for post-communication agreement, confidence, or realized reasoning tokens.

### 7.4 Code-selection repair

Freeze the same BigCodeBench candidate programs before applying alternative selectors:

1. Original AST-based vote, reproduced exactly.
2. Behavioral grouping from execution on **question-derived, selection-visible** inputs, generated without the reference solution, official hidden tests, or candidate outcome labels.
3. Execution-informed selection using generated assertions where the problem specification supports them; unsupported assertions are not treated as ground truth.

Generate a common suite of up to 20 valid inputs per task with one development-frozen generator policy, used for every candidate and arm. Record environment, input values, outputs, exceptions, timeouts, and deterministic state reset. Use exact output equality where specified; a numerical tolerance must be task-defined and frozen, not chosen to merge useful-looking candidates.

A behavioral signature is a vector over the same input suite. Do not merge programs merely because all crash or timeout; retain failure signatures separately. Tasks that cannot support safe deterministic input generation are marked ineligible for behavioral grouping. Use a separate suite of up to 20 held-back audit inputs, plus official test outcomes only after selection is frozen, to inspect false merges and splits.

Passing the same finite tests does not prove program equivalence. Report merge and split audits, selection lift, retained coverage, suite-generation cost, and fraction of tasks eligible. Sandboxes must have no network, no credentials, restricted filesystem, process/memory/time limits, and a fresh workspace per candidate. [D5]

**E0 deliverable:** a source-separated bottleneck map and a short list of contrasts whose measurement and support are adequate for prospective validation. Do not use it to select only favorable confirmatory tasks.

---

## 8. E1: Matched initial disagreements across task families

**Tests:** R1 and the behavioral foundation of R3. **Primary model:** Qwen3-8B/native thinking. **N:** 5. **Revisions:** 4. **Reasoning:** 2,048 tokens per update. **Protocols:** PRIVATE, PEER, FRESH.

### 8.1 Natural cohort

For each of the target 200 unselected source questions per task, draw five independent roots once and fork that exact team into the three protocols. Do not condition recruitment on whether the roots contain a correct answer. Retain invalid roots in this population.

Use one root team and one continuation replicate per source item in the main allocation. A hash-selected 60-item-per-task subset receives a second independent natural root team, with the same three protocol forks, for root variability. This subset is chosen before outcomes and is a robustness block, not another independent-item sample.

All arms have the same root generation and revision allowances, but PEER reads more tokens. Label this a fixed-update-allowance mechanism comparison. The matched-total-resource study is separate in Section 16.

### 8.2 Constructed teams and eligibility

Use only independent donor generations that yield valid, resolved canonical answers. Do not manually repair their answers or rationales.

| Composition | Root construction | Main purpose |
|---|---|---|
| `C1_COHERENT` | 1 correct donor plus 4 distinct generation records sharing one wrong class `w*` | A correct lone minority faces a coherent wrong majority. |
| `C4_COHERENT` | 4 correct donors sharing the correct canonical class plus 1 donor of `w*` | Mirrored wrong-minority challenge. |
| `C1_FRAGMENTED` | 1 correct donor plus wrong counts `[2,1,1]` over three classes | Secondary comparison of concentrated versus fragmented wrong support. |
| `C0_UNANIMOUS` | 5 independent donors in the same wrong class | Optional all-wrong baseline, not a prerequisite. |
| `C5_UNANIMOUS` | 5 independent correct donors in one correct class | Optional all-correct stability baseline. |

The main paired-composition panel requires enough donors to construct both `C1_COHERENT` and `C4_COHERENT` with the **same item and same wrong class**. Select `w*` among qualifying wrong classes by a frozen hash, not by how persuasive its rationale looks. Use one shared correct donor and one shared wrong donor across the mirrored roots where possible; sample additional donor records without replacement **within** each team. Cross-composition donor reuse is recorded and handled by item clustering.

Four independently generated identical texts may legitimately represent four draws; four copies of one generated record do not. Preserve both `generation_uid` and content hash to distinguish those cases. Never reissue an identical seed to manufacture independent support.

Items eligible for only one composition enter an explicitly named supplemental panel. They do not enter the paired-composition primary or its task interaction.

For the fragmented extension, recruit a prespecified target of 100 paired-eligible items per task where both coherent and `[2,1,1]` roots can be built. Hold the correct donor fixed; retain common wrong donors where possible. This changes evidence content as well as wrong-class composition, so it is **not** as content-controlled as E2's endorsement manipulation. No wrong-answer class or rationale is rewritten to force eligibility.

### 8.3 Deterministic construction algorithm

```text
for item in frozen_item_order:
    obtain fixed difficulty bank and separate donor bank
    grade donors in the offline construction service
    if primary eligibility fails at 32 draws:
        extend to 64 only if the frozen rule authorizes extension
    enumerate eligible correct/wrong classes
    select w* with a fixed class hash
    select independent donor records with a construction seed
    build C1 and C4 roots; assign focal slot using a random permutation
    validate exact C0, class occupancy, and distinct generation records
    seal roots, source IDs, donor IDs, and packet hashes
    export a label-free root bundle to the generation service
```

Eligibility is determined before any PEER/PRIVATE/FRESH continuation on that item. The generation service sees neither eligibility flags nor grade-dependent cohort names. Use opaque IDs in requests.

### 8.4 Protocol definitions

| Protocol | Own previous draft | Peer previous drafts | External transcript |
|---|---|---|---|
| `PRIVATE` | Yes | None | Hidden |
| `PEER` | Yes | All N−1 | Hidden |
| `FRESH` | None | None | Hidden |

FRESH slot identity is bookkeeping only; no role label enters the root prompt. Every fresh draw has its own seed. All protocols execute all four revisions even after consensus, because early stopping would change the estimand and the observed risk set.

The constructed initial state contains grading-selected answers. Its subsequent FRESH arm is still valid as a reset intervention, but the union of selected roots and fresh draws is not an unconditional pass@k bank.

### 8.5 Outcomes and primary contrasts

At each `r=0..4`, compute all Section 4 outcomes. Primary E1 endpoints at r=4 are:

- terminal active coverage in `C1_COHERENT`;
- PEER–PRIVATE task heterogeneity of that endpoint;
- wrong-minority propagation to initially correct nonfocal agents in `C4_COHERENT`.

Report continuous preservation, focal retention, correct propagation, delivered plurality accuracy, and all-unknown/invalid outcomes as essential accompanying measures. FRESH is a required comparator, but PEER–PRIVATE is the principal communication contrast.

A lower minority survival rate over later **at-risk** rounds is not by itself temporal decay: the surviving items may change. Report both baseline-root-cohort trajectories and state-conditional hazards. Treat risk-set migration explicitly.

### 8.6 Information-content bridge

On a hash-selected 60-item-per-task subset of the main constructed panel, run an answer-only PEER arm. Reuse the same roots, seeds, graph, and revision template, but remove rationale and claims from peer packets while retaining peer answers and source slots. Own-draft visibility is unchanged.

This provides a limited bridge to Run0's content-sharing question. It does not recreate Run0's raw native-thinking exposure or its serving regime. Report input-length changes and do not identify the difference as “rationale quality” alone.

### 8.7 Acceptance and interpretation

E1 is complete when the frozen source-item quota has been exhausted or filled; every planned episode has either valid data or a typed failure; roots match byte-for-byte across paired arms; and all primary outcomes, uncertainty, missingness bounds, and intended denominators are exported.

The result is interpretable whether task effects persist, shrink, reverse, or are imprecise. Only a powered equivalence test licenses “no practically important residual task difference.”

---

## 9. E2: Evidence evaluation versus endorsement

### 9.1 E2a: Fixed natural evidence packets

**Purpose:** change social endorsement while holding the candidate text fixed.

Recruit 100 independent source questions each from the MATH and MMLU-Pro confirmation frames with one correct and one wrong donor packet; TruthfulQA is an optional third panel. Selection occurs without receiver outcomes. Use the main donor bank; a blinded audit checks that packets are interpretable and do not contain explicit correctness tags. Natural rationale validity is not guaranteed by final-answer correctness; record audited validity separately and keep the unfiltered panel visible.

For every item, freeze two packets and two possible working drafts, one correct and one wrong. The working draft is explicitly an assigned draft, not an assertion that this receiver generated it previously.

Cross:

| Factor | Levels |
|---|---|
| Endorsement cue | Correct candidate endorsed 1:4; correct candidate endorsed 4:1; counts hidden |
| Initial working draft | Correct; incorrect |
| Receiver reasoning cap | 1,024; 4,096 tokens |
| Continuation replicates | 2 independent seed schedules |
| Receiver model | Qwen3-8B; Qwen3-32B |

Every receiver sees both fixed solution packets. Support counts refer to a **constructed panel of five advisers**, not literally the other four agents of E1. They are randomized experimental metadata, not five independently generated votes. Mark this explicitly in the published method and data.

Candidate order, external labels, and support-field order are balanced independently of correctness. The two displayed packet bodies are byte-identical under all support conditions for an item. The 1:4 and 4:1 arms differ only in count fields. Hidden counts are a separate cue-removal intervention and need not have identical token length.

Run a PRIVATE working-draft-only control for each initial-draft/budget/seed combination. It estimates revision without peer evidence. A post-exposure answer change alone is not proof of conformity or evidence transmission.

**Calls:** 100 × 3 × 2 × 2 × 2 = 2,400 one-call receiver episodes per task/model, plus 800 PRIVATE controls. No native multi-agent roots are generated by the receiver model in this replay assay.

### 9.2 E2b: Synthetic mod-97 generator

**Purpose:** manipulate genuinely checkable evidence, controlled dependency structure, and receiver computation.

Generate 300 independent source problems, 100 at each dependency depth `D ∈ {4,12,24}`, plus 30 separate pilot problems. Every source problem has **24 displayed operations**, indexed 1..24:

\[
x_j=(a_jx_{p(j)}+b_j)\bmod 97,
\]

with `x0 ∈ {0..96}`, `a_j ∈ {1..96}`, and `b_j ∈ {0..96}`. Always ask for `x24`.

For depth D, choose an ordered chain of D indices ending at 24; sample its D−1 other indices without replacement from 1..23 and sort them. A chain node depends on the preceding chain node, with the first depending on x0. Every off-chain node depends on x0. Thus all 24 rows are meaningful defined variables, but the answer has a path of exactly D operations. Display all operations in topological order. Random seeds determine coefficients, starting value, chain positions, and packet order.

Construct:

- A correct trace containing all 24 evaluated rows and the correct x24.
- A wrong trace with exactly one violated local equation at a randomly chosen **nonterminal node on the answer chain**. Add a nonzero delta modulo 97 at that node, then recompute descendants consistently from the altered parent value. Other rows remain correct.

Nonzero multipliers preserve a nonzero error along the chain. Independently assert that the final answers differ, that exactly one local equation is violated, and that the correct packet has no violation. The generator and evaluator must be separate implementations, checked against hand-computed fixtures.

Depth changes dependency and error-localization structure at fixed displayed row count. It is not a claim that all other measures of difficulty are held constant. Log token length and error position; balance error position within relative chain-position bins where possible. Hold the same two traces fixed for every endorsement/checker/budget condition of a source problem.

### 9.3 The local-step checker

The only public tool is:

```text
check_step(packet_id, step_id) ->
    {packet_id, step_id, status: "valid" | "invalid" | "disabled" | "bad_request"}
```

The checker tests the selected row's stated output against the stated parent value and the problem's fixed coefficient/offset. It does not recursively recompute the trace, identify the injected error, reveal the correct final answer, or rank packets. A locally consistent downstream row of the wrong trace can return `valid`.

The receiver may request at most **four distinct checks** per episode. Requests refer only to immutable displayed packets and valid step indices. Duplicate/invalid requests consume a request slot and return the fixed typed response; no adaptive exception grants extra checks. No arbitrary Python execution or hidden evaluator API is available.

### 9.4 Two-phase receiver protocol

Use the same two-generation-call structure in both checker conditions:

1. **Plan phase:** read the question, assigned draft, fixed packets, support fields, and whether checking is available. Reason with cap `B/2`; emit up to four `check_step` requests in JSON, with at most 256 public-output tokens.
2. **Check service:** when enabled, return local validity results; when disabled, return the same response schema with status `disabled`. Empty request lists are valid. This is tool access, not a placebo that secretly provides correctness feedback.
3. **Decision phase:** read the same evidence plus plan/check transcript; reason with cap `B/2`; emit the answer schema with at most 512 public-output tokens.

Total **reasoning cap per logical episode** is B, not B per call. Unused phase-1 reasoning is not transferred in the primary protocol. Tool latency, tool requests, repeated prompt processing, plan output, and final output are all metered. Invalid plan output yields an empty check set and continues to decision without a repair call.

The two-phase arrangement controls generation opportunities while separating tool access. This is a new experimental adapter and is not byte-identical to ARD36's one-call assay.

### 9.5 E2b factorial and workload

| Factor | Levels |
|---|---|
| Problem depth | 4, 12, 24; 100 source problems per depth |
| Endorsement | Correct 1:4, correct 4:1, hidden |
| Checker access | None; up to four local checks |
| Initial working draft | Correct trace; wrong trace |
| Total reasoning budget B | 1,024; 4,096 tokens |
| Receiver model | Qwen3-8B; Qwen3-32B |
| Continuation replicates | 2 |

There are 300 × 3 × 2 × 2 × 2 × 2 = **14,400 logical receiver episodes per model**, each with two generation calls: **28,800 calls per model**, excluding controls.

Add working-draft-only, no-peer, no-check controls at both budgets and both initial drafts: 300 × 2 × 2 × 2 = 2,400 logical episodes or 4,800 calls per model. These use the same two-phase adapter with no usable checks.

Do not multiply by the three depth levels again: they are already represented in the 300 source problems. Treat the source problem, not the call or tool request, as the independent cluster.

### 9.6 Primary mechanism contrasts

Let `Y(s,v,b)` denote receiver correctness averaged over the declared source problems, initial drafts, and replicates, with support s, verification v, and budget b.

Define endorsement sensitivity:

\[
E(v,b)=E[Y(majority,v,b)-Y(minority,v,b)].
\]

Primary checker interaction:

\[
I_V=\operatorname{avg}_b E(1,b)-\operatorname{avg}_b E(0,b).
\]

Primary reasoning interaction:

\[
I_B=\operatorname{avg}_v E(v,4096)-\operatorname{avg}_v E(v,1024).
\]

Negative values indicate a smaller majority advantage under usable checking or more reasoning. Test two-sided effects. Main effects of verification and budget, their interactions with depth and starting-draft correctness, and hidden-count comparisons are secondary unless promoted before confirmation under an amendment.

Report checker use, requested-step distribution, actual detection of an invalid row, adherence after detection, confidence, and validity. Checker usage/detection are post-treatment variables: subgroup analyses describe behavior and do not replace the intention-to-treat effect of checker access.

### 9.7 Linking E2 to natural task dynamics

Compare natural-packet endorsement effects with E1 transitions on the same source questions using grouped/cross-fitted analysis. A receiver's measured susceptibility may be a predictive covariate; estimating it requires additional calls and cannot be called free deployment information.

A synthetic interaction does not establish the same mechanism on all benchmarks. A natural-packet interaction is necessary for that extension. Failure to transfer limits the scope, rather than justifying selective omission of the synthetic or natural result.

---

## 10. E3a: Directed minority influence and vulnerability

**Priority:** core. **Primary setting:** Qwen3-8B, N=5, four revisions, 2,048 thinking tokens. **Recruitment:** target 150 paired-composition source questions per primary family from E1, selected by a frozen hash before condition outcomes.

### 10.1 Directed graph intervention

Let f be the focal root minority agent. Define `G[j,i]=1` to mean agent i receives agent j's preceding public packet. No self edge is included; the own-draft channel is separately present in every non-FRESH update.

All directed edges among the four nonfocal agents remain present. Cross the focal agent's incoming and outgoing channels:

| Condition | Focal receives nonfocal packets | Nonfocal agents receive focal packet | Directed peer edges at N=5 |
|---|---|---|---:|
| `IN1_OUT1` | Yes | Yes | 20 |
| `IN0_OUT1` | No | Yes | 16 |
| `IN1_OUT0` | Yes | No | 16 |
| `IN0_OUT0` | No | No | 12 |

Run all four on the **same** `C1_COHERENT` and `C4_COHERENT` roots. The focal identity is fixed throughout an episode, even if the focal answer later joins a majority. Do not recompute the protected slot by oracle correctness each round.

The focal agent generates every round even when isolated. `IN0_OUT0` is not PRIVATE for the whole team: the other four agents still communicate. The `IN1_OUT1` arm may reuse the E1 PEER trace only when the complete request, seed schedule, packet, model, and parent hashes match; mark that reuse explicitly.

### 10.2 Estimands

For the correct focal minority, estimate:

- **Incoming-channel effect on focal correctness:** focal correctness with incoming messages absent minus present, averaging over outgoing status.
- **Outgoing-channel effect on others:** mean correctness of initially wrong nonfocal agents with focal output visible minus hidden, averaging over incoming status.
- **Directional contrast:** `IN0_OUT1 − IN1_OUT0` on delivered accuracy, active coverage, and continuous preservation. These two arms have the same total directed-edge count.

For the wrong focal minority, use the same intervention and measure focal retention of `w*`, nonfocal adoption of `w*`, all correct-to-wrong transitions, and delivered accuracy. A method that preserves both true and false dissent is a structural finding; it is not automatically a beneficial policy.

The initial focal label is used only by the offline constructor/analysis. It is never exposed in prompts and cannot be used by a deployed controller.

### 10.3 Context-load controls

On a prespecified 60-item-per-task subset, repeat the equal-edge directional contrast using fixed packet slots. Replace unavailable packets with a length-matched, task-independent inert record, while leaving positions and all available packet bodies fixed. Use a frozen filler generator with no task facts, recommendations, or correctness cues. Log exact token differences.

This is a diagnostic control, not proof that filler is behaviorally inert. The omitted-message contrast measures total exposure; the matched-slot contrast tests sensitivity to packet presence/length. If filler itself changes behavior, report both rather than claim a pure semantic mechanism.

**Do not use a copied own answer as the primary length control:** that would also increase apparent self-support. A separately labeled `SELF_SUBSTITUTION` diagnostic is allowed, but it is a repetition intervention.

### 10.4 Natural-population protection check

On the natural E1 cohort, compare all-to-all PEER with `IN0_OUT1` protection of one **uniformly random slot chosen before root answers**. Protect that same slot through the episode. This test requires no correctness labels and reveals whether the oracle-targeted minority effect has a useful natural-population analogue.

Any alternative minority-detection rule must use only visible root answers and be frozen before testing. A correct-minority-targeted effect alone is not a deployable improvement.

### 10.5 Interpretation boundary

Directional edges identify behavioral effects of information exposure within this design. They do not establish that a particular neural feature mediates those effects. The claim “minority influence and vulnerability are distinct” requires the outgoing and incoming outcomes, including the wrong-minority mirror, not merely an overall accuracy increase.

---

## 11. E3b and E3c: Restricted structural and model transfer

### 11.1 E3b block A: routing pathways at fixed degree

On 100 hash-selected paired-composition items per family, at N=5 and 2,048 thinking tokens, compare:

- A fixed bidirectional degree-two ring, with slot positions randomized once.
- A newly randomized bidirectional Hamiltonian ring each round, sampled independently of outputs.
- All-to-all PEER.
- PRIVATE.

Fixed versus rewired ring has the same indegree, outdegree, number of packet slots, and update schedule. Actual packet lengths can differ as trajectories evolve; record this. The graph schedule is sealed before outcomes. Random rewiring does not depend on disagreement or correctness.

Record the same round checkpoints and outcomes. Transfer testing concerns whether the earlier mechanism predicts these conditions, not just which row has the largest mean accuracy.

### 11.2 E3b block B: team size and horizon

On a separate preregistered eligible subset, compare `N ∈ {3,5,9}` under PRIVATE and all-to-all PEER, holding model, per-update cap, selector, and memory fixed.

Use two initialization series:

| Series | N=3 | N=5 | N=9 |
|---|---|---|---|
| Lone correct agent | C0=1 | C0=1 | C0=1 |
| Approximately one-third correct | C0=1 | C0=2 | C0=3 |

Wrong agents share a coherent wrong class in the primary series. Report that the fraction at N=5 is 2/5 rather than exactly 1/3. The N=3 condition shared by both series is one condition, not duplicate evidence.

Target 100 independently eligible source questions per task; report common support across N. A 9-agent constructed team may require rare donor configurations even after 64 draws. Do not manufacture supporters or extrapolate a 3/5-agent result to N=9.

Record `r ∈ {0,1,2,3,4,8}` when the extension to eight revisions is enabled. Generate to r=8 once and score prefixes; do not regenerate separate R=1/2/4 endpoints. R appears only in the scheduler, not a prompt saying how many rounds remain. A terminal-aware prompting study is a separate intervention.

Ring and all-to-all are identical at N=3; skip that nominal contrast. More agents at fixed per-call budgets also cost more, so this block is mechanistic scaling, not fixed-compute superiority.

### 11.3 E3b block C: native reasoning budget

At N=5 and r=4, on 100 preregistered paired-composition items per task, replay the same root drafts under `B ∈ {1024,2048,4096}` for PRIVATE and PEER. Change only the thinking cap; keep sampling parameters, final allowance, prompt, and public memory policy fixed.

The primary contrast is receiver-budget-by-protocol interaction with fixed roots. A separate native-root budget panel, if run, measures both improved initial generation and improved revision. Do not conflate the two.

### 11.4 E3c: model size and family

Use 100 prespecified source questions per task for each of two contrasts:

**Fixed-root replay:** Qwen3-8B and Qwen3-32B revise the same label-free root drafts and peer content at the first update. Later rounds evolve endogenously within each model. Compare first-update receiver effects separately from full-trajectory model effects. Root text supplied to a different model is an assigned draft, not evidence of that model's original belief.

**Native initialization:** each model independently produces its own natural roots and matched donor banks on the same source questions. Then compare PEER–PRIVATE under the same logical protocol. Report eligibility and initial answer distributions per model; matching on all models' eligibility changes the population.

Run the family-transfer adapter from Section 5.6 on a frozen subset. Do not describe Qwen8B versus Llama8B as equal competence or identical training. An apparent 8B-to-32B change is a two-point scale contrast, not a scaling law or phase transition. A 14B interpolation point is optional only if a nonlinear scale claim is pursued.

### 11.5 Transfer accounting

Every transfer result must identify what is held out: source question, task family, topology, model, budget, or a combination. Same-question replay across models tests model transfer of the intervention, not independent-item replication. A predictor cannot be claimed to transfer to a topology whose labeled outcomes were used to choose its features or hyperparameters.

---
## 12. E4: Prospective prediction and optional action control

**Priority:** optional. Prediction can be published without a controller. Do not delay the core behavioral experiments to optimize a complex policy.

### 12.1 Separate explanatory and deployable feature sets

| Feature set | Examples | Allowed use |
|---|---|---|
| Offline explanatory | True `C_r`; largest wrong class; grade-based minority identity; independent test-item bank success | Describe and model mechanisms; explicitly not deployable without an oracle. |
| Observable behavioral | Vote shares, answer identities, recent switching, persistence, rationale novelty, packet exposure, public confidence, remaining budget | Forecasting and practical policy. |
| Instrumented | Residual-stream metrics or probes available at the stated decision time | Optional policy; charge measurement overhead and test incremental value. |
| Privileged forbidden | Hidden tests, official gold, future answers, future tool results, treatment branches not yet executed | Never policy input or test-time selector input. |

An independently graded bank on the same test question is not free task-difficulty knowledge. It is an offline explanatory feature unless its acquisition and labels would genuinely be available and charged at deployment.

### 12.2 Prediction targets and model hierarchy

Fit simple baselines before flexible models:

1. **Count-only transition baseline:** smoothed `P(C_{r+1} | C_r, round, protocol)` (default symmetric Dirichlet pseudocount 0.5 per next-state category) on fully gradeable states, with a separate unresolved-state outcome. Use it to test whether correctness count is an adequate coarse state. Do not presume a Markov property.
2. **Richer one-step model:** regularized multinomial/logistic models for next correct-count, last-correct-answer loss, and delivered correctness, adding answer concentration, graph exposure, independent difficulty, and current validity. Compare models with and without lagged state features.
3. **Observable forecasting model:** use only deployable features to predict next-round and final selected correctness, without supplying true `C_r`.
4. **Confidence and geometry additions:** add them sequentially to a frozen behavioral/text baseline and a parameter-matched baseline.

For multi-step predictions from state S_r, use direct horizon-specific models for remaining horizons 1, 2, and 4, or an explicitly validated state-transition simulator. Do not claim autonomous multi-round prediction while feeding the model observed future confidence, agreement, or geometry. The simple count-only model may be rolled forward as a baseline; its failure is evidence that coarse correctness-count dynamics are insufficient.

Use ridge/logistic models as defaults; one small gradient-boosted model is a secondary comparator. Hyperparameters and feature transformations are chosen on development only. Evaluate log loss, Brier score, calibration, and task-resolved forecast error; AUROC is supplementary. A model must improve held-out prediction beyond the appropriate baseline, not just correlate with an outcome.

Use source-item-grouped cross-fitting for within-study prediction. For a task-held-out claim, freeze all decisions using other task families and test on the held-out family without refitting. A cross-fitted result is not an independent external replication; state the distinction.

### 12.3 Fork states to label action values

Sample states at r=1 and r=2 by a frozen rule from the natural and constructed cohorts, recording their sampling probabilities. Do not retain only states whose true correctness makes the action contrast look favorable. Source-item splits precede state sampling; all states from one item remain in one fold.

From each identical state, run:

| Action | Generated continuation | Candidate pool for selection |
|---|---|---|
| `STOP` | None | Current candidates |
| `PRIVATE_DEPTH_1` | One private update for each of N agents | Current plus new candidates |
| `C_FULL` | One peer update for each of N agents | Current plus new candidates |
| `FRESH_N` | N independent fresh solves | Current plus new candidates |

Use one common, frozen candidate-only audit selector and a shared `stage_and_terminal` retention rule. Every non-STOP branch retains the old candidates; future labels do not decide what is retained. The pool difference from STOP is the newly generated evidence, not an arbitrary removal of old options.

Reserve the selector's compute before admitting an action. At each registered budget, disallow actions whose declared envelope cannot fit; STOP always remains admissible. Report same-budget-allowance comparisons and actual expenditure separately. Unspent budget is a real policy outcome.

Generate at least two independent action replicates for development-state labeling where feasible, because one realized branch outcome is a noisy reward label. An oracle maximum across realized branches is an optimistic analysis ceiling, not the expected value of an implementable policy.

### 12.4 Policy and test

Use a small supervised action-value model predicting `Pr(correct | observable state, action, budget)`. Choose the action with the highest predicted correctness among feasible actions; break near-ties with a development-frozen lower-cost preference. Freeze the model and any threshold before the final test.

Compare to STOP, always-PEER, always-PRIVATE, always-FRESH, a random feasible action, and the best fixed action selected on development. Include confidence-weighted and archive-aware selection baselines where they apply; do not compare a policy with a stronger selector to fixed policies with a weaker selector.

Primary optional policy outcome: delivered accuracy under the same budget ceiling. Secondary: measured cost at a prespecified accuracy level, correct-minority rescue, incorrect-minority harm, validity, and calibration. Report both natural-population and challenge performance.

A practical-controller claim requires a source-item-disjoint test untouched by controller development. Legacy data, pilot data, or cross-fitted predictions may train the controller, but all protocol/representation shifts must be declared. Do not reuse the already-opened confirmation set as a new untouched policy test.

---

## 13. Confidence, calibration, and neural measurements

### 13.1 Confidence timing and definitions

Capture public confidence **after the answer-relevant reasoning**. Keep any pre-generation option probe separately named. A pre-generation score may be useful for prediction but is not the agent's post-reasoning confidence. [RUN0, §3.4; ARD, §2.2]

Record raw final-answer token log probabilities where available, with sums, per-token means, answer length, and normalization scheme. A free-form sequence probability is not automatically a probability of correctness. For MCQ option scoring, score all canonical options under the same prospective prefix; handle multi-token option renderings exactly. Do not assign zero probability merely because an option was absent from a truncated top-k response.

Minimum confidence analyses:

- Candidate Brier score, signed bias, and reliability plot.
- Within-source-item and within-team-round discrimination of correct versus wrong candidates, including ties at half credit and only groups containing both classes.
- System confidence versus **selected system correctness**, with one row per selection event.
- Raw versus development-fitted recalibration, with family/protocol-specific and pooled versions labeled separately.
- Actual selection/action utility from using the signal, not merely its average separation.

Use fixed development-defined reliability bins for the main plot and exact atoms for discrete vote fractions. ECE is supplementary to proper scoring rules and depends on the chosen bins. Bootstrap source items, not candidate rows. Report how many mixed groups support within-item AUC; a single system candidate per item cannot define within-item discrimination. [ARD, §2.2; L6]

Never interpret the sign of `ECE(system)-ECE(agent)` as the direction of overconfidence. ECE is unsigned. Use signed bias and reliability curves for over- versus underconfidence. Differences can reflect changed accuracy, resolution, and support, not necessarily corruption of a confidence mechanism.

For confidence-weighted voting, a missing confidence remains null in measurement tables. Its fallback **selection weight** is the development-estimated task-specific root success rate, or a globally frozen development base rate if task-specific support is inadequate. The weight is not imputed as a model statement.

### 13.2 Capture plan

Read-only activation capture must be technically supported from the beginning so data need not be regenerated. Full neural hypothesis testing is optional.

Use a hash-selected target of 100 constructed source questions per family in E1 and the same questions in E3a where available. Capture all five root members and all five members after every update for PRIVATE, PEER, and the directed-channel arms. Select items before outcomes, not only episodes that become unanimous. Capture natural-cohort states on a prespecified subset as a population check.

Anchor definitions:

| Anchor | Exact location | Valid predictive use |
|---|---|---|
| `PROMPT_END` | Last prompt token before this call's generation | Predict the current answer or later transitions. |
| `PRE_FINAL` | Last token before the first token of the final-answer value, after the generated public rationale | Predict current answer outcome, acknowledging that preceding reasoning may already reveal the answer semantically. |
| `ANSWER_END` | Last token of final-answer value, before the public confidence value | Describe answer state or predict the **next** update; not pre-answer prediction. |

Retain delimiter offsets and byte/token span mappings. If an answer is invalid or the anchor does not exist, mark the capture missing with a reason. Do not choose a substitute anchor based on correctness.

Capture at zero-based layers `floor(f*L)-1` for `f ∈ {0.25,0.50,0.75}`, using the actual model layer count L. For Qwen3-8B this gives 8, 17, 26; derive every other model from its pinned config. [D6]

Use exact stored prompt/output token IDs for teacher-forced replay through the same pinned model. Validate tokenizer identity, hook location, and logits on fixtures. Log tolerances and numerical differences rather than demand universally bit-identical GPU kernels. Instrumented inference must not alter behavioral generation.

For cross-model assigned-root replay, distinguish native root activations from teacher-forced activations of a foreign-model draft. They are not interchangeable root measurements.

### 13.3 Geometry metrics

For the N agent vectors at one matched anchor/layer, compute:

- mean/min/max Euclidean distance; centered total variance; vector norm;
- mean cosine distance on normalized nonzero vectors;
- participation ratio `(sum λ)^2 / sum λ^2`;
- entropy effective rank and leading eigenvalue share;
- displacement of each agent and the alignment of displacements between rounds.

Use both raw and development-normalized scale-carrying metrics. Center across agents for covariance and compute eigenvalues via the small N×N Gram matrix. Rank is at most N−1. When total variance is zero, spectral ratios are **undefined**, not evidence of a one-dimensional collapse.

Identical root prompts can produce identical `PROMPT_END` activations under deterministic forward evaluation; that zero spread is expected, not a capture failure. Root answer-state diversity is the relevant baseline for generated-answer geometry.

Do not directly compare raw distances across models with different hidden dimensions/scales. Report within-model contrasts, and common-size agent subsamples for cross-N spectral comparisons. Subsamples are repeated measurements of the same source item, not new independent observations.

### 13.4 Predictive evaluation and claim gate

Compare behavioral/text, behavioral/text+confidence, and behavioral/text+confidence+geometry models using identical folds and development-selected capacity. Include a parameter-matched text expansion so apparent benefit is not simply added model capacity. Control predictive comparisons for context length, answer length, answer identity, and visible agreement, with ablations rather than claims of causal adjustment.

Promote geometry into the headline only if it yields a meaningful, held-out improvement in forecasting an important correctness transition or action value. A contrast showing that peer packets change activations establishes a representational effect, not a causal pathway to error. Causal mediation or activation-edit claims require a separately preregistered intervention and specificity controls.

---

## 14. Primary statistical analysis and power

### 14.1 Frozen primary family

All tests are two-sided. Apply Holm family-wise correction at 0.05 across the following eight hypotheses; publish every estimate and interval regardless of rejection. The Qwen3-8B core supplies the primary tests; Qwen3-32B and family blocks are prespecified transfer analyses.

| ID | Experiment | Primary estimand |
|---|---|---|
| H1 | E1 | Equal-task-weighted PEER−PRIVATE difference in terminal active coverage in `C1_COHERENT`, r=4. |
| H2 | E1 | Omnibus task-by-protocol interaction for the H1 endpoint across the three primary task families. |
| H3 | E1 | PEER−PRIVATE difference in nonfocal adoption of the root wrong class `w*` in `C4_COHERENT`, r=4. |
| H4 | E2b | Checker-access-by-endorsement interaction `I_V`. |
| H5 | E2b | Reasoning-budget-by-endorsement interaction `I_B`. |
| H6 | E3a | Effect of removing focal incoming messages on focal correctness in `C1_COHERENT`, averaged over outgoing status, r=4. |
| H7 | E3a | Effect of enabling focal outgoing messages on mean nonfocal correctness in `C1_COHERENT`, averaged over incoming status, r=4. |
| H8 | E3a | `IN0_OUT1−IN1_OUT1` difference in nonfocal adoption of `w*` in `C4_COHERENT`, r=4. |

H1 uses verified active coverage `Qlo` on all planned items, with semantic bounds and complete-case sensitivity mandatory. H3/H8 count observed adoption of a known wrong root class, not all missing answers as adoption. Report unknown fractions beside them. A large or treatment-differential unknown rate blocks a strong semantic interpretation even when an operational contrast is estimable.

If a hypothesis cannot be estimated, retain it in the family with status `NOT_ESTIMABLE` and conservative `p=1` for family correction; do not replace it after seeing results. A pilot-stage amendment can change the planned family only before confirmation is locked and must preserve its rationale and date.

Endpoint delivery accuracy and continuous preservation are essential secondary outcomes. Do not imply that rejection on coverage proves an accuracy improvement or a successful selector.

### 14.2 Estimation and uncertainty

The primary independent unit is the original source item. Average its prespecified root/seed replicates within each condition before calculating task means. Use equal task weights for the cross-task summary; retain source-sampling weights for each natural-cohort estimate.

Use 10,000 item-cluster bootstrap replicates, paired across every condition of an item, to construct 95% intervals. For tests, use a studentized centered wild-cluster/bootstrap test under the relevant null, with source-item weights shared across arms. Freeze the implementation and seeds. A simple exact McNemar test is an additional check for a binary one-root/one-replicate paired contrast; it is not valid on repeated agent rows treated as independent pairs.

For H2, test equality of task-specific paired effects directly with a joint contrast and its item-cluster covariance, using a null-centered bootstrap to handle heterogeneous task variance. Do not infer interaction from “significant on MATH, nonsignificant on MMLU-Pro.”

For E3 averaged factorial effects, form the within-item factorial contrast first, then resample source items. For E2, aggregate or model the full within-problem factorial structure, including both seeds. Do not count 14,400 receiver episodes as 14,400 independent problems.

Predeclare secondary outcome families and report their multiplicity status. Large exploratory interaction tables may use a separate false-discovery-rate summary, clearly distinct from the eight primary tests. Do not present every uncorrected effect-size interval as a confirmation.

### 14.3 Power and recruitment rule

Run the 60-item-per-task pilot to estimate eligibility, outcome discordance, missingness, within-item variance, and costs. If fewer than 20 items per task support the paired construction, an eligibility-only pilot extension to 120 and then 180 source items is permitted under a predeclared rule. All inspected pilot items remain excluded from confirmation.

Planning minimum effects of interest:

| Contrast type | Proposed target effect |
|---|---:|
| Correct-minority terminal coverage/retention | 10 percentage points |
| Difference between two task-specific communication effects | 10 percentage points |
| Endorsement-by-checker or endorsement-by-budget interaction | 8 percentage points |
| Nonfocal correct propagation | 10 percentage points in mean nonfocal fraction |
| Incorrect-minority propagation harm | 5 percentage points in mean nonfocal fraction |

These are design targets, not expected effects. Simulate the complete paired/factorial analysis at candidate source-item counts `[150,200,300,450,600]`, incorporating observed pilot covariance and eligibility. Use conservative per-test `alpha=0.05/8` for planning, then Holm for the actual family. Choose the smallest feasible allocation reaching at least 80% simulated power for the declared primary contrast(s).

For a rough binary paired check only:

\[
n\approx\frac{(z_{1-\alpha/2}+z_{0.8})^2(p_{discordant}-\delta^2)}{\delta^2}.
\]

Use simulation for interactions and clustered continuous fractions. Power to detect an overall difference does not imply power to detect a task interaction or to establish absence of wrong-minority harm.

The power-selected sample-size lock overrides the provisional module targets. For nested experiments, the parent cohort must contain at least the required number of eligible child-panel items; the planner must detect an impossible E3 subset rather than silently reuse questions.

If available source items or compute cannot support the target, write a signed scope decision: enlarge the permitted source frame, reduce breadth before depth, designate a specific contrast exploratory, or accept wider intervals. Do not automatically replace independent items with extra seeds or redefine the target effect after confirmation results are visible.

For a practical-equivalence claim, predeclare a margin, proposed ±5 percentage points, and use an equivalence test/interval with its own power calculation and multiplicity treatment. A wide interval crossing zero is inconclusive, not “no effect.”

### 14.4 Standardization in the new data

Primary E1 constructed contrasts already fix correct-count and coherent wrong support. Analyze task interactions there first. Then compare them with natural-cohort effects and standardized natural effects to ask how much observed heterogeneity changes when root composition is held comparable.

Use pre-communication variables only. Difficulty-bank outcomes are noisy measurements; provide a sensitivity over prespecified difficulty bins and uncertainty-aware estimates. Avoid a heavily parameterized model that uses the same small minority panel to invent and confirm task features.

Do not average natural and constructed cohorts into a single benchmark accuracy. The constructed cohort is deliberately outcome-selected at initialization and estimates a conditional challenge population.

### 14.5 Safety against selection and reporting artifacts

- No outcome-based reruns, no changing a prompt after its confirmation outcomes are opened, and no best-seed selection.
- Stop recruitment at the frozen quota/frame/cost rule, never when a p-value crosses a threshold.
- Keep risk-set migration separate from within-root longitudinal change.
- Report all hypotheses, including failed mechanism predictions and unsuccessful policies.
- Audit influential discordant items and evaluator disagreements blinded to the desired effect.
- Distinguish sampling uncertainty, finite-benchmark coverage, unresolved grading, and implementation failure.

---

## 15. Pass@k, selection, and failure accounting

### 15.1 Unconditional sampling comparison

On natural-cohort items, the five natural roots plus the 20 FRESH draws form 25 independent sampling opportunities when root and FRESH prompts, sampling, and model contracts match. Use this pool for the matched-generation-count reference. Do not use constructed roots in that pool.

Report `k ∈ {1,3,5,10,25}` using a fixed item set and fixed 25-draw bank. For n resolved independent draws with c correct, the finite-sample estimator is:

\[
\widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom nk},\quad k\le n.
\]

For unresolved draws, report lower/upper versions from the fixed original n opportunities with unknown outcomes assigned wrong/correct respectively. Also report a **fixed** complete-case panel satisfying the largest k, never a different item population at each k.

Estimate vote@k and audit@k on the same deterministic prefixes or frozen subset schedule. The oracle and selector must use the same actual candidate subset for a pointwise selection-gap decomposition; do not subtract an averaged pass@k estimator from a differently sampled vote prefix and call a negative value a true selection gap.

### 15.2 Generations are not total compute

A five-agent root plus four revisions costs 25 solver decisions. The new study must not reuse ARD36's 20-generation caption, which referred to three revisions. Peer prompts are longer, models differ in computation, and audit/verification calls add cost. Plot generation-count comparisons as generation-count comparisons, with separate actual-resource analyses.

An oracle pass@k curve is an upper bound on available correctness, not an attainable system score or a free verified sampling method. Compare deployable selectors and include their cost. [L1]

### 15.3 Operational failure policy

Infrastructure retries: at most two retries of the exact request and seed for transient preemption, transport failure, or unavailable endpoint. Every attempt is logged and costed. A completed model response with a wrong answer, malformed format, refusal, or long reasoning is not eligible for a fresh stochastic retry.

Use atomic output commits and an idempotency key. If a request has possibly completed but the response is not recoverable, flag uncertain execution and its cost; do not count two attempts as two independent samples. A terminal unresolved request remains in the intended denominator with typed missingness.

No fallback to an unrecorded option probe or model-generated repair in the primary study. A parse recovery rule, if deterministic and source-blind, must be fixed before confirmation and audited against adversarial fixtures.

---

## 16. Compute, resource matching, and launch limits

### 16.1 Two different resource comparisons

**Mechanism view:** equal agent count, rounds, and reasoning/final-output allowances; report actual tokens and compute. PEER may read more tokens. This isolates a declared communication operation at fixed update opportunities, not equal total cost.

**Efficiency view:** common per-item resource ceilings, all selector/verifier cost included, with admission/stopping fixed before execution. Compare delivered accuracy versus measured cost and versus allowance. Do not obtain “matched compute” by selecting completed episodes with convenient token counts after the fact.

For efficiency, profile per-request cost by model, prompt length, output phase, and hardware during the pilot. Define budget levels at 0.5×, 1×, and 2× a frozen five-agent/one-peer-update reference cost. Use a conservative envelope to reserve remaining final-output and selection cost. Stop before admitting a full synchronous round that cannot fit; preserve the last complete state. STOP is always feasible. Unused allowance must be reported, not hidden.

As a documented fallback, export `linear_compute_proxy = 2 * parameter_count * (uncached_prompt_tokens + generated_tokens)` in addition to total prompt tokens. This is a rough dense-transformer linear-operation proxy, not a complete FLOP count: attention, caching, and serving overhead require separate accounting. Do not label it measured FLOPs.

A request-count or generated-token proxy is useful but not a hardware-independent FLOP measure. Export actual GPU/CPU telemetry and a separately labeled model-compute estimate whose formula/version is documented.

### 16.2 Physical execution cost versus per-system inference cost

Separate:

- **Campaign cost:** all bank screening, retries, reused generation objects, scoring, profiling, activation replay, and analysis actually executed.
- **Per-system inference cost:** the cost the declared protocol would incur on a new item, charging its own roots, updates, selection, verification, and online feature extraction.

Shared-root caching saves campaign computation but does not make root inference free in a deployment comparison. Shared candidate-scoring caches similarly do not make a verifier free. Dataset-construction oracle labels are offline research costs and unavailable to deployed policies.

Avoid request-wall-time summation as GPU time under batching. Attribute active device time once across concurrent work, including tensor-parallel devices, and verify conservation against device-level totals. Keep user-visible latency, queueing, and physical GPU seconds distinct. [ARD, §2.2]

### 16.3 Workload calculator

Let I be independent source items, K root compositions, P protocols, N agents, R revisions, and S continuation replicates. Solver update calls are:

\[
\text{calls}=I\,K\,P\,N\,R\,S.
\]

The following is an **illustration at planning allocations**, not approval to dispatch everything:

| Block | Logical allocation | New generation calls before optional reuse |
|---|---|---:|
| E1 constructed, per task | 300 items × 2 compositions × 3 protocols × 5 agents × 4 revisions | 36,000 |
| E1 natural, per task | 200 items × 3 protocols × 5 agents × 4 revisions | 12,000 |
| E1 natural roots, per task | 200 items × 5 draws | 1,000 |
| E1 second-root robustness, per task | 60 items × 3 protocols × 5 × 4, plus roots | 3,600 + 300 |
| E1 answer-only bridge, per task | 60 items × 2 compositions × 1 added protocol × 5 × 4 | 2,400 |
| Difficulty+donor banks, per screened item/model | 16 + 32 draws; eligibility extension adds up to 32 | 48–80 |
| E2a, per task/model | 2,400 packet episodes + 800 private controls | 3,200 |
| E2b, per model | 14,400 two-phase packet episodes + 2,400 two-phase controls | 33,600 |
| E3a, per task | 150 items × 2 compositions × 4 arms × 5 × 4 | 24,000; 18,000 if the full reciprocal arm is exactly reused |
| E3a natural random protection, per task | 200 items × 1 added protocol × 5 × 4 | 4,000 |
| E3a matched-slot diagnostic, per task | 60 items × 2 compositions × 2 arms × 5 × 4 | 4,800 |

Rows omit audit-selector calls, invalid-attempt overhead, broader transfer blocks, and activation replay. The screened-item denominator can greatly exceed the retained constructed-cohort denominator. The manifest planner must calculate exact raw requests, cache-unique requests, token envelopes, and projected physical resource use before launch.

### 16.4 Budget authorization

No GPU-hour or monetary ceiling has been newly authorized in this request. `budget_gpu_hours`, `budget_cpu_hours`, `storage_limit_bytes`, and any paid-API cap are required in the execution manifest before confirmation. Retain at least 15% compute and 20% storage headroom as proposed reserves. Do not infer that the allocation in an older specification authorizes this study.

Within an approved allocation, cut optional controller, broad horizons, and extra topologies before cutting paired roots, the wrong-minority mirror, source-item independence, or measurement audits. Do not allocate more seeds merely because recruiting independently eligible questions is inconvenient.

---
## 17. Implementation contract, schemas, and reproducibility

### 17.1 Repository integration

Implement in the existing `agents_scaling` project where available, without changing historical artifacts in place. Proposed namespace: `agents_scaling.nextdyn`. All names below are interfaces to implement or map, not a claim that the modules already exist.

```text
configs/nextdyn/
  study.yaml
  models.lock.json
  datasets.lock.json
  prompts.lock.json
  analysis.yaml
  compute.yaml
  figure_manifest.yaml
src/agents_scaling/nextdyn/
  config.py
  inventory.py
  ids.py
  banks.py
  construct.py
  protocols.py
  prompts.py
  inference.py
  checker.py
  selectors.py
  telemetry.py
  capture.py
  analysis.py
  power.py
  release.py
runs/task_conditioned_dynamics_v1/
  frozen_manifest.json
  amendments.jsonl
  plan.parquet
  source_items.parquet
  item_splits.parquet
  root_teams.parquet
  requests.parquet
  decisions.parquet
  transmissions.parquet
  selections.parquet
  tool_events.parquet
  activations_manifest.parquet
  grades_private.parquet
  derived/
  blobs/
  tests/
  figures/
  reports/
```

The private grading table must not be mounted into the solver, selector, or deployment-policy service. A label-free export from the constructor supplies only opaque identifiers, drafts, questions, and authorized graphs.

Compatibility labels may map `PRIVATE` to a versioned `R_PRIVATE`, `PEER` to a versioned `C_FULL`, and `FRESH` to a versioned `FRESH_N`. A shared label does not imply equivalent prompts or memory. Store both `display_label` and the complete `protocol_hash`. [LEGACY]

### 17.2 Identity and caching

Use immutable composite keys. At minimum:

```text
(study_id, cell_kind, source_item_id, item_variant_id, root_team_id,
 protocol_hash, receiver_model_hash, budget_hash, replicate_id,
 round, agent_slot, phase)
```

`cell_kind` distinguishes `difficulty_bank`, `donor_bank`, `natural_root`, `revision`, `receiver_plan`, `receiver_decision`, `selection`, and `activation_replay`.

Use separate concepts:

- `decision_id`: scientific observation and intended condition.
- `request_hash`: exact bytes/token IDs, model/adapter, all decoding settings, and RNG stream.
- `generation_uid`: physical sample identity, referenced by every intentional reuse.
- `content_hash`: output text identity, which may coincide for different independent samples.

Never use `episode_id` alone as a globally unique join key. Reject duplicate logical decisions, conflicting cached responses, null `source_item_id`, and any bank-to-episode overwrite.

For a paired fork, use a common random-number schedule keyed on source item, root, replicate, round, agent slot, and phase. Actual request identity also includes model/prompt/parent/protocol content, so different treatments cannot silently retrieve each other's descendants. Identical requests under explicitly equivalent conditions may reuse a physical generation and must be reported as reuse, not independent replication.

Randomize condition dispatch order and distribute conditions across serving replicas to avoid confounding treatment with backend load or time. Record replica and batch metadata. Common random seeds do not guarantee bit-identical stochastic outcomes across different kernels; backend parity is a separate check.

### 17.3 Minimum table schemas

| Table | Required fields beyond IDs |
|---|---|
| `source_items` | source dataset/revision, source split, original ID, normalized text hash, variant parent, subject/format, option map, prior-use status, sampling weight, split role |
| `banks` | bank role, draw index, donor model/budget, seed, generation UID, raw output reference, parse/grade status, adaptive-extension stage |
| `root_teams` | ordered donor IDs, independent-generation count, focal slot, construction rule/hash, label-free export hash; oracle composition stored privately |
| `requests` | exact messages/token IDs, prompt hash, model/tokenizer/backend hashes, sampling params, budget, seed, visible parent IDs, dispatch/finish time, device/replica |
| `decisions` | native reasoning reference, public JSON fields, final-answer canonical ID, token offsets, confidence, answer logprobs, validity, termination reason, parent lineage, raw response reference |
| `transmissions` | sender/receiver slots, source and destination rounds, source decision ID, packet fields/hash, transmitted token count, truncation/filler flags, declared graph edge |
| `grades_private` | decision/candidate ID, evaluator version, semantic correctness C/W/U, reason, extracted value, original gold reference, audit status |
| `selections` | selector/version, candidate pool/hash, scope, selected existing ID, tie class IDs, fallback status, cost; grade joined only afterward |
| `tool_events` | E2 packet and step IDs, access condition, request/response, validation status, query count, elapsed time; no hidden oracle payload |
| `activations_manifest` | model/layer/anchor, original token offset, dtype/shape, storage hash, replay parity, missing reason |
| `metrics` | estimand version, source population, weighting, denominator, missingness convention, outcome/time horizon, estimate/interval, multiplicity family |

Do not store only rounded confidence or a selected answer letter. Preserve all original output and mapping data needed for later audit.

### 17.4 Example configuration to compile and lock

The following is valid YAML describing proposed defaults. Null lock/budget fields are **intentional launch blockers**, not values to fill with guesses. The planner materializes a complete final manifest from the source inventories, pilot, and approval record.

```yaml
study:
  id: task_conditioned_dynamics_v1
  schema_version: 1
  spec_version: "1.0"
  seed: 20260915
  mode: pilot_only
  confirmation_authorized: false
  git_commit: null
  container_digests: null
  source_inventory_hash: null
  prior_use_inventory_hash: null

models:
  primary:
    model_id: Qwen/Qwen3-8B
    revision: null
    tokenizer_revision: null
    adapter: native_thinking
  scale:
    model_id: Qwen/Qwen3-32B
    revision: null
    tokenizer_revision: null
    adapter: native_thinking
  family_transfer:
    model_id: meta-llama/Llama-3.1-8B-Instruct
    revision: null
    tokenizer_revision: null
    adapter: total_output_envelope
    enabled: false

sampling:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0

budgets:
  thinking_tokens_main: 2048
  final_phase_tokens_main: 1024
  receiver_thinking_tokens: [1024, 4096]
  served_window_tokens: 32768
  context_reserve_tokens: 256
  campaign_gpu_hours: null
  campaign_cpu_hours: null
  campaign_api_currency_limit: null
  storage_limit_bytes: null
  compute_reserve_fraction: 0.15
  storage_reserve_fraction: 0.20

cohorts:
  families: [MATH, MMLU_PRO, TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE]
  dataset_lock_hash: null
  pilot_items_per_family: 60
  pilot_eligibility_extensions: [120, 180]
  natural_target_per_family: 200
  constructed_target_per_family: 300
  paired_compositions: [C1_COHERENT, C4_COHERENT]
  allow_natural_constructed_item_overlap: true
  allow_pilot_confirmation_overlap: false
  donor_initial_draws: 32
  donor_max_draws: 64
  difficulty_draws: 16
  natural_root_draws: 5

protocol:
  n_agents: 5
  root_round: 0
  revision_rounds: 4
  score_rounds: [0, 1, 2, 3, 4]
  protocols: [PRIVATE, PEER, FRESH]
  synchronous: true
  early_stop_on_consensus: false
  own_previous_public_draft: true
  visible_history: previous_round_only
  native_thinking_shared: false
  confidence_shared: false
  final_phase_constrained_decoding: false
  public_packet_token_cap: 512
  full_archive_stored_externally: true
  prompt_lock_hash: null

selection:
  primary: current_round_plurality
  pool_scope: terminal_only
  tie_break: item_class_hash
  synthesis_allowed: false
  gold_access: false
  secondary: [random_valid, confidence_weighted, pointwise_audit, oracle_analysis]
  audit_model_id: Qwen/Qwen3-32B
  audit_model_revision: null
  audit_thinking_tokens: 1024
  audit_final_tokens: 256
  audit_failure_fallback: current_round_plurality
  audit_prompt_hash: null
  archive_audit_enabled: false

experiments:
  E0: {enabled: true}
  E1: {enabled: true}
  E2a: {enabled: true, items_per_task: 100, tasks: [MATH, MMLU_PRO], replicates: 2}
  E2b:
    enabled: true
    source_problems: 300
    total_displayed_operations: 24
    chain_depths: [4, 12, 24]
    endorsement: [correct_minority, correct_majority, hidden]
    checker_access: [false, true]
    initial_draft: [correct, wrong]
    replicates: 2
    max_checks: 4
    generation_phases: 2
  E3a: {enabled: true, target_items_per_task: 150, natural_random_protection: true}
  E3b: {enabled: false}
  E3c: {enabled: false}
  E4: {enabled: false}

instrumentation:
  enabled: true
  target_items_per_task: 100
  capture_all_root_agents: true
  anchors: [PROMPT_END, PRE_FINAL, ANSWER_END]
  relative_layers: [0.25, 0.50, 0.75]
  mode: read_only_replay
  intervention_enabled: false

analysis:
  primary_hypotheses: [H1, H2, H3, H4, H5, H6, H7, H8]
  primary_endpoint_round: 4
  familywise_alpha: 0.05
  multiplicity: holm
  bootstrap_replicates: 10000
  cluster: source_item_id
  paired: true
  task_summary_weights: equal
  semantic_correctness_states: [C, W, U]
  missingness_bounds_required: true
  power_target: 0.80
  confirmatory_sample_size_lock: null
  analysis_lock_hash: null

failure_policy:
  transient_retries: 2
  retry_same_request_and_seed: true
  model_output_repair_calls: 0
  wrong_answer_retry: false
  context_overflow: typed_failure
  unresolved_semantics: unknown
  null_item_ids: fatal
  duplicate_logical_keys: fatal
  hidden_label_leakage: fatal
```

The compiler must reject unknown fields and incompatible settings. For example, a native thinking-budget field on the Llama adapter is an error; an oracle selector in the deployed-policy list is an error; confirmation with null locks or no authorized resource ceiling is an error.

### 17.5 Required command-line lifecycle

Example interface, to be implemented or mapped onto the current repository:

```bash
# Inventory and validate source artifacts. Does not launch new solver inference.
python -m agents_scaling.nextdyn inventory --config configs/nextdyn/study.yaml
python -m agents_scaling.nextdyn audit-legacy --config configs/nextdyn/study.yaml

# Resolve/cache exact model and data versions; render and test all requests.
python -m agents_scaling.nextdyn lock-inputs --config configs/nextdyn/study.yaml
python -m agents_scaling.nextdyn validate --config configs/nextdyn/study.yaml --smoke

# Pilot only, after its own small resource authorization.
python -m agents_scaling.nextdyn pilot --config configs/nextdyn/study.yaml
python -m agents_scaling.nextdyn power --pilot-root runs/task_conditioned_dynamics_v1/pilot
python -m agents_scaling.nextdyn plan --config configs/nextdyn/study.yaml --dry-run

# Confirmation freeze requires explicit resource, sample-size, and analysis locks.
python -m agents_scaling.nextdyn freeze --config configs/nextdyn/study.yaml
python -m agents_scaling.nextdyn run --manifest runs/task_conditioned_dynamics_v1/frozen_manifest.json

# Idempotent read-only derivation and release validation.
python -m agents_scaling.nextdyn materialize --run-root runs/task_conditioned_dynamics_v1
python -m agents_scaling.nextdyn capture --run-root runs/task_conditioned_dynamics_v1
python -m agents_scaling.nextdyn analyze --run-root runs/task_conditioned_dynamics_v1 --locked
python -m agents_scaling.nextdyn release-check --run-root runs/task_conditioned_dynamics_v1
```

Every command is resumable and reports completed, failed, skipped, blocked, and cache-reused work separately. A planned module is not counted as executed because a directory or all-null output table exists.

---

## 18. Required tests and technical release gates

### 18.1 Identity, splits, and leakage

```text
test_source_item_id_never_null
test_bank_role_cannot_alias_private_episode
test_composite_decision_key_unique
test_independent_draws_have_distinct_generation_uids
test_same_content_does_not_imply_same_sample
test_request_cache_invalidates_on_model_prompt_seed_or_parent_change
test_all_variants_of_source_item_share_split_and_cluster
test_prior_use_inventory_marks_unknown_not_untouched
test_solver_export_contains_no_gold_or_construction_correctness
test_candidate_ids_and_slot_labels_do_not_encode_correctness
test_selector_service_cannot_read_private_grades
test_hidden_code_tests_absent_from_generator_and_selection_sandbox
```

### 18.2 Protocol and construction

```text
test_paired_protocols_have_byte_identical_roots
test_c1_c4_exact_occupancy_and_shared_wrong_class
test_no_donor_record_duplicated_within_team
test_peer_messages_use_only_previous_round
test_private_has_no_peer_content
test_fresh_has_no_own_or_peer_history
test_four_revisions_produce_five_observed_states
test_all_agents_update_after_unanimity
test_c11_c01_c10_c00_have_20_16_16_12_edges
test_isolated_focal_does_not_isolate_other_agents
test_focal_slot_is_fixed_after_root
test_ring_indegree_and_outdegree_are_two
test_n3_ring_equals_all_to_all_is_not_duplicate_treatment
test_r4_prefix_of_r8_is_identical_when_prompts_are_horizon_free
test_missing_packet_padding_has_no_task_content
```

### 18.3 Measurement, selectors, and bounds

```text
test_c_w_u_counts_sum_to_n
test_three_state_transition_identity
test_validity_change_is_not_answer_switch
test_wrong_unanimity_requires_one_shared_wrong_class
test_historical_coverage_is_nondecreasing
test_active_coverage_is_bounded_by_historical_coverage
test_current_selector_accuracy_not_above_current_oracle_on_same_pool
test_failure_decomposition_on_fully_graded_fixture
test_lower_upper_coverage_bounds_include_all_feasible_labels
test_passk_uses_fixed_items_and_correct_bank_role
test_passk_prefix_oracle_and_vote_use_same_actual_candidates
test_tie_break_is_invariant_to_protocol_and_agent_order
test_selector_returns_existing_permitted_candidate
test_missing_or_unfitted_selector_is_unexecuted_not_zero_accuracy
test_within_item_auc_requires_mixed_candidate_groups
test_confidence_not_imputed_as_model_statement
test_option_shuffle_preserves_canonical_answer_identity
```

### 18.4 Receiver and checker

```text
test_mod97_evaluator_independent_of_generator
test_all_depths_show_24_operations
test_answer_dependency_depth_exact
test_wrong_trace_has_exactly_one_violated_local_step
test_wrong_and_correct_trace_final_answers_differ
test_support_conditions_have_identical_packet_bodies
test_support_metadata_is_not_counted_as_independent_generations
test_checker_only_reads_named_public_packet_and_step
test_checker_does_not_reveal_gold_or_recompute_final_answer
test_disabled_checker_never_returns_validity
test_at_most_four_check_requests_and_fixed_duplicate_policy
test_two_phase_thinking_caps_sum_to_declared_episode_budget
test_14400_receiver_episodes_are_28800_generation_calls_per_model
```

### 18.5 Serving, neural capture, and resources

```text
test_reasoning_cap_is_enforced_at_declared_boundary
test_final_phase_has_its_reserved_cap
test_native_and_public_reasoning_fields_not_confused
test_pinned_backend_response_fields_supported
test_worst_case_context_fits_every_compared_model
test_infrastructure_retry_keeps_exact_request_and_seed
test_malformed_model_output_is_not_stochastically_retried
test_activation_layer_and_anchor_offsets_valid
test_teacher_forced_replay_uses_exact_stored_token_ids
test_all_root_agent_captures_exist_or_are_explicitly_missing
test_zero_variance_spectral_metrics_are_undefined
test_geometry_rank_not_above_n_minus_one
test_read_only_capture_cannot_change_generation
test_attributed_gpu_time_conserves_device_totals
test_campaign_and_per_system_cost_are_separate
test_budget_admission_reserves_selection_cost
test_stop_is_available_in_every_action_policy
```

### 18.6 Technical gates before confirmation

Require 100% success for identity, leakage, graph, root-pairing, synthetic-evaluator, and selector-membership fixtures. On the pilot, proposed operational targets are less than 1% unresolved infrastructure failures and less than 2% output-format failures, with less than one percentage point differential infrastructure failure between the primary arms. These are design-quality targets, not grounds to delete failed confirmation examples.

If a pilot fails, repair the implementation or amend the adapter, then rerun on pilot/development only. If confirmation later shows a failure, preserve affected records, halt the affected block, and issue an amendment/reproduction decision. Never rewrite a frozen run to make its protocol look homogeneous.

---

## 19. Execution stages and stopping rules

| Stage | Work | Required output before proceeding |
|---|---|---|
| **S0: source inventory** | Locate legacy raw inputs; document published comparisons; inventory prior item use and exact datasets. | Source hashes, availability report, overlap inventory, E0 blockers. |
| **S1: engineering** | Implement schemas, seed/caching rules, renderer, graders, synchronous protocols, local checker, cost accounting. | Passing fixtures and a complete label-free request audit. |
| **S2: pilot** | 60 items/task, eligible pilot constructions, 30 synthetic problems, throughput and activation smoke tests. | Eligibility, variance, validity, cost, storage, and adapter-parity reports. |
| **S3: freeze** | Choose sample sizes from declared power procedure; set resource ceiling; lock all prompts/models/data/analysis. | Signed frozen manifest, primary-test registry, hash of exact run plan. |
| **S4: core execution** | E1 and E2; E3a after its root panel is sealed; continuous operational validation. | Complete planned outcomes or explicit failures, not only successful cells. |
| **S5: transfer** | Enabled E3b/E3c blocks, without choosing levels from favorable confirmation outcomes. | Prespecified held-out contrast/prediction tables. |
| **S6: analysis** | Materialize outcomes, paired contrasts, bounds, calibration, predictive tests, and figures. | Reproducible locked analysis and claim ledger. |
| **S7: optional extension/release** | E4 or X1/X2 only under separate approvals; package core whether signs are positive, negative, or null. | Release report, provenance, and all failed predictions. |

### 19.1 Exact pilot dispatch subset

Before the full pilot, run an independently authorized throughput smoke test of 20 solver calls per model/adapter and five complete synthetic receiver episodes. The planner uses those costs to approve the larger pilot; it must not silently dispatch the entire confirmation grid in pilot mode.

| Pilot component | Fixed initial scope |
|---|---|
| Bank/eligibility inventory | 60 source questions per primary task, with separate 16-draw difficulty and 32-draw donor banks; use the declared extension rule only for eligibility. |
| Natural trajectories | First 20 hash-selected pilot questions per task, five fresh roots, PRIVATE/PEER/FRESH through r=4. |
| Constructed trajectories | Up to 30 paired-eligible pilot questions per task, both C1/C4 roots, all three protocols through r=4. |
| Directional graphs | First 10 paired-eligible pilot questions per task, all four directed conditions through r=4, reusing exact reciprocal traces where permitted. |
| Natural receiver assay | First 10 eligible pilot questions each from MATH and MMLU-Pro; full support/draft/budget grid, one seed, Qwen3-8B. |
| Synthetic receiver assay | 30 pilot source problems, ten per depth; full support/checker/draft/budget grid, one seed, Qwen3-8B. |
| 32B adapter check | A fixed 12-problem synthetic subset, four per depth, with the same factorial and one seed; not a powered scale study. |
| Capture check | All agents at all required anchors on the first ten eligible pilot questions per task under PRIVATE and PEER. |

These subsets estimate feasibility and variance; they do not choose the scientifically favorable condition. Eligibility extensions to 120/180 pilot source questions are allowed only by the Section 14 rule. The pilot report must state how uncertainty in its variance estimates affects the proposed confirmatory allocation.

No fixed calendar duration is asserted before profiling the actual serving setup. Run E2 and E1 in parallel once their inputs are frozen, because their construction and execution do not depend on a positive result from the other.

During confirmation, operational dashboards may show completion, failures, token counts, cache reuse, and pairing. Do not expose arm-labeled accuracy or layer/probe rankings to people changing code or selecting configurations before the relevant analysis freeze.

**Stop for integrity, feasibility, or resource reasons, not the desired scientific result.** Low eligibility can require a narrower study; weak or reversed effects are legitimate findings. Optional mechanistic or policy claims may be abandoned without abandoning the core characterization.

---

## 20. Required tables, figures, and claim checks

### 20.1 Main figure plan

| Figure | Required content | Conclusion it can support |
|---|---|---|
| **F1: initial-state and task decomposition** | Natural root distributions; natural versus standardized protocol effects; overlap/sample sizes. | Composition contribution versus residual task heterogeneity. |
| **F2: minority trajectories** | C1 and C4 trajectories for individual accuracy, active coverage, and delivered accuracy; continuous survival and validity alongside. | Preservation, propagation, and selection are different outcomes. |
| **F3: receiver mechanism** | Fixed-packet endorsement effects crossed with checker access and budget; synthetic depth and natural-packet validation. | Whether actual checking/reasoning changes endorsement sensitivity. |
| **F4: directed channels** | Four incoming/outgoing cells, focal retention, nonfocal adoption, and wrong-minority mirror. | Influence and vulnerability can be manipulated separately. |
| **F5: transfer and prediction** | Held-out task/model/topology contrasts and prediction errors, including negative transfer. | Scope and usefulness of a proposed explanation. |
| **F6: resource/selection frontier** | Independent sampling coverage; deployable vote/audit; communication/private methods with full cost. | Whether available correctness reaches decisions at useful cost. |

Neural geometry is supplementary by default. Promote it only under the incremental-prediction gate. E4 gets a main figure only if there is an honestly held-out practical result.

### 20.2 Figure metadata

Every figure must have a machine-readable manifest with source hashes, filters, item counts, root/seed counts, task weights, uncertainty procedure, missingness convention, protocol and selector hashes, budget definition, and plot script revision.

Captions must specify whether the population is natural, constructed, mirrored/common-support, or a restricted transfer subset. Use separate panels/figures rather than a pooled mean that hides a sign reversal. Show effect sizes and uncertainty, not only significance stars.

### 20.3 Claim ledger

Create `claim_ledger.csv` with:

```text
claim_id, statement, evidence_type, experiment, estimand_id,
source_population, model_scope, task_scope, primary_or_secondary,
estimate, uncertainty, multiplicity_status, missingness_sensitivity,
audit_status, supporting_artifact_hash, verdict
```

Allowed verdicts: `SUPPORTED_WITHIN_SCOPE`, `NARROWED`, `NOT_SUPPORTED`, `INCONCLUSIVE`, `NOT_ESTIMABLE`, `NOT_RUN`, `REFUTED_BY_CONTROL`.

Examples of prohibited upgrades:

- Coverage loss → “the model forgot the answer permanently.”
- Increased agreement → “truth spread.”
- Current correct candidate exists → “the system can select it.”
- Small correct/wrong confidence difference → “deployable minority detector.”
- Lower effective rank → “smaller distances.”
- Outcome-conditioned geometry association → “causal mechanism.”
- Two model sizes → “universal scaling law.”
- Candidate exists in an archive → “recovery is free.”
- Oracle minority protection → “a practical controller.”

---

## 21. Optional extensions, explicitly not part of the core

### X1. State-matched recovery and correctability

Only prioritize this after the core if active coverage loss persists and archive-aware selection does not trivially recover the effect under acceptable cost.

From identical recorded contested or wrong-consensus states, randomize continuing PEER, PRIVATE, FRESH, or archive-aware verification across additional budgets. Separately compare states produced by earlier randomized PEER/PRIVATE protocols using the **same** recovery policy.

Define `V_rho(s,b)=Pr(correct selected answer after policy rho from state s with budget b)`. Label fixed-policy value separately from an optimized policy-class envelope. STOP and old-candidate retention must be available when claiming an at-most-budget envelope; forced continuation can be worse without demonstrating reduced attainable value.

Do not compare only naturally occurring wrong consensuses across earlier treatment arms and call that the causal effect of earlier communication. Such subsets can contain different questions/states because the treatment changed which states occurred. [ARD, §3.3 motivates the question, not state-matched identification.]

Maintain context scope, archive scope, selector, and root-reset policy as independent factors. A root-reset operation that discards all post-treatment state should have identical inputs across earlier arms and can reuse an exact independent generation; it is an intervention on accessible state, not evidence of erased model parameters. [LEGACY]

### X2. Answer-cardinality and wrong-support concentration

Run0 proposes that answer-space structure may explain agreement-calibration differences; it does not causally establish that account. [RUN0, §12.1, p. 68]

Within one multiple-choice task family, construct paired presentations retaining the gold option and a fixed salient wrong option, with additional incorrect options selected under a frozen rule. Regenerate roots because the task presentation changed, then measure initial wrong-answer concentration, calibration, and communication effects. Use canonical option identities and source-item clustering.

The intervention changes distractor content and difficulty as well as cardinality; do not claim a pure number-of-options effect. A cleaner separate support-count manipulation is E2, where packet text and candidate set stay fixed. If a cardinality result is central, add matched-length/distractor-quality controls and preregister the panel before running it.

### X3. Code round dynamics

Only after E0's grouping audit, run a small natural code panel under the same three protocols and candidate-only execution-informed selection. Preserve code-specific final-output budgets rather than forcing a 1,024-token QA envelope onto long programs. Report this as a separate adapter/task regime.

Do not use generated tests as official ground truth. Evaluate on held-out official tests after selection, include sandbox/test-generation costs, and retain ambiguous behavioral grouping as a measurement limitation.

### X4. Neural intervention

Only if prospective geometry adds reproducible predictive value, design a separate activation edit with no-op, random-direction, matched-norm, and donor-specificity controls; regenerate the full downstream suffix. This requires a new intervention specification. Read-only replay and regression adjustment are insufficient to claim neural causality.

---

## 22. Definition of done and immediate work order

### 22.1 Minimum complete core

A complete core release contains:

1. Source-separated E0 reproduction/audit status and a documented prior-use inventory.
2. E1 natural and paired C1/C4 challenges under PRIVATE, PEER, FRESH, with exact shared roots and all planned round checkpoints.
3. E2 fixed-content endorsement comparisons and an actual local-checker intervention with budget held to the declared episode envelope.
4. E3a both directional channels and the incorrect-minority mirror, not only oracle-protected correct agents.
5. Distinct individual accuracy, active/historical coverage, delivery, validity, and full resource outcomes.
6. Item-paired uncertainty, the complete primary hypothesis family, missingness bounds, and an explicit list of unestimable/failed results.
7. Complete root/lineage provenance and read-only capture status, even if neural prediction is ultimately null or not pursued.

A generalized claim additionally requires the appropriate enabled transfer block. A useful-policy claim additionally requires E4's held-out test. A recovery-cost claim requires X1. None is implied by finishing the core.

### 22.2 Immediate coding order

**First implement and run:** inventory plus key/split/grading audits; synchronous PRIVATE/PEER/FRESH fixtures; 16/32-draw bank separation and C1/C4 constructor; synthetic generator and checker fixtures; a small authorized pilot with token/resource telemetry.

**Then freeze and dispatch:** powered E1/E2 core, followed by the E3a channel intervention on its preselected shared roots. Preserve enough compute for the wrong-minority mirror and at least one relevant transfer contrast before funding a large optional axis sweep.

**Do not start by:** building a learned controller, searching dozens of prompting variants, fitting a high-dimensional geometry probe to a tiny contested panel, or rerunning all older configurations.

### 22.3 Research success condition

The study succeeds scientifically when it can reliably distinguish among these alternatives:

- apparent task dependence is mostly initial answer composition;
- comparable disagreements have different receiver/communication dynamics;
- selector or validity defects account for a substantial part of the apparent difference;
- specified channel or evidence interventions change dynamics in the predicted direction;
- a proposed mechanism does not survive its control or does not transfer.

A null with sufficient precision, a narrower task-specific principle, or a falsified mechanism is a valid outcome. The spec does not require communication to harm, protection to help, or neural geometry to explain correctness.

---

## 23. Sources and provenance

### 23.1 Supplied experimental reports

**[ARD]** *What Communication Does to a Team of Agents: A Characterization of Multi-Agent Round Dynamics, with Emphasis on Correct Minorities*. ARD36, repaired `tables_v2`; report generated September 13, 2026. Supplied file: `091326_AgentRoundDynamics_Report.pdf` (12 pages).

- Repaired comparison and integrity: §§2–3, pp. 2–4.
- Wrong consensus and sampling comparison: §§3.3–4, pp. 5–6.
- Copying, geometry, minority measurements: §§5–8, pp. 7–9.
- Receiver assay and limitations: §§9–13, pp. 9–12.
- SHA-256: `47c95490e1620835f4af5669e04af8387d6e1d8e5ab43f1f4184678595e2bdb2`.

**[ATLAS]** *A Scaffolding Atlas: How Agent-System Configuration Reshapes Model Behaviour, and What Changes Inside the Model*. Exploratory study_v4 report, September 7, 2026. Supplied file: `ScalingAgents_Atlas-1.pdf` (25 pages).

- Design/caveats: §§2–6, pp. 2–4.
- Conditional behavior, selection, and calibration: §7, pp. 5–12.
- Geometry and limits of mechanism: §§8–10, pp. 12–16.
- Follow-ups, defects, statistical status: §§12–13/Appendices, pp. 18–20.
- SHA-256: `415575547faa00494bf60343e264e2475feea380021ef67c38dd2ca9f082f6f3`.

**[RUN0]** *Capacity, Reasoning, Prompting and Coordination in Small Multi-Agent Language Model Systems: A Factorial Sweep of Accuracy and Calibration*. `full_sweep_v1`, compiled July 30, 2026. Supplied file: `Scaling_Agents_Run0_Methods.pdf` (237 pages).

- Protocol and probe timing: §§3–4, pp. 13–25.
- Context-sharing findings: §6.4, p. 41; §7.3, pp. 48–49.
- Task-conditioned interpretation: §12, pp. 68–72.
- Measurement/protocol limitations and status: §13, pp. 73–75; Appendix F.5, p. 237.
- SHA-256: `cdeb2c2031390dbb1ff14120ae963a1aa83f23b558b6ff9a84666f02f9aa2735`.

These hashes identify the PDFs inspected for this specification, not raw-run checksums. Raw data require their own locked inventory.

**[LEGACY]** Earlier project file `agents_scaling_causal_mechanistic_continuation_value_experiment_spec_v1_0.md`, retrieved from the user's prior project files. Referenced for frozen roots, candidate-only selection, explicit context/archive scopes, action naming, and implementation audits. This document does **not** inherit its mandatory benefit-then-harm or neural-intervention gates. No raw implementation or mounted copy of that earlier spec was assumed from its search reference.

### 23.2 Key literature checked for positioning

These are focused precedents, not a fresh exhaustive novelty review. Bibliographic identity and relevant abstracts/pages were checked on 2026-09-15. New hypotheses and operating thresholds in this spec are proposed here, not attributed to these papers.

**[L1]** *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* (2024). Motivation for separating sampling coverage from usable selection.  
`https://arxiv.org/abs/2407.21787`

**[L2]** *Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness* (2025). Direct precedent for conditional rather than universal debate benefits.  
`https://arxiv.org/abs/2505.22960`

**[L3]** *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?* (2025). Debate versus voting and a stochastic-process perspective.  
`https://arxiv.org/abs/2508.17536`

**[L4]** *Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems* (EMNLP 2025). Precedent for experimentally analyzing error and useful-information propagation across communication structures.  
`https://aclanthology.org/2025.emnlp-main.623/`

**[L5]** *The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate* (2026 preprint). Closest precedent for comparing debate with private revision and tracking correct-answer availability.  
`https://arxiv.org/abs/2605.00914`

**[L6]** *Confidence Improves Self-Consistency in LLMs* (Findings of ACL 2025). Confidence-based aggregation and the need to evaluate within-question discrimination and actual selection utility.  
`https://aclanthology.org/2025.findings-acl.1030/`

### 23.3 Official dataset and implementation resources

**[D1]** MATH-500 dataset card.  
`https://huggingface.co/datasets/HuggingFaceH4/MATH-500`

**[D2]** Original MATH project, dataset loaders, and evaluation resources.  
`https://github.com/hendrycks/math`

**[D3]** MMLU-Pro official project and evaluation resources.  
`https://github.com/TIGER-AI-Lab/MMLU-Pro`

**[D4]** TruthfulQA official repository. Its documented newer binary task is distinct from legacy MC1; preserve the intended source revision and scoring adaptation.  
`https://github.com/sylinrl/TruthfulQA`

**[D5]** BigCodeBench official project and evaluator.  
`https://github.com/bigcode-project/bigcodebench`

**[D6]** Qwen3 model cards, native thinking/sampling guidance, and model identities. The study's short caps and public-state policy are experimental choices.  
`https://huggingface.co/Qwen/Qwen3-8B`  
`https://huggingface.co/Qwen/Qwen3-32B`

**[D7]** Llama-3.1-8B-Instruct official model card and access/license conditions.  
`https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`

**[D8]** vLLM official reasoning-output and thinking-budget documentation. Pin and test the actual installed version; the mutable documentation URL is not a software lock.  
`https://docs.vllm.ai/en/latest/features/reasoning_outputs/`

**Final operational rule:** preserve the existing evidence, make the new interventions explicit, and let the measured conditional dynamics determine the paper's conclusion.
