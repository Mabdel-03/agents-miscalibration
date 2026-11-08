# Remaining experiments: compute allocation, discovery, preservation, and selection

**Project:** Compute-Aware Scaling of Agentic Inference  
**Specification date:** 2026-09-25  
**Status:** Proposed implementation plan. No new inference has been run for this specification.  
**Primary deliverable:** A small, reproducible set of experiments that predicts allocation effects and explains their mechanisms.  
**Execution principle:** Generate independent candidate banks once, fork immutable roots, stream grading, and reuse sealed pools for analysis and selection.

## 1. Objective and scope

The paper should answer:

> Given a model, a problem distribution, and an inference budget, which allocation of reasoning depth, independent attempts, and interaction rounds most reliably returns a correct answer, and which generation, preservation, or selection bottleneck explains that choice?

The remaining work should establish **predictive allocation principles**, rather than add a large descriptive grid. The central tests are whether independently estimated depth-response curves predict mixed allocations; whether answer competition explains selection beyond shallow success rate; and whether interaction transitions predict discovery and extinction on held-out continuations.

A complete adaptive orchestrator is optional. The core paper can make a strong contribution through prospective prediction of allocation outcomes without claiming that five samples identify a universally optimal system.

### Recommended order

| Priority | Work package | Required for |
|---|---|---|
| P0 | E0: audit, reconcile, and reanalyze existing logs | Every claim |
| P1 | E1: complete mixed independent depth–width comparisons | Claim that one long trajectory is preferable to distributing compute |
| P1 | E2: repeated response curves and fresh-draw prediction | Predictive depth–width theory |
| P1 | E5: prospective unseen-item predictions | Generalizable regime and allocation claims |
| P2 | E3: narrow interaction grid plus controlled shared-root continuations | Discovery/preservation mechanism and communication claims |
| P2 | E4: frozen generation pools crossed with selectors | Generation–selection distinction and archive utility |
| P2 | E6: natural stopping and measured total-cost sensitivity | Claims beyond enforced reasoning allowances |
| P3 | E7: second model family and executable coding | Broader generality |
| Optional | E8: budgeted five-sample policy and dynamic orchestration baseline | Deployment and adaptive orchestration claims |

**Fast path:** E0 + legacy E1 + E2 on the smaller panel described below, then E5 before strengthening generalization claims. Start E3/E4 implementation concurrently; their inference consumes completed banks.

**Recommended paper package:** E0–E6. Add the narrow E7 replication if the paper claims principles across model families or beyond reasoning benchmarks.

**Do not put on the critical path:** activation extraction, extensive topology search, many frontier models, learned communication graphs, comprehensive tool-use agents, or a new orchestration framework.

## 2. Source basis and current evidence

This specification uses the pasted assessment, the [live Overleaf main.tex](https://www.overleaf.com/project/6aa8506b836c121c0151c228) inspected on 2026-09-25, and the two distinct historical reports. The duplicate attachment pairs are byte-identical copies, not replications. Underlying current-study logs and repository code were not available in this task; E0 must verify their contents before reuse.

### 2.1 Live manuscript

The inspected manuscript specifies:

- A = 1,024 reasoning tokens.
- Configuration (n, R, a), with **R including initialization**, and nominal reasoning budget C = nRa.
- 96 legacy main-panel problems: 24 each from AIME 2025/2026, HMMT February 2025/2026, Omni-MATH difficulty ≥7, and GPQA-Diamond.
- 32 disjoint development problems, eight per family.
- 25 shallow independent draws per main problem; one draw at each deeper allowance.
- Separate THINK and ANSWER calls. THINK is forced to the assigned allowance; ANSWER has a 4,096-token cap.
- Manuscript model identifier: qwen/qwen3.8-27b; temperature 1.0, top-p 0.95, top-k 20. This identifier remains explicitly flagged for verification against run metadata.
- Per-problem provider assignment, with fallback and response caching disabled.
- Math-Verify grading and exact shuffled-option matching for GPQA; invalid answers are incorrect, grading timeouts are unknown.
- 94 fully graded shallow banks and 93 problems in one four-agent complete-case analysis.
- The Theory section is still a placeholder in the inspected version.

Reported results motivate the experiments but are not acceptance targets: 60.3% shallow coverage versus 43.3% voting at 25 samples; 78.5% deep accuracy versus 58.0% shallow oracle coverage at 16A. The Introduction instead says 79.2% deep accuracy. Resolve that discrepancy from one authoritative cohort and aggregation implementation.

### 2.2 Historical studies

| Source | Reusable contribution | Boundary |
|---|---|---|
| AgentRoundDynamics repaired report, Sep 13 | Shared-root design; round transitions; candidate retention; existing receiver probes | Five Qwen3-8B agents, three revisions, MATH-500/MMLU-Pro; separate study |
| ScalingAgents Atlas v4 | Selector/cost infrastructure lessons; executable coding; cached candidate pools | Different models, tasks, controls, and actual budget utilization |
| Pasted assessment | Theory-to-experiment map and proposed priorities | Secondary analysis, not a raw-data audit |

Do not pool these studies with the forced-budget experiment. The older reports describe real failure modes: episode-only joins changed a primary result; nominal independent sample settings aliased cached requests; ungradeable outputs were conflated with errors; several selectors never ran. Incorporate these as E0 checks.

Reuse the already executed 1,280-probe receiver assay only as a separate supporting analysis. The historical constructed-root challenge cohort did not run. Do not regenerate either merely because its description appears in an old plan.

## 3. Scientific contract

### 3.1 Quantities

For problem x and allowance a:

\[
p_x(a)=\Pr(\text{one independent call at allowance }a\text{ is correct}),\quad
h_x(a)=-\log[1-p_x(a)].
\]

For independent calls conditional on x, a fixed model, and a fixed protocol:

\[
G_x(a_1,\ldots,a_n)=1-\prod_{i=1}^n[1-p_x(a_i)].
\]

Estimate this **per problem, then average**. Do not substitute the benchmark-average success probability into the product.

For a realized system, store indicators for delivered correctness S, current coverage H_cur, and archive coverage H_arc. When selection returns an existing current-pool candidate:

\[
1-S=(1-H_{\rm arc})+(H_{\rm arc}-H_{\rm cur})+(H_{\rm cur}-S).
\]

The three terms are generation failure, loss of available correctness, and selection failure. Apply the identity on a common scored cohort. A selector that synthesizes an answer performs additional generation; add that answer to the archive and analyze it as a different system.

For interaction round t, let Z_t indicate current coverage:

\[
G_{t+1}=G_t(1-e_t)+(1-G_t)\gamma_t,
\]

where gamma is discovery conditional on Z_t=0 and e is extinction conditional on Z_t=1. This is an identity. Predictive content comes from predicting these probabilities and the state distribution on independent continuations.

### 3.2 Hypotheses and what would falsify them

| ID | Testable proposition | Supporting outcome | Limiting or negative outcome |
|---|---|---|---|
| H1 | Measured depth-response curves predict mixed independent coverage | Calibrated fresh-draw predictions and correct aggregate ordering | Large systematic residuals or unstable predictions |
| H2 | Depth changes answer separation as well as correctness probability | Answer-distribution model improves held-out vote predictions | Shallow-success-only model predicts equally well |
| H3 | Interaction effects depend on discovery, extinction, and selection jointly | Conditional transition model predicts held-out rounds better than coverage-only model | Good one-step fit but poor rollout prediction |
| H4 | Learned regime information transfers to unseen problems | Observable-feature predictor beats benchmark/configuration-only baseline | Regimes remain retrospective descriptions |
| H5 | Archive availability can translate into delivered improvement with practical selection | Archive selector improves accuracy at its measured added cost | More coverage but no reliable practical selection gain |
| H6 | Preferred allocations survive a change in stopping/cost definition | Qualitatively consistent, uncertainty-aware frontier | Advantage depends on forcing or omitted overhead |

No result is required to favor communication, mixed allocation, or the proposed predictor. An interior optimum, a deep optimum, or no detectable ranking separation are all valid outcomes.

### 3.3 Theory limitations to preserve in implementation

- Zero of 25 successes does not establish impossibility; the one-sided 95% binomial upper bound is about 11.3%.
- Five unlabeled samples reveal agreement and observable behavior, not true correctness probability.
- “Correct answer has probability >0.5” is not the plurality condition. For a single canonical correct class, compare its mass with the largest incorrect class.
- Do not apply the single-correct-class plurality theorem directly to code with multiple correct programs.
- Maximizing h(a)/a describes an efficiency ratio under suitable divisibility assumptions. Solve the finite budget allocation problem by exact enumeration or dynamic programming, accounting for remainders and per-call overhead.
- The Chernoff plurality expression in the pasted assessment is a bound, not a calibrated prediction of finite-sample vote accuracy.
- Do not assume longer reasoning is monotonic in correctness; compare unconstrained and monotone fits on development data.
- Distinguish an observed success after an all-wrong root from a capability unavailable to independent generation.

## 4. Cohorts, splits, and leakage prevention

### 4.1 Default panels

| Panel | Default size | Purpose | Allowed reuse |
|---|---:|---|---|
| L96 | Existing 96 main items | Legacy completion and descriptive analysis | Training and exploratory work only |
| D32 | Existing 32 development items | Harness checks and predictor development | Never prospective confirmation |
| C48 | 48 legacy items, 12 per benchmark | Repeated response curves; interaction screening | Fit curves and models |
| V48 | Same 48 problems, entirely new draw IDs | Fresh-draw validation of measured response composition | Evaluation only for E2 |
| T200 | 200 previously unused problems | Unseen-item prediction and confirmatory comparison | Test only |
| M96 | Up to 96 constructed root groups on ≥48 eligible development problems | Controlled interaction mechanisms | Keep separate from naturally sampled population estimates |
| X80 | 80 prespecified T200 items, proportional to benchmark composition | Cost/stopping and small robustness panels | Test-only sensitivity |

**C48 selection:** Before new generation, sample 12 items per family from eligible L96 items with frozen RNG seed. Use the existing shallow strata for balanced development coverage when feasible; record selection probabilities. Sparse cells may contribute fewer items. Any estimates from this enriched panel are conditional to it unless appropriately weighted.

**T200 default composition:** 20 AIME, 20 HMMT, 100 Omni-MATH, 60 GPQA-Diamond, after excluding all historical main/development items and duplicates. Audit source availability first. This avoids assuming that 50 unseen contest problems remain in each small source. If a quota cannot be met, freeze the reduced quota and revised estimand before seeing new outputs. Never replace difficult or failed-generation items based on their outcomes.

Group equivalent/duplicate problems across all splits by normalized problem hash plus an audit of known benchmark overlaps. Hold out entire problem groups, not requests or roots.

### 4.2 Three different validation claims

1. **New draws on known items:** C48 → V48. Labeled measurements of p_x(a) predict a fresh realization on x.
2. **New items:** Development → T200. Only five unlabeled diagnostic draws and permitted metadata are available to predictors.
3. **New configuration/budget:** Withhold the aggregate 4×4A configuration and all B=8A aggregate targets from predictor fitting/tuning. Primitive single-call measurements at 4A and 8A are allowed. This tests compositional generalization, not extrapolation to an unseen depth.

Save split manifests before dispatch. Save model coefficients, preprocessing, selected comparisons, and per-item predictions before test labels are released to the analysis process.

A separate evaluation service may grade test responses concurrently for throughput, but fitting code must not access these labels until prediction artifacts are sealed.

### 4.3 Independent diagnostic samples

T200 gets five independent A-level diagnostic draws, disjoint from evaluation banks. Default observable features:

- Sorted answer-class frequencies, plurality share, top-two share gap.
- Number of distinct classes, empirical entropy, invalid-format fraction.
- Reasoning/answer lengths, finish reason, and measured latency.
- Benchmark identity and model identity.
- Confidence only if already produced by the fixed protocol; requesting new confidence is a separately costed variant.

Do not include gold correctness, true modal margin, correct-agent count, or hidden-test outcomes. Do not add an expensive embedding or LLM classifier to a “cheap” diagnostic without charging its cost.

Create the same five-sample feature bank on development items. On C48, reserve five compatible legacy A draws outside the 16 fitting/pool draws; the existing 25-draw bank can supply both disjoint sets. If these are unavailable, generate five new diagnostics per C48 item, adding at most 240A and 240 candidates. Their outcomes cannot also be used as supervised response targets. Use problem-grouped development cross-validation and disjoint draw IDs for same-item features versus targets.

For retrospective labeled-regime plots, use a separate development diagnostic bank or cross-fitting. Five-sample observable groups must not be renamed as the old 25-sample correctness strata.

## 5. E0: audit and extract value from existing logs

**Owner:** Data/integrity worker.  
**New inference:** None, except a tiny explicitly separate harness smoke test if necessary.

### Required implementation

1. Inventory manifests, raw requests, THINK outputs, ANSWER outputs, graders, item lists, provider IDs, seeds, prompt templates, and actual usage.
2. Verify the true model/version/provider against returned metadata. Do not silently replace the manuscript's questionable identifier with a familiar checkpoint.
3. Reconcile the 96/94/93 cohorts; produce exclusions with reasons and paired comparison denominators.
4. Reproduce the central numbers, including 78.5% versus 79.2%, using a single metrics library.
5. Recover plurality equivalence and tie policy. Preserve the historical policy when reproducing results; use the explicitly frozen new policy for new runs.
6. Detect request reuse across supposedly independent replicas and configurations. Identical answer text alone is not evidence of a cache collision; compare request identity, response ID, seed, and provenance.
7. Reconstruct current and archive candidate membership for every round.
8. Recompute generation/loss/selection decomposition and correct-count trajectories.
9. Recover actual input, reasoning, answer, selector, retry, and cache usage. Unknown usage stays unknown.
10. Produce a reusable-request manifest: reuse only exact matches in model, provider, problem, prompt, sampler, allowance, stopping, parsing, and split role.

**Outputs:** audit.json, provenance.parquet, candidate_catalog.parquet, exclusions.csv, legacy_metrics.csv, reuse_manifest.json, and three decomposition plots.

**Gate:** Stop large inference on identity collisions, missing model provenance, split overlap, or budget enforcement failure. Fix only the affected subsystem; other independent implementation work can continue. Missing historical logs reduce reuse, not the validity of newly collected data.

## 6. E1: mixed independent depth and width

**Question:** Is the best allocation one long call, many short calls, or several adequately deep calls?

### 6.1 Exact B=16A menu

| Configuration ID | n | R | a/A | Reasoning allowance |
|---|---:|---:|---:|---:|
| independent_16x1 | 16 | 1 | 1 | 16A |
| independent_8x2 | 8 | 1 | 2 | 16A |
| independent_4x4 | 4 | 1 | 4 | 16A |
| independent_2x8 | 2 | 1 | 8 | 16A |
| deep_1x16 | 1 | 1 | 16 | 16A |

Generate one bank per item with counts **(16, 8, 4, 2, 1)** at allowances **(1, 2, 4, 8, 16)A**. This is 31 candidate generations and 80A of experimental reasoning per item. Every evaluated configuration costs 16A, not 80A.

Use prespecified prefixes to also evaluate B=8A: 8×A, 4×2A, 2×4A, 1×8A. These share data with B=16A and are correlated.

Do not cut a long trajectory into synthetic shorter calls. A fresh 8A call must actually use the 8A generation protocol.

### 6.2 Legacy completion

If E0 verifies 25 usable A draws and one draw at every deeper allowance, the missing draws are:

- Seven at 2A.
- Three at 4A.
- One at 8A.

Incremental cost: **11 candidates and 34A per problem**, or 1,056 candidates and 3,264A for L96. Persistent grading unknowns require bounds/cohort disclosure, not silent removal from the launch list.

This cheaply completes the old comparison but remains a legacy analysis. Existing long-call successes were already observed and cannot become prospective evidence.

### 6.3 Readout and analyses

For every allocation report current coverage, plurality accuracy, conditional selection reliability, answer entropy, top wrong-answer mass where labels permit, and full actual resource usage.

For n=2, plurality often ties. Default new-run policy: seeded uniform selection among tied modal classes, with a frozen candidate-order seed. Also report analytically tie-averaged accuracy as a lower-Monte-Carlo diagnostic. Do not quietly grant a verifier only to the two-candidate system.

Use natural independent candidate frequencies; retain duplicates for voting. For each configuration derive both realized-pool coverage and, where the bank is larger than the selected pool, the standard finite-bank pass@k estimate:

\[
\widehat{G}_{k,x}=1-\frac{\binom{m_x-c_x}{k}}{\binom{m_x}{k}},\quad k\le m_x.
\]

Do not extrapolate this estimator beyond the bank size. Candidate regroupings reduce Monte Carlo noise but do not increase independent item N.

**Primary comparison:** Select one mixed allocation from development results only, then compare its plurality accuracy with deep_1x16 on T200. Show the entire prespecified menu with simultaneous or multiplicity-adjusted uncertainty as secondary analysis.

**Done when:** All five allocations are scored on the same documented cohort at 16A; the B=8A comparison is available; the deep-versus-mixed conclusion includes paired uncertainty.

## 7. E2: response curves and prediction of fresh allocations

**Question:** Can measured changes in p_x(a) and the answer distribution predict where additional compute should go?

### 7.1 Calibration bank

On C48, obtain **(16, 8, 8, 8, 8)** independent fitting draws at allowances **(1, 2, 4, 8, 16)A**.

- 48 candidate generations and 256A per item.
- 2,304 candidates and 12,288A gross for C48.
- Existing compatible calibration draws can satisfy these counts.
- All E1 development-bank candidates can be reused if the split contract permits; V48 draws cannot.

Eight deep repeats provide a screening estimate with substantial uncertainty. Do not present a precise individual crossover curve from eight Bernoulli outcomes.

Optional development expansion: increase a selected depth cell from 8 to 16 total draws only when its uncertainty changes the preferred allocation in the frozen menu. Rank cells by predicted decision uncertainty per added cost, cap the incremental expansion at 25% of the initial C48 reasoning allowance, and log selection. This is development sampling; confirmatory sample size remains fixed.

### 7.2 Fresh validation

V48 receives a fresh 31-candidate E1 bank: **3,840A** total. Calibration and validation draw IDs must be disjoint even when the problem is identical.

Seal predicted coverage, voting accuracy, and allocation rankings for V48 before its labels are used. Existing legacy outcomes cannot substitute for this fresh validation bank.

### 7.3 Models to implement

**Transparent probability baseline:** Independent Beta(1/2,1/2) posteriors per item-depth cell. Compute nonlinear coverage predictions by posterior integration, not an infinite empirical h at p-hat=1. Joint posterior draws must reuse the same sampled p for all repeated calls at one depth.

**Pooled response model:** Regularized/hierarchical binomial model with benchmark, depth, and item effects. Use depth as a categorical variable initially. Compare a monotone response restriction only on development data.

**Answer-distribution model:** For scalar math/MC tasks, preserve canonical answer identity, model observed class frequencies with an explicit residual unseen-class category, and simulate multinomial candidate pools under the frozen tie rule. Compare at least two development-chosen treatments of unseen wrong-answer mass, including concentrated residual mass. If these imply materially different voting predictions, report that uncertainty instead of declaring the correct class modal.

**Ablations:**

1. Shallow correctness only, assuming no depth response.
2. Depth-dependent correctness without class competition.
3. Depth response plus answer-class distribution.
4. Unstructured cost-sensitive outcome predictor with the same permitted inputs and training labels.

Evaluate population-average prediction error, Brier/log scores for binary observed pool outcomes, predictive interval coverage, and paired configuration ranking accuracy. Report p_x(a), h_x(a)/a, and modal-margin uncertainty for selected illustrative problems; do not cherry-pick them after seeing validation success.

### 7.4 Crossover test

For B=kA, compare the predicted sign of:

\[
h_x(B)-k h_x(A).
\]

Also evaluate the full finite menu, since a deep-versus-shallow result does not identify the best intermediate split.

For general nonuniform allocations, a small dynamic program may maximize:

\[
\sum_j m_j h_x(a_j)
\quad\text{subject to}\quad
\sum_j m_j c(a_j)\le B,\quad m_j\in\{0,1,\ldots\}.
\]

Here c is the declared cost model. Nonuniform partitions suggested by this DP are **predictions**, not experimentally validated recommendations unless separately evaluated.

**Done when:** V48 predictions were committed before scoring; calibration uncertainty is propagated; aggregate rankings and crossover residuals are reported, including failures.

## 8. E3: interaction, discovery, and preservation

Run the allocation comparison and the constructed-root mechanism test as separate experiments.

### 8.1 E3a: narrow mixed interaction grid

On C48, evaluate the following B=16A configurations:

| (n, R, a/A) | Arms | New behavior tested |
|---|---|---|
| (1, 2, 8) | Private | One revision of an adequately deep attempt |
| (1, 4, 4) | Private | Several intermediate-depth revisions |
| (2, 2, 4) | Private, peer-full | Interaction between two deep candidates |
| (4, 2, 2) | Private, peer-full | Wider intermediate-depth team |
| (2, 4, 2) | Private, peer-full | Additional rounds at fixed total allowance |

These eight logical systems cost at most 128A/item if generated separately, or 6,144A on C48 before root reuse. Reuse exact initialization candidates across paired arms and from compatible calibration banks. Each evaluated system is still charged its own root.

Use synchronous rounds. At a given round, an agent sees only allowed information from the preceding round. In private revision it sees its own previous trace; in peer-full it additionally sees every peer's previous trace in seeded order. Freeze whether older rounds are included. Default new protocol uses previous-round traces only; preserve the historical protocol separately when replicating it.

Choose one peer/private configuration pair on C48 before T200 evaluation. Evaluate that pair on all T200 items with naturally sampled roots. Independent initialization from the T200 bank may be reused physically.

Incremental continuation cost on T200 is 16A/item for the two R=2 pairs or 24A/item for the R=4 pair, totaling **3,200–4,800A**, assuming roots are already available. This is the confirmatory communication comparison.

If testing interaction at a second budget is necessary, add the fixed pair (2,2,2) at B=8A on X80. Its two continuations cost 8A/item when the common root is available, or **640A** total. Keep this sensitivity outside the small primary hypothesis family.

### 8.2 E3b: constructed-root mechanism panel

Build roots only from **distinct authentic candidate generations** on development problems. No duplicate-copying of one correct answer to construct four correct agents. No editing a trace to manufacture an incorrect answer.

Core target: up to **96 roots from at least 48 problems**, at most two core roots per problem. Target 24 roots in each cell:

| Initial correct count | Wrong-answer pattern |
|---:|---|
| 0 of 4 | All wrong answers in one canonical class |
| 0 of 4 | At least two wrong classes; largest wrong class ≤2 |
| 1 of 4 | Three wrong answers in one canonical class |
| 1 of 4 | At least two wrong classes |

Construct uniform-depth A roots first. If A roots are too scarce, report eligibility and separately define a 2A-root panel; do not mix root depths under one label.

Root eligibility and construction use labels for mechanism identification. This panel is not a deployment population. Record all eligible and ineligible cells. If fewer than 48 independent eligible problems exist, keep the smaller mechanistic analysis exploratory and spend remaining resources on natural-root confirmation.

Fork every root into four arms:

1. **Private:** each agent revises its own trace.
2. **Fresh:** four independent new calls seeing the problem only.
3. **Peer-full:** each agent sees the preceding root's four full traces.
4. **Peer-answer:** same structure, sharing only answer strings, with own trace retained.

Core continuation: one revision stage, four agents, A per agent. Cost is **16A/root**, at most **1,536A**, excluding root acquisition.

All arms retain the original root in the analysis archive. Native readout uses the four new candidates only. Archive selection is evaluated separately in E4.

Randomize agent assignment and peer presentation order before outcomes; use independent continuation seeds across arms. Shared roots provide paired comparisons. Randomized protocol assignments/seeds support causal protocol comparisons within roots. Selecting concentrated versus diffuse authentic roots does **not** isolate a causal effect of concentration; trace content differs.

### 8.3 Finite-round validation and optional expansion

Split M96 by problem into roughly equal development/validation portions before continuation. Collect two continuation stages on development roots, so the fitting data include transitions out of states reached after the first revision. Fit the transition model and freeze its state representation before evaluating validation rollouts. For validation roots, generate a second revision in all four arms and predict it before seeing that round's outputs.

For 48 development and 48 validation roots, the added second stages cost **768A each**, or **1,536A total**. The primary model is a regularized time-homogeneous transition over correct count, wrong-class partition, validity, allowance, and protocol. Test time-homogeneity against a round-dependent model where development support permits. Report support for each starting state; forecasts passing through unsupported states are explicitly exploratory. A prediction that plugs in the observed validation-round state is a one-step conditional prediction; a true two-step forecast must integrate over its predicted intermediate state.

Coverage-only transition composition is a baseline. Failure of the richer model to forecast the second stage is evidence against the chosen state sufficiency, not an implementation failure.

Optional breadth: add 64 roots covering C0=2 with concentrated/diffuse wrong answers, C0=3, and C0=4, 16 roots per category. No concentration contrast exists when fewer than two wrong candidates remain.

Optional depth sensitivity: on 48 prespecified roots, run both continuation stages at 4A in the same four arms. Cost: **6,144A**. Keep this off the fast path.

### 8.4 Metrics

At each completed stage, report:

- Discovery among initially/currently uncovered roots.
- Extinction among covered roots.
- Correct-agent count and fraction.
- Per-agent correct→incorrect and incorrect→correct transitions.
- Current and archive coverage; delivered accuracy; selection reliability.
- Answer-class switching separated from validity changes.
- Denominators and uncertainty for every conditional rate.

Fit count-state transitions or regularized conditional models; cluster uncertainty by problem. Do not call answer copying a proven transfer of reasoning.

### 8.5 Additional control required for a positive “interaction-enabled discovery” claim

On all-wrong validation roots, add one agent given all four traces and a 4A continuation allowance. This tests concentration of the same evidence into one trajectory and costs 4A per root, or 192A at 48 roots.

Four independent A calls each given the full original root packet already implement the one-stage peer-full treatment. Do not add or count that as a distinct control. To test a specifically multi-round interaction benefit, compare two adaptive peer stages against eight independent A calls in two batches that always see only the original root packet. Also include one 8A call given that packet. Both controls spend 8A of continuation reasoning per root. Match current readout to the last four outputs for the two-batch system; compare complete archives separately. The two added multi-round controls cost 16A per root, or 768A at 48 roots.

Report root exposure, context limits, and total input cost. A peer advantage over fresh sampling alone does not establish a distinctive communication benefit. Add these controls before making that positive claim; otherwise report new success relative to the observed root.

## 9. E4: generation × selector crossover

**Question:** Is the remaining error due to absent candidates, discarded candidates, or a poor readout?

### 9.1 Frozen pools

Start as soon as E1/E3 seals a pool. Use prespecified pools from:

- 16×A, 4×4A, and 2×8A independent allocations.
- The selected private and peer configurations.
- Current and archive views when those differ.

For independent generation, current and archive are identical: evaluate once. Singleton pools require no choice; do not spend verification on them in the practical system.

Each pool stores immutable ordered candidate IDs, canonical classes, multiplicities, source round, and raw trace references. Exclude gold labels from selector input.

### 9.2 Selectors

| Selector | Contract | Budget/interpretation |
|---|---|---|
| Plurality | Return existing candidate from modal class | No LLM call; account CPU if material |
| Same-model verifier | Rank/select existing candidate ID only | Default 2A reasoning cap plus bounded ID output |
| Independent alternate verifier | Same input and selection contract | Different family; call “stronger” only if calibration supports it |
| Oracle | Choose a correct candidate if available | Diagnostic bound, never deployable accuracy |

Default verifier is listwise: one prompt with candidate IDs, answers, multiplicities, and a fixed per-candidate trace excerpt. Set the excerpt rule on development data to fit the maximum pool in the verified context limit. Do not retain longer traces because they are correct. Candidate order is seeded and balanced; test a second order on a 10% development subset for order sensitivity.

Reserve two independent selector samples only on development if needed to estimate selector instability. Use a fixed single selector realization on T200 unless repetitions were budgeted in advance.

At most seven distinct pools per item are needed for the default five generation settings: three independent pools and current/archive for each of the private and peer settings. Two verifiers at 2A each imply a gross **28A/item** reasoning allowance, or **5,600A on T200**, before identical-pool reuse, singleton bypass, input, and answers.

Preserve answer multiplicities. For archive voting, repeated revisions are correlated votes. Report an optional one-latest-candidate-per-agent selector as a prespecified sensitivity; do not describe archive frequency as independent evidence.

### 9.3 Two different evaluations

1. **Fixed-generation selector comparison:** Apply every selector to the same sealed pool. Plot delivered accuracy against added selector cost. This isolates selection.
2. **Fixed-total-resource system comparison:** Reserve selector cost first and reduce generation accordingly. Evaluate feasible generation+selection systems on the frontier.

The first comparison is not equal total compute. An oracle is not a zero-cost deployable verifier.

Measure conditional selector success, false-positive selection among all-wrong pools, and input/order sensitivity. With abstention, report both coverage of answered items and unconditional delivered correctness; abstention is not a correct answer.

**Done when:** At least one independent mixed pool and both views of the selected interaction pair have valid practical selector results, with explicit added cost and common candidate pools.

## 10. E5: prospective prediction on unseen problems

**Question:** Can the proposed framework predict allocation outcomes using information available before seeing the answer?

T200 receives the E1 bank and five disjoint diagnostics. Gross allowance: **17,000A = 17.408 million reasoning tokens**, covering 7,200 candidate generations before additional interactions/selectors.

### Predictors

- **P0:** Benchmark, model, budget, and configuration only.
- **P1:** P0 plus five-sample observable features.
- **P2:** Mechanism-informed predictor using learned depth response and selection/transition components.
- **P3:** Regularized cost-sensitive direct predictor with the same features and training data.

Implement P0/P1 as regularized logistic models and P3 as a compact nonlinear alternative selected on development only. Encode configurations compositionally using log(n), log(a/A), R, budget, and selected interactions; do not rely exclusively on one-hot configuration IDs that cannot represent the withheld 4×4A case. Apply the same observable configuration information across predictors. P2 must output actual probabilities, not a retrospective regime name.

For unseen x, labeled item-specific p_x(a) is unavailable. P2 must predict it from permitted features using development supervision and propagate that uncertainty. Likewise, a predicted correct-class margin is an estimate; true margin belongs only in the label-informed scientific analysis.

### Minimum executable P2

1. Fit a binomial mixed model to development draw correctness: fixed effects for categorical depth, benchmark, diagnostic features, and depth×diagnostic interactions, with a shared item random intercept. Use ridge regularization selected by five-fold problem-grouped development cross-validation. A random depth slope is an optional development-selected extension.
2. For an unseen item, sample one latent item effect per predictive simulation, conditional on its observable features. Use that same effect for all candidates/depths of the simulated allocation. Compute coverage within each simulation and then average. Integrating this heterogeneity matters: substituting a population mean p into the independent-failure formula is generally biased.
3. Fit a second regularized logistic model for conditional selector reliability eta on covered development pools, using observable features, compositional allocation features, and out-of-fold response-model predictions. Set eta=1 for a singleton candidate-only pool. Do not use true test coverage or correct-answer identity as an input.
4. Predict delivered correctness as predicted G times predicted eta. This is a tractable factorized predictor whose accuracy must be tested; it is not a guarantee that the fitted components are correctly specified.
5. Fit E3's transition kernel separately. Keep the detailed answer-class generator as the E2/H2 extension, and test whether it improves over this minimum P2.

Build all training pools with draw IDs disjoint from their diagnostic features. Use neither V48 results nor withheld 4×4A/B=8A aggregate targets to select hyperparameters. Save the fitted models and simulation seeds.

### Evaluation contract

- Primary predictive metric: mean Brier score of delivered correctness across the five B=16A independent configurations.
- Additional metrics: coverage Brier score, log loss, calibration, aggregate MAE, paired ranking accuracy, and prediction residuals by benchmark.
- Hold out aggregate 4×4A targets from fitting and tuning.
- Hold out all aggregate B=8A targets from fitting and tuning.
- Produce predictions for every prespecified test item, including those with invalid diagnostic outputs.
- Commit the development-selected mixed configuration and peer/private configuration before test labels.
- Report whether adding answer-distribution information improves beyond shallow success on V48, and whether *observable* concentration improves P1/P2 on T200. These are different tests.

With one realized run per test item/configuration, the maximum observed correctness over configurations is a **clairvoyant realized-run ceiling**, not a reliable estimate of each item's optimal expected action. Do not report expected-policy regret to that ceiling as if the optimal action were known.

Useful policy performance can be estimated by selecting a configuration from predictions and reading its independent evaluation outcome. However, this evaluates selection of a 16A action **after additional diagnostics**. It costs 21A in reasoning allowance and is not the deployment policy in E8.

**Done when:** The frozen predictor is evaluated on new problem groups and withheld aggregate targets, compared with simple baselines, and accompanied by uncertainty and failure analysis.

## 11. E6: stopping and full-cost sensitivity

### 11.1 Cost ledger

For each evaluated system, report the resource vector:

- Allowed and realized THINK tokens.
- THINK prompt tokens and ANSWER prompt/output tokens.
- Peer-message tokens, including repeated processing by each receiver.
- Selector prompt/reasoning/output tokens.
- Diagnostics, retries, cached tokens, tool calls, and tool runtime.
- Actual API charges when provided; otherwise price-snapshot estimates clearly labeled.
- Measured end-to-end wall time, queue time, and dependency critical path.

Reasoning-token matching is a controlled experimental resource definition. It is not equal FLOPs, dollars, or total tokens. Tokens on different model families are not interchangeable compute units.

Maintain two ledgers:

1. **Physical experiment spend:** deduplicated executed requests.
2. **Logical deployed-system cost:** all requests needed by that evaluated configuration, including roots and diagnostics reused physically in this study.

A shared root is an experimental saving, not free deployment compute.

### 11.2 Natural stopping

On X80, run natural-stopping versions of four locked configurations: deep_1x16, independent_16x1, the development-selected mixed allocation, and the selected peer configuration. Use the same maximum reasoning allowance and ANSWER protocol, remove forced continuation, and record actual stopping.

This costs at most **5,120A** of additional reasoning allowance before overhead. If two selected configurations coincide, deduplicate.

Report both matched-cap accuracy and the realized resource frontier. Underusing a cap is legitimate in natural stopping, but must not be described as equal realized compute. Do not reallocate saved tokens in this test; reallocation is a different adaptive policy.

### 11.3 Total-budget frontier

First reconstruct actual costs for every completed run and plot accuracy–cost Pareto curves. This requires no additional generation.

For a genuinely hard-budget follow-up, choose two caps from development cost distributions and freeze them before X80 evaluation. The scheduler must reserve the maximum configured prompt/output/selector costs before launching an action; otherwise it must take a feasible fallback. Include a reserve for bounded retries.

Use API-dollar caps for cross-family comparisons, or measured hardware/FLOPs only where defensible measurements exist. If hidden usage makes a hard total-token cap unenforceable, report an allowance/price-capped experiment with its limitation rather than claim exact total compute.

**Done when:** Central frontier points include answer, input, retry, and selection overhead, and the impact of natural stopping is visible.

## 12. E7: focused generalization

This package is recommended for broad claims, but it should not delay core validation.

### 12.1 Second family

Select one different-family reasoning model from accessible infrastructure. Freeze exact model and provider versions after a development-only feasibility check. Choose by task capability, controllable budget, trace availability, and cost, not by which test model agrees with the hypothesis.

Use 20 separate development items for feasibility and threshold calibration. If the provider cannot enforce exact lengths, use a natural-stopping capped-budget replication and label it accordingly.

On X80, run deep_1x16, independent_16x1, independent_4x4, and the locked interaction pair if affordable. The three independent settings cost **48A/item**, or **3,840A** for 80 items. They test directional robustness; they do not reproduce the full frontier.

### 12.2 Executable coding

Use a pinned LiveCodeBench release and a frozen problem-date window, with 20 development and 100 test problems. Select a window later than the model's documented cutoff where possible; do not guarantee absence of contamination when the cutoff is unknown. Pin the official evaluator and dataset revision.

Run deep_1x16, independent_16x1, and independent_4x4, then the locked interaction pair if coding is a main claim. Three independent settings cost **4,800A** on 100 test items.

Generation sees public statements/examples only in the minimal extension. Public-example execution may be added as a separate tool-enabled protocol with identical access and charged tool costs across arms.

Hidden tests are used only for final grading. Practical selection may use public/generated tests, never the hidden grading suite. Run generated code in isolated, resource-limited workers.

Report hidden-test pass@k and selected pass@1. Use a selector appropriate to code, such as same-model verification or public-test selection. Exact text or AST voting is not a reliable semantic-equivalence assumption; retain it only as a labeled baseline. Do not transfer the single-correct-class modal theorem to code without a new argument.

## 13. E8: optional five-sample allocation policy

This is an extension, not a prerequisite for the theory paper.

### 13.1 Budget-correct action menu

For a reasoning allowance of B=16A, five initial A draws leave **11A**. Their outputs remain available. A menu that adds an entire 16A action is a 21A policy and must be labeled accordingly.

A simple continuation menu at 16A is:

| Action | New reasoning after diagnostics | Final readout |
|---|---:|---|
| stop | 0 | Vote over five diagnostics |
| widen | 11 independent A calls | Vote over all 16 |
| deepen | One new 11A call | Return deep call |
| mixed | Two independent 5A calls | Vote over the two new calls; unused 1A remains |
| revise | Choose one diagnostic by frozen plurality rule; one 11A private revision | Return revision |
| verify | One verifier capped at 2A | Select from the five diagnostics |

All new depths and conditioned continuations require their own development outcomes. The static E1 bank cannot supply an 11A call or a revision conditioned on an observed diagnostic trace.

This table matches reasoning allowance only. Under a total-resource cap, reserve answer, prompt, and selector overhead before allocating the remaining reasoning.

### 13.2 Baselines and evaluation

Compare with:

- Strongest fixed allocation selected on development, allowed to use the whole original budget.
- Stop-after-five and a simple agreement-threshold policy tuned on development.
- Direct cost-sensitive learned policy with the same inputs and actions.
- A budget-aware single-agent revision policy.
- If making an orchestration claim, one faithful dynamic baseline such as DyLAN, plus a simple LLM supervisor that chooses from the same action menu.

Charge supervisor decisions, tool calls, message processing, diagnostics, and selection. Pin external implementation commits and disclose modifications needed to enforce the budget. A constrained adaptation is not an unmodified published baseline.

Train on development outcomes; evaluate on T200 only if the policy menu and selection were frozen before test exposure. Otherwise use a new test split. Log each action, predicted utility, remaining budget, fallback, and final cost.

Metrics: unconditional accuracy, cost, budget violations, calibration, average savings from early stopping, and regret only to an independently estimated feasible-menu reference. An oracle requiring gold labels stays a diagnostic.

## 14. Statistical analysis and stopping

### 14.1 Estimands

Report both:

1. **Prespecified T200 mixture mean:** weights 0.10 AIME, 0.10 HMMT, 0.50 Omni-MATH, 0.30 GPQA. This is the default primary prospective population.
2. **Four-family macro mean:** equal family weights, matching the manuscript's style.
3. Per-family results with uncertainty.

The mixture changes the population target, not merely the sample size. State it alongside every primary aggregate. Do not imply that the 200-item mixture has equal-weight macro power of 200 balanced observations.

### 14.2 Primary family

Predeclare three tests:

1. Development-selected mixed allocation versus one-deep delivered accuracy on T200 at B=16A, two-sided.
2. P2 versus P1 mean delivered-correctness Brier score across B=16A allocations, using one item-level average loss difference.
3. Development-selected peer versus private delivered accuracy on natural-root T200, two-sided.

Apply Holm at family-wise alpha=0.05. Provide effect sizes and 95% intervals. Label all other configuration/regime sweeps secondary or exploratory. If E3 is omitted, freeze a two-test family before scoring rather than retaining a fictitious unrun test.

Use at least 5,000 benchmark-stratified paired problem bootstrap replicates for intervals. Carry every draw, configuration, root, and round of a sampled problem together. Resampling individual candidates as independent subjects is invalid.

### 14.3 Power and sample-size interpretation

A rough paired binary comparison uses:

\[
N\approx (1.96+0.84)^2 d/\Delta^2,
\]

where d is the discordant fraction. At d=0.25, approximately 196 independent items are needed for a 10-point difference and 784 for a 5-point difference, before multiplicity. Use development discordance to simulate the actual paired design.

For the 20/20/100/60 composition, the equal-family macro effective N is approximately 126 under equal within-family variance, so its rough detectable difference is about 12.5 points, not 10. Twenty-item contest subsets cannot establish small within-family effects.

More repeats improve probability estimates on an item; they do not create more independent problems. If a communication effect is small, report the uncertainty or enlarge a preregistered new cohort. Do not call a non-significant difference equivalence.

### 14.4 Missingness and stopping

- Preserve correctness as true / false / unknown.
- Invalid model output is false under the frozen answer contract.
- Grader timeout or unavailable reference is unknown.
- Retry evaluation failures separately from generation; retries must not condition on whether a candidate is correct.
- Use common cohorts for paired decompositions and show best/worst-case bounds for unresolved outcomes.
- Include infrastructure-failure rate and an end-to-end operational analysis in which failed delivery is failure.
- Do not expand T200 after looking for significance.
- Development repetitions may use the predeclared uncertainty rule and cost cap.
- Freeze a deadline/cost cap for each optional package. Ending with an inconclusive interval is a valid result.

## 15. Implementation contract

### 15.1 Proposed repository modules

These are modules to implement or map to the existing harness, not assertions about files already present.

| Module | Responsibility |
|---|---|
| manifests | Immutable studies, items, splits, models, prompts, configurations |
| adapters | Provider/model capability checks and THINK/ANSWER calls |
| scheduler | Dependency DAG, leases, token reservations, rate limits, retries |
| banks | Independent draws and deterministic pool materialization |
| interactions | Immutable root forks and synchronous protocols |
| grading | Parsing, equivalence, tri-state grading, code execution |
| selectors | Candidate-only voting and model selection |
| accounting | Physical and logical resource ledgers |
| predictors | Response models, transitions, observable-feature predictions |
| analysis | Paired metrics, bootstrap, figures, report tables |

### 15.2 Required records

~~~yaml
run_manifest:
  study_id: string
  git_commit: string
  dataset_revision: string
  split_manifest_hash: string
  model_id: string
  model_revision: string_or_null
  provider_id: string
  tokenizer_revision: string_or_null
  sampler: {temperature: number, top_p: number, top_k: integer_or_null}
  stopping_mode: forced_or_natural
  prompt_template_hashes: object
  parser_version: string
  grader_version: string
  tie_policy: seeded_uniform_modal_class
  resource_definition: reasoning_allowance_or_total_resource

candidate:
  candidate_id: string
  study_id: string
  item_id: string
  problem_group_id: string
  split: string
  config_id: string
  root_id: string_or_null
  replicate_id: integer
  round: integer
  agent_id: integer
  allowance_tokens: integer
  seed: integer
  provider_seed_supported: boolean
  request_hash: string
  provider_response_id: string_or_null
  parent_candidate_ids: list
  visible_peer_candidate_ids: list
  raw_think_ref: string
  raw_answer_ref: string
  canonical_answer_class: string_or_null
  answer_validity: valid_or_invalid_or_unknown
  correctness: true_false_or_unknown
  grading_reason: string
  request_status: string
  usage_record_ids: list

pool:
  pool_id: string
  item_id: string
  config_id: string
  round: integer
  view: current_or_archive
  candidate_ids: list
  candidate_order_seed: integer
  content_hash: string

selection:
  selection_id: string
  pool_id: string
  selector_id: string
  selector_model_revision: string_or_null
  selected_candidate_id: string_or_null
  selected_correctness: true_false_or_unknown
  abstained: boolean
  usage_record_ids: list
~~~

Correctness and references must live in evaluator-only fields/files that are never inserted into prompts. Every request records purpose: THINK, ANSWER, selector, supervisor, or other.

### 15.3 Identity and reuse

Generate IDs from the full immutable tuple:

~~~text
study + problem_group + item + model_revision + provider +
prompt_hash + stopping_mode + allowance + protocol +
root + replicate + round + agent + purpose + seed
~~~

Never join solely on episode_id. Never infer a missing selector outcome as incorrect. A reused candidate carries its original immutable identity and multiple explicit logical memberships; do not relabel it as an independent draw.

If provider seeds are unsupported, keep a unique requested draw ID and log unsupported status. Distinct seed numbers alone do not prove independent outputs. Disable cross-replicate response caching; resuming the exact same completed job may use its saved response.

### 15.4 Scheduling and failure behavior

- One task is a single independent candidate, continuation, or selector request chain.
- THINK→ANSWER is sequential. Agents within a round run concurrently. Later rounds depend on all required previous-round messages.
- Schedule 16A/8A calls early to avoid a long tail; bucket by expected context/output size.
- Use provider-specific request and token semaphores. Start with eight in-flight requests/provider if allowed, then adjust after a 20-request pilot; respect observed limits.
- Retry transport/rate-limit failures at most three total attempts with exponential backoff and jitter. Preserve costs and response IDs for every attempt.
- Persistent forced-length failure is a failed job, not a valid exact-budget sample. Count attempted spend; do not loop indefinitely.
- Randomize/interleave conditions in time within provider blocks to reduce provider-drift confounding.
- Persist raw responses before grading, then atomically mark completion. Use leases and idempotent result writes to prevent duplicate dispatch.
- Workers append immutable shards. One reducer owns aggregate tables; do not have workers concurrently rewrite shared CSVs.
- Queue downstream selectors only when the named pool is sealed.
- Migrate a problem to another provider only as a separate replication block; do not silently mix providers inside its paired comparison.

Approximate wall time from a pilot as the maximum of token-throughput, request-rate, and dependency-critical-path limits, plus observed retry overhead. Do not promise a fixed completion time without measured throughput.

### 15.5 Essential preflight tests

Use a small fixture set and a 12-item smoke test from D32. These checks prevent expensive invalid inference:

1. Every nominal C equals nRa and R includes initialization.
2. Independent n changes actual request count; replicas do not alias saved responses.
3. Private and peer forks have byte-identical roots and permitted information only.
4. Duplicate class counts, ties, malformed answers, and unknown grades behave correctly.
5. For current-pool readouts, H_arc ≥ H_cur ≥ S and the three-term decomposition exactly matches error. For archive readouts, assert S_arc ≤ H_arc and use 1−S_arc = (1−H_arc)+(H_arc−S_arc); S_arc may exceed H_cur.
6. Fresh sampling has no root content in prompts but retains the root in its analysis archive.
7. Test labels are inaccessible to predictors/selectors.
8. Logical cost charges shared roots; physical cost counts each request once.
9. A scheduler restart does not duplicate successful jobs.
10. All selector IDs refer to actual members of the sealed pool.

## 16. Parallel workstreams and execution DAG

Assign separate owners and directories. Freeze shared contracts before concurrent changes.

| Worker | First task | Can run concurrently with | Handoff |
|---|---|---|---|
| W0: integration/data | E0, schema, manifests, smoke gate | All implementation on fixtures | Frozen data contract and valid reuse list |
| W1: generation | E1 banks, E2 counts, depth adapters | W2–W5 after contract freeze | Immutable candidate shards |
| W2: interaction | E3 protocols, root eligibility, fork scheduling | Independent generation | Root and round pools |
| W3: selectors/cost | E4 selectors, E6 ledger | Generation and root construction | Selection records and cost tables |
| W4: prediction/stats | E2/E5 models, preregistration, bootstrap | All data collection with split restrictions | Sealed predictions and tests |
| W5: evaluation/reporting | Grading, canonicalization, figures | Every producing worker | Validated labels and final tables |

~~~mermaid
flowchart TD
  A["E0: audit and freeze contracts"] --> B["Independent banks"]
  A --> C["Interaction and selector implementation"]
  A --> D["Analysis and prediction implementation"]
  B --> E["C48 fit and V48 validation"]
  B --> F["Root forks and sealed pools"]
  C --> F
  F --> G["Selectors and transition models"]
  E --> H["Freeze predictors and comparisons"]
  D --> H
  G --> H
  H --> I["T200 evaluation"]
  I --> J["Cost sensitivity and final results"]
~~~

T200 candidate generation may start before model fitting completes if labels remain sealed. T200 diagnostic features can be released for prediction; outputs used as prediction targets remain inaccessible. Run the selected T200 interaction pair only after its configuration is locked.

**Critical path:** contracts → compatible banks → development fit/configuration choice → sealed predictions → prospective scoring → paired analysis.

**Shorten it by:** reusing verified roots, launching expensive independent calls first, grading continuously, materializing multiple budgets from one bank, and evaluating selectors without regenerating candidates.

## 17. Costed launch plan

All numbers below are **reasoning allowance**, excluding prompts, ANSWER, selectors unless listed, retries, and root-acquisition overhead. Multiply A by 1,024 for tokens. These are planning estimates, not authorization to spend a particular amount.

### 17.1 Core predictive package

| Component | Gross allowance | Candidate generations |
|---|---:|---:|
| Legacy E1 missing draws, if anchors valid | 3,264A | 1,056 |
| C48 fitting bank | 12,288A | 2,304 |
| V48 independent validation bank | 3,840A | 1,488 |
| T200 independent evaluation bank | 16,000A | 6,200 |
| T200 five-draw diagnostics | 1,000A | 1,000 |
| Conservative subtotal | **36,392A** | **12,048** |

The subtotal is about **37.265 million reasoning tokens**. C48 fitting and legacy completion overlap physically; the scheduler must subtract verified reusable requests. The subtotal does not subtract undocumented reuse; it assumes the five development diagnostics can be reserved from compatible legacy draws. Otherwise add up to 240A and 240 candidates. THINK+ANSWER means each candidate commonly requires two model calls, plus forced-length retries.

**Smaller first release:** C24, six items/family, and V24 cost 6,144A + 1,920A. With legacy completion, that is 11,328A before reuse, roughly 11.600 million reasoning tokens. This is a mechanism-development milestone, not a substitute for T200 generalization.

### 17.2 Add-on packages

| Component | Gross/incremental allowance | Scope |
|---|---:|---|
| C48 interaction screen | ≤6,144A before root reuse | Eight B=16A systems |
| T200 chosen peer/private continuations | 3,200–4,800A | Roots already in independent banks |
| M96 core one-stage assay | ≤1,536A | Roots already available |
| Second stage on 48 development + 48 validation roots | ≤1,536A | Four arms |
| X80 second-budget interaction sensitivity | 640A | Common roots available |
| T200 two verifier selectors, default pool set | ≤5,600A | Seven pools/item, 2A each |
| X80 natural-stopping replication | ≤5,120A | Four configurations |
| Second-family independent replication | 3,840A | Three configurations on 80 items |
| Coding independent replication | 4,800A | Three configurations on 100 items |

Add root acquisition, development feasibility runs, answer/input cost, and any optional controls explicitly. Do not sum upper bounds and present them as a measured invoice.

### 17.3 Dispatch sequence

1. E0 plus parallel implementation of selectors, interaction protocols, and statistical code.
2. Freeze manifests; pass smoke gate.
3. Launch legacy fill and C48 fitting draws in one deduplicated queue; begin V48/T200 banks in separate namespaces.
4. As roots/pools finish, stream development interactions and selector jobs.
5. Fit and lock predictions/comparisons; expose T200 diagnostics but keep outcome labels sealed.
6. Launch the locked T200 interaction pair; seal remaining predictions.
7. Release test labels once; run prespecified analysis.
8. Add cost/stopping checks and focused generalization according to remaining budget.

## 18. Required figures, tables, and completion gates

| Output | Minimum content |
|---|---|
| F1: allocation frontier | All five independent configurations at 16A and four at 8A; coverage and delivered accuracy; paired intervals |
| F2: predictive depth response | C48 fit → V48 predictions, observed outcomes, uncertainty, crossover examples selected by a frozen rule |
| F3: bottleneck decomposition | Generation, availability loss, selection failure across allocation and rounds |
| F4: interaction transitions | Discovery/extinction by initial state; natural roots separated from constructed roots; finite-round predictions |
| F5: selector crossover | Current/archive pools, practical versus oracle readout, added selector cost |
| F6: prospective prediction | P0–P3 calibration, Brier scores, withheld configuration/budget results, benchmark residuals |
| F7: robustness | Natural stopping and actual-cost frontier; second family/coding if run |
| T1: audit/cohorts | Source, N, exclusions, exact model/provider, repeat counts |
| T2: costs | Nominal reasoning, actual input/output, selectors, retries, wall time; physical/logical distinction |
| T3: primary tests | Locked contrasts, paired effects, intervals, Holm-adjusted p-values |

Completion means the planned question can be answered, not that a hypothesis is significant.

- [ ] E0 reproduces central legacy results and resolves model/cohort discrepancies.
- [ ] Missing mixed independent configurations are complete.
- [ ] At least one fresh-draw predictive validation is sealed and scored.
- [ ] New-item prediction is evaluated before calling regimes generalizable.
- [ ] Communication conclusions rely on matched roots and a clearly stated population.
- [ ] Candidate-only selectors obey pool membership and are fully costed.
- [ ] Full-cost and stopping limitations are quantified.
- [ ] Missing outcomes and failed jobs are visible.
- [ ] All figures regenerate from immutable records.
- [ ] Main text states only claims supported by completed packages.

Manuscript integration: keep exposition, figures, and appendix content in main.tex, with references in the separate bibliography as already requested. Replace the abstract's ambiguous “pass@k–pass@1 gap” with coverage versus delivered accuracy. Label legacy strata descriptive and distinguish measured latency from reconstructed critical-path estimates.

## 19. External anchors for implementers

These sources motivate design choices; this specification's matrices, sample sizes, and execution plan are proposed extensions.

1. [Snell et al., Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): difficulty-dependent allocation is prior work. The proposed increment is independently validated, mechanism-based allocation prediction.
2. [Kim et al., Towards a Science of Scaling Agent Systems, v3](https://arxiv.org/abs/2512.08296v3): architecture/task-dependent predictive scaling is prior work. Use the revised version, not only the older blog summary.
3. [Brown et al., Large Language Monkeys](https://arxiv.org/abs/2407.21787): repeated-sampling coverage and practical selection should be evaluated separately.
4. [Chen et al., Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374): finite-bank pass@k estimator.
5. [LiveCodeBench official repository](https://github.com/livecodebench/livecodebench) and [paper](https://arxiv.org/abs/2403.07974): pin the dataset/evaluator for the coding extension.
6. [DyLAN paper](https://arxiv.org/abs/2310.02170) and [official implementation](https://github.com/SALT-NLP/DyLAN): optional dynamic communication/early-stopping baseline, with adaptation and compute accounting disclosed.

Public sources were checked on 2026-09-25. Verify implementation versions when launching. Do not infer model availability, current prices, or provider capabilities from this document.
