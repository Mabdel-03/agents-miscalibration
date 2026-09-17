# Sources and provenance

Documentation checked September 17, 2026. URLs identify official documentation, dataset maintainers, or source repositories. Re-resolve and pin runtime versions before dispatch. Prices and service behavior can change. The numerical budgets, panel sizes, scheduling settings, and statistical contrasts in this package are proposed design choices, not reported benchmark findings.

## API and model facts

**[S01] OpenRouter, Gemini 2.5 Flash-Lite.** Standard listing used for planning: $0.10/M input, $0.40/M output. Endpoint-specific pricing/tier must be checked.  
`https://openrouter.ai/google/gemini-2.5-flash-lite`

**[S02] OpenRouter, Gemini 2.5 Flash.** Standard planning rates: $0.30/M input, $2.50/M output. Numeric reasoning-budget configuration is described on the listing.  
`https://openrouter.ai/google/gemini-2.5-flash`

**[S03] Google Cloud, Thinking.** The Gemini 2.5 section documents a soft thinking budget. Flash-Lite range: 512–24,576; Flash maximum: 24,576. Requested allowance is not guaranteed realized consumption. The page currently redirects from the older Vertex documentation address.  
`https://cloud.google.com/vertex-ai/generative-ai/docs/thinking`

**[S04] OpenRouter, Reasoning Tokens.** Unified reasoning request fields; numeric-budget support is model/provider dependent; reasoning is billed as completion output, and the combined completion cap can be exhausted before visible output. Returned thinking need not expose a full internal trace.  
`https://openrouter.ai/docs/guides/best-practices/reasoning-tokens`

**[S05] OpenRouter, Response Caching.** `X-OpenRouter-Cache: false` disables response caching. A generated-response cache and provider prompt-prefix caching are distinct. Cached completions can be returned verbatim despite a nonzero sampling temperature.  
`https://openrouter.ai/docs/guides/features/response-caching`

**[S06] OpenRouter, Provider Routing and Endpoints API.** Pin provider/endpoint; disable fallback; require parameter support; preserve live endpoint and pricing snapshots.  
`https://openrouter.ai/docs/guides/routing/provider-selection`  
`https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model`

**[S07] OpenRouter, Usage, Limits, Generation Metadata, and FAQ.** Token/cost fields, cached-token details, account/key credit checks, transient capacity errors, and generation metadata. Extra API keys do not create unlimited provider capacity. Verify actual credit-purchase fees rather than treating cash as identical to inference credits.  
`https://openrouter.ai/docs/cookbook/administration/usage-accounting`  
`https://openrouter.ai/docs/api_reference/limits`  
`https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation`  
`https://openrouter.ai/docs/faq`

## Benchmark and grading sources

**[S08] MATH-500 dataset.** Source inventory and schema.  
`https://huggingface.co/datasets/HuggingFaceH4/MATH-500`

**[S09] Math-Verify.** Mathematical parsing/evaluation interface, argument ordering, timeout support, and documented asymmetry of verification. Do not use asymmetric gold grading blindly as an answer-equivalence relation for voting.  
`https://github.com/huggingface/Math-Verify`

**[S10] MMLU-Pro dataset card.** Source schema, benchmark description, and maintenance history, including the January 2026 answer-option whitespace correction.  
`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`

**[S11] BIG-Bench Hard repository.** Public task JSON files; this package selects four subtasks, not the full 23-task suite.  
`https://github.com/suzgunmirac/BIG-Bench-Hard`  
`https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/logical_deduction_seven_objects.json`  
`https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/tracking_shuffled_objects_seven_objects.json`  
`https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/boolean_expressions.json`  
`https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/temporal_sequences.json`

**[S12] EvalPlus repository.** MBPP+ evaluation and code-evaluation tooling. This is the primary source for the coding benchmark and pinned grading semantics.  
`https://github.com/evalplus/evalplus`

**[S13] EvalPlus data/CLI documentation.** Dataset fields, task IDs, solution records, and base/plus test distinction.  
`https://github.com/evalplus/evalplus/blob/master/evalplus/data/mbpp.py`  
`https://github.com/evalplus/evalplus/blob/master/docs/cli.md`

**[S14] EvalPlus leaderboard and release notes.** Maintainer notes about the sanitized MBPP subset, inventory differences, and stronger tests. Resolve the installed release rather than hardcoding the count of tasks.  
`https://evalplus.github.io/leaderboard.html`  
`https://github.com/evalplus/evalplus/releases`

## User-provided prior reports

**[U01] ARD36, “What Communication Does to a Team of Agents,” September 13, 2026.** Repaired-table report supplied by the user. Pages 2–3 describe baseline/control collision and grading repairs; page 6 explains the coverage/selection and fixed-denominator trap; pages 10–11 describe limited independent-item support and scope. It informs tests, not API or new-benchmark claims.

**[U02] “A Scaffolding Atlas,” September 7, 2026.** User-supplied exploratory report. Pages 2–3 identify truncation, syntactic code grouping, and content-addressed aliasing; pages 6–8 identify budget allowances that did not bind; pages 9–10 discuss code-selection limitations. The new package does not import its exploratory effect estimates as new-study results.
