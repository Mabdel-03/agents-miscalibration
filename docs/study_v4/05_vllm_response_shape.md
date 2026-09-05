# Observed vLLM 0.21.0 response shape (real smoke request, 2026-09-05 18:05 EDT, Qwen3-14B `14B-long`, node3904:8813)

Request: `chat.completions.create(model="14B", messages=[{role:user,...}], temperature=0.6, max_tokens=8192, seed=12345, extra_body={"chat_template_kwargs":{"enable_thinking":true},"top_k":20,"top_p":0.95,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"return_token_ids":true})`

- top-level keys: choices, created, id, kv_transfer_params, model, object, prompt_logprobs, prompt_routed_experts, prompt_text, **prompt_token_ids** (list[int], len == usage.prompt_tokens), service_tier, system_fingerprint, usage
- choice keys: finish_reason ("stop"), index, logprobs (None), message, routed_experts, stop_reason, **token_ids** (list[int], len == usage.completion_tokens; includes the thinking tokens)
- message keys: annotations, audio, **content**, function_call, **reasoning** (the parsed `<think>` channel; there is NO `reasoning_content` key), refusal, role, tool_calls
- usage: {completion_tokens, prompt_tokens, total_tokens, completion_tokens_details: None, prompt_tokens_details: None}
- content began with "\n\n" followed by bare JSON (no fence) in this sample → parser: strip whitespace, then at most one enclosing fence.
- single-sequence decode ≈ 47 tok/s on 14B TP1 (unloaded); 839 completion tokens in 17.9 s.
- `extra_body={"guided_json": {...}}` with enable_thinking=false was ACCEPTED (no error) but NOT ENFORCED: the model returned "```json\n{...}\n```". Treat guided decoding as unavailable in this stack; judge/forecast roles use strict JSON parsing with the same one-fence salvage rule (amendment E2 is therefore "not used").
- Server flags in effect: `--reasoning-parser qwen3 --max-num-seqs 64 --max-logprobs 20 --gpu-memory-utilization 0.90 --max-model-len 40960`, env `VLLM_USE_FLASHINFER_SAMPLER=0` (native PyTorch top-k/top-p sampler; FlashInfer's JIT needs nvcc which the hardened template scrubs), all compile caches on node-local /tmp.
