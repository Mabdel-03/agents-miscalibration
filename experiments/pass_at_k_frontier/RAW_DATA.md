# Raw data: prompts, transcripts and per-call records

**The raw data is not on GitHub.** It contains verbatim benchmark questions and answer keys, including GPQA-Diamond, so it
cannot go in this public repository. It stays on the MIT ORCD cluster where the studies ran:

```text
/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/
```

An agent working from a clone cannot read it. The curated tables here answer most questions. If yours needs the prompts or
the models' reasoning, say so and ask the repository owner for the relevant files. **Do not copy their contents into this
repository, or anywhere else public.**

| Study | Run root, under the path above | Size |
|---|---|---|
| 1 | `openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1/` | 3.7 GB |
| 2 | `forced_deep_pilot_v1/<model>/` | 2 MB |
| 3 | `forced_budget_v2/artifacts/` | 923 MB |
| 4 | `dynorch_v1/artifacts/` | 374 MB |
| 5 | `passk_cont_v1/` | 833 MB |

## From a table row to its transcript

Studies 1, 3 and 5 share one layout. In a table row, `node_id` (`candidate_id` in study 5) is a 64-hex id, and:

```text
<root>/raw/<first 2 hex>/<id>_<attempt>.json.gz    full request and response
<root>/parsed/<first 2 hex>/<id>.json              extracted answer
<root>/grades/<first 2 hex>/<id>.json              grade
```

| Study | `<root>` |
|---|---|
| 1 | `openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1` |
| 3 | `forced_budget_v2/artifacts/shards/<shard_id>` |
| 5, `source == "passk"` | `passk_cont_v1/artifacts/shards/<shard>` |
| 5, `source == "legacy"` | `forced_budget_v2/artifacts/shards/<shard>`, because these draws were reused from study 3 |

```python
import gzip, json

# attempt_id in the tables is "<node_id>:<attempt>". Most nodes finished on attempt 1, but not all, so read it.
attempt = attempt_id.rsplit(":", 1)[1]
rec = json.load(gzip.open(f"{root}/raw/{node_id[:2]}/{node_id}_{attempt}.json.gz"))
rec["request_body"]                           # forced studies: {"think": ..., "answer": ...}
body = json.loads(rec["response_body_text"])  # forced studies: "think" and "answer" responses
rec["result"]                                 # status, provider, finish reason, latency
```

Study 5's extensions keep theirs in `passk_cont_v1/e8a/raw/<arm>/`, `e8b/raw/<arm>/` and `lex/raw/`, with call logs beside
them (`e8a/calls.jsonl`).

Study 4 is keyed by run, not by node. Each row of `dynorch_runs` maps to:

```python
f"dynorch_v1/artifacts/raw/{partition}/{arm}/{rung}/{repeat}/{item_id.replace('/', '__')}/a{attempt}/call_0001.json.gz"
```

## Benchmark questions and answers

| Contents | Path |
|---|---|
| Question text per item (`prompt`) | `<run root>/manifests/public/<benchmark>.jsonl` |
| Answer key per item (`gold_answer`) | `<run root>/private/<benchmark>.jsonl` |

Here `<run root>` is `openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1`, `forced_budget_v2/artifacts`
or `passk_cont_v1/artifacts`. Study 4 uses study 3's items.
