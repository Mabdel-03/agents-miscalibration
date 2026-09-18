"""Forced-depth DEEP_SINGLE pilot: one prompt, reasoning held open for exactly k*A tokens, over OpenRouter.

Mechanism (no re-prompting; the model never sees a new instruction):

1. THINK: raw ``/completions`` on the model's own chat template, ending ``<|im_start|>assistant\\n<think>\\n``.
   ``logit_bias`` -100 on ``</think>``, ``<|im_end|>`` and ``<|endoftext|>`` so the model cannot close its reasoning
   or stop; ``max_tokens = k*A`` so the call ends exactly at the budget.
2. ANSWER: the same prompt + the model's own reasoning + ``\\n</think>\\n\\n``; no bias, answer cap = headroom.

This is the API form of a decode-time processor that bans ``</think>`` until k*A then forces it. The one seam is
that the reasoning text is re-tokenized for step 2. ``natural`` arms run step 1 without the bias (the model closes
its own reasoning) as the unforced reference.

Prompts, items and grading are the frozen ones of ``openrouter_agent_allocation_v1`` (root slot-0 requests of the
main panel, text benchmarks only; MBPP+ needs the sandbox and is out of the pilot).

    forced_deep_pilot.py dry-run            # render and print one prompt, no network
    forced_deep_pilot.py probe              # 3 tiny requests: does the ban hold, does the prefill answer
    forced_deep_pilot.py run [--items 32]   # the pilot grid
    forced_deep_pilot.py report
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from agent_allocation._vendor.extraction_tags import TAG_OK, strict_single_pair
from agent_allocation.apikey import load_api_key, redact
from agent_allocation.contracts import Candidate, ParseStatus, PrivateEvalSpec, PublicTask, load_record
from agent_allocation.grading.text.grade import grade_text

SRC = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/artifacts/"
           "openrouter_agent_allocation_v1")
OUT_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_deep_pilot_v1")
OUT = OUT_ROOT

#: (model, provider, (</think>, <|im_end|>, <|endoftext|>) ids from the model's tokenizer.json). ``logit_bias`` is
#: *listed* for many endpoints but ignored by some: DeepInfra and Cerebras dropped a +100 " banana" bias on
#: 2026-09-18 while Together, Parasail and Mancer honoured it. Check with ``bias-check`` before trusting a preset.
PRESETS = {
    "qwen3.5-9b-together": ("qwen/qwen3.5-9b", "together", (248069, 248046, 248044), 41674),
    # Qwen3.5/3.6/3.8 share one tokenizer. Parasail returns reasoning inline in `text` (no server-side parser).
    "qwen3.5-397b-parasail": ("qwen/qwen3.5-397b-a17b", "parasail/fp8", (248069, 248046, 248044), 41674),
    "qwen3-32b-deepinfra": ("qwen/qwen3-32b", "deepinfra/fp8", (151668, 151645, 151643), 43096),  # bias IGNORED
}
MODEL = PROVIDER = BAN = BANANA = None                      # set by configure()


def configure(preset: str) -> None:
    global MODEL, PROVIDER, BAN, BANANA, OUT
    model, provider, ids, BANANA = PRESETS[preset]
    MODEL = model
    PROVIDER = {"only": [provider], "allow_fallbacks": False, "require_parameters": True}
    BAN = {str(t): -100 for t in ids}
    OUT = OUT_ROOT / preset
A = 1024
HEADROOM = 4096                                             # text benchmarks, as in the main study
TEMPERATURE, TOP_P = 0.7, 0.95                              # the study's sampling, not Qwen's recommended 0.6/top_k 20
SPEND_CAP_USD = 8.00
NATURAL_CAP = 16 * 1024 + 4096                              # room to close on its own
BENCH_QUOTA = {"math500": 12, "mmlu_pro": 12, "bbh_logic4": 8}   # BBH: first 2 per subtask
URL = "https://openrouter.ai/api/v1/completions"

_spend_lock = threading.Lock()
_spend = [0.0]


# ---------------------------------------------------------------- items and prompts
def _jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            yield json.loads(line)


def load_items(n_total: int) -> list[dict]:
    """Main-panel items in frozen manifest order with their slot-0 root request messages."""
    tasks = {r["item_id"]: r for b in BENCH_QUOTA for r in _jsonl(SRC / "manifests/public" / f"{b}.jsonl")}
    private = {r["item_id"]: r for b in BENCH_QUOTA for r in _jsonl(SRC / "private" / f"{b}.jsonl")}
    scale = n_total / sum(BENCH_QUOTA.values())
    quota = {b: max(1, round(q * scale)) for b, q in BENCH_QUOTA.items()}
    picked, per_bench, per_sub = [], Counter(), Counter()
    for row in _jsonl(SRC / "manifests/jobs/main.jsonl"):
        rec = row["record"]
        if rec["kind"] != "root" or rec["root_slot"] != 0 or rec["benchmark"] not in quota:
            continue
        b, task = rec["benchmark"], tasks[rec["item_id"]]
        if per_bench[b] >= quota[b]:
            continue
        if b == "bbh_logic4":
            if per_sub[task["subtask"]] >= quota[b] // 4:
                continue
            per_sub[task["subtask"]] += 1
        raws = sorted(glob.glob(str(SRC / "raw" / row["node_id"][:2] / f"{row['node_id']}_*.json.gz")))
        if not raws:
            continue
        with gzip.open(raws[-1]) as fh:
            messages = json.load(fh)["request_body"]["messages"]
        per_bench[b] += 1
        picked.append({"item_id": rec["item_id"], "benchmark": b, "messages": messages,
                       "task": task, "private": private[rec["item_id"]]})
    return picked


def think_prompt(messages: list[dict]) -> str:
    """Qwen3's chat template for (system, user) with the generation prompt and an opened <think>."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n<think>\n"


# ---------------------------------------------------------------- API
def call(client: httpx.Client, key: str, prompt: str, max_tokens: int, bias: dict | None) -> dict:
    with _spend_lock:
        if _spend[0] >= SPEND_CAP_USD:
            raise RuntimeError(f"spend cap ${SPEND_CAP_USD} reached")
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE,
            "top_p": TOP_P, "provider": PROVIDER, "usage": {"include": True}, "stream": False}
    if bias:
        body["logit_bias"] = bias
    headers = {"Authorization": f"Bearer {key}", "X-OpenRouter-Cache": "false", "X-Title": "forced_deep_pilot_v1"}
    t0 = time.time()
    for attempt in range(3):
        resp = client.post(URL, json=body, headers=headers, timeout=900)
        if resp.status_code in (429, 502, 503) and attempt < 2:
            time.sleep(10 * (attempt + 1))
            continue
        break
    out = {"status": resp.status_code, "latency_s": round(time.time() - t0, 2),
           "request": {k: v for k, v in body.items() if k != "prompt"}, "prompt_chars": len(prompt),
           "provider_header": resp.headers.get("x-provider-name")}
    try:
        data = resp.json()
    except ValueError:
        out["error"] = redact(resp.text[:2000])
        return out
    if resp.status_code != 200 or "choices" not in data:
        out["error"] = redact(json.dumps(data)[:2000])
        return out
    ch = data["choices"][0]
    usage = data.get("usage") or {}
    out.update(text=ch.get("text") or "", reasoning=ch.get("reasoning") or "", finish_reason=ch.get("finish_reason"),
               native_finish_reason=ch.get("native_finish_reason"), provider=data.get("provider"),
               model=data.get("model"), usage=usage, generation_id=data.get("id"))
    with _spend_lock:
        _spend[0] += float(usage.get("cost") or 0.0)
    return out


# ---------------------------------------------------------------- analysis helpers
def split_natural(reasoning_field: str, text: str) -> tuple[str, str, bool]:
    """(reasoning, visible answer, closed) for both server styles: a reasoning parser (thinking in `reasoning`,
    answer in `text`) or none (``thinking</think>answer`` all in `text`)."""
    if "</think>" in text:
        pre, _, visible = text.partition("</think>")
        return reasoning_field + pre, visible, True
    if not reasoning_field:                                # no parser and never closed: all of it is thinking
        return text, "", False
    return reasoning_field, text, bool(text.strip())


def loop_onset(text: str, window: int = 3000, step: int = 500, threshold: float = 0.1) -> float | None:
    """Fraction of the reasoning before the first window whose zlib ratio falls below ``threshold`` (a loop)."""
    for start in range(0, max(1, len(text) - window), step):
        chunk = text[start:start + window].encode()
        if len(zlib.compress(chunk, 9)) / max(1, len(chunk)) < threshold:
            return start / len(text)
    return None


def loopiness(text: str, tail_chars: int = 6000) -> float:
    """zlib ratio of the reasoning tail: ~0.3-0.45 for prose, << 0.1 for a repeating loop."""
    tail = text[-tail_chars:].encode()
    return round(len(zlib.compress(tail, 9)) / max(1, len(tail)), 4)


def grade(item: dict, visible: str, finish: str | None, node: str) -> dict:
    task = load_record(PublicTask, item["task"])
    private = load_record(PrivateEvalSpec, item["private"])
    status, inner = strict_single_pair(visible, "<final>", "</final>")
    if status == TAG_OK and inner and inner.strip():
        parse, final = ParseStatus.VALID, inner
    elif "<final>" not in visible and "</final>" not in visible:
        parse, final = ParseStatus.MISSING, None
    else:
        parse, final = ParseStatus.MALFORMED, None
    cand = Candidate(candidate_id=node, node_id=node, item_id=item["item_id"], visible_content=visible,
                     final_answer=final, rationale=None, parse_status=parse, finish_reason=finish,
                     output_kind=task.output_kind)
    g = grade_text(cand, private)
    return {"parse": parse.value, "final": final, "correct": bool(g.semantic_correct), "grade": g.status.value,
            "grade_detail": g.detail_code}


def run_arm(client, key, item: dict, arm: str, k: int) -> dict:
    base = think_prompt(item["messages"])
    budget = k * A
    rec = {"item_id": item["item_id"], "benchmark": item["benchmark"], "arm": arm, "k": k, "budget": budget}
    if arm == "forced":
        think = call(client, key, base, budget, BAN)
        rec["think"] = think
        if "text" not in think:
            return rec
        reasoning = think["reasoning"] + think["text"]        # DeepInfra's parser moves thinking to `reasoning`
        rec["think_text_nonempty"] = bool(think["text"])
        rec["reasoning_tokens"] = think["usage"].get("completion_tokens")
        rec["hit_budget"] = rec["reasoning_tokens"] == budget
        rec["leaked_close"] = "</think>" in reasoning
        answer = call(client, key, base + reasoning + "\n</think>\n\n", HEADROOM, None)
        rec["answer"] = answer
        visible, finish = answer.get("text", ""), answer.get("finish_reason")
        rec["answer_in_reasoning_field"] = bool(answer.get("reasoning"))
        if "usage" in answer:                              # re-tokenization seam: prompt + reasoning + 3 closing tokens
            rec["retok_delta"] = (answer["usage"].get("prompt_tokens", 0) - think["usage"].get("prompt_tokens", 0)
                                  - rec["reasoning_tokens"])
    else:  # natural: the model closes its own reasoning; cap at budget + headroom
        think = call(client, key, base, NATURAL_CAP, None)
        rec["think"] = think
        if "text" not in think:
            return rec
        reasoning, visible, closed = split_natural(think["reasoning"], think["text"])
        rec["closed_think"] = closed
        rec["completion_tokens"] = think["usage"].get("completion_tokens")
        rec["reasoning_tokens"] = None                     # usage does not split thinking from answer; chars reported
        finish = think.get("finish_reason")
    rec["reasoning_chars"] = len(reasoning)
    rec["loopiness"] = loopiness(reasoning)
    rec["loop_onset_frac"] = loop_onset(reasoning)
    rec.update(grade(item, visible, finish, f"{item['item_id']}|{arm}|{k}"))
    rec["cost_usd"] = sum(float((rec.get(s) or {}).get("usage", {}).get("cost") or 0) for s in ("think", "answer"))
    return rec


# ---------------------------------------------------------------- commands
def cmd_dry_run(args):
    items = load_items(args.items)
    print(Counter(i["benchmark"] for i in items))
    print(think_prompt(items[0]["messages"])[-1500:])


def cmd_probe(args):
    """One MATH item, three calls: forced 256, answer after forced, natural (small cap)."""
    key, items = load_api_key(), load_items(32)
    item = next(i for i in items if i["benchmark"] == "math500")
    base = think_prompt(item["messages"])
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        forced = call(client, key, base, 2048, BAN)
        print("FORCED-2048:", {k: forced.get(k) for k in ("status", "finish_reason", "provider", "error")},
              "completion_tokens=", (forced.get("usage") or {}).get("completion_tokens"),
              "has </think>=", "</think>" in forced.get("text", ""))
        print("   reasoning tail:", repr((forced.get("reasoning", "") + forced.get("text", ""))[-400:]), "text:", repr(forced.get("text", "")[:80]))
        ans = call(client, key, base + forced.get("reasoning", "") + forced.get("text", "") + "\n</think>\n\n", 1024, None)
        print("ANSWER:", {k: ans.get(k) for k in ("status", "finish_reason", "error")},
              "completion_tokens=", (ans.get("usage") or {}).get("completion_tokens"))
        print("   text:", repr(ans.get("text", "")[:600]))
        nat = call(client, key, base, 6000, None)
        t = nat.get("text", "")
        print("NATURAL-6000:", {k: nat.get(k) for k in ("status", "finish_reason", "error")},
              "completion_tokens=", (nat.get("usage") or {}).get("completion_tokens"),
              "closed_think=", "</think>" in t)
        print("   after think:", repr(t.partition("</think>")[2][:300]))
    (OUT / "probe.json").write_text(json.dumps({"forced": forced, "answer": ans, "natural": nat}, indent=1))
    print(f"spend ${_spend[0]:.4f}  (probe.json written)")


def cmd_bias_check(args):
    """A +100 bias on " banana": an endpoint that honours logit_bias answers "banana banana ..."."""
    key = load_api_key()
    with httpx.Client() as client:
        out = call(client, key, "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                   12, {str(BANANA): 100})
    text = out.get("text", "") + out.get("reasoning", "")
    print(MODEL, PROVIDER["only"], repr(text[:80]), "->", "HONOURED" if text.count("banana") >= 5 else "IGNORED")


def cmd_run(args):
    key, items = load_api_key(), load_items(args.items)
    grid = [(i, "forced", k) for i in items for k in args.ks] + [(i, "natural", 1) for i in items]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "pilot.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["item_id"], r["arm"], r["k"]) for r in _jsonl(out_path) if complete(r)}
    todo = [g for g in grid if (g[0]["item_id"], g[1], g[2]) not in done]
    print(f"{len(items)} items, {len(grid)} arms, {len(todo)} to run")
    lock = threading.Lock()
    with httpx.Client(limits=httpx.Limits(max_connections=args.workers)) as client, \
            ThreadPoolExecutor(args.workers) as pool, open(out_path, "a") as fh:
        futs = {pool.submit(run_arm, client, key, *g): g for g in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:                       # noqa: BLE001 -- record and continue
                g = futs[fut]
                rec = {"item_id": g[0]["item_id"], "arm": g[1], "k": g[2], "exception": repr(exc)}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"[{n}/{len(todo)}] {rec['arm']}/{rec['k']} {rec['item_id']} "
                  f"tok={rec.get('reasoning_tokens')} ok={rec.get('correct')} spend=${_spend[0]:.3f}", flush=True)
    cmd_report(args)


def complete(r: dict) -> bool:
    """Graded, and for a forced arm the THINK call ended by the length cap at exactly k*A (finish_reason ``None``
    with fewer tokens is a provider-side cut, not the model stopping: it is re-run, never analysed)."""
    if "correct" not in r:
        return False
    return r["arm"] != "forced" or (r.get("hit_budget") and r["think"].get("finish_reason") == "length")


def cmd_report(args):
    latest = {}
    attempts = Counter()
    for r in _jsonl(OUT / "pilot.jsonl"):
        latest[(r["item_id"], r["arm"], r["k"])] = r
        attempts[(r["arm"], r["k"])] += 1
    rows = list(latest.values())
    incomplete = Counter((r["arm"], r["k"]) for r in rows if not complete(r))
    print("attempts per arm:", dict(attempts), " still incomplete:", dict(incomplete))
    rows = [r for r in rows if complete(r)]
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["k"])].append(r)
    lines = ["| arm | k | n | budget | reasoning tokens (min/mean/max) | hit budget | leaked </think> | "
             "loopiness median (min) | looped | loop onset ~tokens (median) | valid | accuracy | cost |",
             "|" + "---|" * 13]
    for (arm, k), rs in sorted(by.items()):
        ok = [r for r in rs if "correct" in r]
        toks = [r["reasoning_tokens"] for r in ok if r.get("reasoning_tokens") is not None]
        loops = sorted(r["loopiness"] for r in ok)
        tok = f"{min(toks)}/{sum(toks)/len(toks):.0f}/{max(toks)}" if toks else \
            f"~{sum(r['reasoning_chars'] for r in ok)/max(1,len(ok))/3.5:.0f} (chars/3.5)"
        hit = f"{sum(r.get('hit_budget', False) for r in ok)}/{len(ok)}" if arm == "forced" else "-"
        leak = f"{sum(r.get('leaked_close', False) for r in ok)}" if arm == "forced" else \
            f"closed {sum(r.get('closed_think', False) for r in ok)}/{len(ok)}"
        onsets = sorted(r["loop_onset_frac"] * (k * A if arm == "forced" else r["reasoning_chars"] / 3.5)
                        for r in ok if r.get("loop_onset_frac") is not None)
        lines.append(f"| {arm} | {k} | {len(ok)}/{len(rs)} | {k*A if arm == 'forced' else NATURAL_CAP} | {tok} | "
                     f"{hit} | {leak} | "
                     f"{loops[len(loops)//2] if loops else '-'} ({loops[0] if loops else '-'}) | "
                     f"{len(onsets)}/{len(ok)} | {onsets[len(onsets)//2] if onsets else '-'} | "
                     f"{sum(r['parse']=='valid' for r in ok)}/{len(ok)} | "
                     f"{sum(r['correct'] for r in ok)/max(1,len(ok)):.3f} | "
                     f"${sum(r.get('cost_usd', 0) for r in ok):.3f} |")
    errs = Counter((r.get("exception") or (r.get("think") or {}).get("error") or "")[:120] for r in rows
                   if "correct" not in r)
    providers = Counter((r.get("think") or {}).get("provider") for r in rows)
    text = "\n".join(lines) + f"\n\nproviders: {dict(providers)}\nfailures: {dict(errs)}\n"
    (OUT / "REPORT.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "bias-check", "probe", "run", "report"])
    ap.add_argument("--preset", default="qwen3.5-9b-together", choices=sorted(PRESETS))
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    configure(args.preset)
    {"bias-check": cmd_bias_check, "dry-run": cmd_dry_run, "probe": cmd_probe, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
