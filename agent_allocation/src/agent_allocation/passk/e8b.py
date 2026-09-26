"""E8b: LLM selectors told the owner's lexicographic objective (``docs/passk_cont/PREREG_E8b_lex_llm.md``).

Same model and forced protocol as E8a (THINK exactly 1,024 tokens + ANSWER), but the prompt states the objective
(accuracy -> correctness stability -> latency, 2-point accuracy ties), offers the 15-option <=16A menu with each
option's typical wall-clock, and is sampled K = 5 times per problem (option order reshuffled per sample) so the
selector's own choice variability enters the system's consistency. Arms: B0 (problem only), B1 (+ the five
diagnostic answers). Panels: C48 (bank-scored) and T200 (realized).

    python -m agent_allocation.passk.e8b run --study-root R [--panels C48+T200] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.passk.e8a import NO_ANSWER, SYSTEM, diagnostic_answers

MENU = ("1x1", "2x1", "4x1", "8x1", "16x1", "1x2", "2x2", "4x2", "8x2", "1x4", "2x4", "4x4", "1x8", "2x8", "1x16")
# typical parallel wall-clock per option (s): C48 population means (R/lex/front_estimates.parquet), rounded to 10 s
TYPICAL_SECONDS = {"1x1": 30, "2x1": 40, "4x1": 40, "8x1": 50, "16x1": 60, "1x2": 50, "2x2": 50, "4x2": 50,
                   "8x2": 60, "1x4": 80, "2x4": 90, "4x4": 100, "1x8": 150, "2x8": 170, "1x16": 280}
ARMS = ("B0", "B1")
PANELS = ("C48", "T200")
SAMPLES = 5
SEED = 20260925
THINK_TOKENS = 1024
MAX_PARSE_ATTEMPTS = 2
MAX_TRANSPORT_ATTEMPTS = 3
CAP_USD = Decimal("14")
PER_SHARD_CONCURRENCY = {"mancer": 48, "parasail": 32}

USER_HEAD = (
    "PROBLEM\n{problem}\n\n"
    "OPTIONS\nA separate solver (the same language model as you) will run exactly one of these options. 1 unit of "
    "reasoning is 1,024 reasoning tokens. Every attempt reasons for exactly its full allowance and then writes its "
    "final answer. Attempts are independent and run in parallel. With several attempts, the delivered answer is the "
    "most common final answer among them (ties are broken uniformly at random); with one attempt it is that "
    "attempt's answer. The typical wall-clock time until the answer is delivered is shown for each option.\n"
    "{options}\n"
)
USER_DIAG = (
    "\nEARLIER ATTEMPTS\nFive independent short attempts (1 unit of reasoning each) were already made on this "
    "problem. Their final answers (correctness unknown) were:\n{answers}\n"
)
USER_TAIL = (
    "\nOBJECTIVE\nChoose the option using these criteria, in this order:\n"
    "1. The highest probability that the delivered answer is correct. Options within 2 percentage points of the "
    "best count as equally accurate.\n"
    "2. Among the equally accurate options, the one whose correctness is most stable across repeated runs (its "
    "probability of being correct is closest to 0 or 1).\n"
    "3. Among options that are still tied, the fastest one.\n"
    "Reply with only the option label inside <final></final>, i.e. <final>LABEL</final> where LABEL is one of the "
    "labels listed above."
)


def _plural(n: int, a: int) -> str:
    attempts = {1: "one attempt", 2: "two attempts", 4: "four attempts", 8: "eight attempts",
                16: "sixteen attempts"}[n]
    unit = "1 unit" if a == 1 else f"{a} units"
    each = "" if n == 1 else " each"
    return f"{attempts} with {unit} of reasoning{each}"


OPTION_TEXT = {label: _plural(*map(int, label.split("x"))) for label in MENU}


def prompt_sha256() -> str:
    blob = json.dumps({"system": SYSTEM, "options": OPTION_TEXT, "seconds": TYPICAL_SECONDS, "head": USER_HEAD,
                       "diag": USER_DIAG, "tail": USER_TAIL, "no_answer": NO_ANSWER, "menu": MENU},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def option_order(arm: str, item_id: str, sample: int) -> list[str]:
    salt = f"passk_cont_v1/e8b/order|seed={SEED}|{arm}|{item_id}|{sample}"
    return sorted(MENU, key=lambda label: hashlib.sha256(f"{salt}|{label}".encode()).hexdigest())


def build_messages(arm: str, item_id: str, sample: int, problem: str,
                   diag_answers: list[str] | None) -> list[dict[str, str]]:
    options = "\n".join(f"- {label}: {OPTION_TEXT[label]} (typical time about {TYPICAL_SECONDS[label]} s)"
                        for label in option_order(arm, item_id, sample))
    text = USER_HEAD.format(problem=problem.strip(), options=options)
    if arm == "B1":
        text += USER_DIAG.format(answers="\n".join(f"{i + 1}. {a}" for i, a in enumerate(diag_answers or [])))
    text += USER_TAIL
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def parse_choice(visible: str | None) -> str | None:
    from agent_allocation._vendor.extraction_tags import TAG_OK, strict_single_pair
    status, inner = strict_single_pair(visible or "", "<final>", "</final>")
    if status != TAG_OK or inner is None:
        return None
    label = re.sub(r"\s+", "", inner.lower()).replace("×", "x")
    return label if label in MENU else None


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


async def run(raw_cfg: Mapping[str, Any], study_root: Path, panels: tuple[str, ...], arms: tuple[str, ...], *,
              cap_usd: Decimal = CAP_USD, limit: int | None = None, samples: int = SAMPLES) -> dict[str, Any]:
    from agent_allocation.api.forced import make_generation_client
    from agent_allocation.apikey import load_api_key
    from agent_allocation.contracts import GenerationKind, TransportStatus, dump_record
    from agent_allocation.live import load_amendments, load_tasks, read_endpoint_locks, standalone_spec
    from agent_allocation.passk.run import resolve

    run_dir = study_root / "artifacts"
    out_dir = study_root / "e8b"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / "calls.jsonl"
    splits = json.loads((run_dir / "manifests" / "splits.json").read_text())
    assignment = json.loads((run_dir / "shards" / "assignment.json").read_text())["assignment"]
    diag = diagnostic_answers(study_root)
    prior = _rows(calls_path)
    done = {(r["arm"], r["item_id"], r["sample"]) for r in prior if r.get("final")}
    state = {"spent": sum(Decimal(str(r.get("cost_usd") or 0)) for r in prior), "stopped": False}
    psha = prompt_sha256()
    key = load_api_key()
    items_by_panel = {p: list(splits["panels"][p])[: limit or None] for p in panels}
    shards: dict[str, dict[str, Any]] = {}
    for shard in sorted({assignment[i] for p in panels for i in items_by_panel[p]}):
        cfg, paths = resolve(raw_cfg, str(run_dir), shard)
        shards[shard] = {"cfg": cfg, "lock": read_endpoint_locks(paths)[cfg["primary_model"]],
                         "client": make_generation_client(cfg, api_key=key), "tasks": load_tasks(paths),
                         "amend": load_amendments(paths), "sem": asyncio.Semaphore(PER_SHARD_CONCURRENCY.get(shard, 16))}
    io_lock = asyncio.Lock()

    async def one(arm: str, panel: str, item_id: str, sample: int) -> None:
        if (arm, item_id, sample) in done:
            return
        sh = shards[assignment[item_id]]
        task = sh["tasks"][item_id]
        messages = build_messages(arm, item_id, sample, task.prompt, diag.get(item_id))
        async with sh["sem"]:
            for parse_try in range(MAX_PARSE_ATTEMPTS):
                slot = sample * 10 + parse_try
                spec = standalone_spec(cfg=sh["cfg"], partition=f"e8b_{arm}", task=task, lock=sh["lock"],
                                       kind=GenerationKind.ROOT, budget=THINK_TOKENS, root_slot=slot, draw_id=slot,
                                       amendments=sh["amend"])
                for transport_try in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
                    if state["spent"] >= cap_usd:
                        state["stopped"] = True
                        return
                    outcome = await sh["client"].generate(spec, messages, attempt_id=f"{spec.node_id}:{transport_try}",
                                                          lock=sh["lock"], sampling=sh["cfg"]["sampling"])
                    res = outcome.result
                    cost = Decimal(str(res.usage.actual_cost_usd)) if res.usage and res.usage.actual_cost_usd else \
                        Decimal(0)
                    raw_path = out_dir / "raw" / arm / f"{spec.node_id}_{transport_try}.json.gz"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.write_bytes(gzip.compress(json.dumps(
                        {"request_body": outcome.request_body,
                         "response_body_text": outcome.response_body.decode("utf-8", "replace"),
                         "response_headers": dict(outcome.response_headers), "result": dump_record(res)},
                        default=str).encode()))
                    ok = res.transport_status is TransportStatus.OK
                    choice = parse_choice(res.visible_content) if ok else None
                    final = ok and (choice is not None or parse_try == MAX_PARSE_ATTEMPTS - 1)
                    row = {"arm": arm, "panel": panel, "item_id": item_id, "benchmark": task.benchmark,
                           "shard": assignment[item_id], "sample": sample, "node_id": spec.node_id,
                           "parse_try": parse_try, "transport_try": transport_try,
                           "transport_status": res.transport_status.value, "choice": choice,
                           "valid": choice is not None, "final": bool(final), "latency_s": res.latency_seconds,
                           "visible": (res.visible_content or "")[:300],
                           "prompt_tokens": res.usage.prompt_tokens if res.usage else None,
                           "completion_tokens": res.usage.completion_tokens if res.usage else None,
                           "reasoning_tokens": res.usage.reasoning_tokens if res.usage else None,
                           "cost_usd": str(cost), "observed_provider": res.observed_provider,
                           "prompt_sha256": psha, "option_order": option_order(arm, item_id, sample),
                           "raw": str(raw_path.relative_to(study_root))}
                    async with io_lock:
                        state["spent"] += cost
                        with calls_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
                    if ok:
                        break
                if choice is not None:
                    return

    jobs = [one(arm, panel, item, k) for panel in panels for arm in arms for item in items_by_panel[panel]
            for k in range(samples)]
    try:
        await asyncio.gather(*jobs)
    finally:
        for sh in shards.values():
            close = getattr(sh["client"], "aclose", None)
            if close:
                await close()
    final = {(r["arm"], r["item_id"], r["sample"]) for r in _rows(calls_path) if r.get("final")}
    return {"planned": len(jobs), "decided": len(final), "spent_usd": str(state["spent"]),
            "stopped_at_cap": state["stopped"], "prompt_sha256": psha, "calls_file": str(calls_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.e8b")
    parser.add_argument("verb", choices=("run",))
    parser.add_argument("--config", default="configs/passk_cont_v1.json")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--panels", default="C48+T200")
    parser.add_argument("--arms", default="B0+B1")
    parser.add_argument("--cap-usd", default=str(CAP_USD))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    summary = asyncio.run(run(load_config(args.config), Path(args.study_root), tuple(args.panels.split("+")),
                              tuple(args.arms.split("+")), cap_usd=Decimal(args.cap_usd), limit=args.limit))
    print(json.dumps(summary, indent=1))
    return 0 if not summary["stopped_at_cap"] else 3


if __name__ == "__main__":
    sys.exit(main())
