"""E8a: an LLM allocator as a comparison (``docs/passk_cont/PREREG_E8a_llm_allocator.md``).

The study's own model, under the study's own forced protocol (THINK exactly 1,024 tokens + ANSWER), reads a problem
(arm A0) or the problem plus the final answers of its five A-level diagnostic draws (arm A1) and picks one action from
the frozen B = 16A menu. Its choice is scored against the already released readouts of all five allocations, so no
solver call is made here. The prompt constants below are frozen by the addendum; their sha256 is recorded per call.

    python -m agent_allocation.passk.e8a run   --config configs/passk_cont_v1.json --study-root R [--panels L96+T200]
    python -m agent_allocation.passk.e8a score --study-root R
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

MENU = ("1x16", "2x8", "4x4", "8x2", "16x1")
FALLBACK = "1x16"
ARMS = ("A0", "A1")
PANELS = ("L96", "T200")
SEED = 20260925
THINK_TOKENS = 1024
MAX_PARSE_ATTEMPTS = 2           # one outcome-blind retry of an invalid choice
MAX_TRANSPORT_ATTEMPTS = 3
CAP_USD = Decimal("8")
PER_SHARD_CONCURRENCY = 16

SYSTEM = (
    "You are a planning assistant. You do not solve the problem yourself. You decide how a fixed reasoning budget "
    "should be spent by a separate solver (the same language model as you) so that the final answer it delivers is "
    "most likely to be correct."
)
OPTION_TEXT = {
    "1x16": "one attempt with 16 units of reasoning",
    "2x8": "two attempts with 8 units of reasoning each",
    "4x4": "four attempts with 4 units of reasoning each",
    "8x2": "eight attempts with 2 units of reasoning each",
    "16x1": "sixteen attempts with 1 unit of reasoning each",
}
USER_HEAD = (
    "PROBLEM\n{problem}\n\n"
    "BUDGET\nThe solver has a total budget of 16 units of reasoning, where 1 unit is 1,024 reasoning tokens. Every "
    "attempt reasons for exactly its full allowance and then writes its final answer. Attempts are independent: they "
    "do not see each other.\n\n"
    "OPTIONS\n{options}\n"
    "With more than one attempt, the delivered answer is the most common final answer among the attempts (ties are "
    "broken uniformly at random). With one attempt, it is that attempt's final answer.\n"
)
USER_DIAG = (
    "\nEARLIER ATTEMPTS\nFive independent short attempts (1 unit of reasoning each) were already made on this "
    "problem. Their final answers (correctness unknown) were:\n{answers}\n"
)
USER_TAIL = (
    "\nTASK\nChoose the option most likely to deliver a correct final answer for this problem. Reply with only the "
    "option label inside <final></final>, i.e. <final>LABEL</final> where LABEL is one of the labels listed above."
)
NO_ANSWER = "(no valid final answer)"


def prompt_sha256() -> str:
    blob = json.dumps({"system": SYSTEM, "options": OPTION_TEXT, "head": USER_HEAD, "diag": USER_DIAG,
                       "tail": USER_TAIL, "no_answer": NO_ANSWER, "menu": MENU}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def option_order(arm: str, item_id: str) -> list[str]:
    """Seeded per-item presentation order (a sort by sha256 key, salt frozen in the addendum)."""
    salt = f"passk_cont_v1/e8a/order|seed={SEED}|{arm}|{item_id}"
    return sorted(MENU, key=lambda label: hashlib.sha256(f"{salt}|{label}".encode()).hexdigest())


def build_messages(arm: str, item_id: str, problem: str, diag_answers: list[str] | None) -> list[dict[str, str]]:
    options = "\n".join(f"- {label}: {OPTION_TEXT[label]}" for label in option_order(arm, item_id))
    text = USER_HEAD.format(problem=problem.strip(), options=options)
    if arm == "A1":
        answers = diag_answers or []
        text += USER_DIAG.format(answers="\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers)))
    text += USER_TAIL
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def parse_choice(visible: str | None) -> str | None:
    from agent_allocation._vendor.extraction_tags import TAG_OK, strict_single_pair
    status, inner = strict_single_pair(visible or "", "<final>", "</final>")
    if status != TAG_OK or inner is None:
        return None
    label = re.sub(r"\s+", "", inner.lower()).replace("×", "x")
    return label if label in MENU else None


# ------------------------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------------------------

def diagnostic_answers(study_root: Path) -> dict[str, list[str]]:
    """item_id -> the five diagnostic final answers in replicate order (answers only; never correctness)."""
    import pandas as pd
    draws = pd.read_parquet(study_root / "analysis_inputs" / "draws.parquet",
                            columns=["item_id", "benchmark", "partition", "allowance_a", "replicate", "parse_status",
                                     "vote_text", "option"])
    l96 = draws[(draws.partition == "legacy_main") & (draws.allowance_a == 1) & draws.replicate.between(20, 24)]
    t200 = draws[draws.partition == "t200_diag"]
    out: dict[str, list[str]] = {}
    for frame in (l96, t200):
        for item, g in frame.sort_values("replicate").groupby("item_id"):
            answers = []
            for row in g.itertuples(index=False):
                text = row.option if row.benchmark == "gpqa_diamond" else row.vote_text
                ok = row.parse_status == "valid" and isinstance(text, str) and text.strip()
                answers.append(text.strip() if ok else NO_ANSWER)
            out[item] = answers
    return out


def panel_items(study_root: Path) -> dict[str, list[str]]:
    splits = json.loads((study_root / "artifacts" / "manifests" / "splits.json").read_text())
    return {p: list(splits["panels"][p]) for p in PANELS}


# ------------------------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------------------------

def _done(calls_path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("final"):
                    done[(row["arm"], row["item_id"])] = row
    return done


def _spent(calls_path: Path) -> Decimal:
    total = Decimal(0)
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            if line.strip():
                total += Decimal(str(json.loads(line).get("cost_usd") or 0))
    return total


async def run(raw_cfg: Mapping[str, Any], study_root: Path, panels: tuple[str, ...], arms: tuple[str, ...],
              *, cap_usd: Decimal = CAP_USD, transport=None, api_key: str | None = None,
              limit: int | None = None) -> dict[str, Any]:
    from agent_allocation.api.forced import make_generation_client
    from agent_allocation.apikey import load_api_key
    from agent_allocation.contracts import GenerationKind, TransportStatus, dump_record
    from agent_allocation.live import load_amendments, load_tasks, read_endpoint_locks, standalone_spec
    from agent_allocation.passk.run import resolve

    run_dir = study_root / "artifacts"
    out_dir = study_root / "e8a"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / "calls.jsonl"
    assignment = json.loads((run_dir / "shards" / "assignment.json").read_text())["assignment"]
    items_by_panel = panel_items(study_root)
    diag = diagnostic_answers(study_root)
    done = _done(calls_path)
    spent = _spent(calls_path)
    psha = prompt_sha256()
    key = api_key if api_key is not None else (None if transport is not None else load_api_key())

    shards: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(assignment[i] for p in panels for i in items_by_panel[p])):
        cfg, paths = resolve(raw_cfg, str(run_dir), shard)
        lock = read_endpoint_locks(paths)[cfg["primary_model"]]
        client = make_generation_client(cfg, api_key=key, transport=transport)
        shards[shard] = {"cfg": cfg, "paths": paths, "lock": lock, "client": client,
                         "tasks": load_tasks(paths), "amend": load_amendments(paths),
                         "sem": asyncio.Semaphore(PER_SHARD_CONCURRENCY)}
    lock_io = asyncio.Lock()
    state = {"spent": spent, "stopped": False}

    async def one(arm: str, panel: str, item_id: str) -> None:
        if (arm, item_id) in done:
            return
        sh = shards[assignment[item_id]]
        task = sh["tasks"][item_id]
        messages = build_messages(arm, item_id, task.prompt, diag.get(item_id))
        choice = None
        async with sh["sem"]:
            for parse_try in range(MAX_PARSE_ATTEMPTS):
                spec = standalone_spec(cfg=sh["cfg"], partition=f"e8a_{arm}", task=task, lock=sh["lock"],
                                       kind=GenerationKind.ROOT, budget=THINK_TOKENS, root_slot=parse_try,
                                       draw_id=parse_try, amendments=sh["amend"])
                outcome = None
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
                    payload = {"request_body": outcome.request_body,
                               "response_body_text": outcome.response_body.decode("utf-8", "replace"),
                               "response_headers": dict(outcome.response_headers), "result": dump_record(res)}
                    raw_path.write_bytes(gzip.compress(json.dumps(payload, default=str).encode()))
                    ok = res.transport_status is TransportStatus.OK
                    choice = parse_choice(res.visible_content) if ok else None
                    final = ok and (choice is not None or parse_try == MAX_PARSE_ATTEMPTS - 1)
                    row = {"arm": arm, "panel": panel, "item_id": item_id, "benchmark": task.benchmark,
                           "shard": assignment[item_id], "node_id": spec.node_id, "parse_try": parse_try,
                           "transport_try": transport_try, "transport_status": res.transport_status.value,
                           "choice": choice, "valid": choice is not None,
                           "chosen": choice if choice is not None else (FALLBACK if final else None),
                           "final": bool(final), "visible": (res.visible_content or "")[:400],
                           "prompt_tokens": res.usage.prompt_tokens if res.usage else None,
                           "completion_tokens": res.usage.completion_tokens if res.usage else None,
                           "reasoning_tokens": res.usage.reasoning_tokens if res.usage else None,
                           "cost_usd": str(cost), "observed_model": res.observed_model,
                           "observed_provider": res.observed_provider, "prompt_sha256": psha,
                           "option_order": option_order(arm, item_id), "raw": str(raw_path.relative_to(study_root))}
                    async with lock_io:
                        state["spent"] += cost
                        with calls_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
                    if ok:
                        break
                if choice is not None or (outcome is not None and outcome.result.transport_status is
                                           TransportStatus.OK and parse_try == MAX_PARSE_ATTEMPTS - 1):
                    return

    jobs = [one(arm, panel, item) for panel in panels for arm in arms
            for item in (items_by_panel[panel][:limit] if limit else items_by_panel[panel])]
    try:
        await asyncio.gather(*jobs)
    finally:
        for sh in shards.values():
            close = getattr(sh["client"], "aclose", None)
            if close:
                await close()
    final = _done(calls_path)
    return {"calls_file": str(calls_path), "decided": len(final), "planned": len(jobs),
            "spent_usd": str(state["spent"]), "stopped_at_cap": state["stopped"], "prompt_sha256": psha}


# ------------------------------------------------------------------------------------------------------------------
# score
# ------------------------------------------------------------------------------------------------------------------

def score(study_root: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from agent_allocation._vendor.ard_stats import holm
    from agent_allocation.passk.stats import MACRO, MIXTURE, contrast, level

    decided = pd.DataFrame(list(_done(study_root / "e8a" / "calls.jsonl").values()))
    calls = pd.read_json(study_root / "e8a" / "calls.jsonl", lines=True)
    ro = pd.read_parquet(study_root / "analysis_inputs" / "readouts.parquet")
    ro = ro[(ro["mode"] == "passk") & ro.panel.isin(PANELS) & ro.config_id.isin(MENU)]
    outcome = ro.pivot_table(index=["panel", "item_id", "benchmark"], columns="config_id", values="S_known",
                             aggfunc="first").reset_index()
    pred = pd.read_parquet(study_root / "sealed" / "t200" / "predictions.parquet")
    p1 = pred[(pred.predictor == "P1") & pred.config_id.isin(MENU)]
    p1_choice = p1.sort_values(["item_id", "p_delivered"], ascending=[True, False]).groupby("item_id").first()
    report: dict[str, Any] = {"prompt_sha256": prompt_sha256(), "panels": {}}
    for panel in PANELS:
        frame = outcome[outcome.panel == panel].copy()
        if frame.empty:
            continue
        for arm in ARMS:
            d = decided[(decided.arm == arm) & (decided.panel == panel)].set_index("item_id")
            frame[f"{arm}_choice"] = frame.item_id.map(d["chosen"])
            frame[f"{arm}_valid"] = frame.item_id.map(d["valid"])
            frame[arm] = [row[c] if isinstance(c, str) else np.nan
                          for row, c in zip(frame.to_dict("records"), frame[f"{arm}_choice"])]
            frame[f"{arm}_invalid0"] = np.where(frame[f"{arm}_valid"] == False, 0.0, frame[arm])   # noqa: E712
        frame["always_deep"] = frame["1x16"]
        frame["random"] = frame[list(MENU)].mean(axis=1, skipna=False)
        frame["oracle"] = frame[list(MENU)].max(axis=1, skipna=False)
        if panel == "T200":
            frame["P1_choice"] = frame.item_id.map(p1_choice["config_id"])
            frame["P1_select"] = [row[c] if isinstance(c, str) else np.nan
                                  for row, c in zip(frame.to_dict("records"), frame["P1_choice"])]
        levels = {}
        for col in ["A0", "A1", "A0_invalid0", "A1_invalid0", "always_deep", "random", "oracle", *MENU] + \
                (["P1_select"] if panel == "T200" else []):
            levels[col] = {"mixture": level(frame, col, weights=MIXTURE), "macro": level(frame, col, weights=MACRO)}
        choices = {arm: frame.groupby("benchmark")[f"{arm}_choice"].value_counts().unstack(fill_value=0)
                   .to_dict(orient="index") for arm in ARMS}
        invalid = {arm: int((frame[f"{arm}_valid"] == False).sum()) for arm in ARMS}   # noqa: E712
        tests = {}
        if panel == "T200":
            t0 = contrast(frame, "A0", "always_deep", weights=MIXTURE, n_family=2)
            t1 = contrast(frame, "A1", "always_deep", weights=MIXTURE, n_family=2)
            adj = holm({"A0_vs_deep": t0["p"], "A1_vs_deep": t1["p"]}, n_primary=2)
            tests["primary"] = {"A0_vs_deep": {**t0, **adj["A0_vs_deep"]}, "A1_vs_deep": {**t1, **adj["A1_vs_deep"]}}
            tests["secondary"] = {
                name: contrast(frame, a, b, weights=MIXTURE, n_family=1)
                for name, a, b in (("A1_vs_A0", "A1", "A0"), ("A0_vs_P1select", "A0", "P1_select"),
                                   ("A1_vs_P1select", "A1", "P1_select"), ("A0_vs_random", "A0", "random"),
                                   ("A1_vs_random", "A1", "random"), ("A0_vs_oracle", "A0", "oracle"),
                                   ("A1_vs_oracle", "A1", "oracle"),
                                   ("A0inv0_vs_deep", "A0_invalid0", "always_deep"),
                                   ("A1inv0_vs_deep", "A1_invalid0", "always_deep"))}
        spend = calls[calls.panel == panel].groupby("arm").apply(
            lambda g: {"calls": int(len(g)), "usd": float(pd.to_numeric(g.cost_usd).sum()),
                       "mean_prompt_tokens": float(g.prompt_tokens.mean()),
                       "mean_completion_tokens": float(g.completion_tokens.mean())}, include_groups=False).to_dict()
        report["panels"][panel] = {"n_items": int(len(frame)), "levels": levels, "choices": choices,
                                   "invalid_after_retry": invalid, "tests": tests, "spend": spend}
        frame.to_parquet(study_root / "e8a" / f"items_{panel}.parquet")
    out = study_root / "e8a" / "results.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.e8a")
    parser.add_argument("verb", choices=("run", "score"))
    parser.add_argument("--config", default="configs/passk_cont_v1.json")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--panels", default="L96+T200")
    parser.add_argument("--arms", default="A0+A1")
    parser.add_argument("--cap-usd", default=str(CAP_USD))
    parser.add_argument("--limit", type=int, default=None, help="first N items per panel (smoke)")
    args = parser.parse_args(argv)
    root = Path(args.study_root)
    if args.verb == "run":
        from agent_allocation.config import load_config
        summary = asyncio.run(run(load_config(args.config), root, tuple(args.panels.split("+")),
                                  tuple(args.arms.split("+")), cap_usd=Decimal(args.cap_usd),
                                  limit=args.limit))
        print(json.dumps(summary, indent=1))
        return 0 if not summary["stopped_at_cap"] else 3
    report = score(root)
    print(json.dumps({p: {"levels": {k: round(v["mixture"]["mean"], 4) if isinstance(v["mixture"], dict) and
                                      "mean" in v["mixture"] else v["mixture"] for k, v in r["levels"].items()},
                          "invalid": r["invalid_after_retry"]} for p, r in report["panels"].items()},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
