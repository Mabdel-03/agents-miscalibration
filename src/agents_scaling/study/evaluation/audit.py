"""Judge audit substitute (WP5, evaluator only): MC letter-vs-judge error rates, the §3.3
two-percentage-point differential-error bound and the blinded human audit export.

Spec §3.3 ("The audit must quantify whether differential errors across conditions could
reverse a claimed effect or move it by more than two percentage points ... Audit roots,
member outputs, coordinator answers ... not only favorable selected results").
Corrections P1-10 / amendment E1: because Qwen3-32B judges its own exact-answer outputs,
the judge is also run on every MC candidate where the exact letter rule is authoritative;
its false-positive / false-negative rates against letter scoring are the only measured
judge-error estimate, and a stratified 100-candidate sample (method × judge label, arm
labels stripped) is exported for a later human audit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agents_scaling.experiment import io
from agents_scaling.study.data.layout import FILE_MODE
from agents_scaling.study.data.protected import load_protected_hle
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.evaluation.hle_judge import (
    EVAL_HLE_SUBDIR,
    HLE_EVAL_SCHEMA_VERSION,
    JUDGE_AMBIGUOUS,
    JUDGE_YES,
)
from agents_scaling.study.selection.seal import POOLS_FILE, SEALS_DIR, load_pool_candidates, load_pools
from agents_scaling.study.types import ProtocolError

AUDIT_SAMPLE_FILE = "hle_audit_sample.jsonl"
AUDIT_KEY_FILE = "hle_audit_key.json"
DIFFERENTIAL_THRESHOLD_PP = 2.0


def _eval_records(run_root: str | os.PathLike) -> list[dict[str, Any]]:
    folder = Path(run_root).joinpath(*EVAL_HLE_SUBDIR)
    out: list[dict[str, Any]] = []
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != HLE_EVAL_SCHEMA_VERSION:
            raise ProtocolError(f"{path} is not an HLE eval record")
        out.append(data)
    return out


def mc_audit(run_root: str | os.PathLike) -> dict[str, Any]:
    """Judge-vs-letter confusion over every judged MC answer (distinct answers per item).

    ``fp_rate`` = P(judge yes | letter incorrect), ``fn_rate`` = P(judge no or ambiguous |
    letter correct); ``ambiguous_rate`` over all judged MC answers.  Counts are of distinct
    (item, final_answer) pairs, each weighted once (the judge is deterministic per answer).
    """
    tp = fp = tn = fn = amb = 0
    n_items = 0
    for rec in _eval_records(run_root):
        if rec.get("answer_format") != "multipleChoice":
            continue
        n_items += 1
        for entry in rec.get("judgements", {}).values():
            letter = entry.get("letter_correct")
            judge = entry.get("judge")
            if letter is None or judge is None:
                raise ProtocolError(f"{rec['source_id']}: MC judgement lacks letter/judge fields")
            if judge == JUDGE_AMBIGUOUS:
                amb += 1
            if letter and judge == JUDGE_YES:
                tp += 1
            elif letter:
                fn += 1
            elif judge == JUDGE_YES:
                fp += 1
            else:
                tn += 1
    pos, neg = tp + fn, fp + tn
    total = pos + neg
    return {
        "n_mc_items": n_items,
        "n_judged_answers": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "ambiguous": amb,
        "fp_rate": (fp / neg) if neg else None,
        "fn_rate": (fn / pos) if pos else None,
        "ambiguous_rate": (amb / total) if total else None,
        "agreement": ((tp + tn) / total) if total else None,
    }


def differential_error_bound(
    fp_rate: float,
    fn_rate: float,
    accuracy_a: float,
    accuracy_b: float,
    *,
    threshold_pp: float = DIFFERENTIAL_THRESHOLD_PP,
) -> dict[str, Any]:
    """Conservative bound on how far judge error could move a two-arm accuracy contrast (§3.3).

    With judge false-positive rate ``fp`` (among truly incorrect answers) and false-negative
    rate ``fn`` (among truly correct ones), an arm with true accuracy ``p`` is observed at
    ``p(1-fn) + (1-p)fp``; the worst-case error of an arm's observed accuracy is bounded by
    ``max(p·fn, (1-p)·fp)`` shifted either way, so the contrast ``A − B`` can move by at
    most ``shift_a + shift_b`` percentage points if the errors are maximally differential.
    Returns the bound and whether it exceeds ``threshold_pp`` (then the claim needs a
    census/human audit or a narrower statement).
    """
    for name, value in (("fp_rate", fp_rate), ("fn_rate", fn_rate), ("accuracy_a", accuracy_a), ("accuracy_b", accuracy_b)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a number in [0, 1]")

    def shift(p: float) -> float:
        return 100.0 * max(p * fn_rate, (1.0 - p) * fp_rate)

    shift_a, shift_b = shift(accuracy_a), shift(accuracy_b)
    bound = shift_a + shift_b
    return {
        "shift_a_pp": shift_a,
        "shift_b_pp": shift_b,
        "differential_bound_pp": bound,
        "threshold_pp": threshold_pp,
        "could_exceed_threshold": bound > threshold_pp,
    }


def _method_of_candidates(run_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """``({candidate_id: method}, {candidate_id: final_answer})`` from every sealed pool register."""
    methods: dict[str, str] = {}
    answers: dict[str, str] = {}
    seals_root = run_root / SEALS_DIR
    if not seals_root.exists():
        return methods, answers
    for pools_path in sorted(seals_root.glob(f"*/{POOLS_FILE}")):
        pools = load_pools(run_root, pools_path.parent.name)
        pool_list = list(pools["pools"].values())
        records = load_pool_candidates(run_root, pool_list)
        for pool in pool_list:
            for cid in pool["candidate_ids"]:
                methods.setdefault(cid, pool["method"])
                rec = records[cid]
                if rec.valid and rec.candidate is not None:
                    answers.setdefault(cid, rec.candidate.final_answer)
    return methods, answers


def export_human_audit_sample(run_root: str | os.PathLike, n: int = 100, *, seed: str = "hle_audit") -> dict[str, Any]:
    """Write ``<run_root>/eval/hle_audit_sample.jsonl`` (0600) + ``hle_audit_key.json`` (0600).

    Stratified by (method × judge label ∈ {yes, no, ambiguous}) with round-robin allocation
    across strata in a deterministic hash order; the sample rows carry the question, the
    candidate's final answer and the authoritative answer only (arm labels and the judge's
    verdict are in the separate key file), so a blinded reviewer can score them (§3.3, E1).
    """
    run_root = Path(run_root)
    if n <= 0:
        raise ValueError("n must be positive")
    tasks = {t.source_id: t for t in load_public_tasks(run_root)}
    labels = load_protected_hle(run_root)
    methods, answers = _method_of_candidates(run_root)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in _eval_records(run_root):
        sid = rec["source_id"]
        for sha, entry in rec.get("judgements", {}).items():
            if entry.get("request_id") is None:
                continue  # empty answers were not judged
            cand_methods = sorted({methods.get(cid, "unknown") for cid in entry.get("candidate_ids", [])})
            method = cand_methods[0] if cand_methods else "unknown"
            strata.setdefault((method, str(entry["judge"])), []).append({"source_id": sid, "sha": sha, "entry": entry, "method": method, "methods": cand_methods})
    for bucket in strata.values():
        bucket.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['source_id']}|{r['sha']}".encode()).hexdigest())
    order = sorted(strata)
    chosen: list[dict[str, Any]] = []
    cursors = {k: 0 for k in order}
    while len(chosen) < n:
        progressed = False
        for key in order:
            bucket = strata[key]
            if cursors[key] < len(bucket) and len(chosen) < n:
                chosen.append(bucket[cursors[key]])
                cursors[key] += 1
                progressed = True
        if not progressed:
            break
    rows: list[dict[str, Any]] = []
    key_rows: dict[str, dict[str, Any]] = {}
    for row in sorted(chosen, key=lambda r: hashlib.sha256(f"{seed}|order|{r['source_id']}|{r['sha']}".encode()).hexdigest()):
        sample_id = hashlib.sha256(f"{seed}|{row['source_id']}|{row['sha']}".encode()).hexdigest()[:16]
        task = tasks[row["source_id"]]
        label = labels[row["source_id"]]
        cids = row["entry"].get("candidate_ids", [])
        texts = {answers[c] for c in cids if c in answers}
        if len(texts) != 1:
            raise ProtocolError(f"{row['source_id']}: judged answer {row['sha'][:12]} maps to {len(texts)} distinct texts")
        rows.append({
            "sample_id": sample_id,
            "answer_format": task.answer_format,
            "question": task.task_text,
            "response_final_answer": texts.pop(),
            "final_answer_sha256": row["sha"],
            "correct_answer": label.answer,
        })
        key_rows[sample_id] = {
            "source_id": row["source_id"],
            "method": row["method"],
            "methods": row["methods"],
            "judge": row["entry"]["judge"],
            "letter_correct": row["entry"].get("letter_correct"),
            "candidate_ids": row["entry"].get("candidate_ids", []),
        }
    out_dir = run_root / "eval"
    sample_path = out_dir / AUDIT_SAMPLE_FILE
    key_path = out_dir / AUDIT_KEY_FILE
    io.write_jsonl(sample_path, rows)
    io.write_json(key_path, {"seed": seed, "n": len(rows), "strata": {f"{m}|{j}": len(b) for (m, j), b in strata.items()}, "keys": key_rows})
    os.chmod(sample_path, FILE_MODE)
    os.chmod(key_path, FILE_MODE)
    return {"sample": str(sample_path), "key": str(key_path), "n": len(rows), "n_strata": len(strata)}


__all__ = [
    "AUDIT_KEY_FILE",
    "AUDIT_SAMPLE_FILE",
    "DIFFERENTIAL_THRESHOLD_PP",
    "differential_error_bound",
    "export_human_audit_sample",
    "mc_audit",
]
