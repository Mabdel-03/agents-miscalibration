"""Synthetic inputs for the N3 tests: in-memory feature frames with a plantable neural
signal, and an on-disk synthetic run root (public export, evaluation join table, report
renders, shadow forecasts, N1 activation store) for the analysis CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.neural import readout as R
from agents_scaling.study.neural import storage as S

METHODS = ("IND_VOTE", "DEC", "CEN_FLAT")
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def synthetic_rows(
    n_per_domain: int,
    rng: np.random.Generator,
    *,
    methods: tuple[str, ...] = METHODS,
    d: int = 32,
    blocks: tuple[int, ...] = (8, 17),
    neural_signal: float = 0.0,
    obs_signal: float = 0.0,
    text_signal: float = 0.0,
    split: str = "dev",
    signal_block: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Rows + per-block state matrices.  ``y`` follows a logistic model of a latent item
    difficulty plus the planted signals; the neural signal lives in coordinate 0 of ``h`` at
    ``signal_block`` (default: the first block) only."""
    signal_block = blocks[0] if signal_block is None else signal_block
    rows: list[dict[str, Any]] = []
    vectors: dict[int, list[np.ndarray]] = {b: [] for b in blocks}
    for dom in ("hle", "bcb"):
        for i in range(n_per_domain):
            sid = f"{dom}:{split}{i:04d}"
            difficulty = rng.normal()
            for m in methods:
                z = 0.4 * difficulty
                h = {b: rng.normal(size=d) * 2.0 for b in blocks}
                s_neural = rng.normal()
                h[signal_block][0] = s_neural * 6.0
                z += neural_signal * s_neural
                conf = float(np.clip(rng.beta(4, 3), 0.01, 0.99))
                z += obs_signal * (conf - 0.55) * 6.0
                topic = int(rng.integers(0, 2))
                z += text_signal * (2.0 * topic - 1.0)
                y = int(rng.random() < _sigmoid(z))
                text = " ".join(rng.choice(WORDS[:9] if topic else WORDS[9:], size=12)) + f" report {m} {dom}"
                q_team = float(np.clip(_sigmoid(z + rng.normal() * 0.5), 0.01, 0.99))
                idx = len(vectors[blocks[0]])
                for b in blocks:
                    vectors[b].append(h[b].astype(np.float32))
                rows.append({
                    "source_id": sid, "method": m, "report_id": identity.sha256_hex(f"{sid}:{m}"), "selection_id": identity.sha256_hex(f"sel:{sid}:{m}"),
                    "seal": "ab" * 32, "superdomain": dom, "cluster": sid, "split": split, "rank": i, "answer_format": "code" if dom == "bcb" else ("multipleChoice" if i % 2 else "exactMatch"),
                    "N": 5, "B": 4, "personal_conf": conf, "personal_missing": False, "selected_is_sentinel": False,
                    "winning_count": float(rng.integers(1, 6)), "tied_classes": float(rng.integers(1, 4)), "all_singleton": 0.0, "valid_count": 5.0, "planned_count": 5.0, "pool_size": 5.0,
                    "calls_root": 5.0, "calls_revise": 5.0 if m == "DEC" else 0.0, "calls_hub": 1.0 if m == "CEN_FLAT" else 0.0, "calls_worker": 4.0 if m == "CEN_FLAT" else 0.0,
                    "calls_total": 10.0 if m == "DEC" else 5.0, "calls_admitted": 6.0, "spent_frac": float(rng.uniform(0.3, 0.9)), "slack_frac": float(rng.uniform(0.05, 0.5)),
                    "stop_reason": "CALL_CAP" if rng.random() < 0.5 else "BUDGET", "task_tokens": float(rng.integers(20, 300)), "selected_len": float(rng.integers(100, 900)),
                    "final_answer_len": float(rng.integers(1, 40)), "prompt_tokens": float(rng.integers(500, 5000)), "evidence_tokens": float(rng.integers(100, 4000)), "packets": 4.0,
                    "text": text, "q_team_now": q_team, "q_personal_forecast": conf, "forecast_valid": True, "forecast_status": "ok", "y": y,
                    **{f"h_row_b{b}": idx for b in blocks},
                })
    X = {b: np.stack(v) for b, v in vectors.items()}
    return rows, X


def synthetic_frames(n_per_domain: int, seed: int, **kwargs: Any) -> R.FeatureFrames:
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows, X = synthetic_rows(n_per_domain, rng, **kwargs)
    blocks = tuple(sorted(X))
    return R.FeatureFrames(rows=pd.DataFrame.from_records(rows), X=X, blocks=blocks, stats={"rows": len(rows)})


# --------------------------------------------------------------------------- on-disk world


def write_world(run_root: Path, *, n_dev: int, n_main: int, seed: int, blocks: tuple[int, ...] = (8, 17), d: int = 32, neural_signal: float = 2.5,
                labels: bool = True, native: bool = True) -> dict[str, Any]:
    """A synthetic run root: ``data/public/tasks.jsonl``, ``tables/selections.parquet`` (when
    ``labels``), ``forecast/reports/*.json`` + ``forecast/*.json``, the N1 ``report`` store at
    STATE_ANCHOR (+ TASK_ONLY_ANCHOR) and a ``native`` store with 5 IND roots per item."""
    import pandas as pd

    from agents_scaling.study.forecast.shadow import forecast_path, report_path
    from tests.study.wp5_support import make_tasks, write_export

    rng = np.random.default_rng(seed)
    tasks = make_tasks(n_dev, split="dev") + make_tasks(n_main, split="main")
    write_export(run_root, tasks)
    task_by_id = {t.source_id: t for t in tasks}
    rows_dev, X_dev = synthetic_rows(n_dev, rng, d=d, blocks=blocks, neural_signal=neural_signal, split="dev")
    rows_main, X_main = synthetic_rows(n_main, rng, d=d, blocks=blocks, neural_signal=neural_signal, split="main")
    all_rows = rows_dev + rows_main
    table = []
    with S.ShardWriter(run_root, "report", "s0", chunk_rows=64) as writer:
        for source_rows, X in ((rows_dev, X_dev), (rows_main, X_main)):
            for r in source_rows:
                task = task_by_id[r["source_id"]]
                report = {
                    "report_id": r["report_id"], "source_id": r["source_id"], "method": r["method"], "seal": r["seal"], "selection_id": r["selection_id"],
                    "cell_id": f"A.{r['method']}.32B.N5.B4.Fnat.e0.s000", "pool_id": identity.sha256_hex("pool:" + r["selection_id"]),
                    "item": {"source_id": r["source_id"], "domain": task.domain.value, "split": task.split, "answer_format": task.answer_format, "method": r["method"],
                             "checkpoint": "32B", "N": 5, "B": 4, "framing": "nat", "episode_rep": 0, "status": "complete"},
                    "task_text": task.task_text, "selected_candidate_id": identity.sha256_hex("cand:" + r["report_id"]),
                    "selected_candidate": {"approach": "x", "evidence": [], "alternatives_considered": [], "failure_checks": [], "final_answer": "4" * int(r["final_answer_len"]), "confidence": r["personal_conf"]},
                    "selected_is_sentinel": False, "selected_personal_confidence": r["personal_conf"], "personal_confidence_missing": False,
                    "decision_rule": "VOTE", "vote_metadata": {"selector_id": "VOTE", "pool_kind": "archive", "pool_size": 5, "planned_count": 5, "valid_count": 5, "no_valid_candidate": False,
                                                              "winning_count": int(r["winning_count"]), "tied_classes": int(r["tied_classes"]), "all_singleton": False, "grouping_mode": "exact_norm", "grouping_mode_counts": {}},
                    "budget_metadata": {"B": 4, "B_flops": 1000.0, "spent_flops": 1000.0 * r["spent_frac"], "slack_flops": 1000.0 * r["slack_frac"], "calls_admitted": 6,
                                        "calls_by_role": {"root": 5, **({"revise": 5} if r["method"] == "DEC" else {}), **({"hub": 1, "worker": 4} if r["method"] == "CEN_FLAT" else {})},
                                        "stop_reason": r["stop_reason"]},
                    "nonselected": [{"i": k} for k in range(4)], "nonselected_total": 4, "packets_clipped_last": False, "packets_dropped": 0, "text": r["text"],
                    "spans": {}, "task_only_anchor": 10, "evidence_tokens": int(r["evidence_tokens"]), "evidence_tokens_cap": 8192,
                }
                render = {"schema_version": 1, "kind": "FINAL_HANDOFF_REPORT", "report_id": r["report_id"], "source_id": r["source_id"], "method": r["method"], "seal": r["seal"],
                          "selection_id": r["selection_id"], "prompt_tokens": int(r["prompt_tokens"]), "evidence_tokens": int(r["evidence_tokens"]),
                          "messages": [{"role": "user", "content": r["text"]}], "prompt_token_ids": [1, 2, 3], "report": report}
                io.write_json(report_path(run_root, r["source_id"], r["method"]), render)
                io.write_json(forecast_path(run_root, r["source_id"], r["method"]), {
                    "schema_version": 1, "kind": "SHADOW_FORECAST", "report_id": r["report_id"], "source_id": r["source_id"], "method": r["method"], "seal": r["seal"],
                    "selection_id": r["selection_id"], "parse_status": "ok", "parsed": {"q_personal": r["personal_conf"], "q_team_now": r["q_team_now"], "q_child_contract": None, "q_recover": None, "q_preserve": None},
                    "selected_personal_confidence": r["personal_conf"], "personal_confidence_missing": False,
                })
                for b in blocks:
                    for anchor, vec in (("STATE_ANCHOR", X[b][r[f"h_row_b{b}"]]), ("TASK_ONLY_ANCHOR", rng.normal(size=d).astype(np.float32))):
                        writer.add(S.ActivationRow(StateSnapshot_id=r["report_id"], consumer_revision="rev", condition=f"report:{r['method']}", child_slot=None, nonce_hash="n",
                                                   block=b, anchor_kind=anchor, structural_span="report", token_offset=5, channel="prompt", generated_token_count=0, missingness=None,
                                                   tensor_hash=None, measurement_cost={}, operationally_available_at_checkpoint=True, stage="report", sequence_id=r["report_id"],
                                                   source_id=r["source_id"], method=r["method"], role="REPORT_READER", phase="FINAL_HANDOFF_REPORT", hidden_size=d), vec)
                table.append({
                    "source_id": r["source_id"], "domain": task.domain.value, "split": task.split, "rank": task.rank, "stratum": task.stratum, "answer_format": task.answer_format,
                    "seal": r["seal"], "selection_id": r["selection_id"], "pool_id": report["pool_id"], "cell_id": report["cell_id"], "module": "A", "method": r["method"], "checkpoint": "32B",
                    "N": 5, "B": 4, "framing": "nat", "episode_rep": 0, "degree": None, "pool_kind": {"IND_VOTE": "archive", "DEC": "latest_slots", "CEN_FLAT": "native"}[r["method"]],
                    "prefix_k": None, "selector_id": "VOTE", "selected_candidate_id": report["selected_candidate_id"], "planned_count": 5, "valid_count": 5,
                    "selected_correct": bool(r["y"]), "candidate_mean": 0.5, "oracle_coverage": float(r["y"] or rng.random() < 0.3), "selection_gap": 0.0, "n_correct": int(r["y"]) + 1,
                    "no_valid_candidate": False, "all_singleton": False, "tied_classes": int(r["tied_classes"]), "winning_count": int(r["winning_count"]), "grouping_mode": "exact_norm",
                    "native_final_correct": None, "sealed_at": 1000.0, "stop_reason": r["stop_reason"], "slack": 1.0, "spent_flops": 1.0, "B_flops": 4.0, "calls_admitted": 6,
                    "calls_by_role": json.dumps(report["budget_metadata"]["calls_by_role"], sort_keys=True), "complete_candidates": 5, "aliased_calls": 0, "generated_calls": 5, "item_status": "complete",
                })
    if labels:
        (run_root / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(table).to_parquet(run_root / "tables" / "selections.parquet", index=False)
    if native:
        with S.ShardWriter(run_root, "native", "s0", chunk_rows=64) as writer:
            for task in tasks:
                for slot in range(5):
                    rid = identity.sha256_hex(f"native:{task.source_id}:{slot}")
                    for b in blocks:
                        writer.add(S.ActivationRow(StateSnapshot_id=rid, consumer_revision="rev", condition="native:IND_VOTE", child_slot=None, nonce_hash="n", block=b,
                                                   anchor_kind="NATIVE_PREFILL", structural_span="prompt", token_offset=3, channel="prompt", generated_token_count=0, missingness=None,
                                                   tensor_hash=None, measurement_cost={}, operationally_available_at_checkpoint=True, stage="native", sequence_id=rid,
                                                   source_id=task.source_id, method="IND_VOTE", role="INDEPENDENT_SOLVER", phase="ROOT", hidden_size=d, extra={"actor_slot": slot}),
                                   rng.normal(size=d).astype(np.float32))
                        writer.add(S.ActivationRow(StateSnapshot_id=rid, consumer_revision="rev", condition="native:IND_VOTE", child_slot=None, nonce_hash="n", block=b,
                                                   anchor_kind="GENERATED_512", structural_span="completion", token_offset=None, channel="completion", generated_token_count=512,
                                                   missingness="NOT_REACHED", tensor_hash=None, measurement_cost={}, operationally_available_at_checkpoint=True, stage="native",
                                                   sequence_id=rid, source_id=task.source_id, method="IND_VOTE", role="INDEPENDENT_SOLVER", phase="ROOT", hidden_size=d, extra={"actor_slot": slot}), None)
    return {"tasks": tasks, "rows": all_rows, "table": table, "blocks": blocks}


__all__ = ["METHODS", "synthetic_frames", "synthetic_rows", "write_world"]
