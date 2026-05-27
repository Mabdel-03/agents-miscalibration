"""Run one ExperimentCell: build agents -> run the topology over N questions -> log JSONL.

The runner is the single place that wires the three axes together for a cell:
  * Axis 1 (capacity): the model served at this cell's ``model_size`` (via the registry).
  * Axis 2 (context): ``cell.context_share_level`` passed to the topology.
  * Axis 3 (prompt): ``cell.prompt_complexity_level`` selects the system prompt.
It writes ``results.jsonl`` (one row per question) and ``meta.json`` (cell config +
prompt token count + quality scores + git commit) into the cell directory.
"""

from __future__ import annotations

import time

from agents_scaling.agents.base_agent import Agent
from agents_scaling.agents.topologies import build_topology
from agents_scaling.benchmarks.grading import grade
from agents_scaling.benchmarks.loaders import load_benchmark
from agents_scaling.calibration.semantic_entropy import semantic_entropy_conf
from agents_scaling.calibration.signals import majority_answer, self_consistency_conf
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.result_schema import CellMeta, QuestionResult
from agents_scaling.models import get_model
from agents_scaling.prompts.prompt_quality import score_prompt_quality
from agents_scaling.prompts.system_prompts import get_prompt, token_count
from agents_scaling.serving.client import LogprobClient
from agents_scaling.serving.registry import wait_for_server


def _build_agents(cell: ExperimentCell, base_url: str, served_model: str) -> list[Agent]:
    system_prompt = get_prompt(cell.prompt_complexity_level)
    n = 1 if cell.topology.value == "single_agent" else cell.n_agents
    agents = []
    for i in range(n):
        client = LogprobClient(base_url=base_url, model=served_model)
        agents.append(
            Agent(
                f"agent{i}",
                client,
                system_prompt,
                temperature=cell.temperature,
                reasoning_level=cell.reasoning_level,  # Axis 4
            )
        )
    return agents


def run_cell(
    cell: ExperimentCell, run_id: str, score_prompt_with_judge: bool = False, shard: int = 0
) -> str:
    """Execute one cell end-to-end; returns the path to its results.jsonl.

    ``shard`` (typically the SLURM array task id) round-robins this cell across all
    registered server endpoints for its model size, spreading load across clusters.
    """
    spec = get_model(cell.model_size)
    run_root = io.run_dir(run_id)
    entry = wait_for_server(run_root, cell.model_size, shard=shard)
    base_url = entry.base_url
    served_model = cell.model_size  # vLLM --served-model-name

    cdir = io.cell_dir(run_id, cell.cell_id)
    results_path = cdir / "results.jsonl"
    meta_path = cdir / "meta.json"

    # Resume: a cell with meta.json is fully done -> skip (idempotent re-submission of the
    # SLURM array after time-limit kills won't redo or duplicate work).
    if meta_path.exists():
        print(f"[run_cell] {cell.cell_id} already complete (meta.json present); skipping")
        return str(results_path)

    # Partial cell: collect qids already answered so we only run the remainder.
    done_qids: set[str] = set()
    if results_path.exists():
        import json as _json

        for line in results_path.read_text().splitlines():
            if line.strip():
                try:
                    done_qids.add(_json.loads(line)["qid"])
                except (ValueError, KeyError):
                    pass
        if done_qids:
            print(f"[run_cell] {cell.cell_id} resuming: {len(done_qids)} questions already done")

    # --- meta (axis-3 measured attributes recorded here) ---
    system_prompt = get_prompt(cell.prompt_complexity_level)
    judge = LogprobClient(base_url=base_url, model=served_model) if score_prompt_with_judge else None
    pq = score_prompt_quality(system_prompt, judge=judge)
    meta = CellMeta(
        cell_id=cell.cell_id,
        config=cell.to_dict(),
        config_hash=cell.config_hash(),
        model_hf_id=spec.hf_id,
        served_model_name=served_model,
        prompt_token_count=token_count(cell.prompt_complexity_level),
        prompt_quality={"heuristic": pq.heuristic, "llm_judge": pq.llm_judge, "features": pq.features},
        git_commit=io.git_commit(),
        started_at=time.time(),
    )

    questions = load_benchmark(cell.benchmark, n=cell.n_questions, seed=cell.seed)
    agents = _build_agents(cell, base_url, served_model)
    reasoning_token_totals: list[int] = []  # per-question reasoning tokens -> meta mean

    for q in questions:
        if q.qid in done_qids:
            continue  # resume: already answered in a prior (killed) run
        t0 = time.time()
        topo = build_topology(
            cell.topology, agents, cell.context_share_level, cell.rounds,
            max_tokens=1024, seed=cell.seed,
        )
        tr = topo.run(q)
        wall_ms = (time.time() - t0) * 1000.0

        # Self-consistency / semantic-entropy from the first agent's samples
        # (cheap extra signal; only for the canonical single-agent view to bound cost).
        sc: dict = {}
        if cell.topology.value == "single_agent" and cell.n_samples > 1:
            samples = agents[0].sample(q, n=cell.n_samples, base_seed=cell.seed + 1000)
            sampled_answers = [s.answer_choice for s in samples]
            maj = majority_answer(sampled_answers)
            conf, ent = semantic_entropy_conf([s.answer_choice or "" for s in samples])
            sc = {
                "samples": sampled_answers,
                "majority": maj,
                "self_consistency_conf": self_consistency_conf(sampled_answers, tr.final_answer or ""),
                "semantic_entropy_conf": conf,
                "semantic_entropy": ent,
            }

        correct = grade(q, tr.final_answer or "") if tr.final_answer is not None else False
        total_reasoning_tokens = sum(o.reasoning_tokens for o in tr.per_agent)
        rec = QuestionResult(
            cell_id=cell.cell_id,
            qid=q.qid,
            benchmark=q.benchmark,
            model_size=cell.model_size,
            topology=cell.topology.value,
            context_share_level=cell.context_share_level.value,
            prompt_complexity_level=cell.prompt_complexity_level,
            reasoning_level=cell.reasoning_level.value,
            final_answer=tr.final_answer,
            answer_key=q.answer_key,
            correct=correct,
            per_agent=[o.to_dict() for o in tr.per_agent],
            system_conf=tr.system_conf,
            self_consistency=sc,
            efficiency_raw={
                "n_turns": tr.n_turns,
                "n_messages": tr.n_messages,
                "n_rounds": tr.n_rounds,
                "n_agents": tr.n_agents,
                "total_prompt_tokens": tr.total_prompt_tokens,
                "total_completion_tokens": tr.total_completion_tokens,
                "total_reasoning_tokens": total_reasoning_tokens,
                "wall_ms": wall_ms,
            },
            timestamp=time.time(),
        )
        io.append_jsonl(results_path, rec.to_dict())
        reasoning_token_totals.append(total_reasoning_tokens)

    meta.n_questions = len(questions)
    meta.mean_reasoning_tokens = (
        sum(reasoning_token_totals) / len(reasoning_token_totals) if reasoning_token_totals else 0.0
    )
    meta.finished_at = time.time()
    io.write_json(meta_path, meta.to_dict())
    return str(results_path)
