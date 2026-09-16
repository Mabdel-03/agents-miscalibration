"""Shared helpers for the WP6 (neural capture and readout) tests.

CPU only.  The model is ``tests/study_neural/support.tiny_qwen3`` — a two-layer random
Qwen3 with the **real** Qwen3 vocabulary and the pinned offline tokenizer — so anchors are
resolved against real byte spans while the forward pass costs milliseconds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agents_scaling.ard import contract as K
from tests.study_neural import support

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "study" / "fixtures" / "request_record.example.json"
REVISION_8B = K.REVISIONS[K.CHECKPOINT_SMALL]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def report_text(
    answer: str | None = "42",
    *,
    evidence: str | None = "checked twice",
    confidence: str | None = "0.87",
    prefix: str = "\n\n",
) -> str:
    """A post-``</think>`` report span in the frozen three-tag schema (spec §6.3)."""
    out = prefix
    if answer is not None:
        out += f"{K.FINAL_OPEN}{answer}{K.FINAL_CLOSE}"
    if evidence is not None:
        out += f"{K.EVIDENCE_OPEN}{evidence}{K.EVIDENCE_CLOSE}"
    if confidence is not None:
        out += f"{K.CONFIDENCE_OPEN}{confidence}{K.CONFIDENCE_CLOSE}"
    return out


def completion_ids(tokenizer, content: str, *, think: str | None = "reasoning 一二三", im_end: bool = True) -> list[int]:
    """``<think>…</think>`` + the report span, as vLLM stores the completion ids."""
    return support.completion_from_text(tokenizer, think, content, im_end=im_end)


def prompt_ids(tokenizer, text: str) -> tuple[list[dict[str, str]], list[int]]:
    messages = [{"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=True, return_dict=False)
    return messages, [int(t) for t in ids]


def request_record(
    tokenizer,
    request_id: str,
    *,
    task: str = "What is 6 times 7?",
    content: str | None = None,
    think: str | None = "reasoning 一二三",
    finish_reason: str = "stop",
    revision: str = REVISION_8B,
    im_end: bool = True,
):
    """A committed ``study.types.RequestRecord`` the capture runner accepts as-is."""
    from agents_scaling.study import types as T

    base = json.loads(FIXTURE.read_text())
    messages, ids = prompt_ids(tokenizer, task)
    body = report_text() if content is None else content
    completion = completion_ids(tokenizer, body, think=think, im_end=im_end)
    base.update({
        "request_id": request_id,
        "messages": messages,
        "prompt_token_ids": list(ids),
        "prompt_tokens": len(ids),
        "chat_template_kwargs": {"enable_thinking": True},
        "response": {**base["response"], "token_ids": completion, "completion_tokens": len(completion),
                     "content": body, "reasoning": think, "finish_reason": finish_reason},
        "identity": {**base["identity"], "model_revision": revision},
    })
    base["model"] = {**base["model"], "model_revision": revision}
    record = T.RequestRecord.from_dict({k: v for k, v in base.items() if k != "content_sha256"})
    return record.with_content_sha256()


def tiny_snapshot(tmp_path: Path, tokenizer) -> Path:
    """A two-layer random Qwen3 over the real vocabulary, beside the real tokenizer files."""
    import shutil

    path = Path(tmp_path)
    model = support.tiny_qwen3(seed=0, vocab_size=int(tokenizer.vocab_size) + len(tokenizer.added_tokens_decoder) + 64)
    model.save_pretrained(path, safe_serialization=True)
    src = support.snapshot_path(K.CHECKPOINT_SMALL)
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, path / name)
    return path


def tiny_engine(tokenizer, *, blocks: tuple[int, ...] = (0, 1), max_batch_tokens: int = 4096) -> Any:
    """A ``CaptureEngine`` around the tiny model on CPU (fp32, causal mask policy)."""
    import torch

    from agents_scaling.study.neural.engine import CaptureEngine

    model = support.tiny_qwen3(seed=0, vocab_size=int(tokenizer.vocab_size) + len(tokenizer.added_tokens_decoder) + 64)
    return CaptureEngine(None, blocks, device_map="cpu", dtype=torch.float32, model=model,
                         max_batch_tokens=max_batch_tokens)


# --------------------------------------------------------------------------- stores/engines


class StubStore:
    """A ``study.inference.store.RequestStore``-shaped store: ``get(request_id)``."""

    def __init__(self, records: Any = ()) -> None:
        self._by_id: dict[str, Any] = {}
        for record in records:
            self.add(record)

    def add(self, record: Any) -> Any:
        self._by_id[str(record.request_id)] = record
        return record

    def get(self, request_id: str) -> Any:
        return self._by_id.get(str(request_id))

    def __len__(self) -> int:
        return len(self._by_id)


class RecordingEngine:
    """Transparent wrapper that keeps every ``CaptureRequest`` the runner submitted.

    Used to assert what the capture pass asked the model for — in particular that
    ``logit_positions`` (the full-vocabulary rows) are requested at the two
    ``contract.ENTROPY_ANCHORS`` and nowhere else.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.requests: list[Any] = []
        self.batches = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def forward(self, requests):
        self.requests.extend(requests)
        self.batches += 1
        return self.engine.forward(requests)


# --------------------------------------------------------------------------- manifests


def work_row(
    decision_id: str,
    *,
    item_id: str = "math500:0001",
    episode_id: str = "ep0",
    agent_id: int = 0,
    round_: int = 0,
    protocol: str = "PEER",
    split: str = "train",
    family: str = "math500",
    boundary_injected: bool = False,
    cell_id: str = "cap.000",
    request_id: str | None = None,
) -> dict[str, Any]:
    """A decision-manifest row in ``decisions.parquet`` field names."""
    return {
        "decision_id": decision_id,
        "request_id": decision_id if request_id is None else request_id,
        "episode_id": episode_id,
        "item_id": item_id,
        "agent_id": int(agent_id),
        "round": int(round_),
        "protocol": protocol,
        "cohort": K.Cohort.CAPTURE.value,
        "split": split,
        "family": family,
        "checkpoint": K.CHECKPOINT_SMALL,
        "boundary_injected": bool(boundary_injected),
        "cell_id": cell_id,
    }


def chunk_rows(run_root: Path, stage: str = "native") -> list[dict[str, Any]]:
    """Every stored ``ActivationRow`` (as a dict) under a run root, in chunk order."""
    from agents_scaling.study.neural import storage as S

    return [dict(row) for _, row in S.iter_rows(run_root, stage)]


# --------------------------------------------------------------------------- probe frames


SIGNAL_SITE: tuple[int, str] = (17, "ANSWER_END")
NOISE_SITE: tuple[int, str] = (8, "ANSWER_END")


def probe_dataset(
    *,
    n_items: int = 24,
    hidden: int = 16,
    sites: tuple[tuple[int, str], ...] = (SIGNAL_SITE,),
    signal_sites: tuple[tuple[int, str], ...] = (SIGNAL_SITE,),
    round_: int = 2,
    split: str = "train",
    item_prefix: str = "it",
    # 1.5 keeps the label separable *within* an item while the per-item offsets keep the
    # pooled problem non-separable — an L2 logistic fit on separable data only crawls
    # towards infinite weights and would make this kit slow for no extra coverage.
    signal: float = 1.5,
    behaviour_signal: float = 0.08,
    seed: int = 0,
):
    """Synthetic captured states with a known within-item structure.

    Every item is *mixed* (1-3 of the five agents correct), so within-item AUC is
    defined everywhere.  At a ``signal_sites`` site the state carries the label along one
    hidden direction; at any other site it carries only the item offset and noise.  The
    behavioural channel carries a weak version of the same label, and the answer class is
    deliberately independent of it so the answer-class baseline is a real control.

    Returns ``(rows, H)`` for :func:`agents_scaling.ard.neural.probes.build_frame`.
    """
    import numpy as np

    from agents_scaling.ard.neural import probes as P

    rng = np.random.default_rng(seed)
    rows: list[Any] = []
    vectors: list[Any] = []
    for i in range(n_items):
        item = f"{item_prefix}{i:03d}"
        family = K.FAMILIES[i % len(K.FAMILIES)].value
        offset = rng.standard_normal(hidden) * 0.5
        labels = np.zeros(K.N_AGENTS)
        labels[: 1 + (i % 3)] = 1.0
        rng.shuffle(labels)
        for agent in range(K.N_AGENTS):
            y = float(labels[agent])
            u = (0.6 if y else -0.4) + float(rng.normal(scale=0.05))
            for layer, anchor in sites:
                vector = offset + rng.standard_normal(hidden) * 0.2
                if (layer, anchor) in signal_sites:
                    vector = vector + np.eye(hidden)[0] * (signal * y)
                vectors.append(vector)
                rows.append(P.ProbeRow(
                    row_id=f"{item}:{agent}:{layer}:{anchor}",
                    item_id=item, episode_id=f"ep{i}", agent_id=agent, protocol="PEER",
                    round_=round_, layer=layer, anchor=anchor, vector_index=len(vectors) - 1,
                    split=split, family=family,
                    behaviour={"confidence": 0.5 + behaviour_signal * y + float(rng.normal(scale=0.25)),
                               "prompt_tokens": 800.0 + float(rng.normal(scale=20.0))},
                    answer_class="ABCD"[int(rng.integers(4))],
                    targets={P.TARGET_CURRENT: y, P.TARGET_EXPOSURE_VALUE: u},
                ))
    return rows, np.asarray(vectors, dtype=float)


# --------------------------------------------------------------------------- capture cells


class FakeTask:
    """The two ``ArdTask`` fields ``runner.write_item`` reads."""

    def __init__(self, item_id: str, *, family: str = "math500", split: str = "train") -> None:
        self.item_id = item_id
        self.family = family
        self.split = split


class FakeCell:
    """The ``ArdCellSpec`` surface ``run_capture_cell`` touches: an id and an options block."""

    def __init__(self, options: dict[str, Any], *, cell_id: str = "CAPTURE.capture.none.8B.T1024.train.s000") -> None:
        self.cell_id = cell_id
        self.options = dict(options)


class FakeRunner:
    """An ``ArdCellRunner``-shaped stand-in for the CAPTURE worker (no endpoint, no Slurm).

    ``write_item`` records the status instead of writing an ``ArdItemResult``, so a test can
    tell "wrote no_decisions and exited 0" from "raised" — the exact distinction the two
    wiring defects turned on.
    """

    def __init__(self, run_root: Path, options: dict[str, Any], *, items: tuple[str, ...] = ()) -> None:
        self.run_root = Path(run_root)
        self.ard_cell = FakeCell(options)
        self.tasks = {item: FakeTask(item) for item in items}
        self.written: list[tuple[str, str]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._t = 0.0

    def clock(self) -> float:
        self._t += 1.0
        return self._t

    def event(self, name: str, **fields: Any) -> None:
        self.events.append((name, dict(fields)))

    def write_item(self, task: Any, episodes: Any, *, status: str = "complete",
                   started: float | None = None, grades: Any = ()) -> None:
        self.written.append((task.item_id, status))


def item_result_file(
    run_root: Path,
    cell_id: str,
    item_id: str,
    *,
    decisions: Sequence[tuple[str, str, int, int]] = (),
    split: str = "train",
    family: str = "math500",
) -> Path:
    """A ``cells/<cell_id>/items/<item_id>.json`` the capture manifest reader accepts.

    ``decisions`` are ``(decision_id, protocol, round, agent_id)`` tuples, grouped into one
    episode per protocol — the shape WP2 writes.
    """
    episodes: dict[str, dict[str, Any]] = {}
    for decision_id, protocol, round_, agent in decisions:
        episode = episodes.setdefault(protocol, {
            "episode_id": f"ep-{protocol}", "protocol": protocol,
            "cohort": K.Cohort.CAPTURE.value, "decisions": []})
        episode["decisions"].append({
            "decision_id": decision_id, "request_id": decision_id,
            "episode_id": episode["episode_id"], "item_id": item_id,
            "agent_id": int(agent), "round": int(round_), "protocol": protocol,
            "checkpoint": K.CHECKPOINT_SMALL, "boundary_injected": False,
        })
    path = Path(run_root) / "cells" / cell_id / "items" / f"{item_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source_id": item_id, "split": split, "family": family,
        "cell": {"cell_id": cell_id}, "episodes": list(episodes.values()),
    }), encoding="utf-8")
    return path
