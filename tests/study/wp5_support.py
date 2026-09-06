"""Shared helpers for the WP5 tests: a synthetic public/protected export, a test freeze
with a chosen B0, and a runner harness against the WP2 fake vLLM server.

The export mirrors ``data/export.py``'s layout (``data/public/tasks.jsonl``,
``data/protected/*.jsonl`` 0600 inside a 0700 dir) so the real loaders (WP1) and the
evaluator identity (WP5) read it unchanged.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agents_scaling.study import types as T
from agents_scaling.study.config import StudyConfig, freeze
from agents_scaling.study.data.layout import DIR_MODE, FILE_MODE, BCB_TESTS_FILE, HLE_LABELS_FILE, protected_dir, public_tasks_path
from agents_scaling.study.inference import client as C
from agents_scaling.study.inference.tokens import render_chat_token_ids
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.runner import run_cell
from tests.study.wp2_support import fake_server  # noqa: F401  (re-exported)

TRIVIAL_TEST = "import unittest\n\nclass TestTrivial(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"
CODE_POOL: tuple[str, ...] = ("x = 1\n", "x = 2\n", "x = 3\n")


def make_tasks(n_per_domain: int, split: str = "main", *, mc_every: int = 2) -> list[T.PublicTask]:
    """``n_per_domain`` HLE (alternating MC / exact) + ``n_per_domain`` BCB tasks, ranks 0..n-1."""
    tasks: list[T.PublicTask] = []
    for i in range(n_per_domain):
        mc = i % mc_every == 0
        text = (f"[{split} hle {i}] Which option is right? A) one B) two C) three" if mc else f"[{split} hle {i}] What is 2 + {i}?")
        tasks.append(T.PublicTask(f"hle:{split}{i:04d}", T.Domain.HLE, split, text, "multipleChoice" if mc else "exactMatch",
                                  "Gold" if i % 3 else "Revision", i, len(text.split()), None, "math"))
    for i in range(n_per_domain):
        text = f"[{split} bcb {i}] Write a function task_func that returns {i}."
        tasks.append(T.PublicTask(f"bcb:{split}{i:04d}", T.Domain.BCB, split, text, "code", "bcb", i, len(text.split()), "task_func", None))
    return tasks


def write_export(run_root: Path, tasks: Sequence[T.PublicTask], *, gold: Callable[[T.PublicTask], str] | None = None,
                 test_src: str = TRIVIAL_TEST) -> None:
    """Public rows + protected labels/tests in the WP1 layout (0700/0600)."""
    pub = public_tasks_path(run_root)
    pub.parent.mkdir(parents=True, exist_ok=True)
    with open(pub, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    prot = protected_dir(run_root)
    prot.mkdir(parents=True, exist_ok=True)
    os.chmod(prot, DIR_MODE)
    gold = gold or (lambda t: "B" if t.answer_format == "multipleChoice" else "4")
    with open(prot / HLE_LABELS_FILE, "w", encoding="utf-8") as handle:
        for task in tasks:
            if task.domain is T.Domain.HLE:
                handle.write(json.dumps({"source_id": task.source_id, "answer": gold(task), "answer_type": task.answer_format, "rationale": "r"}) + "\n")
    with open(prot / BCB_TESTS_FILE, "w", encoding="utf-8") as handle:
        for task in tasks:
            if task.domain is T.Domain.BCB:
                handle.write(json.dumps({"source_id": task.source_id, "task_id": task.source_id[4:], "entry_point": "task_func", "test": test_src,
                                         "canonical_solution": "x = 0\n", "code_prompt": "", "libs": []}) + "\n")
    for name in (HLE_LABELS_FILE, BCB_TESTS_FILE):
        os.chmod(prot / name, FILE_MODE)


def root_reservation(cfg: StudyConfig, tokenizer: Any, task: T.PublicTask, framing: T.Framing = T.Framing.F00, size: str = "32B") -> int:
    n = len(render_chat_token_ids(tokenizer, R.render_root(task, framing).messages, True))
    return FlopOracle.from_table(size).reservation(n, T.SOLVER_OUT_CAP)


def freeze_for_test(run_root: Path, cfg: StudyConfig, b0_flops: int) -> Path:
    return freeze(run_root, {"B0_flops": str(int(b0_flops))}, config=cfg, code_version="test")


def make_cell(method: T.Method, items: Sequence[str], *, module: str = "A", N: int = 5, B: int = 4, framing: T.Framing = T.Framing.F00,
              kind: T.CellKind = T.CellKind.GENERATE, seal: str | None = None, split: str = "main", parallel_items: int = 1,
              max_inflight: int = 8, checkpoint: str = "32B", shard: int = 0, episode_rep: int = 0) -> T.CellSpec:
    if kind is T.CellKind.GENERATE:
        cell_id = f"{module}.{method.value}.{checkpoint}.N{N}.B{B}.F{framing.value}.e{episode_rep}.s{shard:03d}"
    else:
        cell_id = f"eval.{kind.value}.{checkpoint}.N0.B0.Fnat.e0.x{(seal or '')[:8]}.s{shard:03d}"
    deps = () if seal is None else (f"seal:{seal}",)
    return T.CellSpec(cell_id, kind, module, method, checkpoint, N, B, framing, episode_rep, split, tuple(items), deps, max_inflight,
                      parallel_items, checkpoint if kind is T.CellKind.GENERATE else ("eval" if kind is T.CellKind.EVAL_BCB else checkpoint), None)


class Harness:
    """Runs cells of one run root against one fake endpoint with injected tokenizer/oracle."""

    def __init__(self, run_root: Path, cfg: StudyConfig, server: Any, *, size: str = "32B") -> None:
        self.run_root = run_root
        self.cfg = cfg
        self.server = server
        self.size = size
        self.clients: list[Any] = []

    def client_factory(self, hook: Callable[[], None] | None = None):
        harness = self

        def factory(pool, tokenizer, cost_fn):
            class HookedClient(C.VllmChatClient):
                def generate(self, spec, **kwargs):  # type: ignore[override]
                    record = super().generate(spec, **kwargs)
                    if hook is not None:
                        hook()
                    return record

            client = HookedClient(pool, tokenizer, cost_fn=cost_fn, run_id="wp5-test")
            harness.clients.append(client)
            return client

        return factory

    def run(self, cell: T.CellSpec, *, stop_event: threading.Event | None = None, hook: Callable[[], None] | None = None,
            wait_s: float = 5.0, **kwargs: Any) -> int:
        return run_cell(
            cell, self.run_root, self.run_root, 0, stop_event, cfg=self.cfg, tokenizer=self.server.tokenizer,
            oracle_factory=lambda ckpt: FlopOracle.from_table(ckpt.size), client_factory=self.client_factory(hook),
            wait_timeout_s=wait_s, run_id="wp5-test", **kwargs,
        )


def write_cells_manifest(run_root: Path, name: str, cells: Sequence[T.CellSpec]) -> tuple[Path, str]:
    from agents_scaling.study.cells import write_cells_file

    path = run_root / name
    digest = write_cells_file(path, cells, {"test": True})
    return path, digest


__all__ = [name for name in globals() if not name.startswith("_")]
