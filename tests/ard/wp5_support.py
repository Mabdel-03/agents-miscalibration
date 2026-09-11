"""Shared helpers for the ARD36 WP5 (controller / selectors / receiver) tests.

Three kinds of fixture live here:

* **pure** builders — stub decision records, frozen-v1 feature rows, paired and CURRENT
  frames — so the controller assertions are made on hand-written frames with known
  structure rather than on whatever a fake server happened to emit;
* **pool** builders for :mod:`agents_scaling.ard.selectors`, plus a writer that lays down
  the ``cells/<cell_id>/items/<item_id>.json`` files ``selectors.collect_pool`` reads;
* a **cell harness** that drives a real :class:`~agents_scaling.ard.runner.ArdCellRunner`
  over an in-process :class:`~tests.study.fake_vllm.FakeVllmServer`, so the two WP5 workers
  are exercised through WP4's real item-worker seam.

Deliberately not a ``conftest.py``: ``tests/ard/`` is shared ground for every ARD work
package and this file belongs to WP5 alone (the convention of ``wp1_support.py`` ...
``wp4_support.py``).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tests.ard import wp1_support as S1  # sets HF_HOME / offline before transformers loads
from tests.ard import wp2_support as S2
from tests.ard import wp4_support as S4

from agents_scaling.ard import cells as ard_cells
from agents_scaling.ard import contract as K
from agents_scaling.ard import controller as CT
from agents_scaling.ard import features as FE
from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard import selectors as SEL
from agents_scaling.ard.types import ArdCellKind

STUDY_SEED = S2.STUDY_SEED

__all__ = [
    "STUDY_SEED",
    "StubRecord",
    "audit_cell",
    "candidate",
    "cell_runner",
    "current_frame",
    "feature_row",
    "logprobs_block",
    "paired_frame",
    "pool_of",
    "receiver_cell",
    "shared_tokenizer",
    "state_at",
    "stub_controller",
    "write_episode_item",
]


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class StubRecord:
    """The three runtime quantities ``features.features_from_state`` reads off a record."""

    answer_logprob: float | None = -0.5
    thinking_tokens: int = 128
    cap_hit: bool = False


def state_at(
    answers: Sequence[str | None],
    *,
    round_: int = 0,
    confidences: Sequence[float | None] | None = None,
    roots: Sequence[str | None] | None = None,
    item_id: str = "math500:test-0001",
    family: str = K.Family.MATH500.value,
):
    """A :class:`~agents_scaling.ard.states.RoundState` holding exactly ``answers``."""
    confs = list(confidences if confidences is not None else [0.7] * len(answers))
    reports = [S2.report(a, confidence=c) for a, c in zip(answers, confs)]
    root_reports = None if roots is None else [S2.report(a) for a in roots]
    state = S2.round_state(reports, round_=round_, roots=root_reports, item_id=item_id)
    object.__setattr__(state, "family", family)
    return state


def feature_row(
    answers: Sequence[str | None] = ("7", "7", "7", "9", None),
    *,
    slot: int = 0,
    round_: int = 0,
    records: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """One frozen-v1 feature row built by the real builder."""
    state = state_at(answers, round_=round_, **kwargs)
    return FE.features_from_state(state, slot, round_=round_, records=records)


# --------------------------------------------------------------------------- frames


def _rows(n_items: int, *, split: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for index in range(n_items):
        family = K.FAMILIES[index % len(K.FAMILIES)].value
        out.append((f"{family}:{split}-{index:03d}", family, index))
    return out


def paired_frame(n_items: int = 12, *, split: str = "train") -> CT.PairedFrame:
    """A fork frame with a learnable signal: high own-confidence favours PRIVATE.

    The point is not the effect size but that ``fit_controller`` sees separable, item-grouped,
    family-stratified data, so its one-SE penalty selection has something to select on.
    """
    rows: list[CT.PairedRow] = []
    for item_id, family, index in _rows(n_items, split=split):
        for slot in range(2):
            high = (index + slot) % 2 == 0
            answers = ["7", "7", "9", "9", "9"] if high else ["7", "9", "9", "9", "9"]
            confidences = [0.9 if high else 0.2] + [0.5] * 4
            features = feature_row(answers, slot=slot, confidences=confidences,
                                   item_id=item_id, family=family,
                                   records={} if high else None)
            rows.append(CT.PairedRow(
                row_id=f"{item_id}|a{slot}", item_id=item_id, family=family, split=split,
                round_=1 + (index % 2), agent_slot=slot, features=features,
                y_private=1 if high else 0, y_peer=0 if high else 1,
                team_id=f"{item_id}|r1",
            ))
    return CT.PairedFrame(tuple(rows))


def current_frame(n_items: int = 12, *, split: str = "train") -> CT.CurrentFrame:
    """A CURRENT frame whose label tracks the own-class count."""
    rows: list[CT.CurrentRow] = []
    for item_id, family, index in _rows(n_items, split=split):
        for slot in range(2):
            majority = (index + slot) % 2 == 0
            answers = ["7", "7", "7", "9", "9"] if majority else ["9", "7", "7", "7", "7"]
            features = feature_row(answers, slot=slot, item_id=item_id, family=family)
            rows.append(CT.CurrentRow(
                row_id=f"{item_id}|a{slot}|cur", item_id=item_id, family=family, split=split,
                round_=1, agent_slot=slot, features=features,
                y_correct=1 if majority else 0,
            ))
    return CT.CurrentFrame(tuple(rows))


class stub_controller:  # noqa: N801 - a fixture, named like one
    """A minimal frozen-controller stand-in for the two model selectors.

    ``predict(head, rows)`` returns the named feature of each row, so a test can state the
    argmax it expects without fitting anything.
    """

    def __init__(self, column: str = "own_confidence", heads: Sequence[str] = ()) -> None:
        self.column = column
        self.heads = tuple(heads) or (CT.HEAD_CURRENT, CT.HEAD_DYNAMICS)

    def predict(self, head: str, rows: Sequence[dict]) -> list[float]:
        if head not in self.heads:
            raise CT.ControllerError(f"the frozen controller has no head {head!r}")
        return [float(row[self.column]) for row in rows]


# --------------------------------------------------------------------------- pools


def candidate(
    candidate_id: str,
    answer: str | None,
    *,
    valid: bool = True,
    confidence: float | None = 0.5,
    features: dict[str, float] | None = None,
    **kwargs: Any,
) -> SEL.SelectorCandidate:
    return SEL.SelectorCandidate(
        candidate_id=candidate_id,
        answer=answer,
        canonical_class=(S2.upper_class(answer) if valid else None),
        valid=valid and answer is not None,
        confidence=confidence,
        content_sha256=f"{candidate_id}-sha",
        evidence=f"checked the arithmetic ({answer})",
        features=features,
        **kwargs,
    )


def pool_of(
    answers: Sequence[str | None] = ("7", "7", "9", None, "11"),
    *,
    item_id: str = "math500:test-0001",
    pool_kind: str = "root",
    confidences: Sequence[float | None] | None = None,
    features: Sequence[dict[str, float] | None] | None = None,
    round_: int = 0,
) -> SEL.SealedPool:
    confs = list(confidences if confidences is not None else [0.5] * len(answers))
    feats = list(features if features is not None else [None] * len(answers))
    return SEL.sealed_pool(
        item_id, pool_kind,
        [candidate(f"d{index}", answer, valid=answer is not None, confidence=conf,
                   features=feature, agent_slot=index, round_=round_)
         for index, (answer, conf, feature) in enumerate(zip(answers, confs, feats))],
        round_=round_,
    )


def write_episode_item(
    run_root: Path,
    cell_id: str,
    item_id: str,
    *,
    protocol: str,
    round_: int,
    answers: Sequence[str | None],
    checkpoint: str = K.CHECKPOINT_SMALL,
    episode_id: str | None = None,
) -> Path:
    """Lay down one ``cells/<cell_id>/items/<item_id>.json`` the pool reader can read."""
    episode_id = episode_id or f"{cell_id}|{item_id}"
    decisions = []
    for slot, answer in enumerate(answers):
        valid = answer is not None
        decisions.append({
            "decision_id": f"{episode_id}|r{round_}|a{slot}",
            "episode_id": episode_id,
            "agent_id": slot,
            "round": int(round_),
            "phase": "revise",
            "protocol": protocol,
            "checkpoint": checkpoint,
            "answer_raw": answer,
            "answer_class": S2.upper_class(answer) if valid else None,
            "content_sha256": f"sha-{slot}",
            "evidence_text": f"evidence {slot}",
            "confidence": 0.4 + 0.1 * slot,
            "parse_status": K.ParseStatus.OK.value if valid else K.ParseStatus.MISSING_FINAL.value,
            "fallback": False,
        })
    payload = {
        "schema_version": 1,
        "source_id": item_id,
        "status": "complete",
        "episodes": [{
            "episode_id": episode_id, "item_id": item_id, "protocol": protocol,
            "checkpoint": checkpoint, "cohort": K.Cohort.NATURAL.value,
            "decisions": decisions,
        }],
    }
    path = run_root / "cells" / cell_id / "items" / f"{item_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- logprobs


def logprobs_block(
    tokens: Sequence[str],
    *,
    alternatives: dict[int, Sequence[tuple[str, float]]] | None = None,
    logprob: float = -0.25,
    top_k: int = K.RECEIVER_TOP_LOGPROBS,
) -> list[dict[str, Any]]:
    """An OpenAI ``choices[0].logprobs.content`` block with controlled top-k alternatives."""
    alternatives = alternatives or {}
    out: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        tops = [{"token": token, "logprob": logprob}]
        for text, value in alternatives.get(index, ()):  # type: ignore[arg-type]
            tops.append({"token": text, "logprob": float(value)})
        while len(tops) < min(top_k, 3):
            tops.append({"token": f"alt{len(tops)}", "logprob": logprob - len(tops)})
        out.append({"token": token, "logprob": logprob, "top_logprobs": tops[:top_k]})
    return out


# --------------------------------------------------------------------------- cells


def receiver_cell(programs: Sequence[Any], *, presentation: str, **kwargs: Any):
    builder = ard_cells.ArdBuilder()
    made = builder.emit(
        ArdCellKind.RECEIVER, list(programs), cohort=K.Cohort.RECEIVER,
        checkpoint=K.CHECKPOINT_SMALL,
        options={"presentation": presentation,
                 "checks_per_packet": K.RECEIVER_CHECKS_PER_PACKET,
                 "top_logprobs": K.RECEIVER_TOP_LOGPROBS},
        **kwargs)
    return made[0]


def audit_cell(tasks: Sequence[Any], *, pool: str, state_sha: str = "ab" * 32, **kwargs: Any):
    builder = ard_cells.ArdBuilder()
    made = ard_cells.build_e4_selector_audit(
        builder, list(tasks), checkpoint=K.CHECKPOINT_SMALL, state_sha=state_sha,
        pools=(pool,), **kwargs)
    return made[0]


def shared_tokenizer():
    """One :class:`~tests.study.fake_vllm.FakeTokenizer` for the server *and* the runner.

    ``FakeTokenizer.decode`` only knows the tokens its own instance has encoded, so a report
    whose tags never appeared as bare whitespace tokens in the prompt would come back as
    ``<unk>`` through a second instance.  The server encodes the completion it invents, so
    sharing the instance is what makes the decoded report span real text (the same advice
    ``wp1_support.fake_server`` gives).
    """
    from tests.study.fake_vllm import FakeTokenizer

    return FakeTokenizer()


def cell_runner(cell, run_root: Path, *, task_loader=None, **overrides: Any) -> int:
    """Run one ARD cell through the real WP4 runner and return its exit code."""
    kwargs = S4.runner_kwargs(**overrides)
    if task_loader is not None:
        kwargs["task_loader"] = task_loader
    return ard_runner.run_ard_cell(cell, run_root, run_root, 0, threading.Event(), **kwargs)


def freeze_state_register(run_root: Path, sha: str | None = None) -> str:
    """Freeze the ``state:<sha>`` register a SELECTOR_AUDIT cell depends on."""
    return ard_runner.freeze_register(run_root, "state", {"schema": "ard36-state-v1", "states": []})


# Re-exported so a WP5 test never has to import three support modules for one fixture.
ard_config = S1.ard_config
fake_server = S4.fake_server
make_run_root = S4.make_run_root
synthetic_task = S4.synthetic_task
synthetic_tasks = S4.synthetic_tasks
write_items = S4.write_items
upper_class = S2.upper_class
report = S2.report
