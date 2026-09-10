"""Shared helpers for the ARD36 WP2 (protocols / episode) tests.

Two kinds of fixture live here:

* **pure** builders (``report``, ``round_state``) that assemble a
  :class:`~agents_scaling.ard.states.RoundState` from hand-written reports, so the packet
  plans can be tested exactly — a specified class multiset, a specified invalid peer — with
  no server and no grader;
* a **fake-server harness** (:class:`Harness`) that binds a real
  ``ard.client.generate_decision`` to an in-process
  :class:`~tests.study.fake_vllm.FakeVllmServer`, so the episode-loop assertions are made
  against the request bodies the server actually received.

Deliberately not a ``conftest.py``: ``tests/ard/`` is shared ground for every ARD work
package and this file belongs to WP2 alone (the same convention as ``wp1_support.py``).
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from tests.ard import wp1_support as S1  # sets HF_HOME / offline before transformers loads

from agents_scaling.ard import contract as K
from agents_scaling.ard import packets as PK
from agents_scaling.ard import protocols as P
from agents_scaling.ard.client import DecisionRequest, generate_decision
from agents_scaling.ard.episode import run_episode
from agents_scaling.ard.report import Report, parse_report
from agents_scaling.ard.states import AgentState, ReportRef, RoundState
from tests.study import fake_vllm as F

STUDY_SEED = bytes.fromhex("a1" * 32)


# --------------------------------------------------------------------------- items


@dataclass(frozen=True)
class StubTask:
    """The public columns ``run_episode`` reads (``ard.items.ArdTask`` shape, no gold)."""

    item_id: str = "math500:test-0001"
    source_id: str = "math500:test-0001"
    split: str = "test"
    family: str = K.Family.MATH500.value
    task_text: str = "What is 6 times 7? Answer with the integer."


def upper_class(answer: str | None) -> str | None:
    """A gold-independent stand-in for ``ard.graders.canonical_class`` (no sympy needed)."""
    if not answer:
        return None
    return " ".join(str(answer).split()).upper() or None


# --------------------------------------------------------------------------- reports


def report_text(
    answer: str | None, *, evidence: str | None = "checked the arithmetic", confidence: float | None = 0.7
) -> str:
    parts = []
    if answer is not None:
        parts.append(f"{K.FINAL_OPEN}{answer}{K.FINAL_CLOSE}")
    if evidence is not None:
        parts.append(f"{K.EVIDENCE_OPEN}{evidence}{K.EVIDENCE_CLOSE}")
    if confidence is not None:
        parts.append(f"{K.CONFIDENCE_OPEN}{confidence}{K.CONFIDENCE_CLOSE}")
    return "".join(parts)


def completion_ids(n_final: int = 6) -> tuple[int, ...]:
    """A well-formed completion: ``<think>`` … ``</think>`` … ``<|im_end|>``."""
    return (K.THINK_START_ID, 11, 12, K.THINK_END_ID) + tuple(range(20, 20 + n_final)) + (K.IM_END_ID,)


def report(
    answer: str | None,
    *,
    evidence: str | None = "checked the arithmetic",
    confidence: float | None = 0.7,
    finish_reason: str = "stop",
) -> Report:
    """A parsed report with the requested content (``answer=None`` → an invalid report)."""
    return parse_report(report_text(answer, evidence=evidence, confidence=confidence), finish_reason, completion_ids())


def ref(rep: Report, *, slot: int, round_: int, decision_id: str | None = None) -> ReportRef:
    return ReportRef(
        decision_id=decision_id or f"d-{round_}-{slot}-{rep.content_sha256[:8]}",
        report=rep,
        round_=round_,
        agent_slot=slot,
    )


def round_state(
    reports: Sequence[Report],
    *,
    round_: int = 0,
    roots: Sequence[Report] | None = None,
    protocol: K.Protocol = K.Protocol.PEER,
    item_id: str = "math500:test-0001",
    checkpoint: str = K.CHECKPOINT_SMALL,
    cohort: str = K.Cohort.NATURAL.value,
    study_seed: bytes = STUDY_SEED,
    leader: int | None = None,
    count: Callable[[str], int] | None = None,
) -> RoundState:
    """A :class:`RoundState` whose five agents hold exactly ``reports`` at ``round_``."""
    if len(reports) != K.N_AGENTS:
        raise ValueError("five reports, one per agent slot")
    root_reports = list(roots if roots is not None else reports)
    pairing_id = f"{item_id}|{checkpoint}|{cohort}|e0"
    latest_refs = [ref(r, slot=i, round_=round_) for i, r in enumerate(reports)]
    root_refs = [ref(r, slot=i, round_=0) for i, r in enumerate(root_reports)]
    agents = tuple(
        AgentState(
            item_id=item_id,
            checkpoint=checkpoint,
            agent_slot=i,
            round_=round_,
            root=root_refs[i],
            latest=latest_refs[i],
            episode_id=f"{pairing_id}|{protocol.value}",
            protocol=protocol.value,
            pairing_id=pairing_id,
        )
        for i in range(K.N_AGENTS)
    )
    archive = {0: tuple(root_refs)}
    if round_ != 0:
        archive[round_] = tuple(latest_refs)
    return RoundState(
        episode_id=f"{pairing_id}|{protocol.value}",
        pairing_id=pairing_id,
        item_id=item_id,
        protocol=protocol,
        checkpoint=checkpoint,
        round_=round_,
        agents=agents,
        archive=archive,
        pseudonyms=PK.pseudonym_map(study_seed, pairing_id),
        study_seed=study_seed,
        classify=upper_class,
        count=count,
        leader=leader,
        task_text="What is 6 times 7? Answer with the integer.",
        family=K.Family.MATH500.value,
    )


# --------------------------------------------------------------------------- harness


@dataclass
class Harness:
    """A bound ``generate`` over a live fake endpoint, plus the recorded request bodies."""

    server: F.FakeVllmServer
    tokenizer: Any
    generate: Callable[[DecisionRequest], Any]
    study_seed: bytes
    model_cell: str
    count: Callable[[str], int]

    def prompts(self) -> list[str]:
        """Every prompt body the server received, in arrival order."""
        return [
            "\n".join(str(m.get("content", "")) for m in entry["body"]["messages"])
            for entry in self.server.requests
            if entry["path"] == "/v1/chat/completions"
        ]

    def run(self, protocol: K.Protocol, **kwargs: Any):
        item = kwargs.pop("item", StubTask())
        return run_episode(
            item,
            protocol,
            kwargs.pop("checkpoint", K.CHECKPOINT_SMALL),
            generate=self.generate,
            study_seed=self.study_seed,
            model_cell=self.model_cell,
            classify=upper_class,
            count=self.count,
            **kwargs,
        )


@contextmanager
def harness(tmp_path: Path, **server_kwargs: Any) -> Iterator[Harness]:
    """``with harness(tmp_path) as h:`` — fake server, store, pool and a bound generate."""
    cfg = S1.ard_config()
    run_root = S1.make_run_root(tmp_path)
    tokenizer = F.FakeTokenizer()
    context = S1.make_context(cfg, tokenizer)
    cost_fn = S1.cost_fn_for(cfg.checkpoints) if hasattr(S1, "cost_fn_for") else None
    if cost_fn is None:
        from agents_scaling.ard.client import cost_fn_for

        cost_fn = cost_fn_for(cfg.checkpoints)
    server_kwargs.setdefault("tokenizer", tokenizer)
    with S1.fake_server(run_root, K.PROFILE_SMALL, **server_kwargs) as server:
        pool = S1.make_pool(run_root, K.PROFILE_SMALL)
        store = S1.make_store(run_root)
        generate = functools.partial(
            generate_decision, store=store, pool=pool, cost_fn=cost_fn, context=context
        )
        yield Harness(
            server=server,
            tokenizer=tokenizer,
            generate=generate,
            study_seed=bytes.fromhex(cfg.study_seed_hex),
            model_cell=cfg.checkpoints[K.CHECKPOINT_SMALL].model_cell,
            count=PK.make_counter(tokenizer),
        )
