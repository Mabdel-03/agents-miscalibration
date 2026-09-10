"""ARD36 round states: the ``state:<sha>`` append-only register (WP2).

Spec §6.4 fixes what a state *is*: "All iterative protocols use: task, the agent's
original root report, its latest report, and the specified current peer packet.  If root
and latest report are identical, include one copy."  Everything else an agent might
remember (its thinking trace, other rounds' reports, the archive) is deliberately outside
the deployed memory, so an ARD36 **agent state** is exactly

``(item, checkpoint, agent slot, round, own root report, own latest report)``.

That tuple is content-addressed into ``state:<sha>`` and appended to a register modelled
on the frozen ``study/selection/seal.py`` (``_merge`` / ``_write_register``): entries are
append-only, a re-assertion must match byte-for-byte, and a changed entry raises
``ProtocolError`` — seals are immutable.  Plan §5.3 makes ``state:<sha>`` a hard cell
dependency prefix, which is what lets a FORK or SUFFIX cell refuse to run before its
parent trajectory is sealed.

The state id deliberately excludes the protocol and the episode id.  Two protocols that
reach byte-identical memory *have* the same state: at ``t=0`` PRIVATE and PEER share one
root state, which is precisely why the E4 "free" ``t=0`` counterfactual pair costs nothing
(spec §12.1, plan §5.4).  Provenance (which episodes visited a state) is recorded beside
the entry and is excluded from the immutability comparison, exactly as ``seal._merge``
excludes ``score_cell_id``.

Spec §18 ``decisions.parquet`` reads ``parent_state_id`` from here; ``forks.parquet``
reads ``shared_prefix_hash``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard.report import Report
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.parse.candidate import jcs_with_numbers
from agents_scaling.study.types import ProtocolError

__all__ = [
    "AgentState",
    "ReportRef",
    "RoundState",
    "STATE_PREFIX",
    "STATE_SCHEMA_VERSION",
    "StateRegister",
    "state_id",
    "states_path",
]

STATE_PREFIX = "state:"
STATE_SCHEMA_VERSION = 1
STATES_FILE = "states.json"


# --------------------------------------------------------------------------- reports


@dataclass(frozen=True)
class ReportRef:
    """One report together with the decision that produced it.

    ``decision_id`` is the content-addressed request id (spec §18 cache identity), so two
    protocols that issue the identical request share one decision id and one state.
    """

    decision_id: str
    report: Report
    round_: int
    agent_slot: int

    @property
    def content_sha256(self) -> str:
        return self.report.content_sha256

    @property
    def valid(self) -> bool:
        return self.report.valid

    def to_row(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, "round": self.round_, "agent_slot": self.agent_slot,
                **self.report.to_dict()}


# --------------------------------------------------------------------------- agent state


@dataclass(frozen=True)
class AgentState:
    """The deployed memory of one agent at the end of round ``round_`` (spec §6.4)."""

    item_id: str
    checkpoint: str
    agent_slot: int
    round_: int
    root: ReportRef
    latest: ReportRef
    #: Provenance only — never part of the state identity (see the module docstring).
    episode_id: str = ""
    protocol: str = ""
    pairing_id: str = ""

    def __post_init__(self) -> None:
        if not 0 <= int(self.agent_slot) < K.N_AGENTS:
            raise ValueError(f"agent_slot must be in 0..{K.N_AGENTS - 1}, got {self.agent_slot}")
        if int(self.round_) not in K.ROUNDS:
            raise ValueError(f"round must be one of {K.ROUNDS}, got {self.round_}")

    @property
    def deduplicated(self) -> bool:
        """Spec §6.4: "If root and latest report are identical, include one copy.\""""
        return self.root.decision_id == self.latest.decision_id

    def identity_object(self) -> dict[str, Any]:
        """The exact object hashed into :func:`state_id` (no protocol, no episode)."""
        return {
            "schema": "ard_state_v1",
            "schema_version": STATE_SCHEMA_VERSION,
            "item_id": str(self.item_id),
            "checkpoint": str(self.checkpoint),
            "agent_slot": int(self.agent_slot),
            "round": int(self.round_),
            "root_decision_id": str(self.root.decision_id),
            "root_content_sha256": str(self.root.content_sha256),
            "latest_decision_id": str(self.latest.decision_id),
            "latest_content_sha256": str(self.latest.content_sha256),
        }

    @property
    def state_id(self) -> str:
        return state_id(self)

    def to_entry(self) -> dict[str, Any]:
        return dict(self.identity_object())


def state_id(state: AgentState) -> str:
    """``state:<sha256 of the canonical state object>`` (plan §5.3 dependency prefix)."""
    return STATE_PREFIX + identity.sha256_hex(jcs_with_numbers(state.identity_object()))


# --------------------------------------------------------------------------- round state


@dataclass(frozen=True)
class RoundState:
    """The whole team at the end of round ``round_`` — the input of ``packet_plan``.

    ``agents`` is indexed by *true* slot.  ``archive`` holds every report ever produced, per
    round (spec §6.4: "Every protocol has the same external archive for retrospective
    analyses; the deployed agent sees only the memory allowed above") — packets are built
    from ``agents`` and the previous round of ``archive``, never from the whole archive.

    ``classify`` is the injected gold-independent canonical classifier
    (``ard.graders.canonical_class`` bound to this item's family).  Injecting it keeps the
    protocol layer free of the grader import and makes the BALANCED class collapse unit
    testable without sympy.
    """

    episode_id: str
    pairing_id: str
    item_id: str
    protocol: K.Protocol
    checkpoint: str
    round_: int
    agents: tuple[AgentState, ...]
    archive: Mapping[int, tuple[ReportRef, ...]]
    pseudonyms: tuple[str, ...]
    study_seed: bytes
    classify: Callable[[str | None], str | None]
    #: Recipient-token counter of the serving tokenizer (``packets.make_counter``).  The
    #: packet clipper needs it; ``None`` falls back to the conservative approximation of
    #: ``ard.protocols._approx_counter``.
    count: Callable[[str], int] | None = None
    leader: int | None = None
    task_text: str = ""
    family: str = ""

    def __post_init__(self) -> None:
        if len(self.agents) != K.N_AGENTS:
            raise ValueError(f"a team is {K.N_AGENTS} agents, got {len(self.agents)}")
        if len(self.pseudonyms) != K.N_AGENTS:
            raise ValueError("one pseudonym per agent slot")
        for slot, agent in enumerate(self.agents):
            if agent.agent_slot != slot:
                raise ProtocolError("RoundState.agents must be indexed by true agent slot")

    def latest(self, slot: int) -> ReportRef:
        return self.agents[int(slot)].latest

    def reports_of_round(self, round_: int) -> tuple[ReportRef, ...]:
        return tuple(self.archive.get(int(round_), ()))

    def canonical_class(self, report: Report) -> str | None:
        """Gold-independent class of a report; ``None`` for invalid or unkeyable answers."""
        if not report.valid:
            return None
        return self.classify(report.final_answer)

    def archive_classes(self) -> set[str]:
        """Every canonical class ever produced (the monotone archive-coverage series)."""
        out: set[str] = set()
        for refs in self.archive.values():
            for ref in refs:
                name = self.canonical_class(ref.report)
                if name:
                    out.add(name)
        return out

    def state_ids(self) -> tuple[str, ...]:
        return tuple(agent.state_id for agent in self.agents)


# --------------------------------------------------------------------------- register


def states_path(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / "states" / STATES_FILE


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds")


class StateRegister:
    """Append-only ``state:<sha>`` register (``study/selection/seal.py`` ``_merge`` shape).

    * A new state is appended with its seal stamps.
    * A re-asserted state must match byte-for-byte on every identity field; a difference is
      a harness defect and raises ``ProtocolError`` ("seals are immutable").
    * ``episodes`` is provenance: it is unioned across re-assertions and excluded from the
      comparison, exactly as ``seal._merge`` excludes ``score_cell_id``.
    """

    #: Keys stamped by the register itself, never compared.
    STAMPS = ("sealed_at", "sealed_at_iso")
    #: Provenance keys merged rather than compared.
    PROVENANCE = ("episodes",)

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.register: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": STATE_SCHEMA_VERSION, "states": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProtocolError(f"{self.path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("states"), dict):
            raise ProtocolError(f"{self.path} is not a state register")
        return data

    # -- reads ---------------------------------------------------------------
    def __contains__(self, sid: str) -> bool:
        return str(sid) in self.register["states"]

    def __len__(self) -> int:
        return len(self.register["states"])

    def get(self, sid: str) -> dict[str, Any] | None:
        entry = self.register["states"].get(str(sid))
        return dict(entry) if entry is not None else None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.register["states"]))

    def require(self, sid: str) -> dict[str, Any]:
        """Resolve a ``state:`` dependency or refuse (plan §5.3 ``EXIT_DEPS_UNMET``)."""
        entry = self.get(sid)
        if entry is None:
            raise ProtocolError(f"state dependency {sid} is not sealed in {self.path}")
        return entry

    # -- writes --------------------------------------------------------------
    def merge(
        self,
        states: Sequence[AgentState],
        *,
        clock: Callable[[], float] = time.time,
    ) -> tuple[int, int]:
        """Append ``states``; returns ``(added, kept)``.  Raises on a changed entry."""
        now = float(clock())
        added = kept = 0
        for state in states:
            sid = state.state_id
            entry = state.to_entry()
            episodes = [state.episode_id] if state.episode_id else []
            existing = self.register["states"].get(sid)
            if existing is None:
                self.register["states"][sid] = {
                    **entry,
                    "episodes": sorted(set(episodes)),
                    "sealed_at": now,
                    "sealed_at_iso": _iso(now),
                }
                added += 1
                continue
            stripped = {
                k: v for k, v in existing.items() if k not in self.STAMPS + self.PROVENANCE
            }
            if stripped != entry:
                raise ProtocolError(
                    f"state {sid[:18]} already sealed with different content; states are immutable"
                )
            existing["episodes"] = sorted(set(existing.get("episodes", [])) | set(episodes))
            kept += 1
        return added, kept

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        io.atomic_write_text(
            self.path, json.dumps(self.register, indent=1, sort_keys=True) + "\n"
        )
        return self.path
