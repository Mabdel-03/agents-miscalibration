"""The launch gate's call-type table must speak WP2's phase vocabulary (spec §16.3).

``scripts/ard36_smoke.py`` turns pilot decisions into the ``--measured`` JSON the launch gate
reads.  Its phase table was written against a vocabulary ``episode.py`` does not use ("revise",
"hub"), and the lookup ended in ``or "ROOT"`` -- so every revision was silently counted as a
root.  The gate then saw no PEER_REVISE or BALANCED_REVISE row at all, and read a root parse
validity that was really a mixture of roots and revisions.  A defaulting lookup over a
vocabulary owned by another module is the whole defect, so these tests pin both halves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import episode as E
from agents_scaling.ard.gates import CALL_TYPES

_SPEC = importlib.util.spec_from_file_location(
    "ard36_smoke", Path(__file__).resolve().parents[2] / "scripts" / "ard36_smoke.py"
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _dec(phase: str, protocol: str | None = None, budget: int = K.THINKING_TOKENS_LOW) -> dict:
    return {"phase": phase, "protocol": protocol or "NONE", "budget_tokens": budget}


def test_every_phase_wp2_emits_maps_to_a_call_type_the_gate_knows() -> None:
    phases = (E.PHASE_ROOT, E.PHASE_UPDATE, E.PHASE_LEADER, E.PHASE_WORKER, K.PURPOSE_RECEIVER)
    for phase in phases:
        for protocol in (None, "PEER", "PRIVATE", "BALANCED", "FRESH", "HUB5",
                         "GATE", "RANDOM_PROTECT", "CONF_PROTECT"):
            assert smoke.call_type(_dec(phase, protocol)) in CALL_TYPES, (phase, protocol)


def test_the_revision_protocols_are_not_counted_as_roots() -> None:
    """The defect itself: a revision that lands in ROOT costs the gate two rows and one metric."""
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "PEER")) == "PEER_REVISE"
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "PRIVATE")) == "PEER_REVISE"
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "GATE")) == "PEER_REVISE"
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "RANDOM_PROTECT")) == "PEER_REVISE"
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "BALANCED")) == "BALANCED_REVISE"
    assert smoke.call_type(_dec(E.PHASE_LEADER, "HUB5")) == "HUB_LEADER"
    assert smoke.call_type(_dec(E.PHASE_WORKER, "HUB5")) == "HUB_WORKER"
    # A FRESH update really is a fresh independent draw, so it costs what a root costs.
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "FRESH")) == "ROOT"
    # A root keeps its call type even inside a protocol cell, where it carries that tag.
    assert smoke.call_type(_dec(E.PHASE_ROOT, "PEER")) == "ROOT"


def test_the_budget_arm_and_the_receiver_are_separated() -> None:
    assert smoke.call_type(_dec(E.PHASE_UPDATE, "PEER", K.THINKING_TOKENS_HIGH)) == "HIGH_BUDGET"
    # The receiver's own high arm stays RECEIVER: it is a different prompt shape, not E2d.
    assert smoke.call_type(_dec(K.PURPOSE_RECEIVER, None, K.THINKING_TOKENS_HIGH)) == "RECEIVER"
    assert smoke.call_type(_dec(K.PURPOSE_RECEIVER)) == "RECEIVER"


def test_an_unknown_phase_is_loud_rather_than_silently_a_root() -> None:
    """What let the original defect through was the trailing ``or "ROOT"``."""
    with pytest.raises(SystemExit, match="unknown decision phase"):
        smoke.call_type(_dec("some_new_phase", "PEER"))
