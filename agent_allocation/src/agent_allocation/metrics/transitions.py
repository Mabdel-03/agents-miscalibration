"""Per-slot correctness transitions and the counting identity (MASTER_SPEC §12, §15.14).

States are ``C``/``W``/``I``/``U`` (``M``, an absent candidate, is folded into ``U`` here and counted separately).
Sixteen kinds ``"<from><to>"`` are always reported, zeros included. The bookkeeping identity

    C_next - C_prev = sum_{x != C} N_xC - sum_{x != C} N_Cx

holds for every consecutive pair and is checked with a raise: it can fail only when the transition rows and the
team counts come from different populations. On a fully known step (no ``U``) it is the binary identity
``C_next - C_prev = W->C - C->W`` with ``W`` meaning "known not correct" (``W`` or ``I``): ``n_w_to_c = N_WC + N_IC``,
``n_c_to_w = N_CW + N_CI``. Parse failure and a valid wrong answer stay distinguishable in the fine kinds, and an
answer switch (``answer_changed``) is computed only between two valid text answers -- never for code, whose source
differences are not semantic switches.

Config keys consumed: none.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from agent_allocation._vendor.metrics_reference import transition_counts as binary_transition_counts
from agent_allocation.metrics.outcomes import CandidateState

__all__ = ["STATES", "TRANSITION_KINDS", "binary_transition_counts", "check_identity", "fold", "kind_counts"]

STATES: tuple[str, ...] = ("C", "W", "I", "U")
TRANSITION_KINDS: tuple[str, ...] = tuple(a + b for a in STATES for b in STATES)


def fold(state: CandidateState) -> str:
    return "U" if state is CandidateState.MISSING else state.value


def kind_counts(pairs: Iterable[tuple[CandidateState, CandidateState]]) -> dict[str, int]:
    counts = {kind: 0 for kind in TRANSITION_KINDS}
    for before, after in pairs:
        counts[fold(before) + fold(after)] += 1
    return counts


def check_identity(before: Sequence[CandidateState], after: Sequence[CandidateState],
                   counts: Mapping[str, int]) -> int:
    """Raise unless ``C_next - C_prev`` equals the net flow into ``C``; returns ``C_next - C_prev``."""
    if len(before) != len(after):
        raise ValueError("a transition must match the same agent slots")
    delta = sum(s is CandidateState.CORRECT for s in after) - sum(s is CandidateState.CORRECT for s in before)
    inflow = sum(counts[x + "C"] for x in STATES if x != "C")
    outflow = sum(counts["C" + x] for x in STATES if x != "C")
    if delta != inflow - outflow:
        raise ValueError(f"transition identity violated: delta {delta} != {inflow} - {outflow}")
    if all(fold(s) != "U" for s in (*before, *after)):
        binary = binary_transition_counts([int(s is CandidateState.CORRECT) for s in before],
                                          [int(s is CandidateState.CORRECT) for s in after])
        if binary["WC"] != counts["WC"] + counts["IC"] or binary["CW"] != counts["CW"] + counts["CI"]:
            raise ValueError("binary transition identity disagrees with the fine-grained counts")
    return delta
