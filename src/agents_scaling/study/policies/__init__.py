"""Generation policies of the study (WP4; spec §4.2, §4.6; architecture §1.11).

``policy_for(method, cell)`` returns the frozen policy object for a cell.  Modules are
imported lazily so that importing this package never pulls the prompt renderers or the
WP3 contracts before they are needed.  No policy imports ``study.evaluation`` or
``study.data.protected`` (§10.6).
"""

from __future__ import annotations

from typing import Any

from agents_scaling.study.types import CellSpec, Method


def policy_for(cell: CellSpec, **overrides: Any):
    """The :class:`~agents_scaling.study.policies.base.Policy` implementing ``cell.method``."""
    method = Method(cell.method)
    if method is Method.S_FRESH:
        from agents_scaling.study.policies.s_fresh import SFreshPolicy

        return SFreshPolicy(**overrides)
    if method is Method.S_HISTORY:
        from agents_scaling.study.policies.s_history import SHistoryPolicy

        return SHistoryPolicy(**overrides)
    if method is Method.IND_VOTE:
        from agents_scaling.study.policies.ind_vote import IndVotePolicy

        return IndVotePolicy(**overrides)
    if method is Method.DEC:
        from agents_scaling.study.policies.dec import DecPolicy

        return DecPolicy(**overrides)
    if method is Method.DEC_ONE_ROUND:
        from agents_scaling.study.policies.dec import DecOneRoundPolicy

        return DecOneRoundPolicy(**overrides)
    if method is Method.IND_PRIVATE_REVISION:
        from agents_scaling.study.policies.dec import IndPrivateRevisionPolicy

        return IndPrivateRevisionPolicy(**overrides)
    if method is Method.CEN_FLAT:
        from agents_scaling.study.policies.cen_flat import CenFlatPolicy

        return CenFlatPolicy(**overrides)
    if method is Method.DEGREE:
        from agents_scaling.study.policies.degree import DegreePolicy

        return DegreePolicy(**overrides)
    raise ValueError(f"{method.value} is not a generation policy (F banks are produced by the runner)")


__all__ = ["policy_for"]
