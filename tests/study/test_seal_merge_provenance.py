"""seal._merge: sealed decisions are immutable; a re-derivation differing only in JUDGE_BEST provenance is kept."""
import pytest

from agents_scaling.study.selection import seal as S
from agents_scaling.study.types import ProtocolError


def _entry(**over):
    e = {"selector_id": "JUDGE_BEST", "pool_id": "p1", "cell_id": "F.BANK.32B.N1.B0.F10.e0.s000", "score_cell_id": "eval.JUDGE_BEST.x.s000",
         "record": {"selected_candidate_id": "c1", "scores": {"c1": 0.9, "c2": 0.1}}}
    e.update(over)
    return e


def test_merge_keeps_original_when_only_score_cell_differs():
    reg = {"selections": {}}
    added, kept = S._merge(reg, "selections", {"sid": _entry()}, 100.0, "selection")
    assert (added, kept) == (1, 0) and reg["selections"]["sid"]["sealed_at"] == 100.0
    added, kept = S._merge(reg, "selections", {"sid": _entry(score_cell_id="eval.JUDGE_BEST.x.w3.s000")}, 200.0, "selection")
    assert (added, kept) == (0, 1)
    assert reg["selections"]["sid"]["score_cell_id"] == "eval.JUDGE_BEST.x.s000" and reg["selections"]["sid"]["sealed_at"] == 100.0


def test_merge_refuses_a_changed_decision():
    reg = {"selections": {}}
    S._merge(reg, "selections", {"sid": _entry()}, 100.0, "selection")
    with pytest.raises(ProtocolError, match="immutable"):
        S._merge(reg, "selections", {"sid": _entry(score_cell_id="eval.JUDGE_BEST.x.w3.s000", record={"selected_candidate_id": "c2", "scores": {"c1": 0.1, "c2": 0.9}})}, 200.0, "selection")
    with pytest.raises(ProtocolError, match="immutable"):
        S._merge(reg, "selections", {"sid": _entry(pool_id="p2")}, 200.0, "selection")
