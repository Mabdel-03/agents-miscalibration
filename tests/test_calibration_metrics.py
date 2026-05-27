"""ECE/MCE/Brier on synthetic data with known calibration properties; plus the
final-producer system-confidence signal."""

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.aggregate import system_confidences
from agents_scaling.calibration.metrics import compute_calibration


def _out(aid, ans, lp_conf, verbal):
    """An AgentOutput with a given confidence in answer `ans`."""
    return AgentOutput(
        agent_id=aid, round=0, answer_choice=ans, raw_text="", cot_text="",
        intermediate_results="",
        option_logprobs={ans: lp_conf, "X": 1 - lp_conf},
        verbalized_conf=verbal,
    )


def test_final_producer_is_decisive_agreeing_agent():
    # 3 agents pick B with differing confidence; final answer B.
    outs = [_out("a0", "B", 0.5, 0.5), _out("a1", "B", 0.8, 0.7), _out("a2", "C", 0.9, 0.9)]
    sc = system_confidences(outs, "B")
    # decisive producer = most confident agent that chose B (a1, 0.8); not a2 (chose C).
    assert sc["final_producer_logprob"] == 0.8
    assert sc["final_producer_verbal"] == 0.7
    # mean over the winning side (a0,a1) = 0.65
    assert abs(sc["mean_producer_logprob"] - 0.65) < 1e-9


def test_final_producer_override_for_orchestrator():
    subs = [_out("s0", "B", 0.9, 0.9), _out("s1", "B", 0.8, 0.8)]
    orch = _out("orch", "B", 0.4, 0.45)
    sc = system_confidences(subs, "B", producers=[orch])
    # producer is the orchestrator, not the confident sub-agents.
    assert sc["final_producer_logprob"] == 0.4
    assert sc["final_producer_verbal"] == 0.45
    # but the pooled vote view still reflects the sub-agents.
    assert sc["vote_fraction"] == 1.0


def test_perfect_calibration_zero_ece():
    # Confidence exactly equals empirical accuracy within each bin.
    # 10 items at conf 0.9, 9 correct -> bin accuracy 0.9 == confidence.
    confs = [0.9] * 10
    corrects = [True] * 9 + [False]
    rep = compute_calibration(confs, corrects, n_bins=10)
    assert abs(rep.ece) < 1e-9
    assert abs(rep.mce) < 1e-9


def test_overconfident_has_positive_ece():
    # Always 100% confident but only 50% correct -> ECE ~ 0.5.
    confs = [1.0] * 10
    corrects = [True] * 5 + [False] * 5
    rep = compute_calibration(confs, corrects, n_bins=10)
    assert abs(rep.ece - 0.5) < 1e-6
    assert abs(rep.mce - 0.5) < 1e-6


def test_brier_bounds():
    rep = compute_calibration([1.0, 0.0], [False, True], n_bins=10)
    assert abs(rep.brier - 1.0) < 1e-9  # worst case
    rep2 = compute_calibration([1.0, 0.0], [True, False], n_bins=10)
    assert abs(rep2.brier - 0.0) < 1e-9  # best case


def test_empty_input():
    rep = compute_calibration([], [], n_bins=15)
    assert rep.n == 0 and rep.ece == 0.0


def test_confidence_one_lands_in_last_bin():
    # A confidence of exactly 1.0 must be counted (last-bin right edge inclusive).
    rep = compute_calibration([1.0], [True], n_bins=15)
    assert rep.n == 1
    assert sum(b.count for b in rep.bins) == 1
