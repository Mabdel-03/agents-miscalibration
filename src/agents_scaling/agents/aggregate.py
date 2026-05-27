"""Aggregation helpers: turn a set of agent outputs into a system answer + confidences.

System 'confidence' is genuinely ambiguous for an aggregated answer (plan Risk 2), so we
record it under MULTIPLE definitions and let the calibration analysis compute ECE for
each. The winning answer is by majority vote (ties broken by summed option-logprob mass).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from agents_scaling.agents.base_agent import AgentOutput


def majority_vote(outputs: list[AgentOutput]) -> str | None:
    """Most common answer; ties broken by total option-logprob mass for the answer."""
    answers = [o.answer_choice for o in outputs if o.answer_choice is not None]
    if not answers:
        return None
    counts = Counter(answers)
    top = counts.most_common()
    best_count = top[0][1]
    tied = [a for a, c in top if c == best_count]
    if len(tied) == 1:
        return tied[0]
    # Break tie by summed option-logprob probability mass.
    mass: dict[str, float] = defaultdict(float)
    for o in outputs:
        if o.answer_choice in tied:
            mass[o.answer_choice] += o.option_logprobs.get(o.answer_choice, 0.0)
    if mass:
        return max(mass, key=mass.get)
    return tied[0]


def system_confidences(outputs: list[AgentOutput], final_answer: str | None) -> dict[str, float]:
    """Multiple system-confidence definitions for the aggregated ``final_answer``.

    * vote_fraction        — fraction of agents that chose the final answer.
    * mean_agreeing_logprob — mean option-logprob confidence among agreeing agents.
    * mean_agreeing_verbal  — mean verbalized confidence among agreeing agents.
    * mean_all_logprob      — mean option-logprob mass on the final answer over all agents.
    """
    if not outputs or final_answer is None:
        return {}
    agreeing = [o for o in outputs if o.answer_choice == final_answer]
    n = len(outputs)
    vote_fraction = len(agreeing) / n if n else 0.0

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    mean_agreeing_logprob = _mean([o.option_logprobs.get(final_answer, 0.0) for o in agreeing])
    mean_agreeing_verbal = _mean([o.verbalized_conf for o in agreeing if o.verbalized_conf is not None])
    mean_all_logprob = _mean([o.option_logprobs.get(final_answer, 0.0) for o in outputs])

    return {
        "vote_fraction": vote_fraction,
        "mean_agreeing_logprob": mean_agreeing_logprob,
        "mean_agreeing_verbal": mean_agreeing_verbal,
        "mean_all_logprob": mean_all_logprob,
    }
