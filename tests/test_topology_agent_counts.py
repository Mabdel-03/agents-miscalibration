"""Topology accounting for variable-size MAS cells."""

from agents_scaling.agents.aggregate import majority_vote
from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.topologies.centralized import Centralized
from agents_scaling.agents.topologies.decentralized import Decentralized
from agents_scaling.agents.topologies.independent import Independent
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import ContextShareLevel


class ContextTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


TOKENIZER = ContextTokenizer()


class FakeAgent:
    def __init__(self, agent_id: str, answer: str = "A", confidence: float = 0.8):
        self.agent_id = agent_id
        self._answer = answer
        self._confidence = confidence

    def answer(self, q, round_idx=0, peer_context="", max_tokens=1024, seed=None):
        return AgentOutput(
            agent_id=self.agent_id,
            round=round_idx,
            answer_choice=self._answer,
            raw_text=f"Answer: {self._answer}",
            cot_text="because",
            intermediate_results=f"therefore {self._answer}",
            option_logprobs={"A": self._confidence, "B": 1.0 - self._confidence},
            verbalized_conf=self._confidence,
            prompt_tokens=1,
            completion_tokens=1,
        )


def _question():
    return Question(
        qid="q0",
        benchmark="toy",
        prompt_stem="Which option?",
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=["alpha", "beta"],
    )


def _agents(n: int):
    return [FakeAgent(f"agent{i}") for i in range(n)]


def test_independent_accounting_for_variable_agent_counts():
    for n in [2, 4, 5, 6]:
        res = Independent(_agents(n), ContextShareLevel.ARTIFACT_ONLY, rounds=2).run(_question())
        assert res.n_agents == n
        assert res.n_turns == n
        assert res.n_messages == 0
        assert len(res.per_agent) == n


def test_decentralized_accounting_for_variable_agent_counts():
    for n in [2, 4, 5, 6]:
        res = Decentralized(
            _agents(n),
            ContextShareLevel.PLUS_COT,
            rounds=2,
            context_tokenizer=TOKENIZER,
        ).run(_question())
        assert res.n_agents == n
        assert res.n_turns == 2 * n
        assert res.n_messages == n * (n - 1)
        assert len(res.per_agent) == 2 * n


def test_centralized_accounting_for_variable_agent_counts():
    for n in [2, 4, 5, 6]:
        res = Centralized(
            _agents(n),
            ContextShareLevel.PLUS_COT,
            rounds=2,
            context_tokenizer=TOKENIZER,
        ).run(_question())
        assert res.n_agents == n
        assert res.n_turns == 2 * n
        assert res.n_messages == 3 * (n - 1)
        assert len(res.per_agent) == 2 * n


def test_even_agent_vote_tie_uses_summed_option_mass():
    outs = [
        AgentOutput("a0", 0, "A", "", "", "", {"A": 0.4, "B": 0.6}),
        AgentOutput("a1", 0, "B", "", "", "", {"A": 0.1, "B": 0.7}),
    ]
    assert majority_vote(outs) == "B"
