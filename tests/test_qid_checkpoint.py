from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest

from agents_scaling.agents.base_agent import AgentOutput, SelfConsistencySample
from agents_scaling.agents.message_builder import PeerContextRender
from agents_scaling.agents.topologies.base import TopologyResult
from agents_scaling.agents.topologies.independent import Independent
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import (
    ContextShareLevel,
    ExperimentCell,
    ReasoningLevel,
    Topology,
)
from agents_scaling.experiment import io
from agents_scaling.experiment.qid_checkpoint import (
    CheckpointCorruptionError,
    CheckpointIdentityError,
    CheckpointingAgent,
    CoordinateAdmissionClosed,
    QIDCheckpoint,
)
from agents_scaling.serving.client import (
    QWEN_IM_END_TOKEN_ID,
    QWEN_THINK_END_TOKEN_ID,
    GenerationProtocolCensorError,
    GenerationTruncationError,
)
from agents_scaling.serving.profiles import ServingProfile


def _cell(*, topology: Topology = Topology.SINGLE_AGENT, n_agents: int = 1):
    return ExperimentCell(
        model_size="0.6B",
        context_share_level=ContextShareLevel.ARTIFACT_ONLY,
        prompt_complexity_level=0,
        reasoning_level=ReasoningLevel.OFF,
        topology=topology,
        benchmark="fixture",
        n_agents=n_agents,
        rounds=1,
        n_samples=3,
        n_questions=1,
        seed=17,
    )


def _question() -> Question:
    return Question(
        qid="fixture:0",
        benchmark="fixture",
        prompt_stem="What is the answer?",
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=["yes", "no"],
    )


def _profile() -> ServingProfile:
    return ServingProfile(
        name="fixture-profile",
        model_size="0.6B",
        hf_id="fixture/model",
        tp_size=1,
        max_model_len=4_226,
        served_model_name="0.6B",
    )


def _output(agent_id: str, *, round_idx: int = 0, answer: str = "A") -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        round=round_idx,
        answer_choice=answer,
        raw_text=f"reasoning; answer {answer}",
        cot_text="reasoning",
        intermediate_results=f"answer {answer}",
        option_logprobs={"A": 0.8, "B": 0.2},
        verbalized_conf=0.75,
        prompt_tokens=101,
        completion_tokens=19,
        reasoning_text="reasoning",
        reasoning_tokens=7,
        finish_reason="stop",
        reasoning_token_source="server",
        reasoning_word_count_legacy=1,
        generation_phase_finish_reasons=["stop"],
        generation_phase_seeds=[17],
        generation_phase_prompt_tokens=[101],
        generation_phase_completion_tokens=[19],
        generation_phase_requested_max_tokens=[4096],
        generation_phase_prompt_token_id_hashes=["a" * 64],
        generation_phase_completion_token_id_hashes=["b" * 64],
        endpoint_generation="node-a:8000:111",
    )


class _FakeAgent:
    def __init__(self, agent_id: str, *, fail_answer: bool = False) -> None:
        self.agent_id = agent_id
        self.answer_calls = 0
        self.sample_calls: list[int] = []
        self.fail_answer = fail_answer

    def prepare_calibration(self, question: Question):
        return None

    def answer(self, question: Question, *, round_idx: int = 0, **kwargs):
        self.answer_calls += 1
        if self.fail_answer:
            raise ConnectionError(f"{self.agent_id} failed")
        return _output(self.agent_id, round_idx=round_idx)

    def sample_one(
        self,
        question: Question,
        *,
        sample_index: int,
        base_seed: int = 0,
        **kwargs,
    ) -> SelfConsistencySample:
        self.sample_calls.append(sample_index)
        return SelfConsistencySample(
            sample_index=sample_index,
            seed=base_seed + sample_index,
            termination_status="completed",
            agent_output=_output(self.agent_id),
        )


def _checkpoint(path: Path, cell: ExperimentCell) -> QIDCheckpoint:
    return QIDCheckpoint(
        path,
        cell,
        _question(),
        code_version="deadbeef+source.0123456789abcdef",
        serving_profile=_profile(),
        benchmark_contract_sha256="d" * 64,
    )


def _protocol_censor(*, seed: int) -> GenerationProtocolCensorError:
    return GenerationProtocolCensorError(
        protocol_violation_codes=["unexpected_reasoning_delimiter"],
        finish_reason="stop",
        requested_output_tokens=4096,
        completion_tokens=2,
        output_capacity_floor_tokens=4096,
        prompt_tokens=2,
        prompt_token_ids=[10, 11],
        completion_token_ids=[QWEN_THINK_END_TOKEN_ID, QWEN_IM_END_TOKEN_ID],
        decoded_completion="</think>",
        server_content="</think>",
        server_reasoning=None,
        seed=seed,
        serving_profile=_profile().name,
        effective_context_limit=_profile().max_model_len,
        context_reserve_tokens=128,
        completion_think_end_positions=[0],
        endpoint_generation="node-a:8000:111",
        created_at=1234.5,
    )


def _resign_checkpoint(payload: dict) -> None:
    integrity_payload = {
        key: value for key, value in payload.items() if key != "integrity_sha256"
    }
    canonical = json.dumps(
        integrity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()


def test_graceful_drain_finishes_and_journals_inflight_coordinate_then_stops(
    tmp_path: Path,
) -> None:
    """USR1-style admission closure never discards or redraws an admitted response."""

    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)
    request_started = threading.Event()
    release_response = threading.Event()
    drain_requested = threading.Event()

    class BlockingAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, **kwargs):
            self.answer_calls += 1
            request_started.set()
            assert release_response.wait(timeout=5.0)
            return _output(self.agent_id, round_idx=round_idx)

    def admit_coordinate() -> None:
        if drain_requested.is_set():
            raise CoordinateAdmissionClosed("test drain")

    producer = BlockingAgent("agent0")
    guarded = CheckpointingAgent(
        producer,
        checkpoint,
        admit_coordinate=admit_coordinate,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        inflight = executor.submit(
            guarded.answer,
            _question(),
            seed=cell.seed,
        )
        assert request_started.wait(timeout=5.0)
        drain_requested.set()
        release_response.set()
        observed = inflight.result(timeout=5.0)

    assert observed.answer_choice == "A"
    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    assert list(payload["coordinates"]) == ["topology:agent0:0"]
    assert payload["coordinates"]["topology:agent0:0"]["outcome"][
        "agent_output"
    ]["answer_choice"] == "A"

    # Replay is local and remains legal after drain, but a genuinely missing auxiliary
    # coordinate cannot contact the model.
    assert guarded.answer(_question(), seed=cell.seed).answer_choice == "A"
    with pytest.raises(CoordinateAdmissionClosed, match="test drain"):
        guarded.sample_one(
            _question(),
            sample_index=0,
            base_seed=cell.seed + 1000,
        )
    assert producer.answer_calls == 1
    assert producer.sample_calls == []


def test_kill_resume_replays_primary_terminal_and_auxiliary_coordinates(
    tmp_path: Path,
) -> None:
    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)
    first = _FakeAgent("agent0")
    guarded = CheckpointingAgent(first, checkpoint)

    primary = guarded.answer(_question(), seed=cell.seed)
    # The durable coordinate already contains the exact empty peer-context audit.
    assert primary.peer_context_tokens == 0
    terminal = TopologyResult(
        final_answer="A",
        per_agent=[primary],
        n_turns=1,
        n_messages=0,
        n_rounds=1,
        n_agents=1,
        system_conf={"final_producer_logprob": 0.8},
    )
    checkpoint.record_topology_result(terminal, wall_ms=321.25)
    sample0 = guarded.sample_one(
        _question(), sample_index=0, base_seed=cell.seed + 1000
    )
    assert first.answer_calls == 1
    assert first.sample_calls == [0]

    # Simulate preemption: discard every in-memory object and reopen from disk with an
    # agent that would fail loudly if an observed coordinate were reissued.
    resumed = _checkpoint(tmp_path, cell)
    replacement = _FakeAgent("agent0", fail_answer=True)

    loaded_terminal = resumed.topology_terminal()
    assert loaded_terminal is not None
    assert loaded_terminal.wall_ms == pytest.approx(321.25)
    assert loaded_terminal.topology_result is not None
    assert asdict(loaded_terminal.topology_result) == asdict(terminal)

    replayed_primary = CheckpointingAgent(replacement, resumed).answer(
        _question(), seed=cell.seed
    )
    assert replayed_primary.raw_text == primary.raw_text
    assert replacement.answer_calls == 0

    replayed0 = CheckpointingAgent(replacement, resumed).sample_one(
        _question(), sample_index=0, base_seed=cell.seed + 1000
    )
    assert replayed0.to_dict() == sample0.to_dict()
    assert replacement.sample_calls == []

    # A genuinely missing coordinate is the only one allowed to reach the replacement
    # endpoint.
    sample1 = CheckpointingAgent(replacement, resumed).sample_one(
        _question(), sample_index=1, base_seed=cell.seed + 1000
    )
    assert sample1.sample_index == 1
    assert replacement.sample_calls == [1]

    result_path = tmp_path / "results.jsonl"
    io.append_jsonl(result_path, {"qid": _question().qid, "complete": True})
    resumed.delete()
    assert not resumed.path.exists()
    assert json.loads(result_path.read_text().strip())["qid"] == _question().qid


def test_concurrent_sibling_failure_preserves_success_for_retry(tmp_path: Path) -> None:
    cell = _cell(topology=Topology.INDEPENDENT, n_agents=2)
    checkpoint = _checkpoint(tmp_path, cell)
    first0 = _FakeAgent("agent0")
    first1 = _FakeAgent("agent1", fail_answer=True)
    topology = Independent(
        [
            CheckpointingAgent(first0, checkpoint),
            CheckpointingAgent(first1, checkpoint),
        ],
        context_level=cell.context_share_level,
        rounds=1,
        seed=cell.seed,
    )
    with pytest.raises(ConnectionError, match="agent1 failed"):
        topology.run(_question())
    assert first0.answer_calls == 1
    assert first1.answer_calls == 1

    on_disk = json.loads(checkpoint.path.read_text())
    assert len(on_disk["coordinates"]) == 1

    resumed = _checkpoint(tmp_path, cell)
    replacement0 = _FakeAgent("agent0", fail_answer=True)
    replacement1 = _FakeAgent("agent1")
    retried = Independent(
        [
            CheckpointingAgent(replacement0, resumed),
            CheckpointingAgent(replacement1, resumed),
        ],
        context_level=cell.context_share_level,
        rounds=1,
        seed=cell.seed,
    ).run(_question())

    assert replacement0.answer_calls == 0
    assert replacement1.answer_calls == 1
    assert [output.agent_id for output in retried.per_agent] == ["agent0", "agent1"]
    resumed.record_topology_result(retried, wall_ms=12.5)
    reopened = _checkpoint(tmp_path, cell).topology_terminal()
    assert reopened is not None
    assert reopened.topology_result is not None
    assert asdict(reopened.topology_result) == asdict(retried)


def test_duplicate_threaded_coordinate_executes_underlying_agent_once(
    tmp_path: Path,
) -> None:
    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)
    entered = threading.Event()
    release = threading.Event()

    class SlowAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, **kwargs):
            self.answer_calls += 1
            entered.set()
            assert release.wait(timeout=5)
            return _output(self.agent_id, round_idx=round_idx)

    agent = SlowAgent("agent0")
    guarded = CheckpointingAgent(agent, checkpoint)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(guarded.answer, _question(), seed=cell.seed)
        assert entered.wait(timeout=5)
        second = executor.submit(guarded.answer, _question(), seed=cell.seed)
        release.set()
        outputs = [first.result(timeout=5), second.result(timeout=5)]

    assert agent.answer_calls == 1
    assert outputs[0].to_dict() == outputs[1].to_dict()
    assert outputs[0] is not outputs[1]


def test_completed_coordinate_rejects_changed_seed_or_question_identity(
    tmp_path: Path,
) -> None:
    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)
    agent = _FakeAgent("agent0")
    guarded = CheckpointingAgent(agent, checkpoint)
    guarded.answer(_question(), seed=cell.seed)

    with pytest.raises(CheckpointIdentityError, match="different request identity"):
        guarded.answer(_question(), seed=cell.seed + 1)
    changed_question = Question(
        qid=_question().qid,
        benchmark=_question().benchmark,
        prompt_stem="A changed prompt under the same QID",
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=["yes", "no"],
    )
    with pytest.raises(CheckpointIdentityError, match="question does not match"):
        guarded.answer(changed_question, seed=cell.seed)
    assert agent.answer_calls == 1


def test_primary_and_auxiliary_censors_replay_without_resampling(
    tmp_path: Path,
) -> None:
    cell = _cell()
    question = _question()

    def censor(*, seed: int) -> GenerationTruncationError:
        return GenerationTruncationError(
            finish_reason="length",
            requested_output_tokens=4096,
            completion_tokens=4096,
            output_capacity_floor_tokens=4096,
            prompt_tokens=2,
            prompt_token_ids=[10, 11],
            completion_token_ids=[12] * 4096,
            decoded_completion="partial",
            server_content="partial",
            server_reasoning="",
            seed=seed,
            serving_profile=_profile().name,
            effective_context_limit=_profile().max_model_len,
            context_reserve_tokens=128,
            endpoint_generation="node-a:8000:111",
            created_at=1234.5,
        )

    primary_dir = tmp_path / "primary"
    primary_checkpoint = _checkpoint(primary_dir, cell)

    class PrimaryCensorAgent(_FakeAgent):
        def answer(
            self, question: Question, *, round_idx: int = 0, seed=None, **kwargs
        ):
            self.answer_calls += 1
            raise censor(seed=seed)

    primary_agent = PrimaryCensorAgent("agent0")
    with pytest.raises(GenerationTruncationError) as first_censor:
        CheckpointingAgent(primary_agent, primary_checkpoint).answer(
            question, seed=cell.seed
        )
    assert primary_agent.answer_calls == 1

    reopened_primary = _checkpoint(primary_dir, cell)
    replacement = _FakeAgent("agent0", fail_answer=True)
    with pytest.raises(GenerationTruncationError) as replayed_censor:
        CheckpointingAgent(replacement, reopened_primary).answer(
            question, seed=cell.seed
        )
    assert replacement.answer_calls == 0
    assert replayed_censor.value.to_censored_generation() == (
        first_censor.value.to_censored_generation()
    )

    auxiliary_dir = tmp_path / "auxiliary"
    auxiliary_checkpoint = _checkpoint(auxiliary_dir, cell)

    class AuxiliaryCensorAgent(_FakeAgent):
        def sample_one(
            self,
            question: Question,
            *,
            sample_index: int,
            base_seed: int = 0,
            **kwargs,
        ) -> SelfConsistencySample:
            self.sample_calls.append(sample_index)
            seed = base_seed + sample_index
            error = censor(seed=seed).enrich(
                qid=question.qid,
                agent_id=self.agent_id,
                round_idx=0,
                generation_role="self_consistency",
                sample_index=sample_index,
            )
            return SelfConsistencySample(
                sample_index=sample_index,
                seed=seed,
                termination_status="length_censored",
                censored_generation=error.to_censored_generation(),
            )

    auxiliary_agent = AuxiliaryCensorAgent("agent0")
    first_sample = CheckpointingAgent(auxiliary_agent, auxiliary_checkpoint).sample_one(
        question, sample_index=2, base_seed=cell.seed + 1000
    )
    assert auxiliary_agent.sample_calls == [2]
    reopened_auxiliary = _checkpoint(auxiliary_dir, cell)
    auxiliary_replacement = _FakeAgent("agent0")
    replayed_sample = CheckpointingAgent(
        auxiliary_replacement, reopened_auxiliary
    ).sample_one(question, sample_index=2, base_seed=cell.seed + 1000)
    assert auxiliary_replacement.sample_calls == []
    assert replayed_sample.to_dict() == first_sample.to_dict()


def test_protocol_censor_is_journaled_and_replayed_without_second_draw(
    tmp_path: Path,
) -> None:
    cell = _cell()
    question = _question()
    checkpoint = _checkpoint(tmp_path, cell)
    protocol_censor = _protocol_censor(seed=cell.seed)

    class ProtocolCensorAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, seed=None, **kwargs):
            self.answer_calls += 1
            raise protocol_censor

    first = ProtocolCensorAgent("agent0")
    with pytest.raises(GenerationProtocolCensorError) as observed:
        CheckpointingAgent(first, checkpoint).answer(question, seed=cell.seed)
    assert first.answer_calls == 1
    assert observed.value.protocol_violation_codes == (
        "unexpected_reasoning_delimiter",
    )

    reopened = _checkpoint(tmp_path, cell)

    class DivergentSecondDrawAgent(_FakeAgent):
        """Would return a different same-seed outcome if replay were violated."""

        def answer(self, question: Question, *, round_idx: int = 0, seed=None, **kwargs):
            self.answer_calls += 1
            return _output(self.agent_id, round_idx=round_idx, answer="B")

    replacement = DivergentSecondDrawAgent("agent0")
    with pytest.raises(GenerationProtocolCensorError) as replayed:
        CheckpointingAgent(replacement, reopened).answer(question, seed=cell.seed)
    assert replacement.answer_calls == 0
    assert replayed.value.to_censored_generation() == (
        observed.value.to_censored_generation()
    )
    snapshot = reopened.observed_topology_coordinates()
    assert len(snapshot) == 1
    assert snapshot[0]["outcome"]["termination_status"] == "protocol_censored"


def test_corrupt_and_wrong_identity_checkpoints_fail_closed(tmp_path: Path) -> None:
    cell = _cell()
    corrupt_dir = tmp_path / "corrupt"
    checkpoint = _checkpoint(corrupt_dir, cell)
    CheckpointingAgent(_FakeAgent("agent0"), checkpoint).answer(
        _question(), seed=cell.seed
    )

    payload = json.loads(checkpoint.path.read_text())
    coordinate = next(iter(payload["coordinates"].values()))
    coordinate["outcome"]["agent_output"]["raw_text"] = "tampered"
    io.write_json(
        checkpoint.path, payload
    )  # Deliberately retain the old integrity hash.
    with pytest.raises(CheckpointCorruptionError, match="integrity"):
        _checkpoint(corrupt_dir, cell)

    identity_dir = tmp_path / "identity"
    identity_checkpoint = _checkpoint(identity_dir, cell)
    CheckpointingAgent(_FakeAgent("agent0"), identity_checkpoint).answer(
        _question(), seed=cell.seed
    )
    with pytest.raises(CheckpointIdentityError, match="does not match"):
        QIDCheckpoint(
            identity_dir,
            cell,
            _question(),
            code_version="different-code-version",
            serving_profile=_profile(),
            benchmark_contract_sha256="d" * 64,
        )
    with pytest.raises(CheckpointIdentityError, match="does not match"):
        QIDCheckpoint(
            identity_dir,
            cell,
            _question(),
            code_version="deadbeef+source.0123456789abcdef",
            serving_profile=_profile(),
            benchmark_contract_sha256="e" * 64,
        )


def test_nonempty_peer_audit_is_persisted_before_coordinate_publication(
    tmp_path: Path,
) -> None:
    cell = _cell(topology=Topology.CENTRALIZED, n_agents=2)
    checkpoint = _checkpoint(tmp_path, cell)
    agent = _FakeAgent("agent0")
    text = "one exact peer block"
    rendered = PeerContextRender(
        text=text,
        token_count=4,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        block_token_counts=(4,),
        truncation_marker_count=0,
    )
    guarded = CheckpointingAgent(agent, checkpoint)
    output = guarded.answer_with_peer_context_audit(
        _question(),
        round_idx=0,
        peer_context_render=rendered,
        max_tokens=4096,
        seed=cell.seed + 999,
    )
    assert output.peer_context_tokens == 4
    coordinate = checkpoint.observed_topology_coordinates()[0]
    durable = coordinate["outcome"]["agent_output"]
    assert durable["peer_context_sha256"] == rendered.sha256
    assert durable["peer_context_block_token_counts"] == [4]

    replacement = _FakeAgent("agent0", fail_answer=True)
    replayed = CheckpointingAgent(
        replacement,
        _checkpoint(tmp_path, cell),
    ).answer_with_peer_context_audit(
        _question(),
        round_idx=0,
        peer_context_render=rendered,
        max_tokens=4096,
        seed=cell.seed + 999,
    )
    assert replacement.answer_calls == 0
    assert replayed.peer_context_block_token_counts == [4]


def test_nonempty_peer_context_without_render_audit_is_rejected_before_draw(
    tmp_path: Path,
) -> None:
    cell = _cell(topology=Topology.CENTRALIZED, n_agents=2)
    agent = _FakeAgent("agent0")
    guarded = CheckpointingAgent(agent, _checkpoint(tmp_path, cell))
    with pytest.raises(CheckpointIdentityError, match="requires its exact render audit"):
        guarded.answer(
            _question(),
            round_idx=0,
            peer_context="unbound peer text",
            seed=cell.seed + 999,
        )
    assert agent.answer_calls == 0


def test_post_observation_json_failure_latches_and_forbids_redraw(
    tmp_path: Path,
) -> None:
    cell = _cell()

    class NonFiniteAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, **kwargs):
            self.answer_calls += 1
            output = _output(self.agent_id, round_idx=round_idx)
            output.option_logprobs = {"A": float("nan")}
            return output

    agent = NonFiniteAgent("agent0")
    guarded = CheckpointingAgent(agent, _checkpoint(tmp_path, cell))
    with pytest.raises(CheckpointCorruptionError, match="finite canonical JSON"):
        guarded.answer(_question(), seed=cell.seed)
    with pytest.raises(CheckpointCorruptionError, match="finite canonical JSON"):
        guarded.answer(_question(), seed=cell.seed)
    assert agent.answer_calls == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda censor: censor.__setitem__(
                "protocol_violation_codes", ["unknown_terminal_under_v4"]
            ),
            "violation codes",
        ),
        (
            lambda censor: censor.__setitem__("actual_terminal_token_id", 123),
            "terminal token",
        ),
        (
            lambda censor: censor.__setitem__("requested_output_tokens", 4095),
            "treatment floor",
        ),
        (
            lambda censor: censor.__setitem__(
                "completion_think_end_positions", []
            ),
            "does not match retained token IDs",
        ),
    ],
)
def test_protocol_censor_recomputation_rejects_resigned_tampering(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)

    class CensorAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, seed=None, **kwargs):
            self.answer_calls += 1
            raise _protocol_censor(seed=seed)

    with pytest.raises(GenerationProtocolCensorError):
        CheckpointingAgent(CensorAgent("agent0"), checkpoint).answer(
            _question(), seed=cell.seed
        )
    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    censor = next(iter(payload["coordinates"].values()))["outcome"][
        "censored_generation"
    ]
    mutation(censor)
    _resign_checkpoint(payload)
    io.write_json(checkpoint.path, payload)
    with pytest.raises(CheckpointCorruptionError, match=message):
        _checkpoint(tmp_path, cell)


def test_length_censor_semantics_reject_resigned_nonexhaustion(
    tmp_path: Path,
) -> None:
    cell = _cell()

    class LengthCensorAgent(_FakeAgent):
        def answer(self, question: Question, *, round_idx: int = 0, seed=None, **kwargs):
            self.answer_calls += 1
            raise GenerationTruncationError(
                finish_reason="length",
                requested_output_tokens=4096,
                completion_tokens=4096,
                output_capacity_floor_tokens=4096,
                prompt_tokens=2,
                prompt_token_ids=[10, 11],
                completion_token_ids=[12] * 4096,
                decoded_completion="partial",
                server_content="partial",
                server_reasoning=None,
                seed=seed,
                serving_profile=_profile().name,
                effective_context_limit=_profile().max_model_len,
                context_reserve_tokens=128,
                endpoint_generation="node-a:8000:111",
                created_at=1234.5,
            )

    checkpoint = _checkpoint(tmp_path, cell)
    with pytest.raises(GenerationTruncationError):
        CheckpointingAgent(LengthCensorAgent("agent0"), checkpoint).answer(
            _question(), seed=cell.seed
        )
    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    censor = next(iter(payload["coordinates"].values()))["outcome"][
        "censored_generation"
    ]
    censor["finish_reason"] = "stop"
    _resign_checkpoint(payload)
    io.write_json(checkpoint.path, payload)
    with pytest.raises(CheckpointCorruptionError, match="finish_reason must be length"):
        _checkpoint(tmp_path, cell)


def test_request_schema_and_key_are_rejected_before_producer(
    tmp_path: Path,
) -> None:
    cell = _cell()
    checkpoint = _checkpoint(tmp_path, cell)
    calls = 0

    def producer() -> dict:
        nonlocal calls
        calls += 1
        return {
            "termination_status": "completed",
            "agent_output": asdict(_output("agent0")),
            "censored_generation": None,
        }

    request = {
        "generation_role": "topology",
        "qid": _question().qid,
        "agent_id": "agent0",
        "round": 0,
        "seed": cell.seed,
        "sample_index": None,
        "peer_context": {"sha256": hashlib.sha256(b"").hexdigest(), "utf8_bytes": 0},
        "max_tokens": 4096,
        "elicit_cot": True,
    }
    with pytest.raises(CheckpointCorruptionError, match="key"):
        checkpoint.execute_coordinate("topology:agent1:0", request, producer)
    changed = dict(request)
    changed["elicit_cot"] = False
    with pytest.raises(CheckpointCorruptionError, match="elicit_cot"):
        checkpoint.execute_coordinate("topology:agent0:0", changed, producer)
    changed = dict(request)
    changed["unregistered"] = True
    with pytest.raises(CheckpointCorruptionError, match="wrong fields"):
        checkpoint.execute_coordinate("topology:agent0:0", changed, producer)
    assert calls == 0


def test_terminal_censor_binds_to_map_order_coordinate_and_durable_timing(
    tmp_path: Path,
) -> None:
    cell = _cell(topology=Topology.INDEPENDENT, n_agents=2)

    def run_two(checkpoint: QIDCheckpoint):
        barrier = threading.Barrier(2)

        class ConcurrentCensorAgent(_FakeAgent):
            def answer(
                self,
                question: Question,
                *,
                round_idx: int = 0,
                seed=None,
                **kwargs,
            ):
                self.answer_calls += 1
                barrier.wait(timeout=5)
                raise _protocol_censor(seed=seed)

        topology = Independent(
            [
                CheckpointingAgent(ConcurrentCensorAgent("agent0"), checkpoint),
                CheckpointingAgent(ConcurrentCensorAgent("agent1"), checkpoint),
            ],
            context_level=cell.context_share_level,
            rounds=1,
            seed=cell.seed,
        )
        with pytest.raises(GenerationProtocolCensorError) as caught:
            topology.run(_question())
        assert caught.value.agent_id == "agent0"
        assert len(checkpoint.observed_topology_coordinates()) == 2
        return caught.value

    valid = _checkpoint(tmp_path / "valid", cell)
    propagated = run_two(valid)
    valid.record_topology_censor(propagated, wall_ms=10.0)
    assert valid.topology_terminal() is not None
    assert valid.topology_producer_wall_ms() > 0
    assert 0 < valid.topology_critical_path_wall_ms() <= valid.topology_producer_wall_ms()

    invalid = _checkpoint(tmp_path / "invalid", cell)
    run_two(invalid)
    agent1 = next(
        item
        for item in invalid.observed_topology_coordinates()
        if item["request"]["agent_id"] == "agent1"
    )
    request = agent1["request"]
    nonpropagated = invalid._decode_censor(
        agent1["outcome"]["censored_generation"],
        qid=request["qid"],
        agent_id=request["agent_id"],
        round_idx=request["round"],
        generation_role="topology",
        sample_index=None,
        seed=request["seed"],
    )
    with pytest.raises(CheckpointCorruptionError, match="map-order censor"):
        invalid.record_topology_censor(nonpropagated, wall_ms=10.0)
