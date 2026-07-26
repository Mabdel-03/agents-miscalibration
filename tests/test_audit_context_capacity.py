"""Read-only routed long-context capacity audit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from agents_scaling.agents.base_agent import (
    render_agent_user_prompt,
    requested_generation_tokens,
)
from agents_scaling.agents.message_builder import PEER_COT_CHAR_LIMIT, build_peer_context
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.benchmarks.contracts import freeze_benchmark_contracts
from agents_scaling.config import ExperimentCell, ReasoningLevel
from agents_scaling.experiment.manifest import ManifestSnapshot, freeze_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_context_capacity as audit  # noqa: E402
from agents_scaling.serving import context as serving_context  # noqa: E402


class RecordingTokenizer:
    def __init__(self, fixed_count: int | None = None):
        self.fixed_count = fixed_count
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        count = self.fixed_count
        if count is None:
            # Deterministic tokenizer surrogate that preserves prompt-length ordering.
            count = 8 + sum((len(message["content"]) + 3) // 4 for message in conversation)
        return list(range(count))

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        return "".join(chr(token_id) for token_id in token_ids)


def _cell(
    *,
    seed: int = 0,
    n_agents: int = 7,
    reasoning: str = "unlimited",
    prompt: int = 3,
    topology: str = "decentralized",
    context: str = "plus_cot",
    model: str = "32B",
    rounds: int = 2,
) -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": model,
            "context_share_level": context,
            "prompt_complexity_level": prompt,
            "reasoning_level": reasoning,
            "topology": topology,
            "benchmark": "gpqa",
            "n_agents": n_agents,
            "rounds": rounds,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 2,
            "seed": seed,
        }
    )


def _question(qid: str, stem: str) -> Question:
    return Question(
        qid=qid,
        benchmark="gpqa",
        prompt_stem=stem,
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=["first", "second", "third", "fourth"],
    )


def _snapshot(tmp_path: Path, cells: list[ExperimentCell]) -> ManifestSnapshot:
    return ManifestSnapshot(tmp_path / "cells.json", tuple(cells), "manifest-sha")


def test_six_peer_fixture_hits_registered_full_block_cap():
    peers = audit.synthetic_peer_outputs(6)
    assert len(peers) == 6
    assert all(len(peer.cot_text) == PEER_COT_CHAR_LIMIT for peer in peers)
    assert all(peer.intermediate_results for peer in peers)
    assert all(peer.raw_text and peer.answer_choice for peer in peers)
    context = build_peer_context(
        peers,
        _cell().context_share_level,
        tokenizer=RecordingTokenizer(),
    )
    assert context.count("  Intermediate result: ") == 6
    assert context.count("Agent agent") == 6
    assert context.count("PEER_CONTEXT_TRUNCATED") == 6


def test_dense_plus_intermediate_routes_small_model_long_and_hits_boundary(tmp_path):
    cell = _cell(
        model="14B",
        context="plus_intermediate",
        reasoning="b8192",
    )
    report = audit.audit_snapshot(
        _snapshot(tmp_path, [cell]),
        run_id="run",
        all_routed_profiles=True,
        benchmark_loader=lambda *_args, **_kwargs: [_question("q", "stem")],
        tokenizer_loader=lambda _profile: RecordingTokenizer(),
    )
    [cell_report] = report["cells"]
    assert cell_report["serving_profile"] == "14B-long"
    assert cell_report["peer_block_token_counts"] == [4000] * 6
    assert cell_report["peer_truncation_marker_count"] == 6
    assert report["summary"]["passed"]


def test_selection_requires_long_profile_and_applies_all_filters(tmp_path):
    target = _cell()
    cells = [
        target,
        _cell(n_agents=6),
        _cell(reasoning="b8192"),
        _cell(prompt=0),
        _cell(topology="centralized"),
        _cell(topology="independent"),  # never routes to long profile
        _cell(model="14B"),
    ]
    filters = audit.AuditFilters(
        n_agents=frozenset({7}),
        reasoning=frozenset({"unlimited"}),
        prompt_levels=frozenset({3}),
        topologies=frozenset({"decentralized"}),
        context_levels=frozenset({"plus_cot"}),
    )
    assert audit.selected_long_cells(_snapshot(tmp_path, cells), filters) == [target]


def test_all_routed_profiles_selects_standard_profile_and_context_filter(tmp_path):
    target = _cell(model="14B")
    cells = [
        target,
        _cell(model="14B", context="artifact_only"),
        _cell(model="14B", reasoning="b8192"),
    ]
    filters = audit.AuditFilters(
        reasoning=frozenset({"unlimited"}),
        context_levels=frozenset({"plus_cot"}),
    )
    snapshot = _snapshot(tmp_path, cells)
    assert audit.selected_cells(snapshot, filters) == []
    assert audit.selected_cells(
        snapshot, filters, all_routed_profiles=True
    ) == [target]


def test_exact_runtime_render_and_chat_template_path_reports_worst_question(tmp_path):
    cell = _cell()
    tokenizer = RecordingTokenizer()
    loader_calls = []

    def loader(name, n, seed):
        loader_calls.append((name, n, seed))
        return [_question("short", "short stem"), _question("long", "long " * 500)]

    report = audit.audit_snapshot(
        _snapshot(tmp_path, [cell]),
        run_id="run",
        benchmark_loader=loader,
        tokenizer_loader=lambda profile: tokenizer,
    )
    summary = report["summary"]
    assert summary["passed"]
    assert summary["audited_requests"] == 2
    assert summary["maximum_prompt_request"]["qid"] == "long"
    worst = summary["maximum_required_request"]
    assert worst["output_capacity_floor_tokens"] == 12288
    assert worst["requested_output_tokens"] == 40960 - worst["prompt_tokens"] - 128
    assert worst["required_context_tokens"] == 40960
    assert worst["floor_required_context_tokens"] == worst["prompt_tokens"] + 12288 + 128
    assert worst["output_capacity_headroom_tokens"] == (
        worst["requested_output_tokens"] - 12288
    )
    assert worst["full_envelope_context_headroom_tokens"] == 0
    assert summary["maximum_output_capacity_floor_tokens"] == 12288
    assert summary["minimum_requested_output_tokens"] == worst["requested_output_tokens"]
    assert summary["maximum_requested_output_tokens"] > worst["requested_output_tokens"]
    assert summary["minimum_output_capacity_headroom_tokens"] == worst[
        "output_capacity_headroom_tokens"
    ]
    assert summary["context_reserve_tokens"] == 128
    assert worst["context_reserve_tokens"] == 128
    assert (
        worst["prompt_plus_reserve_tokens"]
        == worst["prompt_tokens"] + 128
    )
    assert worst["effective_context_limit"] == 40960
    assert loader_calls == [("gpqa", 2, 0)]

    messages, kwargs = tokenizer.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Confidence: X%" in messages[1]["content"]
    assert messages[1]["content"].count("PEER_CONTEXT_TRUNCATED") == 6
    assert messages[1]["content"].count("  Intermediate result: ") == 6
    assert "Think step by step" not in messages[1]["content"]  # native thinking is on
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "return_dict": False,
    }
    assumptions = report["assumptions"]["peer_context"]
    assert assumptions["cot_character_limit_per_peer"] == 4000
    assert assumptions["rendered_block_token_limit_per_peer"] == 4000
    assert assumptions["cot_fixture_characters"] == 4000
    assert assumptions["intermediate_fixture_characters"] == 4000
    assert assumptions["intermediate_fixture_sha256"]
    assert assumptions["raw_final_text"]
    [cell_report] = report["cells"]
    assert report["schema_version"] == 3
    assert cell_report["output_capacity_floor_tokens"] == 12288
    assert cell_report["minimum_requested_output_tokens"] == worst[
        "requested_output_tokens"
    ]
    assert cell_report["minimum_output_capacity_headroom_tokens"] == worst[
        "output_capacity_headroom_tokens"
    ]
    assert cell_report["peer_block_token_counts"] == [4000] * 6
    assert cell_report["peer_truncation_marker_count"] == 6
    assert cell_report["peer_context_tokens"] >= 6 * 4000


def test_output_capacity_floor_formula_matches_agent_runtime_and_off_prompt_hint():
    expected = {
        ReasoningLevel.OFF: 4096,
        ReasoningLevel.B512: 4608,
        ReasoningLevel.B2048: 6144,
        ReasoningLevel.B8192: 12288,
        ReasoningLevel.UNLIMITED: 12288,
    }
    assert {
        level: requested_generation_tokens(level, 4096) for level in ReasoningLevel
    } == expected
    prompt = render_agent_user_prompt(_question("q", "stem"), ReasoningLevel.OFF)
    assert prompt.startswith("Think step by step. Show your reasoning")


def test_dynamic_envelope_admits_exact_floor_boundary_and_rejects_one_token_less(
    tmp_path,
):
    # 32B-long serves 40,960 tokens.  With the 128-token reserve and the 12,288-token
    # unlimited floor, a 28,544-token prompt is the exact admission boundary.
    def run(prompt_tokens: int):
        return audit.audit_snapshot(
            _snapshot(tmp_path, [_cell()]),
            run_id="run",
            benchmark_loader=lambda *_args, **_kwargs: [_question("q", "stem")],
            tokenizer_loader=lambda _profile: RecordingTokenizer(
                fixed_count=prompt_tokens
            ),
        )

    admitted = run(28544)
    admitted_request = admitted["summary"]["minimum_headroom_request"]
    assert admitted["summary"]["passed"]
    assert admitted_request["output_capacity_floor_tokens"] == 12288
    assert admitted_request["requested_output_tokens"] == 12288
    assert admitted_request["output_capacity_headroom_tokens"] == 0
    assert admitted_request["required_context_tokens"] == 40960
    assert admitted_request["full_envelope_context_headroom_tokens"] == 0

    rejected = run(28545)
    rejected_request = rejected["failure_examples"][0]
    assert not rejected["summary"]["passed"]
    assert rejected_request["output_capacity_floor_tokens"] == 12288
    assert rejected_request["requested_output_tokens"] == 12287
    assert rejected_request["output_capacity_headroom_tokens"] == -1
    assert rejected_request["floor_required_context_tokens"] == 40961
    # The counterfactual full envelope still reaches the exact boundary, but it is
    # never submitted because it cannot preserve the treatment floor.
    assert rejected_request["required_context_tokens"] == 40960
    assert rejected_request["context_preflight_fits"] is False


def test_loader_and_profile_tokenizer_are_cached_across_equivalent_cells(tmp_path):
    cells = [_cell(prompt=0), _cell(prompt=3)]
    loads = []
    tokenizers = []

    def loader(name, n, seed):
        loads.append((name, n, seed))
        return [_question("q", "stem")]

    def tokenizer_loader(profile):
        tokenizers.append(profile)
        return RecordingTokenizer(fixed_count=100)

    report = audit.audit_snapshot(
        _snapshot(tmp_path, cells),
        run_id="run",
        benchmark_loader=loader,
        tokenizer_loader=tokenizer_loader,
    )
    assert report["summary"]["selected_cells"] == 2
    assert loads == [("gpqa", 2, 0)]
    assert tokenizers == ["32B-long"]


def test_exact_prompt_tokens_are_cached_across_thinking_budgets(tmp_path):
    tokenizer = RecordingTokenizer(fixed_count=100)
    report = audit.audit_snapshot(
        _snapshot(
            tmp_path,
            [
                _cell(model="14B", reasoning="b512", n_agents=5),
                _cell(model="14B", reasoning="b2048", n_agents=5),
            ],
        ),
        run_id="run",
        all_routed_profiles=True,
        benchmark_loader=lambda *_args, **_kwargs: [_question("q", "stem")],
        tokenizer_loader=lambda _profile: tokenizer,
    )
    cache = report["assumptions"]["prompt_token_cache"]
    assert report["summary"]["audited_requests"] == 2
    assert cache["unique_rendered_prompts"] == 1
    assert cache["misses"] == 1
    assert cache["hits"] == 1
    assert len(tokenizer.calls) == 1


def test_failures_are_grouped_by_model_profile_and_reasoning(tmp_path):
    loaded_profiles = []

    def tokenizer_loader(profile):
        loaded_profiles.append(profile)
        return RecordingTokenizer(fixed_count=38000 if profile == "14B" else 30000)

    report = audit.audit_snapshot(
        _snapshot(
            tmp_path,
            [
                _cell(model="14B", reasoning="off"),
                _cell(model="32B", reasoning="unlimited"),
            ],
        ),
        run_id="run",
        all_routed_profiles=True,
        benchmark_loader=lambda *_args, **_kwargs: [_question("q", "stem")],
        tokenizer_loader=tokenizer_loader,
    )
    assert loaded_profiles == ["14B", "32B-long"]
    assert report["summary"]["failed_requests"] == 2
    assert len(report["groups"]) == 2
    assert len(report["failure_groups"]) == 2
    by_model = {group["model_size"]: group for group in report["failure_groups"]}
    assert by_model["14B"]["serving_profile"] == "14B"
    assert by_model["14B"]["reasoning_level"] == "off"
    assert by_model["14B"]["maximum_prompt_tokens"] == 38000
    assert by_model["14B"]["maximum_output_capacity_floor_tokens"] == 4096
    assert by_model["14B"]["minimum_requested_output_tokens"] == -5360
    assert by_model["14B"]["maximum_requested_output_tokens"] == -5360
    assert by_model["14B"]["maximum_floor_required_context_tokens"] == 42224
    assert by_model["14B"]["maximum_required_context_tokens"] == 32768
    assert by_model["14B"]["minimum_output_capacity_headroom_tokens"] == -9456
    assert by_model["32B"]["serving_profile"] == "32B-long"
    assert by_model["32B"]["reasoning_level"] == "unlimited"
    assert by_model["32B"]["maximum_output_capacity_floor_tokens"] == 12288
    assert by_model["32B"]["minimum_requested_output_tokens"] == 10832
    assert by_model["32B"]["maximum_requested_output_tokens"] == 10832
    assert by_model["32B"]["maximum_floor_required_context_tokens"] == 42416
    assert by_model["32B"]["maximum_required_context_tokens"] == 40960
    assert by_model["32B"]["minimum_output_capacity_headroom_tokens"] == -1456
    assert by_model["32B"]["minimum_headroom_request"][
        "output_capacity_floor_tokens"
    ] == 12288


def test_decentralized_one_round_has_no_peer_context_but_centralized_has_six():
    assert audit.maximum_peer_count(_cell(rounds=1)) == 0
    assert audit.maximum_peer_count(_cell(rounds=2)) == 6
    assert audit.maximum_peer_count(_cell(topology="centralized", rounds=1)) == 6


def test_audit_counts_every_failure_and_keeps_bounded_examples(tmp_path):
    report = audit.audit_snapshot(
        _snapshot(tmp_path, [_cell()]),
        run_id="run",
        benchmark_loader=lambda *_args, **_kwargs: [
            _question("a", "a"),
            _question("b", "b"),
        ],
        tokenizer_loader=lambda _profile: RecordingTokenizer(fixed_count=30000),
        failure_example_limit=1,
    )
    assert not report["summary"]["passed"]
    assert report["summary"]["failed_requests"] == 2
    assert report["summary"]["failed_cells"] == 1
    assert len(report["failure_examples"]) == 1
    assert report["summary"]["maximum_floor_required_context_tokens"] == 42416
    assert report["summary"]["maximum_required_context_tokens"] == 40960
    assert report["summary"]["minimum_requested_output_tokens"] == 10832
    assert report["summary"]["minimum_output_capacity_headroom_tokens"] == -1456


def test_cli_requires_frozen_manifest_and_returns_one_for_capacity_failure(
    tmp_path, monkeypatch, capsys
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "cells.json").write_text(json.dumps([_cell().to_dict()]))
    assert audit.main(["--run-id", "run", "--results-root", str(tmp_path)]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert "checksum is missing" in missing["error"]["message"]

    freeze_manifest(run_root)
    monkeypatch.setattr(
        audit,
        "load_benchmark",
        lambda *_args, **_kwargs: [
            _question("q1", "stem one"),
            _question("q2", "stem two"),
        ],
    )
    freeze_benchmark_contracts(
        run_root,
        benchmark_loader=lambda *_args, **_kwargs: [
            _question("q1", "stem one"),
            _question("q2", "stem two"),
        ],
    )
    monkeypatch.setattr(
        audit,
        "load_benchmark",
        lambda *_args, **_kwargs: [
            _question("q1", "stem one"),
            _question("q2", "stem two"),
        ],
    )
    monkeypatch.setattr(
        audit,
        "tokenizer_for_profile",
        lambda _profile: RecordingTokenizer(fixed_count=32700),
    )
    before = {path.relative_to(run_root) for path in run_root.rglob("*")}
    assert audit.main(["--run-id", "run", "--results-root", str(tmp_path)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert not failed["summary"]["passed"]
    assert failed["summary"]["failed_requests"] == 2
    assert {path.relative_to(run_root) for path in run_root.rglob("*")} == before
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"


def test_production_audit_rejects_any_benchmark_loader_fallback(tmp_path):
    cell = _cell()

    class FallbackLoader:
        diagnostics = [
            {
                "benchmark": "gpqa",
                "reason": "cache-metadata-fallback",
            }
        ]

        def __call__(self, *_args, **_kwargs):
            return [_question("q", "stem")]

    with pytest.raises(audit.AuditError, match="fallback is forbidden"):
        audit.audit_snapshot(
            _snapshot(tmp_path, [cell]),
            run_id="run",
            benchmark_loader=FallbackLoader(),
            tokenizer_loader=lambda _profile: RecordingTokenizer(),
        )


def test_no_matching_long_cells_is_an_audit_error(tmp_path):
    snapshot = _snapshot(tmp_path, [_cell(topology="independent")])
    with pytest.raises(audit.AuditError, match="no manifest cells"):
        audit.audit_snapshot(
            snapshot,
            run_id="run",
            benchmark_loader=lambda *_args, **_kwargs: [],
            tokenizer_loader=lambda _profile: RecordingTokenizer(),
        )


def test_cached_tokenizer_falls_back_to_slow_backend_on_schema_mismatch(monkeypatch):
    import transformers

    sentinel = RecordingTokenizer()
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(hf_id, **kwargs):
            calls.append((hf_id, kwargs))
            if kwargs.get("use_fast") is False:
                return sentinel
            raise Exception("newer tokenizer.json schema")

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    serving_context.tokenizer_for_profile.cache_clear()
    try:
        assert serving_context.tokenizer_for_profile("32B-long") is sentinel
    finally:
        serving_context.tokenizer_for_profile.cache_clear()
    revision = "9216db5781bf21249d130ec9da846c4624c16137"
    assert calls == [
        (
            "Qwen/Qwen3-32B",
            {"revision": revision, "local_files_only": True},
        ),
        (
            "Qwen/Qwen3-32B",
            {
                "revision": revision,
                "local_files_only": True,
                "use_fast": False,
            },
        ),
    ]


def test_cache_metadata_fallback_retains_mmlu_qid_and_seeded_option_contract(
    monkeypatch, tmp_path
):
    arrow = tmp_path / "mmlu-pro-test.arrow"

    def core_loader(name, *, n, seed, fallback_diagnostics):
        assert (name, n, seed) == ("mmlu_pro", 1, 7)
        fallback_diagnostics.append(
            {
                "benchmark": name,
                "seed": seed,
                "n": n,
                "cached_arrow_path": str(arrow),
            }
        )
        return [
            Question(
                qid="mmlupro-0",
                benchmark="mmlu_pro",
                prompt_stem="Which option?",
                options=["three", "one", "zero", "two"],
                answer_key="D",
                answer_type=AnswerType.MCQ,
            )
        ]

    monkeypatch.setattr(audit, "load_benchmark", core_loader)
    loader = audit.CacheOnlyBenchmarkLoader()
    [question] = loader("mmlu_pro", n=1, seed=7)
    assert question.qid == "mmlupro-0"
    assert question.prompt_stem == "Which option?"
    assert question.options[question.option_letters.index(question.answer_key)] == "two"
    assert loader.diagnostics[0]["cached_arrow_path"] == str(arrow)
