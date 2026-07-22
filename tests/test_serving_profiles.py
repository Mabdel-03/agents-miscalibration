"""Named serving layouts and deterministic long-context routing."""

from agents_scaling.config import ExperimentCell
from agents_scaling.serving.profiles import (
    LONG_CONTEXT_LIMIT,
    LONG_32B_PROFILE,
    LONG_PROFILE_BY_MODEL,
    SELECTIVE_LONG_MODEL_SIZES,
    STANDARD_32B_PROFILE,
    get_serving_profile,
    serving_metadata,
    serving_profile_for_cell,
)


def _cell(**overrides):
    values = {
        "model_size": "32B",
        "context_share_level": "plus_cot",
        "prompt_complexity_level": 3,
        "reasoning_level": "unlimited",
        "topology": "decentralized",
        "benchmark": "gpqa",
        "n_agents": 7,
    }
    values.update(overrides)
    return ExperimentCell(**values)


def test_explicit_32b_profiles_preserve_model_identity():
    normal = get_serving_profile(STANDARD_32B_PROFILE)
    long = get_serving_profile(LONG_32B_PROFILE)

    assert (normal.tp_size, normal.max_model_len) == (1, 16384)
    assert (long.tp_size, long.max_model_len) == (2, 40960)
    assert normal.model_size == long.model_size == "32B"
    assert normal.served_model_name == long.served_model_name == "32B"
    assert normal.hf_id == long.hf_id


def test_all_coordinated_32b_shared_work_cells_route_long():
    for topology in ("centralized", "decentralized"):
        for context in ("plus_intermediate", "plus_cot"):
            for reasoning in ("off", "b8192", "unlimited"):
                profile = serving_profile_for_cell(
                    _cell(
                        topology=topology,
                        context_share_level=context,
                        reasoning_level=reasoning,
                    )
                )
                assert profile.name == LONG_32B_PROFILE


def test_selective_small_model_long_profiles_preserve_identity_and_tp1():
    for model_size in SELECTIVE_LONG_MODEL_SIZES:
        long = get_serving_profile(LONG_PROFILE_BY_MODEL[model_size])
        normal = get_serving_profile(model_size)
        assert long.name == f"{model_size}-long"
        assert long.model_size == long.served_model_name == model_size
        assert long.hf_id == normal.hf_id
        assert (long.tp_size, long.max_model_len) == (1, LONG_CONTEXT_LIMIT)


def test_small_model_long_routing_matches_exact_dense_peer_boundary():
    for model_size in SELECTIVE_LONG_MODEL_SIZES:
        for topology in ("centralized", "decentralized"):
            for context in ("plus_intermediate", "plus_cot"):
                base = {
                    "model_size": model_size,
                    "topology": topology,
                    "context_share_level": context,
                }
                assert serving_profile_for_cell(
                    _cell(**base, n_agents=6, reasoning_level="b8192")
                ).name == f"{model_size}-long"
                assert serving_profile_for_cell(
                    _cell(**base, n_agents=6, reasoning_level="unlimited")
                ).name == f"{model_size}-long"
                assert serving_profile_for_cell(
                    _cell(**base, n_agents=7, reasoning_level="b2048")
                ).name == f"{model_size}-long"

                assert serving_profile_for_cell(
                    _cell(**base, n_agents=5, reasoning_level="unlimited")
                ).name == model_size
                assert serving_profile_for_cell(
                    _cell(**base, n_agents=6, reasoning_level="b2048")
                ).name == model_size
                assert serving_profile_for_cell(
                    _cell(**base, n_agents=7, reasoning_level="b512")
                ).name == model_size


def test_non_sharing_and_short_context_cells_stay_on_normal_profiles():
    assert serving_profile_for_cell(_cell(topology="single_agent")).name == "32B"
    assert serving_profile_for_cell(_cell(topology="independent")).name == "32B"
    assert serving_profile_for_cell(_cell(context_share_level="artifact_only")).name == "32B"
    assert serving_profile_for_cell(
        _cell(model_size="14B", topology="independent")
    ).name == "14B"
    assert serving_profile_for_cell(
        _cell(model_size="14B", context_share_level="artifact_only")
    ).name == "14B"


def test_serving_metadata_records_profile_and_effective_limit():
    metadata = serving_metadata(LONG_32B_PROFILE)
    assert metadata == {
        "serving_profile": "32B-long",
        "served_model_name": "32B",
        "effective_context_limit": 40960,
        "tensor_parallel_size": 2,
    }
