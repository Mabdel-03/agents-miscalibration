from __future__ import annotations

import copy

import pytest

from agent_allocation.api.endpoints import EndpointBlocked, endpoint_candidates, select_standard_endpoint
from tests.unit.api.conftest import FLASH, FLASH_LITE, load_snapshot


def test_flash_lite_locks_the_standard_ai_studio_endpoint():
    lock = select_standard_endpoint(load_snapshot(FLASH_LITE), FLASH_LITE, snapshot_path="locks/snapshots/x.json",
                                    snapshot_sha256="c" * 64, fetched_at="2026-09-17T06:00:00Z")
    assert lock.provider_tag == "google-ai-studio"
    assert lock.provider_name == "Google AI Studio"
    assert lock.service_tier == "standard"
    assert (lock.input_price_per_token, lock.output_price_per_token) == ("0.0000001", "0.0000004")
    assert lock.internal_reasoning_price_per_token == "0.0000004"
    assert lock.max_completion_tokens == 65535 and lock.context_length == 1048576
    assert {"reasoning", "max_tokens", "temperature", "top_p"} <= set(lock.supported_parameters)
    assert lock.snapshot_sha256 == "c" * 64
    volatile = select_standard_endpoint(load_snapshot(FLASH_LITE), FLASH_LITE)
    assert volatile.lock_hash == lock.lock_hash  # snapshot path/hash/time never enter the lock hash


def test_flash_locks_its_own_standard_prices():
    lock = select_standard_endpoint(load_snapshot(FLASH), FLASH)
    assert lock.provider_tag == "google-ai-studio"
    assert (lock.input_price_per_token, lock.output_price_per_token) == ("0.0000003", "0.0000025")


@pytest.mark.parametrize("tag", ["google-ai-studio/flex", "google-ai-studio/priority", "google-vertex/global/priority",
                                 "google-ai-studio:free", "google-ai-studio/batch"])
def test_tier_variants_are_never_selected(tag):
    with pytest.raises(EndpointBlocked, match="tier variant"):
        select_standard_endpoint(load_snapshot(FLASH), FLASH, preferred_tag=tag)


def test_a_pinned_standard_vertex_endpoint_can_be_chosen_explicitly():
    lock = select_standard_endpoint(load_snapshot(FLASH_LITE), FLASH_LITE, preferred_tag="google-vertex")
    assert lock.provider_tag == "google-vertex" and lock.provider_name == "Google"


@pytest.mark.parametrize("model", ["google/gemini-2.5-flash-lite:free", "google/gemini-2.5-flash-lite-preview-09-2025",
                                   "google/gemini-2.0-flash-exp", "openrouter/auto", "google/gemini-experimental"])
def test_preview_free_and_routing_models_are_blocked(model):
    snapshot = copy.deepcopy(load_snapshot(FLASH_LITE))
    snapshot["data"]["id"] = model
    for endpoint in snapshot["data"]["endpoints"]:
        endpoint["model_id"] = model
    with pytest.raises(EndpointBlocked):
        select_standard_endpoint(snapshot, model)


def test_never_invents_a_tag_or_matches_by_prefix():
    snapshot = copy.deepcopy(load_snapshot(FLASH_LITE))
    snapshot["data"]["endpoints"] = [e for e in snapshot["data"]["endpoints"] if e["tag"] != "google-ai-studio"]
    with pytest.raises(EndpointBlocked, match="no endpoint tagged exactly 'google-ai-studio'"):
        select_standard_endpoint(snapshot, FLASH_LITE)
    with pytest.raises(EndpointBlocked, match="no endpoint tagged exactly"):
        select_standard_endpoint(load_snapshot(FLASH_LITE), FLASH_LITE, preferred_tag="google")


def test_snapshot_for_another_model_is_refused():
    with pytest.raises(EndpointBlocked, match="is for"):
        select_standard_endpoint(load_snapshot(FLASH), FLASH_LITE)


@pytest.mark.parametrize("mutate, message", [
    (lambda e: e.update(supported_parameters=["max_tokens", "temperature", "top_p"]), "required parameters"),
    (lambda e: e["pricing"].pop("internal_reasoning"), "internal_reasoning"),
    (lambda e: e["pricing"].update(prompt="0"), "positive price"),
    (lambda e: e["pricing"].update(completion="abc"), "not a decimal"),
    (lambda e: e.update(max_completion_tokens=None), "max_completion_tokens"),
    (lambda e: e.update(model_id="google/gemini-2.5-flash"), "serves"),
])
def test_incomplete_endpoint_metadata_blocks(mutate, message):
    snapshot = copy.deepcopy(load_snapshot(FLASH_LITE))
    for endpoint in snapshot["data"]["endpoints"]:
        if endpoint["tag"] == "google-ai-studio":
            mutate(endpoint)
    with pytest.raises(EndpointBlocked, match=message):
        select_standard_endpoint(snapshot, FLASH_LITE)


def test_duplicate_tags_are_ambiguous():
    snapshot = copy.deepcopy(load_snapshot(FLASH_LITE))
    duplicate = next(e for e in snapshot["data"]["endpoints"] if e["tag"] == "google-ai-studio")
    snapshot["data"]["endpoints"].append(copy.deepcopy(duplicate))
    with pytest.raises(EndpointBlocked, match="share the tag"):
        select_standard_endpoint(snapshot, FLASH_LITE)


def test_candidate_rows_explain_every_listed_endpoint():
    rows = {row["tag"]: row for row in endpoint_candidates(load_snapshot(FLASH), FLASH)}
    assert rows["google-ai-studio"]["standard_selectable"] is True
    assert rows["google-ai-studio/flex"]["standard_selectable"] is False
    assert rows["google-vertex/global/priority"]["problems"]
    assert rows["google-vertex"]["status"] == -5
