"""Lead integration fix: the standard tier is reported as "default" by /generation (observed at the Gate 1 canary)."""
from agent_allocation.api.fake import fake_endpoint_lock
from agent_allocation.api.reconcile import metadata_identity_problems


def _meta(tier):
    lock = fake_endpoint_lock()
    return lock, {"provider_name": lock.provider_name, "service_tier": tier, "response_cache_source_id": None,
                  "model": lock.model}


def test_default_tier_is_accepted_for_a_standard_lock():
    lock, meta = _meta("default")
    assert not [p for p in metadata_identity_problems(meta, lock) if "service_tier" in p]


def test_flex_and_priority_are_still_drift():
    for tier in ("flex", "priority"):
        lock, meta = _meta(tier)
        assert [p for p in metadata_identity_problems(meta, lock) if "service_tier" in p]
