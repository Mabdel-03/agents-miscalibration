"""selection/seal.py: append-only sealed batches, chain verification, write-before-join."""
from __future__ import annotations

import json
import warnings
from dataclasses import replace
from hashlib import sha256

import pytest

from agent_allocation.contracts import SelectionResult, SelectionStatus
from agent_allocation.paths import RunPaths
from agent_allocation.selection.seal import (
    SealDriftWarning, SealedSelections, SealError, SealInfo, load_sealed_selections, read_seals, seal_selections,
)


@pytest.fixture
def paths(tmp_path):
    return RunPaths.for_run(tmp_path / "run", locks_dir=tmp_path / "locks", local_state_dir=tmp_path / "local")


def _sel(policy, selected="c0", status=SelectionStatus.SELECTED, cpu=0.001, pool="pool"):
    return SelectionResult(policy_id=policy, selected_candidate_id=selected, selector_name="plurality_math_exact",
                           selector_version="06-selection-v2", pool_hash=pool, public_evidence_hash=None,
                           cpu_seconds=cpu, status=status)


def test_first_seal_writes_batch_zero_with_a_null_chain_link(paths):
    rows = [_sel("main/b/sample/2/current"),
            _sel("main/a/sample/2/current", None, SelectionStatus.ABSTAINED_ALL_INVALID)]
    digest = seal_selections(paths, "main", rows)
    raw = paths.selections("main", 0).read_bytes()
    assert sha256(raw).hexdigest() == digest
    seal = json.loads(paths.selection_seal("main", 0).read_text())
    assert set(seal) == {"sha256", "count", "sealed_at", "previous_seal_sha256"}
    assert (seal["sha256"], seal["count"], seal["previous_seal_sha256"]) == (digest, 2, None)
    assert [json.loads(line)["policy_id"] for line in raw.splitlines()] == ["main/a/sample/2/current",
                                                                            "main/b/sample/2/current"]
    loaded = load_sealed_selections(paths, "main")
    assert loaded.by_policy["main/b/sample/2/current"] == rows[0] and loaded.seal.sha256 == digest
    assert loaded.by_policy["main/a/sample/2/current"].status is SelectionStatus.ABSTAINED_ALL_INVALID


def test_later_calls_append_only_unsealed_checkpoints(paths):
    first = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=0.5)])
    again = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=9.0)])          # nothing new
    assert again == first and not paths.selections("main", 1).exists()
    second = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=7.0), _sel("main/b/x/current")])
    assert second != first
    batch1 = [json.loads(line)["policy_id"] for line in paths.selections("main", 1).read_bytes().splitlines()]
    assert batch1 == ["main/b/x/current"]
    assert json.loads(paths.selection_seal("main", 1).read_text())["previous_seal_sha256"] == first
    assert paths.selections("main", 0).read_bytes().count(b"\n") == 1                     # batch 0 untouched
    loaded = load_sealed_selections(paths, "main")
    assert [s.batch for s in loaded.seals] == [0, 1] and loaded.by_policy["main/a/x/current"].cpu_seconds == 0.5
    assert read_seals(paths, "main") == loaded.seals


def test_changed_pools_raise_and_changed_selections_warn(paths):
    seal_selections(paths, "main", [_sel("main/a/x/current")])
    with pytest.raises(SealError):
        seal_selections(paths, "main", [_sel("main/a/x/current", pool="other-pool")])
    with pytest.warns(SealDriftWarning):
        seal_selections(paths, "main", [_sel("main/a/x/current", selected="c9")])
    assert load_sealed_selections(paths, "main").by_policy["main/a/x/current"].selected_candidate_id == "c0"


def test_a_paused_selection_resumes_in_a_later_batch(paths):
    paused = _sel("main/a/x/current", selected=None, status=SelectionStatus.PAUSED_CHECK_ERROR)
    seal_selections(paths, "main", [paused])
    with pytest.raises(SealError):
        seal_selections(paths, "main", [_sel("main/a/x/current", pool="other-pool")])
    seal_selections(paths, "main", [_sel("main/a/x/current")])
    loaded = load_sealed_selections(paths, "main")
    assert loaded.by_policy["main/a/x/current"].status is SelectionStatus.SELECTED and len(loaded.seals) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SealDriftWarning):
            seal_selections(paths, "main", [paused])                  # a resumed selection is never re-paused


def test_tampering_gaps_and_missing_seals_refuse_to_load(paths):
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")
    seal_selections(paths, "main", [_sel("main/a/x/current")])
    seal_selections(paths, "main", [_sel("main/b/x/current")])
    batch = paths.selections("main", 1)
    original = batch.read_bytes()
    batch.write_bytes(original.replace(b'"c0"', b'"c1"'))
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")
    batch.write_bytes(original)
    seal0 = paths.selection_seal("main", 0)
    content = seal0.read_text()
    seal0.unlink()
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")                         # seals must be contiguous from batch 0
    seal0.write_text(content)
    link = json.loads(paths.selection_seal("main", 1).read_text())
    paths.selection_seal("main", 1).write_text(json.dumps({**link, "previous_seal_sha256": "0" * 64}))
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")


def test_an_unsealed_batch_left_by_a_crash_is_replaced(paths):
    first = seal_selections(paths, "main", [_sel("main/a/x/current")])
    paths.selections("main", 1).write_bytes(b"partial")
    assert load_sealed_selections(paths, "main").seal.sha256 == first
    seal_selections(paths, "main", [_sel("main/b/x/current")])
    assert set(load_sealed_selections(paths, "main").by_policy) == {"main/a/x/current", "main/b/x/current"}


def test_invalid_inputs(paths):
    with pytest.raises(ValueError):
        seal_selections(paths, "main", [])
    with pytest.raises(ValueError):
        seal_selections(paths, "main", [_sel("pilot/a/x/current")])
    with pytest.raises(ValueError):
        seal_selections(paths, "main", [_sel("main/a/x/current"), _sel("main/a/x/current")])
    with pytest.raises(ValueError):
        seal_selections(paths, "main", [_sel("main/a/x/current", selected=None)])
    with pytest.raises(ValueError):
        seal_selections(paths, "main", [_sel("main/a/x/current", pool="")])
    with pytest.raises(TypeError):
        seal_selections(paths, "main", [replace(_sel("main/a/x/current"), status="selected")])
    with pytest.raises(ValueError):
        seal_selections(paths, "../main", [_sel("../main/a/x/current")])


def test_sealed_selections_cannot_be_forged():
    with pytest.raises(SealError):
        SealedSelections(partition="main", seals=(SealInfo(0, "x", 1, "t", None),), by_policy={})
