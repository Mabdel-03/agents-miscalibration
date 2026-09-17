"""selection/seal.py: write-before-join, digest verification, append-only re-sealing."""
from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from agent_allocation.contracts import SelectionResult, SelectionStatus
from agent_allocation.paths import RunPaths
from agent_allocation.selection.seal import (
    SealedSelections, SealError, SealInfo, load_sealed_selections, read_seal, seal_selections,
)


@pytest.fixture
def paths(tmp_path):
    return RunPaths.for_run(tmp_path / "run", locks_dir=tmp_path / "locks", local_state_dir=tmp_path / "local")


def _sel(policy, selected="c0", status=SelectionStatus.SELECTED, cpu=0.001):
    return SelectionResult(policy_id=policy, selected_candidate_id=selected, selector_name="plurality_math_exact",
                           selector_version="06-selection-v1", pool_hash="pool", public_evidence_hash=None,
                           cpu_seconds=cpu, status=status.value)


def test_seal_writes_sorted_jsonl_and_a_matching_seal(paths):
    rows = [_sel("main/b/sample/2/current"),
            _sel("main/a/sample/2/current", None, SelectionStatus.ABSTAINED_ALL_INVALID)]
    digest = seal_selections(paths, "main", rows)
    raw = paths.selections("main").read_bytes()
    assert sha256(raw).hexdigest() == digest
    seal = json.loads(paths.selection_seal("main").read_text())
    assert set(seal) == {"sha256", "count", "sealed_at"} and seal["sha256"] == digest and seal["count"] == 2
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [line["policy_id"] for line in lines] == ["main/a/sample/2/current", "main/b/sample/2/current"]
    loaded = load_sealed_selections(paths, "main")
    assert loaded.by_policy["main/b/sample/2/current"] == rows[0] and loaded.seal.sha256 == digest


def test_reseal_is_idempotent_and_ignores_cpu_time(paths):
    first = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=0.5)])
    sealed_at = read_seal(paths, "main").sealed_at
    again = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=9.0)])
    assert again == first and read_seal(paths, "main").sealed_at == sealed_at


def test_reseal_may_only_append(paths):
    seal_selections(paths, "main", [_sel("main/a/x/current", cpu=0.5)])
    grown = seal_selections(paths, "main", [_sel("main/a/x/current", cpu=7.0), _sel("main/b/x/current")])
    loaded = load_sealed_selections(paths, "main")
    assert loaded.seal.sha256 == grown and loaded.by_policy["main/a/x/current"].cpu_seconds == 0.5
    with pytest.raises(SealError):
        seal_selections(paths, "main", [_sel("main/a/x/current", selected="c9"), _sel("main/b/x/current")])
    with pytest.raises(SealError):
        seal_selections(paths, "main", [_sel("main/b/x/current")])


def test_tampering_and_missing_seals_refuse_to_load(paths):
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")
    seal_selections(paths, "main", [_sel("main/a/x/current")])
    path = paths.selections("main")
    path.write_bytes(path.read_bytes().replace(b'"c0"', b'"c1"'))
    with pytest.raises(SealError):
        load_sealed_selections(paths, "main")


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
        seal_selections(paths, "../main", [_sel("../main/a/x/current")])


def test_sealed_selections_cannot_be_forged():
    with pytest.raises(SealError):
        SealedSelections(partition="main", seal=SealInfo("x", 0, "t"), by_policy={})
