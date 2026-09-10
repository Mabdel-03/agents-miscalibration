"""WP4 KIND_TABLE: completeness, lanes, endpoints, Slurm shape, dependencies, priority.

Plan §5.3 (cell kinds), §6 (lane sizing and the per-kind cell size), §7 (dispatch priority).
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import kinds as KINDS
from agents_scaling.ard.types import ArdCellKind, ArdManifestError


def test_the_table_covers_every_kind_exactly_once():
    assert set(KINDS.KIND_TABLE) == set(ArdCellKind)
    assert all(spec.kind is kind for kind, spec in KINDS.KIND_TABLE.items())


def test_every_kind_declares_a_module_and_callable_worker():
    for kind, spec in KINDS.KIND_TABLE.items():
        module, sep, attr = spec.worker.partition(":")
        assert sep == ":" and module.startswith("agents_scaling.ard") and attr, kind


def test_every_kind_lands_on_a_declared_lane_and_a_sane_slurm_shape():
    for kind, spec in KINDS.KIND_TABLE.items():
        assert spec.lane in K.LANES
        assert spec.cpus >= 1 and spec.mem.endswith("G") and spec.time.count(":") == 2
        assert spec.items_per_cell >= 1 and spec.parallel_items >= 1 and spec.max_inflight >= 1


@pytest.mark.parametrize("kind,needs", [
    (ArdCellKind.ROOTS, True), (ArdCellKind.EPISODE, True), (ArdCellKind.CHALLENGE, True),
    (ArdCellKind.FORK, True), (ArdCellKind.SUFFIX, True), (ArdCellKind.RECEIVER, True),
    (ArdCellKind.SELECTOR_AUDIT, True), (ArdCellKind.GRADE, False), (ArdCellKind.E0, False),
    (ArdCellKind.COMPACT, False), (ArdCellKind.CAPTURE, False),
])
def test_needs_endpoint_matches_the_lane_plan(kind, needs):
    assert KINDS.spec_for(kind).needs_endpoint is needs


def test_generation_kinds_follow_their_checkpoint_and_cpu_kinds_do_not():
    assert KINDS.spec_for(ArdCellKind.EPISODE).lane_for(K.CHECKPOINT_LARGE) == "32B"
    assert KINDS.spec_for(ArdCellKind.GRADE).lane_for(K.CHECKPOINT_LARGE) == "grade"
    assert KINDS.spec_for(ArdCellKind.COMPACT).lane_for(K.CHECKPOINT_SMALL) == "analysis"
    assert KINDS.spec_for(ArdCellKind.CAPTURE).lane_for(K.CHECKPOINT_SMALL) == "capture"


def test_dependent_kinds_declare_the_prefixes_they_may_carry():
    assert KINDS.CONTROLLER_DEP_PREFIX in KINDS.spec_for(ArdCellKind.EPISODE).dep_prefixes
    assert KINDS.STATE_DEP_PREFIX in KINDS.spec_for(ArdCellKind.FORK).dep_prefixes
    assert KINDS.STATE_DEP_PREFIX in KINDS.spec_for(ArdCellKind.SUFFIX).dep_prefixes
    assert KINDS.spec_for(ArdCellKind.ROOTS).dep_prefixes == ()


def test_controller_protocols_require_the_frozen_controller():
    sha = "ab" * 32
    assert KINDS.required_dependencies(K.Protocol.GATE, controller_sha=sha) == (f"controller:{sha}",)
    assert KINDS.required_dependencies(K.Protocol.CONF_PROTECT, controller_sha=sha)
    assert KINDS.required_dependencies(K.Protocol.PEER, controller_sha=None) == ()
    assert KINDS.required_dependencies(K.Protocol.RANDOM_PROTECT, controller_sha=None) == ()
    with pytest.raises(ArdManifestError):
        KINDS.required_dependencies(K.Protocol.GATE, controller_sha=None)


def test_the_dispatch_priority_key_orders_roots_before_episodes_before_analysis():
    assert KINDS.PRIORITY[ArdCellKind.ROOTS] < KINDS.PRIORITY[ArdCellKind.EPISODE]
    assert KINDS.PRIORITY[ArdCellKind.EPISODE] < KINDS.PRIORITY[ArdCellKind.CHALLENGE]
    assert KINDS.PRIORITY[ArdCellKind.FORK] < KINDS.PRIORITY[ArdCellKind.COMPACT]
    assert set(KINDS.PRIORITY) == set(ArdCellKind)


def test_deadline_bearing_kinds_are_named_so_they_never_use_a_burst_endpoint():
    assert ArdCellKind.FORK in KINDS.DEADLINE_BEARING
    assert ArdCellKind.CAPTURE in KINDS.DEADLINE_BEARING


def test_cpu_kinds_ignore_the_budget_based_cell_sizing():
    assert KINDS.items_per_cell(ArdCellKind.GRADE, reasoning_allowance=K.THINKING_TOKENS_HIGH) == \
        KINDS.spec_for(ArdCellKind.GRADE).items_per_cell
    assert KINDS.items_per_cell(ArdCellKind.COMPACT) == 1
