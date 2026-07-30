from __future__ import annotations

from pathlib import Path

import pytest

from slurm import schema5_control as control
from tests.test_schema5_control import _toolchain_binding, make_pins


@pytest.fixture(autouse=True)
def _stub_conda_toolchain_verification(monkeypatch):
    # Mirrors the stub in test_schema5_control; pin verification is not under test here.
    monkeypatch.setattr(
        control.conda_toolchain,
        "verified_conda_toolchain_binding",
        lambda root, exercise=True: _toolchain_binding(root),
    )


# `email_test` and `external_watchdog` are the only gates that depend on operator
# infrastructure outside the cluster.  A chain may be initialized without them, but the
# waiver has to be explicit and recorded, and everything else must keep failing closed.


def _init(tmp_path: Path, **overrides: bool) -> tuple[Path, dict, dict]:
    # make_pins builds its fixture tree once, so reuse the pins for any re-init.
    state_dir = tmp_path / "results" / ".dispatcher-schema5-v1"
    pins = make_pins(tmp_path)
    control_state = control.initialize_control(
        state_dir,
        pins=pins,
        launch_policy_overrides=overrides or None,
        now=10.0,
    )
    return state_dir, control_state, pins


def test_default_policy_requires_both_optional_gates() -> None:
    assert control.LAUNCH_POLICY_DEFAULTS == {
        "email_test_required": True,
        "external_watchdog_required": True,
    }


def test_control_without_a_policy_field_stays_fully_required() -> None:
    # Any chain created before this policy existed must behave exactly as before.
    assert control.launch_policy({}) == control.LAUNCH_POLICY_DEFAULTS
    assert control.effective_required_gates({}) == control.REQUIRED_GATES
    assert (
        control.effective_production_authorization_gates({})
        == control.PRODUCTION_AUTHORIZATION_GATES
    )


@pytest.mark.parametrize("value", ["false", 0, None, [], {}])
def test_a_malformed_policy_value_falls_back_to_required(value: object) -> None:
    # Only an explicit boolean may relax a gate; anything else fails closed.
    stored = {"launch_policy": {"email_test_required": value}}
    assert control.launch_policy(stored)["email_test_required"] is True


def test_waiving_the_email_gate_drops_only_that_gate() -> None:
    stored = {"launch_policy": {"email_test_required": False}}
    gates = control.effective_required_gates(stored)

    assert "email_test" not in gates
    assert set(gates) == set(control.REQUIRED_GATES) - {"email_test"}
    assert (
        control.effective_production_authorization_gates(stored)
        == control.PRODUCTION_AUTHORIZATION_GATES
    )


def test_waiving_the_watchdog_drops_only_that_authorization() -> None:
    stored = {"launch_policy": {"external_watchdog_required": False}}

    assert control.effective_production_authorization_gates(stored) == (
        "throughput_qualification",
    )
    assert control.effective_required_gates(stored) == control.REQUIRED_GATES


def test_init_records_the_policy_in_control_and_the_journal(tmp_path: Path) -> None:
    _, initialized, _pins = _init(tmp_path, external_watchdog_required=False)

    assert initialized["launch_policy"] == {
        "email_test_required": True,
        "external_watchdog_required": False,
    }
    # The waiver must be visible in the durable transition history, not only in state.
    history = initialized["transition_history"]
    assert history[0]["details"]["launch_policy"] == initialized["launch_policy"]


def test_waived_gates_still_occupy_a_readiness_record(tmp_path: Path) -> None:
    # The readiness map keeps its shape so existing state validation is unchanged; only
    # the enforcement loops narrow.
    _, initialized, _pins = _init(tmp_path, external_watchdog_required=False)

    assert "external_watchdog" in initialized["readiness"]
    assert initialized["readiness"]["external_watchdog"] == {
        "passed": False,
        "evidence": None,
        "attested_at": None,
    }


def test_reinit_with_a_different_policy_is_rejected(tmp_path: Path) -> None:
    state_dir, _initialized, pins = _init(tmp_path, external_watchdog_required=False)

    with pytest.raises(control.ControlError, match="existing launch policy"):
        control.initialize_control(
            state_dir,
            pins=pins,
            launch_policy_overrides={"external_watchdog_required": True},
            now=11.0,
        )


def test_reinit_with_the_same_policy_is_idempotent(tmp_path: Path) -> None:
    state_dir, first, pins = _init(tmp_path, external_watchdog_required=False)

    second = control.initialize_control(
        state_dir,
        pins=pins,
        launch_policy_overrides={"external_watchdog_required": False},
        now=999.0,
    )

    assert second["launch_policy"] == first["launch_policy"]
    assert second["created_timestamp"] == first["created_timestamp"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"no_such_gate_required": False}, "unknown launch policy key"),
        ({"email_test_required": "no"}, "must be a boolean"),
    ],
)
def test_invalid_policy_overrides_are_rejected(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    state_dir = tmp_path / "results" / ".dispatcher-schema5-v1"
    with pytest.raises(control.ControlError, match=message):
        control.initialize_control(
            state_dir,
            pins=make_pins(tmp_path),
            launch_policy_overrides=overrides,
            now=10.0,
        )
