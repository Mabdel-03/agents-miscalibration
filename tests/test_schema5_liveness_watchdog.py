from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import schema5_liveness_watchdog as watchdog


# The probe exists to catch a controller that is scheduled and alive but has stopped
# heartbeating, which neither the dispatcher's own health monitor nor Slurm failure mail
# can observe.  It must never write, and must only enforce on a running chain.


def _control(state_dir: Path, *, desired_state: str, heartbeats: dict) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": watchdog.control.CONTROL_SCHEMA_VERSION,
        "protocol": watchdog.control.CONTROL_PROTOCOL,
        "desired_state": desired_state,
        "controllers": {
            role: (
                {"heartbeat": {"timestamp": value}}
                if value is not None
                else {"heartbeat": None}
            )
            for role, value in heartbeats.items()
        },
    }
    (state_dir / "control.json").write_text(json.dumps(payload), encoding="utf-8")
    return state_dir


def test_fresh_heartbeats_are_healthy(tmp_path: Path) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="running",
        heartbeats={"dispatcher": 990.0, "fleet_supervisor": 995.0},
    )

    report = watchdog.evaluate(state_dir, stale_after_seconds=600.0, now=1000.0)

    assert report["healthy"] is True
    assert report["stale_roles"] == []
    assert report["roles"]["dispatcher"]["liveness_age_seconds"] == 10.0


def test_a_wedged_controller_is_reported_stale(tmp_path: Path) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="running",
        heartbeats={"dispatcher": 100.0, "fleet_supervisor": 995.0},
    )

    report = watchdog.evaluate(state_dir, stale_after_seconds=600.0, now=1000.0)

    assert report["healthy"] is False
    assert report["stale_roles"] == ["dispatcher"]
    assert report["roles"]["dispatcher"]["stale"] is True
    assert report["roles"]["fleet_supervisor"]["stale"] is False


def test_a_paused_chain_is_not_expected_to_heartbeat(tmp_path: Path) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="paused",
        heartbeats={"dispatcher": 0.0, "fleet_supervisor": 0.0},
    )

    report = watchdog.evaluate(state_dir, stale_after_seconds=600.0, now=1000.0)

    assert report["enforced"] is False
    assert report["healthy"] is True
    # Staleness is still observed and reported, just not enforced.
    assert report["stale_roles"] == ["dispatcher", "fleet_supervisor"]


def test_a_never_claimed_role_is_not_called_stale(tmp_path: Path) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="running",
        heartbeats={"dispatcher": None, "fleet_supervisor": 995.0},
    )

    report = watchdog.evaluate(state_dir, stale_after_seconds=600.0, now=1000.0)

    assert report["roles"]["dispatcher"]["never_claimed"] is True
    assert report["roles"]["dispatcher"]["stale"] is False
    assert report["healthy"] is True


def test_cli_exits_two_on_staleness_and_writes_nothing(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="running",
        heartbeats={"dispatcher": 0.0, "fleet_supervisor": 0.0},
    )
    before = {
        path: path.read_bytes() for path in state_dir.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(watchdog.time, "time", lambda: 1000.0)

    code = watchdog.main(["--state-dir", str(state_dir), "--stale-after-seconds", "600"])

    assert code == 2
    assert "stale schema-5 controllers" in capsys.readouterr().err
    after = {
        path: path.read_bytes() for path in state_dir.rglob("*") if path.is_file()
    }
    assert after == before


def test_cli_exits_zero_when_healthy(tmp_path: Path, monkeypatch) -> None:
    state_dir = _control(
        tmp_path / "state",
        desired_state="running",
        heartbeats={"dispatcher": 990.0, "fleet_supervisor": 990.0},
    )
    monkeypatch.setattr(watchdog.time, "time", lambda: 1000.0)

    assert watchdog.main(["--state-dir", str(state_dir)]) == 0


def test_cli_exits_two_when_control_is_unreadable(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "absent"
    missing.mkdir()

    assert watchdog.main(["--state-dir", str(missing)]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_default_threshold_tracks_the_control_plane() -> None:
    # The probe must not drift from the control plane's own staleness definition.
    assert (
        watchdog.control.STALE_HEARTBEAT_SECONDS
        == watchdog._parser().get_default("stale_after_seconds")
    )
