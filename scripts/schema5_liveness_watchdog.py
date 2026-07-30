#!/usr/bin/env python3
"""Report whether the schema-5 controllers are still heartbeating.

The dispatcher already runs a five-minute health monitor, but it runs that monitor
itself: a dispatcher that is alive and scheduled yet wedged stops heartbeating and stops
monitoring at the same moment, and Slurm sees a healthy job, so no failure mail is sent.
This probe closes that specific gap.  It reads control state read-only, never writes,
never submits, and exits non-zero when a role is stale, so an ordinary
``--mail-type=FAIL`` Slurm job turns staleness into mail.

It is deliberately not a replacement for the forced-command external watchdog, which
also survives the cluster itself becoming unavailable.  This probe only survives the
controllers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


_CONTROL_PATH = Path(__file__).resolve().parent.parent / "slurm" / "schema5_control.py"


def _load_control_module():
    name = "_schema5_liveness_control"
    spec = importlib.util.spec_from_file_location(name, _CONTROL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load control module: {_CONTROL_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the control plane defines dataclasses, which
    # resolve their own module out of sys.modules while the class body is processed.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


control = _load_control_module()


def _read_control(state_dir: Path) -> dict[str, Any]:
    """Read control for liveness only.

    Deliberately not ``load_control``: that validates admission, capacity, monitoring
    and finalization state as well, and a probe whose job is to raise the alarm should
    not go silent because some unrelated part of control is malformed.  Only the
    envelope is checked, so the probe cannot be pointed at the wrong file.
    """

    path = state_dir / "control.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"schema-5 control is not initialized: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read schema-5 control {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("schema-5 control must be a JSON object")
    if value.get("schema_version") != control.CONTROL_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported control schema {value.get('schema_version')!r}"
        )
    if value.get("protocol") != control.CONTROL_PROTOCOL:
        raise RuntimeError(f"unexpected control protocol {value.get('protocol')!r}")
    return value


def evaluate(
    state_dir: Path, *, stale_after_seconds: float, now: float | None = None
) -> dict[str, Any]:
    """One read-only liveness observation of every controller role."""

    timestamp = time.time() if now is None else float(now)
    loaded = _read_control(state_dir)
    desired_state = loaded.get("desired_state")
    roles: dict[str, Any] = {}
    stale: list[str] = []
    for role in control.ROLE_NAMES:
        role_state = loaded.get("controllers", {}).get(role, {})
        age = control.controller_liveness_age(role_state, timestamp)
        # A role that has never been claimed has no age; that is only meaningful once
        # the chain is actually running, so it is reported but not called stale here.
        is_stale = age is not None and age > stale_after_seconds
        roles[role] = {
            "liveness_age_seconds": age,
            "stale": is_stale,
            "never_claimed": age is None,
        }
        if is_stale:
            stale.append(role)
    # Only a running chain is expected to heartbeat; a paused one legitimately does not.
    enforced = desired_state in {"resuming", "running"}
    return {
        "observed_at": timestamp,
        "desired_state": desired_state,
        "stale_after_seconds": stale_after_seconds,
        "enforced": enforced,
        "roles": roles,
        "stale_roles": stale,
        "healthy": not (enforced and stale),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=control.STALE_HEARTBEAT_SECONDS,
        help="controller heartbeat age that counts as stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate(
            arguments.state_dir.expanduser().resolve(),
            stale_after_seconds=float(arguments.stale_after_seconds),
        )
    except Exception as error:  # surfaced through the job's failure mail
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["healthy"]:
        print(
            "ERROR: stale schema-5 controllers: "
            + ", ".join(report["stale_roles"]),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
