"""``allocation stop --run-dir RUN [--drain | --clear]``.

* default (STOP): the dispatcher stops admissions, gives in-flight requests a short grace, then abandons the rest;
  abandoned requests keep their full reservation (charged at the maximum), never released as free.
* ``--drain`` (DRAIN): stop admissions and let every in-flight request finish, persist and settle.
* ``--clear``: remove both sentinels. Sentinels persist until cleared; a dispatcher refuses to start while one exists,
  so a requeued job cannot silently resume a stopped run.

Sentinel content: ``{"mode", "requested_at", "host", "pid", "user"}``. Config keys consumed: none.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
from datetime import datetime, timezone

from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic


def write_sentinel(paths: RunPaths, *, drain: bool) -> str:
    target = paths.drain_sentinel if drain else paths.stop_sentinel
    payload = {"mode": "drain" if drain else "stop",
               "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "host": socket.gethostname(), "pid": os.getpid(), "user": getpass.getuser()}
    replace_atomic(target, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    return str(target)


def clear_sentinels(paths: RunPaths) -> list[str]:
    removed = []
    for target in (paths.stop_sentinel, paths.drain_sentinel):
        try:
            target.unlink()
            removed.append(str(target))
        except FileNotFoundError:
            pass
    return removed


def cmd_stop(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation stop")
    parser.add_argument("--run-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--drain", action="store_true")
    group.add_argument("--clear", action="store_true")
    args = parser.parse_args(argv)
    paths = RunPaths.for_run(args.run_dir)
    if not paths.run_dir.is_dir():
        print(f"run dir {paths.run_dir} does not exist", file=sys.stderr)
        return 2
    if args.clear:
        print(json.dumps({"cleared": clear_sentinels(paths)}))
        return 0
    print(json.dumps({"written": write_sentinel(paths, drain=args.drain)}))
    return 0
