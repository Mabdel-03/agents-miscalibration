"""Seal the round-t states a FORK / SUFFIX cell depends on (plan §5.3, spec §12.1).

``ard/runner.py`` resolves a ``state:<sha>`` hard dependency against
``<run_root>/states/<sha>/REGISTER.json`` and refuses the cell (``EXIT_DEPS_UNMET``, no
``meta.json``, zero requests) unless the file exists and hashes to ``sha``.  ``freeze_register``
is the writer that matches that reader.  Nothing called it: ``StateRegister`` in ``states.py``
writes a different layout (``states/<file>``), so the fork stage had a dependency mechanism at
both ends and no step in the middle.

This is that step.  It reads the states each completed episode already records, seals exactly
the parent rounds a fork branches from, and prints the sha the manifest builder pins.

    PYTHONPATH=src python -m agents_scaling.ard.seal --run-root <root> \
        --cells-file cells_P1_8B_trainval_v2.json --round 1 --round 2

Two rules the sealer enforces, because a fork frame is training data for the controller:

* **Only complete cells seal.**  A cell without ``meta.json`` is still generating, and sealing
  its states would freeze a cohort that is about to grow — the sha would then name a set no
  later run can reproduce.
* **The seal is content-addressed over what it contains**, so re-sealing the same states gives
  the same sha and re-sealing a different set gives a different one.  A cell added later
  produces a new register rather than mutating the frozen one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard.cells import load_ard_cells_file
from agents_scaling.ard.runner import freeze_register
from agents_scaling.study.types import ProtocolError

#: What the register declares about itself, so a reader never has to guess what was sealed.
SCHEMA = "ard36-state-register-v1"

#: Identity fields of a sealed state.  ``state_id`` is WP2's content address for the round
#: state; the rest are what a fork needs to rebuild the branch and what an auditor needs to
#: tell two seals apart.
STATE_FIELDS = ("state_id", "episode_id", "round", "decision_ids", "answer_classes",
                "valid", "protected_slot", "leader_id")


def _episode_files(run_root: Path, cell_ids: Sequence[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for cell_id in cell_ids:
        cell_dir = run_root / "cells" / cell_id
        if not (cell_dir / "meta.json").exists():
            continue                       # still generating; sealing it would freeze a moving set
        out += [(cell_id, path) for path in sorted((cell_dir / "items").glob("*.json"))]
    return out


def collect_states(
    run_root: str | os.PathLike,
    cell_ids: Sequence[str],
    *,
    rounds: Iterable[int] = K.FORK_PARENT_ROUNDS,
    protocols: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sealable states of ``cell_ids``, plus a census of what was read."""
    root = Path(run_root)
    wanted = {int(r) for r in rounds}
    by_id: dict[str, dict[str, Any]] = {}
    census = {"cells_read": 0, "cells_skipped_incomplete": 0, "episodes": 0, "items": 0}
    seen_cells: set[str] = set()
    for cell_id in cell_ids:
        if not (root / "cells" / cell_id / "meta.json").exists():
            census["cells_skipped_incomplete"] += 1
    for cell_id, path in _episode_files(root, cell_ids):
        if cell_id not in seen_cells:
            seen_cells.add(cell_id)
            census["cells_read"] += 1
        census["items"] += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode in payload.get("episodes") or []:
            if protocols is not None and str(episode.get("protocol")) not in set(protocols):
                continue
            census["episodes"] += 1
            for state in episode.get("states") or []:
                if int(state.get("round", -1)) not in wanted:
                    continue
                sid = str(state.get("state_id") or "")
                if not sid:
                    raise ProtocolError(f"{path}: a round state carries no state_id")
                entry = {field: state.get(field) for field in STATE_FIELDS}
                entry["protocol"] = episode.get("protocol")
                entry["checkpoint"] = episode.get("checkpoint")
                entry["item_id"] = episode.get("item_id")
                previous = by_id.get(sid)
                if previous is not None and previous != entry:
                    raise ProtocolError(
                        f"state {sid[:18]} appears twice with different content; "
                        "a state id is a content address and must not disagree with itself"
                    )
                by_id[sid] = entry
    return [by_id[sid] for sid in sorted(by_id)], census


def seal_states(
    run_root: str | os.PathLike,
    cell_ids: Sequence[str],
    *,
    rounds: Iterable[int] = K.FORK_PARENT_ROUNDS,
    protocols: Sequence[str] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    """Freeze the states and return ``(sha, census)``; the sha is what a manifest pins."""
    states, census = collect_states(run_root, cell_ids, rounds=rounds, protocols=protocols)
    if not states:
        raise ProtocolError(
            "refusing to seal an empty state register: no completed cell in this manifest "
            f"carries a round in {sorted({int(r) for r in rounds})}. Sealing nothing would "
            "hand FORK a dependency that resolves but contains no parents."
        )
    payload = {
        "schema": SCHEMA,
        "rounds": sorted({int(r) for r in rounds}),
        "protocols": sorted(protocols) if protocols else None,
        "n_states": len(states),
        "states": states,
    }
    sha = freeze_register(run_root, "state", payload)
    return sha, {**census, "states_sealed": len(states)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.seal",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--cells-file", required=True,
                    help="manifest whose completed cells are sealed")
    ap.add_argument("--round", type=int, action="append", dest="rounds", default=[],
                    help="parent round to seal (repeatable; default contract.FORK_PARENT_ROUNDS)")
    ap.add_argument("--protocol", action="append", dest="protocols", default=[],
                    help="restrict to these protocols (repeatable; default all)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be sealed")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    cells_file = Path(args.cells_file)
    if not cells_file.is_absolute():
        cells_file = run_root / cells_file
    cell_ids = [cell.cell_id for cell in load_ard_cells_file(cells_file)]
    rounds = args.rounds or list(K.FORK_PARENT_ROUNDS)
    protocols = args.protocols or None

    if args.dry_run:
        states, census = collect_states(run_root, cell_ids, rounds=rounds, protocols=protocols)
        print(json.dumps({"would_seal": len(states), "rounds": sorted(rounds), **census}, indent=2))
        return 0
    sha, census = seal_states(run_root, cell_ids, rounds=rounds, protocols=protocols)
    print(json.dumps({"state_sha": sha,
                      "register": str(run_root / "states" / sha / "REGISTER.json"),
                      "rounds": sorted(rounds), **census}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
