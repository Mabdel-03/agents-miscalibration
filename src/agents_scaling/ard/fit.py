"""Fit and freeze the E4 controller from the run (spec §12.2; plan §5.4, §7).

The last step of the controller chain: :mod:`agents_scaling.ard.frames` turns the run's
episodes and forks into paired frames, :func:`controller.fit_controller` fits the two heads,
and :func:`controller.freeze_controller` writes ``controller/<sha>/MODEL.json`` — the register
a GATE cell's hard dependency resolves against.

    PYTHONPATH=src python -m agents_scaling.ard.fit --run-root <root> \\
        --peer-cells cells_P1_8B_trainval_v2.json \\
        --private-cells cells_P1_8B_trainval_v2.json \\
        --fork-cells cells_E4_forks_8B.json

The sha it prints is what ``build_e2_natural(controller_sha=...)`` pins, so a controller
re-freeze lands in a new directory and every GATE cell built against the old one keeps
resolving to the model it was actually built for.

**The freeze is refused unless the frame can support it.** A controller fit on a handful of
rows, or on rows where one branch is almost never observed, is worse than no controller: GATE
would still run, still produce a number, and that number would be noise wearing the study's
primary estimand as a label.  So this module checks coverage *before* fitting and says exactly
what is missing when it refuses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import frames as FR
from agents_scaling.ard.cells import load_ard_cells_file
from agents_scaling.ard.controller import fit_controller, freeze_controller
from agents_scaling.study.types import ProtocolError

#: A paired row is only usable for a head when that head's branch outcome is observed.
#: Below this many usable rows per head the fit is refused rather than reported.
MIN_ROWS_PER_HEAD: int = 200

#: Both branches must be observed on enough rows for the *paired* comparator to mean anything.
MIN_PAIRED_ROWS: int = 100


def coverage(rows: Sequence[FR.PairedRow]) -> dict[str, Any]:
    """What the frame can and cannot support, in the terms the refusal message uses."""
    by_round = Counter(r.round_ for r in rows)
    return {
        "rows": len(rows),
        "train_rows": sum(1 for r in rows if r.split == "train"),
        "validation_rows": sum(1 for r in rows if r.split == "validation"),
        "y_private_observed": sum(1 for r in rows if r.y_private is not None),
        "y_peer_observed": sum(1 for r in rows if r.y_peer is not None),
        "both_observed": sum(1 for r in rows
                             if r.y_private is not None and r.y_peer is not None),
        "by_round": {str(r): int(by_round.get(r, 0)) for r in FR.DECISION_ROUNDS},
        "rounds_with_paired_rows": sorted(
            {r.round_ for r in rows if r.y_private is not None and r.y_peer is not None}
        ),
    }


def assert_fittable(cov: Mapping[str, Any]) -> None:
    """Refuse a fit the data cannot support, naming what is missing."""
    problems: list[str] = []
    if int(cov["y_private_observed"]) < MIN_ROWS_PER_HEAD:
        problems.append(
            f"p_private has {cov['y_private_observed']} observed rows (< {MIN_ROWS_PER_HEAD}); "
            "the fork cells have probably not finished or not been graded"
        )
    if int(cov["y_peer_observed"]) < MIN_ROWS_PER_HEAD:
        problems.append(
            f"p_peer has {cov['y_peer_observed']} observed rows (< {MIN_ROWS_PER_HEAD}); "
            "the PEER trajectories have probably not been graded"
        )
    if int(cov["both_observed"]) < MIN_PAIRED_ROWS:
        problems.append(
            f"only {cov['both_observed']} rows carry both branches (< {MIN_PAIRED_ROWS}); "
            "u = p_private - p_peer would be fit on almost no paired evidence"
        )
    if not int(cov["train_rows"]) or not int(cov["validation_rows"]):
        problems.append(
            f"train/validation split is {cov['train_rows']}/{cov['validation_rows']}; "
            "penalty selection needs both"
        )
    missing_rounds = [r for r in FR.DECISION_ROUNDS
                      if r not in set(cov["rounds_with_paired_rows"])]
    if missing_rounds:
        problems.append(
            f"no paired rows at round(s) {missing_rounds}: GATE acts before every round in "
            f"{list(FR.DECISION_ROUNDS)}, so a controller fit without them would be "
            "extrapolating at the rounds it is asked to act on"
        )
    if problems:
        raise ProtocolError(
            "refusing to fit the controller:\n  - " + "\n  - ".join(problems) +
            "\nA controller fit on too little is worse than none: GATE would still run and "
            "still produce a number for the study's primary contrast."
        )


def fit_and_freeze(
    run_root: str | os.PathLike,
    *,
    peer_cells: Sequence[str],
    private_cells: Sequence[str] = (),
    fork_cells: Sequence[str] = (),
    study_seed: bytes = b"\x00" * 32,
    dry_run: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """``(controller_sha, report)``; ``sha`` is ``None`` for a dry run."""
    rows = FR.build_paired_rows(run_root, peer_cells=peer_cells,
                                private_cells=private_cells, fork_cells=fork_cells)
    cov = coverage(rows)
    assert_fittable(cov)
    train, val = FR.split_frames(rows)
    if dry_run:
        return None, {"coverage": cov, "fitted": False}

    controller = fit_controller(
        train, val, study_seed=study_seed,
        meta={"run_root": str(run_root), "n_train": len(train.rows),
              "n_validation": len(val.rows), "coverage": cov,
              "feature_schema_version": K.FEATURE_SCHEMA_VERSION},
    )
    sha = freeze_controller(run_root, controller)
    return sha, {
        "coverage": cov,
        "fitted": True,
        "heads": sorted(controller.models),
        "selections": {k: v.get("penalty") for k, v in controller.selections.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.fit",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--peer-cells", required=True)
    ap.add_argument("--private-cells", default=None)
    ap.add_argument("--fork-cells", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report frame coverage and whether a fit would be admitted")
    args = ap.parse_args(argv)

    root = Path(args.run_root)

    def ids(name: str | None) -> list[str]:
        if not name:
            return []
        path = Path(name) if Path(name).is_absolute() else root / name
        return [cell.cell_id for cell in load_ard_cells_file(path)]

    sha, report = fit_and_freeze(
        root, peer_cells=ids(args.peer_cells), private_cells=ids(args.private_cells),
        fork_cells=ids(args.fork_cells), dry_run=args.dry_run,
    )
    if sha:
        report["controller_sha"] = sha
        report["register"] = str(root / "controller" / sha / "MODEL.json")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
