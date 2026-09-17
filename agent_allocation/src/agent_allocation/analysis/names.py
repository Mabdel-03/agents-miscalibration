"""Checkpoint names from the reference DAG: ``sample/k``, ``private_single/t``, ``private_team/n/r``,
``peer/n/r``, ``deep/m``, ``cp/h``, ``pooled/4``. Analysis parses ``PolicyCheckpoint.name`` (never ``policy_id``,
whose item ids contain ``/``)."""
from __future__ import annotations

from dataclasses import dataclass

FAMILY_ORDER: tuple[str, ...] = ("sample", "private_single", "private_team", "peer", "deep", "cp", "pooled")
PARAM_COUNT = {"sample": 1, "private_single": 1, "private_team": 2, "peer": 2, "deep": 1, "cp": 1, "pooled": 1}
CP_TEAM_SIZE = 4


class CheckpointNameError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointName:
    family: str
    params: tuple[int, ...]

    @property
    def label(self) -> str:
        return "/".join([self.family, *map(str, self.params)])

    @property
    def nominal_generations(self) -> int:
        """Generation opportunities a from-scratch deployment needs (MASTER_SPEC §5)."""
        f, p = self.family, self.params
        if f == "sample" or f == "private_single":
            return p[0]
        if f in ("private_team", "peer"):
            return p[0] * (1 + p[1])
        if f == "deep":
            return 1
        if f == "cp":
            return CP_TEAM_SIZE * (2 + p[0])
        if f == "pooled":
            return p[0] + 1
        raise CheckpointNameError(f)

    @property
    def trajectory(self) -> tuple[tuple, int] | None:
        """(trajectory key, progress index) for nested-prefix families; ``None`` for independent calls."""
        f, p = self.family, self.params
        if f in ("sample", "private_single", "cp"):
            return (f,), p[0]
        if f in ("private_team", "peer"):
            return (f, p[0]), p[1]
        return None

    @property
    def sort_key(self) -> tuple:
        return (FAMILY_ORDER.index(self.family), self.params)


def parse_name(name: str) -> CheckpointName:
    parts = str(name).split("/")
    family = parts[0]
    if family not in PARAM_COUNT or len(parts) != 1 + PARAM_COUNT[family]:
        raise CheckpointNameError(f"unrecognised checkpoint name {name!r}")
    try:
        params = tuple(int(x) for x in parts[1:])
    except ValueError as exc:
        raise CheckpointNameError(f"non-integer parameter in {name!r}") from exc
    return CheckpointName(family, params)


def name_sort_key(name: str) -> tuple:
    try:
        return (0, parse_name(name).sort_key)
    except CheckpointNameError:
        return (1, (str(name),))


def trajectory_label(key: tuple) -> str:
    return "/".join(map(str, key))
