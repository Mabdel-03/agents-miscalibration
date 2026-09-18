"""AIME adapter: AIME 2025 (I + II) and AIME 2026 (I + II) from MathArena, 60 problems.

Sources: ``MathArena/aime_2025`` and ``MathArena/aime_2026`` at pinned revisions, each
``data/train-00000-of-00001.parquet`` (30 rows), read offline from the Hugging Face hub cache under ``$HF_HOME``.
Licence: CC BY-NC-SA 4.0 (MathArena dataset cards); problems (c) MAA.

Config keys consumed: ``benchmarks[id=aime].source`` (must be ``MathArena/aime_2025+MathArena/aime_2026``) and
``.split`` (``train``).

Preprocessing ``aime-prompt-v1``: see :mod:`agent_allocation.benchmarks._matharena`. Every gold is an integer in
[0, 999] rendered with ``str``. ``item_id = "aime/<year>/<problem_idx>"`` (problems 1-15 are contest I, 16-30 are
contest II). Strata: ``year`` (the 2026 snapshot has no ``problem_type`` column).
"""
from __future__ import annotations

from typing import Any, Mapping

from agent_allocation.benchmarks import _matharena
from agent_allocation.benchmarks._matharena import MathArenaSpec, YearSource
from agent_allocation.contracts import InventoryItem
from agent_allocation.paths import RunPaths

BENCHMARK = "aime"
SOURCE_REPO = "MathArena/aime_2025+MathArena/aime_2026"
SOURCE_SPLIT = "train"
SOURCES: tuple[YearSource, ...] = (
    YearSource(2025, "MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302",
               "9f9066ff48ad2e31f9bf1b1ac6d5e80693195f987985f2859f89dd25ffa51c2d", 30),
    YearSource(2026, "MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3",
               "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957", 30),
)
SOURCE_ROWS = 60
LICENSE = "CC BY-NC-SA 4.0 (MathArena/aime_2025, MathArena/aime_2026); problems (c) MAA"
PREPROCESSING_VERSION = "aime-prompt-v1"

#: Frozen pre-generation engineering exclusions ``{item_id: reason}``. Empty: every pinned gold parses and verifies
#: against itself under the pinned Math-Verify rule (recomputed by the text unit tests).
ENGINEERING_EXCLUSIONS: Mapping[str, str] = {}

SPEC = MathArenaSpec(benchmark=BENCHMARK, config_source=SOURCE_REPO, split=SOURCE_SPLIT, sources=SOURCES,
                     license=LICENSE, preprocessing_version=PREPROCESSING_VERSION, stratify_problem_type=False,
                     exclusions=ENGINEERING_EXCLUSIONS)


def source_record() -> dict[str, Any]:
    return _matharena.source_record(SPEC)


def build_item(row: Mapping[str, Any], *, source_revision: str | None = None) -> InventoryItem:
    return _matharena.build_item(SPEC, row, source_revision=source_revision)


def read_rows() -> list[dict[str, Any]]:
    return _matharena.read_rows(SPEC)


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    return _matharena.load_inventory(SPEC, cfg)


#: Dataset-lock provenance read by ``manifest.dataset_lock_payload`` (plain dict so it serializes as JSON).
SOURCE_PROVENANCE: dict[str, Any] = source_record()
