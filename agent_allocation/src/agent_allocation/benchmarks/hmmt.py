"""HMMT February adapter: HMMT February 2025 (30 problems) and 2026 (33 problems) from MathArena, 63 problems.

Sources: ``MathArena/hmmt_feb_2025`` and ``MathArena/hmmt_feb_2026`` at pinned revisions, each
``data/train-00000-of-00001.parquet`` (30 and 33 rows), read offline from the Hugging Face hub cache under
``$HF_HOME``. Licence: CC BY-NC-SA 4.0 (MathArena dataset cards); problems (c) HMMT.

Config keys consumed: ``benchmarks[id=hmmt].source`` (must be ``MathArena/hmmt_feb_2025+MathArena/hmmt_feb_2026``)
and ``.split`` (``train``).

Preprocessing ``hmmt-prompt-v1``: see :mod:`agent_allocation.benchmarks._matharena`. Golds are LaTeX strings kept
verbatim; 33 of 63 are not integers (fractions, radicals, ``\\pi`` expressions, ``3^{2025}``, ``2^{25} \\cdot 26!``, one
two-root list ``hmmt/2025/10`` and one degree answer ``74^\\circ`` at ``hmmt/2026/27``). ``item_id =
"hmmt/<year>/<problem_idx>"``. Strata: ``year`` x first ``problem_type`` (Algebra, Combinatorics, Geometry, Number
Theory; six items list two types and are stratified by the first).
"""
from __future__ import annotations

from typing import Any, Mapping

from agent_allocation.benchmarks import _matharena
from agent_allocation.benchmarks._matharena import MathArenaSpec, YearSource
from agent_allocation.contracts import InventoryItem
from agent_allocation.paths import RunPaths

BENCHMARK = "hmmt"
SOURCE_REPO = "MathArena/hmmt_feb_2025+MathArena/hmmt_feb_2026"
SOURCE_SPLIT = "train"
SOURCES: tuple[YearSource, ...] = (
    YearSource(2025, "MathArena/hmmt_feb_2025", "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
               "ccca64d248b89dc0765e40b893fe4ebeab242226a4336027d95d256886a3d5f2", 30),
    YearSource(2026, "MathArena/hmmt_feb_2026", "02fba4f74d8e68e73e66a02d540fd979c05c274c",
               "e5fcff6b1c2262841c0c37bf6d7b42529f284528d5f0d5c45c52e8bf0654a916", 33),
)
SOURCE_ROWS = 63
LICENSE = "CC BY-NC-SA 4.0 (MathArena/hmmt_feb_2025, MathArena/hmmt_feb_2026); problems (c) HMMT"
PREPROCESSING_VERSION = "hmmt-prompt-v1"

#: Frozen pre-generation engineering exclusions ``{item_id: reason}``. Empty: every pinned gold parses and verifies
#: against itself under the pinned Math-Verify rule (recomputed by the text unit tests).
ENGINEERING_EXCLUSIONS: Mapping[str, str] = {}

SPEC = MathArenaSpec(benchmark=BENCHMARK, config_source=SOURCE_REPO, split=SOURCE_SPLIT, sources=SOURCES,
                     license=LICENSE, preprocessing_version=PREPROCESSING_VERSION, stratify_problem_type=True,
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
