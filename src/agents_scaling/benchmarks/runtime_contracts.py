"""Run-scoped access to frozen, verified benchmark Questions.

Operational readers must not reconstruct scientific truth from whichever dataset happens
to be available when they run.  :class:`VerifiedQuestionCatalog` verifies the immutable
benchmark-contract sidecar up front and lazily verifies each unique normalized Question
sequence against that sidecar.  Cells sharing the same benchmark/count/seed contract
reuse the already verified immutable tuple.
"""

from __future__ import annotations

from pathlib import Path

from agents_scaling.benchmarks.contracts import (
    BenchmarkLoader,
    BenchmarkContractError,
    BenchmarkContractKey,
    FrozenBenchmarkContracts,
    load_frozen_benchmark_contracts,
    load_verified_questions,
)
from agents_scaling.benchmarks.loaders import load_benchmark
from agents_scaling.benchmarks.schema import Question
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest


class VerifiedQuestionCatalog:
    """Verified Question contracts for one immutable run manifest.

    Constructing the catalog fails closed if either sidecar file is absent, malformed,
    checksum-invalid, or detached from the supplied manifest.  ``questions_for`` then
    checks current normalized benchmark content exactly once per unique frozen contract.
    """

    def __init__(
        self,
        run_root: str | Path,
        *,
        snapshot: ManifestSnapshot | None = None,
        benchmark_loader: BenchmarkLoader = load_benchmark,
    ) -> None:
        self.run_root = Path(run_root)
        self.snapshot = load_manifest(self.run_root) if snapshot is None else snapshot
        self.frozen: FrozenBenchmarkContracts = load_frozen_benchmark_contracts(
            self.run_root,
            snapshot=self.snapshot,
        )
        self._manifest_cells = {cell.cell_id: cell for cell in self.snapshot.cells}
        self._questions_by_contract: dict[str, tuple[Question, ...]] = {}
        self._benchmark_loader = benchmark_loader

    @property
    def sidecar_sha256(self) -> str:
        return self.frozen.sidecar_sha256

    def verify_unchanged(self) -> None:
        """Fail if the frozen sidecar changed after this catalog was constructed."""

        observed = load_frozen_benchmark_contracts(
            self.run_root,
            snapshot=self.snapshot,
        )
        if observed.sidecar_sha256 != self.frozen.sidecar_sha256:
            raise BenchmarkContractError(
                f"benchmark contract changed during operation for {self.run_root}: "
                f"expected {self.frozen.sidecar_sha256}, got {observed.sidecar_sha256}"
            )

    def questions_for(self, cell: ExperimentCell) -> tuple[Question, ...]:
        """Return Questions only after exact manifest and frozen-content verification."""

        manifest_cell = self._manifest_cells.get(cell.cell_id)
        if manifest_cell is None or manifest_cell != cell:
            raise BenchmarkContractError(
                f"cell {cell.cell_id!r} is not the exact cell in the frozen manifest "
                f"{self.snapshot.path}"
            )
        contract_id = BenchmarkContractKey.from_cell(cell).contract_id
        cached = self._questions_by_contract.get(contract_id)
        if cached is not None:
            return cached
        questions, entry = load_verified_questions(
            self.run_root,
            manifest_cell,
            snapshot=self.snapshot,
            benchmark_loader=self._benchmark_loader,
        )
        if entry.get("contract_id") != contract_id:
            raise BenchmarkContractError(
                f"verified benchmark contract id mismatch for {cell.cell_id}"
            )
        self._questions_by_contract[contract_id] = questions
        return questions
