"""Shared fixtures for the text-benchmark tests. Real-data tests skip when the pinned offline sources are absent;
everything else runs on synthetic rows."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_allocation.benchmarks import aime, gpqa_diamond, hmmt, omni_math_hard
from agent_allocation.benchmarks.text_manifest import hf_snapshot_file
from agent_allocation.config import load_config
from agent_allocation.contracts import Candidate, OutputKind, ParseStatus
from agent_allocation.paths import RunPaths

ROOT = Path(__file__).resolve().parents[3]
CANARY = "CANARY-GOLD-9f1c2e7a"


def _matharena_available(module) -> bool:
    return all(hf_snapshot_file(s.repo, s.revision, s.file).is_file() for s in module.SOURCES)


def aime_available() -> bool:
    return _matharena_available(aime)


def hmmt_available() -> bool:
    return _matharena_available(hmmt)


def omni_available() -> bool:
    return hf_snapshot_file(omni_math_hard.SOURCE_REPO, omni_math_hard.SOURCE_REVISION,
                            omni_math_hard.SOURCE_FILE).is_file()


def gpqa_available() -> bool:
    return hf_snapshot_file(gpqa_diamond.SOURCE_REPO, gpqa_diamond.SOURCE_REVISION,
                            gpqa_diamond.SOURCE_FILE).is_file()


needs_aime = pytest.mark.skipif(not aime_available(), reason="pinned AIME snapshots not in HF cache")
needs_hmmt = pytest.mark.skipif(not hmmt_available(), reason="pinned HMMT snapshots not in HF cache")
needs_omni = pytest.mark.skipif(not omni_available(), reason="pinned Omni-MATH snapshot not in HF cache")
needs_gpqa = pytest.mark.skipif(not gpqa_available(), reason="pinned GPQA snapshot not in HF cache")


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "configs" / "forced27b.json")


@pytest.fixture
def run_paths(tmp_path: Path) -> RunPaths:
    return RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")


@pytest.fixture(scope="session")
def session_paths(tmp_path_factory: pytest.TempPathFactory) -> RunPaths:
    base = tmp_path_factory.mktemp("text_run")
    return RunPaths.for_run(base / "run", local_state_dir=base / "local")


def _inventory(module, available, what, cfg, session_paths):
    if not available():
        pytest.skip(f"pinned {what} source not in HF cache")
    return module.load_inventory(cfg, session_paths)


@pytest.fixture(scope="session")
def aime_inventory(cfg, session_paths):
    return _inventory(aime, aime_available, "AIME", cfg, session_paths)


@pytest.fixture(scope="session")
def hmmt_inventory(cfg, session_paths):
    return _inventory(hmmt, hmmt_available, "HMMT", cfg, session_paths)


@pytest.fixture(scope="session")
def omni_inventory(cfg, session_paths):
    return _inventory(omni_math_hard, omni_available, "Omni-MATH", cfg, session_paths)


@pytest.fixture(scope="session")
def gpqa_inventory(cfg, session_paths):
    return _inventory(gpqa_diamond, gpqa_available, "GPQA", cfg, session_paths)


def make_candidate(item_id: str, final: str | None, kind: OutputKind, *,
                   status: ParseStatus = ParseStatus.VALID, node: str = "ab" * 32) -> Candidate:
    return Candidate(candidate_id=node, node_id=node, item_id=item_id,
                     visible_content="" if final is None else f"<final>{final}</final>", final_answer=final,
                     rationale=None, parse_status=status, finish_reason="stop", output_kind=kind)
