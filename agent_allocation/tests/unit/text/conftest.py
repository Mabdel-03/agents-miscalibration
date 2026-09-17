"""Shared fixtures for the text-benchmark tests (agent 04). Real-data tests skip when the pinned offline sources
are absent; everything else runs on synthetic rows."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_allocation.benchmarks import bbh_logic4, math500, mmlu_pro
from agent_allocation.benchmarks.text_manifest import hf_snapshot_file
from agent_allocation.config import load_config
from agent_allocation.contracts import Candidate, OutputKind, ParseStatus
from agent_allocation.paths import RunPaths

ROOT = Path(__file__).resolve().parents[3]
CANARY = "CANARY-GOLD-9f1c2e7a"


def math500_available() -> bool:
    return hf_snapshot_file(math500.SOURCE_REPO, math500.SOURCE_REVISION, math500.SOURCE_FILE).is_file()


def mmlu_pro_available() -> bool:
    return hf_snapshot_file(mmlu_pro.SOURCE_REPO, mmlu_pro.SOURCE_REVISION, mmlu_pro.SOURCE_FILE).is_file()


def bbh_available() -> bool:
    return all((bbh_logic4.shared_cache_dir() / bbh_logic4.file_name(s)).is_file() for s in bbh_logic4.SOURCE_FILES)


needs_math500 = pytest.mark.skipif(not math500_available(), reason="pinned MATH-500 snapshot not in HF cache")
needs_mmlu_pro = pytest.mark.skipif(not mmlu_pro_available(), reason="pinned MMLU-Pro snapshot not in HF cache")
needs_bbh = pytest.mark.skipif(not bbh_available(), reason="pinned BBH files not in the shared cache")


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "configs" / "balanced150.json")


@pytest.fixture
def run_paths(tmp_path: Path) -> RunPaths:
    return RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")


@pytest.fixture(scope="session")
def session_paths(tmp_path_factory: pytest.TempPathFactory) -> RunPaths:
    base = tmp_path_factory.mktemp("text_run")
    return RunPaths.for_run(base / "run", local_state_dir=base / "local")


@pytest.fixture(scope="session")
def math_inventory(cfg, session_paths):
    if not math500_available():
        pytest.skip("pinned MATH-500 snapshot not in HF cache")
    return math500.load_inventory(cfg, session_paths)


@pytest.fixture(scope="session")
def mmlu_inventory(cfg, session_paths):
    if not mmlu_pro_available():
        pytest.skip("pinned MMLU-Pro snapshot not in HF cache")
    return mmlu_pro.load_inventory(cfg, session_paths)


@pytest.fixture(scope="session")
def bbh_inventory(cfg, session_paths):
    if not bbh_available():
        pytest.skip("pinned BBH files not in the shared cache")
    return bbh_logic4.load_inventory(cfg, session_paths)


def make_candidate(item_id: str, final: str | None, kind: OutputKind, *,
                   status: ParseStatus = ParseStatus.VALID, node: str = "ab" * 32) -> Candidate:
    return Candidate(candidate_id=node, node_id=node, item_id=item_id,
                     visible_content="" if final is None else f"<final>{final}</final>", final_answer=final,
                     rationale=None, parse_status=status, finish_reason="stop", output_kind=kind)
