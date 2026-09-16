"""Shared fixtures for ``tests/nextdyn`` (plan §6).

* ``tmp_run_root`` — a run root with the nextdyn directory skeleton (cells, registers,
  plan/root_teams, pools/<pool>/servers, tables, protected).
* ``study_cfg`` / ``compute_cfg`` / ``analysis_cfg`` — the committed configs, ``study.yaml`` in
  ``mode: test`` (R=2) and ``analysis.yaml``'s ``test`` profile.
* ``stub_tokenizer`` / ``real_tokenizer_32b`` / ``register_fake_server`` / ``study_config`` —
  re-registered from ``tests/study/conftest.py`` so WP1 can drive the fake vLLM server here.
* HF is forced offline exactly as ``tests/study/conftest.py`` does (importing it does so).
* ``pytest_collection_modifyitems`` records every collected test name on the pytest config
  (attribute ``nextdyn_collected_test_names``) so ``test_spec18_index.py`` can report which
  spec-18 names are not yet collected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tests.study.conftest import (  # noqa: E402,F401  (re-registered fixtures)
    StubTokenizer,
    real_tokenizer_32b,
    register_fake_server,
    stub_tokenizer,
    study_config,
)

from agents_scaling.nextdyn import config as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "nextdyn"
RUN_ROOT_SUBDIRS = (
    "cells", "registers", "plan/root_teams", "plan/manifests", "pools/deadline/servers",
    "pools/low/servers", "pools/audit/servers", "tables", "protected", "requests", "status",
)

COLLECTED_ATTR = "nextdyn_collected_test_names"


def pytest_collection_modifyitems(session, config, items):
    names = getattr(config, COLLECTED_ATTR, None)
    if names is None:
        names = set()
        setattr(config, COLLECTED_ATTR, names)
    for item in items:
        names.add(getattr(item, "originalname", None) or item.name)


@pytest.fixture
def tmp_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run_root"
    for name in RUN_ROOT_SUBDIRS:
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture(scope="session")
def study_cfg() -> C.StudyConfig:
    return C.load_study(CONFIG_DIR / "study.yaml", mode="test")


@pytest.fixture(scope="session")
def compute_cfg() -> C.ComputeConfig:
    return C.load_compute(CONFIG_DIR / "compute.yaml")


@pytest.fixture(scope="session")
def analysis_cfg() -> C.AnalysisConfig:
    return C.load_analysis(CONFIG_DIR / "analysis.yaml", profile="test")
