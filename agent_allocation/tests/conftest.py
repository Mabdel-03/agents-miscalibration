"""Fixtures for the whole suite."""
from __future__ import annotations

import os

import pytest

REHEARSAL = "ALLOCATION_REHEARSAL"


def _restore(value: str | None) -> None:
    if value is None:
        os.environ.pop(REHEARSAL, None)
    else:
        os.environ[REHEARSAL] = value


@pytest.fixture(autouse=True, scope="module")
def _rehearsal_ends_with_its_module():
    """``--rehearsal`` sets ALLOCATION_REHEARSAL for the whole process. That is right for a CLI run, but a test that calls
    the CLI in-process would leave every later test rehearsing: paid-dispatch gates relaxed, and no mock transport reached.
    This also covers module-scoped fixtures, which the per-test fixture below must leave alone."""
    before = os.environ.get(REHEARSAL)
    yield
    _restore(before)


@pytest.fixture(autouse=True)
def _rehearsal_ends_with_its_test():
    before = os.environ.get(REHEARSAL)
    yield
    _restore(before)
