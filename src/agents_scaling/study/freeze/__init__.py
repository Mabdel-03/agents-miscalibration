"""Freeze registers copied into ``<run_root>/freeze/`` by ``config.freeze`` (P1-9, §11.5).

* ``amendments.json`` — the 02_spec_fidelity_audit.md §1 register (F1…X1) verbatim, plus
  the critic's additions E3', N9B1, N1-SRS, E4b, M1b, C1b.
* ``prior_exposure_manifest.json`` — amendment P3: which datasets earlier harness runs
  touched and when HLE-Verified / BigCodeBench were first downloaded.
"""

from __future__ import annotations

from pathlib import Path

FREEZE_DIR = Path(__file__).resolve().parent
AMENDMENTS_PATH = FREEZE_DIR / "amendments.json"
PRIOR_EXPOSURE_PATH = FREEZE_DIR / "prior_exposure_manifest.json"
