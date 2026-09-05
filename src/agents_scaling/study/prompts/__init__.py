"""Frozen prompt artifacts (§5.5 "freeze the clauses literally", P0-2, amendment S1b).

``templates/*.txt`` are byte-exact copies of the handoff templates plus the five authored
files (``dec_root_truthful``, ``hle_judge``, ``cen_final_instruction``,
``dec_revision_contract``, ``dec_revision_contract_n1``).  ``framing_clauses.v4_orcd.json``
is the handoff clause file with the S1b ``CODE_SELECTOR`` wording.  ``PROMPT_HASHES.json``
pins every artifact; ``config.freeze`` copies the hashes into ``FROZEN.yaml`` and
``test_prompts.py`` recomputes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

PROMPTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PROMPTS_DIR / "templates"
FRAMING_CLAUSES_FILE = PROMPTS_DIR / "framing_clauses.v4_orcd.json"
HASHES_FILE = PROMPTS_DIR / "PROMPT_HASHES.json"

TEMPLATE_NAMES: tuple[str, ...] = (
    "independent_root",
    "focal_revision",
    "central_hub",
    "central_worker",
    "common_consumer",
    "judge_best",
    "forecast",
    "rlm_root",
    "rlm_child",
    "transport_mediator",
    "dec_root_truthful",
    "hle_judge",
    "cen_final_instruction",
    "dec_revision_contract",
    "dec_revision_contract_n1",
)
FRAMING_KEYS: tuple[str, ...] = (
    "TEAM_0",
    "TEAM_1",
    "VOTE_0",
    "VOTE_1",
    "HLE_CHOICE_SELECTOR",
    "CODE_SELECTOR",
)


def template_path(name: str) -> Path:
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"unknown prompt template {name!r}")
    return TEMPLATES_DIR / f"{name}.txt"


def load_template(name: str) -> str:
    """Exact template text (no stripping; trailing bytes are part of the frozen artifact)."""
    return template_path(name).read_bytes().decode("utf-8")


def load_framing_clauses() -> dict[str, str]:
    data = json.loads(FRAMING_CLAUSES_FILE.read_text(encoding="utf-8"))
    missing = [key for key in FRAMING_KEYS if key not in data]
    if missing:
        raise KeyError(f"framing clauses missing {missing}")
    return {key: str(data[key]) for key in FRAMING_KEYS}


def _artifact_files() -> list[Path]:
    """Every frozen prompt artifact: ``templates/*.txt`` and ``*.json`` except the hash file."""
    files = sorted(TEMPLATES_DIR.glob("*.txt"))
    files += sorted(p for p in PROMPTS_DIR.glob("*.json") if p.name != HASHES_FILE.name)
    return files


def compute_prompt_hashes() -> dict[str, str]:
    """Recompute sha256 per artifact keyed by path relative to ``prompts/``."""
    return {
        path.relative_to(PROMPTS_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _artifact_files()
    }


def prompt_hashes() -> Mapping[str, str]:
    """The pinned hashes from ``PROMPT_HASHES.json`` (raise if they no longer match)."""
    pinned = json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    current = compute_prompt_hashes()
    if pinned != current:
        changed = sorted(set(pinned) ^ set(current) | {k for k in pinned if pinned.get(k) != current.get(k)})
        raise RuntimeError(f"PROMPT_HASHES.json is stale for {changed}; prompts are frozen artifacts")
    return pinned


def write_prompt_hashes() -> Path:
    """(Re)generate ``PROMPT_HASHES.json`` — only legitimate before the freeze."""
    HASHES_FILE.write_text(json.dumps(compute_prompt_hashes(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HASHES_FILE
