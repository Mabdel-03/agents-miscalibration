"""Omni-MATH (difficulty >= 7) adapter: olympiad-level problems with a single final answer.

Source: ``KbsdJames/Omni-MATH`` at the pinned revision, ``test.jsonl`` (4,428 rows), read offline from the Hugging
Face hub cache under ``$HF_HOME``. Licence: Apache-2.0.

Config keys consumed: ``benchmarks[id=omni_math_hard].source`` (must be the pinned repo), ``.split`` (``test``) and
``.min_difficulty`` (must equal the pinned ``MIN_DIFFICULTY = 7.0``).

Population: rows with ``difficulty >= 7.0`` (916 of 4,428). Lower-difficulty rows are outside the benchmark, not
excluded items. Rows gain one derived field, ``row_index`` (0-based line number among the non-empty lines of the
pinned file), because the source has no id column. ``item_id = "omni_math_hard/<row_index>"``.

Preprocessing ``omni_math_hard-prompt-v1`` (frozen): ``PublicTask.prompt`` is ``problem`` with leading/trailing ASCII
whitespace removed. Public fields: problem only. Private: ``answer`` (gold, verbatim) plus ``solution``,
``difficulty``, ``domain`` and ``source`` inside the hashed private record. Strata: the second level of the first
``domain`` label (``Mathematics -> <domain> -> ...``; every row's top level is ``Mathematics``).

Engineering exclusions (pre-registered, pre-generation, never outcome-based), first match wins:

1. ``empty_problem`` / ``empty_gold_answer``.
2. ``figure_not_included``: the problem refers to a figure/diagram/picture/image (``FIGURE_REFERENCE``) and carries
   no ``[asy]`` source for it.
3. ``gold_prose``: the gold contains words beyond mathematics -- any ``\\text``-like argument with a run of two or
   more letters, a run of three or more letters outside LaTeX commands that is not a standard function name
   (``MATH_WORDS``), or a two-letter English connective (``or``, ``if``, ``is``, ...). This removes proofs, yes/no
   and true/false verdicts, sentence answers and answers with side conditions in words.
4. Math-Verify rules, computed by :func:`audit_gold` with the pinned grader and frozen in
   ``ENGINEERING_EXCLUSIONS`` (the unit tests recompute them): ``gold_unparseable`` (no mathematical object under
   ``math-parse-v3``), ``gold_partial_parse`` (the gold names variables but its parse has no free symbol and no
   function application, i.e. only a numeric fragment of it was read), ``gold_self_verify_failed`` (the gold, bare
   or boxed, does not grade correct against itself).
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from agent_allocation.benchmarks.text_manifest import (
    SourceIntegrityError,
    benchmark_config,
    hf_snapshot_file,
    read_verified,
)
from agent_allocation.contracts import (
    EvalKind,
    InventoryItem,
    OutputKind,
    PrivateEvalSpec,
    PublicTask,
    canonical_hash,
)
from agent_allocation.paths import RunPaths

BENCHMARK = "omni_math_hard"
SOURCE_REPO = "KbsdJames/Omni-MATH"
SOURCE_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
SOURCE_SPLIT = "test"
SOURCE_FILE = "test.jsonl"
SOURCE_SHA256 = "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168"
SOURCE_ROWS = 4428
MIN_DIFFICULTY = 7.0
POPULATION_ROWS = 916
LICENSE = "Apache-2.0 (KbsdJames/Omni-MATH)"
PREPROCESSING_VERSION = "omni_math_hard-prompt-v1"
SOURCE_FIELDS = ("domain", "difficulty", "problem", "solution", "answer", "source")
OUTER_WHITESPACE = " \t\n\r\f\v"

FIGURE_REFERENCE = re.compile(
    r"\b(?:figure|diagram|picture|image)s?\s+(?:below|above)\b"
    r"|\b(?:shown|drawn|depicted|pictured)\s+(?:below|above|in\s+the\s+(?:figure|diagram|picture))\b"
    r"|\b(?:see|in)\s+the\s+(?:figure|diagram|picture)\b"
    r"|\[img\]|\\includegraphics", re.IGNORECASE)
TEXT_COMMAND = re.compile(r"\\(?:text|textbf|textit|textrm|mbox|mathrm|operatorname)\s*\{([^{}]*)\}")
BLACKBOARD = re.compile(r"\\(?:mathbb|mathcal|mathfrak)\s*\{?\s*[A-Za-z]\s*\}?")
LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
WORD3 = re.compile(r"[A-Za-z]{3,}")
CONNECTIVE = re.compile(r"\b(?:or|if|is|of|to|as|at|by|on|an|no|be|it|in)\b", re.IGNORECASE)
MATH_WORDS = frozenset({"sin", "cos", "tan", "cot", "sec", "csc", "log", "exp", "max", "min", "gcd", "lcm", "det",
                        "deg", "mod", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "inf", "sup", "lim"})
CONSTANT_LETTERS = frozenset({"e", "i"})

#: Frozen Math-Verify exclusions ``{item_id: reason}`` over items that pass rules 1-3 (recomputed by the text unit
#: tests with :func:`audit_gold`; regenerate only together with a new PREPROCESSING_VERSION).
ENGINEERING_EXCLUSIONS: Mapping[str, str] = {
    "omni_math_hard/7": "gold_partial_parse",
    "omni_math_hard/55": "gold_partial_parse",
    "omni_math_hard/61": "gold_partial_parse",
    "omni_math_hard/114": "gold_partial_parse",
    "omni_math_hard/125": "gold_unparseable",
    "omni_math_hard/171": "gold_partial_parse",
    "omni_math_hard/252": "gold_partial_parse",
    "omni_math_hard/307": "gold_partial_parse",
    "omni_math_hard/593": "gold_partial_parse",
    "omni_math_hard/1390": "gold_partial_parse",
    "omni_math_hard/1676": "gold_unparseable",
    "omni_math_hard/2407": "gold_unparseable",
    "omni_math_hard/2646": "gold_partial_parse",
    "omni_math_hard/3191": "gold_partial_parse",
    "omni_math_hard/3631": "gold_partial_parse",
    "omni_math_hard/3816": "gold_self_verify_failed",
    "omni_math_hard/3817": "gold_unparseable",
    "omni_math_hard/3857": "gold_unparseable",
    "omni_math_hard/3916": "gold_unparseable",
    "omni_math_hard/3948": "gold_partial_parse",
    "omni_math_hard/3964": "gold_partial_parse",
    "omni_math_hard/4007": "gold_unparseable",
    "omni_math_hard/4012": "gold_partial_parse",
    "omni_math_hard/4110": "gold_partial_parse",
    "omni_math_hard/4117": "gold_partial_parse",
    "omni_math_hard/4119": "gold_unparseable",
    "omni_math_hard/4154": "gold_partial_parse",
    "omni_math_hard/4175": "gold_partial_parse",
    "omni_math_hard/4196": "gold_unparseable",
    "omni_math_hard/4212": "gold_partial_parse",
    "omni_math_hard/4213": "gold_unparseable",
    "omni_math_hard/4242": "gold_partial_parse",
    "omni_math_hard/4244": "gold_unparseable",
    "omni_math_hard/4253": "gold_partial_parse",
    "omni_math_hard/4305": "gold_unparseable",
    "omni_math_hard/4311": "gold_unparseable",
    "omni_math_hard/4403": "gold_partial_parse",
    "omni_math_hard/4406": "gold_unparseable",
}


def source_record() -> dict[str, Any]:
    return {"benchmark": BENCHMARK, "repo": SOURCE_REPO, "revision": SOURCE_REVISION, "split": SOURCE_SPLIT,
            "file": SOURCE_FILE, "sha256": SOURCE_SHA256, "rows": SOURCE_ROWS, "min_difficulty": MIN_DIFFICULTY,
            "population_rows": POPULATION_ROWS, "license": LICENSE, "preprocessing_version": PREPROCESSING_VERSION,
            "item_id_rule": "row_index = 0-based position among non-empty lines of test.jsonl"}


def trim_outer(text: str) -> str:
    return text.strip(OUTER_WHITESPACE)


def is_prose(gold: str) -> bool:
    """Rule 3: the gold carries words beyond mathematics."""
    for match in TEXT_COMMAND.finditer(gold):
        if re.search(r"[A-Za-z]{2,}", match.group(1)):
            return True
    bare = LATEX_COMMAND.sub(" ", TEXT_COMMAND.sub(" ", gold))
    if any(word.lower() not in MATH_WORDS for word in WORD3.findall(bare)):
        return True
    return bool(CONNECTIVE.search(bare))


def figure_missing(problem: str) -> bool:
    """Rule 2: the problem points at a figure it does not include."""
    return bool(FIGURE_REFERENCE.search(problem)) and "[asy]" not in problem


def identifier_letters(gold: str) -> set[str]:
    """ASCII letters the gold uses as names (outside commands, ``\\text`` and blackboard sets; minus e and i)."""
    bare = LATEX_COMMAND.sub(" ", BLACKBOARD.sub(" ", TEXT_COMMAND.sub(" ", gold)))
    return set(re.findall(r"[A-Za-z]", bare)) - CONSTANT_LETTERS


def domain_of(row: Mapping[str, Any]) -> str:
    domains = row.get("domain") or []
    if not domains:
        return "none"
    parts = [part.strip() for part in str(domains[0]).split("->")]
    return parts[1] if len(parts) > 1 else parts[0]


def rule_reason(row: Mapping[str, Any]) -> str | None:
    """Rules 1-3 (pure text rules applied at build time)."""
    problem = row["problem"] if isinstance(row["problem"], str) else ""
    gold = row["answer"] if isinstance(row["answer"], str) else ""
    if not trim_outer(problem):
        return "empty_problem"
    if not gold.strip():
        return "empty_gold_answer"
    if figure_missing(problem):
        return "figure_not_included"
    if is_prose(gold):
        return "gold_prose"
    return None


def audit_gold(gold: str, *, timeout_s: float = 10.0) -> str | None:
    """Rule 4 with the pinned grader; must run on a process's main thread. ``None`` means the gold is usable."""
    from agent_allocation.grading.text import math_equiv
    from agent_allocation.grading.text._limits import wall_clock_limit

    for form in (gold, "\\boxed{" + gold + "}"):
        verdict = math_equiv.verify_gold(gold, form, timeout_s=timeout_s)
        if verdict.code == "gold_unparseable":
            return "gold_unparseable"
        if verdict.correct is not True:
            return "gold_self_verify_failed"
    math_equiv._math_verify()
    with wall_clock_limit(timeout_s):
        parsed = math_equiv.parse_answer(gold)
    if not parsed or any(isinstance(value, str) for value in parsed):
        return "gold_unparseable"
    if identifier_letters(gold):
        from sympy.core.function import AppliedUndef

        def names_something(value: Any) -> bool:
            if getattr(value, "free_symbols", None):
                return True
            atoms = getattr(value, "atoms", None)
            return bool(atoms and atoms(AppliedUndef))

        if not any(names_something(value) for value in parsed):
            return "gold_partial_parse"
    return None


def build_item(row: Mapping[str, Any], *, source_revision: str = SOURCE_REVISION) -> InventoryItem:
    missing = [name for name in (*SOURCE_FIELDS, "row_index") if name not in row]
    if missing:
        raise SourceIntegrityError(f"Omni-MATH row lacks fields {missing}")
    index = row["row_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise SourceIntegrityError(f"Omni-MATH row_index {index!r} is not a nonnegative int")
    item_id = f"{BENCHMARK}/{index}"
    problem = row["problem"] if isinstance(row["problem"], str) else ""
    gold = row["answer"] if isinstance(row["answer"], str) else None
    public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                        prompt=trim_outer(problem), output_kind=OutputKind.MATH)
    private_record = {
        "benchmark": BENCHMARK, "item_id": item_id, "source_revision": source_revision,
        "preprocessing_version": PREPROCESSING_VERSION, "eval_kind": EvalKind.MATH_VERIFY.value,
        "gold_answer": gold, "source_record": {"row_index": index, **{name: row[name] for name in SOURCE_FIELDS}},
    }
    private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                              eval_kind=EvalKind.MATH_VERIFY, gold_answer=gold, private_record_ref=None,
                              private_hash=canonical_hash(private_record))
    reason = rule_reason(row) or ENGINEERING_EXCLUSIONS.get(item_id)
    return InventoryItem(public=public, private=private, strata=(("domain", domain_of(row)),),
                         eligible=reason is None, exclusion_reason=reason)


def read_rows() -> list[dict[str, Any]]:
    """Every row of the pinned file (all difficulties), each with its ``row_index``."""
    path = hf_snapshot_file(SOURCE_REPO, SOURCE_REVISION, SOURCE_FILE)
    data = read_verified(path, SOURCE_SHA256)
    lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
    rows = [{**json.loads(line), "row_index": index} for index, line in enumerate(lines)]
    if len(rows) != SOURCE_ROWS:
        raise SourceIntegrityError(f"Omni-MATH: {len(rows)} rows, pinned {SOURCE_ROWS}")
    return rows


def population(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard = [row for row in rows if float(row["difficulty"]) >= MIN_DIFFICULTY]
    if len(hard) != POPULATION_ROWS:
        raise SourceIntegrityError(f"Omni-MATH: {len(hard)} rows at difficulty >= {MIN_DIFFICULTY}, "
                                   f"pinned {POPULATION_ROWS}")
    return hard


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    entry = benchmark_config(cfg, BENCHMARK, source=SOURCE_REPO)
    if entry.get("split") != SOURCE_SPLIT:
        raise ValueError(f"{BENCHMARK}: config split {entry.get('split')!r} != pinned {SOURCE_SPLIT!r}")
    if float(entry.get("min_difficulty", MIN_DIFFICULTY)) != MIN_DIFFICULTY:
        raise ValueError(f"{BENCHMARK}: config min_difficulty {entry.get('min_difficulty')!r} != pinned "
                         f"{MIN_DIFFICULTY}")
    items = [build_item(row) for row in population(read_rows())]
    ids = [item.public.item_id for item in items]
    unknown = set(ENGINEERING_EXCLUSIONS) - set(ids)
    if unknown:
        raise SourceIntegrityError(f"Omni-MATH: exclusions name unknown items {sorted(unknown)}")
    shadowed = [item.public.item_id for item in items
                if item.public.item_id in ENGINEERING_EXCLUSIONS and item.exclusion_reason
                != ENGINEERING_EXCLUSIONS[item.public.item_id]]
    if shadowed:
        raise SourceIntegrityError(f"Omni-MATH: frozen exclusions shadowed by rules 1-3: {shadowed[:3]}")
    return items


#: Dataset-lock provenance read by ``manifest.dataset_lock_payload`` (plain dict so it serializes as JSON).
SOURCE_PROVENANCE: dict[str, Any] = source_record()
