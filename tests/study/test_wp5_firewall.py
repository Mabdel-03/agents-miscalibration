"""T14 / P1-4: the protected-data firewall as an import-graph assertion.

No module outside ``study/evaluation/`` imports ``agents_scaling.study.data.protected``;
no non-evaluation module (other than the WP1 writer side: ``data/layout.py``,
``data/export.py``, ``data/protected.py`` itself) mentions a ``protected`` path; and the
public export rows carry none of ``answer|json|test|canonical_solution``."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agents_scaling.study.data.public import PROTECTED_KEYS, load_public_tasks
from tests.study.wp5_support import make_tasks, write_export

STUDY_DIR = Path(__file__).resolve().parents[2] / "src" / "agents_scaling" / "study"
WRITER_SIDE = {"data/layout.py", "data/export.py", "data/protected.py"}
_PATH_RE = re.compile(r"(^|/)protected(/|$)")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _path_literals(path: Path) -> list[str]:
    """String constants (docstrings excluded) that look like a ``protected`` path segment."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if _PATH_RE.search(node.value):
                hits.append(node.value)
    return hits


def test_import_graph_firewall():
    offenders = []
    path_hits = []
    for path in sorted(STUDY_DIR.rglob("*.py")):
        rel = path.relative_to(STUDY_DIR).as_posix()
        if rel.startswith("evaluation/"):
            continue
        if "agents_scaling.study.data.protected" in _imports(path):
            offenders.append(rel)
        if rel not in WRITER_SIDE:
            text = path.read_text(encoding="utf-8")
            if "protected_dir(" in text or _path_literals(path):
                path_hits.append(rel)
    assert offenders == [], f"modules importing the protected reader: {offenders}"
    assert path_hits == [], f"modules touching a protected path: {path_hits}"
    # the evaluator identity is the only importer
    importers = [p.relative_to(STUDY_DIR).as_posix() for p in (STUDY_DIR / "evaluation").glob("*.py") if "agents_scaling.study.data.protected" in _imports(p)]
    assert importers, "evaluation/ must import the protected reader"


def test_public_rows_lack_protected_keys(tmp_path: Path):
    run_root = tmp_path / "rr"
    write_export(run_root, make_tasks(3, "dev"))
    tasks = load_public_tasks(run_root)
    assert len(tasks) == 6
    for task in tasks:
        assert not (set(task.to_dict()) & PROTECTED_KEYS)
        assert not (set(task.to_dict()) & {"answer", "json", "test", "canonical_solution"})
