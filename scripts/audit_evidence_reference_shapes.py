#!/usr/bin/env python3
"""Prove every evidence sealer verifies references in the shape it writes them.

The schema-5 sealers share I/O primitives but no evidence schema: each module states
the shape of a marker reference twice, once where it is written and once where it is
verified.  Nothing reconciles the two copies, so a reference gained a field on the
writing side while the verifying side kept a hand-written literal, and the r12 release
rejected its own freshly published r9 marker.

This audit re-derives both key sets from the source and fails when they disagree.  It
is a static check on purpose: it needs no fixture, covers sealers whose proofs are too
expensive to execute in a unit test, and applies unchanged to a sealer added later.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence


# ``_file_ref`` is defined once, in the r5 sealer, and inherited through the module
# chain.  Its key set is the contract every reference is measured against.
FILE_REFERENCE_KEYS = frozenset({"path", "sha256", "size", "mode", "link_count"})

# Constructors that return a complete file reference, mapped to the keys they yield.
REFERENCE_BUILDERS: dict[str, frozenset[str]] = {
    "_file_ref": FILE_REFERENCE_KEYS,
    "_recorder_transcript_ref": FILE_REFERENCE_KEYS | {"transcript_id"},
}

# A reference is recognised by these keys; a dict without them is ordinary payload.
REFERENCE_MARKERS = frozenset({"path", "sha256"})

WRITER_FUNCTIONS = frozenset(
    {"_execute_proof", "seal_failure", "_intent_payload", "_proof_contract", "_binding"}
)
VERIFIER_FUNCTIONS = frozenset({"_verify_proof", "verify_failure_seal"})


class ReferenceShapeError(RuntimeError):
    """A sealer's source cannot be parsed for reference shapes."""


@dataclass(frozen=True)
class Mismatch:
    """One field whose written and verified key sets differ."""

    module: str
    field: str
    writer_keys: frozenset[str]
    writer_lines: tuple[int, ...]
    verifier_keys: frozenset[str]
    verifier_lines: tuple[int, ...]

    @property
    def delta(self) -> tuple[str, ...]:
        return tuple(sorted(self.writer_keys ^ self.verifier_keys))

    def __str__(self) -> str:
        return (
            f"{self.module}: {self.field!r} written as {sorted(self.writer_keys)} "
            f"(line{'s' if len(self.writer_lines) > 1 else ''} "
            f"{', '.join(map(str, self.writer_lines))}) but verified as "
            f"{sorted(self.verifier_keys)} (line{'s' if len(self.verifier_lines) > 1 else ''} "
            f"{', '.join(map(str, self.verifier_lines))}); missing {list(self.delta)}"
        )


def _inferred_keys(node: ast.AST) -> frozenset[str] | None:
    """The key set an expression produces, or None when it cannot be inferred."""
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for key in node.keys:
            # ``**spread`` erases the literal key set.
            if key is None:
                return None
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None
            keys.add(key.value)
        return frozenset(keys)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return REFERENCE_BUILDERS.get(node.func.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _inferred_keys(node.left)
        right = _inferred_keys(node.right)
        if left is None or right is None:
            return None
        return left | right
    return None


def _looks_like_reference(keys: frozenset[str] | None) -> bool:
    return keys is not None and bool(keys & REFERENCE_MARKERS)


def _read_field(node: ast.AST) -> str | None:
    """The field name in ``proof.get("field")`` or ``marker["field"]``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _collect(tree: ast.Module) -> tuple[dict[str, set], dict[str, set]]:
    written: dict[str, set[tuple[frozenset[str], int]]] = {}
    verified: dict[str, set[tuple[frozenset[str], int]]] = {}

    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name in WRITER_FUNCTIONS:
            sink = written
        elif function.name in VERIFIER_FUNCTIONS:
            sink = verified
        else:
            continue

        for node in ast.walk(function):
            # Written form: ``"field": _file_ref(...)`` inside a payload dict.
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if not (
                        isinstance(key, ast.Constant) and isinstance(key.value, str)
                    ):
                        continue
                    keys = _inferred_keys(value)
                    if _looks_like_reference(keys):
                        sink.setdefault(key.value, set()).add((keys, value.lineno))
            # Verified form: ``proof.get("field") != {...}``.
            if isinstance(node, ast.Compare) and len(node.comparators) == 1:
                field = _read_field(node.left)
                if field is None:
                    continue
                expected = node.comparators[0]
                keys = _inferred_keys(expected)
                if _looks_like_reference(keys):
                    verified.setdefault(field, set()).add((keys, expected.lineno))
    return written, verified


def audit_source(source: str, *, module: str) -> list[Mismatch]:
    """Every field this module writes and verifies with differing key sets."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ReferenceShapeError(f"cannot parse {module}: {exc}") from exc

    written, verified = _collect(tree)
    mismatches: list[Mismatch] = []
    for field in sorted(set(written) & set(verified)):
        writer_shapes = {keys for keys, _ in written[field]}
        verifier_shapes = {keys for keys, _ in verified[field]}
        for writer_keys in sorted(writer_shapes, key=sorted):
            for verifier_keys in sorted(verifier_shapes, key=sorted):
                if writer_keys == verifier_keys:
                    continue
                mismatches.append(
                    Mismatch(
                        module=module,
                        field=field,
                        writer_keys=writer_keys,
                        writer_lines=tuple(
                            sorted(l for k, l in written[field] if k == writer_keys)
                        ),
                        verifier_keys=verifier_keys,
                        verifier_lines=tuple(
                            sorted(l for k, l in verified[field] if k == verifier_keys)
                        ),
                    )
                )
    return mismatches


def audit_path(path: Path) -> list[Mismatch]:
    return audit_source(path.read_text(encoding="utf-8"), module=path.name)


def sealer_sources(scripts_root: Path) -> list[Path]:
    """Every evidence sealer, including any release added after this audit."""
    return sorted(scripts_root.glob("seal_schema5_r*.py"))


def audit_all(scripts_root: Path) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for source in sealer_sources(scripts_root):
        mismatches.extend(audit_path(source))
    return mismatches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory holding the seal_schema5_r*.py modules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    sources: Iterable[Path] = sealer_sources(arguments.scripts_root)
    mismatches = audit_all(arguments.scripts_root)
    for mismatch in mismatches:
        print(f"MISMATCH: {mismatch}", file=sys.stderr)
    print(
        f"{len(list(sources))} sealer modules audited, {len(mismatches)} mismatches"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
