#!/usr/bin/env python
"""Fail the build on LaTeX conditions that silently produce a wrong or missing PDF.

Overfull boxes are reported but do not fail: they are cosmetic and unavoidable in a
document with this many wide tables.

Usage:  check_log.py _build/main.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FATAL = [
    (r"Emergency stop", "LaTeX aborted"),
    (r"Output loop---\d+ consecutive dead cycles", "float placement deadlock"),
    (r"Undefined control sequence", "undefined macro"),
    (r"Missing \$ inserted", "unescaped maths character, usually an underscore"),
    (r"Misplaced \\noalign", "table row beginning with a bracket"),
    (r"Label `[^']+' multiply defined", "duplicate label"),
    (r"Citation `[^']+' undefined", "citation key missing from refs.bib"),
    (r"Reference `[^']+' undefined", "cross-reference target missing"),
    (r"File `[^']+' not found", "missing input file"),
    (r"Package longtable Error", "longtable error"),
]

WARN = [
    (r"Overfull \\hbox", "overfull horizontal box"),
    (r"Overfull \\vbox", "overfull vertical box"),
    (r"Underfull \\hbox", "underfull horizontal box"),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_log.py <logfile>", file=sys.stderr)
        return 2
    log = Path(sys.argv[1])
    if not log.exists():
        print(f"FAIL: {log} does not exist; the build did not run", file=sys.stderr)
        return 1
    text = log.read_text(errors="replace")

    failed = False
    for pattern, why in FATAL:
        hits = re.findall(pattern, text)
        if hits:
            failed = True
            print(f"FAIL ({len(hits)}x): {why}", file=sys.stderr)
            for line in text.splitlines():
                if re.search(pattern, line):
                    print(f"    {line.strip()[:160]}", file=sys.stderr)
                    break

    for pattern, why in WARN:
        n = len(re.findall(pattern, text))
        if n:
            print(f"note ({n}x): {why}")

    pdf = log.with_suffix(".pdf")
    if not pdf.exists():
        print(f"FAIL: {pdf} was not produced", file=sys.stderr)
        return 1
    m = re.search(r"Output written on .*\((\d+) pages?, (\d+) bytes\)", text)
    if m:
        print(f"PDF: {m.group(1)} pages, {int(m.group(2)) / 1e6:.1f} MB")

    if failed:
        return 1
    print("log clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
