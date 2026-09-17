#!/usr/bin/env python
"""Assert every number quoted in the report prose traces back to numbers.json.

Numbers that are part of the study design rather than a measurement (item counts, budget levels,
team sizes, years) are declared in DESIGN below so that a genuinely unsupported figure stands out.
"""
from __future__ import annotations
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Design constants and typographic numbers that are not measurements.
DESIGN = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "14", "15", "16", "17", "20",
    "24", "25", "26", "32", "48", "70", "95", "96", "100", "192", "400", "1024", "2048", "4096",
    "2026", "0.7", "0.95", "0.02", "0.5", "1.0", "1.000", "0.000",
}


def prose_lines(path):
    out = []
    for ln in open(path):
        s = ln.strip()
        if s.startswith("%") or s.startswith("\\usepackage") or s.startswith("\\documentclass"):
            continue
        if s.startswith("\\setlength") or s.startswith("\\includegraphics") or s.startswith("\\captionsetup"):
            continue
        if s.startswith("\\renewcommand") or s.startswith("\\vspace") or s.startswith("\\input"):
            continue
        s = s.replace("{,}", "").replace(",", " ")
        out.append(s)
    return out


def known_values(N):
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
        elif isinstance(o, (int, float)):
            for d in (0, 1, 2, 3, 4, 6):
                vals.add(f"{abs(float(o)):.{d}f}")
            # Accuracy points: numbers.json stores fractions, prose often states points.
            for d in (0, 1, 2):
                vals.add(f"{abs(float(o)) * 100:.{d}f}")
            vals.add(str(abs(o)))
    walk(N)
    return {v.rstrip(".") for v in vals}


def main():
    N = json.load(open(f"{HERE}/numbers.json"))
    known = known_values(N) | DESIGN
    bad = []
    for i, ln in enumerate(prose_lines(f"{HERE}/report.tex"), 1):
        for m in re.finditer(r"(?<![A-Za-z0-9_.])(\d+(?:[.,]\d+)?)(?![0-9]*[A-Za-z])", ln):
            tok = m.group(1).replace(",", "")
            if tok in known or tok.rstrip("0").rstrip(".") in known:
                continue
            bad.append((i, tok, ln[:90]))
    if bad:
        print("UNTRACED NUMBERS IN PROSE:")
        for i, tok, ln in bad:
            print(f"  line {i}: {tok!r}  in  {ln}")
        return 1
    print(f"check_numbers: every quoted number traces to numbers.json ({len(known)} known values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
