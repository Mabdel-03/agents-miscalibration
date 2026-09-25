"""Render report.tmpl.tex -> report.tex, filling every {{path|fmt}} from numbers.json.

Paths are dot-separated keys into numbers.json (keys may contain '/', e.g. ``acc.macro.deep/16.acc``).
Formats: ``3`` -> 3 decimals, ``1`` -> 1 decimal, ``+1`` -> signed 1 decimal (typographic minus), ``d`` ->
integer with thousands separators, ``$`` -> dollars with 2 decimals, ``%`` -> percent with 1 decimal, ``tt`` ->
typewriter string, ``s`` -> string. A placeholder that does not resolve is an error, so
no number in the prose can be typed by hand.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(r"\{\{([^}|]+)(?:\|([^}]+))?\}\}")


def resolve(N, path):
    cur = N
    for part in path.strip().split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(f"numbers.json has no {path!r} (stuck at {part!r})")
    return cur


def fmt(v, spec):
    if v is None:
        raise ValueError("value is null")
    spec = (spec or "s").strip()
    if spec == "s":
        return str(v).replace("_", r"\_")
    if spec == "d":
        return f"{int(round(float(v))):,}"
    if spec == "$":
        return f"\\${float(v):,.2f}"
    if spec == "tt":
        return "\\texttt{" + str(v).replace("_", r"\_") + "}"
    if spec == "%":
        return f"{100 * float(v):.1f}\\%"
    if spec.startswith("a"):                           # absolute value ("falls 23.0 points short")
        return f"{abs(float(v)):.{int(spec[1:])}f}"
    if spec.startswith("+"):                           # signed, with a typographic minus
        return f"{float(v):+.{int(spec[1:])}f}".replace("-", "$-$")
    return f"{float(v):.{int(spec)}f}"


def main():
    N = json.loads((HERE / "numbers.json").read_text())
    src = (HERE / "report.tmpl.tex").read_text()
    errors = []

    def sub(m):
        try:
            return fmt(resolve(N, m.group(1)), m.group(2))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{m.group(0)}: {exc}")
            return "??"
    out = PAT.sub(sub, src)
    if errors:
        print("RENDER FAIL:\n  " + "\n  ".join(errors))
        sys.exit(1)
    (HERE / "report.tex").write_text(out)
    print(f"rendered {len(PAT.findall(src))} placeholders")


if __name__ == "__main__":
    main()
