"""dynorch_v1, WP-C: render ``report.tmpl.tex`` -> ``report.tex``, filling every ``{{path|fmt}}`` from ``numbers.json``
(contract section 7; copied from ``docs/forced_v2_report/render.py`` with a callable entry point for the test).

Paths are dot-separated keys into numbers.json (keys may contain '/', e.g. ``acc.H1/16.macro.acc``).
Formats: ``3`` -> 3 decimals, ``1`` -> 1 decimal, ``+1`` -> signed 1 decimal (typographic minus), ``a1`` ->
absolute value, ``d`` -> integer with thousands separators, ``$`` -> dollars with 2 decimals, ``%`` -> percent
with 1 decimal, ``tt`` -> typewriter string, ``s`` -> string. A placeholder that does not resolve, or resolves to
null, is an error (``RenderError``), so no number in the prose can be typed by hand and no placeholder can be
silently left blank.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PAT = re.compile(r"\{\{([^}|]+)(?:\|([^}]+))?\}\}")


class RenderError(ValueError):
    """At least one placeholder did not resolve."""


def resolve(N: Any, path: str) -> Any:
    cur = N
    for part in path.strip().split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(f"numbers.json has no {path!r} (stuck at {part!r})")
    return cur


def fmt(v: Any, spec: str | None) -> str:
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
    if spec.startswith("a"):
        return f"{abs(float(v)):.{int(spec[1:])}f}"
    if spec.startswith("+"):
        return f"{float(v):+.{int(spec[1:])}f}".replace("-", "$-$")
    return f"{float(v):.{int(spec)}f}"


def render_text(template: str, N: Any) -> str:
    """Fill every placeholder of ``template`` from ``N``; raise ``RenderError`` listing every failure."""
    errors: list[str] = []

    def sub(m):
        try:
            return fmt(resolve(N, m.group(1)), m.group(2))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{m.group(0)}: {exc}")
            return "??"
    out = PAT.sub(sub, template)
    if errors:
        raise RenderError("RENDER FAIL:\n  " + "\n  ".join(errors))
    return out


def render(numbers_path: Path | str = HERE / "numbers.json", template_path: Path | str = HERE / "report.tmpl.tex",
           out_path: Path | str | None = HERE / "report.tex") -> str:
    N = json.loads(Path(numbers_path).read_text())
    src = Path(template_path).read_text()
    out = render_text(src, N)
    if out_path is not None:
        Path(out_path).write_text(out)
    return out


def main() -> int:
    try:
        render()
    except RenderError as exc:
        print(exc)
        return 1
    print(f"rendered {len(PAT.findall((HERE / 'report.tmpl.tex').read_text()))} placeholders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
