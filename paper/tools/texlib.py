"""LaTeX emission helpers.  This is the only place ``DataFrame.to_latex`` is called.

Three pandas-3 behaviours make a naive call fatal for this document:

1. ``escape`` defaults to ``None``, so level names such as ``single_agent`` and
   ``mmlu_pro`` reach the .tex verbatim and pdflatex reports ``Missing $ inserted``.
2. ``escape=True`` escapes the *column headers* too, so a maths header such as
   ``$n_{\\mathrm{des}}$`` becomes ``\\$n\\_\\{\\textbackslash mathrm...``.
3. A row whose first cell begins with ``[`` is read as ``\\\\[<dimen>]``, producing
   ``Misplaced \\noalign``.  Confidence intervals are formatted exactly ``[lo, hi]``,
   so this triggers whenever an interval column leads a table.

The contract here is therefore: escape data cells ourselves, hand in headers that are
already LaTeX, call ``to_latex(escape=False)``, then repair the leading-bracket case.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_ESC = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Raw tokens that appear as data values, mapped to their display form.  Values are
# already LaTeX and are not escaped again.
PRETTY = {
    "single_agent": "single agent",
    "independent": "independent",
    "decentralized": "decentralized",
    "centralized": "centralized",
    "artifact_only": "artifact only",
    "plus_intermediate": "plus intermediate",
    "plus_cot": "plus CoT",
    "gpqa": "GPQA",
    "mmlu_pro": "MMLU-Pro",
    "truthfulqa": "TruthfulQA",
    "math": "MATH",
    "model_size": "model size",
    "reasoning_level": "reasoning budget",
    "context_share_level": "context sharing",
    "prompt_complexity_level": "prompt complexity",
    "n_agents": "agent count",
    "topology": "topology",
    "accuracy": "accuracy",
    "pa_ece_prim": "per-agent ECE",
    "vote_ece_prim": "vote ECE",
    "fp_ece_prim": "final-producer ECE",
    "delta_vote_prim": r"$\Delta$ECE (vote)",
    "delta_fp_prim": r"$\Delta$ECE (fp)",
    "pa_signed_gap": "per-agent signed gap",
    "vote_signed_gap": "vote signed gap",
    "mean_total_tokens": "mean total tokens",
    "mean_reasoning_tokens": "mean reasoning tokens",
    "cost": "cost proxy",
    "Ec": r"$E_c$",
    "Ae": r"$A_e$",
    "Opct": r"$O\%$",
    "off": "off",
    "b512": "b512",
    "b2048": "b2048",
    "b8192": "b8192",
    "unlimited": "unlimited",
    "all_systems": "all systems",
    "all core rows": "all core rows",
    "True": "yes",
    "False": "no",
}


def esc(x) -> str:
    """Escape an arbitrary value for use in a LaTeX cell."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return "".join(_ESC.get(c, c) for c in str(x))


def pretty(x) -> str:
    """Display form for a raw level token; anything unknown is escaped."""
    s = str(x)
    if s in PRETTY:
        return PRETTY[s]
    # Composite tokens such as "topology:decentralized" or "3-agent family - 1-agent single".
    if ":" in s:
        head, _, tail = s.partition(":")
        if head in PRETTY or tail in PRETTY:
            return f"{PRETTY.get(head, esc(head))}: {PRETTY.get(tail, esc(tail))}"
    for raw, disp in PRETTY.items():
        if "_" in raw and raw in s:
            s = s.replace(raw, disp)
    return esc(s)


def num(x, d: int = 3, signed: bool = False, na: str = "--") -> str:
    if x is None:
        return na
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if not np.isfinite(v):
        return na
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def intfmt(x, na: str = "--") -> str:
    if x is None:
        return na
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if not np.isfinite(v):
        return na
    return f"{int(round(v)):,}".replace(",", r"\,")


def ci(est, lo, hi, d: int = 3, signed: bool = False) -> list[str]:
    """Format a triple of columns as ``est [lo, hi]``."""
    out = []
    for a, b, c in zip(est, lo, hi):
        if a is None or (isinstance(a, float) and not np.isfinite(a)):
            out.append("--")
            continue
        out.append(f"{num(a, d, signed)} [{num(b, d)}, {num(c, d)}]")
    return out


def sig(lo, hi) -> list[str]:
    """Star a contrast whose interval excludes zero."""
    out = []
    for b, c in zip(lo, hi):
        try:
            b, c = float(b), float(c)
        except (TypeError, ValueError):
            out.append("")
            continue
        out.append(r"$\ast$" if np.isfinite(b) and np.isfinite(c) and (b > 0 or c < 0) else "")
    return out


_FIXES = [
    (r"\cline", r"\cmidrule(lr)"),
    ("Continued on next page", r"\emph{continued on next page}"),
]

MANIFEST: dict[str, dict] = {}


def emit(
    df: pd.DataFrame,
    path,
    *,
    caption: str,
    label: str,
    column_format: str,
    longtable: bool = False,
    note: str | None = None,
    fontsize: str = r"\small",
    source: str = "",
) -> Path:
    """Write ``df`` as a booktabs table.  ``df`` must already hold display strings and
    LaTeX-ready column headers."""
    path = Path(path)
    tex = df.to_latex(
        index=False,
        escape=False,
        longtable=longtable,
        column_format=column_format,
        caption=caption,
        label=label,
        na_rep="--",
    )
    for a, b in _FIXES:
        tex = tex.replace(a, b)
    # A row starting with "[" is otherwise parsed as \\[<dimen>].
    tex = re.sub(r"(\\\\\s*\n)\[", r"\1{[}", tex)

    if note:
        # threeparttable is not installed in this TeX tree; a centred minipage gives
        # the same visual result without the dependency.
        note_tex = (
            "\n\\vspace{-0.6em}\n\\begin{center}\\begin{minipage}{0.94\\linewidth}\n"
            f"\\footnotesize\\textit{{Note.}} {note}\n\\end{{minipage}}\\end{{center}}\n"
        )
        tex = tex + note_tex

    if not longtable:
        # adjustbox cannot wrap a longtable, but for a plain tabular it is the
        # simplest guarantee that a wide table does not run into the margin.
        tex = re.sub(
            r"(\\begin\{tabular\})",
            r"\\adjustbox{max width=\\linewidth}{%\n\\begin{tabular}",
            tex,
            count=1,
        )
        tex = re.sub(r"(\\end\{tabular\})", r"\\end{tabular}}", tex, count=1)

    # Tighter inter-column padding: several of these tables are a few points wider
    # than the text block at the default 6pt.
    body = f"{{{fontsize}\\setlength{{\\tabcolsep}}{{4pt}}\n{tex}}}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)

    MANIFEST[path.stem] = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "caption": caption,
        "label": label,
        "longtable": longtable,
        "source": source,
    }
    return path


def lint(directory) -> list[str]:
    """Report unescaped underscores outside maths in generated .tex files."""
    problems = []
    for p in sorted(Path(directory).glob("*.tex")):
        txt = p.read_text()
        # Strip inline maths before looking for bare underscores.
        stripped = re.sub(r"\$[^$]*\$", "", txt)
        for i, line in enumerate(stripped.splitlines(), 1):
            if re.search(r"(?<!\\)_", line):
                problems.append(f"{p.name}:{i}: unescaped underscore: {line.strip()[:90]}")
            # A doubled backslash before an escape is a double-escaping bug: LaTeX reads
            # it as a line break followed by a bare underscore.
            if re.search(r"\\\\[_&%#$]", line):
                problems.append(f"{p.name}:{i}: double-escaped character: {line.strip()[:90]}")
            if "textbackslash" in line and "\\_" in line:
                problems.append(f"{p.name}:{i}: escaped backslash next to escape: {line.strip()[:90]}")
    return problems
