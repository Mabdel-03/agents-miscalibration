"""Deterministic Markdown/HTML rendering and table export helpers (no timestamps, stable float formatting)."""
from __future__ import annotations

import html
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "unknown"
        return f"{value:.{digits}f}"
    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass
    return str(value)


def md_table(frame: pd.DataFrame | Sequence[dict[str, Any]], columns: Sequence[str] | None = None,
             headers: Sequence[str] | None = None, digits: int | Mapping[str, int] = 4,
             max_rows: int | None = None) -> str:
    """``digits`` is one precision for every float cell or a ``{column: precision}`` map (default 4)."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(list(frame))
    if frame.empty:
        return "_(no rows)_\n"
    columns = list(columns or frame.columns)
    headers = list(headers or columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    shown = frame if max_rows is None else frame.head(max_rows)
    for _, row in shown.iterrows():
        cells = [fmt(row[c], digits.get(c, 4) if isinstance(digits, Mapping) else digits).replace("|", "\\|")
                 if c in row else "" for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"\n_{len(frame) - max_rows} more rows in the exported table._")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_table(frame: pd.DataFrame, tables_dir: Path, name: str) -> list[Path]:
    """Write ``name.parquet`` and ``name.csv`` with deterministic content."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True)
    csv_path = tables_dir / f"{name}.csv"
    parquet_path = tables_dir / f"{name}.parquet"
    tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
    frame.to_csv(tmp_csv, index=False, lineterminator="\n", float_format="%.12g")
    os.replace(tmp_csv, csv_path)
    tmp_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
    _parquet_safe(frame).to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, parquet_path)
    return [parquet_path, csv_path]


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            values = [v for v in out[col] if v is not None and not (isinstance(v, float) and math.isnan(v))]
            kinds = {type(v) for v in values}
            if kinds and not kinds <= {bool} and not kinds <= {int, float} and not kinds <= {str}:
                out[col] = [None if v is None else str(v) for v in out[col]]
    return out


_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = _IMAGE.sub(lambda m: f'<img alt="{m.group(1)}" src="{m.group(2)}">', out)
    out = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    out = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    return out.replace("\\|", "|")


def markdown_to_html(markdown: str, title: str) -> str:
    """Minimal converter for the Markdown this package emits (headings, tables, bullets, paragraphs, images)."""
    body: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            cells = [_split_row(r) for r in block if not re.fullmatch(r"\|(-+\|)+", r.replace(" ", ""))]
            if cells:
                body.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
                            + "</tr></thead><tbody>")
                for row in cells[1:]:
                    body.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
                body.append("</tbody></table>")
            continue
        if line.startswith("- "):
            body.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                body.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            body.append("</ul>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ")):
            para.append(lines[i])
            i += 1
        body.append(f"<p>{_inline(' '.join(para))}</p>")
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;"
             "color:#1f2328;background:#fff}table{border-collapse:collapse;font-size:.85rem;margin:.5rem 0 1rem}"
             "th,td{border:1px solid #d0d7de;padding:.2rem .45rem;text-align:left}th{background:#f6f8fa}"
             "img{max-width:100%}code{background:#f6f8fa;padding:0 .2rem}")
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>" + html.escape(title) + "</title><style>"
            + style + "</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n")


def _split_row(row: str) -> list[str]:
    inner = row.strip()[1:-1]
    parts = re.split(r"(?<!\\)\|", inner)
    return [p.strip() for p in parts]


def bullet_list(items: Iterable[str]) -> str:
    items = list(items)
    return "\n".join(f"- {item}" for item in items) + "\n" if items else "- none\n"
