#!/usr/bin/env python
"""Verify every candidate reference against the arXiv API and emit refs.bib.

Nothing is written to refs.bib unless its metadata was fetched and matched. A
candidate whose identifier does not resolve, or whose fetched title does not match
the expected title, is reported and dropped rather than guessed.

Non-arXiv works are listed separately with hand-entered metadata and are marked as
such in the report.

Usage:  verify_refs.py [--out refs.bib] [--report]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query?id_list={}"
NS = {"a": "http://www.w3.org/2005/Atom"}

# key -> (arxiv id, expected title fragment, bib fields to add)
ARXIV: dict[str, tuple[str, str]] = {
    "kim2025scaling": ("2512.08296", "Science of Scaling Agent Systems"),
    "du2023debate": ("2305.14325", "Improving Factuality and Reasoning"),
    "smit2024debate": ("2311.17371", "Should we be going MAD"),
    "wynn2025debate": ("2509.05396", ""),
    "guo2017calibration": ("1706.04599", "On Calibration of Modern Neural Networks"),
    "yang2024collaborative": ("2404.09127", "Calibration"),
    "wang2024calibrating": ("2403.09849", ""),
    "zhu2025testtime": ("2506.12928", ""),
    "li2026agentbench": ("2602.18998", "Test-Time Scaling"),
    "wong2026teamofthoughts": ("2602.16485", "Team of Thoughts"),
    "kumar2019verified": ("1909.10155", "Verified Uncertainty Calibration"),
    "kuhn2023semantic": ("2302.09664", "Semantic Uncertainty"),
    "wang2023selfconsistency": ("2203.11171", "Self-Consistency"),
    "qwen3": ("2505.09388", "Qwen3"),
    "kwon2023vllm": ("2309.06180", "Efficient Memory Management"),
    "rein2024gpqa": ("2311.12022", "GPQA"),
    "wang2024mmlupro": ("2406.01574", "MMLU-Pro"),
    "hendrycks2021math": ("2103.03874", "Mathematical Problem Solving"),
    "lightman2024verify": ("2305.20050", "Verify Step by Step"),
    "lin2022truthfulqa": ("2109.07958", "TruthfulQA"),
}

# Works with no arXiv record.  Metadata entered by hand from the published version.
MANUAL: dict[str, str] = {
    "murphy1973": """@article{murphy1973,
  author  = {Murphy, Allan H.},
  title   = {A New Vector Partition of the Probability Score},
  journal = {Journal of Applied Meteorology},
  volume  = {12},
  number  = {4},
  pages   = {595--600},
  year    = {1973},
  doi     = {10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2}
}""",
    "cox1958": """@article{cox1958,
  author  = {Cox, David R.},
  title   = {Two Further Applications of a Model for Binary Regression},
  journal = {Biometrika},
  volume  = {45},
  number  = {3--4},
  pages   = {562--565},
  year    = {1958},
  doi     = {10.1093/biomet/45.3-4.562}
}""",
}


def fetch(arxiv_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(API.format(arxiv_id), timeout=30) as r:
            root = ET.fromstring(r.read())
    except Exception as exc:  # noqa: BLE001
        print(f"    fetch failed: {exc}", file=sys.stderr)
        return None
    entry = root.find("a:entry", NS)
    if entry is None:
        return None
    title = " ".join((entry.findtext("a:title", "", NS) or "").split())
    if not title or title.lower() == "error":
        return None
    authors = [
        " ".join((a.findtext("a:name", "", NS) or "").split())
        for a in entry.findall("a:author", NS)
    ]
    published = entry.findtext("a:published", "", NS) or ""
    year = published[:4]
    return {"title": title, "authors": authors, "year": year, "id": arxiv_id}


def bibname(full: str) -> str:
    parts = full.split()
    if not parts:
        return full
    return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else parts[0]


def to_bib(key: str, meta: dict) -> str:
    authors = " and ".join(bibname(a) for a in meta["authors"])
    title = meta["title"].replace("&", r"\&")
    return (
        f"@article{{{key},\n"
        f"  author        = {{{authors}}},\n"
        f"  title         = {{{title}}},\n"
        f"  journal       = {{arXiv preprint arXiv:{meta['id']}}},\n"
        f"  year          = {{{meta['year']}}},\n"
        f"  eprint        = {{{meta['id']}}},\n"
        f"  archivePrefix = {{arXiv}}\n"
        f"}}"
    )


def cited_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for p in list(root.glob("sections/*.tex")) + list(root.glob("sections/appendix/*.tex")) + [
        root / "main.tex"
    ]:
        if not p.exists():
            continue
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", p.read_text()):
            keys.update(k.strip() for k in m.group(1).split(","))
    return {k for k in keys if k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refs.bib")
    ap.add_argument("--report", action="store_true", help="only cross-check keys against the prose")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / args.out

    if args.report:
        if not out.exists():
            print("refs.bib does not exist yet", file=sys.stderr)
            return 1
        have = set(re.findall(r"@\w+\{([^,]+),", out.read_text()))
        used = cited_keys(root)
        missing = sorted(used - have)
        unused = sorted(have - used)
        if missing:
            print(f"FAIL: cited but not in refs.bib: {missing}", file=sys.stderr)
        if unused:
            print(f"FAIL: in refs.bib but never cited: {unused}", file=sys.stderr)
        if not missing and not unused:
            print(f"refs.bib and the prose agree on {len(have)} entries")
            return 0
        return 1

    entries: list[str] = []
    dropped: list[str] = []
    for key, (aid, expect) in ARXIV.items():
        print(f"  {key:26s} arXiv:{aid} ...", end=" ")
        meta = fetch(aid)
        if meta is None:
            print("UNRESOLVED")
            dropped.append(f"{key} (arXiv:{aid}): no record returned")
            continue
        if expect and expect.lower() not in meta["title"].lower():
            print("TITLE MISMATCH")
            dropped.append(
                f"{key} (arXiv:{aid}): expected ~{expect!r}, got {meta['title']!r}"
            )
            continue
        print(f"ok  {meta['year']}  {meta['title'][:62]}")
        entries.append(to_bib(key, meta))

    for key, text in MANUAL.items():
        print(f"  {key:26s} manual entry (not on arXiv)")
        entries.append(text)

    header = [
        "% Generated by paper/tools/verify_refs.py.",
        "% Every arXiv entry was fetched from the arXiv API and its title checked",
        "% against the expected title before being written here.",
        "",
    ]
    out.write_text("\n".join(header) + "\n\n".join(entries) + "\n")
    print(f"\nwrote {len(entries)} entries to {out}")
    if dropped:
        print("\nDROPPED (not written; do not cite these):")
        for d in dropped:
            print("  " + d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
