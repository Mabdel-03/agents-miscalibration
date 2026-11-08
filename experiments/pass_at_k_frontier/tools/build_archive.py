#!/usr/bin/env python3
"""Rebuild the curated, public copy of the Pass@K frontier studies from the cluster run roots.

The run roots hold every prompt and transcript, so they cannot be copied wholesale into a public repository: they
contain verbatim benchmark questions, including GPQA-Diamond, whose maintainers ask that it stay out of public
corpora. This script copies an explicit allow-list and then enforces one rule on the result:

    no file in the public tree may contain eight consecutive words from any benchmark question.

- Table cells that break the rule (a model answer that restates its question) become
  ``[withheld-benchmark-text sha256:<16 hex>]``. The hash is of the original cell, so joins on that column still work.
  Rows and numeric columns are never touched.
- A listed file that is verbatim prompt text is left out and recorded in WITHHELD below.
- Any other non-table file that breaks the rule stops the build. It must be reviewed, not silently dropped.

Everything is recorded in REDACTIONS.csv. Originals stay in the run roots on the cluster (see RAW_DATA.md).

    python build_archive.py --out experiments/pass_at_k_frontier    # full rebuild
    python build_archive.py --out experiments/pass_at_k_frontier --manifest-only
"""
from __future__ import annotations

import argparse
import collections
import csv
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(10**9)

BASE = "/orcd/data/tpoggio/001/mabdel03"
RESULTS = f"{BASE}/agents_scaling_results"
MAIN = f"{BASE}/agents_scaling"
ALLOC = f"{RESULTS}/openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1"
PILOT = f"{RESULTS}/forced_deep_pilot_v1"
FORCED = f"{RESULTS}/forced_budget_v2/artifacts"
DYN = f"{RESULTS}/dynorch_v1/artifacts"
DYN_ROOT = f"{RESULTS}/dynorch_v1"
PASSK = f"{RESULTS}/passk_cont_v1"
DYN_REPORT = f"{BASE}/agents_scaling_dynorch_v1/agent_allocation/docs/dynorch_report"
PACKAGE = f"{MAIN}/docs/passk_cont/results_package/passk_cont_v1_results"

# A CSV larger than this is dropped in favour of its Parquet twin, so the clone stays small.
CSV_MAX_BYTES = 10 * 1024 * 1024
HARD_MAX_BYTES = 95 * 1024 * 1024  # GitHub refuses files over 100 MB

MARKER = "[withheld-benchmark-text sha256:{}]"
MANIFEST_DIRS = [f"{ALLOC}/manifests/public", f"{FORCED}/manifests/public", f"{PASSK}/artifacts/manifests/public"]

# Verbatim prompts: left out of the public tree, present in the results package on the cluster.
WITHHELD = {
    "05_passk_cont_v1/results_package/prompts/examples/solver_think_prompt_aime.txt": "rendered prompt containing an AIME problem",
    "05_passk_cont_v1/results_package/prompts/examples/solver_think_prompt_gpqa_diamond.txt": "rendered prompt containing a GPQA-Diamond question",
    "05_passk_cont_v1/results_package/prompts/examples/solver_think_prompt_hmmt.txt": "rendered prompt containing an HMMT problem",
    "05_passk_cont_v1/results_package/prompts/examples/solver_think_prompt_omni_math_hard.txt": "rendered prompt containing an Omni-MATH problem",
    "05_passk_cont_v1/results_package/prompts/llm_selectors/e8a_example_A0.txt": "rendered selector prompt containing a benchmark problem",
    "05_passk_cont_v1/results_package/prompts/llm_selectors/e8a_example_A1.txt": "rendered selector prompt containing a benchmark problem",
    "05_passk_cont_v1/results_package/prompts/llm_selectors/e8b_example_B0.txt": "rendered selector prompt containing a benchmark problem",
    "05_passk_cont_v1/results_package/prompts/llm_selectors/e8b_example_B1.txt": "rendered selector prompt containing a benchmark problem",
}
RAW_NOTE = "the results package on the cluster (RAW_DATA.md)"

# Generated or hand-written files in the destination that the rebuild must not delete.
KEEP = ("README.md", "RAW_DATA.md", ".gitignore", "tools/*", "*/README.md")

TOKEN = re.compile(r"[a-z0-9]+")
N = 8
MIN_CELL_CHARS = 16  # eight words cannot fit in less


def shingles(text: str):
    tokens = TOKEN.findall(text.lower())
    for i in range(max(0, len(tokens) - N + 1)):
        window = tokens[i:i + N]
        # Require some words: a run of digits matches problem statements and figure tick labels alike. Single letters
        # count, because math phrasing ("let x be a ...") is exactly what must be caught.
        if sum(1 for t in window if t.isalpha()) >= 4:
            yield " ".join(window)


def build_fingerprint() -> set[str]:
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for root in MANIFEST_DIRS:
        for path in glob.glob(f"{root}/*.jsonl"):
            with open(path, errors="replace") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("item_id") and rec.get("prompt"):
                        for s in shingles(rec["prompt"]):
                            owners[s].add(rec["item_id"])
    # A phrase shared by many items is template boilerplate, not a question.
    return {s for s, items in owners.items() if len(items) <= 5}


FINGERPRINT: set[str] = set()


def has_benchmark_text(text: str) -> bool:
    return any(s in FINGERPRINT for s in shingles(text))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Builder:
    def __init__(self, out: str):
        self.out = out
        self.sources: dict[str, str] = {}
        self.dropped_csv: list[str] = []
        self.redactions: list[dict] = []

    def add(self, src: str, rel: str) -> None:
        if rel in WITHHELD:
            return
        dest = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        self.sources[rel] = src

    def add_opt(self, src: str, rel: str) -> None:
        if os.path.exists(src):
            self.add(src, rel)

    def add_glob(self, src_dir: str, pattern: str, rel_dir: str, exclude: tuple[str, ...] = ()) -> None:
        for path in sorted(glob.glob(os.path.join(src_dir, pattern), recursive=True)):
            if not os.path.isfile(path):
                continue
            name = os.path.relpath(path, src_dir)
            if any(fnmatch.fnmatch(name, e) for e in exclude):
                continue
            self.add(path, os.path.join(rel_dir, name))

    def add_tables(self, src_dir: str, rel_dir: str) -> None:
        for path in sorted(glob.glob(f"{src_dir}/*")):
            name = os.path.basename(path)
            if not os.path.isfile(path) or "backup" in name:
                continue
            # A table with no rows was written as a one-byte CSV, which crashes read_csv.
            size = os.path.getsize(path)
            if name.endswith(".csv") and (size > CSV_MAX_BYTES or size <= 1):
                twin = path[:-4] + ".parquet"
                if not os.path.exists(twin):
                    raise SystemExit(f"{path} would be dropped but has no Parquet twin")
                self.dropped_csv.append(os.path.join(rel_dir, name))
                continue
            self.add(path, os.path.join(rel_dir, name))

    # ---- enforcing the rule -----------------------------------------------------------------------------------------
    def enforce(self) -> None:
        offenders = []
        for rel in sorted(self.sources):
            path = os.path.join(self.out, rel)
            ext = os.path.splitext(rel)[1].lower()
            if ext == ".csv":
                self._redact_csv(rel, path)
            elif ext == ".parquet":
                self._redact_parquet(rel, path)
            elif ext in (".md", ".txt", ".json", ".jsonl", ".tex", ".log", ".sh", ".py"):
                with open(path, errors="replace") as fh:
                    if has_benchmark_text(fh.read()):
                        offenders.append(rel)
        if offenders:
            raise SystemExit("benchmark text in non-table files, review before publishing:\n  " + "\n  ".join(offenders))

    def _record(self, rel, column, count, before, path):
        self.redactions.append({"file": rel, "column": column, "cells_withheld": count,
                                "original_sha256": before, "public_sha256": ""})

    def _redact_csv(self, rel, path) -> None:
        before = sha256(path)
        with open(path, newline="", errors="surrogateescape") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return
        counts: collections.Counter = collections.Counter()
        for row in rows[1:]:
            for i, cell in enumerate(row[:len(rows[0])]):
                if len(cell) >= MIN_CELL_CHARS and has_benchmark_text(cell):
                    digest = hashlib.sha256(cell.encode("utf-8", "surrogateescape")).hexdigest()[:16]
                    row[i] = MARKER.format(digest)
                    counts[rows[0][i]] += 1
        if not counts:
            return
        with open(path, "w", newline="", errors="surrogateescape") as fh:
            csv.writer(fh, lineterminator="\n").writerows(rows)
        for column, count in sorted(counts.items()):
            self._record(rel, column, count, before, path)

    def _redact_parquet(self, rel, path) -> None:
        before = sha256(path)
        table = pq.read_table(path)
        counts: collections.Counter = collections.Counter()
        for index, field in enumerate(table.schema):
            if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
                continue
            values = table.column(index).to_pylist()
            changed = False
            for i, cell in enumerate(values):
                if cell is not None and len(cell) >= MIN_CELL_CHARS and has_benchmark_text(cell):
                    values[i] = MARKER.format(hashlib.sha256(cell.encode("utf-8", "surrogatepass")).hexdigest()[:16])
                    counts[field.name] += 1
                    changed = True
            if changed:
                table = table.set_column(index, field, pa.array(values, type=field.type))
        if not counts:
            return
        pq.write_table(table, path, compression="zstd")
        for column, count in sorted(counts.items()):
            self._record(rel, column, count, before, path)

    # ---- generated indexes ------------------------------------------------------------------------------------------
    def write_schemas(self) -> None:
        for rel_dir in ("01_allocation_v1/tables", "03_forced_budget_v2/tables", "04_dynorch_v1/tables",
                        "05_passk_cont_v1/scores", "05_passk_cont_v1/sealed"):
            root = os.path.join(self.out, rel_dir)
            lines = [f"# Table schemas: `{rel_dir}`", "", "Generated by `tools/build_archive.py`.", ""]
            for path in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
                meta = pq.read_metadata(path)
                schema = pq.read_schema(path)
                name = os.path.relpath(path, root)
                lines += [f"## {name}", "", f"{meta.num_rows:,} rows, {meta.num_columns} columns.", ""]
                lines += [f"- `{f.name}`: {f.type}" for f in schema]
                lines.append("")
            with open(os.path.join(root, "SCHEMA.md"), "w") as fh:
                fh.write("\n".join(lines))

    def write_redactions(self) -> None:
        for row in self.redactions:
            row["public_sha256"] = sha256(os.path.join(self.out, row["file"]))
        with open(os.path.join(self.out, "REDACTIONS.csv"), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["kind", "file", "column", "cells_withheld", "original_sha256", "public_sha256", "reason"])
            for row in sorted(self.redactions, key=lambda r: (r["file"], r["column"])):
                w.writerow(["cells", row["file"], row["column"], row["cells_withheld"], row["original_sha256"],
                            row["public_sha256"], "cell restates a benchmark question"])
            for rel, reason in sorted(WITHHELD.items()):
                w.writerow(["file", rel, "", "", "", "", f"{reason}; original in {RAW_NOTE}"])
            for rel in sorted(self.dropped_csv):
                w.writerow(["csv-dropped", rel, "", "", "", "", "over 10 MB or empty; the Parquet twin holds the same data"])


def write_manifest(out: str) -> None:
    rows = []
    for dp, dn, fn in os.walk(out):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), out)
            if rel == "MANIFEST.csv":
                continue
            rows.append((rel, os.path.getsize(os.path.join(dp, f)), sha256(os.path.join(dp, f))))
    with open(os.path.join(out, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(sorted(rows))
    too_big = [r for r in rows if r[1] > HARD_MAX_BYTES]
    if too_big:
        raise SystemExit(f"files too large for GitHub: {too_big}")
    print(f"manifest: {len(rows)} files, {sum(r[1] for r in rows) / 1048576:.1f} MB")


def clean(out: str) -> None:
    """Delete only previously generated data; never the hand-written cards or tools."""
    for dp, _, fn in os.walk(out):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), out)
            if not any(fnmatch.fnmatch(rel, k) for k in KEEP):
                os.remove(os.path.join(dp, f))


def populate(b: Builder) -> None:
    # 01: allocation v1 (Gemini 2.5 Flash-Lite, soft budgets)
    s = "01_allocation_v1"
    b.add(f"{MAIN}/docs/allocation_v1/report/report.pdf", f"{s}/report/report.pdf")
    b.add(f"{MAIN}/docs/allocation_v1/report/numbers.json", f"{s}/report/numbers.json")
    b.add(f"{MAIN}/docs/allocation_v1/report/README.md", f"{s}/report/REPORT_BUILD.md")
    b.add_glob(f"{ALLOC}/reports", "*.md", f"{s}/run_report")
    b.add_opt(f"{ALLOC}/reports/CANARY_REPORT.json", f"{s}/run_report/CANARY_REPORT.json")
    b.add_tables(f"{ALLOC}/tables", f"{s}/tables")
    b.add_glob(f"{ALLOC}/figures", "*.png", f"{s}/figures")
    b.add_glob(f"{ALLOC}/gates", "*.json", f"{s}/gates")
    b.add(f"{ALLOC}/amendments.jsonl", f"{s}/amendments.jsonl")
    b.add_opt(f"{ALLOC}/state/run_decisions.jsonl", f"{s}/run_decisions.jsonl")
    b.add_opt(f"{ALLOC}/manifests/panels.json", f"{s}/manifests/panels.json")

    # 02: forced-depth pilot (raw draws carry prompts, so reports only)
    s = "02_forced_deep_pilot"
    for preset in sorted(os.listdir(PILOT)):
        b.add_opt(f"{PILOT}/{preset}/REPORT.md", f"{s}/{preset}/REPORT.md")

    # 03: forced budget v2 (Qwen3.8-27B, exact budgets)
    s = "03_forced_budget_v2"
    b.add(f"{MAIN}/docs/forced_v2/report/report.pdf", f"{s}/report/report.pdf")
    b.add(f"{MAIN}/docs/forced_v2/report/numbers.json", f"{s}/report/numbers.json")
    b.add(f"{MAIN}/docs/forced_v2/report/README.md", f"{s}/report/REPORT_BUILD.md")
    b.add_glob(f"{MAIN}/docs/forced_v2/report/data", "*.csv", f"{s}/data")
    b.add_tables(f"{FORCED}/merged/tables", f"{s}/tables")
    b.add_glob(f"{FORCED}/merged/reports", "*.md", f"{s}/run_report")
    b.add_glob(f"{FORCED}/merged/figures", "*.png", f"{s}/figures")
    b.add(f"{FORCED}/merged/merge_manifest.json", f"{s}/merge_manifest.json")
    b.add(f"{FORCED}/calibration.json", f"{s}/calibration.json")
    b.add_opt(f"{FORCED}/manifests/panels.json", f"{s}/manifests/panels.json")
    for shard in ("mancer", "parasail"):
        b.add_glob(f"{FORCED}/shards/{shard}/gates", "*.json", f"{s}/gates/{shard}")
        b.add_opt(f"{FORCED}/shards/{shard}/state/run_decisions.jsonl", f"{s}/gates/{shard}/run_decisions.jsonl")

    # 04: dynamic orchestrators (OpenClaude, Codex)
    s = "04_dynorch_v1"
    b.add(f"{DYN_REPORT}/report.pdf", f"{s}/report/report.pdf")
    b.add(f"{DYN_REPORT}/numbers.json", f"{s}/report/numbers.json")
    b.add(f"{DYN_REPORT}/README.md", f"{s}/report/REPORT_BUILD.md")
    b.add_glob(f"{DYN_REPORT}/exports", "*", f"{s}/exports")
    b.add_tables(f"{DYN}/tables", f"{s}/tables")
    b.add_glob(f"{DYN}/gates", "**/*.json", f"{s}/gates")
    b.add_glob(f"{DYN}/manifests", "*.jsonl", f"{s}/manifests")
    b.add_glob(f"{DYN}/reports", "*.md", f"{s}/run_report")
    for name in ("owner_waivers.json", "run_decisions.jsonl"):
        b.add(f"{DYN}/state/{name}", f"{s}/gates/{name}")
    b.add_glob(f"{DYN}/state", "reconciliation_*.json", f"{s}/gates")
    b.add(f"{DYN_ROOT}/rereview_findings.md", f"{s}/notes/rereview_findings.md")
    b.add(f"{DYN_ROOT}/review_issues.json", f"{s}/notes/review_issues.json")

    # 05: passk_cont_v1 (the last study)
    s = "05_passk_cont_v1"
    b.add_glob(PACKAGE, "**/*", f"{s}/results_package")
    b.add_glob(f"{MAIN}/docs/passk_cont/report", "**/*", f"{s}/report")
    b.add_glob(f"{PASSK}/sealed", "**/*", f"{s}/sealed")
    b.add_glob(f"{PASSK}/gates", "*.json", f"{s}/gates")
    b.add_glob(f"{PASSK}/scores", "**/*", f"{s}/scores")
    b.add_glob(f"{PASSK}/scored", "*", f"{s}/scored")
    b.add_glob(f"{PASSK}/e0", "*", f"{s}/e0", exclude=("candidate_catalog.parquet", "reuse_manifest.json"))
    b.add(f"{PASSK}/e8a/results.json", f"{s}/e8a/results.json")
    b.add(f"{PASSK}/lex/front_report.json", f"{s}/lex/front_report.json")
    b.add(f"{PASSK}/lex/selcmp/report.md", f"{s}/lex/selcmp_report.md")
    for name in ("panels", "splits", "waves", "exclusions", "duplicate_groups"):
        b.add(f"{PASSK}/artifacts/manifests/{name}.json", f"{s}/manifests/{name}.json")
    with open(os.path.join(b.out, f"{s}/results_package/prompts/WITHHELD.md"), "w") as fh:
        fh.write("# Withheld prompt files\n\nThe prompt templates here are complete. The rendered examples in `examples/` and "
                 "`llm_selectors/*_example_*.txt` that the package README mentions contain benchmark questions, so they are "
                 "left out of this public copy. See `REDACTIONS.csv` and `RAW_DATA.md` at the top of this investigation.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    if args.manifest_only:
        write_manifest(args.out)
        return 0
    FINGERPRINT.update(build_fingerprint())
    print(f"fingerprint: {len(FINGERPRINT):,} phrases")
    clean(args.out)
    builder = Builder(args.out)
    populate(builder)
    builder.enforce()
    builder.write_schemas()
    builder.write_redactions()
    write_manifest(args.out)
    print(f"copied {len(builder.sources)} files; {len(builder.dropped_csv)} CSVs dropped for a Parquet twin; "
          f"{sum(r['cells_withheld'] for r in builder.redactions)} cells withheld in "
          f"{len({r['file'] for r in builder.redactions})} tables; {len(WITHHELD)} prompt files withheld")
    return 0


if __name__ == "__main__":
    sys.exit(main())
