"""Data export CLI: raw snapshots → public/protected exports, splits, exclusions, hashes (WP1).

    python -m agents_scaling.study.data.export --run-id study_v4 --config configs/study_v4.yaml \
        [--results-root DIR] [--hle-shards DIR] [--bcb-parquet FILE] [--preflight] [--force]

Writes under ``<results_root>/<run_id>/data/``:

* ``public/tasks.jsonl``          — ``PublicTask`` rows only (§10.6, amendment P2);
* ``protected/hle_labels.jsonl``  — ``ProtectedLabel`` rows (dir 0700, file 0600);
* ``protected/bcb_tests.jsonl``   — ``ProtectedBcb`` rows (dir 0700, file 0600);
* ``splits.json``                 — rank rule, counts, per-split id lists, panel sizes;
* ``exclusions.jsonl``            — every excluded row with reason and time (§3.3);
* ``DATA_SHA256.json``            — sha256 of every input and output file;
* ``preflight_report.json``       — with ``--preflight``: task tokens and full root-prompt
  tokens of every dev+main item under all four checkpoint tokenizers and the five root
  cells (00/01/10/11/nat): max and percentiles per cell, wrapper maxima (Table E input for
  ``resources/profile.py``, P1-1) and the over-cap counts, which must be 0 (§4.3; critic
  §4 item 1).

Fail-closed: existing exports are never overwritten without ``--force``; too few eligible
items, a malformed row or an over-cap prompt raise instead of writing a partial export.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study.config import StudyConfig, load_config
from agents_scaling.study.data import bcb as bcb_mod
from agents_scaling.study.data import hle as hle_mod
from agents_scaling.study.data.layout import (
    BCB_TESTS_FILE,
    DIR_MODE,
    FILE_MODE,
    HLE_LABELS_FILE,
    protected_dir,
    public_tasks_path,
)
from agents_scaling.study.data.public import assert_public_row, load_public_tasks
from agents_scaling.study.data.splits import assign_splits, split_summary
from agents_scaling.study.types import (
    PROMPT_TOKENS_CAP,
    TASK_TOKENS_CAP,
    Domain,
    Framing,
    ProtocolError,
    PublicTask,
)

DEFAULT_RESULTS_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results")
RAW_HLE_SHARDS = "raw/hle_shards"
RAW_BCB_PARQUET = "raw/bigcodebench_v0.1.4.parquet"
SPLITS_FILE = "splits.json"
EXCLUSIONS_FILE = "exclusions.jsonl"
HASHES_FILE = "DATA_SHA256.json"
PREFLIGHT_FILE = "preflight_report.json"
ROOT_CELLS: tuple[str, ...] = ("00", "01", "10", "11", "nat")
PERCENTILES: tuple[int, ...] = (50, 90, 99)


@dataclass(frozen=True)
class ExportSummary:
    run_root: Path
    counts: dict[str, dict[str, int]]
    eligible: dict[str, int]
    exclusions: dict[str, dict[str, int]]
    files: dict[str, str]


# --------------------------------------------------------------------------- io helpers


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, mode if mode is not None else 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def _refuse_existing(paths: Sequence[Path], force: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        raise ProtocolError(f"export exists (use --force to overwrite): {[str(p) for p in existing]}")


# --------------------------------------------------------------------------- export


def export_data(
    run_root: str | os.PathLike,
    cfg: StudyConfig,
    *,
    hle_shards: str | os.PathLike | None = None,
    bcb_parquet: str | os.PathLike | None = None,
    tokenizer=None,
    force: bool = False,
    expected_bcb_rows: int | None = bcb_mod.EXPECTED_ROWS,
) -> ExportSummary:
    """Build both superdomains, assign splits and write every export file.

    ``tokenizer`` defaults to the flagship tokenizer (§4.3 envelope authority);
    ``expected_bcb_rows`` pins the snapshot size (1,140; ``None`` only for fixtures).
    """
    root = Path(run_root)
    data_dir = root / "data"
    hle_dir = Path(hle_shards) if hle_shards is not None else data_dir / RAW_HLE_SHARDS
    bcb_path = Path(bcb_parquet) if bcb_parquet is not None else data_dir / RAW_BCB_PARQUET
    public_path = public_tasks_path(root)
    prot_dir = protected_dir(root)
    outputs = {
        "public/tasks.jsonl": public_path,
        "protected/hle_labels.jsonl": prot_dir / HLE_LABELS_FILE,
        "protected/bcb_tests.jsonl": prot_dir / BCB_TESTS_FILE,
        "splits.json": data_dir / SPLITS_FILE,
        "exclusions.jsonl": data_dir / EXCLUSIONS_FILE,
    }
    _refuse_existing(list(outputs.values()), force)

    if tokenizer is None:
        from agents_scaling.study.inference.tokens import load_tokenizer

        tokenizer = load_tokenizer(cfg.flagship_checkpoint)
    hle_build = hle_mod.build_hle(hle_mod.load_hle_rows(hle_dir), tokenizer, cap=cfg.caps.task_tokens)
    bcb_build = bcb_mod.build_bcb(
        bcb_mod.load_bcb_rows(bcb_path, expected_rows=expected_bcb_rows), tokenizer, cap=cfg.caps.task_tokens
    )
    tasks = assign_splits(hle_build.tasks + bcb_build.tasks, cfg)

    public_rows = [task.to_dict() for task in tasks]
    for row in public_rows:
        assert_public_row(row, where=row["source_id"])
    excluded_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    exclusions = [dict(e, excluded_at=excluded_at) for e in hle_build.exclusions + bcb_build.exclusions]
    summary = split_summary(tasks, cfg.items.panels)
    summary["salt_sha256"] = hashlib.sha256(cfg.split_salt).hexdigest()
    summary["config_sha256"] = cfg.config_sha256
    summary["eligible"] = {
        Domain.HLE.value: len(hle_build.tasks),
        Domain.BCB.value: len(bcb_build.tasks),
    }
    summary["strata"] = {
        d.value: {s: {} for s in ("dev", "main", "reserve")} for d in Domain
    }
    for task in tasks:
        block = summary["strata"][Domain(task.domain).value][task.split]
        block[task.stratum] = block.get(task.stratum, 0) + 1

    # protected directory first (0700), then files 0600, then public/aux files
    prot_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(prot_dir, DIR_MODE)
    _atomic_write(outputs["protected/hle_labels.jsonl"], _jsonl(l.to_dict() for l in hle_build.labels.values()), mode=FILE_MODE)
    _atomic_write(outputs["protected/bcb_tests.jsonl"], _jsonl(p.to_dict() for p in bcb_build.protected.values()), mode=FILE_MODE)
    _atomic_write(public_path, _jsonl(public_rows))
    _atomic_write(outputs["splits.json"], json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(outputs["exclusions.jsonl"], _jsonl(exclusions))

    inputs = {f"raw/hle_shards/{p.name}": p for p in sorted(hle_dir.glob("*.parquet"))}
    inputs["raw/" + bcb_path.name] = bcb_path
    hashes = {name: _sha256_file(path) for name, path in {**inputs, **outputs}.items()}
    hashes["_written_at"] = excluded_at
    _atomic_write(data_dir / HASHES_FILE, json.dumps(hashes, indent=2, sort_keys=True) + "\n")

    # re-read through the public loader so the firewall assertion runs on the real file
    load_public_tasks(root)
    excl_counts: dict[str, dict[str, int]] = {d.value: {} for d in Domain}
    for e in exclusions:
        excl_counts[e["domain"]][e["reason"]] = excl_counts[e["domain"]].get(e["reason"], 0) + 1
    return ExportSummary(
        run_root=root,
        counts=summary["counts"],
        eligible=summary["eligible"],
        exclusions=excl_counts,
        files={name: str(path) for name, path in outputs.items()},
    )


# --------------------------------------------------------------------------- preflight


def _percentile(values: Sequence[int], pct: int) -> int:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))  # nearest-rank definition
    return ordered[min(rank, len(ordered)) - 1]


def _stats(values: Sequence[int], cap: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "max": max(values),
        "min": min(values),
        **{f"p{p}": _percentile(values, p) for p in PERCENTILES},
        "cap": cap,
        "n_over_cap": sum(1 for v in values if v > cap),
    }


def preflight(
    run_root: str | os.PathLike,
    cfg: StudyConfig,
    *,
    tasks: Sequence[PublicTask] | None = None,
    tokenizers: Mapping[str, Any] | None = None,
    prompt_cap: int = PROMPT_TOKENS_CAP,
    task_cap: int = TASK_TOKENS_CAP,
) -> dict[str, Any]:
    """Token-envelope preflight over every dev+main item × every checkpoint × five root cells.

    Returns the report dict (also written to ``data/preflight_report.json``).  Raises
    :class:`ProtocolError` after writing when any task exceeds ``task_cap`` or any rendered
    root prompt exceeds ``prompt_cap`` (§4.3 "Preflight checks actual token IDs for every
    checkpoint").
    """
    from agents_scaling.study.inference.tokens import count_tokens, load_tokenizer, render_chat_token_ids
    from agents_scaling.study.prompts.render import render_dec_root, render_root

    root = Path(run_root)
    if tasks is None:
        tasks = [t for t in load_public_tasks(root) if t.split in ("dev", "main")]
    if not tasks:
        raise ProtocolError("preflight: no dev/main tasks")
    if tokenizers is None:
        tokenizers = {size: load_tokenizer(ckpt) for size, ckpt in cfg.checkpoints.items()}

    renders: dict[str, list[dict[str, str]]] = {}
    for cell in ROOT_CELLS:
        for task in tasks:
            if cell == "nat":
                messages = render_dec_root(task, 5).messages
            else:
                messages = render_root(task, Framing(cell)).messages
            renders[f"{cell}:{task.source_id}"] = messages

    report: dict[str, Any] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_items": len(tasks),
        "items_by_split": {s: sum(1 for t in tasks if t.split == s) for s in ("dev", "main")},
        "caps": {"task_tokens": task_cap, "prompt_tokens": prompt_cap},
        "cells": list(ROOT_CELLS),
        "checkpoints": {},
    }
    violations: list[str] = []
    for size, tokenizer in tokenizers.items():
        task_tokens = {t.source_id: count_tokens(t.task_text, tokenizer) for t in tasks}
        exported_mismatch = [t.source_id for t in tasks if size == cfg.flagship and task_tokens[t.source_id] != t.task_tokens]
        block: dict[str, Any] = {
            "task_tokens": _stats(list(task_tokens.values()), task_cap),
            "task_tokens_by_domain": {
                d.value: _stats([task_tokens[t.source_id] for t in tasks if t.domain == d], task_cap) for d in Domain
            },
            "exported_task_tokens_mismatch": exported_mismatch,
            "cells": {},
        }
        for cell in ROOT_CELLS:
            prompt_tokens = {
                t.source_id: len(render_chat_token_ids(tokenizer, renders[f"{cell}:{t.source_id}"], True)) for t in tasks
            }
            wrapper = [prompt_tokens[i] - task_tokens[i] for i in prompt_tokens]
            block["cells"][cell] = {
                "prompt_tokens": _stats(list(prompt_tokens.values()), prompt_cap),
                "wrapper_tokens_max": max(wrapper),
                "wrapper_tokens_min": min(wrapper),
                "root_prompt_max_tokens": task_cap + max(wrapper),
            }
            if block["cells"][cell]["prompt_tokens"]["n_over_cap"]:
                violations.append(f"{size}/{cell}: {block['cells'][cell]['prompt_tokens']['n_over_cap']} prompts > {prompt_cap}")
        if block["task_tokens"]["n_over_cap"]:
            violations.append(f"{size}: {block['task_tokens']['n_over_cap']} tasks > {task_cap}")
        if exported_mismatch:
            violations.append(f"{size}: exported task_tokens differ for {len(exported_mismatch)} items")
        report["checkpoints"][size] = block
    report["root_prompt_max_tokens"] = max(
        c["root_prompt_max_tokens"] for b in report["checkpoints"].values() for c in b["cells"].values()
    )
    report["violations"] = violations
    report["ok"] = not violations
    _atomic_write(root / "data" / PREFLIGHT_FILE, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if violations:
        raise ProtocolError("preflight failed: " + "; ".join(violations))
    return report


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agents_scaling.study.data.export", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=None, help="study yaml (default configs/study_v4.yaml)")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--hle-shards", default=None, help="default <run_root>/data/raw/hle_shards")
    parser.add_argument("--bcb-parquet", default=None, help="default <run_root>/data/raw/bigcodebench_v0.1.4.parquet")
    parser.add_argument("--preflight", action="store_true", help="also run the 4-tokenizer root envelope preflight")
    parser.add_argument("--preflight-only", action="store_true", help="skip the export; preflight the existing one")
    parser.add_argument("--force", action="store_true", help="overwrite an existing export")
    parser.add_argument("--expected-bcb-rows", type=int, default=bcb_mod.EXPECTED_ROWS, help="snapshot row pin (1140)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_root = Path(args.results_root) / args.run_id
    if not args.preflight_only:
        summary = export_data(
            run_root, cfg, hle_shards=args.hle_shards, bcb_parquet=args.bcb_parquet, force=args.force,
            expected_bcb_rows=args.expected_bcb_rows,
        )
        print(json.dumps({"eligible": summary.eligible, "counts": summary.counts, "exclusions": summary.exclusions}, indent=2))
    if args.preflight or args.preflight_only:
        report = preflight(run_root, cfg)
        brief = {
            size: {
                "task_max": b["task_tokens"]["max"],
                "prompt_max_by_cell": {c: v["prompt_tokens"]["max"] for c, v in b["cells"].items()},
            }
            for size, b in report["checkpoints"].items()
        }
        print(json.dumps({"preflight_ok": report["ok"], "root_prompt_max_tokens": report["root_prompt_max_tokens"], **brief}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
