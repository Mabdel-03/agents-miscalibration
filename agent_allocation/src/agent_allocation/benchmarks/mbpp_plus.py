"""MBPP+ inventory and panel draw (agent 05).

Source: the frozen EvalPlus MBPP+ inventory of agent_methods_bench (EvalPlus ``26d6d00b``, MbppPlus release from the
frozen manifest), every file pinned by sha256 below, and the per-task reference oracles certified by
``manifests/evalplus_reference_certification_v4.json`` (paths + sha256). The inventory size is read from the frozen
manifest and checked against the rows; it is never hardcoded.

Public/private split:

* ``PublicTask.prompt`` is the official EvalPlus MBPP+ ``prompt`` verbatim (a docstring holding the task text and
  exactly one example ``assert``). ``public_examples`` is that single assert, parsed from the prompt text. The
  restricted record's ``assertion`` field holds private base tests and is never read here.
* ``PrivateEvalSpec.private_record_ref`` is a run-dir-relative directory under ``paths.code_oracles_dir`` holding
  ``restricted.json`` (the frozen restricted row) and ``oracle.json`` (hardlink or byte copy of the certified
  oracle). ``private_hash`` = ``canonical_hash`` of {task_id, native_revision, restricted_record_sha256 (the
  vendored ``content_hash`` the oracle binds), oracle_sha256 (certified file hash)}: it binds gold and tests.
  ``load_inventory`` installs these files idempotently (0700 dirs, verified sha256).

Config keys consumed: ``benchmarks[id=mbpp_plus].source`` (must be ``evalplus/evalplus``) and ``.grader`` (must be
``evalplus_full``). Environment: ``ALLOCATION_MBPP_SOURCE_ROOT`` overrides the source repository root (tests only).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation._vendor.learnagents_io import content_hash, file_hash, write_json
from agent_allocation.contracts import (EvalKind, InventoryItem, OutputKind, PanelDraw, PrivateEvalSpec,
                                        PublicTask, canonical_hash)
from agent_allocation.paths import PRIVATE_DIR_MODE, RunPaths

BENCHMARK = "mbpp_plus"
SOURCE_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agent_methods_bench")
NATIVE_REVISION = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
FROZEN_DIR = "data/frozen/mbpp_plus_v1"
FROZEN_MANIFEST_SHA256 = "db575ea6a662837a260c596804ee1b96ca309329ce63552f9825eb22f8771c91"
CERTIFICATION = "manifests/evalplus_reference_certification_v4.json"
CERTIFICATION_SHA256 = "ac3055f9a5218ea6625aedd84cfd2bce7ddcab78f06d65d60ff95479c4772889"
PREPROCESSING_VERSION = "mbpp_plus-inventory-v1"
# Verified against the pinned upstream LICENSE (sandbox/upstream/evalplus/26d6d00b…/LICENSE, sha256 fcc1f77f…):
# Apache-2.0, with evalplus/eval/ additionally under MIT (built on OpenAI HumanEval). The frozen MbppPlus release
# manifest records "apache-2.0". The underlying MBPP problems come from google-research/mbpp (CC-BY-4.0); that
# licence is not restated inside the pinned EvalPlus files, so it is named here as the upstream dataset's licence.
LICENSE = ("Apache-2.0 (EvalPlus 26d6d00b code and MbppPlus v0.2.0 release; evalplus/eval/ also MIT); "
           "MBPP source problems CC-BY-4.0 (google-research/mbpp)")
SOURCE_PROVENANCE = {
    "repository": "evalplus/evalplus",
    "revision": NATIVE_REVISION,
    "release": "MbppPlus v0.2.0",
    "frozen_copy": f"{SOURCE_ROOT}/{FROZEN_DIR}",
    "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
    "reference_certification": f"{SOURCE_ROOT}/{CERTIFICATION}",
    "reference_certification_sha256": CERTIFICATION_SHA256,
    "upstream_license_sha256": "fcc1f77fe5443b21bfc4dfcc3d66f7654f599b91582df771850e5e876986a103",
    "public_fields": "EvalPlus prompt verbatim; public_examples = its single assert (parsed from the prompt text)",
    "private_fields": "restricted native record + certified oracle under private/evalplus_oracles/<task>/",
}
PANEL_PROCEDURE = ("mbpp_plus-uniform-v1: eligible item_ids ordered by sha256('{seed}|mbpp_plus|{item_id}') "
                   "(a seeded uniform permutation); dev = first n_dev, main = next n_main in that order, "
                   "reserve = next n_reserve")

# Pre-generation engineering exclusions (reference fails its own prompt assert or cannot run at the frozen
# grading limits). Filled only from the recorded certification job; never from model outcomes.
PRE_GENERATION_EXCLUSIONS: dict[str, str] = {}


def source_root() -> Path:
    return Path(os.environ.get("ALLOCATION_MBPP_SOURCE_ROOT") or SOURCE_ROOT)


def item_id_for(task_id: str) -> str:
    return f"{BENCHMARK}/{task_id.replace('/', '_')}"


def task_dir_name(task_id: str) -> str:
    return task_id.replace("/", "_")


def parse_public_assert(prompt: str) -> str:
    """Return the single ``assert`` statement shown in the official prompt; raise ValueError otherwise."""
    with warnings.catch_warnings():              # prompts quote regexes such as '\w'; not our syntax to police
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        module = ast.parse(prompt)
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr) \
            or not isinstance(module.body[0].value, ast.Constant) or not isinstance(module.body[0].value.value, str):
        raise ValueError("MBPP+ prompt is not a single docstring")
    text = module.body[0].value.value
    starts = [i for i, line in enumerate(text.splitlines()) if line.lstrip().startswith("assert")]
    if len(starts) != 1:
        raise ValueError(f"MBPP+ prompt shows {len(starts)} assert lines; expected exactly one")
    lines = text.splitlines()
    candidate = "\n".join(lines[starts[0]:]).strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        statements = ast.parse(candidate).body
    if len(statements) != 1 or not isinstance(statements[0], ast.Assert):
        raise ValueError("MBPP+ prompt example is not exactly one assert statement")
    return candidate


def _check_sha(path: Path, expected: str) -> None:
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} != pinned {expected}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mbpp_entry(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = [b for b in cfg.get("benchmarks", ()) if b.get("id") == BENCHMARK]
    if len(entries) != 1:
        raise ValueError("config must declare exactly one mbpp_plus benchmark")
    entry = entries[0]
    if entry.get("source") != "evalplus/evalplus" or entry.get("grader") != "evalplus_full":
        raise ValueError(f"unsupported mbpp_plus source/grader: {entry.get('source')!r}/{entry.get('grader')!r}")
    return entry


def read_sources(root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
                                                    dict[str, dict[str, Any]]]:
    """Verify every pinned hash and return (manifest, public rows, restricted rows, certification by problem_id)."""
    root = root or source_root()
    frozen = root / FROZEN_DIR
    _check_sha(frozen / "manifest.json", FROZEN_MANIFEST_SHA256)
    manifest = json.loads((frozen / "manifest.json").read_text())
    _check_sha(frozen / "public.jsonl", manifest["public_sha256"])
    _check_sha(frozen / "restricted.jsonl", manifest["restricted_sha256"])
    public_rows = _read_jsonl(frozen / "public.jsonl")
    restricted_rows = _read_jsonl(frozen / "restricted.jsonl")
    if not (len(public_rows) == len(restricted_rows) == manifest["count"]):
        raise ValueError("MBPP+ frozen inventory count disagrees with its manifest")
    _check_sha(root / CERTIFICATION, CERTIFICATION_SHA256)
    certification = json.loads((root / CERTIFICATION).read_text())
    if certification.get("status") != "passed" or certification.get("native_revision") != NATIVE_REVISION:
        raise ValueError("EvalPlus reference certification is not a passed run of the pinned revision")
    cert = {row["problem_id"]: row for row in certification["results"] if "mbpp_plus" in row["oracle_path"]}
    return manifest, public_rows, restricted_rows, cert


def _decode(restricted: Mapping[str, Any]) -> dict[str, Any]:
    from agent_allocation._vendor.evalplus_external import decode_evalplus_record
    return decode_evalplus_record(dict(restricted))


def build_inventory(paths: RunPaths, root: Path | None = None) -> tuple[list[InventoryItem], dict[str, dict[str, Any]]]:
    """Pure: build items without writing. Returns (items, install plan keyed by item_id)."""
    root = root or source_root()
    manifest, public_rows, restricted_rows, cert = read_sources(root)
    restricted_by_pid = {row["problem_id"]: row for row in restricted_rows}
    release = manifest["metadata"]["source"]["mbpp_version"]
    if manifest["metadata"]["source"]["revision"] != NATIVE_REVISION:
        raise ValueError("frozen MBPP+ manifest revision differs from the pinned EvalPlus revision")
    source_revision = f"evalplus@{NATIVE_REVISION}/MbppPlus-{release}/{PREPROCESSING_VERSION}"
    items: list[InventoryItem] = []
    plan: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for public_row in public_rows:
        pid = public_row["problem_id"]
        restricted = restricted_by_pid[pid]
        native = _decode(restricted)
        task_id = native["task_id"]
        if public_row["provenance_ref"] != f"mbpp_plus@{NATIVE_REVISION}:{task_id}":
            raise ValueError(f"{pid}: public/restricted task identity mismatch")
        prompt = native["prompt"]
        if public_row["public"]["starter_code"] != prompt:
            raise ValueError(f"{task_id}: public starter code differs from the native prompt")
        item_id = item_id_for(task_id)
        if item_id in seen:
            raise ValueError(f"duplicate MBPP+ item {item_id}")
        seen.add(item_id)
        certified = cert.get(pid)
        eligible, reason = True, None
        try:
            public_assert = parse_public_assert(prompt)
            examples: tuple[str, ...] = (public_assert,)
        except (SyntaxError, ValueError) as exc:
            examples, eligible, reason = (), False, f"public_example_unparseable: {exc}"
        if certified is None or certified.get("status") != "reference_verified":
            eligible, reason = False, "reference_not_certified"
        if task_id in PRE_GENERATION_EXCLUSIONS:
            eligible, reason = False, PRE_GENERATION_EXCLUSIONS[task_id]
        restricted_sha = content_hash(restricted)
        oracle_sha = certified["oracle_sha256"] if certified else ""
        private_hash = canonical_hash({"task_id": task_id, "native_revision": NATIVE_REVISION,
                                       "restricted_record_sha256": restricted_sha, "oracle_sha256": oracle_sha})
        ref_dir = paths.code_oracles_dir / task_dir_name(task_id)
        public = PublicTask(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision, prompt=prompt,
                            output_kind=OutputKind.PYTHON, public_examples=examples)
        private = PrivateEvalSpec(item_id=item_id, benchmark=BENCHMARK, source_revision=source_revision,
                                  eval_kind=EvalKind.EVALPLUS_FULL, gold_answer=None,
                                  private_record_ref=_relative(paths, ref_dir), private_hash=private_hash)
        items.append(InventoryItem(public=public, private=private, strata=(), eligible=eligible,
                                   exclusion_reason=reason))
        plan[item_id] = {"dir": ref_dir, "restricted": restricted, "restricted_sha256": restricted_sha,
                         "oracle_source": root / certified["oracle_path"] if certified else None,
                         "oracle_sha256": oracle_sha, "task_id": task_id}
    items.sort(key=lambda it: it.public.item_id)
    return items, plan


def _relative(paths: RunPaths, target: Path) -> str:
    return os.path.relpath(os.path.abspath(target), os.path.abspath(paths.run_dir))


def install_private_record(entry: Mapping[str, Any]) -> None:
    """Idempotently place restricted.json + oracle.json for one task; verify hashes; never overwrite drift."""
    directory = Path(entry["dir"])
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(directory.parent, PRIVATE_DIR_MODE)
    directory.mkdir(exist_ok=True)
    os.chmod(directory, PRIVATE_DIR_MODE)
    restricted_path = directory / "restricted.json"
    if restricted_path.exists():
        if content_hash(json.loads(restricted_path.read_text())) != entry["restricted_sha256"]:
            raise ValueError(f"{restricted_path} differs from the frozen restricted record")
    else:
        write_json(restricted_path, entry["restricted"])
        os.chmod(restricted_path, 0o600)
    oracle_path = directory / "oracle.json"
    if entry["oracle_source"] is None:
        return
    if not oracle_path.exists():
        source = Path(entry["oracle_source"])
        _check_sha(source, entry["oracle_sha256"])
        try:
            os.link(source, oracle_path)
        except OSError:
            tmp = oracle_path.with_suffix(".json.tmp")
            shutil.copyfile(source, tmp)
            os.chmod(tmp, 0o600)
            os.replace(tmp, oracle_path)
    verify_oracle_file(oracle_path, entry["oracle_sha256"])


_VERIFIED: dict[tuple[str, int, int], str] = {}


def verify_oracle_file(path: Path, expected: str) -> None:
    """sha256-verify an oracle once per (path, size, mtime) in this process."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if _VERIFIED.get(key) == expected:
        return
    _check_sha(path, expected)
    _VERIFIED[key] = expected


def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]:
    _mbpp_entry(cfg)
    items, plan = build_inventory(paths)
    for item in items:
        install_private_record(plan[item.public.item_id])
    return items


def select_panels(inventory: Sequence[InventoryItem], *, seed: int, n_dev: int, n_main: int,
                  n_reserve: int) -> PanelDraw:
    if min(n_dev, n_main, n_reserve) < 0:
        raise ValueError("panel sizes must be nonnegative")
    eligible = sorted({item.public.item_id for item in inventory
                       if item.eligible and item.public.benchmark == BENCHMARK})
    needed = n_dev + n_main + n_reserve
    if len(eligible) < needed:
        raise ValueError(f"mbpp_plus has {len(eligible)} eligible items; panels need {needed}")

    def key(item_id: str) -> str:
        return hashlib.sha256(f"{seed}|{BENCHMARK}|{item_id}".encode()).hexdigest()

    order = sorted(eligible, key=lambda item_id: (key(item_id), item_id))
    return PanelDraw(benchmark=BENCHMARK, dev_item_ids=tuple(order[:n_dev]),
                     main_item_ids=tuple(order[n_dev:n_dev + n_main]),
                     reserve_item_ids=tuple(order[n_dev + n_main:needed]), selection_seed=seed,
                     procedure=PANEL_PROCEDURE)
