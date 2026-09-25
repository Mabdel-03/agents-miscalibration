"""Gate P: mechanical seal checks -> ``sealed/SEAL_predictions.json`` (plan "Mechanical seal checks").

All must hold, or nothing is written and the stage exits 1:

1. ``git HEAD`` equals ``gates/analysis_review.json["commit"]`` and the worktree is clean;
2. an AST scan of ``passk/models/*``, ``passk/features.py``, ``passk/pools.py`` and ``passk/readout.py`` finds no
   reference to the ``labels`` release functions, ``private``, ``shards/`` or the HF cache;
3. the fit stages' strace logs (``logs/strace/{e2-fit,e5-fit}.log``, written by ``strace -f -e
   trace=open,openat,openat2``) show no open under ``artifacts/shards/``, ``artifacts/private/``, the HF cache or the
   legacy run root;
4. every artifact under ``sealed/`` (except the seal) is hashed into ``artifacts`` (what ``verify_seal`` checks);
   ``analysis_inputs/`` hashes go to ``inputs_at_seal`` plus a snapshot ``sealed/inputs_manifest_at_seal.json``
   (a later release re-extract rewrites ``analysis_inputs/`` and must not un-verify the seal).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["FIT_STAGES", "SCANNED", "check_strace", "run_seal", "static_scan"]

PASSK_DIR = Path(__file__).resolve().parent
SCANNED = ("models", "features.py", "pools.py", "readout.py")
FIT_STAGES = ("e2-fit", "e5-fit")
HF_CACHE = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"
LEGACY_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2"
_FORBIDDEN_NAMES = {"join_labels", "verify_seal", "blocked_node_ids", "labels", "grade_scope", "extract",
                    "legacy_draws", "passk_draws"}
_FORBIDDEN_TEXT = ("private", "shards/", "huggingface", "HF_HOME", ".cache/huggingface", "grades/")
_OPEN = re.compile(r'open(?:at2?)?\((?:[^,]*,\s*)?"([^"]*)"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scanned_files(root: Path = PASSK_DIR) -> list[Path]:
    files = []
    for name in SCANNED:
        target = root / name
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.exists():
            files.append(target)
    return files


def static_scan(root: Path = PASSK_DIR) -> dict:
    """Findings per file: forbidden imports/attribute names and forbidden string literals."""
    findings: dict[str, list[str]] = {}
    files = _scanned_files(root)
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                      and isinstance(n.body[0].value.value, str)}
        hits = []
        for node in ast.walk(tree):
            if id(node) in docstrings:
                continue                                   # prose may name the fence; code may not touch it
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                for full in [module] + names + [f"{module}.{n}" for n in names]:
                    last = full.rsplit(".", 1)[-1]
                    if last in _FORBIDDEN_NAMES or ".labels" in full or ".grade_scope" in full:
                        hits.append(f"import {full} (line {node.lineno})")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
                hits.append(f"attribute .{node.attr} (line {node.lineno})")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES - {"labels", "extract"}:
                hits.append(f"name {node.id} (line {node.lineno})")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(token in node.value for token in _FORBIDDEN_TEXT):
                    hits.append(f"string {node.value[:60]!r} (line {node.lineno})")
        if hits:
            findings[str(path.relative_to(root))] = sorted(set(hits))
    return {"files": [str(p.relative_to(root)) for p in files], "findings": findings, "ok": not findings}


def check_strace(log: Path, run_root: Path, *, legacy_root: str = LEGACY_ROOT, hf_cache: str = HF_CACHE) -> dict:
    run_root = Path(run_root).resolve()
    forbidden = [str(run_root / "artifacts" / "shards"), str(run_root / "artifacts" / "private"), hf_cache,
                 legacy_root]
    opened = bad = 0
    examples: list[str] = []
    mode = "strace"
    with Path(log).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("# mode=audithook"):
                mode = "audithook"
                continue
            match = _OPEN.search(line)
            if not match:
                continue
            opened += 1
            path = match.group(1)
            relative_bad = path.startswith(("shards/", "private/", "artifacts/shards", "artifacts/private"))
            if relative_bad or any(path.startswith(f) for f in forbidden):
                bad += 1
                if len(examples) < 10:
                    examples.append(path)
    return {"opens": opened, "forbidden_opens": bad, "examples": examples, "mode": mode,
            "ok": opened > 0 and bad == 0}


def _git(args: list[str]) -> str:
    return subprocess.run(["git", "-C", str(PASSK_DIR), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def run_seal(run_root: Path, *, require_git: bool = True, stages: tuple[str, ...] = FIT_STAGES) -> int:
    run_root = Path(run_root)
    problems: list[str] = []
    review_path = run_root / "gates" / "analysis_review.json"
    reviewed = None
    head = None
    if require_git:
        try:
            head = _git(["rev-parse", "HEAD"])
            dirty = _git(["status", "--porcelain", "--untracked-files=no"])
        except (OSError, subprocess.CalledProcessError) as exc:
            problems.append(f"git unavailable: {exc}")
            dirty = "?"
        if not review_path.exists():
            problems.append(f"{review_path} missing")
        else:
            reviewed = json.loads(review_path.read_text(encoding="utf-8")).get("commit")
            if reviewed != head:
                problems.append(f"HEAD {head} != reviewed commit {reviewed}")
        if dirty:
            problems.append("worktree is not clean")
    if not require_git:
        try:
            head = _git(["rev-parse", "HEAD"])
        except (OSError, subprocess.CalledProcessError):
            head = None
    scan = static_scan()
    if not scan["ok"]:
        problems.append(f"static scan findings: {scan['findings']}")
    straces = {}
    for stage in stages:
        log = run_root / "logs" / "strace" / f"{stage}.log"
        if not log.exists():
            problems.append(f"{log} missing (run the fit stage under strace)")
            continue
        result = check_strace(log, run_root)
        result["sha256"] = _sha256(log)
        straces[str(log.relative_to(run_root))] = result
        if not result["ok"]:
            problems.append(f"{stage}: strace shows forbidden opens {result['examples']}")
    # The fit inputs are recorded (``inputs_at_seal``) and their manifest is snapshotted into sealed/, but only
    # sealed/ files are *verified*: a release re-extract rewrites analysis_inputs/, which must not un-verify the seal.
    inputs = {}
    for path in sorted((run_root / "analysis_inputs").rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            inputs[str(path.relative_to(run_root))] = _sha256(path)
    manifest = run_root / "analysis_inputs" / "manifest.json"
    if manifest.exists() and not problems:
        snapshot = run_root / "sealed" / "inputs_manifest_at_seal.json"
        snapshot.write_text(json.dumps({"manifest": json.loads(manifest.read_text()), "sha256": inputs},
                                       indent=2, sort_keys=True) + "\n")
    artifacts = {}
    for path in sorted((run_root / "sealed").rglob("*")):
        if path.is_file() and path.name != "SEAL_predictions.json" and not path.name.startswith("."):
            artifacts[str(path.relative_to(run_root))] = _sha256(path)
    if not any(k.startswith("sealed/") and not k.endswith("inputs_manifest_at_seal.json") for k in artifacts):
        problems.append("nothing under sealed/ to seal")
    if problems:
        print(json.dumps({"seal": "refused", "problems": problems}, indent=2), file=sys.stderr)
        return 1
    seal = {"code_commit": head, "reviewed_commit": reviewed, "artifacts": artifacts, "inputs_at_seal": inputs,
            "strace_logs": straces,
            "static_scan": scan, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    target = run_root / "sealed" / "SEAL_predictions.json"
    if target.exists():
        print(f"{target} exists; a seal is written once", file=sys.stderr)
        return 1
    tmp = target.with_name(f".seal.tmp{os.getpid()}")
    tmp.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    print(json.dumps({"seal": "written", "n_artifacts": len(artifacts)}))
    return 0
