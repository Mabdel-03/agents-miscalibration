"""Gate P: mechanical seal checks -> ``sealed/SEAL_predictions.json`` (plan "Mechanical seal checks").

All must hold, or nothing is written and the stage exits 1:

1. ``git HEAD`` equals ``gates/analysis_review.json["commit"]`` and the worktree is clean;
2. an AST scan of ``passk/models/*``, ``passk/features.py``, ``passk/pools.py`` and ``passk/readout.py`` finds no
   reference to the ``labels`` release functions, ``private``, ``shards/`` or the HF cache;
3. each fit stage ran inside the Gate P bwrap sandbox (``passk-analyze --stage e2-fit|e5-fit --sandbox``):
   ``sealed/runs/{e2-fit,e5-fit}.json`` records rc 0, the exact bwrap argv (its sha256, and every mount on the
   allow-list: system dirs, the env, the code, ``analysis_inputs/`` ro, ``gates/`` ro, ``splits.json`` ro,
   ``sealed/`` rw), the same code commit as HEAD, and the sha256 of every file it wrote, which must still match.
   strace (ptrace is blocked on the compute nodes) and the in-process audit hook are NOT accepted;
3b. the complete PREREG contents exist and came from those sandboxed runs (``REQUIRED``), and the T200
   predictions carry all six predictor sets;
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

__all__ = ["FIT_STAGES", "REQUIRED", "SCANNED", "check_strace", "check_contents", "run_seal", "static_scan"]

PASSK_DIR = Path(__file__).resolve().parent
SCANNED = ("models", "features.py", "pools.py", "readout.py")
FIT_STAGES = ("e2-fit", "e5-fit")
HF_CACHE = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"
LEGACY_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2"
_FORBIDDEN_NAMES = {"join_labels", "verify_seal", "blocked_node_ids", "labels", "grade_scope", "extract",
                    "legacy_draws", "passk_draws"}
_FORBIDDEN_TEXT = ("private", "shards/", "huggingface", "HF_HOME", ".cache/huggingface", "grades/")
_FORBIDDEN_SEGMENTS = {"shards", "private", "grades", "parsed", "forced_budget_v2"}
_FORBIDDEN_MODULES = ("agent_allocation.grading", "agent_allocation.passk.splits", "agent_allocation.passk.checkpoints",
                      "agent_allocation.passk.score", "agent_allocation.passk.e0", "agent_allocation.passk.banks",
                      "agent_allocation.passk.labels", "agent_allocation.passk.grade_scope",
                      "agent_allocation.passk.extract", "agent_allocation.shards", "importlib")
_READERS = {"open", "read_parquet", "read_csv", "read_json", "read_table", "read_text", "read_bytes", "ParquetFile",
            "dataset", "read_pickle", "read_feather", "glob", "rglob", "listdir", "scandir", "walk", "iterdir"}
_ALLOWED_DIRS = ("analysis_inputs", "sealed")
# PREREG seal contents (path under the run root -> producing stage; None = not a sandboxed stage).
REQUIRED = {"sealed/e1_selection.json": None,
            "sealed/v48/predictions.parquet": "e2-fit", "sealed/v48/crossover.parquet": "e2-fit",
            "sealed/v48/population.parquet": "e2-fit",
            "sealed/v48/manifest.json": "e2-fit", "sealed/v48/seeds.json": "e2-fit",
            "sealed/t200/predictions.parquet": "e5-fit", "sealed/t200/features.parquet": "e5-fit",
            "sealed/t200/manifest.json": "e5-fit", "sealed/t200/seeds.json": "e5-fit"}
REQUIRED_MODEL_DIRS = {"sealed/v48/models": "e2-fit", "sealed/t200/models": "e5-fit"}
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
                    if full and any(full == m or full.startswith(m + ".") for m in _FORBIDDEN_MODULES):
                        hits.append(f"import of fenced helper {full} (line {node.lineno})")
                for full in [module] + names + [f"{module}.{n}" for n in names]:
                    last = full.rsplit(".", 1)[-1]
                    if last in _FORBIDDEN_NAMES or ".labels" in full or ".grade_scope" in full:
                        hits.append(f"import {full} (line {node.lineno})")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
                hits.append(f"attribute .{node.attr} (line {node.lineno})")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES - {"labels", "extract"}:
                hits.append(f"name {node.id} (line {node.lineno})")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(token in node.value for token in _FORBIDDEN_TEXT) or node.value in _FORBIDDEN_SEGMENTS:
                    hits.append(f"string {node.value[:60]!r} (line {node.lineno})")
            elif isinstance(node, ast.Name) and node.id == "__import__":
                hits.append(f"dynamic import (line {node.lineno})")
            if isinstance(node, ast.Call):
                func = node.func
                fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if fname in _READERS:
                    consts = [c.value for a in list(node.args) + [k.value for k in node.keywords]
                              for c in ast.walk(a) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                    if isinstance(func, ast.Attribute):              # Path(...).read_text(): scan the receiver too
                        consts += [c.value for c in ast.walk(func.value)
                                   if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                    pathy = [c for c in consts if "/" in c or c.startswith(("artifacts", "catalog", "logs"))
                             or c in ("artifacts", "catalog", "logs", "scores")]
                    if any(c.startswith("/") or c.split("/")[0] in ("artifacts", "catalog", "logs", "scores")
                           for c in pathy) and not any(d in c for c in consts for d in _ALLOWED_DIRS):
                        hits.append(f"{fname}() on a path outside analysis_inputs/sealed (line {node.lineno})")
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


def check_contents(run_root: Path, records: dict[str, dict]) -> list[str]:
    """PREREG contents: every REQUIRED file exists and (for fit outputs) was written by its sandboxed run."""
    run_root = Path(run_root)
    problems = []
    for rel, stage in REQUIRED.items():
        if not (run_root / rel).is_file():
            problems.append(f"{rel} missing (PREREG seal contents)")
        elif stage is not None and rel not in (records.get(stage) or {}).get("outputs", {}):
            problems.append(f"{rel} was not written by the sandboxed {stage} run")
    for rel, stage in REQUIRED_MODEL_DIRS.items():
        files = sorted((run_root / rel).glob("*")) if (run_root / rel).is_dir() else []
        if not files:
            problems.append(f"{rel}/ is empty (fitted models are part of the seal)")
        outs = (records.get(stage) or {}).get("outputs", {})
        for f in files:
            if f.is_file() and str(f.relative_to(run_root)) not in outs:
                problems.append(f"{f.relative_to(run_root)} was not written by the sandboxed {stage} run")
    t200 = run_root / "sealed" / "t200" / "predictions.parquet"
    if t200.is_file():
        try:
            import pandas as pd

            from agent_allocation.passk.models.predictors import PREDICTORS
            got = set(pd.read_parquet(t200, columns=["predictor"])["predictor"].astype(str))
            missing = sorted(set(PREDICTORS) - got)
            if missing:
                problems.append(f"sealed/t200/predictions.parquet lacks predictor sets {missing}")
        except Exception as exc:                                       # noqa: BLE001 - unreadable = refused
            problems.append(f"sealed/t200/predictions.parquet unreadable: {exc}")
    return problems


def _head_and_dirty() -> tuple[str | None, str]:
    """Inside the sandbox the git dir is not mounted: the launcher passes the commit it measured outside."""
    if os.environ.get("PASSK_SANDBOXED") == "1":
        return os.environ.get("PASSK_CODE_COMMIT") or None, ("dirty" if os.environ.get("PASSK_CODE_DIRTY") != "0"
                                                              else "")
    return _git(["rev-parse", "HEAD"]), _git(["status", "--porcelain", "--untracked-files=no"])


def run_seal(run_root: Path, *, require_git: bool = True, stages: tuple[str, ...] = FIT_STAGES) -> int:
    from agent_allocation.passk.analyze import verify_run_record
    from agent_allocation.passk.grade_scope import verify_seal

    run_root = Path(run_root)
    target = run_root / "sealed" / "SEAL_predictions.json"
    if target.exists():                                   # idempotent: never rewrite (or un-verify) a seal
        ok = verify_seal(run_root) is not None
        print(json.dumps({"seal": "exists", "verifies": ok}))
        return 0 if ok else 1
    problems: list[str] = []
    review_path = run_root / "gates" / "analysis_review.json"
    reviewed = None
    head = None
    try:
        head, dirty = _head_and_dirty()
    except (OSError, subprocess.CalledProcessError) as exc:
        if require_git:
            problems.append(f"git unavailable: {exc}")
        dirty = "?"
    if require_git:
        if not head:
            problems.append("code commit unknown")
        if not review_path.exists():
            problems.append(f"{review_path} missing")
        else:
            reviewed = json.loads(review_path.read_text(encoding="utf-8")).get("commit")
            if reviewed != head:
                problems.append(f"HEAD {head} != reviewed commit {reviewed}")
        if dirty:
            problems.append("worktree is not clean")
    scan = static_scan()
    if not scan["ok"]:
        problems.append(f"static scan findings: {scan['findings']}")
    records: dict[str, dict] = {}
    runs = {}
    for stage in stages:
        rec, rec_problems = verify_run_record(run_root, stage)
        problems += rec_problems
        if rec is None:
            continue
        records[stage] = rec
        runs[stage] = {"argv_sha256": rec.get("argv_sha256"), "code_commit": rec.get("code_commit"),
                       "rc": rec.get("rc"), "host": rec.get("host"), "n_outputs": len(rec.get("outputs") or {})}
        if require_git and rec.get("code_commit") != head:
            problems.append(f"{stage}: sandboxed run was at commit {rec.get('code_commit')}, HEAD is {head}")
        if require_git and rec.get("worktree_dirty"):
            problems.append(f"{stage}: sandboxed run was on a dirty worktree")
    problems += check_contents(run_root, records)
    # The fit inputs are recorded (``inputs_at_seal``) and their manifest is snapshotted into sealed/, but only
    # sealed/ files are *verified*: a release re-extract rewrites analysis_inputs/, which must not un-verify the seal.
    inputs = {}
    for path in sorted((run_root / "analysis_inputs").rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            inputs[str(path.relative_to(run_root))] = _sha256(path)
    if problems:
        print(json.dumps({"seal": "refused", "problems": problems}, indent=2), file=sys.stderr)
        return 1
    manifest = run_root / "analysis_inputs" / "manifest.json"
    snapshot = run_root / "sealed" / "inputs_manifest_at_seal.json"
    snap_tmp = snapshot.with_name(f".{snapshot.name}.tmp{os.getpid()}")
    if manifest.exists():
        snap_tmp.write_text(json.dumps({"manifest": json.loads(manifest.read_text()), "sha256": inputs},
                                       indent=2, sort_keys=True) + "\n")
    artifacts = {}
    for path in sorted((run_root / "sealed").rglob("*")):
        if path.is_file() and path.name != "SEAL_predictions.json" and not path.name.startswith("."):
            artifacts[str(path.relative_to(run_root))] = _sha256(path)
    if snap_tmp.exists():
        artifacts[str(snapshot.relative_to(run_root))] = _sha256(snap_tmp)
    splits = [s for s in ("v48", "t200") if f"sealed/{s}/predictions.parquet" in artifacts]
    seal = {"code_commit": head, "reviewed_commit": reviewed, "artifacts": artifacts, "inputs_at_seal": inputs,
            "sandbox_runs": runs, "sandboxed_seal": os.environ.get("PASSK_SANDBOXED") == "1",
            "seal_argv_sha256": os.environ.get("PASSK_SANDBOX_ARGV_SHA256"), "splits": splits,
            "static_scan": scan, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if snap_tmp.exists():
        os.replace(snap_tmp, snapshot)
    tmp = target.with_name(f".seal.tmp{os.getpid()}")
    tmp.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    print(json.dumps({"seal": "written", "n_artifacts": len(artifacts), "splits": splits}))
    return 0
