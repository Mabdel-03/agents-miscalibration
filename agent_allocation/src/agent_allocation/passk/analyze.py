"""``allocation passk-analyze`` (contract §6): one verb, one stage per sbatch job.

Stages: ``extract`` (the single reader), ``pairs`` (fill the pair cache), ``e1-lock``, ``e2-fit`` / ``e5-fit``
(WP-M's ``passk.models.e2/e5.fit_and_predict(run_root)``, imported lazily), ``seal`` (mechanical checks ->
``sealed/SEAL_predictions.json``), ``release`` (extract with ``--release v48|t200`` after a verifying seal),
``score``, ``report``, ``e0``.

Exit codes: 0 ok, 1 check failed, 2 blocked (a dependency is not built yet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

__all__ = ["STAGES", "cmd_passk_analyze", "e1_lock"]

STAGES = ("extract", "pairs", "e1-lock", "e2-fit", "e5-fit", "seal", "release", "score", "report", "e0")
E1_CANDIDATES = ("8x2", "4x4", "2x8")                  # ties -> fewer calls, i.e. this order reversed
EXIT_OK, EXIT_FAIL, EXIT_BLOCKED = 0, 1, 2


class NotReady(RuntimeError):
    """A stage's inputs are not complete yet (exit 2, blocked; the conductor retries later)."""


def _write_once_json(path: Path, payload: dict, *, compare_keys: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if any(prior.get(k) != payload.get(k) for k in compare_keys):
            raise RuntimeError(f"{path} exists with a different result; it is write-once")
        return
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def e1_lock(run_root: Path, *, draws=None, cache=None) -> dict:
    """M* = argmax over {8x2, 4x4, 2x8} of the L96 mixture-weighted tie-averaged plurality accuracy."""
    import pandas as pd

    from agent_allocation.passk.extract import load_draws
    from agent_allocation.passk.metrics import MACRO_WEIGHTS, MIXTURE_WEIGHTS, pool_metrics, weighted_mean
    from agent_allocation.passk.pairs import PairCache
    from agent_allocation.passk.pools import build_pools
    from agent_allocation.passk.readout import readout_pools

    draws = load_draws(run_root) if draws is None else draws
    cache = PairCache(Path(run_root) / "catalog" / "pair_verdicts.parquet") if cache is None else cache
    needed = draws[draws["partition"].isin(["legacy_main", "e1_fill"]) & draws["allowance_a"].isin([2, 4, 8, 16])]
    if "pending" in needed and needed["pending"].fillna(False).astype(bool).any():
        raise NotReady(f"{int(needed['pending'].fillna(False).astype(bool).sum())} E1 nodes have not settled")
    if "graded" in needed:
        ungraded = (~needed["absent"].astype(bool)) & ~needed["graded"].fillna(False).astype(bool)
        if ungraded.any():
            raise NotReady(f"{int(ungraded.sum())} settled E1 nodes are not graded yet")
    pools = build_pools(draws, panels=("L96",), configs=E1_CANDIDATES + ("1x16",))
    counts = pools.groupby("config_id")["item_id"].nunique().to_dict()
    n_items = draws.loc[draws["partition"] == "legacy_main", "item_id"].nunique()
    incomplete = {c: counts.get(c, 0) for c in E1_CANDIDATES if counts.get(c, 0) != n_items}
    if incomplete:
        raise NotReady(f"E1 lock needs every L96 item in each candidate allocation; incomplete: {incomplete}")
    metrics = pool_metrics(readout_pools(pools, draws, cache=cache, mode="passk"), draws)
    table = {}
    for config_id, sub in metrics.groupby("config_id"):
        fam = sub.groupby("benchmark")["S_tie"].mean().to_dict()
        table[config_id] = {"mixture": weighted_mean(fam, MIXTURE_WEIGHTS), "macro": weighted_mean(fam, MACRO_WEIGHTS),
                            "by_family": fam, "n_items": int(sub["item_id"].nunique()),
                            "n_unknown_tie": int(sub["S_tie"].isna().sum())}
    calls = {c: int(c.split("x")[0]) for c in E1_CANDIDATES}
    best = sorted(E1_CANDIDATES, key=lambda c: (-round(table[c]["mixture"], 12), calls[c]))[0]
    payload = {"version": "passk-e1-lock-v1", "selected": best, "rule": "argmax L96 mixture-weighted tie-averaged "
               "plurality accuracy over {8x2,4x4,2x8}; ties -> fewer calls", "weights": MIXTURE_WEIGHTS,
               "table": table, "reference_1x16": table.get("1x16")}
    _write_once_json(Path(run_root) / "sealed" / "e1_selection.json", payload, compare_keys=("selected",))
    _ = pd  # frame library imported for the helpers above
    return payload


def _stage_pairs(run_root: Path, args) -> int:
    import pandas as pd

    from agent_allocation.passk.extract import LEGACY_ROOT, legacy_draws, load_draws
    from agent_allocation.passk.pairs import PairCache, enumerate_pairs, math_votes_from_frame

    frames = []
    inputs = run_root / "analysis_inputs" / "draws.parquet"
    if inputs.exists():
        frames.append(load_draws(run_root))
    elif Path(args.legacy_root).exists():
        frames.append(legacy_draws(Path(args.legacy_root), with_labels=False, read_raw=False))
    if not frames:
        print("blocked: no analysis_inputs and no legacy root", file=sys.stderr)
        return EXIT_BLOCKED
    draws = pd.concat(frames, ignore_index=True)
    pairs = enumerate_pairs(math_votes_from_frame(draws))
    cache = PairCache(run_root / "catalog" / "pair_verdicts.parquet")
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)) or 1)
    summary = cache.fill(pairs, workers=workers, isolate=args.isolate_pairs,
                         progress=lambda i, n: print(f"pairs {i}/{n}", flush=True) if i % 1000 == 0 else None)
    summary["pairs_needed"] = len(pairs)
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK
    _ = LEGACY_ROOT


def _stage_models(run_root: Path, which: str) -> int:
    import importlib
    try:
        module = importlib.import_module(f"agent_allocation.passk.models.{which}")
    except ModuleNotFoundError as exc:
        print(f"blocked: passk.models.{which} not available ({exc})", file=sys.stderr)
        return EXIT_BLOCKED
    module.fit_and_predict(run_root)
    return EXIT_OK


# --------------------------------------------------------------------------- Gate P sandbox (bwrap)
# strace/ptrace is blocked on the compute nodes (probe 23919235), and an in-process audit hook cannot see
# pyarrow's C-level reads, so neither is accepted as the fence. The fit stages (and the seal) instead re-exec
# themselves inside a bwrap mount namespace in which the fenced data (artifacts/shards, artifacts/private, the HF
# cache, the legacy root) does not exist: a forbidden read fails with ENOENT instead of depending on a log.
SANDBOX_STAGES = ("e2-fit", "e5-fit", "seal")
SANDBOX_SYSTEM = ("/usr", "/lib", "/lib64", "/bin", "/etc")
SANDBOX_ENV_ROOT = "/orcd/home/002/mabdel03/conda_envs/consortium"
CODE_ROOT = Path(__file__).resolve().parents[4]          # the worktree (holds agent_allocation/.venv + src)
SANDBOX_RECORD_VERSION = "passk-sandbox-run-v1"
_HASH_ENV = "PASSK_SANDBOX_ARGV_SHA256"


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_sha256(argv: list[str]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(list(argv)).encode("utf-8")).hexdigest()


def sandbox_mounts(run_root: Path, *, code_root: Path = CODE_ROOT, env_root: str = SANDBOX_ENV_ROOT) -> list[tuple]:
    """(flag, source) pairs, in order. Only these paths exist inside the sandbox (plus tmpfs /tmp, /proc, /dev)."""
    rr = Path(run_root).resolve()
    mounts = [("--ro-bind", p) for p in SANDBOX_SYSTEM if Path(p).exists()]
    mounts += [("--ro-bind", str(env_root)), ("--ro-bind", str(Path(code_root).resolve())),
               ("--ro-bind", str(rr / "analysis_inputs"))]
    if (rr / "gates").is_dir():
        mounts.append(("--ro-bind", str(rr / "gates")))
    splits = rr / "artifacts" / "manifests" / "splits.json"       # gold-free panels/strata (Gate S); the fits read it
    if splits.is_file():
        mounts.append(("--ro-bind", str(splits)))
    mounts.append(("--bind", str(rr / "sealed")))
    return mounts


def allowed_sandbox_sources(run_root: Path, *, code_root: Path = CODE_ROOT,
                            env_root: str = SANDBOX_ENV_ROOT) -> dict[str, set[str]]:
    rr = Path(run_root).resolve()
    ro = set(SANDBOX_SYSTEM) | {str(env_root), str(Path(code_root).resolve()), str(rr / "analysis_inputs"),
                                str(rr / "gates"), str(rr / "artifacts" / "manifests" / "splits.json")}
    return {"--ro-bind": ro, "--bind": {str(rr / "sealed")}}


def sandbox_argv(run_root: Path, inner: list[str], *, python: str | None = None, code_root: Path = CODE_ROOT,
                 env_root: str = SANDBOX_ENV_ROOT) -> list[str]:
    """The exact bwrap command (without the argv-hash setenv, which is appended by ``_run_sandboxed``)."""
    rr = Path(run_root).resolve()
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev"]
    for flag, src in sandbox_mounts(rr, code_root=code_root, env_root=env_root):   # after /tmp: binds win
        cmd += [flag, src, src]
    cmd += ["--chdir", str(Path(code_root).resolve()),
            "--setenv", "PASSK_SANDBOXED", "1", "--setenv", "HOME", "/tmp", "--setenv", "MPLCONFIGDIR", "/tmp/mpl",
            "--setenv", "HF_HUB_OFFLINE", "1"]
    cmd += [python or sys.executable, "-m", "agent_allocation.cli", "passk-analyze", *inner,
            "--run-root", str(rr)]
    return cmd


def check_sandbox_argv(argv: list[str], run_root: Path, **kw) -> list[str]:
    """Problems with a recorded bwrap argv (empty list = it is the fence). Used by the seal."""
    problems = []
    if not argv or Path(argv[0]).name != "bwrap":
        return ["not a bwrap command"]
    for flag in ("--unshare-all", "--die-with-parent"):
        if flag not in argv:
            problems.append(f"missing {flag}")
    allowed = allowed_sandbox_sources(run_root, **kw)
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try", "--dev-bind", "--dev-bind-try"):
            src, dst = argv[i + 1], argv[i + 2]
            if tok not in allowed or src not in allowed[tok] or src != dst:
                problems.append(f"mount {tok} {src} -> {dst} is not on the allow-list")
            i += 3
            continue
        if tok in ("--setenv",):
            i += 3
            continue
        if tok in ("--tmpfs", "--proc", "--dev", "--chdir", "--unsetenv"):
            if tok == "--tmpfs" and argv[i + 1] != "/tmp":
                problems.append(f"tmpfs at {argv[i + 1]}")
            i += 2
            continue
        if tok.startswith("--"):
            if tok not in ("--unshare-all", "--die-with-parent", "--new-session"):
                problems.append(f"unexpected bwrap option {tok}")
            i += 1
            continue
        break                                                       # the program starts here
    return problems


def _code_commit() -> tuple[str | None, bool]:
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head, bool(dirty)
    except (OSError, Exception):                                   # noqa: BLE001 - git missing or not a repo
        return None, True


def _sealed_hashes(run_root: Path) -> dict[str, str]:
    out = {}
    sealed = Path(run_root) / "sealed"
    if sealed.is_dir():
        for path in sorted(sealed.rglob("*")):
            rel = str(path.relative_to(run_root))
            if (path.is_file() and not path.name.startswith(".") and path.name != "SEAL_predictions.json"
                    and not rel.startswith("sealed/runs/")):
                out[rel] = _sha256_file(path)
    return out


def run_record_path(run_root: Path, stage: str) -> Path:
    return Path(run_root) / "sealed" / "runs" / f"{stage}.json"


def verify_run_record(run_root: Path, stage: str) -> tuple[dict | None, list[str]]:
    """The sandboxed-run record of a fit stage, and its problems (rc, argv hash, outputs still match)."""
    path = run_record_path(run_root, stage)
    if not path.exists():
        return None, [f"{path} missing (run --stage {stage} --sandbox)"]
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{path} unreadable: {exc}"]
    problems = []
    if rec.get("version") != SANDBOX_RECORD_VERSION or rec.get("sandbox") != "bwrap":
        problems.append(f"{stage}: not a bwrap sandbox record")
    if rec.get("rc") != 0:
        problems.append(f"{stage}: sandboxed run exited {rec.get('rc')}")
    if argv_sha256(rec.get("argv") or []) != rec.get("argv_sha256"):
        problems.append(f"{stage}: argv hash mismatch")
    problems += [f"{stage}: {p}" for p in check_sandbox_argv(rec.get("argv") or [], run_root)]
    outputs = rec.get("outputs") or {}
    if stage != "seal" and not outputs:
        problems.append(f"{stage}: the sandboxed run wrote nothing")
    for rel, sha in outputs.items():
        target = Path(run_root) / rel
        if not target.is_file() or _sha256_file(target) != sha:
            problems.append(f"{stage}: {rel} changed or missing since the sandboxed run")
    return rec, problems


def _run_sandboxed(run_root: Path, stage: str, argv: list[str]) -> int:
    """Re-exec ``passk-analyze --stage <stage>`` inside bwrap; record the exact argv in sealed/runs/<stage>.json."""
    import shutil
    import socket
    import subprocess
    from datetime import datetime, timezone

    run_root = Path(run_root).resolve()
    if stage not in SANDBOX_STAGES:
        print(f"--sandbox applies to {SANDBOX_STAGES} only", file=sys.stderr)
        return EXIT_FAIL
    if shutil.which("bwrap") is None:
        print("bwrap not found on this node; the Gate P fence cannot run here", file=sys.stderr)
        return EXIT_FAIL
    if not (run_root / "analysis_inputs").is_dir():
        print(f"blocked: {run_root}/analysis_inputs missing", file=sys.stderr)
        return EXIT_BLOCKED
    if stage != "seal":
        rec, problems = verify_run_record(run_root, stage)
        if rec is not None and not problems:
            print(json.dumps({"stage": stage, "sandboxed": "already done", "outputs": len(rec["outputs"])}))
            return EXIT_OK                                           # resumable: a verified fit is a no-op
        if rec is not None:
            print(f"{stage}: an earlier sandboxed record does not verify: {problems}", file=sys.stderr)
            return EXIT_FAIL
    (run_root / "sealed").mkdir(parents=True, exist_ok=True)
    head, dirty = _code_commit()
    inner = [a for a in argv if a not in ("--sandbox", "--strace")]
    cmd = sandbox_argv(run_root, inner)
    digest = argv_sha256(cmd)
    env_extra = ["--setenv", _HASH_ENV, digest, "--setenv", "PASSK_CODE_COMMIT", head or "",
                 "--setenv", "PASSK_CODE_DIRTY", "1" if dirty else "0"]
    prog = cmd.index("--chdir")                                      # setenvs go among the bwrap options
    full = cmd[:prog] + env_extra + cmd[prog:]
    before = _sealed_hashes(run_root)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proc = subprocess.run(full, check=False)
    after = _sealed_hashes(run_root)
    outputs = {rel: sha for rel, sha in after.items() if before.get(rel) != sha}
    rec = {"version": SANDBOX_RECORD_VERSION, "stage": stage, "sandbox": "bwrap", "argv": cmd, "argv_sha256": digest,
           "exec_argv": full, "code_commit": head, "worktree_dirty": dirty, "rc": proc.returncode,
           "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "host": socket.gethostname(), "outputs": outputs}
    attempts = run_root / "logs" / "sandbox"
    attempts.mkdir(parents=True, exist_ok=True)
    (attempts / f"{stage}.{started.replace(':', '')}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    if proc.returncode == 0 and stage != "seal":
        target = run_record_path(run_root, stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    return proc.returncode


def cmd_passk_analyze(argv: list[str]) -> int:
    from agent_allocation.passk.extract import LEGACY_ROOT, RUN_ROOT

    parser = argparse.ArgumentParser(prog="allocation passk-analyze")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--legacy-root", default=str(LEGACY_ROOT))
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--release", action="append", default=[], choices=("v48", "t200"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--isolate-pairs", action="store_true",
                        help="pairs: evaluate each pair in a hard-killable child (a C-level stall cannot hang the fill)")
    parser.add_argument("--no-raw", action="store_true", help="extract: skip the THINK generation id (raw gz)")
    parser.add_argument("--sandbox", action="store_true",
                        help="e2-fit/e5-fit/seal: re-exec this stage inside the Gate P bwrap mount namespace "
                             "(only analysis_inputs/ ro, gates/ ro, sealed/ rw, code + env ro); the exact argv is "
                             "recorded in sealed/runs/<stage>.json")
    parser.add_argument("--strace", action="store_true",
                        help="deprecated alias of --sandbox (ptrace is blocked on the compute nodes)")
    args = parser.parse_args(argv)
    run_root = Path(args.run_root)
    if args.stage in ("extract", "release"):
        from agent_allocation.passk.extract import extract
        if args.stage == "release" and not args.release:
            parser.error("--stage release needs --release v48|t200")
        manifest = extract(run_root, legacy_root=Path(args.legacy_root), release=args.release,
                           read_raw=not args.no_raw)
        print(json.dumps({k: manifest[k] for k in ("n_rows", "partitions_read", "labelled_partitions",
                                                   "sha256")}, sort_keys=True))
        return EXIT_OK
    if args.stage == "pairs":
        return _stage_pairs(run_root, args)
    if args.stage == "e1-lock":
        try:
            payload = e1_lock(run_root)
        except NotReady as exc:
            print(f"blocked: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
        print(json.dumps({"selected": payload["selected"]}))
        return EXIT_OK
    if (args.sandbox or args.strace) and args.stage in SANDBOX_STAGES and os.environ.get("PASSK_SANDBOXED") != "1":
        return _run_sandboxed(run_root, args.stage, argv)
    if args.stage in ("e2-fit", "e5-fit"):
        return _stage_models(run_root, "e2" if args.stage == "e2-fit" else "e5")
    if args.stage == "seal":
        from agent_allocation.passk.seal import run_seal
        return run_seal(run_root)
    if args.stage == "score":
        from agent_allocation.passk.score import score
        primary = score(run_root)
        print(json.dumps({"locked": primary["locked"], "holm": primary["holm"]}, default=str))
        return EXIT_OK
    if args.stage == "report":
        import subprocess
        build = Path(__file__).resolve().parents[3] / "docs" / "passk_cont_report" / "build.sh"
        env = dict(os.environ, RUN=str(run_root), PY=sys.executable)
        return subprocess.run(["bash", str(build)], env=env, check=False).returncode
    if args.stage == "e0":
        from agent_allocation.passk.e0.run import run_e0
        run_e0(run_root, legacy_root=Path(args.legacy_root))
        return EXIT_OK
    print(f"blocked: stage {args.stage} is not implemented yet", file=sys.stderr)
    return EXIT_BLOCKED
