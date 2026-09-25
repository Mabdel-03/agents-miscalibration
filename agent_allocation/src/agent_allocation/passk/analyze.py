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
    pools = build_pools(draws, panels=("L96",), configs=E1_CANDIDATES + ("1x16",))
    counts = pools.groupby("config_id")["item_id"].nunique().to_dict()
    n_items = draws.loc[draws["partition"] == "legacy_main", "item_id"].nunique()
    incomplete = {c: counts.get(c, 0) for c in E1_CANDIDATES if counts.get(c, 0) != n_items}
    if incomplete:
        raise RuntimeError(f"E1 lock needs every L96 item in each candidate allocation; incomplete: {incomplete}")
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
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    summary = cache.fill(pairs, workers=workers,
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


def cmd_passk_analyze(argv: list[str]) -> int:
    from agent_allocation.passk.extract import LEGACY_ROOT, RUN_ROOT

    parser = argparse.ArgumentParser(prog="allocation passk-analyze")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--legacy-root", default=str(LEGACY_ROOT))
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--release", action="append", default=[], choices=("v48", "t200"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-raw", action="store_true", help="extract: skip the THINK generation id (raw gz)")
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
        payload = e1_lock(run_root)
        print(json.dumps({"selected": payload["selected"]}))
        return EXIT_OK
    if args.stage in ("e2-fit", "e5-fit"):
        return _stage_models(run_root, "e2" if args.stage == "e2-fit" else "e5")
    if args.stage == "seal":
        from agent_allocation.passk.seal import run_seal
        return run_seal(run_root)
    if args.stage == "e0":
        from agent_allocation.passk.e0.run import run_e0
        run_e0(run_root, legacy_root=Path(args.legacy_root))
        return EXIT_OK
    print(f"blocked: stage {args.stage} is not implemented yet", file=sys.stderr)
    return EXIT_BLOCKED
