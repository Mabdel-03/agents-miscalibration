"""WP-3 merge-shards: two shard run dirs holding disjoint items merge into exactly the single-run analysis of their
union (every summary, contrast, interval and cost table), with shard_id / endpoint_tag covariates added."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.analysis.merge import MergeError, merge_shards
from agent_allocation.config import load_config
from agent_allocation.paths import RunPaths
from agent_allocation.reporting.cli import analyze
from tests.unit.analysis.conftest import REPLICATES
from tests.unit.analysis.synthrun import Grid, SynthSpec, build_run, write_selection_batches

PROJECT = Path(__file__).resolve().parents[3]
TABLES = ("candidates", "policy_outcomes", "accuracy_summary", "primary_contrasts", "policy_costs", "cost_summary",
          "pass_at_k_vs_readout", "transitions_summary", "completeness", "node_completeness", "populations",
          "frontier_descriptive", "sample_bank_items", "transitions_items", "budget_replay_items",
          "budget_replay_curves", "operations")


def _config() -> dict:
    cfg = load_config(PROJECT / "configs" / "balanced150.json")
    cfg = copy.deepcopy(cfg)
    cfg["shards"] = [{"id": "alpha", "endpoint_tag": "prov-a/fp8", "share": 1},
                     {"id": "beta", "endpoint_tag": "prov-b/fp8", "share": 1}]
    return cfg


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def split_run(src: RunPaths, dest: Path, keep: set[str], shard: dict) -> RunPaths:
    """A shard run dir holding only ``keep`` items of a finished run: filtered manifests, per-node files, re-sealed
    selection batches and the ledger slice -- what a shard that collected those items itself would hold."""
    paths = RunPaths.for_run(dest, local_state_dir=dest / "state")
    paths.ensure()
    for bench_file in (src.manifests_dir / "public").glob("*.jsonl"):
        _write(paths.manifests_dir / "public" / bench_file.name,
               [r for r in _jsonl(bench_file) if r["item_id"] in keep])
    panels = json.loads(src.panels.read_text())
    for draw in panels.values():
        for key in ("dev_item_ids", "main_item_ids", "reserve_item_ids"):
            draw[key] = [i for i in draw[key] if i in keep]
    paths.panels.write_text(json.dumps(panels, sort_keys=True))
    waves = json.loads(src.waves.read_text())
    for wave in waves:
        wave["item_ids"] = [i for i in wave["item_ids"] if i in keep]
    paths.waves.write_text(json.dumps([w for w in waves if w["item_ids"]], sort_keys=True))
    nodes: set[str] = set()
    for jobs in (src.manifests_dir / "jobs").glob("*.jsonl"):
        rows = [r for r in _jsonl(jobs) if r["record"]["item_id"] in keep]
        nodes |= {r["node_id"] for r in rows}
        _write(paths.jobs(jobs.stem), rows)
    for policies in (src.manifests_dir / "policies").glob("*.jsonl"):
        kept = [r for r in _jsonl(policies) if r["item_id"] in keep]
        _write(paths.policies(policies.stem), kept)
        kept_ids = {r["policy_id"] for r in kept}
        selections = []
        for batch in sorted(src.selections_dir(policies.stem).glob("batch_*.jsonl")):
            selections += [r for r in _jsonl(batch) if r["policy_id"] in kept_ids]
        write_selection_batches(paths, policies.stem, selections, n_batches=2)
    for sub in ("parsed", "grades", "public_checks", "raw"):
        for f in (src.run_dir / sub).rglob("*"):
            if f.is_file() and f.name.split(".")[0].split("_")[0] in nodes:
                target = paths.run_dir / f.relative_to(src.run_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    events = [e for e in _jsonl(src.ledger_journal) if e.get("node_id") in nodes]
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    _write(paths.ledger_journal, events)
    _write(paths.render_reports, [r for r in _jsonl(src.render_reports) if r["node_id"] in nodes])
    cfg = _config()
    from agent_allocation.shards import shard_config
    derived = shard_config(cfg, shard["id"])
    (paths.state_dir / "shard_config.json").write_text(json.dumps(derived, sort_keys=True))
    return paths


def _key_columns(frame: pd.DataFrame) -> list[str]:
    for keys in (["policy_id"], ["node_id"], ["partition_id", "name", "view", "level", "benchmark"],
                 ["partition_id", "name", "benchmark"], ["population"], ["partition_id", "item_id", "k"],
                 ["partition_id", "item_id", "trajectory", "from_step"]):
        if all(k in frame.columns for k in keys) and not frame.duplicated(keys).any():
            return keys
    return list(frame.columns)


# Provenance, not results: the covariates merge adds, and the (renumbered, per-shard) selection batch a row was sealed
# in -- a single run seals the same selections in differently cut batches.
NOT_COMPARED = ("shard_id", "endpoint_tag", "selection_batch")


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop(columns=[c for c in NOT_COMPARED if c in frame.columns])
    keys = _key_columns(frame)
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)[sorted(frame.columns)]


@pytest.fixture(scope="module")
def shard_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("merge_root")
    union = build_run(root / "union_src", SynthSpec(waves=2, extension_waves=1, grid=Grid(team_sizes=(2, 4)),
                                                    provider_snapshots=False))
    items = sorted(json.loads(l)["item_id"] for f in (union.manifests_dir / "public").glob("*.jsonl")
                   for l in f.read_text().splitlines() if l.strip())
    from agent_allocation.shards import shard_assignment
    assignment = shard_assignment(_config(), items)
    run_root = root / "study"
    for shard in _config()["shards"]:
        split_run(union, run_root / "shards" / shard["id"], {i for i, s in assignment.items() if s == shard["id"]},
                  shard)
    single = analyze(union.run_dir, config=_config(), replicates=REPLICATES, figures=False)
    merged, manifest = merge_shards(run_root, config=_config(), replicates=REPLICATES, figures=False)
    return union, run_root, assignment, single, merged, manifest


def test_shards_are_disjoint_and_cover_the_union(shard_setup):
    union, run_root, assignment, _single, _merged, manifest = shard_setup
    per_shard = {s["shard_id"]: s["items"] for s in manifest["shards"]}
    assert sum(per_shard.values()) == len(assignment) == manifest["items"]
    assert set(per_shard) == {"alpha", "beta"} and all(v > 0 for v in per_shard.values())
    assert all(s["sealed_partitions"] == ["main"] for s in manifest["shards"])


@pytest.mark.parametrize("table", TABLES)
def test_merged_tables_equal_single_run_analysis(shard_setup, table):
    union, run_root, _assignment, _single, _merged, _manifest = shard_setup
    single_path = union.tables_dir / f"{table}.parquet"
    merged_path = run_root / "merged" / "tables" / f"{table}.parquet"
    assert single_path.is_file() == merged_path.is_file(), table
    if not single_path.is_file():
        pytest.skip(f"{table} not produced for this synthetic run")
    a, b = _normalized(pd.read_parquet(single_path)), _normalized(pd.read_parquet(merged_path))
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_merged_rows_carry_shard_covariates(shard_setup):
    _union, run_root, assignment, _single, _merged, _manifest = shard_setup
    for table in ("candidates", "policy_outcomes", "policy_costs"):
        frame = pd.read_parquet(run_root / "merged" / "tables" / f"{table}.parquet")
        assert frame["shard_id"].notna().all(), table
        assert (frame["shard_id"] == frame["item_id"].map(assignment)).all(), table
        tags = dict(zip(frame["shard_id"], frame["endpoint_tag"]))
        assert tags == {"alpha": "prov-a/fp8", "beta": "prov-b/fp8"}


def test_merge_refuses_overlapping_shards(shard_setup, tmp_path):
    union, run_root, assignment, *_ = shard_setup
    clone = tmp_path / "study"
    shutil.copytree(run_root / "shards", clone / "shards")
    shutil.copytree(clone / "shards" / "alpha", clone / "shards" / "gamma")
    with pytest.raises(MergeError):
        merge_shards(clone, config=_config(), replicates=REPLICATES, figures=False)


def test_merged_primary_contrast_matches_single_run(shard_setup):
    _union, _root, _assignment, single, merged, _manifest = shard_setup
    assert single.extras["primary_estimates"] == merged.extras["primary_estimates"]
    assert merged.run.seals["main"].valid
