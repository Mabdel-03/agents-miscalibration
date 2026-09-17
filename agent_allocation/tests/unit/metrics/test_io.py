"""Disk round trip: select + seal from public artifacts, then the sealed grade join through RunPaths."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from agent_allocation.contracts import dump_record
from agent_allocation.metrics import io as metrics_io
from agent_allocation.metrics.tables import build_partition_tables
from agent_allocation.paths import RunPaths
from agent_allocation.selection.partition import select_partition
from agent_allocation.selection.seal import SealError, load_sealed_selections
from tests.fixtures.metrics.synthetic_panel import SEED, build_panel, fixture_math_equivalent, standard_items


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


@pytest.fixture
def run(tmp_path):
    paths = RunPaths.for_run(tmp_path / "run", locks_dir=tmp_path / "locks", local_state_dir=tmp_path / "local")
    panel = build_panel(standard_items(), hashed_ids=True)
    by_benchmark: dict[str, list[str]] = {}
    policies = []
    for item in panel.records.items:
        by_benchmark.setdefault(item.task.benchmark, []).append(json.dumps(dump_record(item.task)))
        policies.extend(json.dumps(dump_record(cp)) for cp in item.checkpoints)
        for nid, candidate in item.candidates.items():
            _write(paths.parsed(nid), json.dumps({"result": dump_record(item.results[nid]),
                                                  "candidate": dump_record(candidate)}))
    for benchmark, lines in by_benchmark.items():
        _write(paths.public_manifest(benchmark), "\n".join(lines) + "\n")
    _write(paths.policies("main"), "\n".join(policies) + "\n")
    for cid, check in panel.public_checks.items():
        _write(paths.public_check(cid), json.dumps(dump_record(check)))
    for cid, grade in panel.grades.items():
        _write(paths.grade(cid), json.dumps(dump_record(grade)))
    return paths, panel


def test_a_parsed_record_without_candidate_is_an_empty_candidate(run):
    paths, panel = run
    item = panel.records.items[1]                                  # mmlu_pro/fixture-2
    nid = panel.node_ids[(item.task.item_id, "root/5")]
    paths.parsed(nid).write_text(json.dumps({"result": None, "candidate": None}))
    select_partition(paths, "main", seed=SEED, math_equivalent=fixture_math_equivalent)
    records = metrics_io.load_partition_records(paths, "main")
    loaded = next(i for i in records.items if i.task.item_id == item.task.item_id)
    assert loaded.candidates[nid].parse_status.value == "empty_content" and nid not in loaded.results
    grades = dict(panel.grades)
    grades.pop(nid)
    tables = build_partition_tables(load_sealed_selections(paths, "main"), records, grades)
    row = next(r for r in tables.checkpoints if r.item_id == item.task.item_id and r.name == "sample/6"
               and r.view == "current")
    assert row.pool_n_invalid == 2 and row.delivered_state != "missing_selection"


def test_unsealed_partition_refuses_before_reading_any_grade(run, monkeypatch):
    paths, _ = run

    def forbidden(*args, **kwargs):
        raise AssertionError("grades were read for an unsealed partition")

    monkeypatch.setattr(metrics_io, "load_grades", forbidden)
    with pytest.raises(SealError):
        metrics_io.load_partition_tables(paths, "main")


def test_select_partition_reads_no_private_or_grade_files(run, monkeypatch):
    paths, _ = run
    opened = []
    real_open = type(paths.run_dir).open
    real_read = type(paths.run_dir).read_text

    def spy_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    def spy_read(self, *args, **kwargs):
        opened.append(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(type(paths.run_dir), "open", spy_open)
    monkeypatch.setattr(type(paths.run_dir), "read_text", spy_read)
    select_partition(paths, "main", seed=SEED, math_equivalent=fixture_math_equivalent)
    assert opened and not any("/grades/" in p or "/private/" in p for p in opened)


def test_disk_round_trip_matches_the_in_memory_join(run, tmp_path):
    paths, panel = run
    selection = select_partition(paths, "main", seed=SEED, math_equivalent=fixture_math_equivalent)
    assert selection.sealed_sha256 and len(selection.pending_policy_ids) == 2      # cp/3 of the item missing cp/3/2
    assert all("/cp/3/" in p for p in selection.pending_policy_ids)
    again = select_partition(paths, "main", seed=SEED, math_equivalent=fixture_math_equivalent)
    assert again.sealed_sha256 == selection.sealed_sha256                          # idempotent re-seal

    from_disk = metrics_io.load_partition_tables(paths, "main")
    in_memory = build_partition_tables(load_sealed_selections(paths, "main"), panel.records, panel.grades)
    def same(a, b, strip=False):
        key = lambda r: repr(sorted(asdict(r).items()))  # noqa: E731
        if strip:
            a, b = ([replace(r, selection_cpu_seconds=None) for r in rows] for rows in (a, b))
        return sorted(a, key=key) == sorted(b, key=key)

    assert same(from_disk.checkpoints, in_memory.checkpoints, strip=True)
    assert same(from_disk.bank, in_memory.bank) and same(from_disk.team_transitions, in_memory.team_transitions)
    assert same(from_disk.slot_transitions, in_memory.slot_transitions)
    assert same(from_disk.root_strata, in_memory.root_strata)
    assert {r.generation_kind for r in from_disk.failures} == {"unknown"}          # no job manifest written

    digests = metrics_io.write_tables(from_disk, tmp_path / "tables")
    assert set(digests) == {"checkpoints.jsonl", "bank.jsonl", "slot_transitions.jsonl", "team_transitions.jsonl",
                            "root_strata.jsonl", "failures.jsonl", "summary_checkpoints.jsonl"}
    first = json.loads((tmp_path / "tables" / "checkpoints.jsonl").read_text().splitlines()[0])
    assert list(first) == sorted(asdict(from_disk.checkpoints[0]))
