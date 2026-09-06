"""WP1 — data/: eligibility fixtures, salted-hash splits, nested panels, export CLI, firewall.

Synthetic parquet fixtures are written into ``tmp_path`` so the unit tests never touch
the real snapshot; the ``test_real_*`` tests read the raw exports under the study_v4 run
root and skip when they are absent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from pathlib import Path

import pytest

from agents_scaling.study.config import load_config
from agents_scaling.study.data import bcb as B
from agents_scaling.study.data import export as X
from agents_scaling.study.data import hle as H
from agents_scaling.study.data import protected as P
from agents_scaling.study.data import public as PUB
from agents_scaling.study.data import splits as S
from agents_scaling.study.types import Domain, ProtocolError, PublicTask

REAL_RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4")
REAL_HLE = REAL_RUN_ROOT / "data" / X.RAW_HLE_SHARDS
REAL_BCB = REAL_RUN_ROOT / "data" / X.RAW_BCB_PARQUET


# ----------------------------------------------------------------------------- fixtures


def hle_row(i: int, cls: str = "Gold subset", *, image: str = "", preview=None, answer_type: str = "multipleChoice", question=None):
    hid = f"id{i:04d}"
    q = question if question is not None else f"Question {i} text here? Answer Choices: A. x B. y"
    blob = {
        "id": hid, "question": q, "image": image, "image_preview": preview, "answer": "A", "answer_type": answer_type,
        "rationale": f"because {i}", "raw_subject": "Math", "category": "Math", "Verified_Classes": cls,
        "canary": "x", "author_name": "a", "rationale_image": None, "verify_meta_info": "{}",
    }
    return {
        "id": hid, "Verified_Classes": cls, "category": "Math", "raw_subject": "Math", "problem_is_valid": "1",
        "problem_error_type": "0", "answer_is_valid": "1", "answer_error_type": "0", "rationale_is_valid": "1",
        "rationale_error_type": "0", "question": q, "answer": "B" if i % 3 == 0 else "A", "json": json.dumps(blob),
    }


def bcb_row(i: int):
    return {
        "task_id": f"BigCodeBench/{i}", "complete_prompt": "import x\n", "instruct_prompt": f"Write task {i} function.",
        "canonical_solution": f"    return {i}\n", "code_prompt": "def task_func():\n", "test": f"import unittest # {i}",
        "entry_point": "task_func", "doc_struct": "{}", "libs": "['random', 'itertools']",
    }


def write_parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def small_config(tmp_path):
    """The committed yaml with tiny item counts (dev 2+2, main 3+3) for synthetic fixtures."""
    import yaml

    raw = yaml.safe_load(load_config().source_path.read_text(encoding="utf-8"))
    raw["items"]["dev"] = {"hle": 2, "bcb": 2}
    raw["items"]["main"] = {"hle": 3, "bcb": 3}
    raw["items"]["panels"] = {"N": 2, "E": 1}
    path = tmp_path / "study_small.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(path)


@pytest.fixture
def synthetic_raw(tmp_path):
    """8 HLE rows (6 eligible) in two shards + 6 BCB rows."""
    run_root = tmp_path / "run"
    long_question = " ".join(["word"] * 4097)
    shard0 = [hle_row(1), hle_row(2, "Revision subset"), hle_row(3, "Uncertain subset"), hle_row(4, image="data:image/png;base64,AAA")]
    shard1 = [hle_row(5, preview={"bytes": "..."}), hle_row(6, "Revision subset", answer_type="exactMatch"), hle_row(1), hle_row(7, question=long_question), hle_row(8), hle_row(9, "Revision subset"), hle_row(10)]
    write_parquet(run_root / "data" / X.RAW_HLE_SHARDS / "0.parquet", shard0)
    write_parquet(run_root / "data" / X.RAW_HLE_SHARDS / "1.parquet", shard1)
    write_parquet(run_root / "data" / X.RAW_BCB_PARQUET, [bcb_row(i) for i in range(6)])
    return run_root


# ----------------------------------------------------------------------------- HLE


def test_hle_eligibility_reasons_in_frozen_order(stub_tokenizer):
    ok, reason = H.hle_eligible(hle_row(1), task_tokens=10)
    assert (ok, reason) == (True, "eligible")
    assert H.hle_eligible(hle_row(2, "Revision subset"), task_tokens=10) == (True, "eligible")
    assert H.hle_eligible(hle_row(3, "Uncertain subset"), task_tokens=10) == (False, "not_gold_or_revision")
    assert H.hle_eligible(hle_row(4, image="x"), task_tokens=10) == (False, "image_dependent")
    assert H.hle_eligible(hle_row(5, preview={"bytes": "b"}), task_tokens=10) == (False, "image_dependent")
    assert H.hle_eligible(hle_row(1), task_tokens=10, seen_ids={"id0001"}) == (False, "duplicate")
    assert H.hle_eligible(hle_row(1), task_tokens=4097) == (False, "envelope")
    assert H.hle_eligible(hle_row(1), task_tokens=4096) == (True, "eligible")
    # order: class before image before duplicate before envelope
    assert H.hle_eligible(hle_row(3, "Uncertain subset", image="x"), task_tokens=9999, seen_ids={"id0003"}) == (False, "not_gold_or_revision")
    assert H.hle_eligible(hle_row(4, image="x"), task_tokens=9999, seen_ids={"id0004"}) == (False, "image_dependent")
    assert H.hle_eligible(hle_row(1), task_tokens=9999, seen_ids={"id0001"}) == (False, "duplicate")
    with pytest.raises(TypeError):
        H.hle_eligible(hle_row(1), task_tokens=None)  # type: ignore[arg-type]


def test_is_image_dependent_literals():
    assert not H.is_image_dependent({"image": "", "image_preview": None})
    assert not H.is_image_dependent({"image": "", "image_preview": "None"})
    assert H.is_image_dependent({"image": "abc", "image_preview": None})
    assert H.is_image_dependent({"image": "", "image_preview": {"bytes": "x"}})


def test_hle_views_and_strata(stub_tokenizer):
    gold = H.hle_public_task(hle_row(1), 7)
    rev = H.hle_public_task(hle_row(2, "Revision subset", answer_type="exactMatch"), 8)
    assert gold.stratum == "Gold" and rev.stratum == "Revision"
    assert gold.source_id == "hle:id0001" and gold.domain is Domain.HLE
    assert gold.answer_format == "multipleChoice" and rev.answer_format == "exactMatch"
    assert gold.split == "unassigned" and gold.rank == -1 and gold.task_tokens == 7 and gold.category == "Math"
    label = H.hle_protected_label(hle_row(3))
    assert label == H.ProtectedLabel("hle:id0003", "B", "multipleChoice", "because 3")  # revised top-level answer
    for key in ("answer", "json", "rationale"):
        assert key not in gold.to_dict()
    bad = hle_row(1)
    bad["json"] = json.dumps({**json.loads(bad["json"]), "answer_type": "essay"})
    with pytest.raises(H.HleRowError):
        H.hle_public_task(bad, 1)
    bad["json"] = "not json"
    with pytest.raises(H.HleRowError):
        H.parse_json_blob(bad)


def test_build_hle_records_every_exclusion(synthetic_raw, stub_tokenizer):
    rows = H.load_hle_rows(synthetic_raw / "data" / X.RAW_HLE_SHARDS)
    assert [r["id"] for r in rows][:4] == ["id0001", "id0002", "id0003", "id0004"]  # shard order 0 then 1
    build = H.build_hle(rows, stub_tokenizer)
    assert [t.source_id for t in build.tasks] == ["hle:id0001", "hle:id0002", "hle:id0006", "hle:id0008", "hle:id0009", "hle:id0010"]
    assert set(build.labels) == {t.source_id for t in build.tasks}
    reasons = {e["source_id"]: e["reason"] for e in build.exclusions}
    assert reasons == {
        "hle:id0003": "not_gold_or_revision", "hle:id0004": "image_dependent", "hle:id0005": "image_dependent",
        "hle:id0001": "duplicate", "hle:id0007": "envelope",
    }
    assert next(e for e in build.exclusions if e["reason"] == "envelope")["task_tokens"] == 4097
    with pytest.raises(FileNotFoundError):
        H.load_hle_rows(synthetic_raw / "nowhere")


# ----------------------------------------------------------------------------- BCB


def test_bcb_views(synthetic_raw, stub_tokenizer):
    rows = B.load_bcb_rows(synthetic_raw / "data" / X.RAW_BCB_PARQUET, expected_rows=6)
    with pytest.raises(B.BcbRowError):
        B.load_bcb_rows(synthetic_raw / "data" / X.RAW_BCB_PARQUET)  # default expects 1,140
    build = B.build_bcb(rows, stub_tokenizer)
    assert len(build.tasks) == 6 and not build.exclusions
    task = build.tasks[0]
    assert task.source_id == "bcb:BigCodeBench/0" and task.answer_format == "code" and task.stratum == "bcb"
    assert task.entry_point == "task_func" and task.task_text == "Write task 0 function."
    prot = build.protected[task.source_id]
    assert prot.libs == ("random", "itertools") and prot.test.startswith("import unittest")
    for key in ("test", "canonical_solution", "code_prompt"):
        assert key not in task.to_dict()
    assert B.parse_libs("[]") == () and B.parse_libs(["a"]) == ("a",)
    with pytest.raises(B.BcbRowError):
        B.parse_libs("random, itertools")
    with pytest.raises(B.BcbRowError):
        B.build_bcb(rows + [rows[0]], stub_tokenizer)  # duplicate task_id is an error, not an exclusion
    long_rows = [dict(rows[0], task_id="BigCodeBench/long", instruct_prompt=" ".join(["w"] * 5000))]
    assert B.build_bcb(long_rows, stub_tokenizer).exclusions[0]["reason"] == "envelope"


# ----------------------------------------------------------------------------- splits


def test_rank_key_is_hmac_sha256():
    salt = bytes(range(32))
    expected = hmac.new(salt, b"rank:hle:abc", hashlib.sha256).digest()
    assert S.rank_key(salt, "hle:abc") == expected
    assert S.rank_key(salt, "hle:abc") != S.rank_key(salt, "hle:abd")
    assert S.rank_key(b"other", "hle:abc") != expected
    with pytest.raises(ValueError):
        S.rank_key(b"", "hle:abc")
    with pytest.raises(ValueError):
        S.rank_key(salt, "")


def make_tasks(n_hle: int, n_bcb: int) -> list[PublicTask]:
    out = []
    for i in range(n_hle):
        stratum = "Gold" if i % 2 else "Revision"
        out.append(PublicTask(f"hle:{i}", Domain.HLE, "unassigned", f"q{i}", "exactMatch", stratum, -1, 5))
    for i in range(n_bcb):
        out.append(PublicTask(f"bcb:{i}", Domain.BCB, "unassigned", f"c{i}", "code", "bcb", -1, 5, entry_point="f"))
    return out


def test_assign_splits_counts_order_and_determinism(small_config):
    tasks = make_tasks(9, 7)
    assigned = S.assign_splits(tasks, small_config)
    counts = {(t.domain.value, t.split): 0 for t in assigned}
    for t in assigned:
        counts[(t.domain.value, t.split)] += 1
    assert counts == {("hle", "dev"): 2, ("hle", "main"): 3, ("hle", "reserve"): 4, ("bcb", "dev"): 2, ("bcb", "main"): 3, ("bcb", "reserve"): 2}
    # rank restarts inside each split; order is domain, split, rank
    assert [t.rank for t in assigned if t.domain is Domain.HLE] == [0, 1, 0, 1, 2, 0, 1, 2, 3]
    assert [t.domain.value for t in assigned] == ["hle"] * 9 + ["bcb"] * 7
    # deterministic and input-order independent
    shuffled = list(reversed(tasks))
    assert S.split_assignment(shuffled, small_config) == S.split_assignment(tasks, small_config)
    # the global order is the HMAC order within each domain
    salt = small_config.split_salt
    hle_sorted = sorted((t for t in tasks if t.domain is Domain.HLE), key=lambda t: S.rank_key(salt, t.source_id))
    assert [t.source_id for t in assigned if t.domain is Domain.HLE] == [t.source_id for t in hle_sorted]
    # SRS: strata are not balanced
    assert {t.stratum for t in assigned if t.split == "dev" and t.domain is Domain.HLE} <= {"Gold", "Revision"}


def test_assign_splits_errors(small_config):
    with pytest.raises(S.SplitError):
        S.assign_splits(make_tasks(4, 7), small_config)  # hle needs 5
    dup = make_tasks(9, 7) + [make_tasks(1, 0)[0]]
    with pytest.raises(S.SplitError):
        S.assign_splits(dup, small_config)


def test_panel_items_are_nested_prefixes(small_config):
    assigned = S.assign_splits(make_tasks(9, 7), small_config)
    main = [t for t in assigned if t.split == "main"]
    p1 = S.panel_items(main, 1)
    p2 = S.panel_items(reversed(main), 2)
    p3 = S.panel_items(main, 3)
    assert [t.source_id for t in p1] == [p2[0].source_id, p2[2].source_id]
    assert {t.source_id for t in p2} <= {t.source_id for t in p3}
    assert all(t.rank < 2 for t in p2) and len(p2) == 4
    with pytest.raises(S.SplitError):
        S.panel_items(main, 4)
    with pytest.raises(S.SplitError):
        S.panel_items(assigned, 1)  # contains dev/reserve
    with pytest.raises(ValueError):
        S.panel_items(main, 0)
    summary = S.split_summary(assigned, {"N": 2})
    assert summary["counts"]["hle"] == {"dev": 2, "main": 3, "reserve": 4}
    assert len(summary["domains"]["bcb"]["main"]) == 3 and summary["panels"] == {"N": 2}


# ----------------------------------------------------------------------------- export + loaders


def test_export_end_to_end(synthetic_raw, small_config, stub_tokenizer):
    summary = X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, expected_bcb_rows=6)
    assert summary.eligible == {"hle": 6, "bcb": 6}
    assert summary.counts == {"hle": {"dev": 2, "main": 3, "reserve": 1}, "bcb": {"dev": 2, "main": 3, "reserve": 1}}
    assert summary.exclusions == {"hle": {"not_gold_or_revision": 1, "image_dependent": 2, "duplicate": 1, "envelope": 1}, "bcb": {}}
    data = synthetic_raw / "data"
    # permissions of the protected export
    assert stat.S_IMODE((data / "protected").stat().st_mode) == 0o700
    for name in ("hle_labels.jsonl", "bcb_tests.jsonl"):
        assert stat.S_IMODE((data / "protected" / name).stat().st_mode) == 0o600
    # public rows carry only public keys
    rows = [json.loads(l) for l in (data / "public" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    for row in rows:
        assert not (set(row) & PUB.PROTECTED_KEYS)
        assert set(row) == PUB.PUBLIC_KEYS
    # loaders
    tasks = PUB.load_public_tasks(synthetic_raw)
    assert len(tasks) == 12 and len(PUB.load_public_tasks(synthetic_raw, "dev")) == 4
    assert {t.source_id for t in PUB.load_public_tasks(synthetic_raw, "main")} == set(
        json.loads((data / "splits.json").read_text())["domains"]["hle"]["main"]
        + json.loads((data / "splits.json").read_text())["domains"]["bcb"]["main"]
    )
    labels = P.load_protected_hle(synthetic_raw)
    tests = P.load_protected_bcb(synthetic_raw)
    assert set(labels) == {t.source_id for t in tasks if t.domain is Domain.HLE}
    assert set(tests) == {t.source_id for t in tasks if t.domain is Domain.BCB}
    assert labels["hle:id0006"].answer_type == "exactMatch"
    # exclusions and hashes
    excl = [json.loads(l) for l in (data / "exclusions.jsonl").read_text().splitlines()]
    assert len(excl) == 5 and all("excluded_at" in e and e["reason"] in H.EXCLUSION_REASONS for e in excl)
    hashes = json.loads((data / "DATA_SHA256.json").read_text())
    assert {"raw/hle_shards/0.parquet", "raw/hle_shards/1.parquet", "raw/bigcodebench_v0.1.4.parquet", "public/tasks.jsonl", "protected/hle_labels.jsonl", "splits.json"} <= set(hashes)
    assert hashes["public/tasks.jsonl"] == hashlib.sha256((data / "public" / "tasks.jsonl").read_bytes()).hexdigest()
    # refuses to overwrite silently
    with pytest.raises(ProtocolError):
        X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, expected_bcb_rows=6)
    X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, force=True, expected_bcb_rows=6)
    with pytest.raises(ValueError):
        PUB.load_public_tasks(synthetic_raw, "test")


def test_public_loader_firewall(synthetic_raw, small_config, stub_tokenizer):
    X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, expected_bcb_rows=6)
    path = PUB.public_tasks_path(synthetic_raw)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for leak in ("answer", "json", "test", "canonical_solution"):
        rows[0][leak] = "LEAK"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ProtocolError):
            PUB.load_public_tasks(synthetic_raw)
        del rows[0][leak]
    rows.append(rows[0])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ProtocolError):
        PUB.load_public_tasks(synthetic_raw)  # duplicate id
    with pytest.raises(FileNotFoundError):
        PUB.load_public_tasks(synthetic_raw / "missing")


def test_protected_loader_refuses_open_permissions(synthetic_raw, small_config, stub_tokenizer):
    X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, expected_bcb_rows=6)
    path = P.protected_dir(synthetic_raw) / P.HLE_LABELS_FILE
    os.chmod(path, 0o644)
    with pytest.raises(ProtocolError):
        P.load_protected_hle(synthetic_raw)
    os.chmod(path, 0o600)
    assert P.load_protected_hle(synthetic_raw)
    with pytest.raises(FileNotFoundError):
        P.load_protected_bcb(synthetic_raw / "missing")


def test_preflight_report_and_failure(synthetic_raw, small_config, stub_tokenizer):
    X.export_data(synthetic_raw, small_config, tokenizer=stub_tokenizer, expected_bcb_rows=6)
    toks = {size: stub_tokenizer for size in small_config.checkpoints}
    report = X.preflight(synthetic_raw, small_config, tokenizers=toks)
    assert report["ok"] and report["n_items"] == 10 and report["items_by_split"] == {"dev": 4, "main": 6}
    for size in small_config.checkpoints:
        block = report["checkpoints"][size]
        assert block["task_tokens"]["n_over_cap"] == 0 and block["exported_task_tokens_mismatch"] == []
        assert set(block["cells"]) == set(X.ROOT_CELLS)
        for cell in X.ROOT_CELLS:
            c = block["cells"][cell]
            assert c["prompt_tokens"]["n_over_cap"] == 0 and c["prompt_tokens"]["max"] >= c["prompt_tokens"]["p50"]
            assert c["root_prompt_max_tokens"] == 4096 + c["wrapper_tokens_max"]
    assert (synthetic_raw / "data" / X.PREFLIGHT_FILE).exists()
    with pytest.raises(ProtocolError):
        X.preflight(synthetic_raw, small_config, tokenizers=toks, prompt_cap=10)
    assert json.loads((synthetic_raw / "data" / X.PREFLIGHT_FILE).read_text())["ok"] is False


def test_percentile_nearest_rank():
    assert X._percentile([1, 2, 3, 4], 50) == 2
    assert X._percentile([5], 99) == 5
    assert X._percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    with pytest.raises(ValueError):
        X._percentile([], 50)


def test_cli_main(synthetic_raw, small_config, stub_tokenizer, monkeypatch, capsys):
    monkeypatch.setattr("agents_scaling.study.inference.tokens.load_tokenizer", lambda ckpt: stub_tokenizer)
    rc = X.main([
        "--run-id", synthetic_raw.name, "--results-root", str(synthetic_raw.parent),
        "--config", str(small_config.source_path), "--preflight", "--expected-bcb-rows", "6",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"eligible"' in out and '"preflight_ok": true' in out
    assert PUB.public_tasks_path(synthetic_raw).exists()


# ----------------------------------------------------------------------------- real snapshot


@pytest.fixture(scope="module")
def real_rows():
    if not REAL_HLE.exists() or not REAL_BCB.exists():
        pytest.skip("raw HLE/BCB snapshot not present")
    return H.load_hle_rows(REAL_HLE), B.load_bcb_rows(REAL_BCB)


def test_real_snapshot_counts(real_rows, real_tokenizer_32b):
    hle_rows, bcb_rows = real_rows
    assert len(hle_rows) == 2500 and len(bcb_rows) == 1140
    assert len({r["id"] for r in hle_rows}) == 2500
    build = H.build_hle(hle_rows, real_tokenizer_32b)
    reasons: dict[str, int] = {}
    for e in build.exclusions:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    assert reasons["not_gold_or_revision"] == 689 and reasons["image_dependent"] == 211
    assert "duplicate" not in reasons
    assert len(build.tasks) + reasons.get("envelope", 0) == 1600
    assert len(build.tasks) >= 950  # §3.2 preflight minimum for 150 + 800
    assert all(t.task_tokens <= 4096 for t in build.tasks)
    strata = {s: sum(1 for t in build.tasks if t.stratum == s) for s in ("Gold", "Revision")}
    assert strata["Gold"] + strata["Revision"] == len(build.tasks)
    formats = {t.answer_format for t in build.tasks}
    assert formats == {"multipleChoice", "exactMatch"}
    bcb = B.build_bcb(bcb_rows, real_tokenizer_32b)
    assert len(bcb.tasks) == 1140 and not bcb.exclusions
    assert all(t.entry_point == "task_func" for t in bcb.tasks)


def test_real_export_present_and_consistent(study_config):
    if not PUB.public_tasks_path(REAL_RUN_ROOT).exists():
        pytest.skip("real export not written yet")
    tasks = PUB.load_public_tasks(REAL_RUN_ROOT)
    for domain, want in ((Domain.HLE, study_config.items.dev.hle), (Domain.BCB, study_config.items.dev.bcb)):
        assert sum(1 for t in tasks if t.domain is domain and t.split == "dev") == want
    for domain, want in ((Domain.HLE, study_config.items.main.hle), (Domain.BCB, study_config.items.main.bcb)):
        main = [t for t in tasks if t.domain is domain and t.split == "main"]
        assert len(main) == want and sorted(t.rank for t in main) == list(range(want))
    reserve = [t for t in tasks if t.split == "reserve"]
    assert len(reserve) >= 0.1 * len(tasks)  # §3.3 reserve
    assigned = S.assign_splits([PublicTask.from_dict({**t.to_dict(), "split": "unassigned", "rank": -1}) for t in tasks], study_config)
    assert {(t.source_id, t.split, t.rank) for t in assigned} == {(t.source_id, t.split, t.rank) for t in tasks}
