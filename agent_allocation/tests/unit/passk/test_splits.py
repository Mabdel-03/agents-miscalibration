"""passk_cont_v1 frozen splits: C48 rule, duplicate registry, T200 assignment, write-once files, and (when the
legacy run root and the pinned HF snapshots are present) the real build: determinism, exclusions, composition and
Gate S. Real-data tests only read the legacy run root; they write under pytest's tmp dir."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_allocation.benchmarks import aime, gpqa_diamond, hmmt, omni_math_hard
from agent_allocation.benchmarks.text_manifest import FIXTURE_REVISION, hf_snapshot_file
from agent_allocation.config import load_config
from agent_allocation.contracts import BENCHMARKS, STUDY_ID
from agent_allocation.passk import splits as S

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "passk_cont_v1.json"


def _real_inputs_available() -> bool:
    if not (S.LEGACY_RUN_ROOT / "manifests" / "panels.json").is_file():
        return False
    if not (S.LEGACY_RUN_ROOT / "private" / "aime.jsonl").is_file():
        return False
    sources = [(omni_math_hard.SOURCE_REPO, omni_math_hard.SOURCE_REVISION, omni_math_hard.SOURCE_FILE),
               (gpqa_diamond.SOURCE_REPO, gpqa_diamond.SOURCE_REVISION, gpqa_diamond.SOURCE_FILE)]
    sources += [(s.repo, s.revision, s.file) for m in (aime, hmmt) for s in m.SOURCES]
    return all(hf_snapshot_file(*s).is_file() for s in sources)


needs_real = pytest.mark.skipif(not _real_inputs_available(),
                                reason="legacy forced_budget_v2 run root or pinned HF snapshots absent")


# ---------------------------------------------------------------------------------------------------------------
# C48 rule on a fixture
# ---------------------------------------------------------------------------------------------------------------

def _bank(benchmark: str, sizes: dict[str, int], start: int = 0) -> dict[str, int]:
    """Synthetic bank: ``sizes[bin]`` items per bin with a count inside that bin."""
    value = {"0": 0, "1-5": 3, "6-15": 10, "16-25": 20}
    out, k = {}, start
    for name, n in sizes.items():
        for _ in range(n):
            out[f"{benchmark}/{k}"] = value[name]
            k += 1
    return out


def test_c48_rule_fixture_allocation_and_inclusion():
    bank = {**_bank("aime", {"0": 12, "1-5": 3, "6-15": 1, "16-25": 8}),
            **_bank("hmmt", {"0": 16, "1-5": 3, "6-15": 0, "16-25": 4}),
            **_bank("omni_math_hard", {"0": 9, "1-5": 7, "6-15": 6, "16-25": 1}),
            **_bank("gpqa_diamond", {"0": 0, "1-5": 2, "6-15": 6, "16-25": 16})}
    bank["hmmt/999"] = None                      # bank-unknown: never eligible
    order = sorted(bank)
    out = S.select_c48(bank, order)
    selected = {b: {n: out["strata"][b][n]["selected"] for n, _, _ in S.C48_BINS} for b in BENCHMARKS}
    # base min(size, 3); the rest by largest remainder of spare-proportional shares (hand-computed).
    assert selected["aime"] == {"0": 4, "1-5": 3, "6-15": 1, "16-25": 4}
    assert selected["hmmt"] == {"0": 6, "1-5": 3, "6-15": 0, "16-25": 3}
    assert selected["omni_math_hard"] == {"0": 4, "1-5": 4, "6-15": 3, "16-25": 1}
    assert selected["gpqa_diamond"] == {"0": 0, "1-5": 2, "6-15": 4, "16-25": 6}
    assert len(out["items"]) == 48 and "hmmt/999" not in out["items"]
    assert out["items"] == [i for i in order if i in set(out["items"])]          # listed in the given order
    for item, p in out["inclusion_prob"].items():
        cell = out["cells"][item]
        table = out["strata"][item.split("/")[0]][cell]
        assert p == table["selected"] / table["size"]
    # Within a cell the lowest sha256(prefix + item) keys are chosen.
    cell = sorted((i for i in bank if i.startswith("aime/") and bank[i] == 0),
                  key=lambda i: S.sha256_text(S.c48_prefix("aime") + i))
    assert set(cell[:4]) == {i for i in out["items"] if out["cells"][i] == "0" and i.startswith("aime/")}
    assert S.select_c48(bank, order) == out                                          # deterministic


def test_c48_small_cells_taken_whole_and_too_few_refuses():
    bank = _bank("aime", {"0": 2, "1-5": 1, "6-15": 20, "16-25": 0})
    out = S.select_c48(bank, sorted(bank))
    assert out["strata"]["aime"]["0"] == {"size": 2, "selected": 2}
    assert out["strata"]["aime"]["1-5"] == {"size": 1, "selected": 1}
    assert out["strata"]["aime"]["6-15"] == {"size": 20, "selected": 9}
    with pytest.raises(S.SplitsRefused):
        S.select_c48(_bank("hmmt", {"0": 5, "1-5": 5}), sorted(_bank("hmmt", {"0": 5, "1-5": 5})))


def test_c48_bins():
    assert [S.c48_bin(v) for v in (0, 1, 5, 6, 15, 16, 25)] == ["0", "1-5", "1-5", "6-15", "6-15", "16-25", "16-25"]
    with pytest.raises(ValueError):
        S.c48_bin(26)


# ---------------------------------------------------------------------------------------------------------------
# Duplicate registry on a fixture
# ---------------------------------------------------------------------------------------------------------------

_TEXTS = {
    "a": "Find the number of ordered pairs of positive integers (m, n) with m + n = 2024 and gcd(m, n) = 8 exactly.",
    "c": "A regular hexagon has side length 7. Compute the area of the region inside it but outside its incircle.",
    "e": "Let p be an odd prime. Determine the largest k such that p to the k divides the binomial of 2p and p.",
    "g": "Sphere packing in dimension three: what fraction of space is filled by the face-centred cubic lattice?",
}


def _omni(row_index: int, text: str, answer: str) -> object:
    row = {"row_index": row_index, "problem": text, "solution": "Synthetic.", "answer": answer,
           "difficulty": 7.5, "domain": ["Mathematics -> Algebra -> Other"], "source": "synthetic"}
    return omni_math_hard.build_item(row, source_revision=FIXTURE_REVISION)


def _registry_fixture():
    items = [
        _omni(10, _TEXTS["a"], "1"),                                   # used (legacy main)
        _omni(20, _TEXTS["a"] + " ", "1"),                             # exact duplicate of a used item
        _omni(5, _TEXTS["c"], "\\frac{1}{2}"),
        _omni(30, _TEXTS["c"].replace("Compute", "Find"), "0.5"),      # near dup, gold equal by Math-Verify
        _omni(40, _TEXTS["e"], "3"),
        _omni(41, _TEXTS["e"].replace("largest", "greatest"), "5"),     # near dup, conflicting gold
        _omni(50, _TEXTS["g"], "\\frac{\\pi}{3\\sqrt{2}}"),            # unrelated
    ]
    reviewed = [("omni_math_hard/30", "omni_math_hard/5"), ("omni_math_hard/40", "omni_math_hard/41")]
    return {"omni_math_hard": items}, reviewed


def test_registry_exclusions_fixture():
    inventories, reviewed = _registry_fixture()
    reg = S.build_registry(inventories, {"omni_math_hard/10"}, reviewed_pairs=reviewed)
    assert reg["excluded"] == {
        "omni_math_hard/20": "partner_of_used:omni_math_hard/10",
        "omni_math_hard/30": "duplicate_of:omni_math_hard/5",          # lowest source row kept (5 before 30)
        "omni_math_hard/40": "conflicting_gold",
        "omni_math_hard/41": "conflicting_gold",
    }
    decisions = {tuple(g["members"]): g["decision"] for g in reg["groups"]}
    assert decisions[("omni_math_hard/10", "omni_math_hard/20")] == "partner_of_used"
    assert decisions[("omni_math_hard/5", "omni_math_hard/30")] == "keep_lowest_row"
    assert decisions[("omni_math_hard/40", "omni_math_hard/41")] == "conflicting_gold"
    statuses = {(f["a"], f["b"]): f["status"] for f in reg["flagged"]}
    assert statuses[("omni_math_hard/10", "omni_math_hard/20")] == "exact_duplicate"
    assert all("omni_math_hard/50" not in pair for pair in statuses)
    payload = json.dumps(S.registry_payload(reg))
    assert "frac" not in payload and "0.5\"" not in payload                    # gold texts are never written


def test_registry_refuses_unflagged_or_ineligible_reviewed_pair():
    inventories, _ = _registry_fixture()
    with pytest.raises(S.SplitsRefused, match="not flagged"):
        S.build_registry(inventories, set(), reviewed_pairs=[("omni_math_hard/5", "omni_math_hard/50")])
    with pytest.raises(S.SplitsRefused, match="not eligible"):
        S.build_registry(inventories, set(), reviewed_pairs=[("omni_math_hard/5", "omni_math_hard/9999")])


def test_gold_relation_one_way_is_not_a_conflict():
    assert S.gold_relation("\\(\\{n \\ge 13\\}\\)", "n \\geq 13", benchmark="omni_math_hard") in ("equal", "one_way")
    assert S.gold_relation("3", "5", benchmark="omni_math_hard") == "conflict"
    assert S.gold_relation("A", "B", benchmark="gpqa_diamond") == "conflict"
    assert S.gold_relation(" 7 ", "7", benchmark="aime") == "equal"


def test_jaccard_and_normalization():
    assert S.normalize_text("  A\n\tB  ") == "a b"
    same = S.char_shingles(S.normalize_text(_TEXTS["a"]))
    assert S.jaccard(same, same) == 1.0
    assert S.jaccard(S.char_shingles("abcdefgh"), S.char_shingles("zyxwvuts")) == 0.0


# ---------------------------------------------------------------------------------------------------------------
# T200 assignment and salts
# ---------------------------------------------------------------------------------------------------------------

def test_t200_assignment_alternates_and_odd_extra_to_parasail():
    t200 = {"aime": [f"aime/x{i}" for i in range(5)], "hmmt": [f"hmmt/x{i}" for i in range(4)]}
    out = S.t200_assignment(t200)
    assert sum(v == "parasail" for k, v in out.items() if k.startswith("aime/")) == 3
    assert sum(v == "mancer" for k, v in out.items() if k.startswith("hmmt/")) == 2
    ranked = sorted(t200["hmmt"], key=lambda i: S.tm.rank_key(S.salt("hmmt", "t200-shard"), "rank", i))
    assert [out[i] for i in ranked] == ["mancer", "parasail", "mancer", "parasail"]
    assert S.t200_assignment({k: list(reversed(v)) for k, v in t200.items()}) == out    # order independent


def test_salts_are_explicit_never_the_default_study_id():
    record = json.dumps(S.salts_record())
    assert STUDY_ID not in record
    assert S.salt("aime", "t200-v1") == "openrouter_passk_cont_v1|aime|seed=20260925|t200-v1"
    assert S.c48_prefix("hmmt") == "openrouter_passk_cont_v1|C48|hmmt|seed=20260925|"


# ---------------------------------------------------------------------------------------------------------------
# Write-once
# ---------------------------------------------------------------------------------------------------------------

def test_write_once_refuses_different_content(tmp_path):
    target = tmp_path / "manifests" / "splits.json"
    digest = S.write_once(target, b"one\n")
    assert S.write_once(target, b"one\n") == digest                              # identical rerun: no-op
    with pytest.raises(S.SplitsRefused, match="write-once"):
        S.write_once(target, b"two\n")
    assert target.read_bytes() == b"one\n"
    assert not [p for p in target.parent.iterdir() if p.name != "splits.json"]  # no temp file left behind
    private = S.write_once(tmp_path / "private" / "x.jsonl", b"{}\n", mode=0o600)
    assert private and oct(os.stat(tmp_path / "private" / "x.jsonl").st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------------------------------------------------------
# Real legacy data (read-only) -- the frozen build
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def written(tmp_path_factory):
    if not _real_inputs_available():
        pytest.skip("legacy run root or pinned HF snapshots absent")
    cfg = load_config(CONFIG)
    run_root = tmp_path_factory.mktemp("passk_run")
    record = S.write_artifacts(cfg, run_root, S.LEGACY_RUN_ROOT, repo=ROOT.parent,
                               local_state=run_root / "local_state")
    return cfg, run_root, record


@needs_real
def test_gate_s_passes_on_real_data(written):
    _cfg, run_root, record = written
    failed = [c["name"] for c in record["checks"] if not c["passed"]]
    assert record["passed"] and not failed, failed
    names = {c["name"] for c in record["checks"]}
    assert {"write_once_sha256", "t200_disjoint_main_dev_duplicates", "t200_composition_20_20_100_60",
            "c48_subset_bank_known_with_inclusion_probs", "legacy_assignment_reproduced_l96_d32",
            "prereg_committed_e781c34", "manifest_rows_legacy_byte_identical"} <= names
    assert (run_root / "gates" / "gate_s.json").is_file()


@needs_real
def test_real_rebuild_is_deterministic_and_rerun_is_noop(written):
    cfg, run_root, record = written
    rebuilt = S.build_splits(cfg, S.Legacy(S.LEGACY_RUN_ROOT))
    assert S.json_bytes(rebuilt["splits"]) == (run_root / "artifacts" / "manifests" / "splits.json").read_bytes()
    again = S.write_artifacts(cfg, run_root, S.LEGACY_RUN_ROOT, repo=ROOT.parent, local_state=run_root / "ls")
    assert again["files"] == record["files"]
    # The frozen run root, once written, must equal the rebuild too.
    frozen = S.PASSK_RUN_ROOT / "artifacts" / "manifests" / "splits.json"
    if frozen.is_file():
        assert frozen.read_bytes() == S.json_bytes(rebuilt["splits"])


@needs_real
def test_real_exclusions_and_composition(written):
    _cfg, run_root, _record = written
    splits = json.loads((run_root / "artifacts" / "manifests" / "splits.json").read_text())
    t200 = set(splits["panels"]["T200"])
    assert {b: len(v) for b, v in splits["panels_by_benchmark"]["T200"].items()} == S.T200_QUOTA
    assert {b: len(v) for b, v in splits["panels_by_benchmark"]["X80"].items()} == S.X80_QUOTA
    assert set(splits["panels"]["X80"]) <= t200
    assert not t200 & set(splits["panels"]["L96"]) and not t200 & set(splits["panels"]["D32"])
    excluded = splits["duplicate_exclusions"]
    assert not t200 & set(excluded)
    assert excluded["omni_math_hard/3611"] == "partner_of_used:omni_math_hard/336"
    assert excluded["omni_math_hard/162"] == "partner_of_used:omni_math_hard/3962"
    assert excluded["omni_math_hard/308"] == excluded["omni_math_hard/3566"] == "conflicting_gold"
    assert sum(r.startswith("duplicate_of:") for r in excluded.values()) == 10
    assert splits["t200_strata"]["omni_math_hard"]["pool"] == 455
    assert {b: splits["t200_strata"][b]["pool"] for b in ("aime", "hmmt", "gpqa_diamond")} == \
        {"aime": 28, "hmmt": 31, "gpqa_diamond": 164}
    c48 = splits["panels"]["C48"]
    assert len(c48) == 48 and set(c48) <= set(splits["c48"]["bank_correct"])
    assert not set(c48) & S.UNKNOWN_BANK_ITEMS and len(splits["c48"]["bank_correct"]) == 94
    assert splits["panels"]["V48"] == c48
    reserve = json.loads((run_root / "artifacts" / "manifests" / "reserve_pool.json").read_text())
    for b in BENCHMARKS:
        assert not set(reserve["items"][b]) & t200
        assert len(reserve["items"][b]) + S.T200_QUOTA[b] == splits["t200_strata"][b]["pool"]


@needs_real
def test_real_manifests_mirror_legacy(written):
    _cfg, run_root, _record = written
    artifacts = run_root / "artifacts"
    legacy = S.Legacy(S.LEGACY_RUN_ROOT)
    for b in BENCHMARKS:
        new = (artifacts / "manifests" / "public" / f"{b}.jsonl").read_bytes().splitlines(keepends=True)
        old = legacy.public_path(b).read_bytes().splitlines(keepends=True)
        assert new[:32] == old[:32]                                               # dev + main byte-identical
        assert len(new) == 32 + S.T200_QUOTA[b]
    assert (artifacts / "manifests" / "waves.json").read_text() == "[]"
    assert (artifacts / "locks" / "datasets.json").read_bytes() == legacy.lock_path.read_bytes()
    panels = json.loads((artifacts / "manifests" / "panels.json").read_text())
    old_panels = json.loads(legacy.panels_path.read_text())
    assert set(panels) == set(old_panels)
    for b in BENCHMARKS:
        assert set(panels[b]) == set(old_panels[b])
        assert panels[b]["main_item_ids"] == old_panels[b]["main_item_ids"]
        assert panels[b]["dev_item_ids"] == old_panels[b]["dev_item_ids"]
        assert panels[b]["selection_seed"] == 20260917 and panels[b]["procedure"] == "passk_cont_v1"
    assert oct(os.stat(artifacts / "private").st_mode & 0o777) == "0o700"


@needs_real
def test_real_write_once_refuses_changed_artifact(written):
    cfg, run_root, _record = written
    target = run_root / "artifacts" / "manifests" / "reserve_pool.json"
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    try:
        with pytest.raises(S.SplitsRefused, match="write-once"):
            S.write_artifacts(cfg, run_root, S.LEGACY_RUN_ROOT, repo=ROOT.parent, local_state=run_root / "ls")
    finally:
        target.write_bytes(original)


@needs_real
def test_load_splits_checks_gate(written, tmp_path):
    _cfg, run_root, record = written
    splits = S.load_splits(run_root)
    assert S.sha256_bytes(S.json_bytes(splits)) == record["splits_sha256"]
    with pytest.raises(S.SplitsRefused, match="Gate S has not run"):
        (tmp_path / "artifacts" / "manifests").mkdir(parents=True)
        (tmp_path / "artifacts" / "manifests" / "splits.json").write_bytes(S.json_bytes(splits))
        S.load_splits(tmp_path)
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "gate_s.json").write_text(json.dumps({**record, "splits_sha256": "0" * 64}))
    with pytest.raises(S.SplitsRefused, match="changed after Gate S"):
        S.load_splits(tmp_path)
    assert S.load_splits(tmp_path, require_gate=False) == splits
