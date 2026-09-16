"""Tests for the ARD36 E6 capture-panel selector (``ard.neural.panel``).

Two layers, on purpose:

* **Synthetic** — a temporary run root holding a 384-row public item table with the real
  natural layout.  These tests pin the selection rule itself (balance, determinism,
  seed sensitivity, fail-closed behaviour, manifest hashing) and run anywhere.
* **Live** — marked ``ard36_live`` and skipped unless the real run root is present.  These
  assert the numbers the module's docstring quotes, so the docstring cannot drift away from
  the run silently.  They are read-only: nothing here writes into a run root, and the
  frozen v1 ``tables/`` directory is never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import items as ARD_ITEMS
from agents_scaling.ard.neural import panel as P

RUN_ROOT = Path(os.environ.get(
    "ARD36_RUN_ROOT", "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36"))
live = pytest.mark.skipif(not (RUN_ROOT / "data" / "public" / "items.parquet").exists(),
                          reason=f"ARD36 run root not mounted at {RUN_ROOT}")


# --------------------------------------------------------------------------- fixtures


def _row(item_id: str, family: str, split: str, rank: int) -> dict:
    """One public item row with every ``ArdTask`` field, so the firewall check passes."""
    return {
        "item_id": item_id, "family": family, "dataset_revision": "rev", "source_id": item_id,
        "split": split, "group_id": item_id, "prompt_hash": "h", "option_permutation": "[]",
        "gold_location": "protected/x#y", "grader_hash": "g", "subject": "Subject",
        "declared_difficulty_metadata": "3",
        "n_options": K.MMLU_REQUIRED_OPTIONS if family == K.Family.MMLUPRO.value else 0,
        "answer_format": "multipleChoice" if family == K.Family.MMLUPRO.value else "exactMatch",
        "stratum": f"{family}/{split}", "rank": rank, "task_text": "t", "options_json": "[]",
    }


@pytest.fixture()
def fake_run_root(tmp_path: Path) -> Path:
    """A run root with the real natural item layout: 48/24/120 per family, 384 total."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for family in (f.value for f in K.FAMILIES):
        for split in K.SPLITS:
            for rank in range(K.SPLIT_COUNTS_PER_FAMILY[split]):
                rows.append(_row(f"{family}-{split}-{rank:04d}", family, split, rank))
    assert len(rows) == K.NATURAL_TOTAL
    path = ARD_ITEMS.items_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).select(list(ARD_ITEMS.PUBLIC_COLUMNS))
    pq.write_table(table, path)
    return tmp_path


# --------------------------------------------------------------------------- the rule


def test_contract_quota_is_self_consistent() -> None:
    """The constants the selector reads must agree before anything is drawn."""
    assert sum(K.CAPTURE_SPLIT_ITEMS.values()) == K.CAPTURE_TOTAL_ITEMS
    assert set(K.CAPTURE_SPLIT_ITEMS) <= set(K.SPLITS)
    for split, want in K.CAPTURE_SPLIT_ITEMS.items():
        per_family = want // len(K.FAMILIES)
        assert per_family <= K.SPLIT_COUNTS_PER_FAMILY[split], (
            f"{split}: quota {want} cannot be met by {K.SPLIT_COUNTS_PER_FAMILY[split]} per family")


def test_panel_is_balanced_and_the_right_size(fake_run_root: Path) -> None:
    items = P.select_capture_items(fake_run_root)
    assert len(items) == K.CAPTURE_TOTAL_ITEMS == 144
    assert len(set(items)) == len(items)
    assert list(items) == sorted(items), "the panel is returned as a sorted set, not a dispatch order"
    counts = P.split_family_counts(fake_run_root, items)
    assert counts == {
        "test/math500": 32, "test/mmlu_pro": 32,
        "train/math500": 32, "train/mmlu_pro": 32,
        "validation/math500": 8, "validation/mmlu_pro": 8,
    }


def test_selection_is_deterministic_and_seed_sensitive(fake_run_root: Path) -> None:
    a = P.select_capture_items(fake_run_root, seed=P.DEFAULT_PANEL_SEED)
    b = P.select_capture_items(fake_run_root, seed=P.DEFAULT_PANEL_SEED)
    c = P.select_capture_items(fake_run_root, seed=P.DEFAULT_PANEL_SEED + 1)
    assert a == b
    assert a != c, "a different seed must draw a different panel"
    # Still balanced under any seed: the quota is structural, not a property of one draw.
    assert P.split_family_counts(fake_run_root, c) == P.split_family_counts(fake_run_root, a)


def test_panel_is_not_the_rank_prefix(fake_run_root: Path) -> None:
    """The docstring's reason for a separate seed: it must not reproduce the 32B panel rule."""
    items = set(P.select_capture_items(fake_run_root))
    tasks = ARD_ITEMS.load_ard_tasks(fake_run_root)
    rank_prefix = {
        task.item_id for task in tasks.values()
        if task.rank < K.CAPTURE_SPLIT_ITEMS[task.split] // len(K.FAMILIES)
    }
    assert items != rank_prefix
    overlap = len(items & rank_prefix) / len(items)
    assert overlap < 0.75, f"hash order collapsed onto rank order ({overlap:.2f} overlap)"


def test_hash_order_is_namespaced() -> None:
    """The sort key is domain-separated; a bare (seed, item) hash must not reproduce it."""
    digest = P._digest(7, "math500-abc")
    assert digest == hashlib.sha256(
        f"{K.SEED_NAMESPACE}|{K.CONTRACT_VERSION}|{P.PANEL_PURPOSE}|7|math500-abc".encode()
    ).hexdigest()
    assert digest != hashlib.sha256(b"7|math500-abc").hexdigest()


def test_family_quota_splits_and_states_its_remainder() -> None:
    assert P.family_quota(64) == {"math500": 32, "mmlu_pro": 32}
    assert P.family_quota(7, ("a", "b")) == {"a": 4, "b": 3}
    assert sum(P.family_quota(7, ("a", "b", "c")).values()) == 7
    with pytest.raises(P.PanelError):
        P.family_quota(-1)


def test_selection_fails_closed_on_a_short_stratum(tmp_path: Path) -> None:
    """A stratum that cannot fill its quota is an error, never a quietly smaller panel."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_row(f"math500-train-{i:04d}", "math500", "train", i) for i in range(4)]
    path = ARD_ITEMS.items_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows).select(list(ARD_ITEMS.PUBLIC_COLUMNS)), path)
    with pytest.raises(P.PanelError, match="needs"):
        P.select_capture_items(tmp_path)


# --------------------------------------------------------------------------- manifest


def test_manifest_hash_is_a_property_of_the_set(fake_run_root: Path) -> None:
    items = P.select_capture_items(fake_run_root)
    manifest = P.panel_manifest(items)
    shuffled = tuple(reversed(items))
    assert P.panel_manifest(shuffled)["sha256"] == manifest["sha256"]
    assert manifest["sha256"] == hashlib.sha256(
        "".join(f"{i}\n" for i in sorted(items)).encode("utf-8")).hexdigest()
    assert manifest["sha256"] != P.panel_sha256(items[:-1])


def test_manifest_carries_the_rule_and_the_nonclaim(fake_run_root: Path) -> None:
    manifest = P.panel_manifest(P.select_capture_items(fake_run_root), run_root=fake_run_root)
    assert manifest["schema"] == P.PANEL_SCHEMA
    assert manifest["study_id"] == K.STUDY_ID
    assert manifest["seed"] == P.DEFAULT_PANEL_SEED
    assert manifest["n_items"] == manifest["expected_n_items"] == K.CAPTURE_TOTAL_ITEMS
    assert manifest["protocols"] == [p.value for p in K.CAPTURE_PROTOCOLS]
    assert manifest["blocks"] == [8, 17, 26]
    assert sorted(manifest["prospective_anchors"]) == sorted(K.PROSPECTIVE_ANCHORS)
    assert "NOT powered" in manifest["nonclaim"]
    assert json.loads(json.dumps(manifest)) == manifest, "the manifest must be JSON-round-trippable"


def test_manifest_refuses_a_degenerate_list() -> None:
    with pytest.raises(P.PanelError):
        P.panel_manifest([])
    with pytest.raises(P.PanelError):
        P.panel_manifest(["a", "a"])


# --------------------------------------------------------------------------- budget


def test_capture_blocks_match_the_engine() -> None:
    """The local copy must equal ``study.neural.engine.blocks_for`` for the pinned model."""
    from agents_scaling.study.neural.engine import blocks_for

    assert P.capture_blocks() == blocks_for(P.NUM_LAYERS_8B, K.CAPTURE_RELATIVE_LAYERS) == (8, 17, 26)
    assert P.capture_blocks()[1] == 17, "the spec's primary depth for Qwen3-8B"


def test_estimated_bytes_is_exact_on_the_vector_term() -> None:
    est = P.estimated_bytes(7200, n_layers=3, n_anchors=2)
    assert est["n_vectors"] == 43_200
    assert est["vector_bytes"] == 43_200 * P.HIDDEN_SIZE_8B * P.STORED_ITEMSIZE == 353_894_400
    assert est["total_bytes_estimate"] > est["vector_bytes"]
    # Halving the anchors halves the vectors: the estimate is linear where it claims to be.
    assert P.estimated_bytes(7200, n_layers=3, n_anchors=4)["n_vectors"] == 2 * est["n_vectors"]
    with pytest.raises(P.PanelError):
        P.estimated_bytes(10, n_layers=0)


def test_missing_tables_directory_is_an_error(fake_run_root: Path) -> None:
    """A run root without ``tables_v2`` must fail rather than fall back to frozen v1."""
    with pytest.raises(P.PanelError, match="tables_v2"):
        P.capture_decision_ids(fake_run_root, ["x"])


# --------------------------------------------------------------------------- live run


@live
def test_live_panel_matches_the_docstring() -> None:
    items = P.select_capture_items(RUN_ROOT)
    assert len(items) == 144
    assert P.split_family_counts(RUN_ROOT, items) == {
        "test/math500": 32, "test/mmlu_pro": 32,
        "train/math500": 32, "train/mmlu_pro": 32,
        "validation/math500": 8, "validation/mmlu_pro": 8,
    }


@live
def test_live_decision_count_and_root_aliasing() -> None:
    items = P.select_capture_items(RUN_ROOT)
    decisions = P.capture_decision_ids(RUN_ROOT, items)
    # 5 shared roots + 3 protocols x 3 update rounds x 5 agents = 50 per item.
    assert len(decisions) == 7_200 == 50 * len(items)
    everything = P.capture_decision_ids(RUN_ROOT, sorted(ARD_ITEMS.load_ard_tasks(RUN_ROOT)))
    assert len(everything) == 19_200, "the whole capture universe, roots deduplicated"


@live
def test_live_token_budget_is_what_the_docstring_quotes() -> None:
    items = P.select_capture_items(RUN_ROOT)
    tokens = P.replayed_tokens(RUN_ROOT, P.capture_decision_ids(RUN_ROOT, items))
    assert tokens["n_records_missing"] == 0, tokens["missing_examples"]
    assert tokens["total_tokens"] == 9_570_118
    assert tokens["over_max_model_len"] == 0
    assert tokens["max_sequence_tokens"] <= K.MAX_MODEL_LEN


@live
def test_live_contested_count_is_reported_and_small() -> None:
    items = P.select_capture_items(RUN_ROOT)
    contested = P.contested_team_rounds(RUN_ROOT, items)
    assert contested["n_rows"] == 48
    assert contested["n_distinct"] == 38
    assert contested["n_items"] == 12
    # The whole point of the nonclaim: this is a small fraction of the panel.
    assert contested["n_items"] / len(items) < 0.10
    assert "MDE80" in contested["nonclaim"]


@live
def test_live_cli_emits_the_hashed_panel(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "panel.json"
    code = P.main(["--run-root", str(RUN_ROOT), "--no-tokens", "--out", str(out)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["sha256"] == P.panel_sha256(P.select_capture_items(RUN_ROOT))
    assert payload["n_capture_eligible_decisions"] == 7_200
    assert json.loads(out.read_text()) == payload


# --------------------------------------------------------------------------- hook


def test_select_capture_panel_matches_the_run_root_entry_point(fake_run_root: Path) -> None:
    """Both entry points are one rule: a manifest builder holding tasks gets the same 144."""
    tasks = list(ARD_ITEMS.load_ard_tasks(fake_run_root).values())
    from_tasks = P.select_capture_panel(tasks)
    assert [t.item_id for t in from_tasks] == list(P.select_capture_items(fake_run_root))
    assert all(isinstance(t, ARD_ITEMS.ArdTask) for t in from_tasks)
    with pytest.raises(P.PanelError, match="repeats"):
        P.select_capture_panel(tasks + tasks[:1])


def test_cells_builder_finds_the_selector() -> None:
    """``cells.capture_panel_items`` looks the selector up by name; keep that name working."""
    from agents_scaling.ard import cells

    selector = getattr(cells, "_panel_selector", None)
    if selector is None:  # the builder-side fix has not landed in this tree
        pytest.skip("cells._panel_selector is not present in this working tree")
    assert selector() is P.select_capture_panel
