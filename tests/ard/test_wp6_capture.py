"""WP6 capture: the decision-manifest replay runner (CPU, tiny random Qwen3).

Every forward pass here is the two-layer random model of ``tests/study_neural/support``
over the *real* Qwen3 vocabulary and tokenizer — never a pinned checkpoint — so the whole
file runs in seconds on a login node while still exercising the real engine, the real
storage layer and the real anchor resolver.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as A
from agents_scaling.ard.neural import capture as C
from agents_scaling.study.neural import storage as S
from tests.ard import wp6_support as W

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def tok():
    try:
        return W.support.qwen_tokenizer(K.CHECKPOINT_SMALL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-8B tokenizer unavailable offline: {exc!r}")


@pytest.fixture(scope="module")
def engine(tok):
    return W.tiny_engine(tok, blocks=(0, 1))


def did(i: int) -> str:
    return "%064x" % i


# --------------------------------------------------------------------------- work list


def test_work_list_puts_train_and_validation_before_test():
    rows = [
        W.work_row(did(3), item_id="i3", split="test"),
        W.work_row(did(1), item_id="i1", split="train"),
        W.work_row(did(4), item_id="i4", split=""),              # unknown split sorts last
        W.work_row(did(2), item_id="i2", split="validation"),
        W.work_row(did(5), item_id="i5", split="val"),           # the alias of "validation"
    ]
    items = C.work_list(rows)
    assert [w.split for w in items] == ["train", "validation", "val", "test", ""]
    assert C.SPLIT_ORDER["train"] < C.SPLIT_ORDER["validation"] == C.SPLIT_ORDER["val"] < C.SPLIT_ORDER["test"]
    # ordering inside a split is deterministic: item, protocol, round, agent
    inside = C.work_list([
        W.work_row(did(20), item_id="b", round_=1, agent_id=2, protocol="PEER"),
        W.work_row(did(21), item_id="a", round_=2, agent_id=0, protocol="PEER"),
        W.work_row(did(22), item_id="a", round_=1, agent_id=4, protocol="PEER"),
        W.work_row(did(23), item_id="a", round_=1, agent_id=1, protocol="BALANCED"),
    ])
    assert [(w.item_id, w.protocol, w.round_, w.agent_id) for w in inside] == [
        ("a", "BALANCED", 1, 1), ("a", "PEER", 1, 4), ("a", "PEER", 2, 0), ("b", "PEER", 1, 2)]


def test_work_list_collapses_a_root_decision_reused_across_protocols():
    """Spec §14.2 "Reuse root captures across protocols": one replay, not three."""
    rows = [W.work_row(did(7), round_=0, protocol=p.value) for p in K.CAPTURE_PROTOCOLS]
    items = C.work_list(rows)
    assert len(items) == 1
    assert items[0].decision_id == did(7)


def test_work_item_refuses_an_impossible_slot_or_round():
    with pytest.raises(ValueError):
        C.CaptureWorkItem(decision_id="", episode_id="e", item_id="i", agent_id=0, round_=0, protocol="PEER")
    with pytest.raises(ValueError):
        C.CaptureWorkItem(decision_id="d", episode_id="e", item_id="i", agent_id=K.N_AGENTS, round_=0, protocol="PEER")
    with pytest.raises(ValueError):
        C.CaptureWorkItem(decision_id="d", episode_id="e", item_id="i", agent_id=0, round_=4, protocol="PEER")


def test_shard_of_is_disjoint_complete_and_order_preserving():
    items = C.work_list([W.work_row(did(i), item_id=f"i{i:02d}") for i in range(11)])
    shards = [C.shard_of(items, s, 3) for s in range(3)]
    assert sorted(w.decision_id for shard in shards for w in shard) == sorted(w.decision_id for w in items)
    assert len({w.decision_id for shard in shards for w in shard}) == len(items)
    for shard in shards:
        assert [w.decision_id for w in shard] == [w.decision_id for w in items if w in shard]
    with pytest.raises(ValueError):
        C.shard_of(items, 3, 3)


def test_load_manifest_reads_jsonl_json_and_orders_it(tmp_path):
    rows = [W.work_row(did(1), item_id="i1", split="test"), W.work_row(did(2), item_id="i2", split="train")]
    jsonl = tmp_path / "m.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    plain = tmp_path / "m.json"
    plain.write_text(json.dumps({"decisions": rows}), encoding="utf-8")
    for path in (jsonl, plain):
        items = C.load_manifest(path)
        assert [w.split for w in items] == ["train", "test"]
        assert items[0].join["decision_id"] == did(2)


def test_manifest_from_item_file_keeps_capture_protocols_and_every_root(tmp_path):
    payload = {
        "source_id": "math500:0007", "split": "train", "family": "math500",
        "cell": {"cell_id": "cap.008"},
        "episodes": [
            {"protocol": "PEER", "episode_id": "e1", "cohort": "capture", "decisions": [
                {"decision_id": did(1), "agent_id": 0, "round": 0},
                {"decision_id": did(2), "agent_id": 0, "round": 1, "boundary_injected": True},
                {"agent_id": 1, "round": 1},                      # context failure: no id
            ]},
            {"protocol": "HUB5", "episode_id": "e2", "cohort": "capture", "decisions": [
                {"decision_id": did(3), "agent_id": 0, "round": 0},   # a root: kept
                {"decision_id": did(4), "agent_id": 0, "round": 2},   # not a CAPTURE protocol
            ]},
        ],
    }
    path = tmp_path / "math500:0007.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    items = C.manifest_from_item_file(path)
    assert sorted(w.decision_id for w in items) == sorted([did(1), did(2), did(3)])
    by_id = {w.decision_id: w for w in items}
    assert by_id[did(2)].boundary_injected is True
    assert by_id[did(1)].split == "train" and by_id[did(1)].cell_id == "cap.008"
    assert all(w.item_id == "math500:0007" for w in items)


def test_parity_panel_is_stratified_over_root_and_rounds_and_protocols():
    rows = []
    n = 0
    for round_ in K.ROUNDS:
        for protocol in K.CAPTURE_PROTOCOLS:
            for rep in range(3):
                n += 1
                rows.append(W.work_row(did(n), item_id=f"i{rep}", round_=round_,
                                       protocol=protocol.value, agent_id=rep))
    panel = C.parity_panel_items(C.work_list(rows), per_cell=2)
    cells = {(int(w.round_), w.protocol) for w in panel}
    assert {r for r, _ in cells} == set(K.ROUNDS)
    for round_ in (1, 2, 3):
        assert {p for r, p in cells if r == round_} == {p.value for p in K.CAPTURE_PROTOCOLS}
    assert sum(1 for w in panel if w.round_ == 0) == 2        # the shared neutral root cell
    assert len(panel) == 2 + 3 * len(K.CAPTURE_PROTOCOLS) * 2


# --------------------------------------------------------------------------- resume


def test_pending_items_skips_only_fully_committed_decisions():
    items = C.work_list([W.work_row(did(i), item_id=f"i{i}") for i in (1, 2, 3)])
    blocks = (8, 17)
    done = {(did(1), b, k) for b in blocks for k in K.CAPTURE_ANCHORS}
    done |= {(did(2), 8, k) for k in K.CAPTURE_ANCHORS}         # only one of two blocks
    done |= {(did(3), b, K.CAPTURE_ANCHORS[0]) for b in blocks}  # only one of four anchors
    pending = C.pending_items(items, done, blocks)
    assert [w.decision_id for w in pending] == [did(2), did(3)]
    assert C.pending_items(items, set(), blocks) == list(items)


# --------------------------------------------------------------------------- entropy


def test_next_token_entropy_is_the_full_vocabulary_entropy():
    uniform = np.zeros(4096)
    out = C.next_token_entropy(uniform)
    assert out["vocab_size"] == 4096
    assert out["entropy_nats"] == pytest.approx(float(np.log(4096)))
    assert out["entropy_bits"] == pytest.approx(12.0)
    assert out["top1_prob"] == pytest.approx(1.0 / 4096)
    assert out["logit_margin"] == pytest.approx(0.0)

    peaked = np.full(4096, -50.0)
    peaked[7] = 50.0
    sharp = C.next_token_entropy(peaked)
    assert sharp["entropy_nats"] == pytest.approx(0.0, abs=1e-9)
    assert sharp["top1_prob"] == pytest.approx(1.0)
    assert sharp["logit_margin"] == pytest.approx(100.0)

    # a scalar or a non-finite row is a refusal, never a silently-zero entropy
    with pytest.raises(ValueError):
        C.next_token_entropy(np.asarray([1.0]))
    with pytest.raises(ValueError):
        C.next_token_entropy(np.asarray([1.0, np.nan, 3.0]))


# --------------------------------------------------------------------------- staging


def test_staging_root_prefers_the_node_local_scratch(tmp_path, monkeypatch):
    local = tmp_path / "node_local"
    local.mkdir()
    monkeypatch.setenv("ARD36_LOCAL_SCRATCH", str(local))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "unused"))
    root = C.staging_root()
    assert root.parent == local and root.is_dir()

    monkeypatch.delenv("ARD36_LOCAL_SCRATCH")
    fallback = tmp_path / "tmpdir"
    fallback.mkdir()
    monkeypatch.setenv("TMPDIR", str(fallback))
    assert C.staging_root().parent == fallback


def test_consolidate_moves_complete_chunks_and_renumbers_them(tmp_path):
    run_root = tmp_path / "run"
    for k in range(2):
        staging = tmp_path / f"stage{k}"
        with S.ShardWriter(staging, C.CAPTURE_STAGE, "ard36.s000") as writer:
            writer.add(S.ActivationRow(
                StateSnapshot_id=did(k), consumer_revision="rev", condition="c", child_slot=None,
                nonce_hash="n", block=8, anchor_kind=A.ANCHOR_PROMPT_END, structural_span="s",
                token_offset=3, channel=A.CHANNEL_PROMPT, generated_token_count=0, missingness=None,
                tensor_hash=None, measurement_cost={}, operationally_available_at_checkpoint=True,
                stage=C.CAPTURE_STAGE, sequence_id=did(k)), np.ones(4, dtype=np.float16))
        moved = C.consolidate_shards(staging, run_root)
        assert len(moved) == 1
        assert not S.list_chunks(S.stage_dir(staging, C.CAPTURE_STAGE))     # the source is drained

    chunks = S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE))
    assert [index for _shard, index, _npz, _jsonl in chunks] == [0, 1]      # renumbered, no collision
    for _shard, _index, npz, jsonl in chunks:
        S.read_chunk(npz, jsonl)                                            # the two files agree
    assert S.existing_keys(run_root, C.CAPTURE_STAGE) == {
        (did(0), 8, A.ANCHOR_PROMPT_END), (did(1), 8, A.ANCHOR_PROMPT_END)}


def test_an_orphan_npz_from_a_crashed_consolidation_is_skipped_not_read(tmp_path):
    """The ``.npz`` lands before its ``.jsonl`` commit marker, so a crash in between
    leaves an orphan ``.npz`` — which must never be counted as a committed chunk."""
    directory = S.stage_dir(tmp_path, C.CAPTURE_STAGE)
    directory.mkdir(parents=True)
    npz, _jsonl = S.chunk_paths(directory, "ard36.s000", 0)
    np.savez(npz, vectors=np.zeros((1, 4), dtype=np.float16), keys=np.asarray(["k"], dtype=str))
    assert S.list_chunks(directory) == []
    assert S.existing_keys(tmp_path, C.CAPTURE_STAGE) == set()
    assert S.next_chunk_index(directory, "ard36.s000") == 1      # never overwritten


# --------------------------------------------------------------------------- the runner


def build_run(tok, tmp_path, *, bodies):
    store = W.StubStore()
    rows = []
    for i, (body, meta) in enumerate(bodies):
        rid = did(100 + i)
        store.add(W.request_record(tok, rid, task=f"Task number {i}?", content=body))
        rows.append(W.work_row(rid, **meta))
    return store, C.work_list(rows)


GOOD = W.report_text("42")
NO_CONF = W.report_text("42", confidence=None)
NO_TAGS = "\n\nI could not produce a report."


def test_run_capture_writes_four_anchors_per_block_with_the_join_keys(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", episode_id="e1", agent_id=0, round_=1, protocol="PEER")),
        (NO_CONF, dict(item_id="i1", split="train", episode_id="e1", agent_id=1, round_=1,
                       protocol="PEER", boundary_injected=True)),
        (NO_TAGS, dict(item_id="i2", split="test", episode_id="e2", agent_id=2, round_=2, protocol="BALANCED")),
    ])
    recording = W.RecordingEngine(engine)
    run_root = tmp_path / "run"
    config = C.CaptureConfig(revision=W.REVISION_8B, shard_name="ard36.s000", group_size=8)
    stats = C.run_capture(items, engine=recording, tokenizer=tok, store=store,
                          run_root=run_root, config=config, staging=tmp_path / "stage")

    assert stats["work_items"] == stats["pending"] == 3
    assert stats["captured"] == 3
    assert stats["rows"] == 3 * len(K.CAPTURE_ANCHORS) * len(engine.blocks) == 24
    assert stats["anchor_errors"] == 0 and stats["missing_records"] == 0

    rows = W.chunk_rows(run_root)
    assert len(rows) == 24
    by_decision: dict[str, list[dict]] = {}
    for row in rows:
        by_decision.setdefault(row["StateSnapshot_id"], []).append(row)
    for decision, group in by_decision.items():
        for block in engine.blocks:
            kinds = [r["anchor_kind"] for r in group if r["block"] == block]
            assert kinds == list(K.CAPTURE_ANCHORS), (decision, block)

    # join keys (plan §5.6) travel in ActivationRow.extra, so a vector can be joined back
    by_id = {w.decision_id: w for w in items}
    for row in rows:
        extra = row["extra"]
        assert set(C.JOIN_KEYS) <= set(extra)
        assert extra == {**extra, **by_id[row["StateSnapshot_id"]].join}
        assert extra["item_id"] and extra["split"] and extra["family"]
        assert extra["feature_time"] == A.feature_time_of(row["anchor_kind"])
        assert extra["last_allowed_round"] == A.last_allowed_round(row["anchor_kind"], extra["round"])
    assert {row["extra"]["boundary_injected"] for row in rows} == {True, False}
    assert {row["extra"]["protocol"] for row in rows} == {"PEER", "BALANCED"}


def test_missing_anchors_are_ledger_rows_without_vectors(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
        (NO_TAGS, dict(item_id="i2", split="train", agent_id=1, round_=1)),
    ])
    run_root = tmp_path / "run"
    stats = C.run_capture(items, engine=engine, tokenizer=tok, store=store, run_root=run_root,
                          config=C.CaptureConfig(revision=W.REVISION_8B), staging=tmp_path / "stage")

    assert stats["anchors_missing"] == {A.MISSING_FINAL_TAG: 3}
    matrix, _frame = S.load_stage(run_root, C.CAPTURE_STAGE)
    rows = W.chunk_rows(run_root)
    present = [r for r in rows if r["vector_index"] >= 0]
    missing = [r for r in rows if r["vector_index"] < 0]
    assert len(present) == (4 + 1) * len(engine.blocks) == 10
    assert len(missing) == 3 * len(engine.blocks) == 6
    assert matrix.shape == (10, engine.hidden_size)
    assert not np.any(np.all(np.asarray(matrix, dtype=np.float32) == 0.0, axis=1))   # no zero rows
    for row in missing:
        assert row["missingness"] == A.MISSING_FINAL_TAG
        assert row["token_offset"] is None and row["tensor_hash"] is None


def test_entropy_is_computed_at_the_two_contract_anchors_and_nowhere_else(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
    ])
    recording = W.RecordingEngine(engine)
    run_root = tmp_path / "run"
    C.run_capture(items, engine=recording, tokenizer=tok, store=store, run_root=run_root,
                  config=C.CaptureConfig(revision=W.REVISION_8B), staging=tmp_path / "stage")

    aset = A.resolve_decision_anchors(store.get(items[0].request_id), tok)
    wanted = {aset.by_kind(k).token_offset for k in K.ENTROPY_ANCHORS}
    assert len(recording.requests) == 1
    request = recording.requests[0]
    # the lm_head is applied at the two designated anchors only, never at every token
    assert set(request.logit_positions) == wanted
    assert len(request.logit_positions) == len(K.ENTROPY_ANCHORS) == 2
    assert set(request.logit_positions) <= set(request.positions)
    assert len(request.logit_positions) < len(request.token_ids)

    rows = W.chunk_rows(run_root)
    entropy_rows = [r for r in rows if r["extra"]["entropy_anchor"]]
    assert {r["anchor_kind"] for r in entropy_rows} == set(K.ENTROPY_ANCHORS)
    assert len(entropy_rows) == len(K.ENTROPY_ANCHORS) * len(engine.blocks)
    for row in entropy_rows:
        extra = row["extra"]
        assert extra["vocab_size"] == int(engine.model.config.vocab_size)     # FULL vocabulary
        assert 0.0 <= extra["entropy_nats"] <= float(np.log(extra["vocab_size"])) + 1e-9
        assert extra["entropy_bits"] == pytest.approx(extra["entropy_nats"] / np.log(2.0))
        assert 0.0 < extra["top1_prob"] <= 1.0
    for row in rows:
        if row["anchor_kind"] in K.ENTROPY_ANCHORS:
            continue
        assert row["extra"]["entropy_anchor"] is False
        assert "entropy_nats" not in row["extra"] and "top1_prob" not in row["extra"]


def test_vocabulary_logits_are_never_persisted(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
    ])
    run_root = tmp_path / "run"
    C.run_capture(items, engine=engine, tokenizer=tok, store=store, run_root=run_root,
                  config=C.CaptureConfig(revision=W.REVISION_8B), staging=tmp_path / "stage")

    vocab = int(engine.model.config.vocab_size)
    for _shard, _index, npz, jsonl in S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE)):
        with np.load(npz, allow_pickle=False) as data:
            assert sorted(data.files) == ["keys", "schema_version", "vectors"]
            assert data["vectors"].shape[1] == engine.hidden_size < vocab
        payload = jsonl.read_text(encoding="utf-8")
        assert len(payload) < 200_000                    # a single vocabulary row would dwarf this
        for line in payload.splitlines():
            row = json.loads(line)

            def walk(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        # the only surviving trace of the vocabulary row is a named scalar
                        assert "logprob" not in k, k
                        assert "logit" not in k or k == "logit_margin", k
                        walk(v)
                elif isinstance(value, list):
                    assert len(value) <= 16, f"a {len(value)}-element list looks like a distribution"
                    for v in value:
                        walk(v)

            walk(row)
            assert isinstance(row["extra"].get("logit_margin", 0.0), float)


def test_shards_stage_on_node_local_scratch_and_land_only_at_consolidation(tok, engine, tmp_path, monkeypatch):
    local = tmp_path / "node_local"
    local.mkdir()
    monkeypatch.setenv("ARD36_LOCAL_SCRATCH", str(local))
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
        (NO_CONF, dict(item_id="i1", split="train", agent_id=1, round_=1)),
    ])
    run_root = tmp_path / "run"
    seen: list[tuple[int, int]] = []

    def on_event(name, payload):
        # mid-run: chunks exist on the node-local disk and *not* in the shared run root
        staged = list(local.rglob("*.jsonl"))
        seen.append((len(staged), len(list((run_root / "neural").rglob("*.jsonl")))))

    stats = C.run_capture(items, engine=engine, tokenizer=tok, store=store, run_root=run_root,
                          config=C.CaptureConfig(revision=W.REVISION_8B, group_size=1),
                          on_event=on_event)

    assert stats["staging"].startswith(str(local))
    assert seen and all(staged >= 1 and landed == 0 for staged, landed in seen)
    assert len(S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE))) >= 1
    assert list(local.iterdir()) == []          # the node-local staging directory is cleaned up
    assert len(W.chunk_rows(run_root)) == 2 * len(K.CAPTURE_ANCHORS) * len(engine.blocks)


def test_a_second_pass_captures_nothing_and_writes_no_new_chunk(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
        (NO_TAGS, dict(item_id="i1", split="train", agent_id=1, round_=1)),
    ])
    run_root = tmp_path / "run"
    config = C.CaptureConfig(revision=W.REVISION_8B)
    first = C.run_capture(items, engine=engine, tokenizer=tok, store=store, run_root=run_root,
                          config=config, staging=tmp_path / "stage1")
    before = {p.name for _s, _i, _n, p in S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE))}

    recording = W.RecordingEngine(engine)
    second = C.run_capture(items, engine=recording, tokenizer=tok, store=store, run_root=run_root,
                           config=config, staging=tmp_path / "stage2")
    after = {p.name for _s, _i, _n, p in S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE))}

    assert first["pending"] == 2 and second["pending"] == 0
    assert second["captured"] == 0 and second["rows"] == 0
    assert recording.requests == []             # nothing was replayed a second time
    assert after == before
    assert len(W.chunk_rows(run_root)) == first["rows"]


def test_a_record_from_another_revision_is_refused(tok, engine, tmp_path):
    store, items = build_run(tok, tmp_path, bodies=[(GOOD, dict(item_id="i1", split="train"))])
    store.add(W.request_record(tok, items[0].request_id, content=GOOD,
                               revision=K.REVISIONS[K.CHECKPOINT_LARGE]))
    with pytest.raises(RuntimeError, match="model_revision"):
        C.run_capture(items, engine=engine, tokenizer=tok, store=store, run_root=tmp_path / "run",
                      config=C.CaptureConfig(revision=W.REVISION_8B), staging=tmp_path / "stage")


def test_a_work_item_without_a_record_is_refused_by_default(tok, engine, tmp_path):
    items = C.work_list([W.work_row(did(900), item_id="i1", split="train")])
    with pytest.raises(RuntimeError, match="not in the store"):
        C.run_capture(items, engine=engine, tokenizer=tok, store=W.StubStore(),
                      run_root=tmp_path / "run", config=C.CaptureConfig(revision=W.REVISION_8B),
                      staging=tmp_path / "stage")


# --------------------------------------------------------------------------- sanity checks


def test_root_prefill_identity_reports_the_realized_numbers():
    base = np.linspace(-1.0, 1.0, 32)
    identical = C.root_prefill_identity({s: base.copy() for s in range(K.N_AGENTS)},
                                        prompt_hashes={s: "h" for s in range(K.N_AGENTS)})
    assert identical["n_agents"] == K.N_AGENTS
    assert identical["max_abs_diff"] == 0.0
    assert identical["min_pairwise_cosine"] == pytest.approx(1.0)
    assert identical["prompt_bytes_identical"] is True
    assert identical["identical"] is True
    assert "sanity check" in identical["nonclaim"]

    drifted = {s: base.copy() for s in range(K.N_AGENTS)}
    drifted[3] = base + 0.5
    verdict = C.root_prefill_identity(drifted)
    assert verdict["max_abs_diff"] == pytest.approx(0.5)
    assert verdict["identical"] is False
    assert verdict["prompt_bytes_identical"] is None

    # a byte-different prompt is reported even if the vectors happen to agree
    assert C.root_prefill_identity({s: base.copy() for s in range(K.N_AGENTS)},
                                   prompt_hashes={0: "a", 1: "b", 2: "b", 3: "b", 4: "b"})["identical"] is False
    with pytest.raises(ValueError):
        C.root_prefill_identity({0: base})


def test_primary_block_and_the_degraded_anchor_set():
    assert C.primary_block(36) == 17                    # floor(L/2)-1 for Qwen3-8B
    assert C.DEGRADED_ANCHORS == (A.ANCHOR_PROMPT_END, A.ANCHOR_ANSWER_END)
    # the preregistered degradation is a complete cohort at reduced depth: both entropy
    # anchors survive it, so the fallback still carries the §14.5 entropy panel
    assert set(K.ENTROPY_ANCHORS) <= set(C.DEGRADED_ANCHORS)


def test_the_preregistered_degradation_is_a_complete_cohort_at_reduced_depth(tok, tmp_path):
    """Plan §5.6: if the hour-30 deadline binds, capture the primary layer and anchors 1
    and 3 only — every decision still gets a complete (reduced) row set, never 72% of the
    cohort, and both entropy anchors survive."""
    engine = W.tiny_engine(tok, blocks=(1,))
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
        (NO_CONF, dict(item_id="i1", split="train", agent_id=1, round_=1)),
    ])
    recording = W.RecordingEngine(engine)
    run_root = tmp_path / "run"
    config = C.CaptureConfig(revision=W.REVISION_8B, anchors=C.DEGRADED_ANCHORS,
                             entropy_anchors=frozenset(a for a in K.ENTROPY_ANCHORS
                                                       if a in C.DEGRADED_ANCHORS))
    stats = C.run_capture(items, engine=recording, tokenizer=tok, store=store,
                          run_root=run_root, config=config, staging=tmp_path / "stage")

    assert stats["captured"] == 2                       # every decision, none dropped
    assert stats["rows"] == 2 * len(C.DEGRADED_ANCHORS) * 1
    rows = W.chunk_rows(run_root)
    assert {r["anchor_kind"] for r in rows} == set(C.DEGRADED_ANCHORS)
    assert {r["block"] for r in rows} == {1}
    assert all(r["extra"]["entropy_anchor"] for r in rows)


def test_a_narrowed_entropy_anchor_set_is_honoured_on_both_sides(tok, engine, tmp_path):
    """The logits requested from the engine and the entropies written must come from one
    decision: narrowing ``entropy_anchors`` must narrow the lm_head work too."""
    store, items = build_run(tok, tmp_path, bodies=[
        (GOOD, dict(item_id="i1", split="train", agent_id=0, round_=1)),
    ])
    recording = W.RecordingEngine(engine)
    run_root = tmp_path / "run"
    C.run_capture(items, engine=recording, tokenizer=tok, store=store, run_root=run_root,
                  config=C.CaptureConfig(revision=W.REVISION_8B,
                                         entropy_anchors=frozenset({A.ANCHOR_PROMPT_END})),
                  staging=tmp_path / "stage")

    aset = A.resolve_decision_anchors(store.get(items[0].request_id), tok)
    assert recording.requests[0].logit_positions == (aset.by_kind(A.ANCHOR_PROMPT_END).token_offset,)
    rows = W.chunk_rows(run_root)
    assert {r["anchor_kind"] for r in rows if r["extra"]["entropy_anchor"]} == {A.ANCHOR_PROMPT_END}
    assert all("entropy_nats" not in r["extra"] for r in rows
               if r["anchor_kind"] != A.ANCHOR_PROMPT_END)


# --------------------------------------------------------------------------- the cell worker

# Regression cover for the two wiring defects that would have made a GPU run silently
# useless: (a) ``cells.build_e6_capture`` emitted no ``cells``/``manifest``/``snapshot_path``,
# so every item resolved an empty work list, and (b) ``run_capture_cell`` wrote item status
# "no_decisions" and exited 0 on that empty list — a cell that captured nothing looked
# complete.  The pair is only visible in the worker, so these tests drive the worker.

CAPTURE_CELL = "EPISODE.natural.PEER.8B.T1024.train.s000"


def capture_run(tok, tmp_path, *, item_id="i1", n=2, cells=(CAPTURE_CELL,)):
    """A run root with one generation cell, ``n`` replayable decisions and their records."""
    run_root = tmp_path / "run"
    from agents_scaling.study.inference.store import RequestStore

    store = RequestStore(run_root / "requests")
    decisions = []
    for i in range(n):
        rid = did(200 + i)
        store.publish(W.request_record(tok, rid, task=f"Task {i}?", content=GOOD))
        decisions.append((rid, "PEER", 1, i))
    W.item_result_file(run_root, cells[0], item_id, decisions=decisions)
    options = {"cells": list(cells), "snapshot_path": "unused (ENGINE_FACTORY is stubbed)",
               "protocols": [p.value for p in K.CAPTURE_PROTOCOLS],
               "anchors": list(K.CAPTURE_ANCHORS), "staging": str(tmp_path / "stage"),
               "revision": W.REVISION_8B}
    return run_root, options


@pytest.fixture
def stub_factory(monkeypatch, tok, engine):
    """Replace the HF snapshot loader: these tests exercise wiring, not the model."""
    monkeypatch.setattr(C, "ENGINE_FACTORY", lambda options: (engine, tok))


def test_a_cell_whose_options_name_no_decision_source_is_refused(tok, tmp_path, stub_factory):
    """Defect (a): the builder emitted only relative_layers/anchors/sha, so the worker had
    nothing to replay — and said so by exiting 0."""
    run_root, options = capture_run(tok, tmp_path)
    for key in ("cells", "manifest"):
        options.pop(key, None)
    runner = W.FakeRunner(run_root, options, items=("i1",))
    with pytest.raises(C.CaptureCellError, match="neither 'cells' nor 'manifest'"):
        C.run_capture_cell(runner, "i1")
    assert runner.written == []                     # nothing was marked complete


def test_a_cell_pointed_at_generation_cells_that_do_not_exist_is_refused(tok, tmp_path, stub_factory):
    run_root, options = capture_run(tok, tmp_path)
    options["cells"] = ["EPISODE.natural.PEER.8B.T1024.train.s404"]
    runner = W.FakeRunner(run_root, options, items=("i1",))
    with pytest.raises(C.CaptureCellError, match="items/ directory"):
        C.run_capture_cell(runner, "i1")
    assert runner.written == []


def test_an_item_with_no_eligible_decision_is_a_clean_no_decisions(tok, tmp_path, stub_factory):
    """The legitimate empty: the generation cells were read, this item has nothing in them
    (every episode was a context failure).  That is not a defect and must not raise."""
    run_root, options = capture_run(tok, tmp_path)
    W.item_result_file(run_root, CAPTURE_CELL, "i2", decisions=())
    runner = W.FakeRunner(run_root, options, items=("i1", "i2"))
    C.run_capture_cell(runner, "i2")
    assert runner.written == [("i2", "no_decisions")]


def test_a_cell_that_captures_nothing_fails_instead_of_looking_complete(tok, tmp_path, stub_factory):
    """Defect (b): a non-empty work list whose every decision was skipped (here: no record
    in the store, with strict_records off) used to write a complete item."""
    run_root, options = capture_run(tok, tmp_path, n=0)
    W.item_result_file(run_root, CAPTURE_CELL, "i1",
                       decisions=[(did(900), "PEER", 1, 0), (did(901), "PEER", 1, 1)])
    options["strict_records"] = False
    runner = W.FakeRunner(run_root, options, items=("i1",))
    with pytest.raises(C.CaptureCellError, match="captured 0 of 2"):
        C.run_capture_cell(runner, "i1")
    assert runner.written == []


def test_a_cell_that_captures_its_decisions_completes_and_reports_the_count(tok, tmp_path, stub_factory):
    run_root, options = capture_run(tok, tmp_path, n=2)
    runner = W.FakeRunner(run_root, options, items=("i1",))
    C.run_capture_cell(runner, "i1")
    assert runner.written == [("i1", "complete")]
    captured = dict(runner.events)["captured"]
    assert captured["captured"] == 2
    assert captured["rows"] == 2 * len(K.CAPTURE_ANCHORS) * len(W.tiny_engine(tok).blocks)

    # ...and a second run over the same committed keys is a no-op resume, not a failure:
    # zero captures is only legitimate when nothing was pending.
    again = W.FakeRunner(run_root, options, items=("i1",))
    C.run_capture_cell(again, "i1")
    assert again.written == [("i1", "complete")]
    assert dict(again.events)["captured"]["captured"] == 0


def test_assert_capture_progress_separates_a_resume_from_a_silent_zero():
    ok_resume = {"work_items": 12, "pending": 0, "captured": 0}
    assert C.assert_capture_progress(ok_resume) is ok_resume
    with pytest.raises(C.CaptureCellError, match="captured 0 of 12"):
        C.assert_capture_progress({"work_items": 12, "pending": 12, "captured": 0,
                                   "missing_records": 0, "anchor_errors": 12})
    # an empty work list is a bug at this point: resolve_cell_work already returned early
    with pytest.raises(C.CaptureCellError):
        C.assert_capture_progress({"work_items": 0, "pending": 0, "captured": 0})


def test_the_engine_factory_names_the_builder_when_the_snapshot_is_unpinned():
    """``options['snapshot_path']`` was the only hard KeyError in the worker, and it was
    reached *after* the empty-work-list exit — so the real run never saw it."""
    with pytest.raises(C.CaptureCellError, match="snapshot_path"):
        C.default_engine_factory({"relative_layers": list(K.CAPTURE_RELATIVE_LAYERS)})
    with pytest.raises(C.CaptureCellError, match="offline HF snapshot"):
        C.default_engine_factory({"snapshot_path": "/nonexistent/snapshot"})


def test_the_protocol_filter_keeps_the_shared_round_0_roots(tok, tmp_path):
    """``options['protocols']`` narrows rounds 1-3 only: one root is replayed once and
    joined by every protocol (spec §14.2 "reuse root captures across protocols")."""
    run_root, options = capture_run(tok, tmp_path, n=0)
    W.item_result_file(run_root, CAPTURE_CELL, "i1", decisions=[
        (did(300), "PEER", 0, 0), (did(301), "PEER", 1, 0), (did(302), "BALANCED", 1, 0)])
    both = C.resolve_cell_work(run_root, "i1", options)
    assert {(w.protocol, w.round_) for w in both} == {("PEER", 0), ("PEER", 1), ("BALANCED", 1)}
    narrowed = C.resolve_cell_work(run_root, "i1", {**options, "protocols": ["PEER"]})
    assert {(w.protocol, w.round_) for w in narrowed} == {("PEER", 0), ("PEER", 1)}
