"""N1 storage: chunked npz/jsonl round trip, atomicity, resume keys, loader filters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents_scaling.study.neural import storage as S


def _row(sid: str, block: int, kind: str, *, missing: str | None = None, **extra) -> S.ActivationRow:
    return S.ActivationRow(
        StateSnapshot_id=sid, consumer_revision="rev", condition="native:IND_VOTE", child_slot=None, nonce_hash="n",
        block=block, anchor_kind=kind, structural_span="prompt[0:10)", token_offset=None if missing else 9, channel="prompt",
        generated_token_count=0, missingness=missing, tensor_hash=None, measurement_cost={"sequence_tokens": 10},
        operationally_available_at_checkpoint=True, stage="native", sequence_id=sid, hidden_size=8, **extra,
    )


def test_round_trip_with_chunking_and_missing_rows(tmp_path: Path):
    rng = np.random.default_rng(0)
    vectors = {}
    with S.ShardWriter(tmp_path, "native", "native.s000of002", chunk_rows=3) as w:
        for i in range(4):
            sid = f"req{i}"
            for block in (1, 3):
                v = rng.standard_normal(8).astype(np.float16)
                vectors[(sid, block, "NATIVE_PREFILL")] = v
                w.add(_row(sid, block, "NATIVE_PREFILL", source_id=f"hle:{i}"), v)
            w.add(_row(sid, 1, "GENERATED_512", missing="NOT_REACHED"), None)
    assert len(w.chunks_written) == 3  # 8 vectors / 3 per chunk, last flush on exit
    names = sorted(p.name for p in (tmp_path / "neural" / "native").iterdir())
    assert names[:2] == ["native.s000of002.c00000.jsonl", "native.s000of002.c00000.npz"]
    matrix, frame = S.load_stage(tmp_path, "native")
    assert matrix.shape == (8, 8) and matrix.dtype == np.float16
    assert len(frame) == 12
    present = frame[frame.vector_index >= 0]
    assert len(present) == 8
    for _, r in present.iterrows():
        assert np.array_equal(matrix[int(r.vector_index)], vectors[(r.StateSnapshot_id, int(r.block), r.anchor_kind)])
        assert r.tensor_hash == S.tensor_hash(vectors[(r.StateSnapshot_id, int(r.block), r.anchor_kind)])
    missing = frame[frame.vector_index < 0]
    assert set(missing.missingness) == {"NOT_REACHED"} and set(missing.anchor_kind) == {"GENERATED_512"}
    assert list(S.ACTIVATION_ROW_FIELDS) == [c for c in frame.columns if c in S.ACTIVATION_ROW_FIELDS]
    # filters
    m2, f2 = S.load_stage(tmp_path, "native", blocks=[3], include_missing=False)
    assert m2.shape == (4, 8) and set(f2.block) == {3} and (f2.vector_index >= 0).all()
    m3, f3 = S.load_stage(tmp_path, "native", anchor_kinds=["GENERATED_512"])
    assert m3.shape[0] == 0 and len(f3) == 4
    # resume keys
    keys = S.existing_keys(tmp_path, "native")
    assert ("req0", 1, "NATIVE_PREFILL") in keys and ("req0", 1, "GENERATED_512") in keys and len(keys) == 12


def test_resume_appends_new_chunks_and_rejects_duplicates(tmp_path: Path):
    v = np.ones(8, dtype=np.float16)
    with S.ShardWriter(tmp_path, "report", "report.s000of001") as w:
        w.add(_row("r1", 1, "STATE_ANCHOR"), v)
    with S.ShardWriter(tmp_path, "report", "report.s000of001") as w:
        assert w._index == 1
        w.add(_row("r2", 1, "STATE_ANCHOR"), v)
    assert [p.name for p in sorted((tmp_path / "neural" / "report").glob("*.jsonl"))] == ["report.s000of001.c00000.jsonl", "report.s000of001.c00001.jsonl"]
    with S.ShardWriter(tmp_path, "report", "other") as w:
        w.add(_row("r1", 1, "STATE_ANCHOR"), v)  # same key from another shard
    with pytest.raises(S.StorageError):
        S.load_stage(tmp_path, "report")


def test_writer_guards(tmp_path: Path):
    v = np.ones(8, dtype=np.float16)
    with S.ShardWriter(tmp_path, "native", "s") as w:
        with pytest.raises(S.StorageError):
            w.add(_row("a", 1, "X"), None)  # no vector, no missingness
        with pytest.raises(S.StorageError):
            w.add(_row("a", 1, "X", missing="NOT_REACHED"), v)  # missing anchors never carry vectors
        with pytest.raises(S.StorageError):
            w.add(_row("a", 1, "X"), np.array([np.nan] * 8, dtype=np.float16))
        w.add(_row("a", 1, "X"), v)
        with pytest.raises(S.StorageError):
            w.add(_row("a", 1, "X"), v)  # duplicate key in one chunk
        with pytest.raises(S.StorageError):
            w.add(_row("b", 1, "X"), np.ones(4, dtype=np.float16))  # hidden size mismatch
    with pytest.raises(ValueError):
        S.ShardWriter(tmp_path, "native", "bad.c0")
    with pytest.raises(ValueError):
        S.stage_dir(tmp_path, "other")


def test_corrupt_chunk_is_detected_and_orphan_npz_is_skipped(tmp_path: Path):
    v = np.ones(8, dtype=np.float16)
    with S.ShardWriter(tmp_path, "native", "s") as w:
        w.add(_row("a", 1, "X"), v)
        w.add(_row("b", 1, "X"), v)
    d = tmp_path / "neural" / "native"
    jsonl = d / "s.c00000.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    rows[1]["vector_index"] = 0  # two rows pointing at one vector
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(S.StorageError):
        S.load_stage(tmp_path, "native")
    jsonl.unlink()  # npz without jsonl = a crashed writer's orphan: ignored by readers, its index is never reused
    assert S.list_chunks(d) == []
    assert S.existing_keys(tmp_path, "native") == set()
    assert S.next_chunk_index(d, "s") == 1
    (d / "s.c00003.jsonl").write_text("{}\n")  # jsonl without npz is a real inconsistency
    with pytest.raises(S.StorageError):
        S.list_chunks(d)


def test_tensor_hash_is_over_fp16_bytes():
    a = np.arange(8, dtype=np.float32)
    assert S.tensor_hash(a) == S.tensor_hash(a.astype(np.float16))
    assert S.tensor_hash(a) != S.tensor_hash(a + 1)
    assert S.parse_key(S.key_str("a|b", 3, "STATE_ANCHOR")) == ("a|b", 3, "STATE_ANCHOR")
