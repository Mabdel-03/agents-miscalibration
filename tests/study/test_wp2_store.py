"""inference/store.py: layout, strict reads, O_EXCL race, aliasing without HTTP (§3.6, §10.4)."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.inference import client as C
from agents_scaling.study.inference.store import RequestStore, load_record_text
from tests.study.wp2_support import fake_server, make_spec


def _record_from_fixture(request_record_fixture) -> T.RequestRecord:
    record = T.RequestRecord.from_dict(request_record_fixture)
    record.verify()
    return record


def test_path_layout_and_id_validation(tmp_run_root):
    store = RequestStore(tmp_run_root / "requests")
    rid = "ab" + "0" * 62
    assert store.path(rid) == tmp_run_root / "requests" / "ab" / f"{rid}.json"
    assert store.get(rid) is None and not store.exists(rid)
    for bad in ("", "AB" + "0" * 62, "x" * 64, "../" + "0" * 61):
        with pytest.raises(T.ProtocolError):
            store.path(bad)


def test_publish_get_roundtrip_and_idempotence(tmp_run_root, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    record = _record_from_fixture(request_record_fixture)
    committed, was_new = store.publish(record)
    assert was_new and committed == record and store.exists(record.request_id)
    again, was_new2 = store.publish(dataclasses.replace(record, timing={"submitted_at": 9.0, "completed_at": 9.5, "latency_s": 0.5}).with_content_sha256())
    assert not was_new2 and again == record  # the first writer's bytes stay
    assert store.get(record.request_id) == record
    assert not list((tmp_run_root / "requests" / record.request_id[:2]).glob(".*.tmp"))
    assert json.loads(store.path(record.request_id).read_text())["content_sha256"] == record.content_sha256


def test_publish_requires_verified_hash(tmp_run_root, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    record = _record_from_fixture(request_record_fixture)
    with pytest.raises(T.ProtocolError, match="content_sha256 is unset"):
        store.publish(dataclasses.replace(record, content_sha256=None))
    with pytest.raises(T.ProtocolError, match="content hash"):
        store.publish(dataclasses.replace(record, content_sha256="0" * 64))
    assert not store.exists(record.request_id)


def test_corrupt_record_is_protocol_error(tmp_run_root, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    record = _record_from_fixture(request_record_fixture)
    store.publish(record)
    path = store.path(record.request_id)
    good = path.read_text()
    # 1) a flipped byte in a stored field
    path.write_text(good.replace('"finish_reason": "stop"', '"finish_reason": "length"'))
    with pytest.raises(T.ProtocolError, match="content hash"):
        store.get(record.request_id)
    # 2) duplicate key
    path.write_text(good.replace('"attempts": 1,', '"attempts": 1, "attempts": 1,'))
    with pytest.raises(T.ProtocolError, match="duplicate key"):
        store.get(record.request_id)
    # 3) truncated JSON
    path.write_text(good[:-40])
    with pytest.raises(T.ProtocolError, match="strict JSON"):
        store.get(record.request_id)
    # 4) NaN constant
    path.write_text(good.replace('"latency_s": 1.5', '"latency_s": NaN'))
    with pytest.raises(T.ProtocolError, match="non-finite"):
        store.get(record.request_id)
    # 5) file name disagrees with the record id
    other = "ff" + record.request_id[2:]
    store.path(other).parent.mkdir(parents=True, exist_ok=True)
    store.path(other).write_text(good)
    with pytest.raises(T.ProtocolError, match="file name disagrees"):
        store.get(other)
    with pytest.raises(T.ProtocolError, match="not a JSON object"):
        load_record_text("[1, 2]")
    with pytest.raises(T.ProtocolError, match="invalid shape"):
        load_record_text('{"request_id": "x"}')


def test_publish_race_exactly_one_new(tmp_run_root, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    base = _record_from_fixture(request_record_fixture)
    n = 8
    records = [dataclasses.replace(base, timing={"submitted_at": float(i), "completed_at": float(i) + 1, "latency_s": 1.0}).with_content_sha256() for i in range(n)]
    assert len({r.content_sha256 for r in records}) == n
    barrier = threading.Barrier(n)
    results: list[tuple[T.RequestRecord, bool]] = [None] * n  # type: ignore[list-item]
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait()
            results[i] = store.publish(records[i])
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sum(1 for _, was_new in results if was_new) == 1
    winner = next(r for r, was_new in results if was_new)
    assert all(r == winner for r, _ in results)
    assert store.get(base.request_id) == winner
    assert not list(store.path(base.request_id).parent.glob(".*.tmp"))


def test_race_with_different_identity_is_protocol_error(tmp_run_root, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    record = _record_from_fixture(request_record_fixture)
    store.publish(record)
    imposter = dataclasses.replace(record, identity=dict(record.identity, study_id="other")).with_content_sha256()
    with pytest.raises(T.ProtocolError, match="different identity"):
        store.publish(imposter)


def test_get_or_generate_alias_hit_makes_zero_http_calls(tmp_run_root, study_config):
    store = RequestStore(tmp_run_root / "requests")
    spec = make_spec(study_config)
    events = []
    with fake_server(tmp_run_root) as srv:
        pool = C.EndpointPool(tmp_run_root, "32B-long", shard=0, refresh_min_interval_s=0.0, probe_timeout=2.0)
        client = C.VllmChatClient(pool, srv.tokenizer, run_id="r")
        first, aliased = store.get_or_generate(spec, client, "A.cell", on_event=lambda k, d: events.append((k, d)))
        assert not aliased and srv.chat_count == 1 and first.producer["cell_id"] == "A.cell"
        # A different cell, a fresh client: the record is read, nothing is generated.
        client2 = C.VllmChatClient(C.EndpointPool(tmp_run_root, "32B-long", shard=0), srv.tokenizer, run_id="r")
        second, aliased2 = store.get_or_generate(spec, client2, "F.cell", on_event=lambda k, d: events.append((k, d)))
        assert aliased2 and second == first and srv.chat_count == 1
        assert events == [("aliased", {"request_id": spec.request_id, "cell_id": "F.cell"})]
        # Even with the server dead the alias is served from disk.
        srv.kill()
        third, aliased3 = store.get_or_generate(spec, client2, "B.cell")
        assert aliased3 and third == first


def test_get_or_generate_lost_race_reports_alias_race(tmp_run_root, study_config, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    spec = make_spec(study_config)
    events = []

    class RacingClient:
        """Generates a record, but a competitor commits the same id first."""

        def __init__(self, template):
            self.template = template
            self.calls = 0

        def generate(self, spec, *, cell_id=None, **kwargs):
            self.calls += 1
            mine = dataclasses.replace(
                self.template, request_id=spec.request_id, identity=spec.identity_fields(), seed_key=spec.seed_key,
                engine_seed=spec.engine_seed, messages=spec.messages, producer={"cell_id": cell_id, "run_id": "r", "slurm_job_id": None},
                timing={"submitted_at": 1.0, "completed_at": 2.0, "latency_s": 1.0},
            ).with_content_sha256()
            theirs = dataclasses.replace(mine, producer={"cell_id": "OTHER", "run_id": "r", "slurm_job_id": None}).with_content_sha256()
            store.publish(theirs)
            return mine

    client = RacingClient(T.RequestRecord.from_dict(request_record_fixture))
    record, aliased = store.get_or_generate(spec, client, "ME", on_event=lambda k, d: events.append((k, d)))
    assert client.calls == 1 and not aliased
    assert record.producer["cell_id"] == "OTHER"  # the committed record wins; ours is discarded
    assert events == [("alias_race", {"request_id": spec.request_id, "cell_id": "ME", "winner_cell_id": "OTHER"})]


def test_get_or_generate_rejects_mismatched_client_output(tmp_run_root, study_config, request_record_fixture):
    store = RequestStore(tmp_run_root / "requests")
    spec = make_spec(study_config)

    class WrongClient:
        def generate(self, spec, *, cell_id=None, **kwargs):
            return T.RequestRecord.from_dict(request_record_fixture)  # fixture id != spec id

    with pytest.raises(T.ProtocolError, match="client returned request_id"):
        store.get_or_generate(spec, WrongClient(), "X")
    assert not store.exists(spec.request_id)


def test_get_or_generate_never_generates_on_corrupt_record(tmp_run_root, study_config):
    store = RequestStore(tmp_run_root / "requests")
    spec = make_spec(study_config)
    path = store.path(spec.request_id)
    path.parent.mkdir(parents=True)
    path.write_text("{corrupt")

    class NeverClient:
        def generate(self, spec, *, cell_id=None, **kwargs):  # pragma: no cover
            raise AssertionError("must not generate over a corrupt record")

    with pytest.raises(T.ProtocolError):
        store.get_or_generate(spec, NeverClient(), "X")


def test_generated_record_round_trips_through_store(tmp_run_root, study_config):
    """T16-style stability: same spec twice → same id, second call is a store hit with equal bytes."""
    store = RequestStore(tmp_run_root / "requests")
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(C.EndpointPool(tmp_run_root, "32B-long", shard=0, probe_timeout=2.0), srv.tokenizer)
        rec1, _ = store.get_or_generate(spec, client, "c1")
        rec2, aliased = store.get_or_generate(spec, client, "c2")
        assert aliased and rec2 == rec1 and rec2.to_json() == store.path(spec.request_id).read_text()
        # A one-byte prompt change is a different id and a fresh generation.
        spec2 = dataclasses.replace(spec, messages=({"role": "user", "content": spec.messages[0]["content"] + "!"},))
        assert spec2.request_id != spec.request_id
        rec3, aliased3 = store.get_or_generate(spec2, client, "c3")
        assert not aliased3 and rec3.request_id == spec2.request_id and srv.chat_count == 2
