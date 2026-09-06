"""N1 engine on CPU with a two-layer random Qwen3 (hidden 32): hook convention, batching /
right-padding correctness, logits at positions, block helper, and the replay fidelity path."""

from __future__ import annotations

import numpy as np
import pytest

from agents_scaling.study.neural import engine as E
from agents_scaling.study.neural import replay as R
from tests.study_neural import support

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model():
    return support.tiny_qwen3(seed=0)


@pytest.fixture(scope="module")
def engine(model):
    eng = E.CaptureEngine(None, blocks=(0, 1), device_map="cpu", dtype=torch.float32, max_batch_tokens=64, model=model)
    yield eng
    eng.close()


def _ids(rng, n):
    return tuple(int(t) for t in rng.integers(1, support.TINY_VOCAB, size=n))


def test_blocks_for_and_plan_batches():
    assert E.blocks_for(64) == (15, 31, 47)
    assert E.blocks_for(36) == (8, 17, 26)
    assert E.blocks_for(40) == (9, 19, 29)
    assert E.blocks_for(2, (0.25, 0.5, 0.75)) == (0,)
    with pytest.raises(ValueError):
        E.blocks_for(0)
    batches = E.plan_batches([10, 30, 5, 30, 12, 1], 64)
    lengths = [10, 30, 5, 30, 12, 1]
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert len(b) * max(lengths[i] for i in b) <= 64
    assert batches[0] == [1, 3]  # the two longest fit together (2*30 <= 64)
    assert E.plan_batches([100], 64) == [[0]]  # an over-long sequence still runs alone


def test_hook_matches_output_hidden_states_and_padding_is_inert(engine, model):
    rng = np.random.default_rng(1)
    seqs = [_ids(rng, n) for n in (7, 13, 4, 20)]
    reqs = [E.CaptureRequest(f"s{i}", s, positions=(0, len(s) // 2, len(s) - 1), logit_positions=(len(s) - 1,)) for i, s in enumerate(seqs)]
    batched = engine.forward(reqs)
    assert [r.seq_id for r in batched] == ["s0", "s1", "s2", "s3"]
    assert engine.batches_run >= 2  # 20+13+7+4 tokens do not fit one 64-token batch when padded
    for req, res in zip(reqs, batched):
        single = engine.forward([req])[0]
        with torch.inference_mode():
            out = model(input_ids=torch.as_tensor([req.token_ids]), use_cache=False, output_hidden_states=True)
        for block in (0, 1):
            assert res.residuals[block].dtype == np.float16 and res.residuals[block].shape == (3, support.TINY_HIDDEN)
            # right padding is inert: batched (padded) capture == single-sequence capture
            np.testing.assert_allclose(res.residuals[block].astype(np.float32), single.residuals[block].astype(np.float32), rtol=2e-2, atol=2e-2)
        # hook convention: block b output == hidden_states[b+1] for b < L-1 (the tuple's last
        # entry is post-final-norm, which test_last_block_is_pre_final_norm covers)
        ref = out.hidden_states[1][0, list(req.positions)].to(torch.float16).numpy()
        np.testing.assert_allclose(res.residuals[0].astype(np.float32), ref.astype(np.float32), rtol=2e-2, atol=2e-2)
        ref_logits = out.logits[0, len(req.token_ids) - 1].numpy()
        np.testing.assert_allclose(res.logits[0], ref_logits, rtol=1e-3, atol=1e-3)
        assert res.measurement_cost["sequence_tokens"] == len(req.token_ids)
        assert res.measurement_cost["batch_tokens"] >= len(req.token_ids)
        assert res.residual_at(1, req.positions[-1]).shape == (support.TINY_HIDDEN,)


def test_last_block_is_pre_final_norm(engine, model):
    rng = np.random.default_rng(2)
    seq = _ids(rng, 9)
    res = engine.forward([E.CaptureRequest("x", seq, positions=(8,))])[0]
    with torch.inference_mode():
        out = model.model(input_ids=torch.as_tensor([seq]), use_cache=False, output_hidden_states=True)
    post_norm = out.last_hidden_state[0, 8].numpy()
    captured = res.residuals[1][0].astype(np.float32)
    assert not np.allclose(captured, post_norm, atol=1e-3)  # hook is before model.model.norm
    np.testing.assert_allclose(model.model.norm(torch.as_tensor(captured)).detach().numpy(), post_norm, rtol=5e-2, atol=5e-2)


def test_request_validation_and_engine_guards(engine):
    with pytest.raises(E.EngineError):
        E.CaptureRequest("a", (1, 2, 3), positions=(3,))
    with pytest.raises(E.EngineError):
        E.CaptureRequest("a", (1, 2, 3), positions=(1, 1))
    with pytest.raises(E.EngineError):
        E.CaptureRequest("a", (), positions=())
    with pytest.raises(E.EngineError):
        E.CaptureEngine(None, blocks=(5,), device_map="cpu", model=engine.model)
    long_engine = E.CaptureEngine(None, blocks=(1,), device_map="cpu", model=engine.model, max_seq_tokens=8)
    with pytest.raises(E.EngineError):
        long_engine.forward([E.CaptureRequest("a", tuple(range(1, 10)), positions=(0,))])
    long_engine.close()
    assert engine.describe()["hook_convention"] == E.HOOK_CONVENTION
    assert engine.describe()["padding"] == "right"


def _mask_labels(results):
    return {r.seq_id: r.extra["attention_mask"] for r in results}


def test_mask_policy_causal_matches_explicit_mask(engine, model):
    """Default policy passes no attention_mask: unpadded and padded batches must equal the
    explicit-mask computation on the captured (real-token) positions."""
    rng = np.random.default_rng(3)
    seqs = [_ids(rng, n) for n in (9, 16, 5, 16)]
    reqs = [E.CaptureRequest(f"m{i}", s, positions=(0, len(s) // 2, len(s) - 1), logit_positions=(len(s) - 1,)) for i, s in enumerate(seqs)]
    assert engine.mask_policy == "causal" == E.DEFAULT_MASK_POLICY
    causal_eng = E.CaptureEngine(None, blocks=(0, 1), device_map="cpu", dtype=torch.float32, max_batch_tokens=32, model=model)
    explicit = E.CaptureEngine(None, blocks=(0, 1), device_map="cpu", dtype=torch.float32, max_batch_tokens=32, model=model, mask_policy="explicit")
    try:
        causal = causal_eng.forward(reqs)
        masked = explicit.forward(reqs)
    finally:
        causal_eng.close()
        explicit.close()
    assert causal_eng.batches_run == 2 == explicit.batches_run  # [16,16] unpadded, then [9,5] padded
    assert set(_mask_labels(causal).values()) == {"none"}
    assert _mask_labels(masked) == {"m0": "padded", "m1": "all_ones", "m2": "padded", "m3": "all_ones"}
    for c, m in zip(causal, masked):
        assert c.seq_id == m.seq_id and c.extra["padded"] == m.extra["padded"]
        for block in (0, 1):
            np.testing.assert_allclose(c.residuals[block].astype(np.float32), m.residuals[block].astype(np.float32), rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(c.logits, m.logits, rtol=1e-3, atol=1e-3)
    # unpadded single rows: mask=None is bit-for-bit the model's own causal path
    single = engine.forward([reqs[0]])[0]
    with torch.inference_mode():
        ref = model(input_ids=torch.as_tensor([reqs[0].token_ids]), use_cache=False, output_hidden_states=True)
    np.testing.assert_allclose(single.residuals[0].astype(np.float32), ref.hidden_states[1][0, list(reqs[0].positions)].to(torch.float16).float().numpy(), rtol=2e-2, atol=2e-2)
    assert single.extra["attention_mask"] == "none" and single.extra["padded"] is False
    assert single.measurement_cost["attention_mask"] == "none" and single.measurement_cost["padded"] is False
    assert masked[0].measurement_cost["attention_mask"] == "padded" and masked[0].measurement_cost["padded"] is True


def test_mask_policy_padded_keeps_mask_only_with_padding(model):
    rng = np.random.default_rng(4)
    seqs = [_ids(rng, n) for n in (16, 16, 10, 3)]
    reqs = [E.CaptureRequest(f"p{i}", s, positions=(len(s) - 1,)) for i, s in enumerate(seqs)]
    eng = E.CaptureEngine(None, blocks=(0, 1), device_map="cpu", dtype=torch.float32, max_batch_tokens=32, model=model, mask_policy="padded")
    try:
        res = eng.forward(reqs)
        labels = _mask_labels(res)
        assert labels["p0"] == "none" and labels["p1"] == "none"  # 2x16 unpadded -> mask dropped
        assert labels["p2"] == "padded" and labels["p3"] == "padded"  # 10+3 padded -> explicit mask kept
        for req, r in zip(reqs, res):
            single = eng.forward([req])[0]
            assert single.extra["attention_mask"] == "none"
            np.testing.assert_allclose(r.residuals[1].astype(np.float32), single.residuals[1].astype(np.float32), rtol=2e-2, atol=2e-2)
    finally:
        eng.close()
    assert eng.describe()["mask_policy"] == "padded"
    with pytest.raises(E.EngineError):
        E.CaptureEngine(None, blocks=(0,), device_map="cpu", dtype=torch.float32, model=model, mask_policy="flash")


def test_max_batch_rows_forces_single_row_batches(model):
    assert E.plan_batches([10, 30, 5, 30, 12, 1], 64, max_batch_rows=1) == [[1], [3], [4], [0], [2], [5]]
    assert E.plan_batches([10, 30, 5, 30, 12, 1], 64, max_batch_rows=2)[0] == [1, 3]
    assert all(len(b) <= 2 for b in E.plan_batches([4] * 9, 64, max_batch_rows=2))
    with pytest.raises(ValueError):
        E.plan_batches([1], 64, max_batch_rows=0)
    rng = np.random.default_rng(5)
    reqs = [E.CaptureRequest(f"r{i}", _ids(rng, n), positions=(0,)) for i, n in enumerate((6, 8, 4))]
    eng = E.CaptureEngine(None, blocks=(0,), device_map="cpu", dtype=torch.float32, max_batch_tokens=64, model=model, max_batch_rows=1)
    try:
        res = eng.forward(reqs)
    finally:
        eng.close()
    assert eng.batches_run == 3 and all(r.batch_rows == 1 and r.extra["padded"] is False for r in res)
    assert eng.describe()["max_batch_rows"] == 1


def test_fidelity_on_greedy_tiny_completion(engine, model):
    rng = np.random.default_rng(3)
    prompt = list(_ids(rng, 6))
    with torch.inference_mode():
        out = model.generate(torch.as_tensor([prompt]), max_new_tokens=12, do_sample=False, pad_token_id=0)
    completion = out[0, len(prompt):].tolist()
    rec = support.synthetic_record(prompt, completion, content=None, enable_thinking=False)
    rep = R.fidelity(rec, engine, n_tokens=12)
    assert rep["scored_tokens"] == 12
    assert rep["argmax_agreement"] == 1.0 and rep["max_rank"] == 0
    assert rep["mean_logprob"] <= 0.0
    assert rep["two_pass_max_abs_logit_diff"] == 0.0
    assert R.teacher_forced_sequence(rec, 5) == tuple(prompt + completion[:5])
    with pytest.raises(R.ReplayError):
        R.teacher_forced_sequence(rec, 13)
    summary = R.summarize_fidelity([rep, rep])
    assert summary["records"] == 2 and summary["argmax_agreement"] == 1.0
