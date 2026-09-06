"""GPU verification of the N1 capture engine (run inside a Slurm GPU job with neural_env).

    PYTHONPATH=src python tests/study_neural/gpu_smoke.py --size 8B --out <report.json>

Checks (a) load + hooks + forward on 3 synthetic sequences (batched vs single, hook capture
vs ``output_hidden_states``) and on up to 3 real request records if the run store has any,
otherwise on greedy HF generations turned into synthetic records; (b) fidelity numbers;
(c) throughput at ``--max-batch-tokens``; (d) an end-to-end anchors → storage round trip.
Writes one JSON report; exit code 0 iff every hard check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from agents_scaling.study.neural import anchors as A  # noqa: E402
from agents_scaling.study.neural import engine as E  # noqa: E402
from agents_scaling.study.neural import replay as R  # noqa: E402
from agents_scaling.study.neural import storage as S  # noqa: E402
from tests.study_neural import support  # noqa: E402

PROMPTS = [
    "What is 2 + 2? Answer with the integer only.",
    "请用中文解释一下什么是残差流 (residual stream)，并给出一个 JSON 例子: {\"a\": 1}。",
    "Write a Python function that returns the n-th Fibonacci number.\n\n=== TASK ===\nfib\n=== END TASK ===\n",
]


def real_records(requests_root: Path, n: int) -> list:
    from agents_scaling.study.inference.store import load_record_text

    out = []
    if not requests_root.is_dir():
        return out
    for sub in sorted(requests_root.iterdir())[:64]:
        if not sub.is_dir():
            continue
        for path in sorted(sub.glob("*.json"))[:n]:
            out.append(load_record_text(path.read_text(encoding="utf-8")))
            if len(out) >= n:
                return out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--requests-root", default="/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/requests")
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--fidelity-tokens", type=int, default=20)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--gen-tokens", type=int, default=64)
    args = ap.parse_args(argv)

    import torch

    report: dict = {"size": args.size, "checks": {}, "hard_failures": []}
    t0 = time.time()
    snapshot = support.snapshot_path(args.size)
    cfg = json.loads((snapshot / "config.json").read_text())
    L = int(cfg["num_hidden_layers"])
    blocks = E.blocks_for(L)
    tok = support.qwen_tokenizer(args.size)
    eng = E.CaptureEngine(str(snapshot), blocks, device_map=args.device_map, max_batch_tokens=args.max_batch_tokens)
    report["engine"] = eng.describe()
    report["engine"]["cuda_devices"] = torch.cuda.device_count()
    report["engine"]["gpu_mem_after_load_gib"] = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None
    print(f"[smoke] loaded {args.size} in {eng.load_seconds:.1f}s blocks={blocks} L={L}", flush=True)

    # ---- (a) synthetic sequences: greedy generations become stored completions --------------
    synthetic = []
    gen_device = eng.input_device
    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
        with torch.inference_mode():
            out = eng.model.generate(
                torch.as_tensor([ids], device=gen_device), max_new_tokens=args.gen_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        completion = out[0, len(ids):].tolist()
        text = tok.decode(completion)
        think_end = tok.convert_tokens_to_ids("</think>")
        if think_end in completion:
            j = completion.index(think_end)
            content = tok.decode(completion[j + 1:], skip_special_tokens=True)
            reasoning = tok.decode(completion[:j])
        else:
            content, reasoning = None, tok.decode(completion)
        rec = support.synthetic_record(ids, completion, messages=messages, content=content, reasoning=reasoning,
                                       finish_reason="stop" if completion and completion[-1] == tok.eos_token_id else "length",
                                       request_id=f"{i:064x}")
        synthetic.append(rec)
        print(f"[smoke] synthetic {i}: prompt={len(ids)} completion={len(completion)} text={text[:60]!r}", flush=True)
    # one hand-written completion with a valid candidate object (exercises FINAL_OBJECT_CLOSE)
    messages = [{"role": "user", "content": PROMPTS[0]}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
    content = "\n\n" + support.CANDIDATE_JSON_CJK
    completion = support.completion_from_text(tok, "两个整数相加。", content)
    synthetic.append(support.synthetic_record(ids, completion, messages=messages, content=content, reasoning="两个整数相加。", request_id=f"{9:064x}"))

    # anchors + batched vs single forward + hook convention check
    requests = []
    anchor_sets = []
    for rec in synthetic:
        anc = A.resolve_native_anchors(rec, tok, ks=(8, 32, 512))
        anchor_sets.append(anc)
        seq = R.teacher_forced_sequence(rec, A.k_max_needed(anc))
        positions = tuple(sorted({a.token_offset for a in anc if a.present}))
        requests.append(E.CaptureRequest(rec["request_id"], seq, positions, logit_positions=(len(rec["prompt_token_ids"]) - 1,)))
        R.check_prompt_identity(rec, tok)
    report["checks"]["anchors"] = [[a.to_dict() for a in anc] for anc in anchor_sets]
    batched = eng.forward(requests)
    singles = [eng.forward([r])[0] for r in requests]
    max_diff, max_rel = 0.0, 0.0
    for b, s in zip(batched, singles):
        for blk in blocks:
            x = b.residuals[blk].astype(np.float32)
            y = s.residuals[blk].astype(np.float32)
            max_diff = max(max_diff, float(np.abs(x - y).max()))
            max_rel = max(max_rel, float(np.abs(x - y).max() / max(1e-6, np.abs(y).max())))
    report["checks"]["batched_vs_single_max_abs_diff"] = max_diff
    report["checks"]["batched_vs_single_max_rel_diff"] = max_rel
    # hook convention vs output_hidden_states on the shortest sequence
    r0 = requests[0]
    with torch.inference_mode():
        hs = eng.base(input_ids=torch.as_tensor([r0.token_ids], device=eng.input_device), use_cache=False, output_hidden_states=True).hidden_states
    conv = {}
    for blk in blocks:
        ref = hs[blk + 1][0, list(r0.positions)].to(torch.float16).cpu().numpy().astype(np.float32)
        conv[blk] = float(np.abs(ref - singles[0].residuals[blk].astype(np.float32)).max())
    report["checks"]["hook_vs_output_hidden_states_max_abs_diff"] = conv
    if max(conv.values()) > 0.0:
        report["hard_failures"].append("hook capture != hidden_states[b+1]")
    if max_rel > 2e-2:
        report["hard_failures"].append(f"batched vs single residual relative diff {max_rel} > 2e-2 (bf16 tolerance)")
    print(f"[smoke] batched-vs-single max abs diff {max_diff:.4g} (rel {max_rel:.3g}); hook-vs-hidden_states {conv}", flush=True)

    # ---- (b) fidelity ---------------------------------------------------------------------
    fid = []
    for rec in synthetic[:3]:
        try:
            fid.append(R.fidelity(rec, eng, args.fidelity_tokens, tokenizer=tok))
        except R.ReplayError as exc:
            fid.append({"request_id": rec["request_id"], "error": str(exc)})
    report["checks"]["fidelity_synthetic_greedy"] = {"summary": R.summarize_fidelity([f for f in fid if "error" not in f]), "per_record": fid}
    greedy_agree = report["checks"]["fidelity_synthetic_greedy"]["summary"].get("argmax_agreement")
    print(f"[smoke] greedy fidelity: {report['checks']['fidelity_synthetic_greedy']['summary']}", flush=True)
    if greedy_agree is not None and greedy_agree < 0.9:
        report["hard_failures"].append(f"greedy argmax agreement {greedy_agree} < 0.9")
    reals = []
    try:
        reals = real_records(Path(args.requests_root), args.n_real)
    except Exception as exc:  # noqa: BLE001
        report["checks"]["real_records_error"] = repr(exc)
    real_fid = []
    real_anchors = []
    for rec in reals:
        try:
            R.check_prompt_identity(rec, tok)
            real_anchors.append([a.to_dict() for a in A.resolve_native_anchors(rec, tok)])
            real_fid.append(R.fidelity(rec, eng, args.fidelity_tokens, tokenizer=tok))
        except Exception as exc:  # noqa: BLE001
            real_fid.append({"request_id": rec.request_id, "error": repr(exc)})
    report["checks"]["real_records"] = {"count": len(reals), "fidelity": real_fid, "anchors": real_anchors,
                                         "summary": R.summarize_fidelity([f for f in real_fid if "error" not in f])}
    print(f"[smoke] real records: {len(reals)}", flush=True)

    # ---- (c) throughput -----------------------------------------------------------------------
    para = tok.encode("The quick brown fox jumps over the lazy dog. " * 40, add_special_tokens=False)
    lengths = [1024, 2048, 4096, 8192, args.max_batch_tokens]
    tp_requests = []
    for n in lengths:
        ids = (para * (n // len(para) + 1))[:n]
        tp_requests.append(E.CaptureRequest(f"tp{n}", tuple(ids), (n - 1,)))
    tp_requests += [E.CaptureRequest(f"tp2048_{i}", tuple((para * 20)[:2048]), (2047,)) for i in range(7)]
    before_tokens, before_seconds = eng.tokens_run, eng.forward_seconds
    started = time.perf_counter()
    res = eng.forward(tp_requests)
    wall = time.perf_counter() - started
    tokens = eng.tokens_run - before_tokens
    fwd = eng.forward_seconds - before_seconds
    per_batch = sorted({(r.batch_index, r.batch_rows, r.batch_tokens, round(r.batch_seconds, 3)) for r in res})
    report["checks"]["throughput"] = {
        "tokens": tokens, "forward_seconds": fwd, "wall_seconds": wall,
        "tokens_per_second_forward": tokens / fwd if fwd else None, "tokens_per_second_wall": tokens / wall,
        "batches": [{"index": b, "rows": r, "tokens": t, "seconds": s} for b, r, t, s in per_batch],
        "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
    }
    print(f"[smoke] throughput {tokens} tokens in {fwd:.2f}s forward = {tokens / fwd if fwd else 0:.0f} tok/s; batches={per_batch}", flush=True)

    # ---- (d) storage round trip ---------------------------------------------------------
    try:
        storage_round_trip(report, synthetic, anchor_sets, batched, blocks, eng, args)
    except Exception as exc:  # noqa: BLE001
        report["checks"]["storage_round_trip"] = {"error": repr(exc), "traceback": traceback.format_exc()}
        report["hard_failures"].append(f"storage round trip raised {exc!r}")

    report["seconds_total"] = time.time() - t0
    report["ok"] = not report["hard_failures"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"[smoke] ok={report['ok']} failures={report['hard_failures']} report={out}", flush=True)
    eng.close()
    return 0 if report["ok"] else 1


def storage_round_trip(report, synthetic, anchor_sets, batched, blocks, eng, args) -> None:
    tmp_root = REPO / ".tmp" / "neural" / f"smoke_{os.environ.get('SLURM_JOB_ID', 'local')}"
    with S.ShardWriter(tmp_root, "native", "smoke", chunk_rows=5) as writer:
        for rec, anc, res_ in zip(synthetic, anchor_sets, batched):
            for a in anc:
                for blk in blocks:
                    vec = res_.residual_at(blk, a.token_offset) if a.present else None
                    row = S.ActivationRow(
                        StateSnapshot_id=rec["request_id"], consumer_revision=support.SNAPSHOTS[args.size][1], condition="smoke",
                        child_slot=None, nonce_hash="", block=blk, anchor_kind=a.kind, structural_span=a.structural_span,
                        token_offset=a.token_offset, channel=a.channel, generated_token_count=a.generated_token_count,
                        missingness=a.missingness, tensor_hash=None, measurement_cost=res_.measurement_cost,
                        operationally_available_at_checkpoint=True, stage="native", sequence_id=rec["request_id"],
                        hidden_size=eng.hidden_size, sequence_tokens=res_.n_tokens,
                    )
                    writer.add(row, vec)
    matrix, frame = S.load_stage(tmp_root, "native")
    n_present = int((frame["vector_index"] >= 0).sum())
    ok = matrix.shape == (n_present, eng.hidden_size) and len(frame) == sum(len(a) for a in anchor_sets) * len(blocks)
    # a stored vector round-trips exactly
    row0 = frame[frame["vector_index"] >= 0].iloc[0]
    res0 = next(r for r in batched if r.seq_id == row0["StateSnapshot_id"])
    exact = np.array_equal(matrix[int(row0["vector_index"])], res0.residual_at(int(row0["block"]), int(row0["token_offset"])))
    report["checks"]["storage_round_trip"] = {"rows": len(frame), "present": n_present, "matrix_shape": list(matrix.shape), "shape_ok": ok, "exact": bool(exact),
                                              "chunks": [p.name for p in writer.chunks_written], "root": str(tmp_root)}
    if not (ok and exact):
        report["hard_failures"].append("storage round trip failed")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
