"""Read-only activation capture (WP6): anchors from real byte spans, replay, ledger.

CPU only.  The model is ``tests/study_neural/support.tiny_qwen3`` -- a two-layer random
Qwen3 over the **real** Qwen3 vocabulary, beside the pinned offline tokenizer -- exactly as
``tests/ard/wp6_support.py`` drives it, so every anchor is resolved against real tokenizer
byte spans while a forward pass costs milliseconds.

What is pinned here (spec 13.2; plan §1.2 WP6):

* the three anchors come from ``PublicOutput.final_answer_span`` / ``confidence_span``, which
  is to say from the bytes the WP2 parser actually read -- never a fixed offset;
* a capture whose anchor does not exist is a typed row with a reason and **no vector**, and
  the substitute anchor is never chosen by correctness;
* ``ANSWER_END`` is pre-confidence-eligible only when confidence had not already been emitted;
* blocks are ``floor(f*L)-1`` at the pinned model's own ``L``, and both the cell options and
  the engine the runner hands over are held to that derivation;
* the replay prefills the **exact stored token ids**, including across the two physical
  requests of a continued draw -- whose phase-2 record carries ``messages=()`` and
  ``enable_thinking=False``, so the public span is located structurally and its origin proved
  against the parser's own content hash, and the identity check runs on phase 1;
* nothing to replay (no request, no stored record) is a typed ledger row, not a cell failure;
* the hook location and the two-pass logit spread are validated on the fixture and logged;
* the capture module cannot generate: its import closure never reaches ``nextdyn.inference``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents_scaling.nextdyn import capture as CAP
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn import schema as SC
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn.errors import ContractViolation, DependencyUnmet
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.study.neural import storage as S
from agents_scaling.study.neural.engine import HOOK_CONVENTION
from tests.study_neural import support

REPO = Path(__file__).resolve().parents[2]
MODEL_HASH = "0123456789abcdef"
TASK = "What is 6 times 7?"
LAYERS_8B, LAYERS_32B = 36, 64
TINY_BLOCKS = (0, 1)

#: The spec-6.1 public object, with the fields in the declared order.
REPORT = ('\n\n{"reasoning": "checked twice", "shareable_claims": ["one claim"], '
          '"final_answer": "42", "confidence": 0.87}')
#: Same object with the confidence value emitted *before* the answer (spec 13.2 eligibility).
REPORT_CONF_FIRST = ('\n\n{"reasoning": "checked twice", "confidence": 0.87, '
                     '"final_answer": "42", "shareable_claims": ["one claim"]}')
#: A parse the answer never survives: no ``final_answer`` member at all.
REPORT_INVALID = '\n\n{"reasoning": "checked twice", "confidence": 0.87}'
#: ``decoding.CONTINUATION_PREFILL_TEXT`` (plan D5) -- spelled out here because the capture
#: import closure may not reach the generation path (``test_read_only_capture...``).
PREFILL_TEXT = '\n\n{\n  "reasoning": "'


# =============================================================================== fixtures


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return support.qwen_tokenizer("8B")
    except (FileNotFoundError, OSError) as exc:                  # pragma: no cover - no cache
        pytest.skip(f"the pinned Qwen3 tokenizer is not in the offline cache: {exc}")


@pytest.fixture
def tokens() -> K.TokenContract:
    return K.TokenContract(thinking_cap=16, final_cap=64, served_window=4096, reserve=8,
                           packet_cap=64, rationale_cap=64, claims_max=4, claim_cap=32)


@pytest.fixture
def tiny_engine(tokenizer):
    torch = pytest.importorskip("torch")
    from agents_scaling.study.neural.engine import CaptureEngine

    vocab = int(tokenizer.vocab_size) + len(tokenizer.added_tokens_decoder) + 64
    model = support.tiny_qwen3(seed=0, vocab_size=vocab)
    return CaptureEngine(None, TINY_BLOCKS, device_map="cpu", dtype=torch.float32,
                         model=model, max_batch_tokens=4096)


class StubStore:
    """A ``study.inference.store.RequestStore``-shaped store: ``get(request_id)``."""

    def __init__(self, records: dict[str, Any] | None = None) -> None:
        self._by_id = dict(records or {})

    def add(self, request_id: str, record: Any) -> Any:
        self._by_id[str(request_id)] = record
        return record

    def get(self, request_id: str) -> Any:
        return self._by_id.get(str(request_id))


class RecordingEngine:
    """Transparent wrapper keeping every :class:`CaptureRequest` the pass submitted."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.requests: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def forward(self, requests):
        self.requests.extend(requests)
        return self.engine.forward(requests)


class FakeCell:
    def __init__(self, options: dict[str, Any], *, cell_id: str = "capture.MATH.8B.s000") -> None:
        self.cell_id = cell_id
        self.kind = CellKind.CAPTURE
        self.options = dict(options)


class FakeRunner:
    """The ``NextdynCellRunner`` surface :func:`capture.capture_worker` touches."""

    def __init__(self, run_root: Path, cfg: Any, options: dict[str, Any], *, store: Any,
                 engine: Any, tokenizer: Any, cell_id: str = "capture.MATH.8B.s000",
                 models_lock: Any = None) -> None:
        self.run_root = Path(run_root)
        self.cfg = cfg
        self.cell = FakeCell(options, cell_id=cell_id)
        self.store = store
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._engine, self._tokenizer = engine, tokenizer
        self.models_lock = models_lock

    def capture_engine(self):
        return self._engine, self._tokenizer

    def event(self, name: str, **fields: Any) -> None:
        self.events.append((name, dict(fields)))


def _tiny_lock(n_layers: int = 2, name: str = "primary"):
    """A ``models.lock.json`` pinning the tiny fixture's depth -- the only place ``L`` comes
    from, so the worker never takes a block number on trust."""
    from agents_scaling.nextdyn.locks import ModelsLock

    return ModelsLock(schema_version=1, locked_at=None, hf_home=None, vllm_version=None,
                      torch_version=None, cuda_version=None, models={name: _lock_entry(n_layers)})


def _tiny_cfg(study_cfg):
    """The committed config at the fixture's depth: ``floor(f*2)-1`` for ``f`` in (0.5, 1.0) is
    exactly the two blocks a two-layer model has, so the spec-13.2 rule still does the deriving."""
    import dataclasses

    return dataclasses.replace(study_cfg, instrumentation=dataclasses.replace(
        study_cfg.instrumentation, relative_layers=(0.5, 1.0)))


def _worker_runner(run_root: Path, study_cfg, options: dict[str, Any], **kwargs: Any) -> FakeRunner:
    return FakeRunner(run_root, _tiny_cfg(study_cfg), {"model": "primary", **options},
                      models_lock=_tiny_lock(), **kwargs)


# =============================================================================== builders


def _record(tokenizer, request_id: str, report: str = REPORT, *,
            think: str | None = "thinking 一二三", task: str = TASK,
            finish_reason: str = "stop", im_end: bool = True) -> dict[str, Any]:
    """A committed ``RequestRecord``-shaped mapping the capture pass accepts as-is."""
    messages = [{"role": "user", "content": task}]
    prompt = [int(t) for t in tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)]
    completion = support.completion_from_text(tokenizer, think, report, im_end=im_end)
    return support.synthetic_record(prompt, completion, messages=messages, content=report,
                                    reasoning=think, finish_reason=finish_reason,
                                    request_id=request_id)


def _output(tokenizer, tokens: K.TokenContract, report: str, *,
            finish_reason: str = "stop") -> SC.PublicOutput:
    ids = tokenizer.encode(report, add_special_tokens=False)
    return SC.parse_public_output(report, finish_reason, ids,
                                  lambda text: len(tokenizer.encode(text)), limits=tokens)


def _decision_row(decision_id: str, out: SC.PublicOutput, *, request_ids: tuple[str, ...],
                  generation_uid: str, round_: int = 0, slot: int = 0,
                  source_item_id: str = "item0001", protocol_hash: str = "cond-peer",
                  episode_id: str = "episode|t|c|m|b|rep0", cell_id: str = "episode.MATH.b0000",
                  is_reference: bool = False) -> T.DecisionRow:
    """The ``decisions`` row a generation cell wrote, carrying the parser's byte spans.

    Field-for-field what ``telemetry.decision_row`` produces (R6 keeps the *producer* single;
    a fixture may still build the frozen dataclass, as ``test_states.py`` does).
    """
    return T.DecisionRow(
        decision_id=decision_id, study_id="task_conditioned_dynamics_v1",
        cell_kind=CellKind.EPISODE.value, source_item_id=source_item_id,
        item_variant_id=f"{source_item_id}.v0", root_team_id="team:abc", protocol_hash=protocol_hash,
        receiver_model_hash=MODEL_HASH, budget_hash="budget08", replicate_id=0, round=round_,
        agent_slot=slot, phase="root_ref" if round_ == 0 else "revision", episode_id=episode_id,
        cell_id=cell_id, display_label=None, generation_uid=generation_uid,
        content_hash=out.content_hash, request_hashes_json=json.dumps(list(request_ids)),
        aliased=False, reuse_of=None, is_reference=is_reference, first_decision_id=None,
        bundle_export_hash=None, parent_decision_ids_json="[]", seen_transmission_ids_json="[]",
        native_reasoning_ref=None, public_reasoning=out.reasoning,
        shareable_claims_json=json.dumps(list(out.shareable_claims)),
        final_answer_raw=out.final_answer, canonical_class=None, confidence=out.confidence,
        answer_logprob_sum=None, answer_logprob_mean=None, answer_token_count=None,
        logprob_norm_scheme=None, parse_status=out.parse_status.value,
        field_order_violation=out.order_violation, validity=out.valid,
        delivery_success=out.valid, infrastructure_status="ok", termination_reason="stop",
        boundary_injected=False, continuation_used=len(request_ids) > 1, discarded_span_tokens=0,
        thinking_tokens=1, final_tokens=1, prompt_tokens=1, cached_prefill_tokens=0,
        token_offsets_json=json.dumps({
            "final_answer_span": list(out.final_answer_span) if out.final_answer_span else None,
            "confidence_span": list(out.confidence_span) if out.confidence_span else None,
            "confidence_emitted_before_answer": bool(out.confidence_emitted_before_answer)}),
        endpoint_id=None, latency_s=None, flops_prefill=None, flops_decode=None,
        flops_total=None, linear_compute_proxy=None)


def _write_cell(run_root: Path, cell_id: str, source_item_id: str,
                rows: tuple[T.DecisionRow, ...]) -> Path:
    item = T.ItemFile(schema_version=T.SCHEMA_VERSION, cell_id=cell_id, cell_kind=CellKind.EPISODE,
                      source_item_id=source_item_id, status=T.ItemStatus.COMPLETE,
                      bundle_export_hash=None, rows={"decisions": rows})
    path = T.item_path(run_root, cell_id, source_item_id)
    item.write(path)
    return path


def _team(tokenizer, tokens, run_root: Path, *, item: str = "item0001",
          reports: tuple[str, ...] = (REPORT,) * 5, cell_id: str = "episode.MATH.b0000",
          ) -> tuple[StubStore, tuple[T.DecisionRow, ...]]:
    """One five-agent root team: five decisions, five records, one generation cell."""
    store, rows = StubStore(), []
    for slot, report in enumerate(reports):
        rid = f"r{slot}"
        store.add(rid, _record(tokenizer, rid, report, task=f"{TASK} (slot {slot})"))
        rows.append(_decision_row(f"d:root:{slot}", _output(tokenizer, tokens, report),
                                  request_ids=(rid,), generation_uid=f"g:{slot}", slot=slot,
                                  source_item_id=item, cell_id=cell_id, is_reference=True))
    _write_cell(run_root, cell_id, item, tuple(rows))
    return store, tuple(rows)


def _panel(run_root: Path, items: tuple[str, ...], **kwargs: Any) -> str:
    return deps.write_register(run_root, deps.REGISTER_PANEL, CAP.panel_payload(items, **kwargs))


# =============================================================================== anchors


def test_activation_layer_and_anchor_offsets_valid(tokenizer, tokens):
    """Spec 13.2: blocks are ``floor(f*L)-1`` at the real ``L``; anchors are real byte spans."""
    fracs = (0.25, 0.50, 0.75)
    assert CAP.capture_blocks(LAYERS_8B, fracs) == (8, 17, 26)
    assert CAP.capture_blocks(LAYERS_32B, fracs) == (15, 31, 47)
    # L is read from the pin, never assumed
    assert CAP.model_layer_count(_lock_entry(LAYERS_32B)) == LAYERS_32B
    assert CAP.blocks_for_model(_lock_entry(LAYERS_8B), fracs) == (8, 17, 26)
    with pytest.raises(ContractViolation, match="layer count"):
        CAP.model_layer_count(_lock_entry(None))
    with pytest.raises(ContractViolation, match="outside"):
        CAP.capture_blocks(2, fracs)                 # the tiny fixture has no quarter depth

    record = _record(tokenizer, "r0")
    out = _output(tokenizer, tokens, REPORT)
    anchors = CAP.resolve_anchors(record, out, tokenizer, decision_id="d:1")
    prompt_len = len(record["prompt_token_ids"])
    sequence = list(record["prompt_token_ids"]) + list(record["response"]["token_ids"])

    assert [a.name for a in anchors.anchors] == list(CAP.ANCHORS)
    assert all(a.present for a in anchors.anchors)
    prompt_end, pre_final, answer_end = anchors.anchors
    assert prompt_end.token_offset == prompt_len - 1
    assert prompt_len <= pre_final.token_offset < answer_end.token_offset < len(sequence)

    # every offset is derived from the span the parser recorded, not from a constant
    body = REPORT.encode("utf-8")
    start, stop = out.final_answer_span
    assert body[start:stop] == b'"42"'
    assert pre_final.byte_offset == start and answer_end.byte_offset == stop
    assert pre_final.detail["token_byte_span"][0] < start <= pre_final.detail["token_byte_span"][1]
    assert answer_end.detail["token_byte_span"][0] < stop <= answer_end.detail["token_byte_span"][1]
    # PRE_FINAL's token stops at the opening quote: its residual predicts the answer's first
    # token and holds no answer character.  ANSWER_END's covers the value's last byte.  Both
    # record that a delimiter and the value share one token rather than shifting silently.
    assert pre_final.detail["token_byte_span"][1] <= start + 1
    assert b"4" not in body[pre_final.detail["token_byte_span"][0]:pre_final.detail["token_byte_span"][1]]
    assert answer_end.detail["token_byte_span"][1] >= stop
    assert body[:stop].endswith(b'"42"')
    assert pre_final.detail["shared_token"] and answer_end.detail["shared_token"]
    # the confidence value is strictly after ANSWER_END (spec 13.2 "before the public confidence")
    assert out.confidence_span[0] >= stop


def _lock_entry(n_layers: int | None):
    from agents_scaling.nextdyn.locks import ModelLockEntry

    return ModelLockEntry(status="locked", blocked_reason=None, hf_id="Qwen/Qwen3-8B",
                          revision="rev", tokenizer_revision="rev", adapter="native_thinking",
                          config_sha256=None, chat_template_sha256=None, safetensors=None,
                          n_layers=n_layers, dtype=None, reasoning_parser=None)


def test_the_anchor_is_never_a_fixed_offset_from_either_end_of_the_sequence(tokenizer, tokens):
    """The anchor follows the span, so no constant reproduces it from either direction."""

    def anchor_of(report: str) -> tuple[int, int, int]:
        record, out = _record(tokenizer, "r", report), _output(tokenizer, tokens, report)
        anchors = CAP.resolve_anchors(record, out, tokenizer)
        offset = anchors.by_name(CAP.ANCHOR_ANSWER_END).token_offset
        return offset, offset - anchors.public_span[0], anchors.sequence_tokens - offset

    # more rationale before the answer moves it away from the start of the public span
    longer_prefix = REPORT.replace("checked twice", "checked twice and then a third time over")
    assert anchor_of(REPORT)[1] < anchor_of(longer_prefix)[1]
    # more text after the answer moves it away from the end
    longer_suffix = REPORT.replace('"confidence": 0.87', '"confidence": 0.8712345678901234')
    assert anchor_of(REPORT)[2] < anchor_of(longer_suffix)[2]
    # and the field order alone changes where it lands
    assert anchor_of(REPORT)[1] != anchor_of(REPORT_CONF_FIRST)[1]


def test_answer_end_is_pre_confidence_only_when_confidence_not_yet_emitted(tokenizer, tokens):
    """Spec 13.2/6.1 (the capture half): eligibility is a recorded label, not a filter.

    Both orders resolve an ``ANSWER_END`` anchor and both get a vector -- the state exists
    either way.  What changes is whether the analysis may treat it as a *pre-confidence*
    state, and that fact travels on the row rather than being re-derived downstream.
    """
    declared = CAP.resolve_anchors(_record(tokenizer, "r", REPORT),
                                   _output(tokenizer, tokens, REPORT), tokenizer)
    early = CAP.resolve_anchors(_record(tokenizer, "r", REPORT_CONF_FIRST),
                                _output(tokenizer, tokens, REPORT_CONF_FIRST), tokenizer)

    assert declared.by_name(CAP.ANCHOR_ANSWER_END).pre_confidence_eligible is True
    assert early.by_name(CAP.ANCHOR_ANSWER_END).pre_confidence_eligible is False
    assert early.by_name(CAP.ANCHOR_ANSWER_END).present        # still captured, just not eligible
    # the other two anchors never claim the property at all
    for anchors in (declared, early):
        for name in (CAP.ANCHOR_PROMPT_END, CAP.ANCHOR_PRE_FINAL):
            assert anchors.by_name(name).pre_confidence_eligible is None
    # and the confidence really did precede the answer in the early case
    early_out = _output(tokenizer, tokens, REPORT_CONF_FIRST)
    assert early_out.confidence_emitted_before_answer
    assert early_out.confidence_span[1] < early_out.final_answer_span[0]


def test_a_missing_anchor_is_typed_and_the_substitute_is_never_chosen_by_correctness(tokenizer, tokens):
    """An invalid answer, a truncated span and an unclosed thinking channel each get a reason."""
    invalid = CAP.resolve_anchors(_record(tokenizer, "r", REPORT_INVALID),
                                  _output(tokenizer, tokens, REPORT_INVALID), tokenizer)
    assert invalid.by_name(CAP.ANCHOR_PROMPT_END).present      # the prompt is still there
    for name in (CAP.ANCHOR_PRE_FINAL, CAP.ANCHOR_ANSWER_END):
        anchor = invalid.by_name(name)
        assert not anchor.present and anchor.missing_reason == CAP.MISSING_INVALID_ANSWER
        assert anchor.token_offset is None

    # a span that points past the stored bytes (a cut completion) is SPAN_TRUNCATED, not a
    # shifted anchor: nothing looks for the "nearest" token instead
    out = _output(tokenizer, tokens, REPORT)
    beyond = CAP.AnswerSpans(final_answer_span=(len(REPORT.encode()) + 500, len(REPORT.encode()) + 520),
                             confidence_span=None, confidence_emitted_before_answer=False,
                             parse_status="ok")
    cut = CAP.resolve_anchors(_record(tokenizer, "r", REPORT), beyond, tokenizer)
    assert {a.missing_reason for a in cut.missing} == {CAP.MISSING_SPAN_TRUNCATED}

    # no closing delimiter at all: there is no public span to anchor in
    unclosed = _record(tokenizer, "r", REPORT)
    unclosed["response"]["token_ids"] = list(
        tokenizer.encode("<think>still thinking", add_special_tokens=False))
    none_ = CAP.resolve_anchors(unclosed, out, tokenizer)
    assert {a.missing_reason for a in none_.missing} == {CAP.MISSING_NO_PUBLIC_SPAN}

    # a decision that never reached vLLM has no ids and therefore no anchor at all
    empty = dict(unclosed, prompt_token_ids=[])
    refused = CAP.resolve_anchors(empty, out, tokenizer)
    assert {a.missing_reason for a in refused.anchors} == {CAP.MISSING_CONTEXT_FAILURE}
    assert all(a.token_offset is None for a in refused.anchors)


def _continued(tokenizer, tokens, *, task: str = TASK, request_ids=("phase1", "phase2")):
    """The two records a continued draw really stores (amendment A3; ``continuation.py``).

    Phase 1 hits the thinking cap, so its report is never written; phase 2 is a
    ``/v1/completions`` call whose prompt is an exact token prefix -- phase 1's prompt, the
    thinking span through ``</think>``, then the frozen prefill -- and which therefore carries
    ``messages=()`` (``ContinuationSpec`` refuses messages) and ``enable_thinking=False``
    (no chat template is rendered at all).  The decision's final text is
    ``decode(prefill + kept)``, exactly as ``continuation.merged_decision`` builds it.
    """
    first = _record(tokenizer, request_ids[0], REPORT, task=task)
    prompt = list(first["prompt_token_ids"])
    think = tokenizer.encode("<think>\nstill deriving\n</think>", add_special_tokens=False)
    first["response"]["token_ids"] = list(think)
    first["response"]["content"] = ""
    first["response"]["finish_reason"] = "length"
    prefill = tokenizer.encode(PREFILL_TEXT, add_special_tokens=False)
    tail = ('checked twice", "shareable_claims": ["one claim"], '
            '"final_answer": "42", "confidence": 0.87}')
    generated = tokenizer.encode(tail, add_special_tokens=False)
    final_text = tokenizer.decode(list(prefill) + list(generated))
    second = support.synthetic_record(
        prompt + list(think) + list(prefill),
        list(generated) + [tokenizer.convert_tokens_to_ids("<|im_end|>")],
        messages=[], enable_thinking=False, content=tail, reasoning=None,
        request_id=request_ids[1])
    out = _output(tokenizer, tokens, final_text)
    return first, second, out, len(prompt), len(prompt) + len(think)


def test_the_two_physical_requests_of_a_continued_draw_compose_into_one_sequence(tokenizer, tokens):
    """Amendment A3: the prompt anchor stays the decision's own prompt, not the prefill's end.

    The channel is located structurally, so the phase-2 record's ``enable_thinking=False`` --
    what ``decoding.continuation_decoding`` really stores -- cannot move the origin: the
    thinking span lives inside phase 2's *prompt*, and an anchor measured from there would
    land in the reasoning text with a vector written and no missing reason.
    """
    first, second, out, prompt_len, public_start = _continued(tokenizer, tokens)
    assert second["chat_template_kwargs"] == {"enable_thinking": False}
    assert second["messages"] == []

    work = CAP.CaptureWorkItem(
        generation_uid="g:cont", decision_ids=("d:cont",), request_ids=("phase1", "phase2"),
        source_item_id="item0001", item_variant_id="item0001.v0", episode_ids=("e",),
        cell_ids=("c",), cell_kind="episode", protocol_hash="cond", model_hash=MODEL_HASH,
        round_=1, agent_slot=0, is_reference=False,
        spans=CAP.AnswerSpans.from_output(out))
    store = StubStore({"phase1": first, "phase2": second})
    source = CAP._records_for(work, store)

    assert source.record is second and source.own_record is first
    assert source.prompt_tokens == prompt_len and source.continued
    anchors = CAP.resolve_anchors(source.record, work.spans, tokenizer,
                                  prompt_tokens=source.prompt_tokens)
    assert anchors.by_name(CAP.ANCHOR_PROMPT_END).token_offset == prompt_len - 1
    assert all(a.present for a in anchors.anchors)
    # the public span begins after the thinking close, i.e. at the frozen prefill
    assert anchors.public_span[0] == public_start
    assert anchors.origin_proof == "content_hash" and anchors.public_text_offset == 0

    # the two content anchors sit on the answer's bytes, not inside the reasoning text
    sequence = list(second["prompt_token_ids"]) + list(second["response"]["token_ids"])
    pre_final, answer_end = (anchors.by_name(n) for n in (CAP.ANCHOR_PRE_FINAL, CAP.ANCHOR_ANSWER_END))
    assert tokenizer.decode(sequence[pre_final.token_offset]).endswith('"')
    assert "42" in tokenizer.decode(sequence[pre_final.token_offset: answer_end.token_offset + 1])
    assert "still deriving" not in tokenizer.decode(
        sequence[pre_final.token_offset - 1: answer_end.token_offset + 1])


def test_a_continued_draw_is_identity_checked_against_its_own_phase_one_record(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    """The identity check re-renders a chat template, and only phase 1 has one.

    ``check_prompt_identity`` on the phase-2 record cannot pass by construction (``messages``
    is empty, and its prompt is a token prefix no render reproduces), so a pass that checked
    the replayed record would kill every CAPTURE cell holding a continued decision -- about a
    third of them at ARD36's measured boundary rate.  The phase-2 evidence is the prefix
    relation instead, and both hold before a row claims ``replay_parity == 'ok'``.
    """
    first, second, out, prompt_len, _ = _continued(tokenizer, tokens)
    store = StubStore({"phase1": first, "phase2": second})
    row = _decision_row("d:cont", out, request_ids=("phase1", "phase2"), generation_uid="g:cont",
                        round_=1, slot=0)
    _write_cell(tmp_run_root, "episode.MATH.b0000", "item0001", (row,))
    items = CAP.capture_manifest((row,), None)

    manifest_rows, stats = CAP.run_capture(
        items, engine=tiny_engine, tokenizer=tokenizer, store=store, run_root=tmp_run_root,
        config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_cont"))

    assert stats["captured"] == 1 and stats["missing_records"] == 0 and stats["anchor_errors"] == 0
    assert len(manifest_rows) == len(CAP.ANCHORS) * len(TINY_BLOCKS)
    assert {r.replay_parity for r in manifest_rows} == {CAP.PARITY_OK}
    assert all(r.missing_reason is None and r.storage_hash for r in manifest_rows)
    # PROMPT_END is phase 1's last prompt token; both content anchors are past the prefill
    by_anchor = {r.anchor: r for r in manifest_rows if r.layer == TINY_BLOCKS[0]}
    assert by_anchor[CAP.ANCHOR_PROMPT_END].token_offset == prompt_len - 1
    assert by_anchor[CAP.ANCHOR_PRE_FINAL].token_offset > len(second["prompt_token_ids"]) - 1 - len(
        tokenizer.encode(PREFILL_TEXT, add_special_tokens=False))


def test_an_anchor_whose_origin_cannot_be_proved_is_missing_rather_than_placed(tokenizer, tokens):
    """The origin of a byte span is proved, never assumed (spec 13.2 correctness-blindness).

    A record whose stored ids do not carry the text the parser measured -- here the public
    channel holds a different report -- yields no offset at all: a typed missing anchor beats
    a vector taken from whatever those bytes happen to be.
    """
    record = _record(tokenizer, "r", REPORT)
    other = _output(tokenizer, tokens, REPORT.replace('"42"', '"1007"'))
    spans = CAP.AnswerSpans(final_answer_span=(0, 4), confidence_span=None,
                            confidence_emitted_before_answer=False, parse_status="ok",
                            content_hash=other.content_hash, final_answer="1007")
    anchors = CAP.resolve_anchors(record, spans, tokenizer)

    assert anchors.by_name(CAP.ANCHOR_PROMPT_END).present
    assert {a.missing_reason for a in anchors.missing} == {CAP.MISSING_ANCHOR_UNRESOLVED}
    assert anchors.public_text_offset is None and anchors.origin_proof == ""
    # the same spans against the record that really produced them resolve
    good = CAP.resolve_anchors(record, _output(tokenizer, tokens, REPORT), tokenizer)
    assert all(a.present for a in good.anchors)


def test_a_non_thinking_companion_has_no_think_close_and_still_anchors(tokenizer, tokens):
    """Spec 5.6: the ``total_output_envelope`` companion emits no ``</think>`` at all.

    Its whole completion is the public span, and the channel is settled by the content hash
    rather than by a flag defaulted to "thinking was on" -- which would look for a delimiter
    that never exists and lose every content anchor silently.
    """
    prompt = [int(t) for t in tokenizer.apply_chat_template(
        [{"role": "user", "content": TASK}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False)]
    completion = support.completion_from_text(tokenizer, None, REPORT)
    record = support.synthetic_record(prompt, completion, messages=[{"role": "user", "content": TASK}],
                                      content=REPORT, enable_thinking=False, request_id="env")
    anchors = CAP.resolve_anchors(record, _output(tokenizer, tokens, REPORT), tokenizer)

    assert all(a.present for a in anchors.anchors)
    assert anchors.public_span == (len(prompt), len(prompt) + len(completion))
    assert anchors.origin_proof == "content_hash"
    body = REPORT.encode()
    start, stop = _output(tokenizer, tokens, REPORT).final_answer_span
    assert body[start:stop] == b'"42"'
    assert anchors.by_name(CAP.ANCHOR_ANSWER_END).byte_offset == stop


# =============================================================================== replay


def test_teacher_forced_replay_uses_exact_stored_token_ids(tokenizer, tokens, tiny_engine, tmp_run_root):
    """Spec 13.2: the replay prefills the stored ids, unchanged, and nothing else."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    items = CAP.capture_manifest(rows, None)
    engine = RecordingEngine(tiny_engine)
    config = CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test")

    manifest_rows, stats = CAP.run_capture(items, engine=engine, tokenizer=tokenizer,
                                           store=store, run_root=tmp_run_root, config=config)

    assert stats["captured"] == len(rows) == 5 and stats["missing_records"] == 0
    # the replay forwards are the capture ones; the two extra are the spec-13.2 tolerance
    # report, the only thing in the pass that asks for a vocabulary row at all
    replayed = [r for r in engine.requests if not r.logit_positions]
    parity = [r for r in engine.requests if r.logit_positions]
    assert len(replayed) == 5
    assert len(parity) == 2 and parity[0] == parity[1]     # one prefix, forwarded twice
    for request, work in zip(replayed, items):
        record = store.get(work.request_ids[0])
        stored = list(record["prompt_token_ids"]) + list(record["response"]["token_ids"])
        assert list(request.token_ids) == stored[: len(request.token_ids)]
        assert request.token_ids == tuple(stored[: len(request.token_ids)])
        # the prefix stops exactly at the last anchor: no token is invented and none past it
        # is paid for
        assert max(request.positions) == len(request.token_ids) - 1
        assert request.seq_id == work.generation_uid
        assert request.logit_positions == ()          # read-only: no vocabulary row is asked for

    # every row is 3 anchors x 2 blocks, and the vectors that landed are the model's
    assert len(manifest_rows) == 5 * len(CAP.ANCHORS) * len(TINY_BLOCKS)
    assert {r.layer for r in manifest_rows} == set(TINY_BLOCKS)
    assert {r.anchor for r in manifest_rows} == set(CAP.ANCHORS)
    assert all(r.replay_parity == CAP.PARITY_OK for r in manifest_rows)
    assert all(r.dtype == CAP.VECTOR_DTYPE and r.storage_hash for r in manifest_rows)
    assert all(json.loads(r.shape_json) == [tiny_engine.hidden_size] for r in manifest_rows)

    # spec 13.2 asks for logged tolerances, not bit-identical kernels: the pass reports the
    # argmax agreement, the mean log-prob of the stored tokens and the two-pass logit spread
    report = stats["fidelity"]
    assert 0.0 <= report["argmax_agreement"] <= 1.0 and report["mean_logprob"] <= 0.0
    assert report["max_abs_logit_diff"] == pytest.approx(0.0, abs=1e-3)
    assert report["scored_tokens"] > 0 and "error" not in report
    assert report["identity"]["decode_matches_render"] and report["identity"]["retokenize_matches"]
    # and they travel with the vectors, so a gate reads them off the ledger
    stored_rows = _chunk(tmp_run_root)[2]
    carried = [r for r in stored_rows if (r["extra"] or {}).get("fidelity")]
    assert carried and carried[0]["extra"]["fidelity"]["hook_convention"] == HOOK_CONVENTION
    assert all(r["extra"]["origin_proof"] == "content_hash" for r in stored_rows
               if r["extra"].get("public_text_offset") is not None)


def test_a_missing_anchor_carries_no_vector_and_is_never_a_zero_vector(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    """Spec 13.2: a missing capture is a ledger row with a reason, not a zeroed state."""
    store, rows = _team(tokenizer, tokens, tmp_run_root,
                        reports=(REPORT, REPORT, REPORT_INVALID, REPORT, REPORT))
    items = CAP.capture_manifest(rows, None)
    manifest_rows, stats = CAP.run_capture(
        items, engine=tiny_engine, tokenizer=tokenizer, store=store, run_root=tmp_run_root,
        config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test"))

    missing = [r for r in manifest_rows if r.missing_reason is not None]
    assert {r.anchor for r in missing} == {CAP.ANCHOR_PRE_FINAL, CAP.ANCHOR_ANSWER_END}
    assert len(missing) == 2 * len(TINY_BLOCKS)          # the one invalid answer, both blocks
    assert {r.missing_reason for r in missing} == {CAP.MISSING_INVALID_ANSWER}
    for row in missing:
        assert row.token_offset is None and row.storage_hash is None
        assert row.dtype is None and row.shape_json is None
        assert row.replay_parity == CAP.PARITY_MISSING
    assert stats["anchors_missing"] == {CAP.MISSING_INVALID_ANSWER: 2}

    # and on disk: the chunk holds no vector row for those keys, and no zero vector anywhere
    vectors, keys, stored = _chunk(tmp_run_root)
    by_key = {str(r["StateSnapshot_id"]) + "|" + str(r["anchor_kind"]): r for r in stored}
    absent = by_key["g:2|" + CAP.ANCHOR_ANSWER_END]
    assert absent["vector_index"] == -1 and absent["missingness"] == CAP.MISSING_INVALID_ANSWER
    assert absent["tensor_hash"] is None
    assert vectors.shape[0] == len(keys) < len(stored)
    assert not np.any(np.all(vectors == 0, axis=1)), "a stored vector is all zeros"


def _chunk(run_root: Path):
    chunks = S.list_chunks(S.stage_dir(run_root, CAP.CAPTURE_STAGE))
    assert chunks, "no committed chunk"
    vectors, keys, rows = S.read_chunk(chunks[0][2], chunks[0][3])
    return vectors, keys, rows


def test_the_npz_is_committed_before_its_jsonl_marker(tokenizer, tokens, tiny_engine, tmp_run_root):
    """Consolidation order: a crash leaves an orphan ``.npz``, never a chunk with no vectors."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    CAP.run_capture(CAP.capture_manifest(rows, None), engine=tiny_engine, tokenizer=tokenizer,
                    store=store, run_root=tmp_run_root,
                    config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test"))
    for _shard, _index, npz, jsonl in S.list_chunks(S.stage_dir(tmp_run_root, CAP.CAPTURE_STAGE)):
        assert npz.is_file() and jsonl.is_file()
        assert npz.stat().st_mtime <= jsonl.stat().st_mtime


def test_a_second_pass_resumes_by_committed_key_and_still_accounts_for_every_sample(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    items = CAP.capture_manifest(rows, None)
    config = CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test")
    first, _ = CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                               run_root=tmp_run_root, config=config)
    again, stats = CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                                   run_root=tmp_run_root, config=config)

    assert stats["pending"] == 0 and stats["captured"] == 0
    assert CAP.assert_capture_progress(stats) is stats        # a full resume is not a failure
    assert len(again) == len(first)                           # the ledger still covers every row
    assert {r.replay_parity for r in again} == {CAP.PARITY_RESUMED}
    assert {(r.decision_id, r.layer, r.anchor) for r in again} == \
           {(r.decision_id, r.layer, r.anchor) for r in first}
    assert len(_chunk(tmp_run_root)[2]) == len(first)         # nothing was written twice


def test_the_committed_ledger_can_be_read_once_and_handed_to_every_item_of_a_cell(
        tokenizer, tokens, tiny_engine, tmp_run_root, monkeypatch):
    """The resume filter is the whole ``native`` stage, and that stage is shared by every
    capture cell of the run while the worker contract is *per item*: a cell that replays
    sixteen items would re-parse the campaign's ledger sixteen times.  A caller that drives
    more than one item reads it once and hands it to each pass."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    items = CAP.capture_manifest(rows, None)
    config = CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test")
    first, _ = CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                               run_root=tmp_run_root, config=config)

    committed = CAP.read_committed(tmp_run_root)
    assert {key[0] for key in committed} == {w.generation_uid for w in items}
    assert len(committed) == len(items) * len(CAP.ANCHORS) * len(TINY_BLOCKS)
    # and it narrows to the samples a pass can ask about, so a late cell's filter stays small
    one = CAP.read_committed(tmp_run_root, uids=(items[0].generation_uid,))
    assert {key[0] for key in one} == {items[0].generation_uid}

    reads: list[Any] = []
    real = S.iter_rows
    monkeypatch.setattr(S, "iter_rows", lambda *a, **k: (reads.append(a), real(*a, **k))[1])
    again, stats = CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                                   run_root=tmp_run_root, config=config, committed=committed)

    assert reads == []                                   # the pass re-parsed nothing
    assert stats["pending"] == 0 and stats["captured"] == 0
    assert {r.replay_parity for r in again} == {CAP.PARITY_RESUMED}
    assert {(r.decision_id, r.layer, r.anchor) for r in again} == \
           {(r.decision_id, r.layer, r.anchor) for r in first}


def test_a_cell_that_captures_nothing_it_was_asked_for_fails_loudly():
    with pytest.raises(CAP.CaptureError, match="must fail"):
        CAP.assert_capture_progress({"work_items": 3, "pending": 3, "captured": 0}, cell_id="x")


# =============================================================================== the panel


def test_the_work_list_is_the_panel_register_and_one_entry_per_physical_sample(
        tokenizer, tokens, tmp_run_root):
    """Root captures are reused across arms: one replay, one ledger row per decision."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    # the same five roots referenced again by a second arm: different decision ids, same uids
    peer = tuple(
        _decision_row(f"d:peer:{i}", _output(tokenizer, tokens, REPORT), request_ids=(f"r{i}",),
                      generation_uid=f"g:{i}", slot=i, protocol_hash="cond-private",
                      episode_id="episode|t|p|m|b|rep0", cell_id="episode.MATH.b0001",
                      is_reference=True)
        for i in range(5))
    _write_cell(tmp_run_root, "episode.MATH.b0001", "item0001", peer)

    items = CAP.capture_manifest(rows + peer, None)
    assert len(items) == 5                                     # five samples, ten decisions
    assert sum(len(w.decision_ids) for w in items) == 10
    assert all(len(set(w.decision_ids)) == len(w.decision_ids) for w in items)
    assert [w.order_key[1:3] for w in items] == [(0, s) for s in range(5)]

    # the sealed panel decides which items and arms are in scope, before any outcome
    sha = _panel(tmp_run_root, ("item0001",), protocol_hashes=("cond-peer",))
    panel = CAP.load_panel(tmp_run_root, sha)
    narrowed = CAP.capture_manifest(rows + peer, panel)
    assert {d for w in narrowed for d in w.decision_ids} == {r.decision_id for r in rows}
    assert CAP.capture_manifest(rows + peer, CAP.load_panel(
        tmp_run_root, _panel(tmp_run_root, ("other-item",)))) == ()

    with pytest.raises(CAP.CaptureError, match="empty capture panel"):
        CAP.panel_payload(())
    with pytest.raises(DependencyUnmet):
        CAP.load_panel(tmp_run_root, "deadbeef")


def test_one_pass_replays_one_model_and_one_sample_has_one_public_output(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    """Spec 13.2/plan D14: two refusals that would otherwise corrupt a root measurement."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    items = CAP.capture_manifest(rows, None)
    with pytest.raises(CAP.CaptureError, match="foreign-model"):
        CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                        run_root=tmp_run_root,
                        config=CAP.CaptureConfig(model_hash="fedcba9876543210",
                                                 shard_name="nextdyn_test"))

    # two decisions naming one generation_uid must agree about what that sample said
    clash = _decision_row("d:clash", _output(tokenizer, tokens, REPORT_CONF_FIRST),
                          request_ids=("r0",), generation_uid="g:0", slot=0)
    with pytest.raises(CAP.CaptureError, match="different answer spans"):
        CAP.capture_manifest(rows + (clash,), None)


def test_all_root_agent_captures_exist_or_are_explicitly_missing(
        tokenizer, tokens, tiny_engine, tmp_run_root, study_cfg):
    """Spec 13.2: every root member of the panel is captured, or says why it is not."""
    store, rows = _team(tokenizer, tokens, tmp_run_root,
                        reports=(REPORT, REPORT, REPORT_INVALID, REPORT, REPORT))
    sha = _panel(tmp_run_root, ("item0001",))
    runner = _worker_runner(tmp_run_root, study_cfg,
                            {"cells": ("episode.MATH.b0000",), "panel": sha, "blocks": TINY_BLOCKS},
                            store=store, engine=tiny_engine, tokenizer=tokenizer)

    item = CAP.capture_worker(runner, "item0001")
    assert item.cell_kind is CellKind.CAPTURE and item.status is T.ItemStatus.COMPLETE
    manifest = item.rows["activations_manifest"]

    # every (root decision, block, anchor) exists exactly once -- nothing is silently dropped
    keys = [(r.decision_id, r.layer, r.anchor) for r in manifest]
    assert len(keys) == len(set(keys)) == 5 * len(TINY_BLOCKS) * len(CAP.ANCHORS)
    assert {r.decision_id for r in manifest} == {r.decision_id for r in rows}
    for row in manifest:
        assert (row.storage_hash is None) == (row.missing_reason is not None)
        assert row.model_hash == MODEL_HASH
    # the one invalid answer is present-with-a-reason, never absent
    invalid = [r for r in manifest if r.decision_id == "d:root:2"]
    assert len(invalid) == len(TINY_BLOCKS) * len(CAP.ANCHORS)
    assert {r.missing_reason for r in invalid} == {None, CAP.MISSING_INVALID_ANSWER}

    census = CAP.root_agent_coverage(CAP.capture_manifest(rows, None), manifest,
                                     capture_all_root_agents=True)
    assert census["root_decisions"] == census["root_decisions_with_rows"] == 5
    assert census["root_decisions_without_any_row"] == []
    assert study_cfg.instrumentation.capture_all_root_agents is True

    # drop one root's rows and the census refuses: a partial cohort never looks complete
    partial = tuple(r for r in manifest if r.decision_id != "d:root:4")
    with pytest.raises(CAP.CaptureError, match="no activation row"):
        CAP.root_agent_coverage(CAP.capture_manifest(rows, None), partial,
                                capture_all_root_agents=True)


def test_the_worker_separates_an_unusable_options_block_from_an_item_with_no_decisions(
        tokenizer, tokens, tiny_engine, tmp_run_root, study_cfg):
    store, _rows = _team(tokenizer, tokens, tmp_run_root)
    engine_kwargs = dict(store=store, engine=tiny_engine, tokenizer=tokenizer)

    with pytest.raises(CAP.CaptureError, match="no 'cells'"):
        CAP.capture_worker(FakeRunner(tmp_run_root, study_cfg, {}, **engine_kwargs), "item0001")
    with pytest.raises(CAP.CaptureError, match="items/ directory"):
        CAP.capture_worker(FakeRunner(tmp_run_root, study_cfg, {"cells": ("episode.NOPE",)},
                                      **engine_kwargs), "item0001")

    # the cells are there and were read; this item simply has none -- a complete, empty file
    runner = _worker_runner(tmp_run_root, study_cfg, {"cells": ("episode.MATH.b0000",)},
                            **engine_kwargs)
    item = CAP.capture_worker(runner, "item-not-in-this-cell")
    assert item.status is T.ItemStatus.COMPLETE and item.rows["activations_manifest"] == ()


def test_the_worker_derives_its_blocks_and_holds_the_engine_to_them(
        tokenizer, tokens, tiny_engine, tmp_run_root, study_cfg):
    """Spec 13.2: the hooks sit at ``floor(f*L)-1`` of the **pinned** model, always derived.

    On the production path the runner hands over an engine it built, so a worker that took
    ``engine.blocks`` (or an ``options['blocks']`` literal) on trust would never enforce the
    rule at the real layer count -- the derivation would be dead code exactly where it matters.
    """
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    kwargs = dict(store=store, engine=tiny_engine, tokenizer=tokenizer)

    # no 'blocks' option at all: the layer count in models.lock.json is what decides
    runner = _worker_runner(tmp_run_root, study_cfg, {"cells": ("episode.MATH.b0000",)}, **kwargs)
    manifest = CAP.capture_worker(runner, "item0001").rows["activations_manifest"]
    assert {r.layer for r in manifest} == set(TINY_BLOCKS)
    assert CAP.blocks_for_model(_lock_entry(2), (0.5, 1.0)) == TINY_BLOCKS

    # an options literal that disagrees with the derivation is refused, not preferred
    wrong = _worker_runner(tmp_run_root, study_cfg,
                           {"cells": ("episode.MATH.b0000",), "blocks": (0,)}, **kwargs)
    with pytest.raises(ContractViolation, match="floor"):
        CAP.capture_worker(wrong, "item0001")

    # and so is an engine whose hooks are somewhere else
    class OtherBlocks(RecordingEngine):
        blocks = (0,)

    mismatched = _worker_runner(tmp_run_root, study_cfg, {"cells": ("episode.MATH.b0000",)},
                                store=store, engine=OtherBlocks(tiny_engine), tokenizer=tokenizer)
    with pytest.raises(ContractViolation, match="hooks blocks"):
        CAP.capture_worker(mismatched, "item0001")

    # a model whose depth nothing pins is a refusal, never a guess
    unpinned = FakeRunner(tmp_run_root, _tiny_cfg(study_cfg), {"cells": ("episode.MATH.b0000",)},
                          models_lock=None, **kwargs)
    with pytest.raises(ContractViolation, match="layer count"):
        CAP.capture_worker(unpinned, "item0001")


def test_a_sample_with_no_request_or_no_stored_record_is_typed_missingness(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    """Spec 13.2: "if an answer is invalid or the anchor does not exist, mark the capture
    missing with a reason" -- including when the reason is that there is nothing to replay.

    A build-time context refusal has no physical request at all and a store that lost a record
    has no ids; both are ledger rows with a reason and no vector.  Killing the cell instead
    would throw away every other sample it was asked for, and ``assert_capture_progress``
    already refuses a cell that captured *nothing*.
    """
    store, rows = _team(tokenizer, tokens, tmp_run_root, reports=(REPORT,))
    good = CAP.capture_manifest(rows, None)
    refused = CAP.CaptureWorkItem(
        generation_uid="g:refused", decision_ids=("d:refused",), request_ids=(),
        source_item_id="item0001", item_variant_id="item0001.v0", episode_ids=("e",),
        cell_ids=("c",), cell_kind="episode", protocol_hash="cond", model_hash=MODEL_HASH,
        round_=1, agent_slot=1, is_reference=False, spans=good[0].spans)
    lost = CAP.CaptureWorkItem(
        generation_uid="g:lost", decision_ids=("d:lost",), request_ids=("never-stored",),
        source_item_id="item0001", item_variant_id="item0001.v0", episode_ids=("e",),
        cell_ids=("c",), cell_kind="episode", protocol_hash="cond", model_hash=MODEL_HASH,
        round_=2, agent_slot=1, is_reference=False, spans=good[0].spans)

    manifest, stats = CAP.run_capture(
        (*good, refused, lost), engine=tiny_engine, tokenizer=tokenizer, store=store,
        run_root=tmp_run_root,
        config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_missing"))

    assert stats["captured"] == 1 and stats["context_failures"] == 1 and stats["missing_records"] == 1
    assert CAP.assert_capture_progress(stats) is stats
    by_decision = {r.decision_id: [x for x in manifest if x.decision_id == r.decision_id]
                   for r in manifest}
    for decision_id, reason in (("d:refused", CAP.MISSING_CONTEXT_FAILURE),
                                ("d:lost", CAP.MISSING_RECORD)):
        emitted = by_decision[decision_id]
        assert len(emitted) == len(CAP.ANCHORS) * len(TINY_BLOCKS)
        assert {r.missing_reason for r in emitted} == {reason}
        assert all(r.storage_hash is None and r.dtype is None for r in emitted)
        assert {r.replay_parity for r in emitted} == {CAP.PARITY_MISSING}
    assert stats["anchors_missing"][CAP.MISSING_CONTEXT_FAILURE] == len(CAP.ANCHORS)
    # the good sample was still captured: one unreplayable item does not cost the cell
    assert all(r.storage_hash for r in by_decision["d:root:0"])


def test_the_capture_hook_sits_at_the_block_output_and_two_passes_agree(tokenizer, tiny_engine):
    """Spec 13.2: "validate tokenizer identity, hook location, and logits on fixtures".

    The hook location is the fact every vector's meaning rests on, so it is checked against the
    model itself -- the residual this pass stores for block ``b`` is block ``b``'s output, the
    input to block ``b+1`` -- and the two-pass logit difference is *logged*, not demanded to be
    bit-identical on some other GPU.
    """
    torch = pytest.importorskip("torch")
    from agents_scaling.study.neural.engine import CaptureRequest

    ids = [int(t) for t in tokenizer.encode("the hook location is a fixture fact",
                                            add_special_tokens=False)]
    position = len(ids) - 1
    first, = tiny_engine.forward([CaptureRequest("fixture", tuple(ids), (position,))])
    second, = tiny_engine.forward([CaptureRequest("fixture", tuple(ids), (position,))])

    seen: dict[int, Any] = {}

    def record(block):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            seen[block] = tensor.detach()
        return hook

    handles = [tiny_engine.layers[b].register_forward_hook(record(b)) for b in TINY_BLOCKS]
    try:
        with torch.no_grad():
            reference = tiny_engine.model(torch.tensor([ids], device=tiny_engine.input_device),
                                          output_hidden_states=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    for block in TINY_BLOCKS:
        stored = np.asarray(first.residual_at(block, position))
        assert stored.dtype == np.float16 and stored.shape == (tiny_engine.hidden_size,)
        np.testing.assert_allclose(
            stored.astype(np.float32),
            seen[block][0, position].to(torch.float32).numpy(), rtol=0, atol=1e-2)
        # the same numbers twice: the pass is deterministic on one device
        np.testing.assert_array_equal(stored, np.asarray(second.residual_at(block, position)))
    # and the convention the ledger records is the one that was just checked: block b's output
    # is the input to block b+1 (pre-final-norm), which is hidden_states[b+1] below the top
    np.testing.assert_allclose(
        np.asarray(first.residual_at(TINY_BLOCKS[0], position)).astype(np.float32),
        reference.hidden_states[TINY_BLOCKS[0] + 1][0, position].to(torch.float32).numpy(),
        rtol=0, atol=1e-2)
    assert "block OUTPUT" in HOOK_CONVENTION and "use_cache=False" in HOOK_CONVENTION


# =============================================================================== read-only


def test_read_only_capture_cannot_change_generation():
    """Spec 13.2: instrumented inference must not alter behavioural generation.

    Structural, not promised: the whole import closure of ``nextdyn.capture`` inside the
    package is walked, and the generation path -- the client, the endpoint pool, the adapters,
    the decoding identity and the two-phase continuation -- is not in it.  A CAPTURE cell
    therefore cannot issue a request even by accident, and ``needs_endpoint`` is False.
    """
    from agents_scaling.nextdyn import kinds
    from agents_scaling.nextdyn.ids import CellKind as CK

    package = Path(CAP.__file__).resolve().parent
    closure, stack = set(), ["capture"]
    while stack:
        module = stack.pop()
        if module in closure:
            continue
        closure.add(module)
        path = package / f"{module.replace('.', '/')}.py"
        if not path.is_file():
            path = package / module.replace(".", "/") / "__init__.py"
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith("agents_scaling.nextdyn"):
                base = node.module[len("agents_scaling.nextdyn"):].lstrip(".")
                targets = [f"{base}.{a.name}" if base else a.name for a in node.names]
            elif isinstance(node, ast.Import):
                targets = [a.name[len("agents_scaling.nextdyn."):] for a in node.names
                           if a.name.startswith("agents_scaling.nextdyn.")]
            for target in targets:
                candidate = target if (package / f"{target.replace('.', '/')}.py").is_file() \
                    else target.rsplit(".", 1)[0]
                if candidate:
                    stack.append(candidate)

    forbidden = {"inference", "continuation", "adapters", "decoding", "episode", "telemetry"}
    assert not (closure & forbidden), f"capture reaches the generation path: {closure & forbidden}"
    assert "grading" not in closure and "construct" not in closure      # R7 as well

    spec = kinds.spec_for(CK.CAPTURE)
    assert spec.needs_endpoint is False and spec.lane is kinds.Lane.CAPTURE
    assert spec.worker == "agents_scaling.nextdyn.capture:capture_worker"
    # the module offers no way to generate: no client, no pool, no sampler
    names = {n for n in dir(CAP) if not n.startswith("_")}
    assert not names & {"generate", "generate_decision", "NextdynChatClient", "EndpointPool"}


def test_the_capture_table_is_declared_once_and_produced_here():
    """R1/R3: ``activations_manifest`` columns live in tables.py and this module is its producer."""
    schema = T.TABLES["activations_manifest"]
    assert schema.producer == "agents_scaling.nextdyn.capture"
    assert schema.primary_key == ("decision_id", "layer", "anchor")
    names = set(schema.column_names)
    assert {"token_offset", "missing_reason", "pre_confidence_eligible", "replay_parity"} <= names
    assert CAP.ANCHORS == ("PROMPT_END", "PRE_FINAL", "ANSWER_END")
    assert set(CAP.MISSING_REASONS) >= {CAP.MISSING_INVALID_ANSWER, CAP.MISSING_CONTEXT_FAILURE}


def test_the_capture_verb_reports_the_ledger_without_touching_a_gpu(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    store, rows = _team(tokenizer, tokens, tmp_run_root,
                        reports=(REPORT, REPORT, REPORT_INVALID, REPORT, REPORT))
    CAP.run_capture(CAP.capture_manifest(rows, None), engine=tiny_engine, tokenizer=tokenizer,
                    store=store, run_root=tmp_run_root,
                    config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_test"))

    report = CAP.main(["--run-root", str(tmp_run_root)])
    assert report.completed == 5 * len(TINY_BLOCKS) * len(CAP.ANCHORS) - 2 * len(TINY_BLOCKS)
    assert report.skipped == 2 * len(TINY_BLOCKS)
    assert CAP.MISSING_INVALID_ANSWER in report.detail
    blocked = CAP.main(["--run-root", str(tmp_run_root / "nowhere")])
    assert blocked.blocked == 1 and blocked.completed == 0


def test_the_capture_sbatch_template_takes_its_lane_from_compute_yaml(compute_cfg, tmp_path):
    """The CAPTURE lane job: every dispatch number is a placeholder, rc 5 continues, staging
    is node-local, and the interpreter is the replay env -- not the harness or the serve env."""
    import subprocess

    from agents_scaling.nextdyn import kinds

    template = (REPO / "slurm" / "nextdyn_capture.sbatch.tmpl").read_text(encoding="utf-8")
    shape = kinds.cell_shape(CellKind.CAPTURE, compute=compute_cfg)

    # the compute.yaml leaves this template is the declared consumer of are named in it
    for leaf in ("compute.capture.gpu_type", "compute.capture.gpus_per_job",
                 "compute.capture.partition"):
        assert leaf in template
    assert "/orcd/scratch" not in template                      # D3
    assert "envs/neural_env/bin/python" in template             # the replay env, not the harness
    assert "NEXTDYN_LOCAL_SCRATCH" in template and "consolidat" in template
    # the executable half never activates the harness or serve env and never starts a server
    body = "\n".join(l for l in template.splitlines() if not l.lstrip().startswith("#"))
    assert "common.sh" not in body and "mamba activate" not in body
    assert "vllm" not in body.lower() and "serve_qwen" not in body
    # exit 5 (EXIT_DEPS_UNMET) continues to the next cell; anything else stops the job
    assert 'if [ "$rc" -eq 5 ]; then' in body and "continue" in body
    assert 'if [ "$rc" -ne 0 ]; then' in body and 'exit "$rc"' in body
    # no partition, gpu type or cell shape is hard-coded: they all arrive as placeholders
    directives = "\n".join(l for l in template.splitlines() if l.startswith("#SBATCH"))
    for value in (compute_cfg.capture.partition, compute_cfg.capture.gpu_type,
                  str(compute_cfg.capture.gpus_per_job), shape.mem, shape.time):
        assert value not in directives

    rendered = template
    for key, value in {
        "PARTITION": compute_cfg.capture.partition, "GPU_TYPE": compute_cfg.capture.gpu_type,
        "GPUS_PER_JOB": str(compute_cfg.capture.gpus_per_job), "CPUS": str(shape.cpus),
        "MEM": shape.mem, "TIME": shape.time, "RUN_ID": "nd1", "REPO": str(REPO),
        "RUN_ROOT": "/tmp/run", "LOG_DIR": "/tmp/run/logs", "MANIFEST": "/tmp/run/m.json",
        "FIRST_INDEX": "0", "LAST_INDEX": "3",
    }.items():
        rendered = rendered.replace("{" + key + "}", value)
    runnable = "\n".join(l for l in rendered.splitlines() if not l.lstrip().startswith("#")
                         or l.startswith("#SBATCH") or l.startswith("#!"))
    assert "{" not in runnable.replace("${", "").replace("$(", "")
    assert f"--gres=gpu:{compute_cfg.capture.gpu_type}:{compute_cfg.capture.gpus_per_job}" in rendered
    assert f"--partition={compute_cfg.capture.partition}" in rendered

    script = tmp_path / "capture.sbatch"
    script.write_text(rendered, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_a_partial_resume_still_emits_a_row_for_every_committed_key(
        tokenizer, tokens, tiny_engine, tmp_run_root):
    """A pass that narrows the anchor set must not return a short ledger for the keys the
    earlier pass already committed: the census is read off the table, so a hole there is a
    coverage claim nobody made."""
    store, rows = _team(tokenizer, tokens, tmp_run_root)
    items = CAP.capture_manifest(rows, None)
    CAP.run_capture(items, engine=tiny_engine, tokenizer=tokenizer, store=store,
                    run_root=tmp_run_root,
                    config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_a",
                                             anchors=(CAP.ANCHOR_PROMPT_END,)))
    later, stats = CAP.run_capture(
        items, engine=tiny_engine, tokenizer=tokenizer, store=store, run_root=tmp_run_root,
        config=CAP.CaptureConfig(model_hash=MODEL_HASH, shard_name="nextdyn_b"))

    assert stats["pending"] == 5 and stats["captured"] == 5
    assert stats["rows_skipped_committed"] == 5 * len(TINY_BLOCKS)     # the PROMPT_END keys
    assert len(later) == 5 * len(CAP.ANCHORS) * len(TINY_BLOCKS)
    by_anchor = {a: [r for r in later if r.anchor == a] for a in CAP.ANCHORS}
    assert {r.replay_parity for r in by_anchor[CAP.ANCHOR_PROMPT_END]} == {CAP.PARITY_RESUMED}
    assert {r.replay_parity for r in by_anchor[CAP.ANCHOR_ANSWER_END]} == {CAP.PARITY_OK}
    assert all(r.storage_hash and r.token_offset is not None for r in later)
