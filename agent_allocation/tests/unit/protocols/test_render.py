"""Prompt rendering: templates, hash lock, packets, blind order, trimming, leakage guards (agent 03)."""
from __future__ import annotations

import json
from pathlib import Path
import re
from dataclasses import replace

import pytest

from agent_allocation.contracts import (
    CheckpointView, EvalKind, GenerationKind, OutputKind, ParseStatus, PrivateEvalSpec, PublicTask,
)
from agent_allocation.prompts import render
from agent_allocation.prompts.render import (
    ABSENT_MARKER, OMISSION_MARKER, PACKET_ENVELOPE_CHARS, RAW_MARKER, ContextLimitEvent, PromptError,
    bound_packet_text, output_requirement, prompt_hashes, prompt_lock_record, prompt_version, render_messages,
    render_messages_with_report, verify_prompt_lock,
)
from agent_allocation.protocols.grid import checkpoint_by_name, node_index

from .helpers import CODE_TASK, MATH_TASK, ROOT, SEED, candidate, graph, parents_for


def _template(name: str) -> str:
    return (ROOT / "prompts" / name).read_text()


def _fill(name: str, **values: str) -> str:
    text = _template(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def test_prompt_hashes_match_pack_checksums_and_version_format():
    pack = json.loads((ROOT / "CHECKSUMS.json").read_text())
    hashes = prompt_hashes()
    assert set(hashes) == {"system.txt", "root.txt", "private.txt", "peer.txt", "pooled.txt", "output_requirements.json"}
    registry = json.loads((ROOT / "docs" / "amendments" / "prompt_amendments.json").read_text())["amendments"]
    amended = {Path(a["template"]).name: a for a in registry}
    for name, digest in hashes.items():
        if name in amended:
            # A documented, owner-approved amendment: the pack hash is the recorded "before", the file the "after".
            assert pack[f"prompts/{name}"] == amended[name]["before_sha256"], name
            assert digest == amended[name]["after_sha256"], name
        else:
            assert pack[f"prompts/{name}"] == digest, name
    assert re.fullmatch(r"prompts-v1-[0-9a-f]{12}", prompt_version())
    verify_prompt_lock(prompt_lock_record())
    with pytest.raises(PromptError):
        verify_prompt_lock({**prompt_lock_record(), "packet_envelope_chars": PACKET_ENVELOPE_CHARS + 1})


def test_output_requirements_by_benchmark_and_kind():
    reqs = json.loads(_template("output_requirements.json"))
    assert output_requirement(MATH_TASK) == reqs["math500"]
    assert output_requirement(CODE_TASK) == reqs["mbpp_plus"]
    mmlu = PublicTask("mmlu_pro/1", "mmlu_pro", "rev", "Q?\n(A) x\n(B) y", OutputKind.OPTION, allowed_choices=("A", "B"))
    assert output_requirement(mmlu) == reqs["mmlu_pro"]
    boolean = PublicTask("bbh/b/1", "bbh_logic4", "rev", "not True is", OutputKind.BOOLEAN, subtask="boolean_expressions")
    choice = PublicTask("bbh/t/1", "bbh_logic4", "rev", "Q", OutputKind.OPTION, subtask="temporal_sequences")
    assert output_requirement(boolean) == reqs["bbh_boolean"]
    assert output_requirement(choice) == reqs["bbh_choice"]
    for bad in (replace(MATH_TASK, output_kind=OutputKind.OPTION), replace(boolean, output_kind=OutputKind.OPTION),
                replace(choice, output_kind=OutputKind.MATH), replace(MATH_TASK, benchmark="gsm8k")):
        with pytest.raises(PromptError):
            output_requirement(bad)


def test_root_and_deep_render_exactly_the_templates():
    g = graph("main")
    root = g.nodes[0]
    deep = next(n for n in g.nodes if n.kind is GenerationKind.DEEP)
    expected = [{"role": "system", "content": _template("system.txt")},
                {"role": "user", "content": _fill("root.txt", task_prompt=MATH_TASK.prompt,
                                                  output_requirement=output_requirement(MATH_TASK))}]
    assert render_messages(root, MATH_TASK, [], seed=SEED) == expected
    assert render_messages(deep, MATH_TASK, [], seed=SEED) == expected
    assert "{braces}" in expected[1]["content"] and "\\frac{1}{2}" in expected[1]["content"]


def test_private_and_cp_relay_only_the_parent_visible_content():
    g = graph("main_extension")
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE and n.update_round == 3 and n.root_slot == 0)
    parent = index[priv.parent_ids[0]]
    cand = replace(candidate(parent), final_answer="SECRET-PARSED", rationale="SECRET-RATIONALE-FIELD")
    messages, report = render_messages_with_report(priv, MATH_TASK, [(parent, cand)], seed=SEED)
    assert messages[1]["content"] == _fill("private.txt", task_prompt=MATH_TASK.prompt,
                                           output_requirement=output_requirement(MATH_TASK),
                                           own_packet=cand.visible_content)
    assert "SECRET" not in messages[1]["content"]
    assert report["packets"][0]["availability"] == "valid" and report["template"] == "private.txt"
    cp1 = next(n for n in g.nodes if n.kind is GenerationKind.CP and n.update_round == 1)
    peer_parent = index[cp1.parent_ids[0]]
    cp_messages = render_messages(cp1, MATH_TASK, [(peer_parent, candidate(peer_parent))], seed=SEED)
    assert cp_messages[1]["content"] == _fill("private.txt", task_prompt=MATH_TASK.prompt,
                                              output_requirement=output_requirement(MATH_TASK),
                                              own_packet=candidate(peer_parent).visible_content)


def test_absent_and_malformed_priors_follow_one_rule():
    g = graph("main")
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE)
    parent = index[priv.parent_ids[0]]
    for cand in (None, candidate(parent, "", status=ParseStatus.EMPTY), candidate(parent, "  \n", status=ParseStatus.EMPTY)):
        messages, report = render_messages_with_report(priv, MATH_TASK, [(parent, cand)], seed=SEED)
        assert f"YOUR PREVIOUS WRITTEN SOLUTION\n{ABSENT_MARKER}\n\n" in messages[1]["content"]
        assert report["packets"][0]["availability"] == "absent"
    raw = "I think the answer is 1 but I forgot the tags."
    for status in (ParseStatus.MISSING, ParseStatus.MALFORMED, ParseStatus.REFUSAL):
        messages, report = render_messages_with_report(priv, MATH_TASK, [(parent, candidate(parent, raw, status=status))],
                                                       seed=SEED)
        assert f"YOUR PREVIOUS WRITTEN SOLUTION\n{RAW_MARKER}\n{raw}\n\n" in messages[1]["content"]
        assert report["packets"][0]["availability"] == "raw"
    for marker in (ABSENT_MARKER, RAW_MARKER, OMISSION_MARKER):
        assert not re.search(r"wrong|incorrect|invalid|error|fail", marker, re.IGNORECASE)
    assert ABSENT_MARKER == "[No prior candidate response is available.]"


def _peer_request(g, n, r, j, contents=None):
    index = node_index(g)
    node = index[checkpoint_by_name(g, f"peer/{n}/{r}").candidate_ids[j]]
    cands = {p: candidate(index[p], contents[index[p].root_slot] if contents else None) for p in node.parent_ids}
    return node, parents_for(g, node, cands)


def test_peer_packets_own_first_then_blind_neutral_order():
    g = graph("main")
    contents = {s: f"<rationale>\nDraft text {chr(65 + s)}.\n</rationale>\n<final>\n{s}\n</final>" for s in range(5)}
    node, parents = _peer_request(g, 5, 2, 3, contents)
    messages, report = render_messages_with_report(node, MATH_TASK, parents, seed=SEED)
    user = messages[1]["content"]
    order = report["display_order_slots"]
    assert sorted(order) == [0, 1, 2, 4]
    peer_block = "\n\n".join(f"CANDIDATE {i}\n{contents[slot]}" for i, slot in enumerate(order, start=1))
    assert user == _fill("peer.txt", task_prompt=MATH_TASK.prompt, output_requirement=output_requirement(MATH_TASK),
                         own_packet=contents[3], peer_packets=peer_block)
    # order is independent of the order parents are supplied in (never return order) and stable across calls
    assert render_messages(node, MATH_TASK, list(reversed(parents)), seed=SEED) == messages
    # order is a function of (seed, item, n, round, receiver): over receivers/rounds several permutations appear
    perms = set()
    for r in range(1, 5):
        for j in range(5):
            node_rj, parents_rj = _peer_request(g, 5, r, j, contents)
            perms.add(tuple(render_messages_with_report(node_rj, MATH_TASK, parents_rj, seed=SEED)[1]["display_order_slots"]))
    assert len(perms) > 5
    other_seed = render_messages_with_report(node, MATH_TASK, parents, seed=SEED + 1)[1]["display_order_slots"]
    assert sorted(other_seed) == sorted(order)
    for word in ("expert", "majority", "minority", "correct", "checked", "verified", "agent", "team", "round",
                 "peer 1", "slot"):
        assert word not in peer_block.lower()


def test_pooled_reads_the_four_roots():
    g = graph("main_extension")
    index = node_index(g)
    node = index[checkpoint_by_name(g, "pooled/4").candidate_ids[0]]
    parents = parents_for(g, node, {p: candidate(index[p]) for p in node.parent_ids})
    messages, report = render_messages_with_report(node, MATH_TASK, parents, seed=SEED)
    order = report["display_order_slots"]
    assert sorted(order) == [0, 1, 2, 3]
    roots = {index[p].root_slot: candidate(index[p]).visible_content for p in node.parent_ids}
    block = "\n\n".join(f"CANDIDATE {i}\n{roots[s]}" for i, s in enumerate(order, start=1))
    assert messages[1]["content"] == _fill("pooled.txt", task_prompt=MATH_TASK.prompt,
                                           output_requirement=output_requirement(MATH_TASK), candidate_packets=block)


def test_no_horizon_round_slot_or_team_size_in_prompts():
    """Same parent text -> byte-identical prompt, whatever the round, slot, chain or horizon."""
    g = graph("main")
    index = node_index(g)
    text = "<rationale>\nSame text.\n</rationale>\n<final>\n2\n</final>"
    prompts = set()
    for node in g.nodes:
        if node.kind is GenerationKind.PRIVATE:
            parent = index[node.parent_ids[0]]
            prompts.add(render_messages(node, MATH_TASK, [(parent, candidate(parent, text))], seed=SEED)[1]["content"])
    assert len(prompts) == 1
    same = {}
    for n in (2, 3, 4, 5):
        for r in (1, 2, 3, 4):
            for j in range(n):
                node, parents = _peer_request(g, n, r, j, {s: text for s in range(5)})
                user = render_messages(node, MATH_TASK, parents, seed=SEED)[1]["content"]
                same.setdefault(n, set()).add(user)
    assert all(len(v) == 1 for v in same.values())   # only the number of packets varies with n
    for user in (u for v in same.values() for u in v):
        assert not re.search(r"\b(round|horizon|remaining|team|agent|role|slot|iteration|cycle)\b", user, re.IGNORECASE)


def test_identical_prompts_never_imply_identical_nodes():
    g = graph("main")
    roots = [n for n in g.nodes if n.kind is GenerationKind.ROOT]
    rendered = {json.dumps(render_messages(n, MATH_TASK, [], seed=SEED)) for n in roots}
    assert len(rendered) == 1 and len({n.node_id for n in roots}) == 25


def test_private_canary_never_reaches_prompts():
    canary = "CANARY-7f3a91-gold"
    private = PrivateEvalSpec(item_id=MATH_TASK.item_id, benchmark="math500", source_revision=MATH_TASK.source_revision,
                              eval_kind=EvalKind.MATH_VERIFY, gold_answer=canary, private_record_ref=f"private/{canary}",
                              private_hash=canary)
    g = graph("main_extension")
    index = node_index(g)
    cands = {n.node_id: candidate(n) for n in g.nodes}
    for node in g.nodes:
        messages, report = render_messages_with_report(node, MATH_TASK, parents_for(g, node, cands), seed=SEED)
        blob = json.dumps(messages) + json.dumps(report)
        assert canary not in blob and private.private_hash not in blob
    assert "private" not in " ".join(render.render_messages.__annotations__.values()).lower()


def test_trimming_keeps_final_symmetric_and_reports_bytes():
    final = "<final>\ndef f(x):\n    return {x: 'é'}\n</final>"
    rationale = "<rationale>\n" + "".join(chr(0x61 + i % 26) for i in range(5000)) + "é" * 5000 + "\n</rationale>\n"
    text = rationale + final + "\n"
    bounded, rep = bound_packet_text(text, envelope=2000, source_node_id="n")
    assert len(bounded) <= 2000 and final in bounded and OMISSION_MARKER in bounded
    head, tail = bounded.split(f"\n{OMISSION_MARKER}\n")
    assert text.startswith(head) and text.endswith(tail)
    assert len(head) - (len(tail) - len(final)) in (0, 1)          # ceil(K/2) head, floor(K/2) tail outside <final>
    assert len(bounded) == 2000                                     # the envelope is used exactly, not undershot
    assert rep["trimmed"] and rep["cuts"] == 1 and rep["chars_removed"] == len(text) - len(head) - len(tail)
    removed = text[len(head):len(text) - len(tail)]
    assert rep["bytes_removed"] == len(removed.encode("utf-8")) > rep["chars_removed"]
    untouched, rep0 = bound_packet_text("short <final>1</final>", envelope=2000)
    assert untouched == "short <final>1</final>" and not rep0["trimmed"] and rep0["bytes_removed"] == 0
    with pytest.raises(ContextLimitEvent) as info:
        bound_packet_text("x" * 100 + "<final>" + "y" * 3000 + "</final>", envelope=2000, source_node_id="src")
    assert info.value.record["source_node_id"] == "src" and info.value.record["event"] == "context_limit"


def test_trimming_straddling_final_and_malformed_multiple_finals():
    text = "a" * 3000 + "<final>1</final>" + "b" * 3000 + "<final>2</final>" + "c" * 3000
    bounded, rep = bound_packet_text(text, envelope=1000)
    assert len(bounded) <= 1000 and "<final>1</final>" in bounded and "<final>2</final>" in bounded
    assert bounded.startswith("a") and bounded.endswith("c") and rep["final_elements"] == 2


def test_render_reports_trimming_and_context_limit_receiver():
    g = graph("main", CODE_TASK)
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE)
    parent = index[priv.parent_ids[0]]
    long_text = "<rationale>\n" + "z" * (2 * PACKET_ENVELOPE_CHARS) + "\n</rationale>\n<final>\nprint(1)\n</final>"
    messages, report = render_messages_with_report(priv, CODE_TASK, [(parent, candidate(parent, long_text,
                                                   output_kind=OutputKind.PYTHON))], seed=SEED)
    assert report["trimmed_packets"] == 1 and report["bytes_removed_total"] > PACKET_ENVELOPE_CHARS
    assert "<final>\nprint(1)\n</final>" in messages[1]["content"]
    assert report["rendered_chars"] == sum(len(m["content"]) for m in messages)
    huge_final = "<final>\n" + "x = 1\n" * PACKET_ENVELOPE_CHARS + "</final>"
    with pytest.raises(ContextLimitEvent) as info:
        render_messages(priv, CODE_TASK, [(parent, candidate(parent, huge_final, output_kind=OutputKind.PYTHON))], seed=SEED)
    assert info.value.record["receiver_node_id"] == priv.node_id


def test_render_refuses_mismatched_requests():
    g = graph("main")
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE)
    parent = index[priv.parent_ids[0]]
    good = [(parent, candidate(parent))]
    other_root = g.nodes[1]
    with pytest.raises(PromptError):
        render_messages(priv, MATH_TASK, [], seed=SEED)                              # missing parent
    with pytest.raises(PromptError):
        render_messages(priv, MATH_TASK, [(other_root, candidate(other_root))], seed=SEED)   # wrong parent
    with pytest.raises(PromptError):
        render_messages(priv, MATH_TASK, [(parent, candidate(other_root))], seed=SEED)       # candidate/parent mismatch
    with pytest.raises(PromptError):
        render_messages(priv, replace(MATH_TASK, prompt="Other task"), good, seed=SEED)      # task hash mismatch
    stale = graph("main", version="prompts-v1-000000000000")
    with pytest.raises(PromptError):
        render_messages(stale.nodes[0], MATH_TASK, [], seed=SEED)
    with pytest.raises(PromptError):
        render_messages(g.nodes[0], MATH_TASK, good, seed=SEED)                               # root with parents
    assert checkpoint_by_name(g, "sample/1", CheckpointView.ARCHIVE)


LEAD_PACKET = "<rationale>\nAgent 0 says N2 is 78%.\n</rationale>\n<final>\nB\n</final>"


def test_final_elements_counted_for_packets_inside_the_envelope():
    """Lead regression: an in-envelope PEER packet with one closed <final> reported final_elements == 0."""
    g = graph("main")
    node, parents = _peer_request(g, 3, 1, 0, {s: LEAD_PACKET for s in range(5)})
    messages, report = render_messages_with_report(node, MATH_TASK, parents, seed=SEED)
    assert [p["final_elements"] for p in report["packets"]] == [1, 1, 1]
    assert all(not p["trimmed"] and p["finals_protected"] for p in report["packets"])
    assert messages[1]["content"].count("<final>\nB\n</final>") == 3
    assert bound_packet_text(LEAD_PACKET)[1]["final_elements"] == 1
    assert bound_packet_text("no tags at all")[1]["final_elements"] == 0


MULTILINE_FINAL = "<final>\nThe answer is\n\n$x = \\frac{3}{4}$\n\nafter simplification.\n</final>"
FENCED_FINAL = ("<final>\n```python\nimport math\n\ndef similar_elements(a, b):\n    if len(a) > 0 and len(b) > 0:\n"
                "        return set(a) & set(b)\n    return {x for x in ()}\n```\n</final>")


@pytest.mark.parametrize("final", [MULTILINE_FINAL, FENCED_FINAL], ids=["multiline", "fenced_python"])
def test_over_envelope_packet_keeps_multiline_and_fenced_finals(final):
    """Lead regression: packets over the envelope keep the whole <final>...</final> and count exactly one."""
    rationale = "<rationale>\n" + "\n".join(f"Step {i}: Agent 0 says N2 is 78%." for i in range(900)) + "\n</rationale>\n"
    text = rationale + final + "\n"
    assert len(text) > PACKET_ENVELOPE_CHARS
    bounded, rep = bound_packet_text(text)
    assert final in bounded and bounded.count("<final>") == 1 and bounded.rstrip().endswith("</final>")
    assert rep["final_elements"] == 1 and rep["trimmed"] and rep["finals_protected"]
    assert len(bounded) <= PACKET_ENVELOPE_CHARS
    # through the renderer, as a PEER packet (own and peer positions)
    g = graph("main")
    node, parents = _peer_request(g, 2, 1, 0, {0: text, 1: text})
    messages, report = render_messages_with_report(node, MATH_TASK, parents, seed=SEED)
    assert messages[1]["content"].count(final) == 2
    assert [(p["role"], p["final_elements"], p["trimmed"]) for p in report["packets"]] == [("own", 1, True), ("peer", 1, True)]


def test_raw_relay_is_always_bounded_even_when_its_final_cannot_fit():
    huge = "<rationale>\nshort\n</rationale>\n<final>\n" + "y" * (2 * PACKET_ENVELOPE_CHARS) + "\n</final>"
    with pytest.raises(ContextLimitEvent):
        bound_packet_text(huge)                                    # VALID: never cut the final, record an event
    bounded, rep = bound_packet_text(huge, strict_final=False)    # raw relay: bounded, marked, reported
    assert len(bounded) <= PACKET_ENVELOPE_CHARS and not rep["finals_protected"] and rep["final_elements"] == 1
    assert bounded.startswith("<rationale>") and bounded.endswith("</final>") and OMISSION_MARKER in bounded
    g = graph("main")
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE)
    parent = index[priv.parent_ids[0]]
    messages, report = render_messages_with_report(
        priv, MATH_TASK, [(parent, candidate(parent, huge, status=ParseStatus.MALFORMED))], seed=SEED)
    assert f"{RAW_MARKER}\n<rationale>" in messages[1]["content"] and report["packets"][0]["availability"] == "raw"


def test_symmetric_trim_property_over_random_texts():
    import random
    rng = random.Random(20260917)
    marker = f"\n{OMISSION_MARKER}\n"
    for _ in range(300):
        parts = []
        for _ in range(rng.randint(1, 4)):
            parts.append("".join(rng.choice("ab é\n<>/") for _ in range(rng.randint(0, 400))))
            if rng.random() < 0.6:
                parts.append("<final>" + "".join(rng.choice("xyz\n{}") for _ in range(rng.randint(0, 60))) + "</final>")
        text = "".join(parts)
        envelope = rng.randint(2 * len(marker) + 1, 900)
        spans = render._final_spans(text)
        try:
            bounded, rep = bound_packet_text(text, envelope=envelope)
        except ContextLimitEvent:
            # finals plus one marker per non-empty gap the symmetric cut must cross exceed the envelope
            bounds = [0] + [x for span in spans for x in span] + [len(text)]
            gaps = sum(1 for a, b in zip(bounds[::2], bounds[1::2]) if b > a)
            assert sum(b - a for a, b in spans) + len(marker) * max(1, gaps) > envelope
            continue
        assert len(bounded) <= envelope and rep["final_elements"] == len(spans)
        assert all(text[a:b] in bounded for a, b in spans)
        if not rep["trimmed"]:
            assert bounded == text
            continue
        # removing the markers leaves the original minus the cut ranges; kept outside text splits ceil/floor
        pieces = bounded.split(marker)
        assert len(pieces) == rep["cuts"] + 1
        assert text.startswith(pieces[0]) and text.endswith(pieces[-1])
        kept = sum(len(p) for p in pieces)
        assert kept + rep["chars_removed"] == len(text)
        assert rep["bytes_removed"] == len(text.encode()) - sum(len(p.encode()) for p in pieces)
        assert len(bounded) == envelope                                  # K is exactly what fits


def test_final_span_scanner_matches_reference_regex_and_is_linear():
    import random
    import time
    reference = re.compile(r"<final\b[^>]{0,64}>.*?</final\s*>", re.IGNORECASE | re.DOTALL)
    rng = random.Random(7)
    alphabet = ["<final>", "</final>", "<FINAL >", "</final >", "<final x='1'>", "<finally>", "a", "\n", ">", "<"]
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert render._final_spans(text) == [m.span() for m in reference.finditer(text)], text
    degenerate = "<final>x " * 20000
    start = time.perf_counter()
    assert render._final_spans(degenerate) == []
    assert time.perf_counter() - start < 0.5


def test_lone_surrogate_in_a_response_does_not_crash_rendering():
    content = json.loads('"<rationale>\\nbad \\ud800 byte\\n</rationale>\\n<final>\\n1\\n</final>"')
    g = graph("main")
    index = node_index(g)
    priv = next(n for n in g.nodes if n.kind is GenerationKind.PRIVATE)
    parent = index[priv.parent_ids[0]]
    messages, report = render_messages_with_report(priv, MATH_TASK, [(parent, candidate(parent, content))], seed=SEED)
    assert "\ud800" in messages[1]["content"] and report["packets"][0]["final_elements"] == 1
    long_content = content.replace("bad", "bad" + "\ud800" * (2 * PACKET_ENVELOPE_CHARS))
    _, trimmed = render_messages_with_report(priv, MATH_TASK, [(parent, candidate(parent, long_content))], seed=SEED)
    assert trimmed["bytes_removed_total"] > 0
    ordinary = render_messages_with_report(priv, MATH_TASK, [(parent, candidate(parent))], seed=SEED)
    assert ordinary[1]["messages_sha256"] == render.canonical_hash(ordinary[0])
