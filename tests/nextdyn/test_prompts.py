"""Templates are files, byte-identical to the spec, and name no condition (plan §1.6, R5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from agents_scaling.nextdyn import prompts
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn.errors import ManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "expermt_agent_specs" / "taskCondAgentDynV1" / \
    "task_conditioned_agent_dynamics_experiment_spec_v1_0.md"


@dataclass(frozen=True)
class _Item:
    rendered_question = "What is 2 + 2?\nA. three\nB. four"
    source_item_id = "item-1"
    item_variant_id = "item-1:canonical"


@dataclass(frozen=True)
class _Family:
    answer_instruction = "Give the option letter."


def _spec_block(heading: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    start = text.index(heading)
    body = text.index("```text", start) + len("```text\n")
    return text[body:text.index("```", body)]


def test_root_and_revision_templates_match_spec_6_2_6_3():
    """The spec's own fenced blocks, byte for byte -- not a paraphrase and not reflowed."""
    assert prompts.template_text(prompts.ROOT) == _spec_block("### 6.2 Root template")
    assert prompts.template_text(prompts.REVISION) == _spec_block("### 6.3 Common revision template")


def test_every_declared_template_exists_and_carries_both_roles():
    for name in prompts.TEMPLATE_NAMES:
        messages = prompts.render_template(name, _placeholders(name))
        assert [m["role"] for m in messages] == [prompts.ROLE_SYSTEM, prompts.ROLE_USER]
        assert all(m["content"].strip() for m in messages)


def _placeholders(name: str) -> dict[str, str]:
    body = prompts.template_text(name)
    return {token: "X" for token in re.findall(r"\{([A-Za-z_]+)\}", body)}


#: Every condition name, legacy alias and graph word a template must never contain.
_FORBIDDEN = tuple(sorted({
    *P.load_registry().episodes, *P.load_registry().receivers,
    *(s.legacy_alias for s in P.load_registry().episodes.values() if s.legacy_alias),
    "peer", "private", "fresh", "focal", "graph", "edge", "neighbour", "neighbor",
    "channel", "network", "majority", "minority", "adviser", "advisor", "vote",
    "agent-0", "slot",
}))


def _placeholder_free(text: str) -> str:
    """Template text with the ``{placeholder}`` names removed: they are substituted away
    before a prompt exists, and the spec spells one of them ``visible_peer_packets...``."""
    return re.sub(r"\{[A-Za-z_]+\}", " ", text)


def test_no_template_contains_a_protocol_label():
    """Rule R5: nothing a model reads names a condition or describes the graph, so the only
    thing that differs between arms is the content of the OTHER DRAFTS block."""
    offenders = []
    for path in sorted(prompts.TEMPLATES_DIR.rglob("*")):
        if not path.is_file():
            continue
        text = _placeholder_free(path.read_text(encoding="utf-8")).lower()
        offenders.extend(f"{path.name}: {word}" for word in _FORBIDDEN if word.lower() in text)
    assert not offenders, offenders


def test_no_rendered_prompt_contains_a_protocol_label():
    rendered = [
        *prompts.render_root(_Item(), _Family()),
        *prompts.render_revision(_Item(), '{"from":"agent-1","final_answer":"B"}',
                                 ['{"from":"agent-2","final_answer":"A"}']),
        *prompts.render_audit(_Item(), '{"from":"agent-1","final_answer":"B"}'),
    ]
    blob = "\n".join(m["content"] for m in rendered).lower()
    forbidden = [w for w in _FORBIDDEN if w.lower() in blob and w not in ("agent-0", "slot")]
    assert not forbidden, forbidden


def test_no_template_states_a_horizon():
    """Horizon-free: nothing says how many rounds remain, which is what makes the round-4
    prompt of an R=8 episode identical to the round-4 prompt of an R=4 one."""
    for name in prompts.TEMPLATE_NAMES:
        text = prompts.template_text(name).lower()
        for word in ("round", "remaining", "final round", "last round", "iteration", "step "):
            assert word not in text, (name, word)
    # the two episode-loop templates carry no number at all; the audit template states the
    # probability range [0,1], which is a field domain and not a horizon
    for name in (prompts.ROOT, prompts.REVISION):
        assert not re.search(r"\d", _placeholder_free(prompts.template_text(name))), name


def test_private_renders_literal_NONE():
    """Spec 6.3: an agent with no visible peer packets sees the literal NONE, not an empty
    block, so the arm is indistinguishable from a peer arm whose senders were all removed."""
    messages = prompts.render_revision(_Item(), '{"from":"agent-1","final_answer":"B"}', [])
    user = messages[1]["content"]
    assert "\nOTHER DRAFTS\n" + prompts.NONE_LITERAL + "\n" in user


def test_the_root_prompt_carries_no_slot_role_or_social_count():
    """Spec 6.2: "No role diversity, expertise persona, social-count hint, or agent-slot
    identifier appears in the root prompt."."""
    user = prompts.render_root(_Item(), _Family())[1]["content"]
    assert _Item.rendered_question in user
    assert _Family.answer_instruction in user
    assert "agent" not in user.lower() and "expert" not in user.lower()


def test_peer_packets_render_one_per_line_in_the_given_order():
    peers = ['{"from":"agent-3","final_answer":"A"}', '{"from":"agent-1","final_answer":"B"}']
    user = prompts.render_revision(_Item(), None, peers)[1]["content"]
    block = user.split("OTHER DRAFTS\n", 1)[1].split("\n\n", 1)[0]
    assert block.split("\n") == peers


def test_a_missing_placeholder_or_template_is_a_typed_refusal():
    with pytest.raises(ManifestError):
        prompts.render_template(prompts.ROOT, {"not_a_placeholder": "x"})
    with pytest.raises(Exception):
        prompts.template_text("nope")


def test_the_filler_vocabulary_has_no_digit_and_no_bare_option_letter():
    words = prompts.filler_words()
    assert len(words) >= 16
    assert all(w.isalpha() and len(w) > 1 for w in words)
    assert not {w.upper() for w in words} & {chr(c) for c in range(ord("A"), ord("K"))}


def test_template_hashes_cover_every_file_and_detect_drift(tmp_path):
    hashes = prompts.template_hashes()
    assert set(hashes) == {p.name for p in prompts.TEMPLATES_DIR.iterdir() if p.is_file()}

    class _Lock:
        templates = {**hashes, "root.txt": "0" * 64}

    with pytest.raises(ManifestError, match="drifted"):
        prompts.assert_prompts_locked(_Lock())

    class _Good:
        templates = hashes

    prompts.assert_prompts_locked(_Good())
